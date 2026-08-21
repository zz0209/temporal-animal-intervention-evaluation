from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .baseline_ranking import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    STATIC_SUMMARY_COLUMNS,
    animal_system_family,
)


KEY_COLUMNS = [*CONTEXT_COLUMNS, "candidate_id"]


def _weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    root_weights = np.sqrt(weights)[:, None]
    weighted_design = design * root_weights
    weighted_y = y * root_weights[:, 0]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_y,
    )


def _predict_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target: str,
    alpha: float,
) -> np.ndarray:
    context_size = train.groupby(CONTEXT_COLUMNS, observed=True)[
        "candidate_id"
    ].transform("size")
    context_count = train[CONTEXT_COLUMNS].drop_duplicates().shape[0]
    weights = 1.0 / (context_size.to_numpy(dtype=float) * context_count)
    coefficients = _weighted_ridge(
        train[columns].to_numpy(dtype=float),
        train[target].to_numpy(dtype=float),
        weights,
        alpha,
    )
    design = np.column_stack(
        [np.ones(len(test)), test[columns].to_numpy(dtype=float)]
    )
    return design @ coefficients


def _random_score(label_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{label_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _add_percentile_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in FEATURE_COLUMNS:
        output[f"pct_{column}"] = output.groupby(
            CONTEXT_COLUMNS, observed=True
        )[column].rank(method="average", pct=True)
    return output


def add_strictly_prior_candidate_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add candidate summaries that use labels strictly before each anchor time."""
    output = frame.copy()
    output["anchor_time"] = pd.to_datetime(output["anchor_time"], format="mixed")
    output["stable_prior_score"] = np.nan
    output["stable_prior_raw_value"] = np.nan
    output["last_prior_score"] = np.nan
    output["prior_observations"] = 0
    output["stable_prior_known"] = False
    for dataset_id, dataset_frame in output.groupby(
        "dataset_id", sort=True, observed=True
    ):
        for anchor_time in sorted(dataset_frame["anchor_time"].unique()):
            test_mask = output["dataset_id"].eq(dataset_id) & output[
                "anchor_time"
            ].eq(anchor_time)
            prior = output.loc[
                output["dataset_id"].eq(dataset_id)
                & output["anchor_time"].lt(anchor_time)
            ]
            if prior.empty:
                output.loc[test_mask, "stable_prior_score"] = 0.5
                output.loc[test_mask, "stable_prior_raw_value"] = 0.0
                output.loc[test_mask, "last_prior_score"] = 0.5
                continue
            score_mean = prior.groupby("candidate_id", observed=True)[
                "robust_priority_percentile"
            ].mean()
            raw_mean = prior.groupby("candidate_id", observed=True)[
                "robust_intervention_value"
            ].mean()
            counts = prior.groupby("candidate_id", observed=True).size()
            last = (
                prior.sort_values("anchor_time", kind="stable")
                .groupby("candidate_id", observed=True)
                .tail(1)
                .set_index("candidate_id")["robust_priority_percentile"]
            )
            candidates = output.loc[test_mask, "candidate_id"]
            output.loc[test_mask, "stable_prior_score"] = (
                candidates.map(score_mean).fillna(0.5).to_numpy()
            )
            output.loc[test_mask, "stable_prior_raw_value"] = (
                candidates.map(raw_mean).fillna(float(prior["robust_intervention_value"].mean())).to_numpy()
            )
            output.loc[test_mask, "last_prior_score"] = (
                candidates.map(last).fillna(0.5).to_numpy()
            )
            output.loc[test_mask, "prior_observations"] = (
                candidates.map(counts).fillna(0).astype(int).to_numpy()
            )
            output.loc[test_mask, "stable_prior_known"] = candidates.isin(
                score_mean.index
            ).to_numpy()
    output["pct_prior_observations"] = output.groupby(
        CONTEXT_COLUMNS, observed=True
    )["prior_observations"].rank(method="average", pct=True)
    if output[
        ["stable_prior_score", "stable_prior_raw_value", "last_prior_score"]
    ].isna().any().any():
        raise ValueError("strictly prior candidate summaries are incomplete")
    return output


def build_forward_predictions(
    feature_labels: pd.DataFrame,
    *,
    min_prior_anchor_times: int = 2,
    ridge_alpha: float = 1.0,
    seed: int = 20260816,
) -> pd.DataFrame:
    """Build expanding-window predictions using only strictly earlier anchors."""
    if min_prior_anchor_times < 1:
        raise ValueError("min_prior_anchor_times must be positive")
    frame = feature_labels.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], format="mixed")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("forward feature-label keys are not unique")
    frame = add_strictly_prior_candidate_history(_add_percentile_features(frame))
    percentile_columns = [f"pct_{column}" for column in FEATURE_COLUMNS]
    static_columns = [f"pct_{column}" for column in STATIC_SUMMARY_COLUMNS]
    hybrid_columns = [
        *percentile_columns,
        "stable_prior_score",
        "pct_prior_observations",
        "stable_prior_known",
    ]
    rows: list[pd.DataFrame] = []
    for dataset_id, dataset_frame in frame.groupby(
        "dataset_id", sort=True, observed=True
    ):
        anchor_times = sorted(dataset_frame["anchor_time"].unique())
        for anchor_index, anchor_time in enumerate(anchor_times):
            if anchor_index < min_prior_anchor_times:
                continue
            train = dataset_frame.loc[dataset_frame["anchor_time"].lt(anchor_time)]
            test = dataset_frame.loc[dataset_frame["anchor_time"].eq(anchor_time)].copy()
            if train["anchor_time"].nunique() != anchor_index:
                raise ValueError("forward split contains an unexpected anchor-time gap")
            test["score_random"] = test["label_id"].map(
                lambda value: _random_score(str(value), seed)
            )
            test["score_stable_watchlist"] = test["stable_prior_score"]
            test["score_last_observed_value"] = test["last_prior_score"]
            test["score_current_activity"] = test["pct_event_rate_per_day"]
            test["score_current_composite"] = test[
                [
                    "pct_event_rate_per_day",
                    "pct_observed_partner_count",
                    "pct_contact_opportunity_rate",
                    "pct_recency_score",
                    "pct_recent_activity_fraction",
                ]
            ].mean(axis=1)
            model_columns = {
                "forward_ridge_static": static_columns,
                "forward_ridge_current": percentile_columns,
                "forward_ridge_hybrid": hybrid_columns,
            }
            for method, columns in model_columns.items():
                test[f"score_{method}"] = _predict_ridge(
                    train,
                    test,
                    columns,
                    "robust_priority_percentile",
                    ridge_alpha,
                )
                test[f"raw_prediction_{method}"] = _predict_ridge(
                    train,
                    test,
                    columns,
                    "robust_intervention_value",
                    ridge_alpha,
                )
            test["raw_prediction_stable_watchlist"] = test[
                "stable_prior_raw_value"
            ]
            test["score_future_oracle"] = test["robust_intervention_value"]
            test["raw_prediction_future_oracle"] = test[
                "robust_intervention_value"
            ]
            test["prior_anchor_times"] = anchor_index
            test["candidate_seen_before"] = test["stable_prior_known"].astype(bool)
            rows.append(test)
    if not rows:
        raise ValueError("no anchors satisfy the forward evaluation requirement")
    output = pd.concat(rows, ignore_index=True)
    output["system_family"] = output["dataset_id"].map(animal_system_family)
    score_columns = [column for column in output if column.startswith("score_")]
    if output[score_columns].isna().any().any():
        raise ValueError("forward predictions are incomplete")
    return output


def balanced_variance_decomposition(
    labels: pd.DataFrame,
    *,
    target: str = "robust_priority_percentile",
) -> pd.DataFrame:
    """Decompose balanced common-support variation by animal and anchor."""
    frame = labels.copy()
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], format="mixed")
    rows: list[dict[str, object]] = []
    for (dataset_id, network_id), unit in frame.groupby(
        ["dataset_id", "network_id"], sort=True, observed=True
    ):
        pivot = unit.pivot(index="candidate_id", columns="anchor_time", values=target)
        complete = pivot.dropna(axis=0, how="any")
        if complete.shape[0] < 2 or complete.shape[1] < 2:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "network_id": network_id,
                    "system_family": animal_system_family(str(dataset_id)),
                    "animals_ever_observed": pivot.shape[0],
                    "common_support_animals": complete.shape[0],
                    "anchors": complete.shape[1],
                    "common_support_fraction": (
                        complete.shape[0] / pivot.shape[0] if pivot.shape[0] else 0.0
                    ),
                    "status": "not_estimable",
                    "individual_fraction": np.nan,
                    "anchor_fraction": np.nan,
                    "individual_anchor_residual_fraction": np.nan,
                }
            )
            continue
        values = complete.to_numpy(dtype=float)
        grand = float(values.mean())
        animal_effect = values.mean(axis=1) - grand
        anchor_effect = values.mean(axis=0) - grand
        residual = values - grand - animal_effect[:, None] - anchor_effect[None, :]
        individual_ss = values.shape[1] * float(np.square(animal_effect).sum())
        anchor_ss = values.shape[0] * float(np.square(anchor_effect).sum())
        residual_ss = float(np.square(residual).sum())
        total_ss = individual_ss + anchor_ss + residual_ss
        rows.append(
            {
                "dataset_id": dataset_id,
                "network_id": network_id,
                "system_family": animal_system_family(str(dataset_id)),
                "animals_ever_observed": pivot.shape[0],
                "common_support_animals": complete.shape[0],
                "anchors": complete.shape[1],
                "common_support_fraction": complete.shape[0] / pivot.shape[0],
                "status": "estimated" if total_ss > 0 else "constant_target",
                "individual_fraction": individual_ss / total_ss if total_ss > 0 else 0.0,
                "anchor_fraction": anchor_ss / total_ss if total_ss > 0 else 0.0,
                "individual_anchor_residual_fraction": (
                    residual_ss / total_ss if total_ss > 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_raw_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate raw intervention-value calibration within each forward context."""
    prediction_columns = [
        column for column in predictions if column.startswith("raw_prediction_")
    ]
    rows: list[dict[str, object]] = []
    for context, frame in predictions.groupby(
        CONTEXT_COLUMNS, sort=True, observed=True
    ):
        observed = frame["robust_intervention_value"].to_numpy(dtype=float)
        scale = float(np.quantile(observed, 0.9) - np.quantile(observed, 0.1))
        for column in prediction_columns:
            predicted = frame[column].to_numpy(dtype=float)
            error = predicted - observed
            design = np.column_stack([np.ones(len(predicted)), predicted])
            coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
            rows.append(
                {
                    "dataset_id": context[0],
                    "network_id": context[1],
                    "anchor_time": context[2],
                    "system_family": animal_system_family(str(context[0])),
                    "method": column.removeprefix("raw_prediction_"),
                    "candidate_count": len(frame),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                    "normalized_mae": (
                        float(np.abs(error).mean()) / scale if scale > 1e-15 else np.nan
                    ),
                    "calibration_intercept": float(coefficients[0]),
                    "calibration_slope": float(coefficients[1]),
                }
            )
    return pd.DataFrame(rows)
