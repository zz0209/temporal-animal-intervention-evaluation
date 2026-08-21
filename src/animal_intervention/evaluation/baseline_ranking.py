from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = ["dataset_id", "network_id", "anchor_time"]
FEATURE_COLUMNS = [
    "event_rate_per_day",
    "contact_opportunity_rate",
    "weighted_exposure_rate",
    "eligible_partner_fraction",
    "observed_partner_count",
    "location_count",
    "mean_group_size",
    "recency_score",
    "recent_activity_fraction",
    "active_span_fraction",
]
STATIC_SUMMARY_COLUMNS = FEATURE_COLUMNS[:-3]
TEMPORAL_SUMMARY_COLUMNS = FEATURE_COLUMNS[-3:]


def animal_system_family(dataset_id: str) -> str:
    if dataset_id in {"wytham_great_tits_divorce", "experimental_wild_songbirds"}:
        return "linked_wytham_songbirds"
    return dataset_id


def _percentile_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in FEATURE_COLUMNS:
        output[f"pct_{column}"] = output.groupby(CONTEXT_COLUMNS, observed=True)[
            column
        ].rank(method="average", pct=True)
    return output


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


def _deterministic_random_score(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def fit_baseline_scores(
    feature_labels: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    seed: int = 20260816,
) -> pd.DataFrame:
    """Create non-learned scores and strict leave-one-system-out ridge scores."""
    frame = _percentile_features(feature_labels)
    frame["system_family"] = frame["dataset_id"].map(animal_system_family)
    frame["score_random"] = frame["label_id"].map(
        lambda value: _deterministic_random_score(str(value), seed)
    )
    frame["score_activity"] = frame["pct_event_rate_per_day"]
    frame["score_partner_diversity"] = frame["pct_observed_partner_count"]
    frame["score_contact_opportunity"] = frame["pct_contact_opportunity_rate"]
    frame["score_recent_activity"] = (
        frame["pct_recency_score"] + frame["pct_recent_activity_fraction"]
    ) / 2
    frame["score_composite"] = frame[
        [
            "score_activity",
            "score_partner_diversity",
            "score_contact_opportunity",
            "score_recent_activity",
        ]
    ].mean(axis=1)
    percentile_columns = [f"pct_{column}" for column in FEATURE_COLUMNS]
    frame["score_ridge_loso"] = np.nan
    anchor_key = frame[CONTEXT_COLUMNS].astype(str).agg("|".join, axis=1)
    anchor_sizes = anchor_key.map(anchor_key.value_counts())
    family_anchor_counts = frame.groupby("system_family", observed=True).apply(
        lambda value: value[CONTEXT_COLUMNS].drop_duplicates().shape[0],
        include_groups=False,
    )
    row_weights = 1.0 / (
        anchor_sizes.to_numpy(dtype=float)
        * frame["system_family"].map(family_anchor_counts).to_numpy(dtype=float)
    )
    for family in sorted(frame["system_family"].unique()):
        train = frame["system_family"].ne(family).to_numpy()
        test = ~train
        coefficients = _weighted_ridge(
            frame.loc[train, percentile_columns].to_numpy(dtype=float),
            frame.loc[train, "robust_priority_percentile"].to_numpy(dtype=float),
            row_weights[train],
            ridge_alpha,
        )
        test_design = np.column_stack(
            [
                np.ones(test.sum()),
                frame.loc[test, percentile_columns].to_numpy(dtype=float),
            ]
        )
        frame.loc[test, "score_ridge_loso"] = test_design @ coefficients
    if frame["score_ridge_loso"].isna().any():
        raise ValueError("leave-one-system-out predictions are incomplete")
    return frame


def fit_feature_ablation_scores(
    feature_labels: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Fit matched LOSO ridge models before and after temporal summaries."""
    frame = _percentile_features(feature_labels)
    frame["system_family"] = frame["dataset_id"].map(animal_system_family)
    anchor_key = frame[CONTEXT_COLUMNS].astype(str).agg("|".join, axis=1)
    anchor_sizes = anchor_key.map(anchor_key.value_counts())
    family_anchor_counts = frame.groupby("system_family", observed=True).apply(
        lambda value: value[CONTEXT_COLUMNS].drop_duplicates().shape[0],
        include_groups=False,
    )
    row_weights = 1.0 / (
        anchor_sizes.to_numpy(dtype=float)
        * frame["system_family"].map(family_anchor_counts).to_numpy(dtype=float)
    )
    feature_sets = {
        "ridge_static_summary_loso": STATIC_SUMMARY_COLUMNS,
        "ridge_temporal_summary_loso": FEATURE_COLUMNS,
    }
    for score_name in feature_sets:
        frame[f"score_{score_name}"] = np.nan
    for family in sorted(frame["system_family"].unique()):
        train = frame["system_family"].ne(family).to_numpy()
        test = ~train
        for score_name, columns in feature_sets.items():
            percentile_columns = [f"pct_{column}" for column in columns]
            coefficients = _weighted_ridge(
                frame.loc[train, percentile_columns].to_numpy(dtype=float),
                frame.loc[train, "robust_priority_percentile"].to_numpy(dtype=float),
                row_weights[train],
                ridge_alpha,
            )
            test_design = np.column_stack(
                [
                    np.ones(test.sum()),
                    frame.loc[test, percentile_columns].to_numpy(dtype=float),
                ]
            )
            frame.loc[test, f"score_{score_name}"] = test_design @ coefficients
    output_columns = [
        *CONTEXT_COLUMNS,
        "candidate_id",
        "robust_intervention_value",
        "robust_priority_percentile",
        "score_ridge_static_summary_loso",
        "score_ridge_temporal_summary_loso",
    ]
    output = frame[output_columns].copy()
    score_columns = [column for column in output if column.startswith("score_")]
    if output[score_columns].isna().any().any():
        raise ValueError("feature-ablation LOSO predictions are incomplete")
    return output


def evaluate_baseline_scores(
    scored: pd.DataFrame,
    *,
    top_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between zero and one")
    score_columns = [column for column in scored if column.startswith("score_")]
    rows: list[dict[str, object]] = []
    for context, frame in scored.groupby(CONTEXT_COLUMNS, sort=True, observed=True):
        n = len(frame)
        k = max(1, math.ceil(n * top_fraction))
        values = frame["robust_intervention_value"].to_numpy(dtype=float)
        oracle = float(np.sort(values)[-k:].mean())
        random_expected = float(values.mean())
        denominator = oracle - random_expected
        label_ranks = frame["robust_priority_percentile"]
        for score_column in score_columns:
            method = score_column.removeprefix("score_")
            scores = frame[score_column]
            oracle_ids = set(frame.nlargest(k, "robust_intervention_value")["candidate_id"])
            if method == "random":
                selected_value = random_expected
                expected_oracle_overlap = k * k / n
                has_variation = True
                spearman = 0.0
                selection_evaluation = "analytic_random_expectation"
            else:
                remaining = k
                expected_value_sum = 0.0
                expected_oracle_overlap = 0.0
                for score_value in sorted(frame[score_column].unique(), reverse=True):
                    tied = frame.loc[frame[score_column].eq(score_value)]
                    if remaining >= len(tied):
                        fraction = 1.0
                    else:
                        fraction = remaining / len(tied)
                    expected_value_sum += fraction * float(
                        tied["robust_intervention_value"].sum()
                    )
                    expected_oracle_overlap += fraction * sum(
                        candidate in oracle_ids for candidate in tied["candidate_id"]
                    )
                    remaining -= min(remaining, len(tied))
                    if remaining == 0:
                        break
                selected_value = expected_value_sum / k
                score_ranks = scores.rank(method="average")
                target_ranks = label_ranks.rank(method="average")
                has_variation = (
                    score_ranks.nunique() > 1 and target_ranks.nunique() > 1
                )
                spearman = (
                    float(score_ranks.corr(target_ranks)) if has_variation else 0.0
                )
                selection_evaluation = "tie_aware_score_expectation"
            rows.append(
                {
                    "dataset_id": context[0],
                    "network_id": context[1],
                    "anchor_time": context[2],
                    "system_family": animal_system_family(str(context[0])),
                    "method": method,
                    "candidate_count": n,
                    "selection_count": k,
                    "spearman": spearman,
                    "score_unique_count": int(scores.nunique()),
                    "score_has_variation": bool(has_variation),
                    "selection_evaluation": selection_evaluation,
                    "selected_mean_value": selected_value,
                    "oracle_mean_value": oracle,
                    "random_expected_mean_value": random_expected,
                    "value_capture_above_random": (
                        (selected_value - random_expected) / denominator
                        if denominator > 1e-15
                        else np.nan
                    ),
                    "oracle_regret": oracle - selected_value,
                    "top_set_overlap": expected_oracle_overlap / k,
                }
            )
    context_metrics = pd.DataFrame(rows)
    family_metrics = (
        context_metrics.groupby(["system_family", "method"], observed=True)
        .agg(
            anchors=("anchor_time", "size"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            mean_value_capture=("value_capture_above_random", "mean"),
            mean_top_set_overlap=("top_set_overlap", "mean"),
            mean_oracle_regret=("oracle_regret", "mean"),
        )
        .reset_index()
    )
    return context_metrics, family_metrics
