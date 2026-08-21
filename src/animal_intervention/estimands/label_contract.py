from __future__ import annotations

from typing import Any

import pandas as pd

from animal_intervention.estimands.intervention_value import rolling_anchors


LABEL_CONTRACT_VERSION = "2.0.0"

MODEL_READY_LABEL_COLUMNS = (
    "label_contract_version",
    "label_id",
    "dataset_id",
    "network_id",
    "experiment_id",
    "profile",
    "anchor_id",
    "candidate_id",
    "history_start",
    "anchor_time",
    "horizon_end",
    "lookback_seconds",
    "horizon_seconds",
    "eligible_population",
    "outcome_population",
    "primary_mapper",
    "beta_unit",
    "intervention_type",
    "intervention_duration_seconds",
    "random_block_count",
    "introduction_sampling",
    "non_index_introductions_per_candidate_block",
    "non_index_sampling_fraction",
    "self_index_replicates_per_candidate_block",
    "parameter_contexts",
    "robust_intervention_value",
    "minimum_scenario_value",
    "maximum_scenario_value",
    "disease_scenario_sd",
    "mean_random_block_sd",
    "robust_priority_percentile",
    "minimum_priority_percentile",
    "maximum_priority_percentile",
    "robust_rank",
)


def _constant_by_key(
    frame: pd.DataFrame, keys: list[str], columns: list[str]
) -> pd.DataFrame:
    grouped = frame.groupby(keys, observed=True, sort=False)
    for column in columns:
        inconsistent = grouped[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            examples = list(inconsistent.index[inconsistent][:3])
            raise ValueError(f"{column} is inconsistent within label keys: {examples}")
    return grouped[columns].first().reset_index()


def build_model_ready_labels(
    *,
    labels: pd.DataFrame,
    block_estimates: pd.DataFrame,
    stream: Any,
    config: dict[str, Any],
    profile: str,
    anchor_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach cross-dataset semantics and provenance to model-facing labels."""

    labels = labels.copy()
    block_estimates = block_estimates.copy()
    if "network_id" not in labels:
        labels["network_id"] = "all"
    if "network_id" not in block_estimates:
        block_estimates["network_id"] = "all"
    labels["network_id"] = labels["network_id"].astype(str)
    block_estimates["network_id"] = block_estimates["network_id"].astype(str)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    block_estimates["candidate_id"] = block_estimates["candidate_id"].astype(str)
    if labels.duplicated(["network_id", "anchor_id", "candidate_id"]).any():
        raise ValueError("robust labels must be unique by network, anchor, and candidate")

    selected_profile = config["profiles"][profile]
    windows = config["windows"]
    if anchor_metadata is None:
        anchors = rolling_anchors(
            stream,
            lookback=pd.Timedelta(windows["lookback"]),
            horizon=pd.Timedelta(windows["horizon"]),
            step=pd.Timedelta(windows["step"]),
            max_anchors=int(selected_profile["max_anchors"]),
        )
        anchor_frame = pd.DataFrame(
            {
                "network_id": "all",
                "anchor_id": anchor.anchor_id,
                "history_start": anchor.history_start,
                "anchor_time": anchor.anchor_time,
                "horizon_end": anchor.horizon_end,
            }
            for anchor in anchors
        )
    else:
        required_anchor_columns = {
            "network_id",
            "anchor_id",
            "history_start",
            "anchor_time",
            "horizon_end",
        }
        missing_anchor_columns = required_anchor_columns.difference(
            anchor_metadata.columns
        )
        if missing_anchor_columns:
            raise ValueError(
                f"anchor metadata is missing columns: {sorted(missing_anchor_columns)}"
            )
        anchor_frame = anchor_metadata.loc[:, sorted(required_anchor_columns)].copy()
        for column in ("history_start", "anchor_time", "horizon_end"):
            anchor_frame[column] = pd.to_datetime(anchor_frame[column])
        anchor_frame["network_id"] = anchor_frame["network_id"].astype(str)
    required_anchors = set(
        map(tuple, labels[["network_id", "anchor_id"]].drop_duplicates().to_numpy())
    )
    available_anchors = set(
        map(tuple, anchor_frame[["network_id", "anchor_id"]].drop_duplicates().to_numpy())
    )
    if not required_anchors.issubset(available_anchors):
        raise ValueError(
            f"label anchors are not reproducible from the configured windows: "
            f"{sorted(required_anchors - available_anchors)}"
        )

    population = _constant_by_key(
        block_estimates,
        ["network_id", "anchor_id", "candidate_id"],
        [
            "eligible_population",
            "outcome_population",
            "self_index_worlds",
            "non_index_worlds",
        ],
    )
    random_blocks = int(block_estimates["block_id"].nunique())
    if random_blocks < 2:
        raise ValueError("model-ready labels require at least two random blocks")

    model = labels.merge(
        anchor_frame,
        on=["network_id", "anchor_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        population,
        on=["network_id", "anchor_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if model.isna().any().any():
        missing = model.columns[model.isna().any()].tolist()
        raise ValueError(f"model-ready label merge produced missing values: {missing}")

    expected_non_index = model["eligible_population"] - 1
    model["introduction_sampling"] = "uniform_without_replacement"
    model.loc[
        model["non_index_worlds"].eq(expected_non_index), "introduction_sampling"
    ] = "exhaustive"
    model["non_index_sampling_fraction"] = (
        model["non_index_worlds"] / expected_non_index
    )

    dataset_id = str(stream.dataset_id)
    experiment_id = str(config["experiment"]["id"])
    model.insert(0, "profile", profile)
    model.insert(0, "experiment_id", experiment_id)
    model.insert(0, "dataset_id", dataset_id)
    model.insert(
        0,
        "label_id",
        dataset_id
        + "::"
        + model["network_id"]
        + "::"
        + pd.to_datetime(model["anchor_time"]).dt.strftime("%Y%m%dT%H%M%S%f")
        + "::"
        + model["candidate_id"],
    )
    model.insert(0, "label_contract_version", LABEL_CONTRACT_VERSION)
    model["lookback_seconds"] = (
        model["anchor_time"] - model["history_start"]
    ).dt.total_seconds()
    model["horizon_seconds"] = (
        model["horizon_end"] - model["anchor_time"]
    ).dt.total_seconds()
    model["primary_mapper"] = str(stream.metadata.get("mapper", "unknown"))
    model["beta_unit"] = str(stream.metadata.get("beta_unit", "unknown"))
    model["intervention_type"] = str(config["intervention"]["action_type"])
    model["intervention_duration_seconds"] = pd.Timedelta(
        config["intervention"]["duration"]
    ).total_seconds()
    model["random_block_count"] = random_blocks
    model = model.rename(
        columns={
            "non_index_worlds": "non_index_introductions_per_candidate_block",
            "self_index_worlds": "self_index_replicates_per_candidate_block",
        }
    )
    return model.loc[:, MODEL_READY_LABEL_COLUMNS].sort_values(
        ["dataset_id", "network_id", "anchor_time", "candidate_id"],
        ignore_index=True,
    )


def validate_model_ready_labels(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the invariant semantics required before model training."""

    missing_columns = sorted(set(MODEL_READY_LABEL_COLUMNS) - set(frame.columns))
    duplicate_label_ids = (
        int(frame["label_id"].duplicated().sum()) if "label_id" in frame else None
    )
    missing_values = int(frame.isna().sum().sum())
    time_order_valid = bool(
        (
            pd.to_datetime(frame["history_start"])
            < pd.to_datetime(frame["anchor_time"])
        ).all()
        and (
            pd.to_datetime(frame["anchor_time"])
            < pd.to_datetime(frame["horizon_end"])
        ).all()
    )
    value_range_valid = bool(
        frame["robust_intervention_value"].between(-1.0, 1.0).all()
        and frame["robust_priority_percentile"].between(0.0, 1.0).all()
    )
    candidate_coverage = frame.groupby(
        ["dataset_id", "network_id", "anchor_id"], observed=True
    ).agg(
        candidate_count=("candidate_id", "nunique"),
        eligible_population=("eligible_population", "first"),
    )
    candidate_coverage_valid = bool(
        candidate_coverage["candidate_count"]
        .eq(candidate_coverage["eligible_population"])
        .all()
    )
    checks = {
        "required_columns": not missing_columns,
        "no_missing_values": missing_values == 0,
        "unique_label_id": duplicate_label_ids == 0,
        "strict_time_order": time_order_valid,
        "value_ranges": value_range_valid,
        "complete_candidate_coverage": candidate_coverage_valid,
    }
    return {
        "status": "passed" if all(checks.values()) else "needs_revision",
        "checks": checks,
        "rows": int(len(frame)),
        "datasets": int(frame["dataset_id"].nunique()),
        "anchors": int(
            frame[["dataset_id", "network_id", "anchor_time"]]
            .drop_duplicates()
            .shape[0]
        ),
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_label_ids": duplicate_label_ids,
    }
