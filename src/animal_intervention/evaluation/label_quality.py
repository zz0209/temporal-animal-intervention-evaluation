from __future__ import annotations

import math

import pandas as pd

from .rank_stability import pairwise_rank_stability


def spearman_brown_reliability(single_measure_correlation: float, repeats: int) -> float:
    """Estimate reliability after averaging exchangeable repeated measurements.

    Negative repeat correlations indicate no reproducible signal and are
    conservatively reported as zero. This also avoids the singularity at
    ``r = -1 / (repeats - 1)`` from producing meaningless extreme values.
    """

    if repeats < 2:
        raise ValueError("repeats must be at least two")
    correlation = float(single_measure_correlation)
    if not math.isfinite(correlation):
        return float("nan")
    if correlation <= 0:
        return 0.0
    denominator = 1.0 + (repeats - 1) * correlation
    if denominator <= 0:
        return float("nan")
    return repeats * correlation / denominator


def aggregate_label_precision(
    block_estimates: pd.DataFrame, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | None]]:
    """Audit Monte Carlo precision at the grain of the delivered label."""

    required = {
        "anchor_id",
        "parameter_id",
        "block_id",
        "candidate_id",
        "unconditional_value",
    }
    missing = required.difference(block_estimates.columns)
    if missing:
        raise ValueError(f"block estimates are missing columns: {sorted(missing)}")

    robust_blocks = (
        block_estimates.groupby(
            ["anchor_id", "block_id", "candidate_id"], observed=True, as_index=False
        )["unconditional_value"]
        .mean()
    )
    stability_rows: list[pd.DataFrame] = []
    separation_rows: list[dict[str, float | int | str]] = []
    for anchor_id, group in robust_blocks.groupby("anchor_id", observed=True):
        compared = pairwise_rank_stability(
            group,
            context_columns=["block_id"],
            item_column="candidate_id",
            value_column="unconditional_value",
            top_k=top_k,
        )
        compared["anchor_id"] = anchor_id
        stability_rows.append(compared)

        pivot = group.pivot(
            index="candidate_id", columns="block_id", values="unconditional_value"
        ).dropna()
        block_count = int(pivot.shape[1])
        if block_count < 2 or len(pivot) < 2:
            raise ValueError(
                f"{anchor_id} requires at least two blocks and two candidates"
            )
        mean_square_between = block_count * float(pivot.mean(axis=1).var(ddof=1))
        mean_square_within = float(pivot.var(axis=1, ddof=1).mean())
        denominator = mean_square_between + (block_count - 1) * mean_square_within
        icc = (
            (mean_square_between - mean_square_within) / denominator
            if denominator > 0
            else float("nan")
        )
        mean_icc = (
            (mean_square_between - mean_square_within) / mean_square_between
            if mean_square_between > 0
            else float("nan")
        )
        anchor_correlations = compared["spearman"].dropna()
        anchor_repeat = (
            float(anchor_correlations.median())
            if not anchor_correlations.empty
            else float("nan")
        )
        separation_rows.append(
            {
                "anchor_id": str(anchor_id),
                "candidate_count": int(len(pivot)),
                "block_count": block_count,
                "single_block_rank_correlation": anchor_repeat,
                "averaged_block_rank_reliability": spearman_brown_reliability(
                    anchor_repeat, block_count
                ),
                "single_block_candidate_separation_icc": icc,
                "averaged_block_candidate_separation_icc": mean_icc,
                "aggregate_label_candidate_separation_icc": mean_icc,
            }
        )

    stability = pd.concat(stability_rows, ignore_index=True)
    separation = pd.DataFrame(separation_rows)
    valid_correlations = stability["spearman"].dropna()
    raw_repeat_value = (
        float(valid_correlations.median())
        if not valid_correlations.empty
        else float("nan")
    )
    block_counts = separation["block_count"].unique()
    if len(block_counts) != 1:
        raise ValueError("all anchors must use the same random block count")
    block_count = int(block_counts[0])
    reliability_value = spearman_brown_reliability(raw_repeat_value, block_count)

    def finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    raw_repeat = finite_or_none(raw_repeat_value)
    reliability = finite_or_none(reliability_value)
    metrics: dict[str, float | int | None] = {
        "aggregate_label_single_block_spearman": raw_repeat,
        "aggregate_label_block_count": block_count,
        "aggregate_label_spearman_brown_reliability": reliability,
        "minimum_anchor_spearman_brown_reliability": finite_or_none(
            float(separation["averaged_block_rank_reliability"].min())
        ),
        "maximum_anchor_spearman_brown_reliability": finite_or_none(
            float(separation["averaged_block_rank_reliability"].max())
        ),
        "aggregate_label_candidate_separation_icc": finite_or_none(
            float(separation["aggregate_label_candidate_separation_icc"].median())
        ),
        "aggregate_label_single_block_candidate_separation_icc": finite_or_none(
            float(separation["single_block_candidate_separation_icc"].median())
        ),
        "aggregate_label_mean_candidate_separation_icc": finite_or_none(
            float(separation["averaged_block_candidate_separation_icc"].median())
        ),
    }
    return stability, separation, metrics
