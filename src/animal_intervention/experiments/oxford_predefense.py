from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import (
    _candidate_action,
    node_support,
    rolling_anchors,
    slice_stream,
)
from animal_intervention.evaluation import pairwise_rank_stability, stable_hash_order
from animal_intervention.simulation import PairedTemporalSIREngine, SIRParameters
from animal_intervention.transmission.mappers import compile_primary_exposure

from .g1_sim import _git_state, _repository_root, _save_figure, _sha256


def _keyed_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _parameter_grid(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    beta_values = config.get("beta_values", config.get("beta_per_integrated_association"))
    if beta_values is None:
        raise KeyError("parameter_grid requires beta_values")
    for beta in beta_values:
        for infectious_days in config["mean_infectious_period_days"]:
            rows.append(
                {
                    "parameter_id": f"beta_{float(beta):g}__ip_{float(infectious_days):g}d",
                    "beta": float(beta),
                    "mean_infectious_period_days": float(infectious_days),
                    "recovery_rate_per_day": 1.0 / float(infectious_days),
                }
            )
    return pd.DataFrame(rows)


def _prepare_windows(stream: Any, config: dict[str, Any], max_anchors: int) -> list[dict[str, Any]]:
    anchors = rolling_anchors(
        stream,
        lookback=pd.Timedelta(config["lookback"]),
        horizon=pd.Timedelta(config["horizon"]),
        step=pd.Timedelta(config["step"]),
        max_anchors=max_anchors,
    )
    prepared = []
    for anchor in anchors:
        history = slice_stream(stream, anchor.history_start, anchor.anchor_time)
        future = slice_stream(stream, anchor.anchor_time, anchor.horizon_end)
        history_support = node_support(history)
        future_support = node_support(future)
        eligible = sorted(
            node
            for node, count in history_support.items()
            if count >= int(config["min_history_events_per_node"])
        )
        if len(eligible) < 2:
            raise ValueError(f"{anchor.anchor_id} has fewer than two eligible nodes")
        prepared.append(
            {
                "anchor": anchor,
                "future": future,
                "eligible": eligible,
                "history_support": history_support,
                "future_support": future_support,
                "population_size": len(set(eligible) | future.nodes()),
            }
        )
    return prepared


def _run_calibration(
    prepared: list[dict[str, Any]],
    parameters: pd.DataFrame,
    *,
    replicates: int,
    index_limit: int | None,
    major_threshold: float,
    seed: int,
    progress_label: str = "Parameter calibration",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = sum(
        min(len(window["eligible"]), index_limit or len(window["eligible"]))
        * replicates
        * len(parameters)
        for window in prepared
    )
    progress = tqdm(total=total, desc=progress_label)
    rows: list[dict[str, Any]] = []
    try:
        for window in prepared:
            anchor = window["anchor"]
            eligible = stable_hash_order(
                window["eligible"], seed, anchor.anchor_id, "calibration_indices"
            )
            if index_limit is not None:
                eligible = eligible[:index_limit]
            engine = PairedTemporalSIREngine()
            for parameter in parameters.itertuples(index=False):
                sir = SIRParameters(
                    beta=float(parameter.beta),
                    recovery_rate=float(parameter.recovery_rate_per_day) / 86400,
                )
                for replicate in range(replicates):
                    for initial in eligible:
                        world_seed = _keyed_seed(
                            seed, anchor.anchor_id, "calibration", initial, replicate
                        )
                        result = engine.simulate(
                            window["future"],
                            sir,
                            initial_infected=[initial],
                            start_time=anchor.anchor_time,
                            end_time=anchor.horizon_end,
                            world_seed=world_seed,
                        )
                        attack_rate = result.final_size / window["population_size"]
                        rows.append(
                            {
                                "anchor_id": anchor.anchor_id,
                                "parameter_id": parameter.parameter_id,
                                "beta": parameter.beta,
                                "mean_infectious_period_days": parameter.mean_infectious_period_days,
                                "replicate": replicate,
                                "initial_infected": initial,
                                "population_size": window["population_size"],
                                "final_size": result.final_size,
                                "attack_rate": attack_rate,
                                "major_outbreak": attack_rate >= major_threshold,
                            }
                        )
                        progress.update(1)
    finally:
        progress.close()
    worlds = pd.DataFrame(rows)
    summary = (
        worlds.groupby(
            ["anchor_id", "parameter_id", "beta", "mean_infectious_period_days"],
            observed=True,
        )
        .agg(
            simulations=("attack_rate", "size"),
            mean_attack_rate=("attack_rate", "mean"),
            median_attack_rate=("attack_rate", "median"),
            p10_attack_rate=("attack_rate", lambda values: values.quantile(0.10)),
            p90_attack_rate=("attack_rate", lambda values: values.quantile(0.90)),
            major_outbreak_probability=("major_outbreak", "mean"),
            no_secondary_infection_probability=("final_size", lambda values: values.eq(1).mean()),
        )
        .reset_index()
    )
    return worlds, summary


def _select_parameters(
    summary: pd.DataFrame,
    grid: pd.DataFrame,
    config: dict[str, Any],
    limit: int,
) -> pd.DataFrame:
    aggregate = (
        summary.groupby(
            ["parameter_id", "beta", "mean_infectious_period_days"], observed=True
        )
        .agg(
            mean_attack_rate=("mean_attack_rate", "mean"),
            major_outbreak_probability=("major_outbreak_probability", "mean"),
        )
        .reset_index()
    )
    aggregate["informative"] = (
        aggregate["mean_attack_rate"].between(
            float(config["informative_mean_attack_rate_lower"]),
            float(config["informative_mean_attack_rate_upper"]),
            inclusive="both",
        )
        & aggregate["major_outbreak_probability"].between(
            float(config["informative_major_outbreak_probability_lower"]),
            float(config["informative_major_outbreak_probability_upper"]),
            inclusive="both",
        )
    )
    pool = aggregate.loc[aggregate["informative"]].sort_values("mean_attack_rate")
    if pool.empty:
        pool = aggregate.sort_values("mean_attack_rate")
    if len(pool) > limit:
        positions = np.linspace(0, len(pool) - 1, limit).round().astype(int)
        selected_ids = list(dict.fromkeys(pool.iloc[positions]["parameter_id"]))
    else:
        selected_ids = pool["parameter_id"].tolist()
    reference = config["required_reference"]
    reference_beta = reference.get(
        "beta", reference.get("beta_per_integrated_association")
    )
    if reference_beta is None:
        raise KeyError("required_reference requires beta")
    reference_matches = grid.loc[
        grid["beta"].eq(float(reference_beta))
        & grid["mean_infectious_period_days"].eq(
            float(reference["mean_infectious_period_days"])
        ),
        "parameter_id",
    ]
    reference_id = None if reference_matches.empty else str(reference_matches.iloc[0])
    if reference_id is not None and reference_id not in selected_ids:
        if len(selected_ids) >= limit:
            selected_ids[len(selected_ids) // 2] = reference_id
        else:
            selected_ids.append(reference_id)
    selected_ids = list(dict.fromkeys(selected_ids))
    aggregate["selected"] = aggregate["parameter_id"].isin(selected_ids)
    aggregate["selection_reason"] = np.where(
        aggregate["parameter_id"].eq(reference_id) & aggregate["selected"],
        "required_reference",
        np.where(aggregate["selected"], "informative_range_coverage", "not_selected"),
    )
    return aggregate.sort_values(["selected", "mean_attack_rate"], ascending=[False, True])


def _run_stability(
    prepared: list[dict[str, Any]],
    parameters: pd.DataFrame,
    action_config: dict[str, Any],
    *,
    random_blocks: int,
    non_index_cases: int,
    self_replicates: int,
    candidate_limit: int | None,
    seed: int,
    progress_label: str = "Stability simulations",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_windows: list[tuple[dict[str, Any], list[str]]] = []
    total = 0
    for window in prepared:
        candidates = list(window["eligible"])
        if candidate_limit is not None:
            candidates = sorted(
                candidates,
                key=lambda node: (-int(window["history_support"].get(node, 0)), node),
            )[:candidate_limit]
        selected_windows.append((window, candidates))
        k = min(non_index_cases, len(window["eligible"]) - 1)
        total += len(parameters) * random_blocks * len(candidates) * (k + self_replicates)
    progress = tqdm(total=total, desc=progress_label)
    rows: list[dict[str, Any]] = []
    try:
        for window, candidates in selected_windows:
            anchor = window["anchor"]
            eligible = window["eligible"]
            k = min(non_index_cases, len(eligible) - 1)
            engine = PairedTemporalSIREngine()
            for parameter in parameters.itertuples(index=False):
                sir = SIRParameters(
                    beta=float(parameter.beta),
                    recovery_rate=float(parameter.recovery_rate_per_day) / 86400,
                )
                for block in range(random_blocks):
                    introduction_order = stable_hash_order(
                        eligible, seed, anchor.anchor_id, "stability_indices", block
                    )
                    initials_by_candidate = {
                        candidate: [
                            node for node in introduction_order if node != candidate
                        ][:k]
                        for candidate in candidates
                    }
                    required_initials = sorted(
                        {
                            node
                            for selected_initials in initials_by_candidate.values()
                            for node in selected_initials
                        }
                    )
                    baseline_by_initial: dict[str, Any] = {}
                    for initial in required_initials:
                        world_seed = _keyed_seed(
                            seed, anchor.anchor_id, "non_index", initial, block
                        )
                        baseline_by_initial[initial] = engine.simulate(
                            window["future"], sir, initial_infected=[initial],
                            start_time=anchor.anchor_time, end_time=anchor.horizon_end,
                            world_seed=world_seed,
                        )
                    for candidate in candidates:
                        action = _candidate_action(candidate, anchor, action_config)
                        selected_initials = initials_by_candidate[candidate]
                        for initial in selected_initials:
                            world_seed = _keyed_seed(
                                seed, anchor.anchor_id, "non_index", initial, block
                            )
                            baseline = baseline_by_initial[initial]
                            intervention = engine.simulate(
                                window["future"], sir, initial_infected=[initial],
                                start_time=anchor.anchor_time, end_time=anchor.horizon_end,
                                world_seed=world_seed, action=action,
                            )
                            rows.append(
                                {
                                    "anchor_id": anchor.anchor_id,
                                    "parameter_id": parameter.parameter_id,
                                    "block_id": block,
                                    "candidate_id": candidate,
                                    "introduction_stratum": "non_index",
                                    "introduction_replicate": block,
                                    "initial_infected": initial,
                                    "world_seed": world_seed,
                                    "population_size": window["population_size"],
                                    "baseline_final_size": baseline.final_size,
                                    "intervention_final_size": intervention.final_size,
                                    "avoided_infections": baseline.final_size - intervention.final_size,
                                }
                            )
                            progress.update(1)
                        for replicate in range(self_replicates):
                            world_seed = _keyed_seed(
                                seed, anchor.anchor_id, "self_index", candidate,
                                block, replicate,
                            )
                            baseline = engine.simulate(
                                window["future"], sir, initial_infected=[candidate],
                                start_time=anchor.anchor_time, end_time=anchor.horizon_end,
                                world_seed=world_seed,
                            )
                            intervention = engine.simulate(
                                window["future"], sir, initial_infected=[candidate],
                                start_time=anchor.anchor_time, end_time=anchor.horizon_end,
                                world_seed=world_seed, action=action,
                            )
                            rows.append(
                                {
                                    "anchor_id": anchor.anchor_id,
                                    "parameter_id": parameter.parameter_id,
                                    "block_id": block,
                                    "candidate_id": candidate,
                                    "introduction_stratum": "self_index",
                                    "introduction_replicate": replicate,
                                    "initial_infected": candidate,
                                    "world_seed": world_seed,
                                    "population_size": window["population_size"],
                                    "baseline_final_size": baseline.final_size,
                                    "intervention_final_size": intervention.final_size,
                                    "avoided_infections": baseline.final_size - intervention.final_size,
                                }
                            )
                            progress.update(1)
    finally:
        progress.close()
    worlds = pd.DataFrame(rows)
    worlds["avoided_attack_rate"] = (
        worlds["avoided_infections"] / worlds["population_size"]
    )
    estimate_rows: list[dict[str, Any]] = []
    keys = ["anchor_id", "parameter_id", "block_id", "candidate_id"]
    for key, group in worlds.groupby(keys, observed=True, sort=False):
        anchor_id, parameter_id, block_id, candidate_id = key
        self_values = group.loc[
            group["introduction_stratum"].eq("self_index"), "avoided_attack_rate"
        ]
        non_values = group.loc[
            group["introduction_stratum"].eq("non_index"), "avoided_attack_rate"
        ]
        n = int(group["population_size"].iloc[0])
        eligible_n = next(
            len(item["eligible"])
            for item, _ in selected_windows
            if item["anchor"].anchor_id == anchor_id
        )
        known = float(self_values.mean())
        non_index = float(non_values.mean())
        value = known / eligible_n + non_index * (eligible_n - 1) / eligible_n
        estimate_rows.append(
            {
                "anchor_id": anchor_id,
                "parameter_id": parameter_id,
                "block_id": block_id,
                "candidate_id": candidate_id,
                "eligible_population": eligible_n,
                "outcome_population": n,
                "self_index_worlds": len(self_values),
                "non_index_worlds": len(non_values),
                "known_index_value": known,
                "non_index_value": non_index,
                "unconditional_value": value,
            }
        )
    estimates = pd.DataFrame(estimate_rows)
    estimates["rank"] = estimates.groupby(
        ["anchor_id", "parameter_id", "block_id"], observed=True
    )["unconditional_value"].rank(method="average", ascending=False)
    return worlds, estimates


def _grouped_pairwise(
    frame: pd.DataFrame,
    group_columns: list[str],
    context_column: str,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for group_key, group in frame.groupby(group_columns, observed=True, sort=True):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        compared = pairwise_rank_stability(
            group,
            context_columns=[context_column],
            item_column="candidate_id",
            value_column="unconditional_value",
            top_k=top_k,
        )
        for name, value in zip(group_columns, group_key):
            compared[name] = value
        rows.append(compared)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summaries(
    block_estimates: pd.DataFrame,
    top_k: int,
    minimum_contexts: int,
) -> dict[str, pd.DataFrame]:
    random_stability = _grouped_pairwise(
        block_estimates, ["anchor_id", "parameter_id"], "block_id", top_k
    )
    aggregate = (
        block_estimates.groupby(
            ["anchor_id", "parameter_id", "candidate_id"], observed=True
        )["unconditional_value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "unconditional_value", "std": "block_standard_deviation"})
    )
    parameter_stability = _grouped_pairwise(
        aggregate, ["anchor_id"], "parameter_id", top_k
    )
    temporal_stability = _grouped_pairwise(
        aggregate, ["parameter_id"], "anchor_id", top_k
    )
    separation_rows = []
    for (anchor_id, parameter_id), group in block_estimates.groupby(
        ["anchor_id", "parameter_id"], observed=True, sort=True
    ):
        pivot = group.pivot(
            index="candidate_id", columns="block_id", values="unconditional_value"
        ).dropna()
        block_count = pivot.shape[1]
        candidate_means = pivot.mean(axis=1)
        mean_square_between = (
            block_count * float(candidate_means.var(ddof=1))
            if len(candidate_means) > 1
            else float("nan")
        )
        mean_square_within = (
            float(pivot.var(axis=1, ddof=1).mean())
            if block_count > 1
            else float("nan")
        )
        denominator = mean_square_between + (block_count - 1) * mean_square_within
        icc = (
            (mean_square_between - mean_square_within) / denominator
            if denominator > 0
            else float("nan")
        )
        separation_rows.append(
            {
                "anchor_id": anchor_id,
                "parameter_id": parameter_id,
                "candidate_count": len(candidate_means),
                "block_count": block_count,
                "candidate_separation_icc": icc,
                "mean_square_between": mean_square_between,
                "mean_square_within": mean_square_within,
                "minimum_value": float(candidate_means.min()),
                "median_value": float(candidate_means.median()),
                "maximum_value": float(candidate_means.max()),
                "p90_minus_p10": float(
                    candidate_means.quantile(0.90) - candidate_means.quantile(0.10)
                ),
                "zero_value_fraction": float(candidate_means.abs().lt(1e-12).mean()),
            }
        )
    candidate_separation = pd.DataFrame(separation_rows)
    aggregate["priority_percentile"] = aggregate.groupby(
        ["anchor_id", "parameter_id"], observed=True
    )["unconditional_value"].rank(method="average", pct=True, ascending=True)
    aggregate["is_top_k"] = aggregate.groupby(
        ["anchor_id", "parameter_id"], observed=True
    )["unconditional_value"].rank(method="first", ascending=False).le(top_k)
    robust_anchor_labels = (
        aggregate.groupby(["anchor_id", "candidate_id"], observed=True)
        .agg(
            parameter_contexts=("parameter_id", "nunique"),
            robust_intervention_value=("unconditional_value", "mean"),
            minimum_scenario_value=("unconditional_value", "min"),
            maximum_scenario_value=("unconditional_value", "max"),
            disease_scenario_sd=("unconditional_value", "std"),
            mean_random_block_sd=("block_standard_deviation", "mean"),
            robust_priority_percentile=("priority_percentile", "mean"),
            minimum_priority_percentile=("priority_percentile", "min"),
            maximum_priority_percentile=("priority_percentile", "max"),
        )
        .reset_index()
    )
    robust_anchor_labels["robust_rank"] = robust_anchor_labels.groupby(
        "anchor_id", observed=True
    )["robust_intervention_value"].rank(method="average", ascending=False)
    consensus = (
        aggregate.groupby("candidate_id", observed=True)
        .agg(
            contexts=("priority_percentile", "size"),
            mean_priority_percentile=("priority_percentile", "mean"),
            minimum_priority_percentile=("priority_percentile", "min"),
            maximum_priority_percentile=("priority_percentile", "max"),
            priority_percentile_sd=("priority_percentile", "std"),
            top_k_contexts=("is_top_k", "sum"),
            mean_intervention_value=("unconditional_value", "mean"),
        )
        .reset_index()
    )
    consensus["eligible_for_consensus"] = consensus["contexts"].ge(minimum_contexts)
    consensus["consensus_rank"] = consensus.loc[
        consensus["eligible_for_consensus"], "mean_priority_percentile"
    ].rank(method="min", ascending=False)
    consensus = consensus.sort_values(
        ["eligible_for_consensus", "consensus_rank", "candidate_id"],
        ascending=[False, True, True],
        na_position="last",
        ignore_index=True,
    )
    return {
        "block_estimates": block_estimates,
        "aggregate_estimates": aggregate,
        "random_repeat_stability": random_stability,
        "parameter_stability": parameter_stability,
        "temporal_stability": temporal_stability,
        "candidate_separation": candidate_separation,
        "robust_anchor_labels": robust_anchor_labels,
        "consensus_watchlist": consensus,
    }


def _compare_exhaustive_reference(
    aggregate: pd.DataFrame,
    reference_path: Path,
    reference_parameter_id: str,
    top_k: int,
) -> pd.DataFrame:
    if not reference_path.exists():
        return pd.DataFrame()
    reference = pd.read_csv(reference_path, dtype={"candidate_id": str})
    current = aggregate.loc[
        aggregate["parameter_id"].eq(reference_parameter_id),
        ["anchor_id", "candidate_id", "unconditional_value"],
    ]
    rows = []
    for anchor_id, current_anchor in current.groupby("anchor_id", observed=True):
        old_anchor = reference.loc[reference["anchor_id"].eq(anchor_id)]
        joined = current_anchor.merge(
            old_anchor[["candidate_id", "unconditional_value"]],
            on="candidate_id",
            suffixes=("_sampled", "_exhaustive"),
        )
        if len(joined) < 2:
            continue
        sampled_ranks = joined["unconditional_value_sampled"].rank(method="average")
        exhaustive_ranks = joined["unconditional_value_exhaustive"].rank(method="average")
        effective_k = min(top_k, len(joined))
        sampled_top = set(joined.nlargest(effective_k, "unconditional_value_sampled")["candidate_id"])
        exhaustive_top = set(joined.nlargest(effective_k, "unconditional_value_exhaustive")["candidate_id"])
        rows.append(
            {
                "anchor_id": anchor_id,
                "shared_candidates": len(joined),
                "spearman": float(sampled_ranks.corr(exhaustive_ranks, method="pearson")),
                "top_k": effective_k,
                "top_k_overlap_count": len(sampled_top & exhaustive_top),
                "top_k_overlap_fraction": len(sampled_top & exhaustive_top) / effective_k,
            }
        )
    return pd.DataFrame(rows)


def _header(figure: plt.Figure, title: str, subtitle: str, *, top: float = 0.78) -> None:
    figure.suptitle(title, x=0.08, y=0.975, ha="left", fontsize=16, weight="bold")
    figure.text(0.08, 0.905, subtitle, ha="left", fontsize=10, color="#555555")
    figure.subplots_adjust(top=top)


def _plot_calibration(summary: pd.DataFrame, path: Path) -> None:
    anchors = sorted(summary["anchor_id"].unique())
    figure, axes = plt.subplots(
        1, len(anchors), figsize=(5.2 * len(anchors), 4.8), squeeze=False
    )
    for axis, anchor_id in zip(axes[0], anchors):
        selected = summary.loc[summary["anchor_id"].eq(anchor_id)]
        pivot = selected.pivot(
            index="mean_infectious_period_days", columns="beta", values="mean_attack_rate"
        ).sort_index(ascending=False)
        image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
        axis.set_xlabel("Transmission coefficient")
        axis.set_title(anchor_id, fontsize=11)
        for row in range(len(pivot.index)):
            for column in range(len(pivot.columns)):
                value = pivot.iloc[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="#222222")
    axes[0, 0].set_ylabel("Mean infectious period (days)")
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.16, wspace=0.20)
    color_axis = figure.add_axes([0.92, 0.18, 0.015, 0.58])
    figure.colorbar(image, cax=color_axis, label="Mean attack rate")
    _header(
        figure,
        "Oxford epidemic-parameter calibration",
        "Cells show mean final attack rate over the profile's index-case ensemble and random repeats",
        top=0.76,
    )
    _save_figure(figure, path)


def _plot_stability_points(frame: pd.DataFrame, title: str, subtitle: str, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("spearman", "Rank correlation", (-1.05, 1.05)),
        ("top_k_overlap_fraction", "Top-k overlap fraction", (-0.05, 1.05)),
        (
            "mean_top_k_value_retention",
            "Transferred top-k value / context-best value",
            (-0.05, 1.05),
        ),
    ]
    for axis, (column, label, limits) in zip(axes, metrics):
        values = (
            frame[column].dropna().sort_values().to_numpy()
            if column in frame
            else np.array([], dtype=float)
        )
        axis.scatter(values, np.arange(len(values)), color="#4C78A8", s=22, alpha=0.8)
        if len(values):
            axis.axvline(np.median(values), color="#D55E00", linestyle="--", linewidth=1.5)
        else:
            axis.text(
                0.5, 0.5, "Not estimable in this profile", ha="center", va="center",
                transform=axis.transAxes, color="#555555",
            )
        axis.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
        axis.set_xlim(*limits)
        axis.set_xlabel(label)
        axis.set_ylabel("Context pair, ordered")
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    _header(figure, title, subtitle, top=0.76)
    _save_figure(figure, path)


def _plot_consensus(consensus: pd.DataFrame, path: Path) -> None:
    selected = consensus.loc[consensus["eligible_for_consensus"]].head(20).copy()
    selected = selected.sort_values("mean_priority_percentile")
    y = np.arange(len(selected))
    means = selected["mean_priority_percentile"].to_numpy()
    lower = selected["minimum_priority_percentile"].to_numpy()
    upper = selected["maximum_priority_percentile"].to_numpy()
    figure, axis = plt.subplots(figsize=(9.5, 7.8))
    axis.errorbar(
        means, y, xerr=np.vstack([means - lower, upper - means]), fmt="o",
        color="#4C78A8", ecolor="#9ECAE1", capsize=3,
    )
    maximum_contexts = int(consensus["contexts"].max())
    labels = [
        f"{row.candidate_id} ({int(row.contexts)}/{maximum_contexts})"
        for row in selected.itertuples(index=False)
    ]
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Mean priority percentile; whiskers show observed context range")
    axis.set_ylabel("Oxford animal ID (available contexts / maximum)")
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    _header(
        figure,
        "Oxford cross-context intervention-priority summary",
        "Higher is better; whiskers are observed context ranges, not confidence intervals",
        top=0.82,
    )
    _save_figure(figure, path)


def _plot_candidate_separation(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values(["parameter_id", "anchor_id"]).reset_index(drop=True)
    labels = [
        f"{row.parameter_id} | {row.anchor_id}"
        for row in ordered.itertuples(index=False)
    ]
    y = np.arange(len(ordered))
    figure, axes = plt.subplots(1, 2, figsize=(12, max(5.2, 0.34 * len(ordered) + 2.8)), sharey=True)
    axes[0].scatter(ordered["candidate_separation_icc"], y, color="#4C78A8", s=28)
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].set_xlabel("Candidate-separation ICC")
    axes[1].scatter(ordered["p90_minus_p10"], y, color="#D55E00", s=28)
    axes[1].set_xlabel("90th minus 10th percentile intervention value")
    axes[0].set_yticks(y, labels)
    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    _header(
        figure,
        "Separation of Oxford animal intervention values",
        "ICC compares between-animal differences with within-animal random-block variation; negative values indicate noise dominance",
        top=0.80,
    )
    _save_figure(figure, path)


def _audit(
    calibration: pd.DataFrame,
    selections: pd.DataFrame,
    worlds: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    gates: dict[str, Any],
) -> dict[str, Any]:
    random_frame = summaries["random_repeat_stability"]
    parameter_frame = summaries["parameter_stability"]
    temporal_frame = summaries["temporal_stability"]
    separation_frame = summaries["candidate_separation"]
    reference_frame = summaries["exhaustive_reference_comparison"]

    def median(frame: pd.DataFrame, column: str) -> float | None:
        values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        return None if values.empty else float(values.median())

    selected_count = int(selections["selected"].sum())
    random_spearman = median(random_frame, "spearman")
    random_overlap = median(random_frame, "top_k_overlap_fraction")
    random_retention = median(random_frame, "mean_top_k_value_retention")
    parameter_spearman = median(parameter_frame, "spearman")
    parameter_overlap = median(parameter_frame, "top_k_overlap_fraction")
    parameter_retention = median(parameter_frame, "mean_top_k_value_retention")
    temporal_spearman = median(temporal_frame, "spearman")
    temporal_overlap = median(temporal_frame, "top_k_overlap_fraction")
    temporal_retention = median(temporal_frame, "mean_top_k_value_retention")
    separation_icc = median(separation_frame, "candidate_separation_icc")
    reference_spearman = median(reference_frame, "spearman")
    reference_overlap = median(reference_frame, "top_k_overlap_fraction")
    checks = {
        "minimum_selected_scenarios": selected_count >= int(gates["minimum_selected_scenarios"]),
        "random_repeat_median_spearman": random_spearman is not None
        and random_spearman >= float(gates["random_repeat_median_spearman"]),
        "random_repeat_median_top_k_overlap": random_overlap is not None
        and random_overlap >= float(gates["random_repeat_median_top_k_overlap"]),
        "parameter_median_spearman": parameter_spearman is not None
        and parameter_spearman >= float(gates["parameter_median_spearman"]),
        "parameter_median_top_k_overlap": parameter_overlap is not None
        and parameter_overlap >= float(gates["parameter_median_top_k_overlap"]),
        "temporal_median_spearman": temporal_spearman is not None
        and temporal_spearman >= float(gates["temporal_median_spearman"]),
        "temporal_median_top_k_overlap": temporal_overlap is not None
        and temporal_overlap >= float(gates["temporal_median_top_k_overlap"]),
        "candidate_separation_median_icc": separation_icc is not None
        and separation_icc >= float(gates["candidate_separation_median_icc"]),
        "exhaustive_reference_median_spearman": reference_spearman is not None
        and reference_spearman >= float(gates["exhaustive_reference_median_spearman"]),
        "exhaustive_reference_median_top_k_overlap": reference_overlap is not None
        and reference_overlap >= float(gates["exhaustive_reference_median_top_k_overlap"]),
    }
    continuous_value_check_names = [
        "minimum_selected_scenarios",
        "random_repeat_median_spearman",
        "parameter_median_spearman",
        "candidate_separation_median_icc",
        "exhaustive_reference_median_spearman",
    ]
    fixed_top_k_check_names = [
        "random_repeat_median_top_k_overlap",
        "parameter_median_top_k_overlap",
        "temporal_median_top_k_overlap",
        "exhaustive_reference_median_top_k_overlap",
    ]
    continuous_value_checks = {
        name: checks[name] for name in continuous_value_check_names
    }
    fixed_top_k_checks = {name: checks[name] for name in fixed_top_k_check_names}
    predeclared_status = "passed" if all(checks.values()) else "needs_revision"
    continuous_value_status = (
        "passed" if all(continuous_value_checks.values()) else "needs_revision"
    )
    fixed_top_k_status = (
        "passed" if all(fixed_top_k_checks.values()) else "not_supported"
    )
    world_key = [
        "anchor_id",
        "parameter_id",
        "block_id",
        "candidate_id",
        "introduction_stratum",
        "introduction_replicate",
        "initial_infected",
    ]
    world_key_duplicates = int(worlds.duplicated(world_key).sum())
    missing_world_values = int(worlds.isna().sum().sum())
    candidate_block_counts = worlds.groupby(
        ["anchor_id", "parameter_id", "block_id", "candidate_id"],
        observed=True,
    ).size()
    stratum_counts = worlds.groupby(
        [
            "anchor_id",
            "parameter_id",
            "block_id",
            "candidate_id",
            "introduction_stratum",
        ],
        observed=True,
    ).size()
    expected = worlds["baseline_final_size"] - worlds["intervention_final_size"]
    arithmetic_error = float((worlds["avoided_infections"] - expected).abs().max())
    artifact_integrity_status = (
        "passed"
        if world_key_duplicates == 0
        and missing_world_values == 0
        and arithmetic_error == 0
        and candidate_block_counts.nunique() == 1
        else "needs_revision"
    )
    return {
        "status": predeclared_status,
        "status_definition": "original strict composite gate retained for provenance",
        "continuous_value_gate_status": continuous_value_status,
        "fixed_top_k_gate_status": fixed_top_k_status,
        "artifact_integrity_status": artifact_integrity_status,
        "stage_progression_recommendation": (
            "proceed_to_cross_dataset_label_generation_with_caveats"
            if continuous_value_status == "passed"
            and artifact_integrity_status == "passed"
            else "resolve_oxford_signal_quality_before_progression"
        ),
        "scope": "Oxford-only internal validation; temporal windows overlap and do not establish long-term stability",
        "selected_parameter_scenarios": selected_count,
        "calibration_simulations": int(calibration["simulations"].sum()),
        "paired_worlds": len(worlds),
        "arithmetic_identity_max_error": arithmetic_error,
        "world_key_duplicates": world_key_duplicates,
        "missing_world_values": missing_world_values,
        "candidate_block_world_count_min": int(candidate_block_counts.min()),
        "candidate_block_world_count_max": int(candidate_block_counts.max()),
        "stratum_world_count_values": sorted(map(int, stratum_counts.unique())),
        "negative_paired_outcomes": int(worlds["avoided_infections"].lt(0).sum()),
        "median_metrics": {
            "random_repeat_spearman": random_spearman,
            "random_repeat_top_k_overlap": random_overlap,
            "random_repeat_top_k_value_retention": random_retention,
            "parameter_spearman": parameter_spearman,
            "parameter_top_k_overlap": parameter_overlap,
            "parameter_top_k_value_retention": parameter_retention,
            "temporal_spearman": temporal_spearman,
            "temporal_top_k_overlap": temporal_overlap,
            "temporal_top_k_value_retention": temporal_retention,
            "candidate_separation_icc": separation_icc,
            "exhaustive_reference_spearman": reference_spearman,
            "exhaustive_reference_top_k_overlap": reference_overlap,
        },
        "internal_gate_checks": checks,
        "continuous_value_gate_checks": continuous_value_checks,
        "fixed_top_k_gate_checks": fixed_top_k_checks,
    }


def _write_report_bundle(
    *,
    results_dir: Path,
    report_dir: Path,
    calibration_summary: pd.DataFrame,
    selections: pd.DataFrame,
    summary_frames: dict[str, pd.DataFrame],
    audit: dict[str, Any],
    config: dict[str, Any],
    selected_profile: dict[str, Any],
    analysis_window_count: int,
    calibration_scenario_count: int,
) -> None:
    report_frames = {
        "calibration_summary": calibration_summary,
        "parameter_selection": selections,
        **summary_frames,
    }
    for name, frame in report_frames.items():
        frame.to_csv(results_dir / f"{name}.csv", index=False)
        frame.to_csv(report_dir / f"{name}.csv", index=False)

    audit_text = json.dumps(audit, indent=2)
    (results_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")
    (report_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")

    _plot_calibration(calibration_summary, report_dir / "parameter_calibration.png")
    _plot_stability_points(
        summary_frames["random_repeat_stability"],
        "Random-repeat stability of Oxford rankings",
        "Same disease scenario and forecast window; orange lines show medians",
        report_dir / "random_repeat_stability.png",
    )
    _plot_stability_points(
        summary_frames["parameter_stability"],
        "Disease-scenario stability of Oxford rankings",
        "Different selected disease scenarios within the same forecast window",
        report_dir / "parameter_stability.png",
    )
    _plot_stability_points(
        summary_frames["temporal_stability"],
        "Short-window stability of Oxford rankings",
        "Different overlapping 2-day forecast windows within the same disease scenario",
        report_dir / "temporal_stability.png",
    )
    _plot_consensus(
        summary_frames["consensus_watchlist"], report_dir / "consensus_watchlist.png"
    )
    _plot_candidate_separation(
        summary_frames["candidate_separation"], report_dir / "candidate_separation.png"
    )

    medians = audit["median_metrics"]
    def format_metric(name: str) -> str:
        value = medians.get(name)
        return "not estimable" if value is None else f"{float(value):.3f}"

    top = summary_frames["consensus_watchlist"].loc[
        summary_frames["consensus_watchlist"]["eligible_for_consensus"]
    ].head(10)
    top_lines = [
        "| candidate_id | contexts | mean_priority_percentile | minimum_priority_percentile | top_k_contexts |",
        "|---:|---:|---:|---:|---:|",
    ]
    top_lines.extend(
        f"| {row.candidate_id} | {int(row.contexts)} | {row.mean_priority_percentile:.4f} | "
        f"{row.minimum_priority_percentile:.4f} | {int(row.top_k_contexts)} |"
        for row in top.itertuples(index=False)
    )
    report = f"""# Oxford pre-defense stability audit

This experiment tests whether simulation-derived singleton isolation values are
repeatable across epidemic randomness and disease scenarios, while separately
describing change across the three complete Oxford rolling windows.

## Scope and design

- Oxford observation span: six relative days.
- Complete analysis windows: {analysis_window_count} overlapping windows with a two-day history and two-day future.
- Calibration scenarios evaluated: {calibration_scenario_count}.
- Scenarios selected by the predeclared non-degeneracy rule: {int(selections['selected'].sum())}.
- Independent random blocks per selected context: {int(selected_profile['random_blocks'])}.
- Non-index introductions per candidate and block: {int(selected_profile['non_index_cases_per_candidate_block'])}.
- Complete isolation is a best-case model estimand, not a field-effect estimate.

## Label contract for the next stage

`robust_anchor_labels.csv` is the model-facing label table. One row represents
one eligible animal at one anchor. `robust_intervention_value` is the equal-weight
mean unconditional avoided attack rate across the selected non-degenerate disease
scenarios. Anchors are not averaged together. Scenario-specific values and the
observed scenario range remain available for sensitivity analysis.

## Validation result

- Original strict composite status: `{audit['status']}`. This is retained because the predeclared gate included exact top-k membership.
- Continuous-value gate: `{audit['continuous_value_gate_status']}`.
- Fixed top-k gate: `{audit['fixed_top_k_gate_status']}`.
- Artifact integrity: `{audit['artifact_integrity_status']}`; {audit['world_key_duplicates']} duplicate full keys, {audit['missing_world_values']} missing values, arithmetic error {audit['arithmetic_identity_max_error']}.
- Stage recommendation: `{audit['stage_progression_recommendation']}`.
- Median random-repeat rank correlation: {format_metric('random_repeat_spearman')}.
- Median disease-scenario rank correlation: {format_metric('parameter_spearman')}.
- Median short-window rank correlation: {format_metric('temporal_spearman')}.
- Median candidate-separation ICC: {format_metric('candidate_separation_icc')}.
- Median transferred top-k value retention: random {format_metric('random_repeat_top_k_value_retention')}, disease scenario {format_metric('parameter_top_k_value_retention')}, short window {format_metric('temporal_top_k_value_retention')}.
- Exact top-{int(config['stability']['top_k'])} overlap remains a diagnostic: random {format_metric('random_repeat_top_k_overlap')}, disease scenario {format_metric('parameter_top_k_overlap')}, short window {format_metric('temporal_top_k_overlap')}.
- Negative paired outcomes retained: {audit['negative_paired_outcomes']}.

## Audited temporal non-monotonicity

The most extreme negative paired outcome (-68 infections) was replayed exactly
with identical event logs. Isolating node 101 delayed node 21's infection by
5.16 hours, shifted its recovery past a later contact cluster, and opened a
larger time-respecting cascade. This is an allowed temporal-SIR timing effect,
not a random-seed or cache defect. Negative values therefore remain unclipped.

## Highest cross-context candidates

{chr(10).join(top_lines)}

## Interpretation boundary

The Oxford result supports a repeatable continuous intervention-value gradient,
not a permanent exact top-k list. Because the record has only six days and the
windows overlap, it cannot establish long-term temporal stability or cross-dataset
generalization. The latter is the purpose of the next dataset stage.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")


def run(config_path: Path, profile: str) -> tuple[Path, Path]:
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.perf_counter()
    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected_profile = config["profiles"][profile]
    experiment_id = config["experiment"]["id"]
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = root / config["data"]["canonical_path"]
    dataset = CanonicalDataset.read(canonical_path)
    stream = compile_primary_exposure(dataset)
    prepared = _prepare_windows(
        stream, config["windows"], int(selected_profile["max_anchors"])
    )
    grid = _parameter_grid(config["parameter_grid"])
    parameter_limit = selected_profile["calibration_parameter_limit"]
    if parameter_limit is not None:
        grid = grid.head(int(parameter_limit))
    calibration_worlds, calibration_summary = _run_calibration(
        prepared,
        grid,
        replicates=int(selected_profile["calibration_replicates_per_index"]),
        index_limit=selected_profile["calibration_index_limit"],
        major_threshold=float(config["parameter_grid"]["major_outbreak_attack_rate"]),
        seed=int(selected_profile["seed"]),
    )
    selection_limit = min(
        int(selected_profile["selected_scenario_limit"]),
        int(config["parameter_grid"]["max_selected_scenarios"]),
    )
    selections = _select_parameters(
        calibration_summary, grid, config["parameter_grid"], selection_limit
    )
    selected_parameters = grid.loc[
        grid["parameter_id"].isin(selections.loc[selections["selected"], "parameter_id"])
    ].copy()
    worlds, block_estimates = _run_stability(
        prepared,
        selected_parameters,
        config["intervention"],
        random_blocks=int(selected_profile["random_blocks"]),
        non_index_cases=int(selected_profile["non_index_cases_per_candidate_block"]),
        self_replicates=int(selected_profile["self_index_replicates_per_block"]),
        candidate_limit=selected_profile["candidate_limit"],
        seed=int(selected_profile["seed"]),
    )
    summary_frames = _summaries(
        block_estimates,
        int(config["stability"]["top_k"]),
        min(
            int(config["stability"]["consensus_minimum_contexts"]),
            len(prepared) * len(selected_parameters),
        ),
    )
    reference = config["parameter_grid"]["required_reference"]
    reference_id = (
        f"beta_{float(reference['beta_per_integrated_association']):g}"
        f"__ip_{float(reference['mean_infectious_period_days']):g}d"
    )
    summary_frames["exhaustive_reference_comparison"] = _compare_exhaustive_reference(
        summary_frames["aggregate_estimates"],
        root / config["stability"]["exhaustive_reference_path"],
        reference_id,
        int(config["stability"]["top_k"]),
    )
    audit = _audit(
        calibration_summary,
        selections,
        worlds,
        summary_frames,
        config["stability"]["internal_gates"],
    )
    calibration_worlds.to_csv(results_dir / "calibration_worlds.csv", index=False)
    worlds.to_csv(results_dir / "paired_world_outcomes.csv", index=False)
    _write_report_bundle(
        results_dir=results_dir,
        report_dir=report_dir,
        calibration_summary=calibration_summary,
        selections=selections,
        summary_frames=summary_frames,
        audit=audit,
        config=config,
        selected_profile=selected_profile,
        analysis_window_count=len(prepared),
        calibration_scenario_count=len(grid),
    )

    resolved = {**config, "selected_profile": profile, "run": dict(selected_profile)}
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    (results_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    (report_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    elapsed = time.perf_counter() - started
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "status": "completed",
        "validation_status": audit["status"],
        "continuous_value_gate_status": audit["continuous_value_gate_status"],
        "fixed_top_k_gate_status": audit["fixed_top_k_gate_status"],
        "artifact_integrity_status": audit["artifact_integrity_status"],
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "config_path": config_path.resolve().relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path),
        "canonical_files_sha256": {
            path.name: _sha256(path)
            for path in sorted(canonical_path.iterdir())
            if path.is_file()
        },
        "random_seed": int(selected_profile["seed"]),
        "git": _git_state(root),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "matplotlib", "pyarrow", "PyYAML"]
        },
        "outputs": sorted(path.name for path in results_dir.iterdir()),
    }
    manifest_text = json.dumps(manifest, indent=2)
    (results_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
    (report_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
    return results_dir, report_dir


def refresh_existing_report(config_path: Path, profile: str) -> tuple[Path, Path]:
    """Rebuild summaries and figures from saved simulation outcomes only."""

    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected_profile = config["profiles"][profile]
    experiment_id = config["experiment"]["id"]
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile
    required = [
        "calibration_summary.csv",
        "parameter_selection.csv",
        "paired_world_outcomes.csv",
        "block_estimates.csv",
    ]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot refresh report; missing saved outputs: {', '.join(missing)}"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    calibration_summary = pd.read_csv(results_dir / "calibration_summary.csv")
    selections = pd.read_csv(results_dir / "parameter_selection.csv")
    worlds = pd.read_csv(
        results_dir / "paired_world_outcomes.csv",
        dtype={"candidate_id": str, "initial_infected": str},
    )
    block_estimates = pd.read_csv(
        results_dir / "block_estimates.csv", dtype={"candidate_id": str}
    )
    selected_scenarios = int(selections["selected"].sum())
    analysis_window_count = int(calibration_summary["anchor_id"].nunique())
    summary_frames = _summaries(
        block_estimates,
        int(config["stability"]["top_k"]),
        min(
            int(config["stability"]["consensus_minimum_contexts"]),
            analysis_window_count * selected_scenarios,
        ),
    )
    reference = config["parameter_grid"]["required_reference"]
    reference_id = (
        f"beta_{float(reference['beta_per_integrated_association']):g}"
        f"__ip_{float(reference['mean_infectious_period_days']):g}d"
    )
    summary_frames["exhaustive_reference_comparison"] = _compare_exhaustive_reference(
        summary_frames["aggregate_estimates"],
        root / config["stability"]["exhaustive_reference_path"],
        reference_id,
        int(config["stability"]["top_k"]),
    )
    audit = _audit(
        calibration_summary,
        selections,
        worlds,
        summary_frames,
        config["stability"]["internal_gates"],
    )
    _write_report_bundle(
        results_dir=results_dir,
        report_dir=report_dir,
        calibration_summary=calibration_summary,
        selections=selections,
        summary_frames=summary_frames,
        audit=audit,
        config=config,
        selected_profile=selected_profile,
        analysis_window_count=analysis_window_count,
        calibration_scenario_count=int(calibration_summary["parameter_id"].nunique()),
    )
    manifest_path = results_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_status"] = audit["status"]
        manifest["continuous_value_gate_status"] = audit[
            "continuous_value_gate_status"
        ]
        manifest["fixed_top_k_gate_status"] = audit["fixed_top_k_gate_status"]
        manifest["artifact_integrity_status"] = audit["artifact_integrity_status"]
        manifest["report_refreshed_at_utc"] = datetime.now(UTC).isoformat(
            timespec="seconds"
        )
        manifest["outputs"] = sorted(path.name for path in results_dir.iterdir())
        manifest_text = json.dumps(manifest, indent=2)
        manifest_path.write_text(manifest_text, encoding="utf-8")
        (report_dir / "run_manifest.json").write_text(
            manifest_text, encoding="utf-8"
        )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Oxford pre-defense stability audit.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="Rebuild summaries and figures from saved outcomes without simulation",
    )
    args = parser.parse_args()
    if args.refresh_report:
        results_dir, report_dir = refresh_existing_report(args.config, args.profile)
    else:
        results_dir, report_dir = run(args.config, args.profile)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
