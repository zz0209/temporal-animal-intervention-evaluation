from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import AnchorWindow, node_support
from animal_intervention.evaluation import aggregate_label_precision
from animal_intervention.simulation import PairedTemporalSIREngine, SIRParameters
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import GroupMixingMapper

from .baboon_validation import _audit_results
from .g1_sim import _git_state, _repository_root, _save_figure, _sha256
from .oxford_predefense import (
    _parameter_grid,
    _run_calibration,
    _run_stability,
    _select_parameters,
    _summaries,
)
from .stability_parallel import run_checkpointed_stability, summarize_stability_worlds


SEASON_BOUNDS = (
    ("winter_2011_12", pd.Timestamp("2011-11-01"), pd.Timestamp("2012-06-01")),
    ("winter_2012_13", pd.Timestamp("2012-11-01"), pd.Timestamp("2013-06-01")),
    ("winter_2013_14", pd.Timestamp("2013-11-01"), pd.Timestamp("2014-06-01")),
)


def _season_id(timestamp: pd.Timestamp) -> str:
    for season, start, end in SEASON_BOUNDS:
        if start <= timestamp < end:
            return season
    raise ValueError(f"Timestamp is outside the three deposited winters: {timestamp}")


def _host_group_stream(
    dataset: CanonicalDataset, host_species_code: str
) -> ExposureStream:
    host_nodes = set(
        dataset.individuals.loc[
            dataset.individuals["species"].eq(host_species_code), "node_id"
        ].astype(str)
    )
    if not host_nodes:
        raise ValueError(f"No individuals have host species code {host_species_code}")
    full = GroupMixingMapper(mode="frequency_dependent").compile(dataset)
    memberships = full.group_memberships.loc[
        full.group_memberships["node_id"].astype(str).isin(host_nodes)
    ].copy()
    group_ids = set(memberships["group_event_id"].astype(str))
    groups = full.group_exposures.loc[
        full.group_exposures["group_event_id"].astype(str).isin(group_ids)
    ].copy()
    stream = ExposureStream(
        dataset_id=full.dataset_id,
        population_nodes=tuple(sorted(host_nodes)),
        group_exposures=groups,
        group_memberships=memberships,
        metadata={
            **full.metadata,
            "host_species_code": host_species_code,
            "population_definition": "all_deposited_host_species_individuals",
        },
    )
    stream.validate()
    return stream


def _weekend_table(stream: ExposureStream) -> pd.DataFrame:
    groups = stream.group_exposures.copy()
    groups["start_time"] = pd.to_datetime(groups["start_time"])
    groups["end_time"] = pd.to_datetime(groups["end_time"])
    groups["season"] = groups["start_time"].map(_season_id)
    groups["weekend_start"] = (
        groups["start_time"].dt.to_period("W-SUN").dt.start_time
    )
    return groups


def _prepare_windows(
    stream: ExposureStream,
    windows: dict[str, Any],
    max_anchors: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    groups = _weekend_table(stream)
    memberships = stream.group_memberships.copy()
    memberships["node_id"] = memberships["node_id"].astype(str)
    history_weekends = int(windows["history_observed_weekends"])
    forecast_indices = [int(value) for value in windows["forecast_weekend_indices"]]
    threshold = int(windows["min_history_events_per_node"])
    prepared: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for season, season_groups in groups.groupby("season", observed=True, sort=True):
        weekend_starts = sorted(season_groups["weekend_start"].unique())
        local_anchor_index = 0
        for forecast_index in forecast_indices:
            if forecast_index < history_weekends or forecast_index >= len(weekend_starts):
                raise ValueError(
                    f"{season} cannot support forecast weekend index {forecast_index}"
                )
            history_starts = weekend_starts[
                forecast_index - history_weekends : forecast_index
            ]
            forecast_start = weekend_starts[forecast_index]
            history_groups = season_groups.loc[
                season_groups["weekend_start"].isin(history_starts)
            ]
            forecast_groups = season_groups.loc[
                season_groups["weekend_start"].eq(forecast_start)
            ]
            history_group_ids = set(history_groups["group_event_id"].astype(str))
            forecast_group_ids = set(forecast_groups["group_event_id"].astype(str))
            history_memberships = memberships.loc[
                memberships["group_event_id"].astype(str).isin(history_group_ids)
            ]
            history_support = (
                history_memberships.groupby("node_id", observed=True).size().astype(int)
            )
            eligible = sorted(
                str(node) for node, count in history_support.items() if count >= threshold
            )
            if len(eligible) < 2:
                raise ValueError(f"{season} forecast index {forecast_index} has <2 candidates")
            eligible_set = set(eligible)
            future_memberships = memberships.loc[
                memberships["group_event_id"].astype(str).isin(forecast_group_ids)
                & memberships["node_id"].isin(eligible_set)
            ].copy()
            retained_sizes = future_memberships.groupby(
                "group_event_id", observed=True
            ).size()
            transmission_group_ids = set(retained_sizes[retained_sizes.ge(2)].index.astype(str))
            future_memberships = future_memberships.loc[
                future_memberships["group_event_id"].astype(str).isin(
                    transmission_group_ids
                )
            ].copy()
            future_groups = forecast_groups.loc[
                forecast_groups["group_event_id"].astype(str).isin(
                    transmission_group_ids
                )
            ].drop(columns=["season", "weekend_start"])
            if future_groups.empty:
                raise ValueError(f"{season} forecast index {forecast_index} has no host dyads")
            future = ExposureStream(
                dataset_id=stream.dataset_id,
                population_nodes=tuple(eligible),
                group_exposures=future_groups,
                group_memberships=future_memberships,
                metadata={
                    **stream.metadata,
                    "network_id": season,
                    "population_definition": "history_eligible_host_species_cohort",
                },
            )
            future.validate()
            anchor_time = pd.Timestamp(forecast_groups["start_time"].min())
            horizon_end = pd.Timestamp(forecast_groups["end_time"].max())
            history_start = pd.Timestamp(history_groups["start_time"].min())
            local_anchor_index += 1
            anchor_id = f"{season}::anchor_{local_anchor_index:03d}"
            anchor = AnchorWindow(
                anchor_id=anchor_id,
                history_start=history_start,
                anchor_time=anchor_time,
                horizon_end=horizon_end,
            )
            future_support = node_support(future)
            future_active = {node for node, count in future_support.items() if count > 0}
            history_weekend_spans = [
                (
                    pd.Timestamp(
                        history_groups.loc[
                            history_groups["weekend_start"].eq(weekend), "start_time"
                        ].min()
                    ),
                    pd.Timestamp(
                        history_groups.loc[
                            history_groups["weekend_start"].eq(weekend), "end_time"
                        ].max()
                    ),
                )
                for weekend in history_starts
            ]
            prepared.append(
                {
                    "anchor": anchor,
                    "future": future,
                    "eligible": eligible,
                    "history_support": history_support,
                    "future_support": future_support,
                    "population_size": len(eligible),
                    "network_id": season,
                    "history_weekend_starts": list(map(pd.Timestamp, history_starts)),
                    "history_weekend_spans": history_weekend_spans,
                    "forecast_weekend_start": pd.Timestamp(forecast_start),
                    "future_active_count": len(future_active),
                }
            )
            metadata_rows.append(
                {
                    "dataset_id": stream.dataset_id,
                    "network_id": season,
                    "anchor_id": anchor_id,
                    "history_start": history_start,
                    "anchor_time": anchor_time,
                    "horizon_end": horizon_end,
                    "history_observed_weekends": history_weekends,
                    "forecast_weekend_start": pd.Timestamp(forecast_start),
                    "eligible_population": len(eligible),
                    "future_active_population": len(future_active),
                    "future_active_fraction": len(future_active) / len(eligible),
                    "transmission_group_events": len(future_groups),
                }
            )
            if len(prepared) >= max_anchors:
                return prepared, pd.DataFrame(metadata_rows)
    return prepared, pd.DataFrame(metadata_rows)


def _data_quality_audit(
    dataset: CanonicalDataset,
    stream: ExposureStream,
    prepared: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    groups = _weekend_table(stream)
    memberships = stream.group_memberships.copy()
    memberships["node_id"] = memberships["node_id"].astype(str)
    sizes = memberships.groupby("group_event_id", observed=True).size().rename("host_group_size")
    groups = groups.merge(sizes, on="group_event_id", how="left", validate="one_to_one")
    groups["date"] = groups["start_time"].dt.floor("D")
    daily = (
        groups.groupby(["season", "date"], observed=True)
        .agg(group_events=("group_event_id", "size"))
        .reset_index()
    )
    daily_active = (
        memberships.merge(
            groups[["group_event_id", "season", "date"]],
            on="group_event_id",
            how="left",
            validate="many_to_one",
        )
        .groupby(["season", "date"], observed=True)["node_id"]
        .nunique()
        .rename("active_host_individuals")
        .reset_index()
    )
    daily = daily.merge(daily_active, on=["season", "date"], validate="one_to_one")
    host_roster = dataset.individuals.loc[
        dataset.individuals["species"].eq(stream.metadata["host_species_code"])
    ]
    duration = (
        pd.to_datetime(groups["end_time"]) - pd.to_datetime(groups["start_time"])
    ).dt.total_seconds()
    season_weekends = groups.groupby("season", observed=True)["weekend_start"].nunique()
    transmission_durations = duration.loc[sizes.reindex(groups["group_event_id"]).ge(2).to_numpy()]
    p99_duration = float(transmission_durations.quantile(0.99))
    checks = {
        "three_seasons": season_weekends.size == 3,
        "thirteen_recording_weekends_per_season": season_weekends.eq(13).all(),
        "positive_group_intervals": duration.gt(0).all(),
        "unique_group_memberships": not memberships.duplicated(
            ["group_event_id", "node_id"]
        ).any(),
        "host_roster_reconciles": len(stream.population_nodes) == len(host_roster),
        "all_windows_stay_within_one_season": all(
            _season_id(item["anchor"].history_start)
            == _season_id(item["anchor"].horizon_end)
            for item in prepared
        ),
        "closed_future_population": all(
            item["future"].nodes() == set(item["eligible"]) for item in prepared
        ),
        "all_forecast_groups_have_two_hosts": all(
            item["future"].group_memberships.groupby("group_event_id").size().ge(2).all()
            for item in prepared
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "status": "passed" if all(checks.values()) else "needs_revision",
        "checks": checks,
        "all_species_individuals": int(len(dataset.individuals)),
        "host_species_roster_individuals": int(len(host_roster)),
        "observed_host_species_individuals": int(memberships["node_id"].nunique()),
        "all_species_group_events": int(len(dataset.group_events)),
        "excluded_nonpositive_all_species_group_events": int(
            stream.metadata["excluded_nonpositive_or_missing_intervals"]
        ),
        "host_present_group_events": int(len(groups)),
        "host_group_memberships": int(len(memberships)),
        "host_singleton_group_events": int(sizes.eq(1).sum()),
        "host_transmission_group_events": int(sizes.ge(2).sum()),
        "recording_dates": int(daily["date"].nunique()),
        "recording_weekends_by_season": {
            key: int(value) for key, value in season_weekends.items()
        },
        "host_group_size": {
            "median": float(sizes.median()),
            "p90": float(sizes.quantile(0.90)),
            "p99": float(sizes.quantile(0.99)),
            "maximum": int(sizes.max()),
        },
        "group_duration_seconds": {
            "median": float(duration.median()),
            "p90": float(duration.quantile(0.90)),
            "p99": float(duration.quantile(0.99)),
            "maximum": float(duration.max()),
        },
        "transmission_event_duration_tail": {
            "p99_seconds": p99_duration,
            "events_over_one_hour": int(transmission_durations.gt(3600).sum()),
            "duration_share_over_one_hour": float(
                transmission_durations.loc[transmission_durations.gt(3600)].sum()
                / transmission_durations.sum()
            ),
            "duration_share_at_or_above_p99": float(
                transmission_durations.loc[transmission_durations.ge(p99_duration)].sum()
                / transmission_durations.sum()
            ),
        },
        "analysis_windows": len(prepared),
        "eligible_population_range": [
            min(len(item["eligible"]) for item in prepared),
            max(len(item["eligible"]) for item in prepared),
        ],
        "future_active_fraction_range": [
            min(item["future_active_count"] / len(item["eligible"]) for item in prepared),
            max(item["future_active_count"] / len(item["eligible"]) for item in prepared),
        ],
        "primary_mapper": stream.metadata["mapper"],
        "group_mixing_mode": stream.metadata["mode"],
        "beta_unit": stream.metadata["beta_unit"],
        "edge_semantics": "inferred RFID co-flocking association, not physical contact",
    }
    group_sizes = sizes.reset_index()
    return audit, daily, group_sizes


def _attach_network(frame: pd.DataFrame, anchor_metadata: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "anchor_id" not in frame:
        return frame
    if "network_id" in frame:
        return frame
    return frame.merge(
        anchor_metadata[["anchor_id", "network_id"]],
        on="anchor_id",
        how="left",
        validate="many_to_one",
    )


def _plot_data_quality(
    daily: pd.DataFrame,
    group_sizes: pd.DataFrame,
    path: Path,
    dataset_label: str = "Wytham great-tit",
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for season, group in daily.groupby("season", observed=True):
        axes[0].plot(group["date"], group["active_host_individuals"], marker="o", label=season)
        axes[1].plot(group["date"], group["group_events"], marker="o", label=season)
    axes[0].set_title("Individual coverage on recording days")
    axes[0].set_ylabel("Active individuals")
    axes[1].set_title("Inferred flock-event volume")
    axes[1].set_ylabel("Group events")
    axes[1].legend(frameon=False, fontsize=8)
    maximum_group_size = int(group_sizes["host_group_size"].max())
    axes[2].hist(
        group_sizes["host_group_size"],
        bins=np.arange(0.5, maximum_group_size + 1.5, 1),
        color="#4C78A8",
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Inferred group-size distribution")
    axes[2].set_xlabel("Tagged individuals in inferred group")
    axes[2].set_ylabel("Group events (log scale)")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=35)
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{dataset_label} primary-stream audit", fontsize=16, weight="bold")
    _save_figure(figure, path)


def _plot_windows(
    prepared: list[dict[str, Any]],
    path: Path,
    dataset_label: str = "Wytham great-tit",
    history_label: str = "four sampled history weekends",
    forecast_label: str = "one forecast weekend",
) -> None:
    seasons = list(
        dict.fromkeys(
            item.get("observation_unit_id", item["network_id"]) for item in prepared
        )
    )
    figure, axes = plt.subplots(
        len(seasons), 1, figsize=(13, 2.35 * len(seasons) + 1.7),
        constrained_layout=True, squeeze=False,
    )
    for axis, season in zip(axes.flat, seasons):
        rows = [
            item
            for item in prepared
            if item.get("observation_unit_id", item["network_id"]) == season
        ]
        for y, item in enumerate(reversed(rows), start=1):
            for observed_start, observed_end in item["history_weekend_spans"]:
                axis.barh(
                    y, observed_end - observed_start, left=observed_start,
                    height=0.42, color="#4C78A8",
                )
            axis.barh(
                y,
                item["anchor"].horizon_end - item["anchor"].anchor_time,
                left=item["anchor"].anchor_time,
                height=0.42,
                color="#F28E2B",
            )
        labels = [item["anchor"].anchor_id.rsplit("::", 1)[-1] for item in reversed(rows)]
        axis.set_yticks(range(1, len(rows) + 1), labels)
        axis.set_title(season, loc="left", fontsize=11, weight="bold")
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="x", rotation=20)
    axes[-1, 0].set_xlabel("Calendar time within each observation unit")
    figure.suptitle(
        f"{dataset_label} observed history and forecast windows\n"
        f"Blue = {history_label}; orange = {forecast_label}; blank gaps are unobserved",
        fontsize=15,
        weight="bold",
    )
    _save_figure(figure, path)


def _plot_calibration(
    summary: pd.DataFrame,
    path: Path,
    dataset_label: str = "Wytham",
) -> None:
    anchors = sorted(summary["anchor_id"].unique())
    columns = min(3, len(anchors))
    rows = int(np.ceil(len(anchors) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.1 * columns, 4.2 * rows), constrained_layout=True, squeeze=False
    )
    image = None
    for axis, anchor_id in zip(axes.flat, anchors):
        selected = summary.loc[summary["anchor_id"].eq(anchor_id)]
        pivot = selected.pivot(
            index="mean_infectious_period_days", columns="beta", values="mean_attack_rate"
        ).sort_index(ascending=False)
        image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
        axis.set_title(anchor_id, fontsize=10)
        axis.set_xlabel("Transmission coefficient")
        axis.set_ylabel("Mean infectious period (days)")
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                value = pivot.iloc[row_index, column_index]
                if pd.notna(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    for axis in axes.flat[len(anchors) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.72, label="Mean final attack rate")
    figure.suptitle(
        f"{dataset_label} epidemic-parameter calibration", fontsize=16, weight="bold"
    )
    _save_figure(figure, path)


def _plot_stability(frame: pd.DataFrame, title: str, top_k: int, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), constrained_layout=True)
    metrics = [
        ("spearman", "Rank correlation", (-1.05, 1.05)),
        ("top_k_overlap_fraction", f"Top-{top_k} overlap", (-0.05, 1.05)),
        ("mean_top_k_value_retention", "Transferred value retention", (-0.05, 1.05)),
    ]
    for index, (axis, (column, label, limits)) in enumerate(zip(axes, metrics)):
        values = frame[column].dropna().sort_values().to_numpy() if column in frame else np.array([])
        axis.scatter(values, np.arange(len(values)), color="#4C78A8", s=24, alpha=0.8)
        if len(values):
            axis.axvline(float(np.median(values)), color="#D55E00", linestyle="--")
        else:
            axis.text(0.5, 0.5, "Not estimable", ha="center", va="center", transform=axis.transAxes)
        axis.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
        axis.set_xlim(*limits)
        axis.set_xticks(
            [-1.0, -0.5, 0.0, 0.5, 1.0]
            if column == "spearman"
            else [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        axis.set_xlabel(label, fontsize=10, labelpad=7)
        axis.set_ylabel("Comparison, ordered" if index == 0 else "")
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title, fontsize=16, weight="bold")
    _save_figure(figure, path)


def _plot_label_distributions(
    labels: pd.DataFrame,
    path: Path,
    dataset_label: str = "Wytham great-tit",
) -> None:
    order = sorted(labels["anchor_id"].unique())
    values = [
        labels.loc[labels["anchor_id"].eq(anchor), "robust_intervention_value"].to_numpy()
        for anchor in order
    ]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.boxplot(
        values,
        tick_labels=order,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35},
    )
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_ylabel("Mean avoided attack rate")
    axis.set_xlabel("Forecast anchor")
    axis.set_title(
        f"Distribution of {dataset_label} singleton isolation values\n"
        "Each box contains all history-eligible birds at one forecast period",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def _evaluation_units(anchor_metadata: pd.DataFrame) -> pd.DataFrame:
    units = anchor_metadata[["dataset_id", "network_id"]].drop_duplicates().copy()
    units["evaluation_unit_id"] = units["dataset_id"] + "::" + units["network_id"]
    units["independent_unit_type"] = "winter_observation_season"
    units["split_constraint"] = "keep_all_anchors_within_one_winter_in_one_fold"
    return units.sort_values("network_id", ignore_index=True)


def _label_precision_diagnostics(
    worlds: pd.DataFrame,
    block_estimates: pd.DataFrame,
    aggregate_separation: pd.DataFrame,
    gates: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auxiliary = (
        block_estimates.groupby(["anchor_id", "candidate_id"], observed=True)
        .agg(
            auxiliary_known_index_value=("known_index_value", "mean"),
            minimum_known_index_value=("known_index_value", "min"),
            maximum_known_index_value=("known_index_value", "max"),
            known_index_contexts=("known_index_value", "size"),
        )
        .reset_index()
    )
    auxiliary["auxiliary_known_index_rank"] = auxiliary.groupby(
        "anchor_id", observed=True
    )["auxiliary_known_index_value"].rank(method="average", ascending=False)

    rows: list[dict[str, Any]] = []
    for anchor_id, anchor_worlds in worlds.groupby("anchor_id", observed=True):
        anchor_blocks = block_estimates.loc[block_estimates["anchor_id"].eq(anchor_id)]
        pivot = (
            anchor_blocks.groupby(["block_id", "candidate_id"], observed=True)[
                "known_index_value"
            ]
            .mean()
            .unstack("block_id")
        )
        known_index_block_spearman = (
            float(pivot.corr(method="spearman").iloc[0, 1])
            if pivot.shape[1] == 2
            else float("nan")
        )
        anchor_precision = aggregate_separation.loc[
            aggregate_separation["anchor_id"].eq(anchor_id)
        ].iloc[0]
        non_index = anchor_worlds.loc[
            anchor_worlds["introduction_stratum"].eq("non_index")
        ]
        known_index = anchor_worlds.loc[
            anchor_worlds["introduction_stratum"].eq("self_index")
        ]
        non_index_counts = non_index.groupby(
            ["parameter_id", "candidate_id"], observed=True
        ).size()
        rank_reliability = float(anchor_precision["averaged_block_rank_reliability"])
        separation = float(anchor_precision["averaged_block_candidate_separation_icc"])
        primary_label_ready = bool(
            rank_reliability >= float(gates["aggregate_label_reliability"])
            and separation
            >= float(gates["aggregate_label_candidate_separation_icc"])
        )
        rows.append(
            {
                "anchor_id": anchor_id,
                "candidate_count": int(anchor_blocks["candidate_id"].nunique()),
                "non_index_worlds_per_candidate_scenario": float(non_index_counts.median()),
                "non_index_zero_fraction": float(non_index["avoided_attack_rate"].eq(0).mean()),
                "known_index_zero_fraction": float(known_index["avoided_attack_rate"].eq(0).mean()),
                "known_index_block_spearman": known_index_block_spearman,
                "primary_rank_reliability": rank_reliability,
                "primary_candidate_separation_icc": separation,
                "primary_label_ready": primary_label_ready,
                "diagnosis": (
                    "passed"
                    if primary_label_ready
                    else "insufficient_non_index_monte_carlo_precision"
                ),
            }
        )
    return auxiliary, pd.DataFrame(rows)


def diagnose_saved_non_index_worlds(
    root: Path, config: dict[str, Any], results_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay unique baseline worlds and attribute singleton zero effects."""

    dataset = CanonicalDataset.read(root / config["data"]["canonical_path"])
    stream = _host_group_stream(dataset, str(config["data"]["host_species_code"]))
    prepared, _ = _prepare_windows(
        stream, config["windows"], int(config["profiles"]["full"]["max_anchors"])
    )
    windows = {item["anchor"].anchor_id: item for item in prepared}
    selections = pd.read_csv(results_dir / "parameter_selection.csv")
    parameters = {
        str(row.parameter_id): SIRParameters(
            beta=float(row.beta),
            recovery_rate=1.0 / (float(row.mean_infectious_period_days) * 86400.0),
        )
        for row in selections.loc[selections["selected"]].itertuples(index=False)
    }
    worlds = pd.read_csv(results_dir / "paired_world_outcomes.csv", dtype={"candidate_id": str, "initial_infected": str})
    non_index = worlds.loc[worlds["introduction_stratum"].eq("non_index")].copy()
    key_columns = [
        "anchor_id",
        "parameter_id",
        "block_id",
        "initial_infected",
        "world_seed",
    ]
    unique_worlds = non_index[key_columns + ["baseline_final_size"]].drop_duplicates()
    replay_rows: list[dict[str, Any]] = []
    engine = PairedTemporalSIREngine()
    for row in unique_worlds.itertuples(index=False):
        window = windows[str(row.anchor_id)]
        anchor = window["anchor"]
        result = engine.simulate(
            window["future"],
            parameters[str(row.parameter_id)],
            initial_infected=[str(row.initial_infected)],
            start_time=anchor.anchor_time,
            end_time=anchor.horizon_end,
            world_seed=int(row.world_seed),
        )
        infected = set(
            result.event_log.loc[
                result.event_log["event"].isin(["initial_infection", "infection"]),
                "node_id",
            ].astype(str)
        )
        transmitters = set(
            result.event_log.loc[
                result.event_log["event"].eq("infection"), "source_id"
            ].dropna().astype(str)
        )
        if result.final_size != int(row.baseline_final_size):
            raise AssertionError("Replayed baseline final size does not match saved result")
        replay_rows.append(
            {
                **{column: getattr(row, column) for column in key_columns},
                "baseline_final_size": result.final_size,
                "infected_nodes": "|".join(sorted(infected)),
                "transmitting_nodes": "|".join(sorted(transmitters)),
            }
        )
    replayed = pd.DataFrame(replay_rows)
    attributed = non_index.merge(replayed, on=key_columns + ["baseline_final_size"], validate="many_to_one")
    attributed["candidate_infected_in_baseline"] = attributed.apply(
        lambda row: str(row["candidate_id"]) in set(str(row["infected_nodes"]).split("|")),
        axis=1,
    )
    attributed["candidate_transmitted_in_baseline"] = attributed.apply(
        lambda row: str(row["candidate_id"]) in set(str(row["transmitting_nodes"]).split("|")),
        axis=1,
    )
    support_lookup = {
        (anchor_id, str(node)): int(count)
        for anchor_id, window in windows.items()
        for node, count in window["future_support"].items()
    }
    attributed["candidate_future_event_support"] = [
        support_lookup.get((str(anchor), str(candidate)), 0)
        for anchor, candidate in zip(attributed["anchor_id"], attributed["candidate_id"])
    ]
    attributed["baseline_no_secondary_infection"] = attributed["baseline_final_size"].eq(1)
    attributed["positive_intervention_effect"] = attributed["avoided_infections"].gt(0)
    summary = (
        attributed.groupby(["parameter_id", "anchor_id"], observed=True)
        .agg(
            candidate_world_rows=("candidate_id", "size"),
            unique_baseline_worlds=("world_seed", "nunique"),
            baseline_no_secondary_fraction=("baseline_no_secondary_infection", "mean"),
            candidate_zero_future_support_fraction=("candidate_future_event_support", lambda value: value.eq(0).mean()),
            candidate_infected_fraction=("candidate_infected_in_baseline", "mean"),
            candidate_transmitted_fraction=("candidate_transmitted_in_baseline", "mean"),
            positive_intervention_effect_fraction=("positive_intervention_effect", "mean"),
        )
        .reset_index()
    )
    return attributed, summary


def run(
    config_path: Path,
    profile: str,
    *,
    stream_builder: Any | None = None,
    window_builder: Any | None = None,
    audit_builder: Any | None = None,
    evaluation_units_builder: Any | None = None,
    data_quality_plotter: Any | None = None,
) -> tuple[Path, Path]:
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.perf_counter()
    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected_profile = config["profiles"][profile]
    experiment_id = str(config["experiment"]["id"])
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = results_dir / "run_manifest.json"
    previous_manifest = (
        json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        if previous_manifest_path.exists()
        else None
    )

    dataset_label = str(config["data"]["display_name"])
    dataset = CanonicalDataset.read(root / config["data"]["canonical_path"])
    stream = (
        stream_builder(dataset, config)
        if stream_builder is not None
        else _host_group_stream(dataset, str(config["data"]["host_species_code"]))
    )
    prepared, anchor_metadata = (
        window_builder(stream, config["windows"], int(selected_profile["max_anchors"]))
        if window_builder is not None
        else _prepare_windows(stream, config["windows"], int(selected_profile["max_anchors"]))
    )
    data_audit, daily, group_sizes = (
        audit_builder(dataset, stream, prepared)
        if audit_builder is not None
        else _data_quality_audit(dataset, stream, prepared)
    )
    if data_audit["status"] != "passed":
        raise ValueError(f"{dataset_label} data-quality gate failed; simulation is not allowed")

    full_grid = _parameter_grid(config["parameter_grid"])
    grid = full_grid.copy()
    parameter_limit = selected_profile["calibration_parameter_limit"]
    if parameter_limit is not None:
        grid = grid.head(int(parameter_limit))
    calibration_source = config["experiment"].get("calibration_source_experiment_id")
    if calibration_source is None:
        calibration_worlds, calibration_summary = _run_calibration(
            prepared,
            grid,
            replicates=int(selected_profile["calibration_replicates_per_index"]),
            index_limit=selected_profile["calibration_index_limit"],
            major_threshold=float(config["parameter_grid"]["major_outbreak_attack_rate"]),
            seed=int(selected_profile["seed"]),
            progress_label=f"{dataset_label} parameter calibration",
        )
        selections = _select_parameters(
            calibration_summary,
            grid,
            config["parameter_grid"],
            min(
                int(selected_profile["selected_scenario_limit"]),
                int(config["parameter_grid"]["max_selected_scenarios"]),
            ),
        )
    else:
        calibration_dir = root / "results" / str(calibration_source) / "full"
        calibration_worlds = pd.read_csv(calibration_dir / "calibration_worlds.csv")
        calibration_summary = pd.read_csv(calibration_dir / "calibration_summary.csv")
        selections = pd.read_csv(calibration_dir / "parameter_selection.csv")
    selected_parameters = grid.loc[
        grid["parameter_id"].isin(selections.loc[selections["selected"], "parameter_id"])
    ].copy()
    execution = config.get("execution", {})
    if execution.get("runner") == "checkpointed_process_pool":
        prefix_config = execution.get("reuse_precision_prefix")
        reused_worlds = pd.DataFrame()
        prepared_to_run = prepared
        if prefix_config:
            prepared_anchor_ids = {
                window["anchor"].anchor_id for window in prepared
            }
            prefix_anchors = set(map(str, prefix_config["anchor_ids"])) & prepared_anchor_ids
            prefix_path = (
                root
                / "results"
                / str(prefix_config["experiment_id"])
                / "full"
                / "paired_world_outcomes.csv"
            )
            reused_worlds = pd.read_csv(
                prefix_path, dtype={"candidate_id": str, "initial_infected": str}
            )
            maximum_position = int(
                selected_profile["non_index_cases_per_candidate_block"]
            )
            reused_worlds = reused_worlds.loc[
                reused_worlds["anchor_id"].isin(prefix_anchors)
                & reused_worlds["parameter_id"].isin(
                    selected_parameters["parameter_id"]
                )
                & reused_worlds["block_id"].lt(int(selected_profile["random_blocks"]))
                & (
                    reused_worlds["introduction_stratum"].eq("self_index")
                    | reused_worlds["introduction_position"].lt(maximum_position)
                )
            ].copy()
            if prefix_anchors and set(reused_worlds["anchor_id"].unique()) != prefix_anchors:
                raise ValueError("precision-prefix source does not cover requested anchors")
            expected_worlds_per_candidate_block = maximum_position + int(
                selected_profile["self_index_replicates_per_block"]
            )
            prefix_counts = reused_worlds.groupby(
                ["anchor_id", "parameter_id", "block_id", "candidate_id"],
                observed=True,
            ).size()
            if not prefix_counts.eq(expected_worlds_per_candidate_block).all():
                raise ValueError("precision-prefix source has incomplete candidate blocks")
            prepared_to_run = [
                window
                for window in prepared
                if window["anchor"].anchor_id not in prefix_anchors
            ]
        generated_worlds, _ = run_checkpointed_stability(
            prepared_to_run,
            selected_parameters,
            config["intervention"],
            random_blocks=int(selected_profile["random_blocks"]),
            non_index_cases=int(selected_profile["non_index_cases_per_candidate_block"]),
            self_replicates=int(selected_profile["self_index_replicates_per_block"]),
            candidate_limit=selected_profile["candidate_limit"],
            seed=int(selected_profile["seed"]),
            checkpoint_dir=results_dir / "checkpoints",
            max_workers=int(selected_profile["max_workers"]),
            progress_label=f"{dataset_label} checkpointed intervention-label simulations",
        )
        worlds = pd.concat([reused_worlds, generated_worlds], ignore_index=True)
        worlds["candidate_id"] = worlds["candidate_id"].astype(str)
        worlds["initial_infected"] = worlds["initial_infected"].astype(str)
        block_estimates = summarize_stability_worlds(worlds)
    else:
        worlds, block_estimates = _run_stability(
            prepared,
            selected_parameters,
            config["intervention"],
            random_blocks=int(selected_profile["random_blocks"]),
            non_index_cases=int(selected_profile["non_index_cases_per_candidate_block"]),
            self_replicates=int(selected_profile["self_index_replicates_per_block"]),
            candidate_limit=selected_profile["candidate_limit"],
            seed=int(selected_profile["seed"]),
            progress_label=f"{dataset_label} intervention-label simulations",
        )
    summaries = _summaries(
        block_estimates,
        int(config["stability"]["top_k"]),
        min(
            int(config["stability"]["consensus_minimum_contexts"]),
            len(prepared) * len(selected_parameters),
        ),
    )
    summaries["exhaustive_reference_comparison"] = pd.DataFrame()
    aggregate_stability, aggregate_separation, aggregate_metrics = aggregate_label_precision(
        block_estimates, int(config["stability"]["top_k"])
    )
    summaries["aggregate_label_random_stability"] = aggregate_stability
    summaries["aggregate_label_candidate_separation"] = aggregate_separation
    summaries["aggregate_label_precision_metrics"] = aggregate_metrics
    auxiliary, precision_diagnostics = _label_precision_diagnostics(
        worlds,
        block_estimates,
        aggregate_separation,
        config["stability"]["gates"],
    )
    summaries["auxiliary_known_index_labels"] = auxiliary
    summaries["label_precision_diagnostics"] = precision_diagnostics
    audit = _audit_results(
        data_audit,
        calibration_summary,
        selections,
        worlds,
        summaries,
        config["stability"]["gates"],
    )
    audit["primary_registry_ready"] = bool(
        audit["status"] == "passed" and precision_diagnostics["primary_label_ready"].all()
    )
    audit["primary_failure_diagnosis"] = (
        None
        if audit["primary_registry_ready"]
        else (
            "insufficient_non_index_monte_carlo_precision"
            if not audit["gate_checks"]["aggregate_label_candidate_separation_icc"]
            else "see_gate_checks"
        )
    )

    anchor_frames = {
        "calibration_worlds": calibration_worlds,
        "calibration_summary": calibration_summary,
        "paired_world_outcomes": worlds,
        "block_estimates": block_estimates,
        **{name: frame for name, frame in summaries.items() if isinstance(frame, pd.DataFrame)},
    }
    frames = {
        "recording_day_coverage": daily,
        "host_group_sizes": group_sizes,
        "anchor_metadata": anchor_metadata,
        "evaluation_units": (
            evaluation_units_builder(anchor_metadata)
            if evaluation_units_builder is not None
            else _evaluation_units(anchor_metadata)
        ),
        "parameter_selection": selections,
        **{
            name: _attach_network(frame, anchor_metadata)
            for name, frame in anchor_frames.items()
        },
    }
    for name, frame in frames.items():
        frame.to_csv(results_dir / f"{name}.csv", index=False)
        if name not in {"calibration_worlds", "paired_world_outcomes"}:
            frame.to_csv(report_dir / f"{name}.csv", index=False)
    for directory in (results_dir, report_dir):
        (directory / "data_quality_audit.json").write_text(
            json.dumps(data_audit, indent=2), encoding="utf-8"
        )
        (directory / "audit_summary.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        (directory / "aggregate_label_precision_metrics.json").write_text(
            json.dumps(aggregate_metrics, indent=2), encoding="utf-8"
        )

    report_config = config.get("report", {})
    if data_quality_plotter is None:
        _plot_data_quality(
            daily,
            group_sizes,
            report_dir / "data_quality_overview.png",
            dataset_label,
        )
    else:
        data_quality_plotter(
            daily,
            group_sizes,
            data_audit,
            report_dir / "data_quality_overview.png",
        )
    _plot_windows(
        prepared,
        report_dir / "timeline.png",
        dataset_label,
        str(report_config.get("history_label", "sampled history periods")),
        str(report_config.get("forecast_label", "one forecast period")),
    )
    _plot_calibration(
        calibration_summary,
        report_dir / "parameter_calibration.png",
        dataset_label,
    )
    top_k = int(config["stability"]["top_k"])
    _plot_stability(
        summaries["aggregate_label_random_stability"],
        f"{dataset_label} final-label random-repeat stability",
        top_k,
        report_dir / "aggregate_label_random_stability.png",
    )
    _plot_stability(
        summaries["parameter_stability"],
        f"{dataset_label} disease-scenario ranking stability",
        top_k,
        report_dir / "parameter_stability.png",
    )
    _plot_stability(
        summaries["temporal_stability"],
        f"{dataset_label} across-window ranking stability",
        top_k,
        report_dir / "temporal_stability.png",
    )
    _plot_label_distributions(
        frames["robust_anchor_labels"],
        report_dir / "label_value_distributions.png",
        dataset_label,
    )

    resolved = {**config, "selected_profile": profile, "run": dict(selected_profile)}
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    (results_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    (report_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    elapsed = time.perf_counter() - started
    initial_started_at = (
        previous_manifest.get(
            "initial_full_compute_started_at_utc",
            previous_manifest.get("started_at_utc", started_at),
        )
        if previous_manifest
        else started_at
    )
    initial_completed_at = (
        previous_manifest.get(
            "initial_full_compute_completed_at_utc",
            previous_manifest.get("completed_at_utc"),
        )
        if previous_manifest
        else datetime.now(UTC).isoformat(timespec="seconds")
    )
    initial_elapsed = (
        previous_manifest.get(
            "initial_full_compute_elapsed_seconds",
            previous_manifest.get("elapsed_seconds", elapsed),
        )
        if previous_manifest
        else elapsed
    )
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "status": "completed",
        "validation_status": audit["status"],
        "data_quality_status": data_audit["status"],
        "artifact_integrity_status": audit["artifact_integrity_status"],
        "started_at_utc": initial_started_at,
        "completed_at_utc": initial_completed_at,
        "elapsed_seconds": initial_elapsed,
        "initial_full_compute_started_at_utc": initial_started_at,
        "initial_full_compute_completed_at_utc": initial_completed_at,
        "initial_full_compute_elapsed_seconds": initial_elapsed,
        "latest_refresh_started_at_utc": started_at,
        "latest_refresh_completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "latest_refresh_elapsed_seconds": elapsed,
        "config_path": config_path.resolve().relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path),
        "canonical_files_sha256": {
            path.name: _sha256(path)
            for path in sorted((root / config["data"]["canonical_path"]).iterdir())
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

    medians = audit["median_metrics"]
    observation_lines = report_config.get(
        "observation_design",
        [
            "Each network unit is an independently sampled observation season.",
            "History-only eligibility precedes one offline epidemic replay period.",
            "Unobserved gaps are not treated as zero-contact epidemic time.",
        ],
    )
    observation_text = "\n".join(f"- {line}" for line in observation_lines)
    roster_individuals = data_audit.get(
        "host_species_roster_individuals",
        data_audit.get("registered_individuals", "not reported"),
    )
    observed_individuals = data_audit.get(
        "observed_host_species_individuals",
        data_audit.get("observed_group_stream_individuals", "not reported"),
    )
    roster_text = (
        f"{roster_individuals:,}"
        if isinstance(roster_individuals, (int, float))
        else str(roster_individuals)
    )
    observed_text = (
        f"{observed_individuals:,}"
        if isinstance(observed_individuals, (int, float))
        else str(observed_individuals)
    )
    aggregate_blocks = int(selected_profile["random_blocks"])
    introduction = str(
        report_config.get(
            "introduction",
            "This run applies the shared prospective singleton-isolation estimand "
            "to author-provided temporal association events.",
        )
    )
    population_noun = str(report_config.get("population_noun", "animals"))
    readme = f"""# {dataset_label} label-generation validation

{introduction}

## Observation design

{observation_text}

## Result

- Validation status: `{audit['status']}`.
- Roster / observed analysis individuals: {roster_text} / {observed_text}.
- Analysis anchors: {len(prepared)} across {anchor_metadata['network_id'].nunique()} analysis populations.
- Eligible population range: {data_audit['eligible_population_range'][0]}–{data_audit['eligible_population_range'][1]} {population_noun} per anchor.
- Calibration simulations: {audit['calibration_simulations']:,}.
- Paired intervention worlds: {audit['paired_worlds']:,}.
- Selected informative scenarios: {audit['informative_selected_parameter_scenarios']} of {audit['selected_parameter_scenarios']} selected.
- Final-label {aggregate_blocks}-block reliability estimate: {medians['aggregate_label_spearman_brown_reliability']}.
- Median disease-scenario rank correlation: {medians['parameter_spearman']}.
- Median across-window rank correlation: {medians['temporal_spearman']}.
- Delivered label candidate-separation ICC: {medians['aggregate_label_mean_candidate_separation_icc']}.

`robust_anchor_labels.csv` contains one continuous simulation-derived intervention
value for every history-eligible {population_noun[:-1] if population_noun.endswith('s') else population_noun} at every anchor. These are model-based
offline targets, not field-observed causal effects. Exact top-20 membership is a
diagnostic; the continuous value is the primary target.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Wytham great-tit data, calibration, and label validation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    results_dir, report_dir = run(args.config, args.profile)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
