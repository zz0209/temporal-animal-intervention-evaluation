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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import (
    AnchorWindow,
    node_support,
    rolling_anchors,
    slice_stream,
)
from animal_intervention.evaluation import aggregate_label_precision
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import CoalescedDurationContactMapper

from .baboon_validation import _audit_results, _median
from .g1_sim import _git_state, _repository_root, _save_figure, _sha256
from .oxford_predefense import (
    _parameter_grid,
    _run_calibration,
    _run_stability,
    _select_parameters,
    _summaries,
)
from .stability_parallel import run_checkpointed_stability, summarize_stability_worlds


STUDY_PHASE_BY_WEEK = {
    1: "Pre-parasite",
    2: "Pre-patent",
    3: "Pre-patent",
    4: "Pre-patent",
    5: "Patent-parasite",
    6: "Patent-parasite",
    7: "Patent-parasite",
    8: "Post-parasite",
    9: "Post-parasite",
}


def _phase_by_week_start(week_one_start: str | pd.Timestamp) -> dict[object, str]:
    """Build the publication-defined phase calendar from the frozen study start."""

    start = pd.Timestamp(week_one_start).normalize()
    result: dict[object, str] = {}
    for week, phase in STUDY_PHASE_BY_WEEK.items():
        week_start = start + pd.Timedelta(weeks=week - 1)
        for offset in range(7):
            result[(week_start + pd.Timedelta(days=offset)).date()] = phase
    return result


def _load_phase_by_date(raw_path: Path) -> dict[object, str]:
    """Return publication-defined study phases for each experimental date.

    The deposited behavior table contains a legacy ``Phase`` encoding that does
    not match the four phases defined in the associated paper. The animal
    measurement table records study week and the publication-consistent phase,
    so it is the authoritative source for this contextual metadata.
    """

    raw = pd.read_csv(raw_path, usecols=["Date", "Week", "Phase"])
    raw["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="raise")
    raw["week"] = pd.to_numeric(raw["Week"], errors="raise").astype(int)
    study = raw.loc[raw["week"].isin(STUDY_PHASE_BY_WEEK)].copy()
    study["expected_phase"] = study["week"].map(STUDY_PHASE_BY_WEEK)
    inconsistent = study["Phase"].astype(str).ne(study["expected_phase"])
    if inconsistent.any():
        examples = study.loc[inconsistent, ["Date", "Week", "Phase"]].head(3)
        raise ValueError(
            "Animal measurement phases disagree with the publication-defined "
            f"study weeks: {examples.to_dict(orient='records')}"
        )
    phase_counts = study.groupby("week", observed=True)["expected_phase"].nunique()
    if phase_counts.ne(1).any():
        raise ValueError("Animal measurement table assigns multiple phases to one week")
    week_starts = (
        study.groupby("week", observed=True, as_index=False)
        .agg(date=("date", "min"), expected_phase=("expected_phase", "first"))
        .sort_values("week")
    )
    if set(week_starts["week"]) != set(STUDY_PHASE_BY_WEEK):
        raise ValueError("Animal measurement table does not define weeks 1 through 9")
    week_one_start = week_starts.loc[week_starts["week"].eq(1), "date"].iloc[0]
    expected_starts = {
        week: week_one_start + pd.Timedelta(weeks=week - 1)
        for week in STUDY_PHASE_BY_WEEK
    }
    observed_starts = dict(zip(week_starts["week"], week_starts["date"], strict=True))
    if any(observed_starts[week] != expected_starts[week] for week in expected_starts):
        raise ValueError("Animal measurement study weeks are not seven days apart")
    return _phase_by_week_start(week_one_start)


def _substream(stream: ExposureStream, nodes: set[str], network_id: str) -> ExposureStream:
    selected = stream.dyadic_exposures.loc[
        stream.dyadic_exposures["source_id"].isin(nodes)
        & stream.dyadic_exposures["target_id"].isin(nodes)
    ].copy()
    result = ExposureStream(
        dataset_id=stream.dataset_id,
        dyadic_exposures=selected,
        metadata={**stream.metadata, "network_id": network_id},
    )
    result.validate()
    return result


def _prepare_network_windows(
    stream: ExposureStream,
    dataset: CanonicalDataset,
    windows: dict[str, Any],
    max_anchors: int,
    phase_by_date: dict[object, str],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    base_anchors = rolling_anchors(
        stream,
        lookback=pd.Timedelta(windows["lookback"]),
        horizon=pd.Timedelta(windows["horizon"]),
        step=pd.Timedelta(windows["step"]),
        max_anchors=max_anchors,
    )
    roster = dataset.individuals[["node_id", "group_id"]].copy()
    roster["node_id"] = roster["node_id"].astype(str)
    roster["group_id"] = roster["group_id"].astype(str)
    prepared: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for network_id, group in roster.groupby("group_id", observed=True, sort=True):
        nodes = set(group["node_id"])
        network_stream = _substream(stream, nodes, str(network_id))
        for base_anchor in base_anchors:
            anchor = AnchorWindow(
                anchor_id=f"{network_id}::{base_anchor.anchor_id}",
                history_start=base_anchor.history_start,
                anchor_time=base_anchor.anchor_time,
                horizon_end=base_anchor.horizon_end,
            )
            history = slice_stream(
                network_stream, anchor.history_start, anchor.anchor_time
            )
            future = slice_stream(network_stream, anchor.anchor_time, anchor.horizon_end)
            history_support = node_support(history)
            future_support = node_support(future)
            eligible = sorted(
                node
                for node in nodes
                if int(history_support.get(node, 0))
                >= int(windows["min_history_events_per_node"])
            )
            if len(eligible) != len(nodes):
                raise ValueError(
                    f"{anchor.anchor_id} has {len(eligible)} of {len(nodes)} eligible animals"
                )
            future_population = set(eligible) | future.nodes()
            if future_population != nodes:
                raise ValueError(
                    f"{anchor.anchor_id} does not preserve its five-animal cohort"
                )
            prepared.append(
                {
                    "anchor": anchor,
                    "future": future,
                    "eligible": eligible,
                    "history_support": history_support,
                    "future_support": future_support,
                    "population_size": len(nodes),
                    "network_id": str(network_id),
                    "base_anchor_id": base_anchor.anchor_id,
                }
            )
            metadata_rows.append(
                {
                    "network_id": str(network_id),
                    "anchor_id": anchor.anchor_id,
                    "base_anchor_id": base_anchor.anchor_id,
                    "history_start": anchor.history_start,
                    "anchor_time": anchor.anchor_time,
                    "horizon_end": anchor.horizon_end,
                    "phase": phase_by_date.get(anchor.anchor_time.date(), "unknown"),
                    "treatment_class": str(network_id).rsplit(".", 1)[0],
                }
            )
    return prepared, pd.DataFrame(metadata_rows)


def _data_quality_audit(
    dataset: CanonicalDataset,
    stream: ExposureStream,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    roster = dataset.individuals[["node_id", "group_id"]].copy()
    roster["node_id"] = roster["node_id"].astype(str)
    roster["group_id"] = roster["group_id"].astype(str)
    group_sizes = roster.groupby("group_id", observed=True).size().rename("animals")
    node_to_group = roster.set_index("node_id")["group_id"]
    events = stream.dyadic_exposures.copy()
    events["network_id"] = events["source_id"].map(node_to_group)
    target_groups = events["target_id"].map(node_to_group)
    cross_group = int(events["network_id"].ne(target_groups).sum())
    duration = (
        pd.to_datetime(events["end_time"]) - pd.to_datetime(events["start_time"])
    ).dt.total_seconds()
    group_coverage = (
        events.groupby("network_id", observed=True)
        .agg(
            exposure_events=("exposure_id", "size"),
            first_event=("start_time", "min"),
            last_event=("end_time", "max"),
            source_nodes=("source_id", "nunique"),
        )
        .join(group_sizes)
        .reset_index()
    )
    group_coverage["node_union"] = [
        len(
            set(events.loc[events["network_id"].eq(network), "source_id"])
            | set(events.loc[events["network_id"].eq(network), "target_id"])
        )
        for network in group_coverage["network_id"]
    ]
    daily = (
        events.assign(day=pd.to_datetime(events["start_time"]).dt.floor("D"))
        .groupby(["network_id", "day"], observed=True)
        .size()
        .rename("exposure_events")
        .reset_index()
    )
    expected_mapped = (
        int(len(dataset.dyadic_events))
        - int(stream.metadata["excluded_exact_source_duplicates"])
        - int(stream.metadata["merged_source_exposure_count"])
    )
    checks = {
        "canonical_rows_reconcile": int(len(dataset.dyadic_events)) == 299_587,
        "duplicate_exclusion_reconciles": int(
            stream.metadata["input_exposures_before_coalescing"]
        )
        == int(len(dataset.dyadic_events))
        - int(stream.metadata["excluded_exact_source_duplicates"]),
        "coalesced_rows_reconcile": int(len(events)) == expected_mapped,
        "twelve_groups_of_five": len(group_sizes) == 12 and group_sizes.eq(5).all(),
        "no_cross_group_exposures": cross_group == 0,
        "all_group_nodes_observed": group_coverage["node_union"].eq(5).all(),
        "positive_intervals": duration.gt(0).all(),
        "complete_daily_group_coverage": daily.groupby("network_id").size().eq(64).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "status": "passed" if all(checks.values()) else "needs_revision",
        "checks": checks,
        "canonical_dyadic_rows": int(len(dataset.dyadic_events)),
        "excluded_exact_source_duplicates": int(
            stream.metadata["excluded_exact_source_duplicates"]
        ),
        "pre_coalescing_exposures": int(
            stream.metadata["input_exposures_before_coalescing"]
        ),
        "coalesced_primary_exposures": int(len(events)),
        "merged_source_exposure_count": int(
            stream.metadata["merged_source_exposure_count"]
        ),
        "overlapping_or_touching_clusters": int(
            stream.metadata["overlapping_or_touching_clusters"]
        ),
        "overlap_duration_removed_seconds": float(
            stream.metadata["overlap_duration_removed_seconds"]
        ),
        "cross_group_exposures": cross_group,
        "network_count": int(len(group_sizes)),
        "animals_per_network_values": sorted(map(int, group_sizes.unique())),
        "primary_mapper": str(stream.metadata["mapper"]),
        "beta_unit": str(stream.metadata["beta_unit"]),
        "duration_seconds": {
            "minimum": float(duration.min()),
            "median": float(duration.median()),
            "p90": float(duration.quantile(0.9)),
            "p99": float(duration.quantile(0.99)),
            "maximum": float(duration.max()),
        },
    }
    return audit, group_coverage, daily


def _attach_network_columns(
    frame: pd.DataFrame, anchor_metadata: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "network_id",
        "anchor_id",
        "base_anchor_id",
        "phase",
        "treatment_class",
    ]
    return frame.merge(
        anchor_metadata[columns], on="anchor_id", how="left", validate="many_to_one"
    )


def _plot_data_quality(
    stream: ExposureStream,
    group_coverage: pd.DataFrame,
    daily: pd.DataFrame,
    audit: dict[str, Any],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ordered = group_coverage.sort_values("network_id")
    axes[0, 0].barh(ordered["network_id"], ordered["exposure_events"], color="#4C78A8")
    axes[0, 0].set_title("Coalesced exposure events by five-animal social group")
    axes[0, 0].set_xlabel("Exposure intervals")
    duration = (
        pd.to_datetime(stream.dyadic_exposures["end_time"])
        - pd.to_datetime(stream.dyadic_exposures["start_time"])
    ).dt.total_seconds()
    axes[0, 1].hist(duration, bins=np.logspace(0, np.log10(duration.max()), 45), color="#F28E2B")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Encounter duration distribution")
    axes[0, 1].set_xlabel("Seconds (log scale)")
    pivot = daily.pivot(index="network_id", columns="day", values="exposure_events")
    image = axes[1, 0].imshow(pivot, aspect="auto", cmap="Blues")
    axes[1, 0].set_yticks(range(len(pivot.index)), pivot.index)
    axes[1, 0].set_title("Daily event coverage; no empty group-days")
    axes[1, 0].set_xlabel("Study day")
    figure.colorbar(image, ax=axes[1, 0], label="Exposure intervals")
    counts = [
        audit["canonical_dyadic_rows"],
        audit["pre_coalescing_exposures"],
        audit["coalesced_primary_exposures"],
    ]
    axes[1, 1].bar(
        ["Canonical\nrows", "After exact\nduplicates", "After interval\nunion"],
        counts,
        color=["#9C755F", "#59A14F", "#76B7B2"],
    )
    axes[1, 1].set_ylim(min(counts) - 1500, max(counts) + 500)
    axes[1, 1].set_title("Auditable exposure reconciliation")
    for x, value in enumerate(counts):
        axes[1, 1].text(x, value + 80, f"{value:,}", ha="center")
    figure.suptitle("Domestic sheep primary-contact data audit", fontsize=20, fontweight="bold")
    _save_figure(figure, path)


def _plot_timeline(anchor_metadata: pd.DataFrame, path: Path) -> None:
    anchors = anchor_metadata.drop_duplicates("base_anchor_id").sort_values("anchor_time")
    figure, axis = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    y = np.arange(len(anchors))
    for position, row in zip(y, anchors.itertuples(index=False)):
        axis.barh(
            position,
            (row.anchor_time - row.history_start).total_seconds() / 86400,
            left=row.history_start,
            height=0.55,
            color="#4C78A8",
        )
        axis.barh(
            position,
            (row.horizon_end - row.anchor_time).total_seconds() / 86400,
            left=row.anchor_time,
            height=0.55,
            color="#F28E2B",
        )
    phase_column_x = anchors["horizon_end"].max() + pd.Timedelta(days=2)
    axis.axvline(
        phase_column_x - pd.Timedelta(days=1),
        color="#B8B8B8",
        linewidth=0.8,
        linestyle=":",
    )
    for position, row in zip(y, anchors.itertuples(index=False)):
        axis.text(
            phase_column_x,
            position,
            row.phase,
            ha="left",
            va="center",
            fontsize=9,
            color="#333333",
        )
    axis.text(
        phase_column_x,
        -0.62,
        "Study phase at anchor",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#555555",
    )
    axis.set_xlim(
        anchors["history_start"].min(),
        phase_column_x + pd.Timedelta(days=12),
    )
    axis.set_yticks(y, anchors["base_anchor_id"])
    axis.invert_yaxis()
    axis.set_xlabel("Calendar time")
    figure.suptitle(
        "Shared rolling windows for all 12 sheep social groups",
        fontsize=17,
        fontweight="bold",
    )
    axis.text(
        0.5,
        1.015,
        "Blue = 14-day observed history; orange = 7-day future used only for offline labels",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        color="#555555",
    )
    _save_figure(figure, path)


def _plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    aggregate = (
        calibration.groupby(
            ["beta", "mean_infectious_period_days"], observed=True, as_index=False
        )
        .agg(
            mean_attack_rate=("mean_attack_rate", "mean"),
            major_outbreak_probability=("major_outbreak_probability", "mean"),
            no_secondary_infection_probability=(
                "no_secondary_infection_probability",
                "mean",
            ),
        )
    )
    metrics = [
        ("mean_attack_rate", "Mean final attack rate"),
        ("major_outbreak_probability", "Major-outbreak probability"),
        ("no_secondary_infection_probability", "No-secondary-infection probability"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        pivot = aggregate.pivot(
            index="mean_infectious_period_days", columns="beta", values=metric
        ).sort_index(ascending=False)
        image = axis.imshow(pivot, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns], rotation=35, ha="right")
        axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
        axis.set_xlabel("Transmission hazard per contact-second")
        axis.set_ylabel("Mean infectious period (days)")
        axis.set_title(title)
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
                    )
    figure.colorbar(image, ax=axes, label="Probability / proportion", shrink=0.88)
    figure.suptitle("Sheep epidemic-parameter calibration across groups and windows", fontsize=18, fontweight="bold")
    _save_figure(figure, path)


def _plot_label_heatmap(
    labels: pd.DataFrame, anchor_metadata: pd.DataFrame, path: Path
) -> None:
    enriched = labels.merge(
        anchor_metadata[["anchor_id", "anchor_time"]],
        on="anchor_id",
        validate="many_to_one",
    )
    order = (
        enriched[["network_id", "candidate_id"]]
        .drop_duplicates()
        .sort_values(["network_id", "candidate_id"])
    )
    enriched["row_label"] = enriched["network_id"] + " / " + enriched["candidate_id"]
    row_order = (order["network_id"] + " / " + order["candidate_id"]).tolist()
    pivot = enriched.pivot(
        index="row_label", columns="anchor_time", values="robust_intervention_value"
    ).reindex(row_order)
    bound = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 1e-6)
    figure, axis = plt.subplots(figsize=(13, 18), constrained_layout=True)
    image = axis.imshow(
        pivot,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    axis.set_xticks(
        range(len(pivot.columns)),
        [timestamp.strftime("%m-%d") for timestamp in pivot.columns],
        rotation=35,
        ha="right",
    )
    for boundary in range(5, len(pivot), 5):
        axis.axhline(boundary - 0.5, color="white", linewidth=1.5)
    axis.set_xlabel("Intervention anchor")
    axis.set_ylabel("Social group / sheep ID")
    axis.set_title("Dynamic singleton isolation values for all sheep", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Mean avoided attack rate")
    _save_figure(figure, path)


def _plot_stability_summary(summaries: dict[str, pd.DataFrame], path: Path) -> None:
    contexts = [
        ("Random blocks", summaries["aggregate_label_random_stability"]),
        ("Disease scenarios", summaries["parameter_stability"]),
        ("Time windows", summaries["temporal_stability"]),
    ]
    metrics = [
        ("spearman", "Rank correlation"),
        ("top_k_overlap_fraction", "Exact top-2 overlap"),
        ("mean_top_k_value_retention", "Transferred top-2 value retention"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        plotted_values = []
        plotted_positions = []
        plotted_colors = []
        colors = ["#4C78A8", "#F28E2B", "#59A14F"]
        for position, ((_, frame), color) in enumerate(zip(contexts, colors), start=1):
            values = (
                frame[metric].dropna().to_numpy()
                if metric in frame
                else np.array([])
            )
            if len(values):
                plotted_values.append(values)
                plotted_positions.append(position)
                plotted_colors.append(color)
            else:
                axis.text(
                    position,
                    0.5,
                    "Not\nestimable",
                    ha="center",
                    va="center",
                    color="#777777",
                    fontsize=9,
                )
        if plotted_values:
            boxes = axis.boxplot(
                plotted_values,
                positions=plotted_positions,
                patch_artist=True,
                showfliers=False,
            )
            for patch, color in zip(boxes["boxes"], plotted_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
        axis.set_xticks(range(1, len(contexts) + 1), [name for name, _ in contexts])
        axis.set_ylim(-1.05 if metric == "spearman" else -0.05, 1.05)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Sheep intervention-value stability diagnostics", fontsize=18, fontweight="bold")
    _save_figure(figure, path)


def _build_label_quality_annotations(
    labels: pd.DataFrame,
    separation: pd.DataFrame,
    gates: dict[str, Any],
) -> pd.DataFrame:
    """Attach local Monte Carlo precision flags without discarding labels."""

    columns = [
        "anchor_id",
        "single_block_rank_correlation",
        "averaged_block_rank_reliability",
        "single_block_candidate_separation_icc",
        "averaged_block_candidate_separation_icc",
    ]
    annotated = labels.merge(
        separation[columns],
        on="anchor_id",
        how="left",
        validate="many_to_one",
    )
    if annotated[columns[1:]].isna().any().any():
        raise ValueError("Local precision annotations contain missing values")
    reliability_gate = float(gates["aggregate_label_reliability"])
    separation_gate = float(gates["aggregate_label_candidate_separation_icc"])
    annotated["passes_local_rank_reliability"] = annotated[
        "averaged_block_rank_reliability"
    ].ge(reliability_gate)
    annotated["passes_local_candidate_separation"] = annotated[
        "averaged_block_candidate_separation_icc"
    ].ge(separation_gate)
    annotated["local_precision_flag"] = "passes_both"
    annotated.loc[
        ~annotated["passes_local_rank_reliability"]
        & annotated["passes_local_candidate_separation"],
        "local_precision_flag",
    ] = "low_rank_reliability"
    annotated.loc[
        annotated["passes_local_rank_reliability"]
        & ~annotated["passes_local_candidate_separation"],
        "local_precision_flag",
    ] = "low_candidate_separation"
    annotated.loc[
        ~annotated["passes_local_rank_reliability"]
        & ~annotated["passes_local_candidate_separation"],
        "local_precision_flag",
    ] = "low_on_both_diagnostics"
    return annotated.sort_values(
        ["network_id", "base_anchor_id", "candidate_id"], ignore_index=True
    )


def _build_evaluation_units(anchor_metadata: pd.DataFrame) -> pd.DataFrame:
    """Declare independent social groups that must remain intact across folds."""

    units = anchor_metadata[["network_id", "treatment_class"]].drop_duplicates().copy()
    units.insert(0, "dataset_id", "domestic_sheep_sirtrack")
    units["evaluation_unit_id"] = units["dataset_id"] + "::" + units["network_id"]
    units["independent_unit_type"] = "experimental_social_group"
    units["split_constraint"] = "keep_all_animals_and_anchors_in_one_fold"
    return units.sort_values("network_id", ignore_index=True)


def _write_outputs(
    *,
    results_dir: Path,
    report_dir: Path,
    frames: dict[str, pd.DataFrame],
    data_audit: dict[str, Any],
    audit: dict[str, Any],
    config: dict[str, Any],
    profile: str,
    started_at: str,
    elapsed: float,
    root: Path,
    config_path: Path,
    stream: ExposureStream,
) -> None:
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
        (directory / "exposure_metadata.json").write_text(
            json.dumps(stream.metadata, indent=2), encoding="utf-8"
        )
    resolved = {**config, "selected_profile": profile}
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    (results_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    (report_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    medians = audit["median_metrics"]
    def format_metric(name: str) -> str:
        value = medians.get(name)
        return "not estimable" if value is None else f"{float(value):.3f}"

    report = f"""# Domestic sheep intervention-label validation

The 60 lambs are analyzed as 12 independent five-animal social networks. The
primary transmission stream is the time union of overlapping reciprocal Sirtrack
logger intervals; exact source duplicates and overlap reconciliation remain
auditable.

## Audited result

- Data-quality status: `{data_audit['status']}`; {data_audit['canonical_dyadic_rows']:,} canonical rows become {data_audit['coalesced_primary_exposures']:,} non-overlapping primary exposures.
- Independent social networks: {data_audit['network_count']}; animals per network: {data_audit['animals_per_network_values']}.
- Network-anchors: {int(frames['anchor_metadata'].shape[0])}; final label rows: {len(frames['robust_anchor_labels'])}.
- Calibration simulations: {audit['calibration_simulations']:,}; paired worlds: {audit['paired_worlds']:,}.
- Selected informative disease scenarios: {audit['informative_selected_parameter_scenarios']} of {audit['selected_parameter_scenarios']} selected.
- Artifact integrity: `{audit['artifact_integrity_status']}`; final validation: `{audit['status']}`.
- Study-phase metadata: `{audit['study_phase_metadata_status']}`.
- Independent evaluation units: {audit['independent_evaluation_units']} social groups; local precision flags by anchor: {audit['local_precision_flag_counts']}.

## Precision and stability

- Scenario-level single-block rank correlation: {format_metric('random_repeat_spearman')}.
- Final-label single-block rank correlation: {format_metric('aggregate_label_single_block_spearman')}.
- Reliability estimate for the delivered block-mean ranking: {format_metric('aggregate_label_spearman_brown_reliability')}.
- Minimum anchor reliability estimate: {format_metric('minimum_anchor_spearman_brown_reliability')}.
- Scenario-averaged single-block candidate-separation ICC: {format_metric('aggregate_label_single_block_candidate_separation_icc')}.
- Delivered block-mean candidate-separation ICC: {format_metric('aggregate_label_mean_candidate_separation_icc')}.
- Disease-scenario rank correlation: {format_metric('parameter_spearman')}.
- Across-window rank correlation within social groups: {format_metric('temporal_spearman')}.
- Negative paired temporal outcomes retained without clipping: {audit['negative_paired_outcomes']} ({audit['negative_paired_outcome_fraction']:.6f} of paired worlds).

All reliability values are Monte Carlo precision diagnostics, not field-effect
accuracy. Exact top-2 membership remains diagnostic only. Parasite phase and
treatment are context variables from the source study, not transmission-chain
ground truth for this SIR counterfactual. Rare negative paired outcomes can
occur in a temporal SIR process when blocking an early route delays infection
onto a later contact path and changes recovery timing; they are audited rather
than clipped to zero.

`label_quality_annotations.csv` retains every label while exposing local
Monte Carlo reliability and candidate-separation flags. `evaluation_units.csv`
declares the 12 experimental social groups as indivisible train/validation/test
units; rows, animals, or anchors from one group must not be split across folds.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": str(config["experiment"]["id"]),
        "profile": profile,
        "status": "completed",
        "validation_status": audit["status"],
        "data_quality_status": data_audit["status"],
        "artifact_integrity_status": audit["artifact_integrity_status"],
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path),
        "random_seed": int(config["profiles"][profile]["seed"]),
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


def run(config_path: Path, profile: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected_profile = config["profiles"][profile]
    experiment_id = str(config["experiment"]["id"])
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = CanonicalDataset.read(root / config["data"]["canonical_path"])
    stream = CoalescedDurationContactMapper().compile(dataset)
    phase_by_date = _load_phase_by_date(
        root / config["data"]["raw_measurements_path"]
    )
    data_audit, group_coverage, daily = _data_quality_audit(dataset, stream)
    if data_audit["status"] != "passed":
        raise ValueError("sheep data-quality gate failed")
    prepared, anchor_metadata = _prepare_network_windows(
        stream,
        dataset,
        config["windows"],
        int(selected_profile["max_anchors"]),
        phase_by_date,
    )
    grid = _parameter_grid(config["parameter_grid"])
    calibration_source = config["experiment"].get("calibration_source_experiment_id")
    if calibration_source:
        source_dir = root / "results" / str(calibration_source) / "full"
        calibration_worlds = pd.read_csv(
            source_dir / "calibration_worlds.csv", dtype={"initial_infected": str}
        )
        calibration_summary = pd.read_csv(source_dir / "calibration_summary.csv")
        selections = pd.read_csv(source_dir / "parameter_selection.csv")
        metadata_columns = {"network_id", "base_anchor_id", "phase", "treatment_class"}
        calibration_worlds = calibration_worlds.drop(
            columns=list(metadata_columns.intersection(calibration_worlds.columns))
        )
        calibration_summary = calibration_summary.drop(
            columns=list(metadata_columns.intersection(calibration_summary.columns))
        )
    else:
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
            progress_label="Sheep parameter calibration",
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
    selected_parameters = grid.loc[
        grid["parameter_id"].isin(selections.loc[selections["selected"], "parameter_id"])
    ].copy()
    max_workers = selected_profile.get("max_workers")
    if max_workers is None:
        worlds, block_estimates = _run_stability(
            prepared,
            selected_parameters,
            config["intervention"],
            random_blocks=int(selected_profile["random_blocks"]),
            non_index_cases=int(selected_profile["non_index_cases_per_candidate_block"]),
            self_replicates=int(selected_profile["self_index_replicates_per_block"]),
            candidate_limit=selected_profile["candidate_limit"],
            seed=int(selected_profile["seed"]),
            progress_label="Sheep intervention-label simulations",
        )
    else:
        simulation_windows = prepared
        reused_worlds = pd.DataFrame()
        if bool(selected_profile.get("reuse_precision_pilot", False)):
            pilot_source = config["experiment"].get(
                "precision_pilot_source_experiment_id"
            )
            if not pilot_source:
                raise ValueError("precision pilot reuse requested without a source")
            pilot_path = (
                root
                / "results"
                / str(pilot_source)
                / "full"
                / "paired_world_outcomes.csv.gz"
            )
            reused_worlds = pd.read_csv(
                pilot_path, dtype={"candidate_id": str, "initial_infected": str}
            )
            reused_anchors = set(reused_worlds["anchor_id"].astype(str))
            simulation_windows = [
                window
                for window in prepared
                if window["anchor"].anchor_id not in reused_anchors
            ]
            expected_parameters = set(selected_parameters["parameter_id"].astype(str))
            if set(reused_worlds["parameter_id"].astype(str)) != expected_parameters:
                raise ValueError("precision pilot parameter set does not match full run")
            if reused_worlds["block_id"].nunique() != int(
                selected_profile["random_blocks"]
            ):
                raise ValueError("precision pilot block coverage does not match full run")
            expected_reused_rows = (
                len(reused_anchors)
                * len(expected_parameters)
                * int(selected_profile["random_blocks"])
                * 5
                * (
                    int(selected_profile["non_index_cases_per_candidate_block"])
                    + int(selected_profile["self_index_replicates_per_block"])
                )
            )
            if len(reused_worlds) != expected_reused_rows:
                raise ValueError("precision pilot row count does not match full design")
        new_worlds, _ = run_checkpointed_stability(
            simulation_windows,
            selected_parameters,
            config["intervention"],
            random_blocks=int(selected_profile["random_blocks"]),
            non_index_cases=int(selected_profile["non_index_cases_per_candidate_block"]),
            self_replicates=int(selected_profile["self_index_replicates_per_block"]),
            candidate_limit=selected_profile["candidate_limit"],
            seed=int(selected_profile["seed"]),
            checkpoint_dir=results_dir / "checkpoints",
            max_workers=int(max_workers),
            progress_label="Sheep checkpointed label simulations",
        )
        worlds = pd.concat([reused_worlds, new_worlds], ignore_index=True)
        extended_anchors: set[str] = set()
        if bool(selected_profile.get("reuse_residual_precision", False)):
            residual_source = config["experiment"].get(
                "residual_precision_source_experiment_id"
            )
            if not residual_source:
                raise ValueError("residual precision reuse requested without a source")
            residual_worlds = pd.read_csv(
                root
                / "results"
                / str(residual_source)
                / "full"
                / "paired_world_outcomes.csv.gz",
                dtype={"candidate_id": str, "initial_infected": str},
            )
            extended_anchors = set(residual_worlds["anchor_id"].astype(str))
            if not extended_anchors:
                raise ValueError("residual precision source has no anchors")
            if set(residual_worlds["parameter_id"].astype(str)) != set(
                selected_parameters["parameter_id"].astype(str)
            ):
                raise ValueError("residual precision parameter set does not match full run")
            if residual_worlds["block_id"].nunique() <= int(
                selected_profile["random_blocks"]
            ):
                raise ValueError("residual precision source does not extend block coverage")
            worlds = pd.concat(
                [
                    worlds.loc[~worlds["anchor_id"].isin(extended_anchors)],
                    residual_worlds,
                ],
                ignore_index=True,
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
        if worlds.duplicated(world_key).any():
            raise ValueError("combined sheep worlds contain duplicate keys")
        block_estimates = summarize_stability_worlds(worlds)
    worlds = _attach_network_columns(worlds, anchor_metadata)
    block_estimates = _attach_network_columns(block_estimates, anchor_metadata)
    summaries = _summaries(
        block_estimates,
        int(config["stability"]["top_k"]),
        int(config["stability"]["consensus_minimum_contexts"]),
    )
    summaries["robust_anchor_labels"] = _attach_network_columns(
        summaries["robust_anchor_labels"], anchor_metadata
    )
    summaries["exhaustive_reference_comparison"] = pd.DataFrame()
    aggregate_stability, aggregate_separation, aggregate_metrics = (
        aggregate_label_precision(
            block_estimates, int(config["stability"]["top_k"])
        )
    )
    summaries["aggregate_label_random_stability"] = aggregate_stability
    summaries["aggregate_label_candidate_separation"] = aggregate_separation
    summaries["aggregate_label_precision_metrics"] = aggregate_metrics
    audit = _audit_results(
        data_audit,
        calibration_summary,
        selections,
        worlds,
        summaries,
        config["stability"]["gates"],
    )
    audit["network_count"] = int(anchor_metadata["network_id"].nunique())
    audit["network_anchors"] = int(len(anchor_metadata))
    audit["study_phase_metadata_status"] = "publication_definition_verified"
    audit["scientific_readiness_status"] = (
        "qualified_for_grouped_uncertainty_aware_modeling"
    )
    if max_workers is not None:
        audit["baseline_random_blocks"] = int(selected_profile["random_blocks"])
        audit["precision_extended_anchor_count"] = int(len(extended_anchors))
        audit["maximum_random_blocks"] = int(block_estimates["block_id"].max()) + 1

    calibration_summary = _attach_network_columns(
        calibration_summary, anchor_metadata
    )
    calibration_worlds = _attach_network_columns(calibration_worlds, anchor_metadata)
    frames: dict[str, pd.DataFrame] = {
        "group_coverage": group_coverage,
        "daily_group_coverage": daily,
        "anchor_metadata": anchor_metadata,
        "calibration_worlds": calibration_worlds,
        "calibration_summary": calibration_summary,
        "parameter_selection": selections,
        "paired_world_outcomes": worlds,
        **{
            name: frame
            for name, frame in summaries.items()
            if isinstance(frame, pd.DataFrame)
        },
    }
    frames["label_quality_annotations"] = _build_label_quality_annotations(
        summaries["robust_anchor_labels"],
        aggregate_separation,
        config["stability"]["gates"],
    )
    frames["evaluation_units"] = _build_evaluation_units(anchor_metadata)
    audit["local_precision_flag_counts"] = {
        str(key): int(value)
        for key, value in frames["label_quality_annotations"]
        .drop_duplicates("anchor_id")["local_precision_flag"]
        .value_counts()
        .items()
    }
    audit["independent_evaluation_units"] = int(len(frames["evaluation_units"]))
    _plot_data_quality(stream, group_coverage, daily, data_audit, report_dir / "data_quality_overview.png")
    _plot_timeline(anchor_metadata, report_dir / "timeline.png")
    _plot_calibration(calibration_summary, report_dir / "parameter_calibration.png")
    _plot_label_heatmap(summaries["robust_anchor_labels"], anchor_metadata, report_dir / "label_heatmap.png")
    _plot_stability_summary(summaries, report_dir / "stability_summary.png")
    elapsed = time.perf_counter() - started
    _write_outputs(
        results_dir=results_dir,
        report_dir=report_dir,
        frames=frames,
        data_audit=data_audit,
        audit=audit,
        config=config,
        profile=profile,
        started_at=started_at,
        elapsed=elapsed,
        root=root,
        config_path=config_path,
        stream=stream,
    )
    return results_dir, report_dir


def refresh_audit(config_path: Path, profile: str) -> tuple[Path, Path]:
    """Recompute sheep summaries and figures from saved simulation outputs."""

    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile
    required = [
        "anchor_metadata.csv",
        "calibration_summary.csv",
        "parameter_selection.csv",
        "paired_world_outcomes.csv",
        "block_estimates.csv",
        "data_quality_audit.json",
    ]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot refresh audit; missing: {', '.join(missing)}")

    anchor_metadata = pd.read_csv(
        results_dir / "anchor_metadata.csv",
        parse_dates=["history_start", "anchor_time", "horizon_end"],
    )
    phase_by_date = _load_phase_by_date(
        root / config["data"]["raw_measurements_path"]
    )
    anchor_metadata["phase"] = anchor_metadata["anchor_time"].map(
        lambda timestamp: phase_by_date.get(timestamp.date(), "unknown")
    )
    if anchor_metadata["phase"].eq("unknown").any():
        raise ValueError("One or more sheep anchors lack publication-defined phase metadata")
    calibration = pd.read_csv(results_dir / "calibration_summary.csv")
    selections = pd.read_csv(results_dir / "parameter_selection.csv")
    worlds = pd.read_csv(
        results_dir / "paired_world_outcomes.csv",
        dtype={"candidate_id": str, "initial_infected": str},
    )
    block_estimates = pd.read_csv(
        results_dir / "block_estimates.csv", dtype={"candidate_id": str}
    )
    data_audit = json.loads(
        (results_dir / "data_quality_audit.json").read_text(encoding="utf-8")
    )
    summaries = _summaries(
        block_estimates,
        int(config["stability"]["top_k"]),
        int(config["stability"]["consensus_minimum_contexts"]),
    )
    summaries["robust_anchor_labels"] = _attach_network_columns(
        summaries["robust_anchor_labels"], anchor_metadata
    )
    summaries["exhaustive_reference_comparison"] = pd.DataFrame()
    aggregate_stability, aggregate_separation, aggregate_metrics = (
        aggregate_label_precision(
            block_estimates, int(config["stability"]["top_k"])
        )
    )
    summaries["aggregate_label_random_stability"] = aggregate_stability
    summaries["aggregate_label_candidate_separation"] = aggregate_separation
    summaries["aggregate_label_precision_metrics"] = aggregate_metrics
    audit = _audit_results(
        data_audit,
        calibration,
        selections,
        worlds,
        summaries,
        config["stability"]["gates"],
    )
    audit["network_count"] = int(anchor_metadata["network_id"].nunique())
    audit["network_anchors"] = int(len(anchor_metadata))
    audit["study_phase_metadata_status"] = "publication_definition_verified"
    audit["scientific_readiness_status"] = (
        "qualified_for_grouped_uncertainty_aware_modeling"
    )
    quality_annotations = _build_label_quality_annotations(
        summaries["robust_anchor_labels"],
        aggregate_separation,
        config["stability"]["gates"],
    )
    evaluation_units = _build_evaluation_units(anchor_metadata)
    audit["local_precision_flag_counts"] = {
        str(key): int(value)
        for key, value in quality_annotations.drop_duplicates("anchor_id")[
            "local_precision_flag"
        ].value_counts().items()
    }
    audit["independent_evaluation_units"] = int(len(evaluation_units))

    for name, frame in summaries.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        frame.to_csv(results_dir / f"{name}.csv", index=False)
        frame.to_csv(report_dir / f"{name}.csv", index=False)
    for directory in (results_dir, report_dir):
        anchor_metadata.to_csv(directory / "anchor_metadata.csv", index=False)
        quality_annotations.to_csv(
            directory / "label_quality_annotations.csv", index=False
        )
        evaluation_units.to_csv(directory / "evaluation_units.csv", index=False)
        resolved = {**config, "selected_profile": profile}
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    audit_text = json.dumps(audit, indent=2)
    (results_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")
    (report_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")
    _plot_label_heatmap(
        summaries["robust_anchor_labels"],
        anchor_metadata,
        report_dir / "label_heatmap.png",
    )
    _plot_timeline(anchor_metadata, report_dir / "timeline.png")
    _plot_stability_summary(summaries, report_dir / "stability_summary.png")

    medians = audit["median_metrics"]

    def format_metric(name: str) -> str:
        value = medians.get(name)
        return "not estimable" if value is None else f"{float(value):.3f}"

    report = f"""# Domestic sheep intervention-label validation

The 60 lambs are analyzed as 12 independent five-animal social networks. The
primary transmission stream is the time union of overlapping reciprocal Sirtrack
logger intervals; exact source duplicates and overlap reconciliation remain
auditable.

## Audited result

- Data-quality status: `{data_audit['status']}`; {data_audit['canonical_dyadic_rows']:,} canonical rows become {data_audit['coalesced_primary_exposures']:,} non-overlapping primary exposures.
- Independent social networks: {data_audit['network_count']}; animals per network: {data_audit['animals_per_network_values']}.
- Network-anchors: {len(anchor_metadata)}; final label rows: {len(summaries['robust_anchor_labels'])}.
- Calibration simulations: {audit['calibration_simulations']:,}; paired worlds: {audit['paired_worlds']:,}.
- Selected informative disease scenarios: {audit['informative_selected_parameter_scenarios']} of {audit['selected_parameter_scenarios']} selected.
- Artifact integrity: `{audit['artifact_integrity_status']}`; final validation: `{audit['status']}`.
- Study-phase metadata: `{audit['study_phase_metadata_status']}`.
- Independent evaluation units: {audit['independent_evaluation_units']} social groups; local precision flags by anchor: {audit['local_precision_flag_counts']}.

## Precision and stability

- Scenario-level single-block rank correlation: {format_metric('random_repeat_spearman')}.
- Final-label single-block rank correlation: {format_metric('aggregate_label_single_block_spearman')}.
- Reliability estimate for the delivered block-mean ranking: {format_metric('aggregate_label_spearman_brown_reliability')}.
- Minimum anchor reliability estimate: {format_metric('minimum_anchor_spearman_brown_reliability')}.
- Scenario-averaged single-block candidate-separation ICC: {format_metric('aggregate_label_single_block_candidate_separation_icc')}.
- Delivered block-mean candidate-separation ICC: {format_metric('aggregate_label_mean_candidate_separation_icc')}.
- Disease-scenario rank correlation: {format_metric('parameter_spearman')}.
- Across-window rank correlation within social groups: {format_metric('temporal_spearman')}.
- Negative paired temporal outcomes retained without clipping: {audit['negative_paired_outcomes']} ({audit['negative_paired_outcome_fraction']:.6f} of paired worlds).

All reliability values are Monte Carlo precision diagnostics, not field-effect
accuracy. Exact top-2 membership remains diagnostic only. Parasite phase and
treatment are context variables from the source study, not transmission-chain
ground truth for this SIR counterfactual. Rare negative paired outcomes can
occur in a temporal SIR process when blocking an early route delays infection
onto a later contact path and changes recovery timing; they are audited rather
than clipped to zero.

`label_quality_annotations.csv` retains every label while exposing local
Monte Carlo reliability and candidate-separation flags. `evaluation_units.csv`
declares the 12 experimental social groups as indivisible train/validation/test
units; rows, animals, or anchors from one group must not be split across folds.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    manifest_path = results_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_status"] = audit["status"]
        manifest["scientific_readiness_status"] = audit[
            "scientific_readiness_status"
        ]
        manifest["config_sha256"] = _sha256(config_path)
        manifest["audit_refreshed_at_utc"] = datetime.now(UTC).isoformat(
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
    parser = argparse.ArgumentParser(
        description="Generate and audit domestic sheep intervention-value labels"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260815-002_sheep_validation.yaml"),
    )
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--refresh-audit",
        action="store_true",
        help="Recompute summaries and figures from saved simulations",
    )
    args = parser.parse_args()
    if args.refresh_audit:
        refresh_audit(args.config.resolve(), args.profile)
    else:
        run(args.config.resolve(), args.profile)


if __name__ == "__main__":
    main()
