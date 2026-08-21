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
from animal_intervention.evaluation import (
    aggregate_label_precision,
    pairwise_rank_stability,
    stable_hash_order,
)
from animal_intervention.simulation import PairedTemporalSIREngine, SIRParameters
from animal_intervention.transmission.mappers import compile_primary_exposure

from .g1_sim import _git_state, _repository_root, _save_figure, _sha256
from .g1_sim_balanced import _cumulative
from .oxford_predefense import (
    _parameter_grid,
    _prepare_windows,
    _run_calibration,
    _run_stability,
    _select_parameters,
    _summaries,
)


def _median(frame: pd.DataFrame, column: str) -> float | None:
    values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
    return None if values.empty else float(values.median())


def _data_quality_audit(
    dataset: CanonicalDataset,
    stream: Any,
    raw_proximity_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_proximity_path, sep="\t")
    events = dataset.dyadic_events
    proximity = events.loc[events["edge_semantics"].eq("sensor_proximity")].copy()
    behavior = events.loc[events["edge_semantics"].eq("direct_behavior")].copy()
    exposures = stream.dyadic_exposures.copy()
    endpoint_nodes = set(exposures["source_id"].astype(str)) | set(
        exposures["target_id"].astype(str)
    )
    starts = pd.to_datetime(exposures["start_time"])
    ends = pd.to_datetime(exposures["end_time"])
    daily = (
        exposures.assign(day=starts.dt.floor("D"))
        .groupby("day", observed=True)
        .agg(
            exposure_events=("exposure_id", "size"),
            source_nodes=("source_id", "nunique"),
            target_nodes=("target_id", "nunique"),
        )
        .reset_index()
    )
    daily["active_nodes"] = [
        len(
            set(exposures.loc[starts.dt.floor("D").eq(day), "source_id"].astype(str))
            | set(exposures.loc[starts.dt.floor("D").eq(day), "target_id"].astype(str))
        )
        for day in daily["day"]
    ]
    pairs = (
        exposures.groupby(["source_id", "target_id"], observed=True)
        .agg(
            detected_bins=("exposure_id", "size"),
            detected_seconds=(
                "exposure_id",
                lambda values: float(len(values) * 20),
            ),
        )
        .reset_index()
    )
    raw_times = pd.to_datetime(raw["t"], unit="s", errors="coerce")
    printed_times = pd.to_datetime(raw["DateTime"], dayfirst=True, errors="coerce")
    timezone_offsets = (
        printed_times - raw_times.dt.floor("min")
    ).dt.total_seconds()
    expected_local_starts = printed_times + pd.to_timedelta(
        pd.to_numeric(raw["t"], errors="coerce") % 60, unit="s"
    )
    canonical_starts = pd.to_datetime(
        proximity.sort_values("source_record_id", key=lambda values: values.astype(int))["start_time"]
    ).reset_index(drop=True)
    canonical_timestamp_matches = canonical_starts.eq(expected_local_starts.reset_index(drop=True))
    canonical_durations = (ends - starts).dt.total_seconds()
    audit = {
        "status": "passed",
        "raw_proximity_rows": int(len(raw)),
        "canonical_proximity_rows": int(len(proximity)),
        "primary_exposure_rows": int(len(exposures)),
        "canonical_behavior_rows_retained_separately": int(len(behavior)),
        "canonical_all_modality_individuals": int(len(dataset.individuals)),
        "primary_endpoint_union_individuals": int(len(endpoint_nodes)),
        "expected_tagged_individuals": 13,
        "raw_to_canonical_row_reconciliation": bool(len(raw) == len(proximity)),
        "canonical_to_exposure_row_reconciliation": bool(len(proximity) == len(exposures)),
        "raw_timezone_offset_seconds_values": sorted(
            float(value) for value in timezone_offsets.dropna().unique()
        ),
        "canonical_timestamp_match_rows": int(canonical_timestamp_matches.sum()),
        "canonical_timestamp_mismatch_rows": int((~canonical_timestamp_matches).sum()),
        "missing_primary_timestamps": int((starts.isna() | ends.isna()).sum()),
        "nonpositive_primary_intervals": int((canonical_durations <= 0).sum()),
        "duplicate_primary_event_ids": int(exposures["exposure_id"].duplicated().sum()),
        "duplicate_pair_start_rows": int(
            exposures.duplicated(["source_id", "target_id", "start_time"]).sum()
        ),
        "calendar_days_with_events": int(daily["day"].nunique()),
        "calendar_span_days": int((daily["day"].max() - daily["day"].min()).days + 1),
        "minimum_daily_active_nodes": int(daily["active_nodes"].min()),
        "maximum_daily_active_nodes": int(daily["active_nodes"].max()),
        "time_start": starts.min().isoformat(),
        "time_end": ends.max().isoformat(),
        "primary_mapper": stream.metadata.get("mapper"),
        "beta_unit": stream.metadata.get("beta_unit"),
        "primary_modality": "sensor_proximity",
        "behavior_modality_pooled": False,
    }
    required = [
        audit["raw_to_canonical_row_reconciliation"],
        audit["canonical_to_exposure_row_reconciliation"],
        audit["raw_timezone_offset_seconds_values"] == [7200.0],
        audit["canonical_timestamp_mismatch_rows"] == 0,
        audit["missing_primary_timestamps"] == 0,
        audit["nonpositive_primary_intervals"] == 0,
        audit["duplicate_primary_event_ids"] == 0,
        audit["duplicate_pair_start_rows"] == 0,
        audit["primary_endpoint_union_individuals"] == audit["expected_tagged_individuals"],
        audit["calendar_days_with_events"] == audit["calendar_span_days"],
    ]
    audit["status"] = "passed" if all(required) else "needs_revision"
    return audit, daily, pairs


def _plot_data_quality(
    daily: pd.DataFrame, pairs: pd.DataFrame, audit: dict[str, Any], path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].bar(daily["day"], daily["exposure_events"], color="#4C78A8")
    axes[0].set_title("Daily proximity-event volume")
    axes[0].set_ylabel("20-second detections")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].plot(daily["day"], daily["active_nodes"], marker="o", color="#D55E00")
    axes[1].set_ylim(0, audit["expected_tagged_individuals"] + 1)
    axes[1].set_title("Daily tagged-animal coverage")
    axes[1].set_ylabel("Endpoint-union animals")
    axes[1].tick_params(axis="x", rotation=35)
    axes[2].hist(pairs["detected_bins"], bins=18, color="#72B7B2")
    axes[2].set_title("Dyad exposure heterogeneity")
    axes[2].set_xlabel("Detected 20-second bins per dyad")
    axes[2].set_ylabel("Dyads")
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Guinea baboon primary-stream data audit",
        fontsize=16,
        weight="bold",
    )
    _save_figure(figure, path)


def _plot_windows(prepared: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    for index, window in enumerate(prepared):
        anchor = window["anchor"]
        y = len(prepared) - index
        axis.barh(
            y,
            anchor.anchor_time - anchor.history_start,
            left=anchor.history_start,
            color="#4C78A8",
            height=0.5,
        )
        axis.barh(
            y,
            anchor.horizon_end - anchor.anchor_time,
            left=anchor.anchor_time,
            color="#F28E2B",
            height=0.5,
        )
    axis.set_yticks(
        range(1, len(prepared) + 1),
        [window["anchor"].anchor_id for window in prepared][::-1],
    )
    axis.set_xlabel("Calendar time")
    axis.set_title(
        "Baboon rolling history and future-label windows\n"
        "Blue: observed history for eligibility/features; orange: future replay used only to construct labels",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=14,
    )
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.autofmt_xdate()
    _save_figure(figure, path)


def _plot_calibration(summary: pd.DataFrame, path: Path) -> None:
    anchors = sorted(summary["anchor_id"].unique())
    columns = 3
    rows = int(np.ceil(len(anchors) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(14.5, 4.2 * rows), constrained_layout=True, squeeze=False
    )
    image = None
    for axis, anchor_id in zip(axes.flat, anchors):
        selected = summary.loc[summary["anchor_id"].eq(anchor_id)]
        pivot = selected.pivot(
            index="mean_infectious_period_days", columns="beta", values="mean_attack_rate"
        ).sort_index(ascending=False)
        image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        probabilities = 1.0 - np.exp(-20.0 * pivot.columns.to_numpy(dtype=float))
        axis.set_xticks(range(len(pivot.columns)), [f"{value:.4f}" for value in probabilities])
        axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
        axis.set_title(anchor_id)
        axis.set_xlabel("Transmission probability per 20-second detection")
        axis.set_ylabel("Mean infectious period (days)")
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                axis.text(
                    column_index,
                    row_index,
                    f"{pivot.iloc[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    for axis in axes.flat[len(anchors):]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.72, label="Mean final attack rate")
    figure.suptitle("Guinea baboon epidemic-parameter calibration", fontsize=16, weight="bold")
    _save_figure(figure, path)


def _plot_stability(frame: pd.DataFrame, title: str, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    metrics = [
        ("spearman", "Rank correlation", (-1.05, 1.05)),
        ("top_k_overlap_fraction", "Top-3 overlap", (-0.05, 1.05)),
        ("mean_top_k_value_retention", "Transferred value retention", (-0.05, 1.05)),
    ]
    for axis, (column, label, limits) in zip(axes, metrics):
        values = (
            frame[column].dropna().sort_values().to_numpy()
            if column in frame
            else np.array([], dtype=float)
        )
        axis.scatter(values, np.arange(len(values)), color="#4C78A8", s=20, alpha=0.8)
        if len(values):
            axis.axvline(np.median(values), color="#D55E00", linestyle="--")
        else:
            axis.text(
                0.5,
                0.5,
                "Not estimable in this profile",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#555555",
            )
        axis.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
        axis.set_xlim(*limits)
        axis.set_xlabel(label)
        axis.set_ylabel("Comparison, ordered")
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title, fontsize=16, weight="bold")
    _save_figure(figure, path)


def _plot_label_heatmap(labels: pd.DataFrame, path: Path) -> None:
    pivot = labels.pivot(
        index="candidate_id", columns="anchor_id", values="robust_intervention_value"
    )
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    figure, axis = plt.subplots(
        figsize=(10.5, max(5.8, 0.42 * len(pivot))), constrained_layout=True
    )
    image = axis.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_xlabel("Forecast anchor")
    axis.set_ylabel("Tagged baboon")
    axis.set_title(
        "Robust singleton isolation values by animal and time\n"
        "Each cell is mean avoided attack rate across selected disease scenarios",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    figure.colorbar(image, ax=axis, shrink=0.75, label="Avoided attack rate")
    _save_figure(figure, path)


def _plot_baseline_ensemble(
    prepared: list[dict[str, Any]], parameter: pd.Series, seed: int, path: Path
) -> None:
    window = prepared[0]
    anchor = window["anchor"]
    sir = SIRParameters(
        beta=float(parameter["beta"]),
        recovery_rate=float(parameter["recovery_rate_per_day"]) / 86400,
    )
    results = []
    engine = PairedTemporalSIREngine()
    for index, initial in enumerate(sorted(window["eligible"])):
        results.append(
            engine.simulate(
                window["future"],
                sir,
                initial_infected=[initial],
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=seed + index,
            )
        )
    grid = pd.date_range(anchor.anchor_time, anchor.horizon_end, periods=180)
    matrix = np.vstack([_cumulative(result, grid) for result in results])
    figure, axis = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    for values in matrix:
        axis.step(grid, values, where="post", color="#BDBDBD", alpha=0.35)
    axis.step(grid, np.median(matrix, axis=0), where="post", color="#4C78A8", linewidth=2.5)
    axis.set_xlabel("Future replay time")
    axis.set_ylabel("Cumulative ever-infected animals")
    axis.set_title(
        "Baboon baseline outbreaks across all eligible index animals\n"
        f"{len(results)} worlds at {anchor.anchor_id}; "
        f"p20={1.0 - np.exp(-20.0 * float(parameter['beta'])):.4f}, "
        f"mean infectious period={float(parameter['mean_infectious_period_days']):g}d; "
        "blue is the median",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    figure.autofmt_xdate()
    _save_figure(figure, path)


def _audit_results(
    data_audit: dict[str, Any],
    calibration: pd.DataFrame,
    selections: pd.DataFrame,
    worlds: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    gates: dict[str, Any],
) -> dict[str, Any]:
    selected_count = int(selections["selected"].sum())
    informative_selected = int(
        (selections["selected"] & selections["informative"]).sum()
    )
    random_spearman = _median(summaries["random_repeat_stability"], "spearman")
    parameter_spearman = _median(summaries["parameter_stability"], "spearman")
    temporal_spearman = _median(summaries["temporal_stability"], "spearman")
    separation_icc = _median(summaries["candidate_separation"], "candidate_separation_icc")
    aggregate_metrics = summaries["aggregate_label_precision_metrics"]
    checks = {
        "data_quality": data_audit["status"] == "passed",
        "minimum_selected_scenarios": selected_count >= int(gates["minimum_selected_scenarios"]),
        "minimum_informative_selected_scenarios": informative_selected
        >= int(gates["minimum_informative_selected_scenarios"]),
        "aggregate_label_reliability": aggregate_metrics[
            "aggregate_label_spearman_brown_reliability"
        ]
        is not None
        and aggregate_metrics["aggregate_label_spearman_brown_reliability"]
        >= float(gates["aggregate_label_reliability"]),
        "parameter_median_spearman": parameter_spearman is not None
        and parameter_spearman >= float(gates["parameter_median_spearman"]),
        "aggregate_label_candidate_separation_icc": aggregate_metrics[
            "aggregate_label_candidate_separation_icc"
        ]
        is not None
        and aggregate_metrics["aggregate_label_candidate_separation_icc"]
        >= float(gates["aggregate_label_candidate_separation_icc"]),
    }
    key = [
        "anchor_id", "parameter_id", "block_id", "candidate_id",
        "introduction_stratum", "introduction_replicate", "initial_infected",
    ]
    expected = worlds["baseline_final_size"] - worlds["intervention_final_size"]
    integrity = {
        "duplicate_world_keys": int(worlds.duplicated(key).sum()),
        "missing_world_values": int(worlds.isna().sum().sum()),
        "arithmetic_identity_max_error": float(
            (worlds["avoided_infections"] - expected).abs().max()
        ),
    }
    integrity_passed = (
        integrity["duplicate_world_keys"] == 0
        and integrity["missing_world_values"] == 0
        and integrity["arithmetic_identity_max_error"] == 0
    )
    return {
        "status": "passed" if all(checks.values()) and integrity_passed else "needs_revision",
        "artifact_integrity_status": "passed" if integrity_passed else "needs_revision",
        "gate_checks": checks,
        "selected_parameter_scenarios": selected_count,
        "informative_selected_parameter_scenarios": informative_selected,
        "calibration_simulations": int(calibration["simulations"].sum()),
        "paired_worlds": int(len(worlds)),
        "negative_paired_outcomes": int(worlds["avoided_infections"].lt(0).sum()),
        "negative_paired_outcome_fraction": float(
            worlds["avoided_infections"].lt(0).mean()
        ),
        "median_metrics": {
            "random_repeat_spearman": random_spearman,
            "parameter_spearman": parameter_spearman,
            "temporal_spearman": temporal_spearman,
            "candidate_separation_icc": separation_icc,
            "random_repeat_top_k_overlap": _median(
                summaries["random_repeat_stability"], "top_k_overlap_fraction"
            ),
            "parameter_top_k_overlap": _median(
                summaries["parameter_stability"], "top_k_overlap_fraction"
            ),
            "temporal_top_k_overlap": _median(
                summaries["temporal_stability"], "top_k_overlap_fraction"
            ),
            **aggregate_metrics,
        },
        **integrity,
        "interpretation": (
            "Temporal stability is reported as a scientific result, not a pass/fail gate; "
            "a low value would favor time-conditioned prediction over a permanent watchlist."
        ),
    }


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

    dataset = CanonicalDataset.read(root / config["data"]["canonical_path"])
    stream = compile_primary_exposure(dataset)
    if stream.metadata.get("mapper") != "DetectionIntervalMapper":
        raise ValueError("Baboon primary analysis requires the RFID detection mapper")
    data_audit, daily, pairs = _data_quality_audit(
        dataset, stream, root / config["data"]["raw_proximity_path"]
    )
    if data_audit["status"] != "passed":
        raise ValueError("Baboon data-quality gate failed; full simulation is not allowed")

    prepared = _prepare_windows(
        stream, config["windows"], int(selected_profile["max_anchors"])
    )
    full_grid = _parameter_grid(config["parameter_grid"])
    grid = full_grid.copy()
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
        progress_label="Baboon parameter calibration",
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
    worlds, block_estimates = _run_stability(
        prepared,
        selected_parameters,
        config["intervention"],
        random_blocks=int(selected_profile["random_blocks"]),
        non_index_cases=int(selected_profile["non_index_cases_per_candidate_block"]),
        self_replicates=int(selected_profile["self_index_replicates_per_block"]),
        candidate_limit=selected_profile["candidate_limit"],
        seed=int(selected_profile["seed"]),
        progress_label="Baboon intervention-label simulations",
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

    frames = {
        "daily_primary_coverage": daily,
        "dyad_primary_coverage": pairs,
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
    for name, frame in frames.items():
        if name in {"calibration_worlds", "paired_world_outcomes"}:
            frame.to_csv(results_dir / f"{name}.csv", index=False)
        else:
            frame.to_csv(results_dir / f"{name}.csv", index=False)
            frame.to_csv(report_dir / f"{name}.csv", index=False)
    for directory in (results_dir, report_dir):
        (directory / "data_quality_audit.json").write_text(
            json.dumps(data_audit, indent=2), encoding="utf-8"
        )
        (directory / "audit_summary.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )

    _plot_data_quality(daily, pairs, data_audit, report_dir / "data_quality_overview.png")
    _plot_windows(prepared, report_dir / "timeline.png")
    _plot_calibration(calibration_summary, report_dir / "parameter_calibration.png")
    _plot_stability(
        summaries["random_repeat_stability"],
        "Baboon scenario-level random-repeat stability",
        report_dir / "random_repeat_stability.png",
    )
    _plot_stability(
        summaries["aggregate_label_random_stability"],
        "Baboon final-label random-repeat stability",
        report_dir / "aggregate_label_random_stability.png",
    )
    _plot_stability(
        summaries["parameter_stability"],
        "Baboon disease-scenario ranking stability",
        report_dir / "parameter_stability.png",
    )
    _plot_stability(
        summaries["temporal_stability"],
        "Baboon across-window ranking stability",
        report_dir / "temporal_stability.png",
    )
    _plot_label_heatmap(summaries["robust_anchor_labels"], report_dir / "label_heatmap.png")
    selected_attack_rates = selections.loc[
        selections["selected"], ["parameter_id", "mean_attack_rate"]
    ]
    diagnostic_id = str(
        selected_attack_rates.iloc[
            (selected_attack_rates["mean_attack_rate"] - 0.5).abs().to_numpy().argmin()
        ]["parameter_id"]
    )
    reference_row = full_grid.loc[full_grid["parameter_id"].eq(diagnostic_id)].iloc[0]
    _plot_baseline_ensemble(
        prepared,
        reference_row,
        int(selected_profile["seed"]),
        report_dir / "baseline_ensemble.png",
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
        "data_quality_status": data_audit["status"],
        "artifact_integrity_status": audit["artifact_integrity_status"],
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
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
    report = f"""# Guinea baboon label-generation validation

This run applies the shared prospective singleton-isolation label contract to
the primary 20-second RFID proximity stream. Direct behavioral observations
remain a separate modality and are not pooled into the transmission process.

## Data and windows

- Raw, canonical, and mapped proximity rows: {data_audit['raw_proximity_rows']:,} / {data_audit['canonical_proximity_rows']:,} / {data_audit['primary_exposure_rows']:,}.
- Tagged animals in the endpoint union: {data_audit['primary_endpoint_union_individuals']}.
- Complete daily coverage span: {data_audit['calendar_days_with_events']} days.
- Analysis windows: {len(prepared)}; seven-day history and three-day non-overlapping future replay windows.

## Result

- Validation status: `{audit['status']}`.
- Calibration simulations: {audit['calibration_simulations']:,}.
- Paired intervention worlds: {audit['paired_worlds']:,}.
- Selected informative disease scenarios: {audit['informative_selected_parameter_scenarios']} of {audit['selected_parameter_scenarios']} selected.
- Median random-repeat rank correlation: {medians['random_repeat_spearman']}.
- Final-label single-block rank correlation: {medians['aggregate_label_single_block_spearman']}.
- Three-block final-label reliability estimate: {medians['aggregate_label_spearman_brown_reliability']}.
- Median disease-scenario rank correlation: {medians['parameter_spearman']}.
- Median across-window rank correlation: {medians['temporal_spearman']}.
- Median candidate-separation ICC: {medians['candidate_separation_icc']}.
- Scenario-averaged single-block candidate-separation ICC: {medians['aggregate_label_single_block_candidate_separation_icc']}.
- Delivered three-block-mean candidate-separation ICC: {medians['aggregate_label_mean_candidate_separation_icc']}.
- Negative paired outcomes retained without clipping: {audit['negative_paired_outcomes']}.

`robust_anchor_labels.csv` is the model-facing table. Each row is one tagged
animal at one forecast anchor; anchors are not averaged. The label is the mean
paired avoided attack rate across selected informative epidemic scenarios.

Across-window stability is descriptive rather than a gate. Low temporal
stability would be evidence for a time-conditioned intervention model, not a
reason to discard otherwise reproducible per-anchor labels.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    return results_dir, report_dir


def refresh_audit(config_path: Path, profile: str) -> tuple[Path, Path]:
    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = config["experiment"]["id"]
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile
    required = [
        "calibration_summary.csv",
        "parameter_selection.csv",
        "paired_world_outcomes.csv",
        "block_estimates.csv",
        "data_quality_audit.json",
    ]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot refresh audit; missing: {', '.join(missing)}")
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
        min(
            int(config["stability"]["consensus_minimum_contexts"]),
            int(block_estimates["anchor_id"].nunique())
            * int(block_estimates["parameter_id"].nunique()),
        ),
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
    for name, frame in summaries.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        frame.to_csv(results_dir / f"{name}.csv", index=False)
        frame.to_csv(report_dir / f"{name}.csv", index=False)
    audit_text = json.dumps(audit, indent=2)
    (results_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")
    (report_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")
    _plot_stability(
        aggregate_stability,
        "Baboon final-label random-repeat stability",
        report_dir / "aggregate_label_random_stability.png",
    )
    _plot_label_heatmap(
        summaries["robust_anchor_labels"], report_dir / "label_heatmap.png"
    )
    medians = audit["median_metrics"]
    report = f"""# Guinea baboon label-generation validation

The primary analysis uses only the 20-second RFID proximity stream. The direct
behavioral-observation modality remains separate and is not pooled.

## Audited result

- Data-quality status: `{data_audit['status']}`; {data_audit['raw_proximity_rows']:,} raw rows reconcile one-to-one with canonical events and mapped exposures.
- Tagged endpoint-union population: {data_audit['primary_endpoint_union_individuals']} animals.
- Forecast anchors: {int(summaries['robust_anchor_labels']['anchor_id'].nunique())}; final label rows: {len(summaries['robust_anchor_labels'])}.
- Calibration simulations: {audit['calibration_simulations']:,}; paired worlds: {audit['paired_worlds']:,}.
- Selected informative disease scenarios: {audit['informative_selected_parameter_scenarios']} of {audit['selected_parameter_scenarios']} selected.
- Artifact integrity: `{audit['artifact_integrity_status']}`.
- Final validation status: `{audit['status']}`.

## Precision and stability

- Scenario-level single-block rank correlation: {medians['random_repeat_spearman']:.3f}.
- Final-label single-block rank correlation after averaging disease scenarios: {medians['aggregate_label_single_block_spearman']:.3f}.
- Reliability of the delivered three-block mean label (Spearman-Brown): {medians['aggregate_label_spearman_brown_reliability']:.3f}.
- Scenario-averaged single-block candidate-separation ICC: {medians['aggregate_label_single_block_candidate_separation_icc']:.3f}.
- Delivered three-block-mean candidate-separation ICC: {medians['aggregate_label_mean_candidate_separation_icc']:.3f}.
- Disease-scenario rank correlation: {medians['parameter_spearman']:.3f}.
- Across-window rank correlation: {medians['temporal_spearman']:.3f}.

The scenario-level diagnostic is deliberately retained: one disease scenario
and one random block are noisy in this 13-animal system. The model-facing label,
however, is the predeclared mean across selected scenarios and three independent
blocks; its reliability and candidate separation are the applicable gates.

Across-window stability is descriptive, not a pass/fail condition. A changing
ranking supports time-conditioned prediction rather than invalidating valid
per-anchor labels. Negative paired temporal outcomes remain unclipped.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    analysis_resolved = {**config, "selected_profile": profile}
    analysis_text = yaml.safe_dump(analysis_resolved, sort_keys=False)
    (results_dir / "analysis_resolved_config.yaml").write_text(
        analysis_text, encoding="utf-8"
    )
    (report_dir / "analysis_resolved_config.yaml").write_text(
        analysis_text, encoding="utf-8"
    )
    dataset = CanonicalDataset.read(root / config["data"]["canonical_path"])
    stream = compile_primary_exposure(dataset)
    prepared = _prepare_windows(
        stream, config["windows"], int(config["profiles"][profile]["max_anchors"])
    )
    full_grid = _parameter_grid(config["parameter_grid"])
    selected_attack_rates = selections.loc[
        selections["selected"], ["parameter_id", "mean_attack_rate"]
    ]
    diagnostic_id = str(
        selected_attack_rates.iloc[
            (selected_attack_rates["mean_attack_rate"] - 0.5).abs().to_numpy().argmin()
        ]["parameter_id"]
    )
    diagnostic_parameter = full_grid.loc[
        full_grid["parameter_id"].eq(diagnostic_id)
    ].iloc[0]
    _plot_baseline_ensemble(
        prepared,
        diagnostic_parameter,
        int(config["profiles"][profile]["seed"]),
        report_dir / "baseline_ensemble.png",
    )
    manifest_path = results_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_status"] = audit["status"]
        manifest["artifact_integrity_status"] = audit["artifact_integrity_status"]
        manifest["audit_refreshed_at_utc"] = datetime.now(UTC).isoformat(
            timespec="seconds"
        )
        manifest["analysis_config_sha256"] = _sha256(config_path)
        manifest_text = json.dumps(manifest, indent=2)
        manifest_path.write_text(manifest_text, encoding="utf-8")
        (report_dir / "run_manifest.json").write_text(
            manifest_text, encoding="utf-8"
        )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Guinea baboon data, calibration, and label validation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--refresh-audit",
        action="store_true",
        help="Recompute label summaries and audit from saved full-run outcomes.",
    )
    args = parser.parse_args()
    if args.refresh_audit:
        results_dir, report_dir = refresh_audit(args.config, args.profile)
    else:
        results_dir, report_dir = run(args.config, args.profile)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
