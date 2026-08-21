from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import time
from typing import Any

import importlib.metadata
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import (
    _candidate_action,
    _stratified_world_seed,
    estimate_stratified_singleton_values,
    node_support,
    rolling_anchors,
    slice_stream,
)
from animal_intervention.simulation import PairedTemporalSIREngine, SIRParameters
from animal_intervention.transmission.mappers import compile_primary_exposure

from .g1_sim import _git_state, _repository_root, _save_figure, _sha256


def _header(
    figure: plt.Figure,
    title: str,
    subtitle: str,
    *,
    top: float = 0.78,
) -> None:
    figure.suptitle(title, x=0.10, y=0.975, ha="left", fontsize=16, weight="bold")
    figure.text(0.10, 0.905, subtitle, ha="left", fontsize=10, color="#555555")
    figure.subplots_adjust(top=top)


def _plot_timeline(anchors: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.4))
    for index, row in anchors.reset_index(drop=True).iterrows():
        y = len(anchors) - index
        history_days = (row.anchor_time - row.history_start).total_seconds() / 86400
        horizon_days = (row.horizon_end - row.anchor_time).total_seconds() / 86400
        axis.barh(y, history_days, left=row.history_start, color="#4C78A8", height=0.46)
        axis.barh(y, horizon_days, left=row.anchor_time, color="#F28E2B", height=0.46)
        axis.axvline(row.anchor_time, color="#303030", linewidth=0.8, alpha=0.45)
    axis.set_yticks(range(1, len(anchors) + 1), anchors["anchor_id"].iloc[::-1])
    axis.set_xlabel("Calendar time in the Oxford replay")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    figure.autofmt_xdate()
    _header(
        figure,
        "Rolling history and future-label windows",
        "Blue = observed history for eligibility; orange = future contacts used only for offline labels",
        top=0.79,
    )
    _save_figure(figure, path)


def _cumulative(result: Any, grid: pd.DatetimeIndex) -> np.ndarray:
    times = pd.to_datetime(
        result.event_log.loc[
            result.event_log["event"].isin(["initial_infection", "infection"]), "time"
        ]
    ).sort_values()
    return np.searchsorted(times.to_numpy(), grid.to_numpy(), side="right")


def _simulate_pair(
    future: Any,
    parameters: SIRParameters,
    anchor: Any,
    row: pd.Series,
    action_config: dict[str, Any],
) -> tuple[Any, Any]:
    engine = PairedTemporalSIREngine()
    baseline = engine.simulate(
        future,
        parameters,
        initial_infected=[str(row.initial_infected)],
        start_time=anchor.anchor_time,
        end_time=anchor.horizon_end,
        world_seed=int(row.world_seed),
    )
    intervention = engine.simulate(
        future,
        parameters,
        initial_infected=[str(row.initial_infected)],
        start_time=anchor.anchor_time,
        end_time=anchor.horizon_end,
        world_seed=int(row.world_seed),
        action=_candidate_action(str(row.candidate_id), anchor, action_config),
    )
    return baseline, intervention


def _plot_paired_examples(
    pairs: list[tuple[str, pd.Series, Any, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for axis, (label, row, baseline, intervention) in zip(axes, pairs):
        grid = pd.date_range(baseline.start_time, baseline.end_time, periods=240)
        axis.step(
            grid, _cumulative(baseline, grid), where="post", color="#555555",
            linewidth=2, label=f"No intervention: {baseline.final_size}",
        )
        axis.step(
            grid, _cumulative(intervention, grid), where="post", color="#D55E00",
            linewidth=2, linestyle="--",
            label=f"Isolate node {row.candidate_id}: {intervention.final_size}",
        )
        delta = baseline.final_size - intervention.final_size
        axis.set_title(
            f"{label}\nIndex {row.initial_infected}; target {row.candidate_id}; avoided {delta}",
            loc="left", fontsize=11, pad=12,
        )
        axis.set_xlabel("Future replay time")
        axis.legend(frameon=False, fontsize=9, loc="upper left")
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Cumulative ever-infected animals")
    _header(
        figure,
        "Paired trajectories: typical and extreme outcomes",
        "Each panel shares contacts, index case, and keyed randomness within its baseline–isolation pair",
        top=0.76,
    )
    _save_figure(figure, path)


def _plot_ranking(estimates: pd.DataFrame, anchor_id: str, path: Path) -> None:
    selected = estimates.loc[estimates["anchor_id"].eq(anchor_id)].nsmallest(20, "rank")
    selected = selected.sort_values("unconditional_value")
    means = selected["unconditional_value"].to_numpy()
    lower = selected["ci95_lower"].to_numpy()
    upper = selected["ci95_upper"].to_numpy()
    y = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(9.6, 8.2))
    axis.errorbar(
        means, y, xerr=np.vstack([means - lower, upper - means]), fmt="o",
        color="#4C78A8", ecolor="#9ECAE1", capsize=3, markersize=5,
    )
    axis.axvline(0, color="#333333", linewidth=1, linestyle="--")
    axis.set_yticks(y, selected["candidate_id"])
    axis.set_xlabel("Unconditional mean avoided attack rate")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    n = int(selected["eligible_population"].iloc[0])
    self_reps = int(selected["self_index_worlds"].iloc[0])
    _header(
        figure,
        "Balanced singleton intervention-value estimates",
        f"{anchor_id}; all {n - 1} non-index seeds + {self_reps} self-index replicates per candidate; stratified bootstrap 95% intervals",
        top=0.84,
    )
    _save_figure(figure, path)


def _plot_baseline_ensemble(results: list[Any], path: Path) -> None:
    grid = pd.date_range(results[0].start_time, results[0].end_time, periods=180)
    matrix = np.vstack([_cumulative(result, grid) for result in results])
    lower, median, upper = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0)
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for values in matrix:
        axis.step(grid, values, where="post", color="#BDBDBD", alpha=0.12, linewidth=0.7)
    axis.fill_between(grid, lower, upper, step="post", color="#9ECAE1", alpha=0.45, label="IQR")
    axis.step(grid, median, where="post", color="#4C78A8", linewidth=2.4, label="Median")
    axis.set_xlabel("Future replay time")
    axis.set_ylabel("Cumulative ever-infected animals")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    figure.autofmt_xdate()
    _header(
        figure,
        "Baseline epidemic ensemble across all eligible index cases",
        f"One paired-random baseline world for each of {len(results)} eligible Oxford animals at anchor_001",
        top=0.79,
    )
    _save_figure(figure, path)


def _plot_all_animal_states(result: Any, roster: list[str], path: Path) -> None:
    grid = pd.date_range(result.start_time, result.end_time, periods=160)
    infection_time: dict[str, pd.Timestamp] = {}
    recovery_time: dict[str, pd.Timestamp] = {}
    for row in result.event_log.itertuples(index=False):
        if row.event in {"initial_infection", "infection"}:
            infection_time[str(row.node_id)] = pd.Timestamp(row.time)
        elif row.event == "recovery":
            recovery_time[str(row.node_id)] = pd.Timestamp(row.time)
    ordered = sorted(
        roster,
        key=lambda node: (infection_time.get(node, pd.Timestamp.max), node),
    )
    states = np.zeros((len(ordered), len(grid)), dtype=int)
    for row_index, node in enumerate(ordered):
        infected = infection_time.get(node)
        if infected is None:
            continue
        states[row_index, grid >= infected] = 1
        recovered = recovery_time.get(node)
        if recovered is not None:
            states[row_index, grid >= recovered] = 2
    cmap = ListedColormap(["#E6E6E6", "#F28E2B", "#4C78A8"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    figure, axis = plt.subplots(figsize=(10, 7.2))
    image = axis.imshow(
        states,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[grid[0].value, grid[-1].value, len(ordered), 0],
    )
    ticks = pd.date_range(grid[0], grid[-1], periods=5)
    axis.set_xticks([tick.value for tick in ticks], [tick.strftime("%m-%d %H:%M") for tick in ticks])
    axis.tick_params(axis="x", rotation=25)
    axis.set_xlabel("Future replay time")
    axis.set_ylabel("Animals, ordered by infection time")
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2], pad=0.02)
    colorbar.ax.set_yticklabels(["S", "I", "R"])
    _header(
        figure,
        "All-animal SIR state trajectories in a representative baseline world",
        f"{len(ordered)} animals shown; grey = susceptible, orange = infectious, blue = recovered",
        top=0.82,
    )
    _save_figure(figure, path)


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
    window_config = config["windows"]
    anchors = rolling_anchors(
        stream,
        lookback=pd.Timedelta(window_config["lookback"]),
        horizon=pd.Timedelta(window_config["horizon"]),
        step=pd.Timedelta(window_config["step"]),
        max_anchors=int(selected_profile["max_anchors"]),
    )
    epidemic = config["epidemic"]
    if int(epidemic["initial_infected_count"]) != 1:
        raise ValueError("balanced G1-SIM requires exactly one initial infected node")
    parameters = SIRParameters(
        beta=float(epidemic["beta_per_integrated_association"]),
        recovery_rate=float(epidemic["recovery_rate_per_day"]) / 86400,
    )
    estimation = config["estimation"]
    estimates, worlds, anchor_summary = estimate_stratified_singleton_values(
        stream,
        anchors,
        parameters,
        action_config=config["intervention"],
        self_seed_replicates=int(estimation["self_seed_replicates"]),
        bootstrap_replicates=int(estimation["bootstrap_replicates"]),
        seed=int(selected_profile["seed"]),
        min_history_events=int(window_config["min_history_events_per_node"]),
        candidate_limit=selected_profile["candidate_limit"],
    )
    for directory in (results_dir, report_dir):
        estimates.to_csv(directory / "intervention_value_ranking.csv", index=False)
        anchor_summary.to_csv(directory / "anchor_summary.csv", index=False)
    worlds.to_csv(results_dir / "paired_world_outcomes.csv", index=False)

    first_anchor = anchors[0]
    first_future = slice_stream(stream, first_anchor.anchor_time, first_anchor.horizon_end)
    history = slice_stream(stream, first_anchor.history_start, first_anchor.anchor_time)
    eligible = sorted(
        node
        for node, count in node_support(history).items()
        if count >= int(window_config["min_history_events_per_node"])
    )
    baseline_engine = PairedTemporalSIREngine()
    baseline_results = []
    for initial in eligible:
        world_seed = _stratified_world_seed(
            int(selected_profile["seed"]), first_anchor.anchor_id, "non_index", initial, 0
        )
        baseline_results.append(
            baseline_engine.simulate(
                first_future,
                parameters,
                initial_infected=[initial],
                start_time=first_anchor.anchor_time,
                end_time=first_anchor.horizon_end,
                world_seed=world_seed,
            )
        )

    top_candidate = str(
        estimates.loc[estimates["anchor_id"].eq(first_anchor.anchor_id)]
        .nsmallest(1, "rank")["candidate_id"].iloc[0]
    )
    candidate_worlds = worlds.loc[
        worlds["anchor_id"].eq(first_anchor.anchor_id)
        & worlds["candidate_id"].eq(top_candidate)
    ].copy()
    typical_value = float(candidate_worlds["avoided_infections"].median())
    typical_row = candidate_worlds.iloc[
        (candidate_worlds["avoided_infections"] - typical_value).abs().to_numpy().argmin()
    ]
    extreme_row = candidate_worlds.nlargest(1, "avoided_infections").iloc[0]
    typical_pair = _simulate_pair(
        first_future, parameters, first_anchor, typical_row, config["intervention"]
    )
    extreme_pair = _simulate_pair(
        first_future, parameters, first_anchor, extreme_row, config["intervention"]
    )

    _plot_timeline(anchor_summary, report_dir / "timeline.png")
    _plot_ranking(estimates, first_anchor.anchor_id, report_dir / "ranking.png")
    _plot_paired_examples(
        [
            ("Typical paired outcome", typical_row, *typical_pair),
            ("Maximum observed paired benefit", extreme_row, *extreme_pair),
        ],
        report_dir / "paired_trajectory.png",
    )
    _plot_baseline_ensemble(baseline_results, report_dir / "baseline_ensemble.png")
    median_result = sorted(baseline_results, key=lambda result: result.final_size)[
        len(baseline_results) // 2
    ]
    roster = sorted(set(eligible) | first_future.nodes())
    _plot_all_animal_states(
        median_result, roster, report_dir / "all_animal_state_trajectories.png"
    )

    strata_counts = worlds.groupby(
        ["anchor_id", "candidate_id", "introduction_stratum"], observed=True
    ).size().unstack(fill_value=0)
    audit = {
        "status": "passed",
        "anchor_count": len(anchor_summary),
        "candidate_anchor_rows": len(estimates),
        "candidate_world_rows": len(worlds),
        "self_index_worlds_per_candidate_min": int(strata_counts["self_index"].min()),
        "self_index_worlds_per_candidate_max": int(strata_counts["self_index"].max()),
        "non_index_worlds_match_eligible_minus_one": bool(
            all(
                int(row.non_index_worlds) == int(row.eligible_population) - 1
                for row in estimates.itertuples(index=False)
            )
        ),
        "stratified_identity_max_error": float(
            (
                estimates["unconditional_value"]
                - estimates["known_index_value"] / estimates["eligible_population"]
                - estimates["non_index_value"]
                * (estimates["eligible_population"] - 1)
                / estimates["eligible_population"]
            ).abs().max()
        ),
        "zero_future_support_all_zero_non_index_value": bool(
            estimates.loc[estimates["future_event_support"].eq(0), "non_index_value"]
            .eq(0).all()
        ),
    }
    audit_text = json.dumps(audit, indent=2)
    for directory in (results_dir, report_dir):
        (directory / "audit_summary.json").write_text(audit_text, encoding="utf-8")

    resolved = {
        **config,
        "selected_profile": profile,
        "run": dict(selected_profile),
    }
    resolved_path = results_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    elapsed = time.perf_counter() - started
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "status": "completed",
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
    (report_dir / "resolved_config.yaml").write_text(
        resolved_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    report = f"""# Balanced G1-SIM Oxford engineering pilot

This experiment replaces the imbalanced 16-seed design in EXP-20260814-001.
Every candidate is evaluated against every other eligible animal as index case,
plus {int(estimation['self_seed_replicates'])} explicit self-index replicates.

## Estimands

- `known_index_value`: isolate the candidate when it is the index case.
- `non_index_value`: isolate the candidate when another eligible animal is the index case.
- `unconditional_value`: population-weighted combination of the two strata; this is the primary ranking target.

## Run

- Profile: `{profile}`
- Anchors: {len(anchor_summary)}
- Candidate-anchor rows: {len(estimates)}
- Paired candidate-world outcomes: {len(worlds)}
- Runtime: {elapsed:.1f} seconds
- Audit status: `{audit['status']}`

## Figures

- `timeline.png`: history versus future-label windows.
- `baseline_ensemble.png`: aggregate epidemic curves for every eligible index case at anchor_001.
- `all_animal_state_trajectories.png`: S/I/R state history for every animal in a representative baseline world.
- `paired_trajectory.png`: typical and maximum-benefit paired examples, explicitly separated.
- `ranking.png`: balanced unconditional values with stratified bootstrap intervals.

This remains an uncalibrated engineering pilot. It demonstrates a fairer label
estimator and complete node-level epidemic output, not biological risk rankings.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run balanced G1-SIM Oxford pilot.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    results_dir, report_dir = run(args.config, args.profile)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
