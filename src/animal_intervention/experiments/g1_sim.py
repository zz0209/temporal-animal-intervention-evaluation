from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import (
    estimate_singleton_values,
    rolling_anchors,
    slice_stream,
)
from animal_intervention.simulation import InterventionAction, PairedTemporalSIREngine, SIRParameters
from animal_intervention.transmission.mappers import compile_primary_exposure


def _repository_root(config_path: Path) -> Path:
    for parent in [config_path.resolve().parent, *config_path.resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("could not locate repository root from config path")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_timeline(anchors: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 3.8))
    for index, row in anchors.reset_index(drop=True).iterrows():
        y = len(anchors) - index
        history_days = (row.anchor_time - row.history_start).total_seconds() / 86400
        horizon_days = (row.horizon_end - row.anchor_time).total_seconds() / 86400
        axis.barh(y, history_days, left=row.history_start, color="#4C78A8", height=0.5)
        axis.barh(y, horizon_days, left=row.anchor_time, color="#F28E2B", height=0.5)
        axis.axvline(row.anchor_time, color="#303030", linewidth=0.8, alpha=0.45)
    axis.set_yticks(range(1, len(anchors) + 1), anchors["anchor_id"].iloc[::-1])
    axis.set_xlabel("Replay time")
    figure.suptitle("Rolling prediction windows", x=0.125, y=0.98, ha="left", weight="bold")
    figure.text(
        0.125, 0.91,
        "Blue: observed history used for eligibility | Orange: future used only for offline labels",
        fontsize=9, color="#555555",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    figure.autofmt_xdate()
    figure.subplots_adjust(top=0.84)
    _save_figure(figure, path)


def _cumulative_infections(result: Any, grid: pd.DatetimeIndex) -> np.ndarray:
    infections = pd.to_datetime(
        result.event_log.loc[
            result.event_log["event"].isin(["initial_infection", "infection"]), "time"
        ]
    ).sort_values()
    values = np.searchsorted(infections.to_numpy(), grid.to_numpy(), side="right")
    return values


def _plot_paired_trajectory(
    baseline: Any,
    intervention: Any,
    candidate: str,
    world_index: int,
    path: Path,
) -> None:
    grid = pd.date_range(baseline.start_time, baseline.end_time, periods=240)
    figure, axis = plt.subplots(figsize=(9.5, 4.7))
    axis.step(
        grid, _cumulative_infections(baseline, grid), where="post",
        color="#555555", linewidth=2.2, label=f"No intervention (final={baseline.final_size})",
    )
    axis.step(
        grid, _cumulative_infections(intervention, grid), where="post",
        color="#D55E00", linewidth=2.2,
        label=f"Isolate {candidate} (final={intervention.final_size})",
    )
    axis.set_ylabel("Cumulative infected animals")
    axis.set_xlabel("Future replay time")
    figure.suptitle(
        "Paired epidemic trajectories in one random world",
        x=0.125, y=0.98, ha="left", weight="bold",
    )
    figure.text(
        0.125, 0.91,
        f"World {world_index}; identical future contacts, initial infection, and keyed random primitives",
        fontsize=9, color="#555555",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.autofmt_xdate()
    figure.subplots_adjust(top=0.84)
    _save_figure(figure, path)


def _plot_ranking(estimates: pd.DataFrame, anchor_id: str, path: Path) -> None:
    selected = estimates.loc[estimates["anchor_id"].eq(anchor_id)].nsmallest(15, "rank")
    selected = selected.sort_values("mean_avoided_attack_rate")
    y = np.arange(len(selected))
    means = selected["mean_avoided_attack_rate"].to_numpy()
    lower = selected["ci95_lower"].to_numpy()
    upper = selected["ci95_upper"].to_numpy()
    figure, axis = plt.subplots(figsize=(9, 6.5))
    axis.errorbar(
        means, y, xerr=np.vstack([means - lower, upper - means]), fmt="o",
        color="#4C78A8", ecolor="#9ECAE1", capsize=3, markersize=5,
    )
    axis.axvline(0, color="#333333", linewidth=1, linestyle="--")
    axis.set_yticks(y, selected["candidate_id"])
    axis.set_xlabel("Mean avoided attack rate (paired baseline minus isolation)")
    figure.suptitle(
        "Highest estimated singleton intervention values",
        x=0.125, y=0.98, ha="left", weight="bold",
    )
    worlds = int(selected["worlds"].iloc[0])
    figure.text(
        0.125, 0.92,
        f"{anchor_id}; {worlds} paired Monte Carlo worlds; bars are normal 95% MC intervals",
        fontsize=9, color="#555555",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    figure.subplots_adjust(top=0.86)
    _save_figure(figure, path)


def _write_report(
    config: dict[str, Any],
    profile: str,
    estimates: pd.DataFrame,
    worlds: pd.DataFrame,
    anchors: pd.DataFrame,
    results_dir: Path,
    report_dir: Path,
    elapsed: float,
) -> None:
    top = estimates.sort_values(["anchor_id", "rank"]).groupby("anchor_id").head(1)
    rows = "\n".join(
        f"| {row.anchor_id} | {row.candidate_id} | {row.mean_avoided_attack_rate:.4f} "
        f"| [{row.ci95_lower:.4f}, {row.ci95_upper:.4f}] |"
        for row in top.itertuples(index=False)
    )
    zero_future = estimates.loc[estimates["future_event_support"].eq(0)]
    max_zero_future = float(zero_future["mean_avoided_attack_rate"].abs().max())
    negative_worlds = int(worlds["avoided_infections"].lt(0).sum())
    baseline_min = int(worlds["baseline_final_size"].min())
    baseline_max = int(worlds["baseline_final_size"].max())
    relative_results = (
        f"{config['outputs']['results_root']}/{config['experiment']['id']}/{profile}"
    )
    text = f"""# G1-SIM Oxford engineering pilot

This run validates the paired simulation and label-construction machinery. It is
not a calibrated epidemiological result and must not be interpreted as field
evidence about Oxford birds.

## Run summary

- Experiment: `{config['experiment']['id']}`
- Profile: `{profile}`
- Anchors: {len(anchors)}
- Candidate-anchor rows: {len(estimates)}
- Monte Carlo worlds per candidate: {int(estimates['worlds'].iloc[0])}
- Runtime: {elapsed:.1f} seconds
- Primary engineering label: paired mean avoided attack rate
- Eligibility: history-only event support; future support is saved only as an offline diagnostic
- Future network: observed replay with no behavioral rewiring

## Top engineering-pilot estimates

| Anchor | Candidate | Mean avoided attack rate | 95% MC interval |
|---|---|---:|---:|
{rows}

## Visual audit map

- `timeline.png`: verifies lookback, anchor, and future-label boundaries.
- `paired_trajectory.png`: verifies a baseline/action comparison inside one shared random world.
- `ranking.png`: checks magnitude, uncertainty, ties, and zero-crossing for the first anchor.

## Machine-readable artifacts

- `anchor_summary.csv`, `intervention_value_ranking.csv`, and `audit_summary.json` are copied into this report directory.
- Full candidate-world results, resolved config, and run manifest are in `{relative_results}`.

## Automated audit

- Baseline final-size range: {baseline_min}–{baseline_max} animals.
- Candidates with zero future event support: {len(zero_future)}; maximum absolute mean value among them: {max_zero_future:.6f}.
- Negative candidate-world differences: {negative_worlds}/{len(worlds)}. In a temporal SIR replay, removing a contact can rarely alter infection timing and expose a later path; these rows remain visible rather than being clipped.

## Scope limitations

The transmission parameters are engineering values, the association-to-hazard
mapping is not biologically calibrated, confidence intervals describe Monte
Carlo error only, and complete isolation assumes no contact rewiring. Publication
claims require the later parameter, missingness, rewiring, and external-system arms.
"""
    (report_dir / "README.md").write_text(text, encoding="utf-8")


def run(config_path: Path, profile: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    root = _repository_root(config_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if profile not in raw_config["profiles"]:
        raise ValueError(f"unknown profile: {profile}")
    resolved = {**raw_config, "selected_profile": profile, "run": raw_config["profiles"][profile]}
    experiment_id = raw_config["experiment"]["id"]
    results_dir = root / raw_config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / raw_config["outputs"]["report_root"] / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = root / raw_config["data"]["canonical_path"]
    dataset = CanonicalDataset.read(canonical_path)
    stream = compile_primary_exposure(dataset)
    windows = raw_config["windows"]
    anchors = rolling_anchors(
        stream,
        lookback=pd.Timedelta(windows["lookback"]),
        horizon=pd.Timedelta(windows["horizon"]),
        step=pd.Timedelta(windows["step"]),
        max_anchors=int(windows["max_anchors"]),
    )
    epidemic = raw_config["epidemic"]
    if int(epidemic["initial_infected_count"]) != 1:
        raise ValueError("G1-SIM currently supports exactly one initial infected node")
    parameters = SIRParameters(
        beta=float(epidemic["beta_per_integrated_association"]),
        recovery_rate=float(epidemic["recovery_rate_per_day"]) / 86400.0,
    )
    selected_profile = raw_config["profiles"][profile]
    estimates, worlds, anchor_summary, baseline_results = estimate_singleton_values(
        stream,
        anchors,
        parameters,
        action_config=raw_config["intervention"],
        worlds=int(selected_profile["worlds"]),
        seed=int(selected_profile["seed"]),
        min_history_events=int(windows["min_history_events_per_node"]),
        candidate_limit=(
            None
            if selected_profile["candidate_limit"] is None
            else int(selected_profile["candidate_limit"])
        ),
    )
    estimates.to_csv(results_dir / "intervention_value_ranking.csv", index=False)
    worlds.to_csv(results_dir / "paired_world_outcomes.csv", index=False)
    anchor_summary.to_csv(results_dir / "anchor_summary.csv", index=False)
    estimates.to_csv(report_dir / "intervention_value_ranking.csv", index=False)
    anchor_summary.to_csv(report_dir / "anchor_summary.csv", index=False)
    baseline_groups = worlds.groupby(["anchor_id", "world_index"], observed=True)
    candidate_groups = worlds.groupby(["anchor_id", "candidate_id"], observed=True)
    recomputed = candidate_groups["avoided_attack_rate"].mean().rename("recomputed")
    joined = estimates.set_index(["anchor_id", "candidate_id"]).join(recomputed)
    zero_future = estimates.loc[estimates["future_event_support"].eq(0)]
    audit = {
        "status": "passed",
        "anchor_count": len(anchor_summary),
        "candidate_anchor_rows": len(estimates),
        "candidate_world_rows": len(worlds),
        "worlds_per_candidate_min": int(candidate_groups.size().min()),
        "worlds_per_candidate_max": int(candidate_groups.size().max()),
        "max_baseline_final_size_variants_within_paired_world": int(
            baseline_groups["baseline_final_size"].nunique().max()
        ),
        "max_initial_seed_variants_within_paired_world": int(
            baseline_groups["initial_infected"].nunique().max()
        ),
        "max_label_recomputation_error": float(
            (joined["mean_avoided_attack_rate"] - joined["recomputed"]).abs().max()
        ),
        "zero_future_support_candidates": len(zero_future),
        "zero_future_support_all_zero_value": bool(
            zero_future["mean_avoided_attack_rate"].eq(0).all()
        ),
        "negative_candidate_world_differences": int(
            worlds["avoided_infections"].lt(0).sum()
        ),
    }
    audit_text = json.dumps(audit, indent=2)
    (results_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")
    (report_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")

    selected = worlds.sort_values(
        ["avoided_infections", "anchor_id", "candidate_id"], ascending=[False, True, True]
    ).iloc[0]
    anchor = next(item for item in anchors if item.anchor_id == selected.anchor_id)
    future = slice_stream(stream, anchor.anchor_time, anchor.horizon_end)
    action_config = raw_config["intervention"]
    action_start = anchor.anchor_time + pd.Timedelta(action_config["delay"])
    action = InterventionAction(
        name=action_config["name"], action_type=action_config["action_type"],
        target_nodes=(str(selected.candidate_id),), start_time=action_start,
        end_time=min(anchor.horizon_end, action_start + pd.Timedelta(action_config["duration"])),
        contact_multiplier=float(action_config["contact_multiplier"]),
        susceptibility_multiplier=float(action_config["susceptibility_multiplier"]),
        infectivity_multiplier=float(action_config["infectivity_multiplier"]),
        recovery_rate_multiplier=float(action_config["recovery_rate_multiplier"]),
    )
    intervention = PairedTemporalSIREngine().simulate(
        future, parameters, initial_infected=[str(selected.initial_infected)],
        start_time=anchor.anchor_time, end_time=anchor.horizon_end,
        world_seed=int(selected.world_seed), action=action,
    )
    baseline = baseline_results[(str(selected.anchor_id), int(selected.world_index))]

    _plot_timeline(anchor_summary, report_dir / "timeline.png")
    _plot_paired_trajectory(
        baseline, intervention, str(selected.candidate_id), int(selected.world_index),
        report_dir / "paired_trajectory.png",
    )
    _plot_ranking(estimates, anchors[0].anchor_id, report_dir / "ranking.png")

    elapsed = time.perf_counter() - started
    resolved_path = results_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "status": "completed",
        "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "config_path": config_path.resolve().relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path),
        "resolved_config_sha256": _sha256(resolved_path),
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
    manifest_path = results_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "resolved_config.yaml").write_text(
        resolved_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (report_dir / "run_manifest.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_report(
        raw_config, profile, estimates, worlds, anchor_summary,
        results_dir, report_dir, elapsed,
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G1-SIM paired Oxford pilot.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    results_dir, report_dir = run(args.config, args.profile)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
