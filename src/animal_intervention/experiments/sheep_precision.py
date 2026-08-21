from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.evaluation import aggregate_label_precision
from animal_intervention.transmission.mappers import CoalescedDurationContactMapper

from .g1_sim import _git_state, _repository_root, _save_figure, _sha256
from .sheep_validation import _load_phase_by_date, _prepare_network_windows
from .stability_parallel import run_checkpointed_stability, summarize_stability_worlds


def _selected_parameters(root: Path, source_experiment_id: str) -> pd.DataFrame:
    selection = pd.read_csv(
        root / "results" / source_experiment_id / "full" / "parameter_selection.csv"
    )
    selected = selection.loc[selection["selected"].astype(bool)].copy()
    selected["recovery_rate_per_day"] = 1.0 / selected[
        "mean_infectious_period_days"
    ]
    return selected[
        [
            "parameter_id",
            "beta",
            "mean_infectious_period_days",
            "recovery_rate_per_day",
        ]
    ]


def precision_curve(
    worlds: pd.DataFrame,
    block_levels: list[int],
    top_k: int,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Evaluate label precision using nested prefixes of random blocks."""

    rows: list[dict[str, Any]] = []
    labels_by_level: dict[int, pd.DataFrame] = {}
    maximum_level = max(block_levels)
    for level in block_levels:
        selected_worlds = worlds.loc[worlds["block_id"].lt(level)].copy()
        estimates = summarize_stability_worlds(selected_worlds)
        _, separation, metrics = aggregate_label_precision(estimates, top_k)
        labels = (
            estimates.groupby(["anchor_id", "candidate_id"], observed=True)[
                "unconditional_value"
            ]
            .mean()
            .rename("intervention_value")
            .reset_index()
        )
        labels_by_level[level] = labels
        for record in separation.itertuples(index=False):
            rows.append(
                {
                    "anchor_id": str(record.anchor_id),
                    "random_blocks": level,
                    "candidate_count": int(record.candidate_count),
                    "single_block_rank_correlation": float(
                        record.single_block_rank_correlation
                    ),
                    "averaged_block_rank_reliability": float(
                        record.averaged_block_rank_reliability
                    ),
                    "averaged_block_candidate_separation_icc": float(
                        record.averaged_block_candidate_separation_icc
                    ),
                    "aggregate_reliability": float(
                        metrics["aggregate_label_spearman_brown_reliability"]
                    ),
                }
            )
    maximum = labels_by_level[maximum_level].rename(
        columns={"intervention_value": "maximum_level_value"}
    )
    convergence_rows: list[dict[str, Any]] = []
    for level, labels in labels_by_level.items():
        compared = labels.merge(
            maximum, on=["anchor_id", "candidate_id"], validate="one_to_one"
        )
        for anchor_id, group in compared.groupby("anchor_id", observed=True):
            convergence_rows.append(
                {
                    "anchor_id": anchor_id,
                    "random_blocks": level,
                    "rank_correlation_to_maximum_level": float(
                        group["intervention_value"].rank(method="average").corr(
                            group["maximum_level_value"].rank(method="average")
                        )
                    ),
                    "mean_absolute_value_change_to_maximum_level": float(
                        (group["intervention_value"] - group["maximum_level_value"])
                        .abs()
                        .mean()
                    ),
                }
            )
    curve = pd.DataFrame(rows).merge(
        pd.DataFrame(convergence_rows),
        on=["anchor_id", "random_blocks"],
        validate="one_to_one",
    )
    return curve, labels_by_level


def _plot_curve(curve: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    metrics = [
        ("averaged_block_rank_reliability", "Rank reliability", 0.70),
        (
            "averaged_block_candidate_separation_icc",
            "Candidate-separation ICC",
            0.40,
        ),
        (
            "rank_correlation_to_maximum_level",
            "Rank correlation to largest pilot",
            0.90,
        ),
    ]
    colors = {
        "passes_both": "#4C78A8",
        "low_rank_reliability": "#F28E2B",
        "low_on_both_diagnostics": "#E15759",
    }
    for axis, (column, label, threshold) in zip(axes, metrics):
        for anchor_id, group in curve.groupby("anchor_id", observed=True):
            precision_class = str(group["original_precision_class"].iloc[0])
            axis.plot(
                group["random_blocks"],
                group[column],
                marker="o",
                color=colors[precision_class],
                alpha=0.72,
                linewidth=1.2,
            )
        medians = curve.groupby("random_blocks", observed=True)[column].median()
        axis.plot(
            medians.index,
            medians.values,
            color="#222222",
            marker="D",
            linewidth=2.4,
            label="Pilot median",
        )
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticks(sorted(curve["random_blocks"].unique()))
        axis.set_ylim(-0.12, 1.05)
        axis.set_xlabel("Independent random blocks")
        axis.set_ylabel(label)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=color, marker="o", label=label.replace("_", " "))
        for label, color in colors.items()
    ]
    handles.append(plt.Line2D([0], [0], color="#222222", marker="D", label="pilot median"))
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "Sheep label precision as independent epidemic repeats increase",
        fontsize=17,
        weight="bold",
    )
    _save_figure(figure, path)


def run(config_path: Path) -> tuple[Path, Path]:
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    results_dir = root / config["outputs"]["results_root"] / experiment_id / "full"
    report_dir = root / config["outputs"]["report_root"] / experiment_id / "full"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = CanonicalDataset.read(root / config["data"]["canonical_path"])
    stream = CoalescedDurationContactMapper().compile(dataset)
    phases = _load_phase_by_date(root / config["data"]["raw_measurements_path"])
    prepared, _ = _prepare_network_windows(
        stream,
        dataset,
        config["windows"],
        int(config["windows"]["max_anchors"]),
        phases,
    )
    requested = set(map(str, config["pilot"]["anchor_ids"]))
    selected_windows = [
        window for window in prepared if window["anchor"].anchor_id in requested
    ]
    if {window["anchor"].anchor_id for window in selected_windows} != requested:
        raise ValueError("not all requested sheep pilot anchors were prepared")
    parameters = _selected_parameters(
        root, str(config["experiment"]["source_experiment_id"])
    )
    levels = sorted(map(int, config["pilot"]["random_block_levels"]))
    worlds, _ = run_checkpointed_stability(
        selected_windows,
        parameters,
        config["intervention"],
        random_blocks=max(levels),
        non_index_cases=int(config["pilot"]["non_index_cases_per_candidate_block"]),
        self_replicates=int(config["pilot"]["self_index_replicates_per_block"]),
        candidate_limit=config["pilot"]["candidate_limit"],
        seed=int(config["pilot"]["seed"]),
        checkpoint_dir=results_dir / "checkpoints",
        max_workers=int(config["pilot"]["max_workers"]),
        progress_label="Sheep precision pilot",
    )
    curve, labels = precision_curve(worlds, levels, int(config["stability"]["top_k"]))
    classes = config["pilot"]["original_precision_classes"]
    curve["original_precision_class"] = curve["anchor_id"].map(classes)
    if curve["original_precision_class"].isna().any():
        raise ValueError("pilot precision classes do not cover all requested anchors")
    maximum_rows = curve.loc[curve["random_blocks"].eq(max(levels))]
    gates = config["stability"]["gates"]
    checks = {
        "all_anchor_rank_reliability": bool(
            maximum_rows["averaged_block_rank_reliability"]
            .ge(float(gates["aggregate_label_reliability"]))
            .all()
        ),
        "all_anchor_candidate_separation": bool(
            maximum_rows["averaged_block_candidate_separation_icc"]
            .ge(float(gates["aggregate_label_candidate_separation_icc"]))
            .all()
        ),
    }
    audit = {
        "status": "passed" if all(checks.values()) else "mixed_precision",
        "checks": checks,
        "pilot_anchors": sorted(requested),
        "random_block_levels": levels,
        "maximum_random_blocks": max(levels),
        "paired_worlds": int(len(worlds)),
        "maximum_level_pass_counts": {
            "rank_reliability": int(
                maximum_rows["averaged_block_rank_reliability"]
                .ge(float(gates["aggregate_label_reliability"]))
                .sum()
            ),
            "candidate_separation": int(
                maximum_rows["averaged_block_candidate_separation_icc"]
                .ge(float(gates["aggregate_label_candidate_separation_icc"]))
                .sum()
            ),
            "both": int(
                (
                    maximum_rows["averaged_block_rank_reliability"].ge(
                        float(gates["aggregate_label_reliability"])
                    )
                    & maximum_rows["averaged_block_candidate_separation_icc"].ge(
                        float(gates["aggregate_label_candidate_separation_icc"])
                    )
                ).sum()
            ),
        },
    }
    worlds.to_csv(results_dir / "paired_world_outcomes.csv.gz", index=False, compression="gzip")
    curve.to_csv(results_dir / "precision_curve.csv", index=False)
    curve.to_csv(report_dir / "precision_curve.csv", index=False)
    for level, frame in labels.items():
        frame.to_csv(results_dir / f"labels_at_{level:02d}_blocks.csv", index=False)
    elapsed = time.perf_counter() - started
    manifest = {
        "experiment_id": experiment_id,
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path),
        "random_seed": int(config["pilot"]["seed"]),
        "git": _git_state(root),
    }
    for directory in (results_dir, report_dir):
        (directory / "audit_summary.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (directory / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    _plot_curve(curve, report_dir / "precision_convergence.png")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sheep precision pilot.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    results_dir, report_dir = run(args.config.resolve())
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
