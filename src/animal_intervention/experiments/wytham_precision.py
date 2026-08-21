from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.evaluation import aggregate_label_precision

from .g1_sim import _repository_root, _save_figure
from .stability_parallel import run_checkpointed_stability, summarize_stability_worlds
from .wytham_validation import _host_group_stream, _prepare_windows


def _selected_parameters(root: Path, source_experiment_id: str) -> pd.DataFrame:
    source = root / "results" / source_experiment_id / "full"
    selection = pd.read_csv(source / "parameter_selection.csv")
    selected = selection.loc[selection["selected"]].copy()
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


def _precision_curve(
    worlds: pd.DataFrame, levels: list[int], top_k: int
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    labels_by_level: dict[int, pd.DataFrame] = {}
    maximum_level = max(levels)
    for level in levels:
        selected_worlds = worlds.loc[
            worlds["introduction_stratum"].eq("self_index")
            | worlds["introduction_position"].lt(level)
        ].copy()
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
        for anchor_id, anchor_metrics in separation.groupby("anchor_id", observed=True):
            anchor_worlds = selected_worlds.loc[
                selected_worlds["anchor_id"].eq(anchor_id)
                & selected_worlds["introduction_stratum"].eq("non_index")
            ]
            record = anchor_metrics.iloc[0]
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "non_index_cases_per_candidate_block": level,
                    "random_blocks": int(record["block_count"]),
                    "candidate_count": int(record["candidate_count"]),
                    "non_index_candidate_worlds": len(anchor_worlds),
                    "non_index_zero_fraction": float(
                        anchor_worlds["avoided_infections"].eq(0).mean()
                    ),
                    "single_block_rank_correlation": float(
                        record["single_block_rank_correlation"]
                    ),
                    "averaged_block_rank_reliability": float(
                        record["averaged_block_rank_reliability"]
                    ),
                    "averaged_block_candidate_separation_icc": float(
                        record["averaged_block_candidate_separation_icc"]
                    ),
                    "aggregate_reliability": metrics[
                        "aggregate_label_spearman_brown_reliability"
                    ],
                }
            )
    maximum = labels_by_level[maximum_level].rename(
        columns={"intervention_value": "maximum_level_value"}
    )
    curve = pd.DataFrame(rows)
    convergence_rows = []
    for level, labels in labels_by_level.items():
        compared = labels.merge(
            maximum, on=["anchor_id", "candidate_id"], validate="one_to_one"
        )
        for anchor_id, group in compared.groupby("anchor_id", observed=True):
            convergence_rows.append(
                {
                    "anchor_id": anchor_id,
                    "non_index_cases_per_candidate_block": level,
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
    curve = curve.merge(
        pd.DataFrame(convergence_rows),
        on=["anchor_id", "non_index_cases_per_candidate_block"],
        validate="one_to_one",
    )
    return curve, labels_by_level


def _plot_curve(curve: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
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
    for axis, (column, label, threshold) in zip(axes, metrics):
        for anchor_id, group in curve.groupby("anchor_id", observed=True):
            axis.plot(
                group["non_index_cases_per_candidate_block"],
                group[column],
                marker="o",
                label=anchor_id,
            )
        axis.axhline(threshold, color="#D55E00", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            curve["non_index_cases_per_candidate_block"].unique(),
            curve["non_index_cases_per_candidate_block"].unique(),
        )
        axis.set_ylim(-0.1, 1.05)
        axis.set_xlabel("Non-index introductions per candidate and block")
        axis.set_ylabel(label)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Wytham Monte Carlo precision as outbreak coverage increases",
        fontsize=16,
        weight="bold",
    )
    _save_figure(figure, path)


def run(config_path: Path) -> tuple[Path, Path]:
    root = _repository_root(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    results_dir = root / config["outputs"]["results_root"] / experiment_id / "full"
    report_dir = root / config["outputs"]["report_root"] / experiment_id / "full"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset = CanonicalDataset.read(root / config["data"]["canonical_path"])
    stream = _host_group_stream(dataset, str(config["data"]["host_species_code"]))
    prepared, _ = _prepare_windows(stream, config["windows"], max_anchors=6)
    requested = set(map(str, config["pilot"]["anchor_ids"]))
    selected_windows = [
        window for window in prepared if window["anchor"].anchor_id in requested
    ]
    if {window["anchor"].anchor_id for window in selected_windows} != requested:
        raise ValueError("not all requested Wytham pilot anchors were prepared")
    parameters = _selected_parameters(
        root, str(config["experiment"]["source_experiment_id"])
    )
    levels = sorted(map(int, config["pilot"]["non_index_case_levels"]))
    worlds, _ = run_checkpointed_stability(
        selected_windows,
        parameters,
        config["intervention"],
        random_blocks=int(config["pilot"]["random_blocks"]),
        non_index_cases=max(levels),
        self_replicates=int(config["pilot"]["self_index_replicates_per_block"]),
        candidate_limit=config["pilot"]["candidate_limit"],
        seed=int(config["pilot"]["seed"]),
        checkpoint_dir=results_dir / "checkpoints",
        max_workers=int(config["pilot"]["max_workers"]),
        progress_label="Wytham precision pilot",
    )
    curve, labels = _precision_curve(
        worlds, levels, int(config["stability"]["top_k"])
    )
    maximum_level = max(levels)
    gates = config["stability"]["gates"]
    maximum_rows = curve.loc[
        curve["non_index_cases_per_candidate_block"].eq(maximum_level)
    ]
    checks = {
        "rank_reliability": bool(
            maximum_rows["averaged_block_rank_reliability"].ge(
                float(gates["aggregate_label_reliability"])
            ).all()
        ),
        "candidate_separation": bool(
            maximum_rows["averaged_block_candidate_separation_icc"].ge(
                float(gates["aggregate_label_candidate_separation_icc"])
            ).all()
        ),
    }
    audit = {
        "status": "passed" if all(checks.values()) else "needs_more_precision",
        "checks": checks,
        "pilot_anchors": sorted(requested),
        "maximum_non_index_cases_per_candidate_block": maximum_level,
        "random_blocks": int(config["pilot"]["random_blocks"]),
        "paired_worlds": len(worlds),
    }
    worlds.to_csv(results_dir / "paired_world_outcomes.csv", index=False)
    curve.to_csv(results_dir / "precision_curve.csv", index=False)
    curve.to_csv(report_dir / "precision_curve.csv", index=False)
    for level, frame in labels.items():
        frame.to_csv(results_dir / f"labels_at_{level:03d}_introductions.csv", index=False)
    for directory in (results_dir, report_dir):
        (directory / "audit_summary.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    _plot_curve(curve, report_dir / "precision_convergence.png")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Wytham precision pilot.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    results_dir, report_dir = run(args.config)
    print(f"Results: {results_dir}")
    print(f"Report: {report_dir}")


if __name__ == "__main__":
    main()
