from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.evaluation import stable_hash_order
from animal_intervention.simulation import DetectionProfile, detection_time_from_seed

from .outbreak_response_pilot import (
    DATASET_LABELS,
    _git_value,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _selected_parameters,
    _sha256,
)
from .set_value_pilot import (
    CONTEXT_KEYS,
    FEATURE_COLUMNS,
    _context_summary,
    _run_contexts,
)


ANCHOR_KEYS = ["dataset_id", "network_id", "anchor_id"]


def _dataset_budget_summary(contexts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_id, budget_fraction), group in contexts.groupby(
        ["dataset_id", "budget_fraction"], observed=True, sort=True
    ):
        multinode = group.loc[group["budget"].gt(1)]
        rows.append(
            {
                "dataset_id": dataset_id,
                "budget_fraction": budget_fraction,
                "contexts": len(group),
                "multinode_contexts": len(multinode),
                "median_budget": float(group["budget"].median()),
                "variable_multinode_context_fraction": (
                    float(multinode["distinct_values"].gt(1).mean())
                    if len(multinode)
                    else np.nan
                ),
                "mean_multinode_spread": (
                    float(multinode["value_spread"].mean())
                    if len(multinode)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _anchor_summary(contexts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ANCHOR_KEYS + ["budget_fraction"]
    for key, group in contexts.groupby(keys, observed=True, sort=True):
        multinode = group.loc[group["budget"].gt(1)]
        rows.append(
            {
                **dict(zip(keys, key)),
                "contexts": len(group),
                "multinode_contexts": len(multinode),
                "variable_multinode_context_fraction": (
                    float(multinode["distinct_values"].gt(1).mean())
                    if len(multinode)
                    else np.nan
                ),
                "mean_multinode_spread": (
                    float(multinode["value_spread"].mean())
                    if len(multinode)
                    else np.nan
                ),
                "has_variable_multinode_context": bool(
                    len(multinode) and multinode["distinct_values"].gt(1).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def _family_gate_summary(
    contexts: pd.DataFrame,
    *, minimum_variable_fraction: float,
    minimum_variable_anchors: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in contexts.groupby(
        "system_family", observed=True, sort=True
    ):
        multinode = group.loc[group["budget"].gt(1)]
        variable = multinode.loc[multinode["distinct_values"].gt(1)]
        variable_anchors = variable[ANCHOR_KEYS].drop_duplicates().shape[0]
        fraction = (
            float(multinode["distinct_values"].gt(1).mean())
            if len(multinode)
            else 0.0
        )
        rows.append(
            {
                "system_family": family,
                "contexts": len(group),
                "multinode_contexts": len(multinode),
                "multinode_anchors": multinode[ANCHOR_KEYS].drop_duplicates().shape[0],
                "variable_multinode_context_fraction": fraction,
                "variable_anchors": variable_anchors,
                "mean_multinode_spread": (
                    float(multinode["value_spread"].mean())
                    if len(multinode)
                    else 0.0
                ),
                "qualifies": bool(
                    len(multinode)
                    and fraction >= minimum_variable_fraction
                    and variable_anchors >= minimum_variable_anchors
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_dataset_budget(
    summary: pd.DataFrame, output_path: Path
) -> None:
    datasets = list(summary["dataset_id"].drop_duplicates())
    fractions = sorted(summary["budget_fraction"].unique())
    y = np.arange(len(datasets))
    height = 0.34
    colors = ["#4C78A8", "#F58518"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for position, fraction in enumerate(fractions):
        frame = summary.loc[summary["budget_fraction"].eq(fraction)].set_index(
            "dataset_id"
        ).reindex(datasets)
        offset = (position - (len(fractions) - 1) / 2) * height
        values = frame["variable_multinode_context_fraction"].fillna(0)
        bars = axes[0].barh(
            y + offset,
            values,
            height,
            color=colors[position],
            label=f"{100 * fraction:.0f}% budget",
        )
        axes[1].barh(
            y + offset,
            100 * frame["mean_multinode_spread"].fillna(0),
            height,
            color=colors[position],
            label=f"{100 * fraction:.0f}% budget",
        )
        for bar, count in zip(bars, frame["multinode_contexts"]):
            if count == 0:
                axes[0].text(
                    0.01,
                    bar.get_y() + bar.get_height() / 2,
                    "singleton",
                    va="center",
                    color="#666666",
                    fontsize=9,
                )
    labels = [DATASET_LABELS.get(item, item) for item in datasets]
    for axis in axes:
        axis.set_yticks(y, labels)
        axis.grid(axis="x", alpha=0.18)
        axis.legend(frameon=False, loc="lower right")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Variable multi-node context fraction")
    axes[0].set_title("Set-value separation", fontweight="bold")
    axes[1].set_xlabel("Mean best-minus-worst value (percentage points)")
    axes[1].set_title("Outcome spread", fontweight="bold")
    maximum_spread = float((100 * summary["mean_multinode_spread"].fillna(0)).max())
    axes[1].set_xlim(0, max(1.0, 1.08 * maximum_spread))
    fig.suptitle(
        "Fixed-budget set-label support across datasets",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.92,
        "Budget-1 contexts are shown as singleton boundary cases and do not count toward the set gate",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.83, bottom=0.12, wspace=0.42)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_anchor_support(summary: pd.DataFrame, output_path: Path) -> None:
    datasets = list(summary["dataset_id"].drop_duplicates())
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), squeeze=False)
    colors = {0.05: "#4C78A8", 0.10: "#F58518"}
    markers = {0.05: "o", 0.10: "x"}
    for axis, dataset_id in zip(axes.flat, datasets):
        frame = summary.loc[summary["dataset_id"].eq(dataset_id)].copy()
        anchor_units = frame[["network_id", "anchor_id"]].drop_duplicates()
        order = {
            tuple(row): index
            for index, row in enumerate(anchor_units.itertuples(index=False, name=None))
        }
        for fraction, group in frame.groupby("budget_fraction", observed=True):
            x = [order[(row.network_id, row.anchor_id)] for row in group.itertuples()]
            y = group["variable_multinode_context_fraction"]
            axis.scatter(
                x,
                y,
                s=34,
                color=colors.get(float(fraction), "#777777"),
                marker=markers.get(float(fraction), "o"),
                label=f"{100 * fraction:.0f}% budget",
            )
        axis.axhline(0.10, color="#555555", linestyle="--", linewidth=1)
        axis.set_ylim(-0.04, 1.04)
        if len(order) == 1:
            axis.set_xlim(-0.5, 0.5)
            axis.set_xticks([0])
        if frame["multinode_contexts"].fillna(0).eq(0).all():
            axis.text(
                0.5,
                0.5,
                "Both budgets resolve to one animal\n(singleton boundary; excluded from set gate)",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontsize=9,
            )
        axis.set_title(DATASET_LABELS.get(dataset_id, dataset_id), fontweight="bold")
        axis.set_xlabel("Ordered forward anchor")
        axis.set_ylabel("Variable-context fraction")
        axis.grid(alpha=0.16)
    for axis in axes.flat[len(datasets):]:
        axis.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.925))
    fig.suptitle("Temporal coverage of set-label separation", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.875, "Dashed line marks the 10% within-family support threshold", ha="center", color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.07, hspace=0.48, wspace=0.28)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_family_gate(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary.sort_values("variable_multinode_context_fraction")
    labels = [item.replace("_", " ") for item in frame["system_family"]]
    colors = ["#4C78A8" if value else "#B8B8B8" for value in frame["qualifies"]]
    fig, axis = plt.subplots(figsize=(11, 6))
    bars = axis.barh(labels, frame["variable_multinode_context_fraction"], color=colors)
    axis.axvline(0.10, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Variable multi-node context fraction")
    axis.grid(axis="x", alpha=0.18)
    for bar, anchors in zip(bars, frame["variable_anchors"]):
        text_x = bar.get_width() + 0.02
        if bar.get_width() < 0.10:
            text_x = 0.12
        axis.text(
            min(text_x, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{int(anchors)} variable anchors",
            va="center",
            fontsize=10,
        )
    fig.suptitle(
        "Independent-family set-label gate",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.90,
        "A family qualifies only with >=10% variable contexts and >=2 distinct variable anchors",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.31, right=0.98, top=0.80, bottom=0.13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][profile_name]
    experiment_id = str(config["experiment"]["id"])
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stable_path = Path(config["data"]["stable_prediction_path"])
    predictions = pd.read_csv(
        stable_path, dtype={"candidate_id": str, "network_id": str}
    )
    predictions["anchor_time"] = pd.to_datetime(
        predictions["anchor_time"], format="mixed"
    )
    fingerprint = hashlib.sha256(
        config_path.read_bytes()
        + stable_path.read_bytes()
        + Path(__file__).read_bytes()
        + Path(__file__).with_name("set_value_pilot.py").read_bytes()
    ).hexdigest()[:12]
    detections = [
        DetectionProfile(**item) for item in config["decision"]["detection_profiles"]
    ]
    budgets = list(map(float, config["decision"]["budget_fractions"]))
    tasks: list[tuple[Any, ...]] = []
    support_rows: list[dict[str, Any]] = []

    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network = str(specification.get("network_id", "all"))
        for window in windows:
            window.setdefault("network_id", default_network)
        available = set(
            predictions.loc[
                predictions["dataset_id"].eq(dataset_id),
                ["network_id", "anchor_time"],
            ].itertuples(index=False, name=None)
        )
        windows = [
            window
            for window in windows
            if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time))
            in available
        ]
        maximum = profile.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        parameters = _selected_parameters(
            Path(specification["source_results"]) / "parameter_selection.csv", None
        )
        for window in windows:
            anchor = window["anchor"]
            compatible = []
            for parameter in parameters.itertuples(index=False):
                supported = all(
                    detection_time_from_seed(
                        anchor.anchor_time,
                        anchor.horizon_end,
                        pd.Timedelta(days=float(parameter.mean_infectious_period_days)),
                        detection,
                    )
                    is not None
                    for detection in detections
                )
                support_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "network_id": str(window["network_id"]),
                        "anchor_id": anchor.anchor_id,
                        "parameter_id": parameter.parameter_id,
                        "mean_attack_rate": float(parameter.mean_attack_rate),
                        "supported": supported,
                        "selected": False,
                    }
                )
                if supported:
                    compatible.append(parameter)
            if not compatible:
                continue
            compatible.sort(key=lambda item: float(item.mean_attack_rate))
            parameter = compatible[len(compatible) // 2]
            for row in support_rows:
                if (
                    row["dataset_id"] == dataset_id
                    and row["network_id"] == str(window["network_id"])
                    and row["anchor_id"] == anchor.anchor_id
                    and row["parameter_id"] == parameter.parameter_id
                ):
                    row["selected"] = True
            for detection in detections:
                for evidence in config["decision"]["evidence_profiles"]:
                    for budget in budgets:
                        tasks.append(
                            (
                                dataset_id,
                                specification,
                                window,
                                parameter,
                                detection,
                                evidence,
                                budget,
                            )
                        )

    frames: list[pd.DataFrame] = []
    progress = tqdm(tasks, desc=f"Set coverage {profile_name}", unit="task")
    for dataset_id, specification, window, parameter, detection, evidence, budget in progress:
        anchor = window["anchor"]
        identity = (
            f"{fingerprint}|{dataset_id}|{window['network_id']}|{anchor.anchor_id}|"
            f"{parameter.parameter_id}|{detection.name}|{evidence['name']}|{budget}"
        )
        checkpoint = checkpoint_dir / (
            f"{dataset_id}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.csv.gz"
        )
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(
                checkpoint,
                dtype={"initial_infected": str, "network_id": str},
            )
            if not frame.empty:
                frames.append(frame)
                progress.set_postfix_str(f"{dataset_id} cached")
                continue
        stable = _matching_stable_scores(
            predictions,
            dataset_id,
            str(window["network_id"]),
            anchor.anchor_time,
            window["eligible"],
        )
        seeds = stable_hash_order(
            list(map(str, window["eligible"])),
            int(config["evaluation"]["seed"]),
            dataset_id,
            anchor.anchor_id,
            "set_coverage_seeds",
        )[: int(profile["seeds_per_anchor"])]
        frame = _run_contexts(
            dataset_id=dataset_id,
            system_family=str(specification["system_family"]),
            window=window,
            parameter=parameter,
            detection_profile=detection,
            evidence_profile=str(evidence["name"]),
            secondary_case_sensitivity=float(evidence["secondary_case_sensitivity"]),
            stable_scores=stable,
            seed_nodes=seeds,
            config=config,
            budget_fraction=budget,
            random_blocks=int(profile["random_blocks"]),
        )
        frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
        progress.set_postfix_str(f"{dataset_id} complete")

    if not frames:
        raise ValueError("set coverage produced no rows")
    values = pd.concat(frames, ignore_index=True)
    contexts = _context_summary(values)
    dataset_budget = _dataset_budget_summary(contexts)
    anchor_summary = _anchor_summary(contexts)
    family_summary = _family_gate_summary(
        contexts,
        minimum_variable_fraction=float(
            config["evaluation"]["minimum_family_variable_context_fraction"]
        ),
        minimum_variable_anchors=int(
            config["evaluation"]["minimum_variable_anchors_per_family"]
        ),
    )
    eligible_families = family_summary.loc[
        family_summary["multinode_contexts"].gt(0)
    ]
    family_equal_fraction = (
        float(eligible_families["variable_multinode_context_fraction"].mean())
        if len(eligible_families)
        else 0.0
    )
    qualifying_families = int(family_summary["qualifies"].sum())
    enforce_gate = bool(profile.get("enforce_scientific_gate", False))
    scientific_pass = bool(
        qualifying_families
        >= int(config["evaluation"]["minimum_qualifying_families"])
        and family_equal_fraction
        >= float(config["evaluation"]["minimum_family_equal_variable_fraction"])
    )

    values.to_csv(
        results_dir / "sampled_set_values.csv.gz", index=False, compression="gzip"
    )
    contexts.to_csv(results_dir / "context_set_summary.csv", index=False)
    dataset_budget.to_csv(results_dir / "dataset_budget_summary.csv", index=False)
    anchor_summary.to_csv(results_dir / "anchor_support_summary.csv", index=False)
    family_summary.to_csv(results_dir / "family_gate_summary.csv", index=False)
    pd.DataFrame(support_rows).to_csv(
        results_dir / "parameter_detection_support.csv", index=False
    )

    detected_disjoint = values.apply(
        lambda row: not bool(
            set(str(row.selected_nodes).split("|"))
            & set(str(row.detected_nodes).split("|"))
        ),
        axis=1,
    )
    set_sizes = values["selected_nodes"].map(lambda value: len(str(value).split("|")))
    shared_keys = [key for key in CONTEXT_KEYS if key != "budget_fraction"]
    standard_across_budgets = values.groupby(
        shared_keys, observed=True
    )["standard_final_size"].nunique()
    detected_across_budgets = values.groupby(
        shared_keys, observed=True
    )["detected_nodes"].nunique()
    checks = {
        "set_keys_unique": not values.duplicated(
            CONTEXT_KEYS + ["set_signature"]
        ).any(),
        "detected_nodes_excluded": bool(detected_disjoint.all()),
        "fixed_budget_within_context": bool(set_sizes.eq(values["budget"]).all()),
        "standard_shared_within_context": bool(
            values.groupby(CONTEXT_KEYS, observed=True)["standard_final_size"]
            .nunique()
            .eq(1)
            .all()
        ),
        "natural_and_standard_shared_across_budgets": bool(
            standard_across_budgets.eq(1).all()
        ),
        "detected_evidence_shared_across_budgets": bool(
            detected_across_budgets.eq(1).all()
        ),
        "paired_arithmetic": bool(
            np.allclose(
                values["set_attack_rate_value"],
                (values["standard_final_size"] - values["set_final_size"])
                / values["population_size"],
            )
        ),
        "finite_features_and_outcomes": bool(
            np.isfinite(
                values[FEATURE_COLUMNS + ["set_attack_rate_value"]].to_numpy(float)
            ).all()
        ),
        "configured_random_blocks_present": bool(
            values.groupby(
                [
                    "dataset_id",
                    "network_id",
                    "anchor_id",
                    "parameter_id",
                    "detection_profile",
                    "evidence_profile",
                    "budget_fraction",
                    "initial_infected",
                ],
                observed=True,
            )["random_block"]
            .nunique()
            .eq(int(profile["random_blocks"]))
            .all()
        ),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scientific_gate": {
            "status": (
                "pass" if scientific_pass else "fail"
            )
            if enforce_gate
            else "not_evaluated",
            "qualifying_families": qualifying_families,
            "minimum_qualifying_families": int(
                config["evaluation"]["minimum_qualifying_families"]
            ),
            "family_equal_variable_context_fraction": family_equal_fraction,
            "minimum_family_equal_variable_fraction": float(
                config["evaluation"]["minimum_family_equal_variable_fraction"]
            ),
            "minimum_family_variable_context_fraction": float(
                config["evaluation"]["minimum_family_variable_context_fraction"]
            ),
            "minimum_variable_anchors_per_family": int(
                config["evaluation"]["minimum_variable_anchors_per_family"]
            ),
        },
        "sets": len(values),
        "contexts": len(contexts),
        "anchors": contexts[ANCHOR_KEYS].drop_duplicates().shape[0],
        "datasets": values["dataset_id"].nunique(),
        "families": values["system_family"].nunique(),
    }
    if audit["status"] != "pass":
        raise ValueError(f"set-coverage artifact audit failed: {audit}")
    (results_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    _plot_dataset_budget(
        dataset_budget, report_dir / "dataset_budget_support.png"
    )
    _plot_anchor_support(anchor_summary, report_dir / "anchor_support.png")
    _plot_family_gate(family_summary, report_dir / "family_gate.png")

    gate_status = audit["scientific_gate"]["status"]
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "python": platform.python_version(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "artifact_audit": audit["status"],
        "scientific_gate": gate_status,
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    family_lines = "\n".join(
        f"- {row.system_family}: {row.variable_multinode_context_fraction:.1%} variable contexts, "
        f"{int(row.variable_anchors)} variable anchors, qualifies={bool(row.qualifies)}."
        for row in family_summary.itertuples(index=False)
    )
    (report_dir / "README.md").write_text(
        f"# Multi-anchor fixed-budget set-label coverage\n\n"
        f"Profile: **{profile_name}**. Sets: {len(values):,}; contexts: {len(contexts):,}; "
        f"anchors: {audit['anchors']}; datasets: {audit['datasets']}; independent families: {audit['families']}. "
        f"Artifact audit: **{audit['status']}**. Scientific gate: **{gate_status}**.\n\n"
        f"Family-equal variable-context fraction: {family_equal_fraction:.1%}; "
        f"qualifying independent families: {qualifying_families}.\n\n{family_lines}\n\n"
        "This stage audits label support only. It does not train or evaluate a learned planner.\n",
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-anchor fixed-budget set-label coverage audit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260816-012_set_value_coverage.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
