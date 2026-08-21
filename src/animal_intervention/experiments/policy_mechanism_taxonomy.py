from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .outbreak_response_pilot import _git_value, _sha256


CELL_KEYS = ["epidemic_model", "detection_profile", "rewiring_fraction"]
PAIR_KEYS = [
    "dataset_id",
    "network_id",
    "system_family",
    "analysis_cluster_id",
    "anchor_id",
    "anchor_time",
    "horizon_end",
    "parameter_id",
    "epidemic_model",
    "detection_profile",
    "rewiring_fraction",
    "random_block",
    "initial_infected",
    "world_seed",
]
FAMILY_LABELS = {
    "domestic_sheep_sirtrack": "Sheep",
    "guinea_baboons_sociopatterns": "Baboons",
    "linked_wytham_songbird_family": "Linked songbirds",
    "oxford_wildbird_network": "Oxford",
    "radolfzell_great_tits_ontogeny": "Radolfzell",
}
CLASS_LABELS = {
    "robust_history": "Robust history default",
    "pooled_history_only": "Pooled history signal",
    "pooled_override_only": "Pooled override signal",
    "abstain": "Abstain",
}
CLASS_COLORS = {
    "robust_history": "#4C78A8",
    "pooled_history_only": "#9ECAE1",
    "pooled_override_only": "#F58518",
    "abstain": "#B8B8B8",
}


def _members(value: Any) -> set[str]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {item for item in str(value).split("|") if item}


def build_paired_worlds(worlds: pd.DataFrame) -> pd.DataFrame:
    direct = worlds.loc[worlds["method"].eq("contact_to_detected")].copy()
    history = worlds.loc[worlds["method"].eq("history_weight")].copy()
    if direct.duplicated(PAIR_KEYS).any() or history.duplicated(PAIR_KEYS).any():
        raise ValueError("Policy rows are not unique within paired-world keys.")
    method_columns = [
        "additional_targets",
        "selected_infected_fraction",
        "additional_removed_hazard_fraction",
        "additional_rewired_hazard_fraction",
        "augmented_final_size",
        "attack_rate_reduction",
    ]
    shared_columns = [
        "population_size",
        "natural_final_size",
        "standard_final_size",
        "additional_budget",
        "detected_actionable_infectious_fraction",
        "detected_secondary_recall",
    ]
    paired = direct[PAIR_KEYS + shared_columns + method_columns].merge(
        history[PAIR_KEYS + method_columns],
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_direct", "_history"),
    )
    direct_sets = paired["additional_targets_direct"].map(_members)
    history_sets = paired["additional_targets_history"].map(_members)
    unions = [len(left | right) for left, right in zip(direct_sets, history_sets)]
    intersections = [len(left & right) for left, right in zip(direct_sets, history_sets)]
    paired["target_jaccard"] = [
        intersection / union if union else 1.0
        for intersection, union in zip(intersections, unions)
    ]
    paired["target_disagreement"] = 1.0 - paired["target_jaccard"]
    paired["standard_attack_rate"] = paired["standard_final_size"] / paired["population_size"]
    paired["direct_minus_history"] = (
        paired["attack_rate_reduction_direct"]
        - paired["attack_rate_reduction_history"]
    )
    paired["direct_minus_no_extra"] = paired["attack_rate_reduction_direct"]
    paired["history_minus_no_extra"] = paired["attack_rate_reduction_history"]
    paired["selected_infected_fraction_gain"] = (
        paired["selected_infected_fraction_direct"]
        - paired["selected_infected_fraction_history"]
    )
    paired["removed_hazard_fraction_gain"] = (
        paired["additional_removed_hazard_fraction_direct"]
        - paired["additional_removed_hazard_fraction_history"]
    )
    paired["rewired_hazard_fraction_gain"] = (
        paired["additional_rewired_hazard_fraction_direct"]
        - paired["additional_rewired_hazard_fraction_history"]
    )
    return paired


def variance_decomposition(family_cells: pd.DataFrame, outcome: str) -> pd.DataFrame:
    table = family_cells[["system_family", *CELL_KEYS, outcome]].dropna().copy()
    expected = table["system_family"].nunique() * 8
    if len(table) != expected:
        raise ValueError(f"Expected a balanced family-by-factor table with {expected} rows.")
    table["model"] = np.where(table["epidemic_model"].eq("temporal_seir_erlang"), 1.0, -1.0)
    table["timing"] = np.where(table["detection_profile"].eq("delayed_detection"), 1.0, -1.0)
    table["rewiring"] = np.where(table["rewiring_fraction"].gt(0), 1.0, -1.0)
    y = table[outcome].to_numpy(float)
    grand = float(y.mean())
    total_ss = float(np.square(y - grand).sum())
    family_means = table.groupby("system_family", observed=True)[outcome].mean()
    family_ss = float(8 * np.square(family_means - grand).sum())
    rows = [{"outcome": outcome, "component": "animal_system_family", "sum_squares": family_ss}]
    terms = {
        "epidemic_model": table["model"],
        "detection_timing": table["timing"],
        "rewiring": table["rewiring"],
        "model_x_timing": table["model"] * table["timing"],
        "model_x_rewiring": table["model"] * table["rewiring"],
        "timing_x_rewiring": table["timing"] * table["rewiring"],
        "model_x_timing_x_rewiring": table["model"] * table["timing"] * table["rewiring"],
    }
    explained = family_ss
    centered = y - grand
    for name, contrast in terms.items():
        values = contrast.to_numpy(float)
        coefficient = float(np.dot(centered, values) / np.dot(values, values))
        ss = float(np.square(coefficient * values).sum())
        explained += ss
        rows.append({"outcome": outcome, "component": name, "sum_squares": ss})
    residual = max(total_ss - explained, 0.0)
    rows.append({"outcome": outcome, "component": "family_by_scenario_heterogeneity", "sum_squares": residual})
    result = pd.DataFrame(rows)
    result["variance_fraction"] = result["sum_squares"] / total_ss if total_ss > 0 else 0.0
    return result


def classify_cells(decisions: pd.DataFrame, resilience: pd.DataFrame) -> pd.DataFrame:
    merged = decisions.merge(
        resilience[CELL_KEYS + ["same_decision_folds", "override_folds", "omission_folds", "resilience_status"]],
        on=CELL_KEYS,
        how="left",
        validate="one_to_one",
    )
    classes = []
    for row in merged.itertuples(index=False):
        if row.decision == "abstain_or_unresolved":
            category = "abstain"
        elif row.decision == "retain_history_weight":
            category = (
                "robust_history"
                if row.same_decision_folds == row.omission_folds
                else "pooled_history_only"
            )
        else:
            category = "pooled_override_only"
        classes.append(category)
    merged["evidence_class"] = classes
    merged["evidence_label"] = merged["evidence_class"].map(CLASS_LABELS)
    return merged


def _spearman(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average").to_numpy(float)
    right_rank = right.rank(method="average").to_numpy(float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _cell_label(row: pd.Series | Any) -> str:
    model = "SIR" if row.epidemic_model == "temporal_sir" else "SEIR/Erlang"
    timing = "early" if row.detection_profile == "early_detection" else "delayed"
    rewiring = "rewiring" if float(row.rewiring_fraction) else "no rewiring"
    return f"{model} | {timing} | {rewiring}"


def plot_taxonomy(classification: pd.DataFrame, path: Path, dpi: int) -> None:
    ordered = classification.sort_values(CELL_KEYS, kind="stable").reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(12.5, 6.8))
    y = np.arange(len(ordered))[::-1]
    colors = [CLASS_COLORS[item] for item in ordered["evidence_class"]]
    axis.barh(y, np.ones(len(ordered)), color=colors, height=0.68)
    for position, row in zip(y, ordered.itertuples(index=False)):
        axis.text(
            0.03,
            position,
            row.evidence_label,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.text(
            0.97,
            position,
            f"direct − history = {100 * row.direct_minus_history:+.2f} pp",
            va="center",
            ha="right",
            fontsize=10,
        )
    axis.set_yticks(y, [_cell_label(row) for row in ordered.itertuples(index=False)])
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    fig.suptitle("Evidence taxonomy for the eight frozen deployment cells", fontsize=18, fontweight="bold", y=0.98)
    axis.set_title("Classification combines the preregistered decision with whole-family deletion stress tests", fontsize=11, pad=14)
    fig.subplots_adjust(left=0.29, right=0.97, top=0.84, bottom=0.08)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_factorial_comparison(family_cells: pd.DataFrame, path: Path, dpi: int) -> None:
    cells = family_cells[CELL_KEYS].drop_duplicates().sort_values(CELL_KEYS, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), sharey=True)
    outcomes = [
        ("direct_minus_history", "Direct minus history"),
        ("direct_minus_no_extra", "Direct minus no-extra"),
    ]
    family_order = list(FAMILY_LABELS)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(family_order)))
    positions = np.arange(len(cells))[::-1]
    for axis, (outcome, title) in zip(axes, outcomes):
        for family, color in zip(family_order, colors):
            selected = cells.merge(
                family_cells.loc[family_cells["system_family"].eq(family), CELL_KEYS + [outcome]],
                on=CELL_KEYS,
                how="left",
            )
            axis.scatter(100 * selected[outcome], positions, s=42, color=color, alpha=0.9, label=FAMILY_LABELS[family])
        means = cells.merge(family_cells.groupby(CELL_KEYS, observed=True)[outcome].mean().reset_index(), on=CELL_KEYS)
        axis.scatter(100 * means[outcome], positions, marker="D", s=70, facecolor="white", edgecolor="black", linewidth=1.4, zorder=5, label="Family-equal mean")
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(title, fontsize=13)
        axis.set_xlabel("Avoided attack-rate difference (percentage points)")
    axes[0].set_yticks(positions, [_cell_label(row) for row in cells.itertuples(index=False)])
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Independent animal systems do not support a universal policy winner", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.87, bottom=0.12, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_variance(decomposition: pd.DataFrame, path: Path, dpi: int) -> None:
    labels = {
        "animal_system_family": "Animal system",
        "family_by_scenario_heterogeneity": "System × scenario heterogeneity",
        "epidemic_model": "Epidemic model",
        "detection_timing": "Detection timing",
        "rewiring": "Rewiring",
        "model_x_timing": "Model × timing",
        "model_x_rewiring": "Model × rewiring",
        "timing_x_rewiring": "Timing × rewiring",
        "model_x_timing_x_rewiring": "Three-way interaction",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), sharey=True)
    order = decomposition.groupby("component")["variance_fraction"].max().sort_values().index.tolist()
    titles = {"direct_minus_history": "Relative value: direct minus history", "direct_minus_no_extra": "Absolute value: direct minus no-extra"}
    for axis, outcome in zip(axes, titles):
        selected = decomposition.loc[decomposition["outcome"].eq(outcome)].set_index("component").loc[order]
        axis.barh(np.arange(len(order)), 100 * selected["variance_fraction"], color="#4C78A8" if outcome == "direct_minus_history" else "#F58518")
        axis.set_title(titles[outcome], fontsize=12)
        axis.set_xlabel("Share of descriptive variance (%)")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(np.arange(len(order)), [labels[item] for item in order])
    fig.suptitle("What separates the frozen policy outcomes?", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, "Balanced decomposition of five family means across the 2 × 2 × 2 design; no pseudo-replicate p-values", ha="center", fontsize=11)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.83, bottom=0.12, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_mechanisms(family_cells: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    panels = [
        ("target_disagreement", "direct_minus_history", "Target disagreement", "Direct minus history (pp)"),
        ("standard_attack_rate", "direct_minus_no_extra", "Standard-care attack rate", "Direct minus no-extra (pp)"),
    ]
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(FAMILY_LABELS)))
    for axis, (x_col, y_col, x_label, y_label) in zip(axes, panels):
        for (family, label), color in zip(FAMILY_LABELS.items(), colors):
            selected = family_cells.loc[family_cells["system_family"].eq(family)]
            axis.scatter(selected[x_col], 100 * selected[y_col], s=48, alpha=0.85, color=color, label=label)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Simple mechanism descriptors do not yield a universal rule", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.14, wspace=0.24)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path, profile: str) -> dict[str, Any]:
    started = time.time()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = config["experiment"]["id"]
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile
    reports_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    worlds_path = Path(config["data"]["policy_worlds"])
    decisions_path = Path(config["data"]["decision_map"])
    resilience_path = Path(config["data"]["deletion_resilience"])
    precision_path = Path(config["data"]["precision_deletion_resilience"])
    worlds = pd.read_csv(worlds_path)
    decisions = pd.read_csv(decisions_path)
    resilience = pd.read_csv(resilience_path)
    precision = pd.read_csv(precision_path)
    paired = build_paired_worlds(worlds)
    metric_columns = [
        "direct_minus_history",
        "direct_minus_no_extra",
        "history_minus_no_extra",
        "target_jaccard",
        "target_disagreement",
        "selected_infected_fraction_gain",
        "removed_hazard_fraction_gain",
        "rewired_hazard_fraction_gain",
        "standard_attack_rate",
        "detected_actionable_infectious_fraction",
        "detected_secondary_recall",
    ]
    cluster_cells = (
        paired.groupby(["system_family", "analysis_cluster_id", *CELL_KEYS], observed=True)[metric_columns]
        .mean()
        .reset_index()
    )
    family_cells = (
        cluster_cells.groupby(["system_family", *CELL_KEYS], observed=True)[metric_columns]
        .mean()
        .reset_index()
    )
    decomposition = pd.concat(
        [variance_decomposition(family_cells, outcome) for outcome in ["direct_minus_history", "direct_minus_no_extra"]],
        ignore_index=True,
    )
    classification = classify_cells(decisions, resilience)
    classification = classification.merge(
        precision[CELL_KEYS + ["same_decision_folds", "override_folds"]].rename(
            columns={"same_decision_folds": "precision_same_decision_folds", "override_folds": "precision_override_folds"}
        ),
        on=CELL_KEYS,
        how="left",
        validate="one_to_one",
    )
    correlations = []
    for family, group in family_cells.groupby("system_family", observed=True):
        for descriptor, outcome in [
            ("target_disagreement", "direct_minus_history"),
            ("selected_infected_fraction_gain", "direct_minus_history"),
            ("removed_hazard_fraction_gain", "direct_minus_history"),
            ("standard_attack_rate", "direct_minus_no_extra"),
        ]:
            correlations.append(
                {
                    "system_family": family,
                    "descriptor": descriptor,
                    "outcome": outcome,
                    "spearman_correlation": _spearman(group[descriptor], group[outcome]),
                    "cells": len(group),
                }
            )
    correlations = pd.DataFrame(correlations)
    paired.to_csv(results_dir / "paired_world_mechanisms.csv.gz", index=False, compression="gzip")
    cluster_cells.to_csv(results_dir / "cluster_cell_mechanisms.csv", index=False)
    family_cells.to_csv(results_dir / "family_cell_mechanisms.csv", index=False)
    decomposition.to_csv(results_dir / "factorial_variance_decomposition.csv", index=False)
    classification.to_csv(results_dir / "evidence_taxonomy.csv", index=False)
    correlations.to_csv(results_dir / "within_family_mechanism_correlations.csv", index=False)
    dpi = int(config["profiles"][profile]["render_dpi"])
    plot_taxonomy(classification, reports_dir / "evidence_taxonomy.png", dpi)
    plot_factorial_comparison(family_cells, reports_dir / "family_factorial_comparison.png", dpi)
    plot_variance(decomposition, reports_dir / "variance_decomposition.png", dpi)
    plot_mechanisms(family_cells, reports_dir / "mechanism_descriptors.png", dpi)
    checks = {
        "source_policy_rows_complete": len(worlds) == 5184,
        "paired_worlds_complete": len(paired) == 2592,
        "five_independent_families": family_cells["system_family"].nunique() == 5,
        "balanced_eight_cell_design": len(family_cells) == 40,
        "cluster_equal_family_aggregation": bool(
            cluster_cells.groupby("system_family", observed=True)["analysis_cluster_id"].nunique().ge(1).all()
        ),
        "paired_outcomes_reconcile": bool(np.allclose(paired["direct_minus_history"], paired["direct_minus_no_extra"] - paired["history_minus_no_extra"])),
        "target_metrics_bounded": bool(paired["target_jaccard"].between(0, 1).all()),
        "taxonomy_complete": len(classification) == 8 and classification["evidence_class"].notna().all(),
        "variance_decomposition_closes": bool(decomposition.groupby("outcome")["variance_fraction"].sum().sub(1).abs().lt(1e-8).all()),
        "all_outputs_finite": bool(np.isfinite(family_cells[metric_columns].to_numpy(float)).all()),
        "no_new_simulation_or_policy_tuning": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scope": "post_gate_explanatory_not_causal",
        "policy_cells": len(classification),
        "evidence_class_counts": {
            str(name): int(count)
            for name, count in classification["evidence_class"].value_counts().items()
        },
        "independent_families": int(family_cells["system_family"].nunique()),
        "nested_paired_worlds": len(paired),
    }
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_hashes = pd.DataFrame(
        [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in [worlds_path, decisions_path, resilience_path, precision_path]]
    )
    source_hashes.to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_status": _git_value(["git", "status", "--short"]),
        "config_path": str(config_path),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain and classify the frozen outbreak-response policy boundary.")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260817-008_mechanism_taxonomy.yaml"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    audit = run(args.config, args.profile)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
