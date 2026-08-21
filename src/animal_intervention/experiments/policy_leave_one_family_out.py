from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import yaml

from .history_baseline_substitution import _decision_map, _markdown_table
from .intervention_delivery_sensitivity import _hierarchical_summary
from .outbreak_response_pilot import _git_value, _sha256


CELL_KEYS = ["epidemic_model", "detection_profile", "rewiring_fraction"]
DECISION_ORDER = [
    "abstain_or_unresolved",
    "retain_history_weight",
    "override_with_detected_case_contacts",
]
DECISION_LABELS = {
    "abstain_or_unresolved": "Abstain",
    "retain_history_weight": "Retain history",
    "override_with_detected_case_contacts": "Override",
}
FAMILY_LABELS = {
    "domestic_sheep_sirtrack": "Sheep",
    "guinea_baboons_sociopatterns": "Baboons",
    "linked_wytham_songbird_family": "Linked songbirds",
    "oxford_wildbird_network": "Oxford",
    "radolfzell_great_tits_ontogeny": "Radolfzell",
}


def _cell_label(row: pd.Series | Any) -> str:
    model = "SIR" if row.epidemic_model == "temporal_sir" else "SEIR/Erlang"
    timing = "early" if row.detection_profile == "early_detection" else "delayed"
    rewiring = "no rewiring" if float(row.rewiring_fraction) == 0 else "full rewiring"
    return f"{model} | {timing} | {rewiring}"


def _summarize_resilience(
    full_decisions: pd.DataFrame, loo_decisions: pd.DataFrame, total_omissions: int
) -> pd.DataFrame:
    rows = []
    for full in full_decisions.itertuples(index=False):
        selected = loo_decisions.copy()
        for key in CELL_KEYS:
            selected = selected.loc[selected[key].eq(getattr(full, key))]
        same = int(selected["decision"].eq(full.decision).sum())
        overrides = int(
            selected["decision"].eq("override_with_detected_case_contacts").sum()
        )
        if full.decision == "override_with_detected_case_contacts":
            if overrides == total_omissions:
                status = "deletion_robust"
            elif overrides >= max(total_omissions - 1, 1):
                status = "mostly_robust"
            else:
                status = "fragile"
        else:
            status = "decision_invariant" if same == total_omissions else "decision_changes"
        rows.append(
            {
                **{key: getattr(full, key) for key in CELL_KEYS},
                "full_decision": full.decision,
                "omission_folds": total_omissions,
                "same_decision_folds": same,
                "override_folds": overrides,
                "relative_ci_low_min": float(selected["direct_minus_history_ci_low"].min()),
                "relative_ci_low_max": float(selected["direct_minus_history_ci_low"].max()),
                "absolute_ci_low_min": float(selected["direct_absolute_ci_low"].min()),
                "absolute_ci_low_max": float(selected["direct_absolute_ci_low"].max()),
                "resilience_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(CELL_KEYS, kind="stable").reset_index(drop=True)


def _plot_decision_deletions(
    full_decisions: pd.DataFrame,
    loo_decisions: pd.DataFrame,
    families: list[str],
    path: Path,
) -> None:
    ordered = full_decisions.sort_values(CELL_KEYS, kind="stable").reset_index(drop=True)
    columns = ["Full"] + [f"Without\n{FAMILY_LABELS.get(item, item)}" for item in families]
    matrix = np.zeros((len(ordered), len(columns)), dtype=int)
    annotations: list[list[str]] = [["" for _ in columns] for _ in range(len(ordered))]
    for row_index, full in ordered.iterrows():
        matrix[row_index, 0] = DECISION_ORDER.index(full["decision"])
        annotations[row_index][0] = DECISION_LABELS[full["decision"]]
        for column_index, family in enumerate(families, start=1):
            selected = loo_decisions.loc[loo_decisions["omitted_family"].eq(family)]
            for key in CELL_KEYS:
                selected = selected.loc[selected[key].eq(full[key])]
            decision = str(selected.iloc[0]["decision"])
            matrix[row_index, column_index] = DECISION_ORDER.index(decision)
            annotations[row_index][column_index] = DECISION_LABELS[decision]
    color_map = ListedColormap(["#D9D9D9", "#4C78A8", "#F58518"])
    fig, axis = plt.subplots(figsize=(13.5, 7.4))
    image = axis.imshow(matrix, cmap=color_map, vmin=-0.5, vmax=2.5, aspect="auto")
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(ordered)), [_cell_label(row) for _, row in ordered.iterrows()])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, annotations[row][column], ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=axis, ticks=range(3), pad=0.02)
    colorbar.ax.set_yticklabels([DECISION_LABELS[item] for item in DECISION_ORDER])
    fig.suptitle(
        "Does the frozen policy decision survive deleting one animal system?",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    axis.set_title(
        "Frozen primary analysis: 4,000 bootstrap replicates per deletion with the unchanged two-gate rule",
        fontsize=11,
        pad=14,
    )
    fig.subplots_adjust(left=0.25, right=0.91, top=0.83, bottom=0.17)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_override_intervals(
    full_decisions: pd.DataFrame,
    loo_decisions: pd.DataFrame,
    families: list[str],
    path: Path,
) -> None:
    overrides = full_decisions.loc[
        full_decisions["decision"].eq("override_with_detected_case_contacts")
    ].sort_values(CELL_KEYS, kind="stable")
    fig, axes = plt.subplots(len(overrides), 2, figsize=(13.0, 7.0), squeeze=False)
    metrics = [
        ("direct_minus_history", "direct_minus_history_ci_low", "direct_minus_history_ci_high", "Direct minus history"),
        ("direct_absolute", "direct_absolute_ci_low", "direct_absolute_ci_high", "Direct minus no-extra"),
    ]
    for row_index, full in enumerate(overrides.itertuples(index=False)):
        selected = loo_decisions.copy()
        for key in CELL_KEYS:
            selected = selected.loc[selected[key].eq(getattr(full, key))]
        selected = selected.set_index("omitted_family").loc[families].reset_index()
        positions = np.arange(len(families))[::-1]
        for column_index, (mean_col, low_col, high_col, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            values = 100 * selected[mean_col].to_numpy(float)
            lows = 100 * selected[low_col].to_numpy(float)
            highs = 100 * selected[high_col].to_numpy(float)
            axis.errorbar(
                values,
                positions,
                xerr=np.vstack([values - lows, highs - values]),
                fmt="o",
                color="#4C78A8" if column_index == 0 else "#F58518",
                capsize=3,
            )
            axis.axvline(0, color="#555555", linestyle="--", linewidth=1.1)
            axis.set_yticks(
                positions,
                [f"Without {FAMILY_LABELS.get(item, item)}" for item in families]
                if column_index == 0
                else [],
            )
            axis.grid(axis="x", alpha=0.25)
            axis.set_title(title, fontsize=11)
            if row_index == len(overrides) - 1:
                axis.set_xlabel("Avoided attack-rate difference (percentage points)")
        axes[row_index, 0].set_ylabel(_cell_label(full), labelpad=10)
    fig.suptitle(
        "Numerical convergence of leave-one-system-out uncertainty",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Original override cells only; 20,000 bootstrap replicates per deletion",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.28, right=0.98, top=0.87, bottom=0.12, hspace=0.45, wspace=0.16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    source_root = Path(config["source"]["results_path"])
    source_audit_path = source_root / str(config["source"]["audit"])
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("status") != "pass":
        raise ValueError("source experiment audit must pass")

    relative_path = source_root / str(config["source"]["relative_worlds"])
    policy_path = source_root / str(config["source"]["policy_worlds"])
    decision_path = source_root / str(config["source"]["decision_map"])
    relative_worlds = pd.read_csv(relative_path)
    policy_worlds = pd.read_csv(policy_path)
    full_decisions = pd.read_csv(decision_path)
    families = sorted(relative_worlds["system_family"].unique())
    profile = config["profiles"][profile_name]
    maximum = profile.get("max_omitted_families")
    selected_families = families if maximum is None else families[: int(maximum)]
    repetitions = int(profile["bootstrap_replicates"])
    seed = int(config["evaluation"]["seed"])
    method = str(config["evaluation"]["primary_method"])
    baseline = str(config["evaluation"]["primary_baseline"])

    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    relative_parts = []
    absolute_parts = []
    decision_parts = []
    for index, omitted in enumerate(selected_families):
        relative_subset = relative_worlds.loc[~relative_worlds["system_family"].eq(omitted)]
        policy_subset = policy_worlds.loc[~policy_worlds["system_family"].eq(omitted)]
        relative_summary, _ = _hierarchical_summary(
            relative_subset,
            value_column="increment",
            group_columns=CELL_KEYS,
            bootstrap_replicates=repetitions,
            seed=seed + 100 * index,
        )
        absolute_summary, _ = _hierarchical_summary(
            policy_subset,
            value_column="attack_rate_reduction",
            group_columns=CELL_KEYS + ["method"],
            bootstrap_replicates=repetitions,
            seed=seed + 100 * index + 1,
        )
        decisions = _decision_map(
            relative_summary,
            absolute_summary,
            method=method,
            baseline=baseline,
        )
        for frame in (relative_summary, absolute_summary, decisions):
            frame.insert(0, "omitted_family", omitted)
        relative_parts.append(relative_summary)
        absolute_parts.append(absolute_summary)
        decision_parts.append(decisions)

    relative_loo = pd.concat(relative_parts, ignore_index=True)
    absolute_loo = pd.concat(absolute_parts, ignore_index=True)
    decisions_loo = pd.concat(decision_parts, ignore_index=True)
    resilience = _summarize_resilience(
        full_decisions, decisions_loo, len(selected_families)
    )
    original_overrides = resilience.loc[
        resilience["full_decision"].eq("override_with_detected_case_contacts")
    ]
    strong_pass = bool(
        len(selected_families) == len(families)
        and not original_overrides.empty
        and original_overrides["override_folds"].eq(len(families)).all()
    )
    partial_pass = bool(
        len(selected_families) == len(families)
        and not original_overrides.empty
        and original_overrides["override_folds"].ge(len(families) - 1).all()
    )

    precision_decisions = pd.DataFrame()
    precision_resilience = pd.DataFrame()
    precision_repetitions = profile.get("numerical_convergence_bootstrap_replicates")
    if precision_repetitions is not None and len(selected_families) == len(families):
        precision_parts = []
        original_cells = full_decisions.loc[
            full_decisions["decision"].eq("override_with_detected_case_contacts")
        ]
        precision_seed = int(config["evaluation"]["numerical_convergence_seed"])
        for family_index, omitted in enumerate(selected_families):
            for cell_index, cell in enumerate(original_cells.itertuples(index=False)):
                relative_subset = relative_worlds.loc[
                    ~relative_worlds["system_family"].eq(omitted)
                ].copy()
                policy_subset = policy_worlds.loc[
                    ~policy_worlds["system_family"].eq(omitted)
                ].copy()
                for key in CELL_KEYS:
                    relative_subset = relative_subset.loc[
                        relative_subset[key].eq(getattr(cell, key))
                    ]
                    policy_subset = policy_subset.loc[
                        policy_subset[key].eq(getattr(cell, key))
                    ]
                relative_precision, _ = _hierarchical_summary(
                    relative_subset,
                    value_column="increment",
                    group_columns=CELL_KEYS,
                    bootstrap_replicates=int(precision_repetitions),
                    seed=precision_seed + 100 * family_index + 2 * cell_index,
                )
                absolute_precision, _ = _hierarchical_summary(
                    policy_subset,
                    value_column="attack_rate_reduction",
                    group_columns=CELL_KEYS + ["method"],
                    bootstrap_replicates=int(precision_repetitions),
                    seed=precision_seed + 100 * family_index + 2 * cell_index + 1,
                )
                decision_precision = _decision_map(
                    relative_precision,
                    absolute_precision,
                    method=method,
                    baseline=baseline,
                )
                decision_precision.insert(0, "omitted_family", omitted)
                precision_parts.append(decision_precision)
        precision_decisions = pd.concat(precision_parts, ignore_index=True)
        precision_resilience = _summarize_resilience(
            full_decisions.loc[
                full_decisions["decision"].eq("override_with_detected_case_contacts")
            ],
            precision_decisions,
            len(selected_families),
        )
    gate_resilience = precision_resilience if not precision_resilience.empty else original_overrides
    convergence_strong_pass = bool(
        not gate_resilience.empty
        and gate_resilience["override_folds"].eq(len(selected_families)).all()
    )
    convergence_partial_pass = bool(
        not gate_resilience.empty
        and gate_resilience["override_folds"].ge(max(len(selected_families) - 1, 1)).all()
    )

    fold_counts = decisions_loo.groupby("omitted_family", observed=True).size()
    retained_counts = relative_worlds.loc[
        relative_worlds["system_family"].isin(selected_families)
    ].groupby("system_family", observed=True).size()
    checks = {
        "source_audit_passed": source_audit.get("status") == "pass",
        "source_has_five_independent_families": len(families) == 5,
        "source_decision_map_has_eight_cells": len(full_decisions) == 8,
        "every_deletion_has_eight_cells": bool(fold_counts.eq(8).all()),
        "every_deletion_retains_n_minus_one_families": bool(
            relative_loo["families"].eq(len(families) - 1).all()
            and absolute_loo["families"].eq(len(families) - 1).all()
        ),
        "all_selected_families_evaluated": set(decisions_loo["omitted_family"].unique())
        == set(selected_families),
        "source_families_have_worlds": bool(retained_counts.gt(0).all()),
        "decision_values_valid": set(decisions_loo["decision"].unique()).issubset(
            set(DECISION_ORDER)
        ),
        "finite_effects": bool(
            np.isfinite(
                decisions_loo[
                    [
                        "direct_minus_history",
                        "direct_minus_history_ci_low",
                        "direct_minus_history_ci_high",
                        "direct_absolute",
                        "direct_absolute_ci_low",
                        "direct_absolute_ci_high",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "intervals_ordered": bool(
            decisions_loo["direct_minus_history_ci_low"].le(
                decisions_loo["direct_minus_history_ci_high"]
            ).all()
            and decisions_loo["direct_absolute_ci_low"].le(
                decisions_loo["direct_absolute_ci_high"]
            ).all()
        ),
        "resilience_rows_complete": len(resilience) == 8,
        "numerical_convergence_preserves_gate_failure": bool(
            precision_resilience.empty
            or (not convergence_strong_pass and not convergence_partial_pass)
        ),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_experiment": config["source"]["experiment_id"],
        "source_policy_evaluations": len(policy_worlds),
        "independent_families": len(families),
        "evaluated_deletions": len(selected_families),
        "decision_cells_per_deletion": 8,
        "original_override_cells": len(original_overrides),
        "strong_transportability_gate": "pass" if strong_pass else "fail",
        "partial_transportability_gate": "pass" if partial_pass else "fail",
        "numerical_convergence_replicates": (
            int(precision_repetitions) if precision_repetitions is not None else None
        ),
        "converged_strong_transportability_gate": (
            "pass" if convergence_strong_pass else "fail"
        ),
        "converged_partial_transportability_gate": (
            "pass" if convergence_partial_pass else "fail"
        ),
        "interpretation": "sensitivity_analysis_not_a_new_pass_opportunity",
    }
    if audit["status"] != "pass":
        raise ValueError(f"leave-one-family-out audit failed: {audit}")

    relative_loo.to_csv(results_dir / "loo_relative_summary.csv", index=False)
    absolute_loo.to_csv(results_dir / "loo_absolute_summary.csv", index=False)
    decisions_loo.to_csv(results_dir / "loo_decision_map.csv", index=False)
    resilience.to_csv(results_dir / "decision_resilience.csv", index=False)
    if not precision_decisions.empty:
        precision_decisions.to_csv(
            results_dir / "precision_loo_decision_map.csv", index=False
        )
        precision_resilience.to_csv(
            results_dir / "precision_decision_resilience.csv", index=False
        )
    pd.DataFrame(
        [
            {"artifact": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (source_audit_path, relative_path, policy_path, decision_path)
        ]
    ).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
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
        "audit_status": audit["status"],
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if len(selected_families) == len(families):
        _plot_decision_deletions(
            full_decisions, decisions_loo, families, report_dir / "decision_deletion_matrix.png"
        )
        _plot_override_intervals(
            full_decisions,
            precision_decisions if not precision_decisions.empty else decisions_loo,
            families,
            report_dir / "override_loo_intervals.png",
        )
    display = resilience.copy()
    for column in (
        "relative_ci_low_min",
        "relative_ci_low_max",
        "absolute_ci_low_min",
        "absolute_ci_low_max",
    ):
        display[column] = 100 * display[column]
    readme = f"""# Leave-one-animal-system-family-out policy stress test

This frozen post-simulation analysis deletes one complete independent animal-system family at a time, reruns the hierarchical bootstrap, and reapplies the unchanged relative-plus-absolute decision rule. It does not generate new epidemic worlds. Any newly appearing override in a deletion fold is recorded only as decision instability and cannot count as a new positive claim.

- Source experiment: {audit['source_experiment']}
- Source policy evaluations: {audit['source_policy_evaluations']}
- Independent families: {audit['independent_families']}
- Original override cells: {audit['original_override_cells']}
- Technical audit: **{audit['status']}**
- Strong transportability gate: **{audit['strong_transportability_gate']}**
- Partial transportability gate: **{audit['partial_transportability_gate']}**
- Numerical-convergence replicates: {audit['numerical_convergence_replicates']}
- Converged strong gate: **{audit['converged_strong_transportability_gate']}**
- Converged partial gate: **{audit['converged_partial_transportability_gate']}**

Strong requires every original override to remain an override in all five deletion folds. Partial requires every original override to remain an override in at least four of five folds. With four families remaining, the unchanged 80% direction requirement becomes four of four positive family means.

The 4,000-replicate deletion matrix is the frozen primary analysis. Because several interval bounds were close to zero, the original override cells alone were recomputed with 20,000 bootstrap replicates as a numerical-convergence audit. This audit may make the interpretation more conservative but cannot create a new positive claim.

## Decision resilience

{_markdown_table(display)}
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leave-one-animal-system-family-out policy stress test"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260817-007_policy_leave_one_family_out.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
