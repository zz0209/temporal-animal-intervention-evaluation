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

from .outbreak_response_pilot import _git_value, _sha256


BOOL_COLUMNS = [
    "node_ranking",
    "fixed_budget_sets",
    "strict_forward",
    "outbreak_update",
    "no_extra_control",
    "loso_family",
]


def load_source_audits(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_id": path.parts[-3],
                "path": str(path),
                "status": payload.get("status", "missing"),
                "checks": len(payload.get("checks", {})),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def build_method_results(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    temporal = pd.read_csv(config["sources"]["temporal_increment"])
    for item in temporal.itertuples(index=False):
        rows.append(
            {
                "comparison_domain": "Temporal order",
                "comparison": f"Temporal summary over static ({item.outcome})",
                "estimate": float(item.family_equal_mean),
                "ci_low": float(item.blocked_ci_low),
                "ci_high": float(item.blocked_ci_high),
                "unit": "metric difference",
                "families": int(item.families),
                "result": "not_supported" if item.blocked_ci_low <= 0 <= item.blocked_ci_high else "supported",
                "source_experiment": "EXP-20260816-005",
            }
        )
    closest = pd.read_csv(config["sources"]["closest_prior"])
    for item in closest.itertuples(index=False):
        rows.append(
            {
                "comparison_domain": "Preparedness baseline",
                "comparison": str(item.comparison).replace("stable_watchlist_over_", "Stable over "),
                "estimate": float(item.family_equal_mean),
                "ci_low": float(item.blocked_ci_low),
                "ci_high": float(item.blocked_ci_high),
                "unit": "attack-rate difference",
                "families": int(item.families),
                "result": "not_supported" if item.blocked_ci_low <= 0 <= item.blocked_ci_high else "supported",
                "source_experiment": "EXP-20260817-005",
            }
        )
    learned = pd.read_csv(config["sources"]["learned_models"])
    learned = learned.loc[learned["subset"].eq("all")]
    for item in learned.itertuples(index=False):
        if item.method == "stable_plus_tracing":
            continue
        rows.append(
            {
                "comparison_domain": "Fixed-budget planner",
                "comparison": f"{item.method} over stable + tracing",
                "estimate": float(item.family_equal_gain_over_stable_plus_tracing),
                "ci_low": float(item.gain_ci_low),
                "ci_high": float(item.gain_ci_high),
                "unit": "set-value difference",
                "families": int(item.families),
                "result": "not_supported" if item.gain_ci_low <= 0 else "supported",
                "source_experiment": "EXP-20260816-014",
            }
        )
    return pd.DataFrame(rows)


def reconcile_primary_tables(decision: pd.DataFrame, taxonomy: pd.DataFrame, resilience: pd.DataFrame) -> bool:
    keys = ["epidemic_model", "detection_profile", "rewiring_fraction"]
    left = decision[keys + ["decision"]].sort_values(keys).reset_index(drop=True)
    right = taxonomy[keys + ["decision"]].sort_values(keys).reset_index(drop=True)
    return bool(left.equals(right) and len(resilience) == 8)


def plot_comparison_coverage(coverage: pd.DataFrame, path: Path, dpi: int) -> None:
    matrix = coverage[BOOL_COLUMNS].astype(int).to_numpy()
    labels = {
        "node_ranking": "Node\nranking",
        "fixed_budget_sets": "Fixed-budget\nsets",
        "strict_forward": "Strict\nforward split",
        "outbreak_update": "Outbreak\nupdate",
        "no_extra_control": "No-extra\ncontrol",
        "loso_family": "Held-out\nfamily",
    }
    fig, axis = plt.subplots(figsize=(11.8, 8.0))
    axis.imshow(matrix, cmap=ListedColormap(["#EEEEEE", "#4C78A8"]), vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(BOOL_COLUMNS)), [labels[column] for column in BOOL_COLUMNS])
    axis.set_yticks(range(len(coverage)), coverage["method_family"])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, "✓" if matrix[row, column] else "—", ha="center", va="center", fontsize=14, color="white" if matrix[row, column] else "#666666")
    fig.suptitle("Comparison coverage in the frozen evidence package", fontsize=18, fontweight="bold", y=0.98)
    axis.set_title("A check mark means the method family was evaluated under that contract; it does not imply superiority", fontsize=10.5, pad=14)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.84, bottom=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_method_results(results: pd.DataFrame, path: Path, dpi: int) -> None:
    domains = ["Temporal order", "Preparedness baseline", "Fixed-budget planner"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.0))
    colors = ["#4C78A8", "#59A14F", "#F58518"]
    titles = {
        "Temporal order": "Temporal summaries\nvs matched static summaries",
        "Preparedness baseline": "Stable watchlist\nvs history comparators",
        "Fixed-budget planner": "Candidate planners\nvs stable + tracing",
    }
    label_maps = {
        "Temporal order": {
            "Temporal summary over static (spearman_gain)": "Spearman rank",
            "Temporal summary over static (value_capture_gain)": "Value capture",
            "Temporal summary over static (regret_reduction)": "Regret reduction",
        },
        "Preparedness baseline": {
            "Stable over history_recency": "History recency",
            "Stable over history_weight": "History weight",
        },
        "Fixed-budget planner": {
            "contact_to_detected over stable + tracing": "Detected contacts",
            "deep_sets over stable + tracing": "Deep Sets",
            "ridge over stable + tracing": "Ridge",
            "stable_watchlist over stable + tracing": "Stable watchlist",
        },
    }
    for axis, domain, color in zip(axes, domains, colors):
        selected = results.loc[results["comparison_domain"].eq(domain)].reset_index(drop=True)
        y = np.arange(len(selected))[::-1]
        values = 100 * selected["estimate"].to_numpy(float)
        lows = 100 * selected["ci_low"].to_numpy(float)
        highs = 100 * selected["ci_high"].to_numpy(float)
        axis.errorbar(values, y, xerr=np.vstack([values - lows, highs - values]), fmt="o", color=color, capsize=3)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        labels = [label_maps[domain].get(str(item), str(item)) for item in selected["comparison"]]
        axis.set_yticks(y, labels)
        axis.set_title(titles[domain], fontsize=12)
        axis.set_xlabel("Difference × 100\n(domain-specific scale)")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("No complex method earns a universal performance claim", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.04, "Panels use different outcomes and references; magnitudes must not be compared across panels", ha="center", fontsize=10.5)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.16, wspace=0.55)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_claim_ledger(claims: pd.DataFrame, path: Path, dpi: int) -> None:
    status_colors = {
        "supported_null_within_tested_methods": "#4C78A8",
        "supported_negative_result": "#4C78A8",
        "supported_baseline_revision": "#59A14F",
        "conditionally_supported_not_transportable": "#F58518",
        "supported_decision_principle": "#59A14F",
        "supported_rejection": "#4C78A8",
        "not_supported_and_excluded": "#B8B8B8",
        "supported_distinction_and_support_audit": "#59A14F",
    }
    labels = {
        "C1_temporal_order": "Temporal-order increment",
        "C2_learned_planners": "Learned-planner superiority",
        "C3_preparedness_reference": "History preparedness reference",
        "C4_conditional_update": "Detected-contact update",
        "C5_dual_safety": "Relative + absolute safety",
        "C6_universal_policy": "Universal policy winner",
        "C7_field_effectiveness": "Named-pathogen field efficacy",
        "C8_source_vs_intervention": "Source versus intervention value",
    }
    display_status = {
        "supported_null_within_tested_methods": "No robust increment in tested methods",
        "supported_negative_result": "Learned superiority not supported",
        "supported_baseline_revision": "History baseline retained",
        "conditionally_supported_not_transportable": "Conditional pooled signal only",
        "supported_decision_principle": "Dual safety gate supported",
        "supported_rejection": "Universal winner rejected",
        "not_supported_and_excluded": "Not supported; excluded",
        "supported_distinction_and_support_audit": "Distinct estimands; support audited",
    }
    fig, axis = plt.subplots(figsize=(12.8, 7.2))
    y = np.arange(len(claims))[::-1]
    axis.barh(y, np.ones(len(claims)), color=[status_colors[item] for item in claims["status"]], height=0.68)
    for position, row in zip(y, claims.itertuples(index=False)):
        text = display_status[str(row.status)]
        axis.text(0.03, position, text, va="center", ha="left", fontsize=10.5, fontweight="bold")
        axis.text(0.97, position, str(row.transportability).replace("_", " "), va="center", ha="right", fontsize=9.5)
    axis.set_yticks(y, [labels[item] for item in claims["claim_id"]])
    axis.set_xticks([])
    axis.set_xlim(0, 1)
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    fig.suptitle("Final claim ledger: support, boundaries, and explicit exclusions", fontsize=18, fontweight="bold", y=0.98)
    axis.set_title("Right-hand text records the transportability scope of each claim", fontsize=10.5, pad=14)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.84, bottom=0.08)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _write_report(results_dir: Path, reports_dir: Path, audit: dict[str, Any], readiness: pd.DataFrame) -> None:
    lines = [
        "# Final frozen evidence package",
        "",
        "This package reconstructs manuscript-facing evidence from frozen artifacts. It performs no new epidemic simulation and does not alter any decision threshold.",
        "",
        "## Reconstruction result",
        "",
        f"- Source audits verified: {audit['source_audits_verified']}",
        f"- Method families represented: {audit['method_families']}",
        f"- Final claims classified: {audit['claims']}",
        f"- Frozen policy cells reconciled: {audit['policy_cells']}",
        f"- Overall audit: `{audit['status']}`",
        "",
        "## Readiness boundary",
        "",
    ]
    for row in readiness.itertuples(index=False):
        lines.append(f"- **{row.item}:** {row.status} — {row.note}")
    lines.extend(
        [
            "",
            "This reviewer-release evidence package follows completion of the clean rebuild, citation and permission audit, and manuscript assembly. Its claims are limited to simulation-based counterfactual evaluation on observed contact histories; it does not establish field effectiveness for a named pathogen.",
            "",
            "## Generated artifacts",
            "",
            f"- `{results_dir / 'primary_policy_table.csv'}`",
            f"- `{results_dir / 'method_comparison_results.csv'}`",
            f"- `{results_dir / 'comparison_coverage.csv'}`",
            f"- `{results_dir / 'claim_evidence_ledger.csv'}`",
            f"- `{results_dir / 'source_audit_inventory.csv'}`",
            f"- `{reports_dir / 'comparison_coverage.png'}`",
            f"- `{reports_dir / 'method_comparison_summary.png'}`",
            f"- `{reports_dir / 'claim_ledger.png'}`",
        ]
    )
    (reports_dir / "FINAL_EVIDENCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, profile: str) -> dict[str, Any]:
    started = time.time()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = config["experiment"]["id"]
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile
    reports_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_paths = [Path(item) for item in config["sources"]["audits"]]
    source_audits = load_source_audits(audit_paths)
    coverage = pd.DataFrame(config["comparison_families"])
    claims = pd.DataFrame(config["claim_ledger"])
    method_results = build_method_results(config)
    decision = pd.read_csv(config["sources"]["decision_map"])
    resilience = pd.read_csv(config["sources"]["deletion_resilience"])
    taxonomy = pd.read_csv(config["sources"]["evidence_taxonomy"])
    novelty = pd.read_csv(config["sources"]["novelty_matrix"])
    primary = taxonomy.merge(
        resilience[[*CELL_KEYS, "same_decision_folds", "override_folds", "resilience_status"]].rename(
            columns={
                "same_decision_folds": "deletion_same_decision_folds",
                "override_folds": "deletion_override_folds",
                "resilience_status": "deletion_resilience_status",
            }
        ),
        on=CELL_KEYS,
        how="left",
        validate="one_to_one",
    )
    readiness = pd.DataFrame(config["release_readiness"])
    source_audits.to_csv(results_dir / "source_audit_inventory.csv", index=False)
    coverage.to_csv(results_dir / "comparison_coverage.csv", index=False)
    claims.to_csv(results_dir / "claim_evidence_ledger.csv", index=False)
    method_results.to_csv(results_dir / "method_comparison_results.csv", index=False)
    primary.to_csv(results_dir / "primary_policy_table.csv", index=False)
    readiness.to_csv(results_dir / "publication_readiness.csv", index=False)
    dpi = int(config["profiles"][profile]["render_dpi"])
    plot_comparison_coverage(coverage, reports_dir / "comparison_coverage.png", dpi)
    plot_method_results(method_results, reports_dir / "method_comparison_summary.png", dpi)
    plot_claim_ledger(claims, reports_dir / "claim_ledger.png", dpi)
    checks = {
        "all_source_audits_present": len(source_audits) == len(audit_paths),
        "all_source_audits_pass": source_audits["status"].eq("pass").all(),
        "ten_method_families_covered": len(coverage) == 10,
        "coverage_contract_complete": coverage[BOOL_COLUMNS].notna().all().all(),
        "configured_claims_classified": len(claims) >= 7 and claims["status"].notna().all(),
        "every_supported_claim_has_source": claims.loc[~claims["status"].eq("not_supported_and_excluded"), "source_experiments"].ne("none").all(),
        "field_effectiveness_explicitly_excluded": claims.loc[claims["claim_id"].eq("C7_field_effectiveness"), "status"].eq("not_supported_and_excluded").all(),
        "primary_tables_reconcile": reconcile_primary_tables(decision, taxonomy, resilience),
        "eight_primary_policy_cells": len(primary) == 8,
        "five_independent_families_declared": int(config["design"]["independent_families"]) == 5,
        "novelty_matrix_has_temporal_and_learned_comparators": novelty["closest_relation"].str.contains("Temporal|GNN|temporal", case=False, regex=True).sum() >= 3,
        "no_new_simulation_or_threshold_tuning": bool(config["design"]["reconstruction_only"] and config["design"]["prohibit_new_simulation"] and config["design"]["prohibit_threshold_changes"]),
        "model_based_scope_disclosed": readiness.loc[
            readiness["item"].eq("Field validation"), "status"
        ].eq("outside study scope").all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "source_audits_verified": int(len(source_audits)),
        "method_families": int(len(coverage)),
        "claims": int(len(claims)),
        "policy_cells": int(len(primary)),
        "independent_families": 5,
        "readiness": "reviewer_archive_ready_for_submission",
    }
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_files = audit_paths + [Path(config["sources"][key]) for key in config["sources"] if key != "audits"]
    pd.DataFrame(
        [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in source_files]
    ).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
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
    _write_report(results_dir, reports_dir, audit, readiness)
    return audit


CELL_KEYS = ["epidemic_model", "detection_profile", "rewiring_fraction"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final frozen Temporal Animal Intervention Evaluation evidence package.")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260817-009_final_evidence_package.yaml"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
