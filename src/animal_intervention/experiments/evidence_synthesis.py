from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from .outbreak_response_pilot import _git_value, _sha256


SUMMARY_FILES = {
    "absolute": ("absolute_policy_summary.csv", "absolute_family_summary.csv"),
    "relative": ("relative_policy_summary.csv", "relative_family_summary.csv"),
}


def _apply_filters(frame: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    selected = frame.copy()
    for column, expected in filters.items():
        if column not in selected:
            raise ValueError(f"filter column is missing: {column}")
        if isinstance(expected, list):
            selected = selected.loc[selected[column].isin(expected)]
        else:
            selected = selected.loc[selected[column].eq(expected)]
    return selected


def _effect_status(mean: float, low: float, high: float) -> str:
    if low > 0:
        return "supported_benefit"
    if high < 0:
        return "supported_harm"
    if mean > 0:
        return "possible_benefit"
    if mean < 0:
        return "possible_harm"
    return "null_or_unresolved"


def _evaluate_claim(claim: dict[str, Any], ledger: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = ledger.loc[
        ledger["experiment_key"].eq(claim["source"])
        & ledger["estimand"].eq(claim["estimand"])
        & ledger["method"].eq(claim["method"])
    ]
    selected = _apply_filters(selected, claim.get("filters", {})).copy()
    if selected.empty:
        raise ValueError(f"claim has no matching evidence cells: {claim['claim_id']}")
    gate = claim["gate"]
    selected["point_gate"] = selected["family_equal_mean"].gt(0)
    selected["interval_gate"] = selected["ci_low"].gt(0)
    selected["family_direction_gate"] = selected["positive_families"].ge(
        int(gate.get("minimum_positive_families", 0))
    )
    requirements = []
    if gate.get("require_point_positive", False):
        requirements.append("point_gate")
    if gate.get("require_ci_low_positive", False):
        requirements.append("interval_gate")
    if int(gate.get("minimum_positive_families", 0)) > 0:
        requirements.append("family_direction_gate")
    selected["cell_passes"] = selected[requirements].all(axis=1) if requirements else True
    passing = int(selected["cell_passes"].sum())
    required_fraction = float(gate["minimum_passing_cell_fraction"])
    passed = passing / len(selected) + 1e-12 >= required_fraction
    result = {
        "claim_id": claim["claim_id"],
        "claim_text": claim["claim_text"],
        "source_experiment": selected["experiment_id"].iloc[0],
        "domain": selected["domain"].iloc[0],
        "estimand": claim["estimand"],
        "method": claim["method"],
        "cells": len(selected),
        "point_positive_cells": int(selected["point_gate"].sum()),
        "interval_supported_cells": int(selected["interval_gate"].sum()),
        "family_direction_cells": int(selected["family_direction_gate"].sum()),
        "minimum_positive_families": int(gate.get("minimum_positive_families", 0)),
        "passing_cells": passing,
        "required_passing_fraction": required_fraction,
        "gate_passed": bool(passed),
        "publication_status": "supported_within_scope" if passed else "not_supported",
        "scope": claim["scope"],
    }
    selected["claim_id"] = claim["claim_id"]
    return result, selected


def _load_evidence(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    policy_parts = []
    family_parts = []
    audits: dict[str, dict[str, Any]] = {}
    for key, specification in config["sources"].items():
        root = Path(specification["results_path"])
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
        audits[key] = audit
        for estimand, (policy_file, family_file) in SUMMARY_FILES.items():
            policy = pd.read_csv(root / policy_file)
            family = pd.read_csv(root / family_file)
            for frame in (policy, family):
                frame.insert(0, "domain", specification["domain"])
                frame.insert(0, "experiment_id", specification["experiment_id"])
                frame.insert(0, "experiment_key", key)
                frame.insert(3, "estimand", estimand)
            policy_parts.append(policy)
            family_parts.append(family)
    policy_ledger = pd.concat(policy_parts, ignore_index=True, sort=False)
    family_ledger = pd.concat(family_parts, ignore_index=True, sort=False)
    policy_ledger["effect_status"] = [
        _effect_status(mean, low, high)
        for mean, low, high in policy_ledger[
            ["family_equal_mean", "ci_low", "ci_high"]
        ].itertuples(index=False, name=None)
    ]
    return policy_ledger, family_ledger, audits


def _safety_frontier(
    policy_ledger: pd.DataFrame, source: str, method: str
) -> pd.DataFrame:
    source_rows = policy_ledger.loc[
        policy_ledger["experiment_key"].eq(source)
        & policy_ledger["method"].eq(method)
    ].copy()
    absolute = source_rows.loc[source_rows["estimand"].eq("absolute")]
    relative = source_rows.loc[source_rows["estimand"].eq("relative")]
    dimensions = [
        column
        for column in (
            "disease_regime",
            "detection_profile",
            "action_delay_fraction",
            "residual_contact_multiplier",
            "secondary_case_sensitivity",
            "false_positive_rate",
            "rewiring_fraction",
            "rewiring_mode",
        )
        if column in absolute and column in relative
    ]
    metrics = ["family_equal_mean", "ci_low", "ci_high", "positive_families"]
    merged = relative[dimensions + metrics].merge(
        absolute[dimensions + metrics],
        on=dimensions,
        suffixes=("_relative", "_absolute"),
        validate="one_to_one",
    )
    merged["relative_status"] = [
        _effect_status(mean, low, high)
        for mean, low, high in merged[
            ["family_equal_mean_relative", "ci_low_relative", "ci_high_relative"]
        ].itertuples(index=False, name=None)
    ]
    merged["absolute_status"] = [
        _effect_status(mean, low, high)
        for mean, low, high in merged[
            ["family_equal_mean_absolute", "ci_low_absolute", "ci_high_absolute"]
        ].itertuples(index=False, name=None)
    ]
    merged["decision"] = np.where(
        merged["ci_low_absolute"].gt(0) & merged["ci_low_relative"].gt(0),
        "direct_targeting_supported",
        np.where(
            merged["ci_low_absolute"].gt(0),
            "additional_targeting_supported_policy_unresolved",
            "abstain_or_require_external_calibration",
        ),
    )
    return merged


def _modality_summary(
    family_ledger: pd.DataFrame,
    *,
    source: str,
    method: str,
    filters: dict[str, Any],
    modality_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = family_ledger.loc[
        family_ledger["experiment_key"].eq(source)
        & family_ledger["estimand"].eq("relative")
        & family_ledger["method"].eq(method)
    ].copy()
    selected = _apply_filters(selected, filters)
    selected["modality"] = selected["system_family"].map(modality_map)
    if selected["modality"].isna().any():
        missing = sorted(selected.loc[selected["modality"].isna(), "system_family"].unique())
        raise ValueError(f"system families lack modality mapping: {missing}")
    summary = (
        selected.groupby(["disease_regime", "modality"], observed=True)["mean_value"]
        .agg(modality_equal_mean="mean", families="size")
        .reset_index()
    )
    return selected, summary


def _plot_claim_map(claims: pd.DataFrame, path: Path) -> None:
    columns = [
        ("point_positive_cells", "Point estimate"),
        ("interval_supported_cells", "Interval support"),
        ("family_direction_cells", "Family direction"),
        ("passing_cells", "Final cell gate"),
    ]
    matrix = np.column_stack(
        [claims[column].to_numpy(float) / claims["cells"].to_numpy(float) for column, _ in columns]
    )
    family_not_required = claims["minimum_positive_families"].eq(0).to_numpy()
    matrix[family_not_required, 2] = np.nan
    fig, axis = plt.subplots(figsize=(12.5, 6.2))
    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#DDDDDD")
    image = axis.imshow(matrix, cmap=color_map, vmin=0, vmax=1, aspect="auto")
    for row in range(len(claims)):
        for column, (field, _) in enumerate(columns):
            label = (
                "N/A"
                if column == 2 and family_not_required[row]
                else f"{int(claims.iloc[row][field])}/{int(claims.iloc[row]['cells'])}"
            )
            axis.text(column, row, label, ha="center", va="center", fontsize=11)
    axis.set_xticks(range(len(columns)), [label for _, label in columns])
    axis.set_yticks(range(len(claims)), claims["claim_id"])
    axis.set_title("Frozen claim-evidence map", fontsize=20, fontweight="bold", pad=20)
    axis.set_xlabel("Evidence component; green means a larger fraction of prespecified cells passed")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Fraction of cells")
    fig.subplots_adjust(left=0.28, right=0.93, top=0.86, bottom=0.16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_safety_frontier(frame: pd.DataFrame, path: Path) -> None:
    colors = {"low": "#4C78A8", "middle": "#F28E2B", "high": "#B23A48"}
    profiles = ["early_detection", "delayed_detection"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), sharex=True, sharey=True)
    for axis, profile in zip(axes, profiles):
        subset = frame.loc[frame["detection_profile"].eq(profile)].sort_values(
            ["disease_regime", "rewiring_fraction"]
        )
        axis.axhline(0, color="#555555", linewidth=1)
        axis.axvline(0, color="#555555", linewidth=1)
        for row in subset.itertuples(index=False):
            x = 100 * row.family_equal_mean_relative
            y = 100 * row.family_equal_mean_absolute
            axis.errorbar(
                x,
                y,
                xerr=[[100 * (row.family_equal_mean_relative - row.ci_low_relative)], [100 * (row.ci_high_relative - row.family_equal_mean_relative)]],
                yerr=[[100 * (row.family_equal_mean_absolute - row.ci_low_absolute)], [100 * (row.ci_high_absolute - row.family_equal_mean_absolute)]],
                fmt="o" if row.rewiring_fraction == 0 else "s",
                color=colors[row.disease_regime],
                markerfacecolor="white" if row.rewiring_fraction == 0 else colors[row.disease_regime],
                capsize=3,
                markersize=7,
                alpha=0.9,
            )
            if row.rewiring_fraction == 1.0:
                label_offsets = {
                    "low": (6, -16),
                    "middle": (6, 10),
                    "high": (6, -2),
                }
                axis.annotate(
                    row.disease_regime.title(),
                    (x, y),
                    xytext=label_offsets[row.disease_regime],
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_title(profile.replace("_", " ").title())
        axis.set_xlabel("Direct over stable (attack-rate percentage points)")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Direct over detected-case isolation alone (percentage points)")
    legend = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="black", markerfacecolor="white", label="No rewiring"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor="black", markerfacecolor="#777777", label="Full rewiring"),
        *[
            Line2D([0], [0], marker="o", color="none", markeredgecolor=color, markerfacecolor=color, label=regime.title())
            for regime, color in colors.items()
        ],
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Relative policy advantage is not absolute intervention safety", fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, "Upper-right favors direct targeting and shows positive absolute benefit; lower-right favors direct only because stable is worse", ha="center", color="#555555")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.18, wspace=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_modality_boundary(family: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    regimes = ["low", "middle", "high"]
    positions = np.arange(len(regimes))
    colors = {"association_or_feeder": "#4C78A8", "proximity_sensor": "#E45756"}
    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    for offset, (modality, group) in zip((-0.08, 0.08), summary.groupby("modality", observed=True, sort=True)):
        ordered = group.set_index("disease_regime").reindex(regimes)
        axis.plot(positions + offset, 100 * ordered["modality_equal_mean"], marker="D", linewidth=2.2, color=colors[modality], label=f"{modality.replace('_', ' ').title()} mean")
        raw = family.loc[family["modality"].eq(modality)]
        for regime_index, regime in enumerate(regimes):
            values = 100 * raw.loc[raw["disease_regime"].eq(regime), "mean_value"].to_numpy(float)
            jitter = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(regime_index + offset + jitter, values, color=colors[modality], alpha=0.35, s=28)
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(positions, [item.title() for item in regimes])
    axis.set_ylabel("Direct-over-stable increment (attack-rate percentage points)")
    axis.set_xlabel("Within-dataset calibrated disease regime")
    axis.set_title("Primary policy boundary by observation modality", fontsize=20, fontweight="bold", pad=18)
    axis.text(0.5, 1.01, "Early detection with full compensatory rewiring; diamonds are modality means and faint points are independent families", transform=axis.transAxes, ha="center", color="#555555")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.18)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    policy_ledger, family_ledger, source_audits = _load_evidence(config)

    claim_rows = []
    claim_cell_parts = []
    claims = config["claims"] if profile_name == "full" else config["claims"][:2]
    for claim in claims:
        row, cells = _evaluate_claim(claim, policy_ledger)
        claim_rows.append(row)
        claim_cell_parts.append(cells)
    claim_summary = pd.DataFrame(claim_rows)
    claim_cells = pd.concat(claim_cell_parts, ignore_index=True, sort=False)

    frontier_config = config["safety_frontier"]
    frontier = _safety_frontier(
        policy_ledger,
        source=frontier_config["source"],
        method=frontier_config["method"],
    )
    modality_family, modality_summary = _modality_summary(
        family_ledger,
        source=frontier_config["source"],
        method=frontier_config["method"],
        filters=frontier_config["modality_filters"],
        modality_map=config["modalities"],
    )
    negative_cells = policy_ledger.loc[
        policy_ledger["effect_status"].isin(["possible_harm", "supported_harm"])
    ].copy()
    negative_claims = claim_summary.loc[~claim_summary["gate_passed"]].copy()
    valid_effect_statuses = {
        "supported_benefit",
        "supported_harm",
        "possible_benefit",
        "possible_harm",
        "null_or_unresolved",
    }

    expected_families = set(config["modalities"])
    observed_families = set(modality_family["system_family"])
    audit = {
        "status": "pass",
        "checks": {
            "all_source_artifacts_passed": all(item.get("status") == "pass" for item in source_audits.values()),
            "all_claims_have_evidence": bool(claim_summary["cells"].gt(0).all()),
            "claim_ids_unique": not claim_summary["claim_id"].duplicated().any(),
            "finite_policy_evidence": bool(np.isfinite(policy_ledger[["family_equal_mean", "ci_low", "ci_high"]].to_numpy(float)).all()),
            "intervals_ordered": bool((policy_ledger["ci_low"] <= policy_ledger["family_equal_mean"]).all() and (policy_ledger["family_equal_mean"] <= policy_ledger["ci_high"]).all()),
            "safety_frontier_pairs_one_to_one": len(frontier) == int(frontier_config["expected_cells"]),
            "all_families_have_one_modality": observed_families == expected_families and not modality_family["modality"].isna().any(),
            "both_modalities_have_multiple_families": bool(modality_family.groupby("modality", observed=True)["system_family"].nunique().ge(2).all()),
            "experiments_kept_as_assumption_domains": policy_ledger["experiment_id"].nunique() == len(config["sources"]),
            "all_effects_classified": bool(
                policy_ledger["effect_status"].notna().all()
                and set(policy_ledger["effect_status"]).issubset(valid_effect_statuses)
            ),
        },
        "source_experiments": len(config["sources"]),
        "claims": len(claim_summary),
        "supported_claims": int(claim_summary["gate_passed"].sum()),
        "not_supported_claims": int((~claim_summary["gate_passed"]).sum()),
        "possible_or_supported_harm_cells": len(negative_cells),
        "supported_harm_cells": int(negative_cells["effect_status"].eq("supported_harm").sum()),
        "publication_claims": claim_summary[["claim_id", "publication_status", "scope"]].to_dict(orient="records"),
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"evidence-synthesis audit failed: {audit}")

    policy_ledger.to_csv(results_dir / "policy_evidence_ledger.csv", index=False)
    family_ledger.to_csv(results_dir / "family_evidence_ledger.csv", index=False)
    claim_summary.to_csv(results_dir / "claim_evidence_summary.csv", index=False)
    claim_cells.to_csv(results_dir / "claim_evidence_cells.csv", index=False)
    frontier.to_csv(results_dir / "policy_safety_frontier.csv", index=False)
    modality_family.to_csv(results_dir / "modality_family_evidence.csv", index=False)
    modality_summary.to_csv(results_dir / "modality_summary.csv", index=False)
    negative_cells.to_csv(results_dir / "negative_effect_cells.csv", index=False)
    negative_claims.to_csv(results_dir / "negative_claim_ledger.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    _plot_claim_map(claim_summary, report_dir / "claim_evidence_map.png")
    _plot_safety_frontier(frontier, report_dir / "policy_safety_frontier.png")
    _plot_modality_boundary(modality_family, modality_summary, report_dir / "modality_policy_boundary.png")

    source_hashes = {
        f"{key}_audit": _sha256(Path(item["results_path"]) / "audit.json")
        for key, item in config["sources"].items()
    }
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
        "input_hashes": source_hashes,
        "audit_status": audit["status"],
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "README.md").write_text(
        f"# {config['experiment']['name'].replace('_', ' ').title()}\n\n"
        f"Profile: **{profile_name}**. Audit: **{audit['status']}**. "
        f"Supported frozen claims: **{audit['supported_claims']}/{audit['claims']}**.\n\n"
        "Experiments are retained as separate assumption domains and are not pooled as independent replications. "
        "The safety frontier separates relative policy advantage from absolute benefit over detected-case isolation alone.\n",
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize frozen intervention evidence and claim-safety gates")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260817-002_evidence_synthesis.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
