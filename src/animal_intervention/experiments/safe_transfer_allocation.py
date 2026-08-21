"""Evaluate conservative cross-system resource allocation with baseline fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def _policy(sentinel_fraction: float, response_fraction: float) -> str:
    return f"s{round(100 * sentinel_fraction):02d}_r{round(100 * response_fraction):02d}"


def _candidate_scores(
    training: pd.DataFrame,
    feasible: pd.DataFrame,
    reference_policy: str,
) -> pd.DataFrame:
    values = training.merge(feasible, on=["sentinel_fraction", "response_fraction", "policy"])
    wide = values.pivot(index="system_family", columns="policy", values="mean_value")
    if reference_policy not in wide:
        raise ValueError("reference policy is absent from training data")
    improvements = wide[reference_policy].to_numpy()[:, None] - wide.to_numpy()
    best = wide.min(axis=1).to_numpy()[:, None]
    regrets = wide.to_numpy() - best
    rows = []
    for index, policy in enumerate(wide.columns):
        metadata = feasible.loc[feasible["policy"].eq(policy)].iloc[0]
        rows.append(
            {
                "policy": policy,
                "sentinel_fraction": float(metadata["sentinel_fraction"]),
                "response_fraction": float(metadata["response_fraction"]),
                "nominal_cost": float(metadata["nominal_cost"]),
                "mean_improvement": float(improvements[:, index].mean()),
                "minimum_improvement": float(improvements[:, index].min()),
                "maximum_regret": float(regrets[:, index].max()),
                "negative_training_families": int((improvements[:, index] < -1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def _select(scores: pd.DataFrame, selector: str) -> pd.Series:
    if selector == "pooled_mean":
        order = ["mean_improvement", "minimum_improvement", "nominal_cost", "policy"]
        ascending = [False, False, True, True]
    elif selector == "maximin_reference_anchored":
        order = ["minimum_improvement", "mean_improvement", "nominal_cost", "policy"]
        ascending = [False, False, True, True]
    elif selector == "minimax_regret":
        order = ["maximum_regret", "mean_improvement", "nominal_cost", "policy"]
        ascending = [True, False, True, True]
    else:
        raise ValueError(f"unknown selector: {selector}")
    return scores.sort_values(order, ascending=ascending, kind="stable").iloc[0]


def select_allocations(
    family_policy: pd.DataFrame,
    selectors: list[str],
    cost_ratios: list[float],
    reference_sentinel_fraction: float,
    reference_response_fraction: float,
    heldout_families: list[str] | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_policy = _policy(reference_sentinel_fraction, reference_response_fraction)
    rows: list[dict[str, Any]] = []
    score_rows: list[pd.DataFrame] = []
    families = sorted(family_policy["system_family"].unique())
    selected_families = families if heldout_families == "all" else list(heldout_families)
    for (model, sensitivity), context in family_policy.groupby(
        ["epidemic_model", "recognition_sensitivity"], observed=True, sort=True
    ):
        policies = context[["sentinel_fraction", "response_fraction", "policy"]].drop_duplicates()
        for cost_ratio in cost_ratios:
            budget = reference_sentinel_fraction + cost_ratio * reference_response_fraction
            feasible = policies.loc[
                policies["sentinel_fraction"]
                + cost_ratio * policies["response_fraction"]
                <= budget + 1e-12
            ].copy()
            feasible["nominal_cost"] = (
                feasible["sentinel_fraction"] + cost_ratio * feasible["response_fraction"]
            )
            if reference_policy not in set(feasible["policy"]):
                raise ValueError("reference policy is not feasible")
            for heldout in selected_families:
                training = context.loc[context["system_family"].ne(heldout)]
                held = context.loc[context["system_family"].eq(heldout)]
                scores = _candidate_scores(training, feasible, reference_policy)
                scores = scores.assign(
                    epidemic_model=model,
                    recognition_sensitivity=float(sensitivity),
                    cost_ratio=float(cost_ratio),
                    heldout_family=heldout,
                )
                score_rows.append(scores)
                reference = float(held.loc[held["policy"].eq(reference_policy), "mean_value"].iloc[0])
                for selector in selectors:
                    selected = _select(scores, selector)
                    chosen = held.loc[held["policy"].eq(selected["policy"]), "mean_value"]
                    if len(chosen) != 1:
                        raise ValueError("held-out policy value is incomplete")
                    rows.append(
                        {
                            "epidemic_model": model,
                            "recognition_sensitivity": float(sensitivity),
                            "cost_ratio": float(cost_ratio),
                            "system_family": heldout,
                            "analysis_cluster_id": heldout,
                            "selector": selector,
                            "selected_policy": selected["policy"],
                            "selected_sentinel_fraction": selected["sentinel_fraction"],
                            "selected_response_fraction": selected["response_fraction"],
                            "training_mean_improvement": selected["mean_improvement"],
                            "training_minimum_improvement": selected["minimum_improvement"],
                            "training_maximum_regret": selected["maximum_regret"],
                            "heldout_attack_rate": float(chosen.iloc[0]),
                            "reference_attack_rate": reference,
                            "value": reference - float(chosen.iloc[0]),
                            "used_reference": selected["policy"] == reference_policy,
                            "training_families": "|".join(name for name in families if name != heldout),
                        }
                    )
    return pd.DataFrame(rows), pd.concat(score_rows, ignore_index=True)


def _bootstrap_summary(table: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["epidemic_model", "recognition_sensitivity", "cost_ratio", "selector"]
    for key, frame in table.groupby(keys, observed=True, sort=True):
        values = frame["value"].to_numpy(float)
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        rows.append(
            dict(
                zip(keys, key),
                family_equal_mean=float(values.mean()),
                ci_low=float(np.quantile(samples, 0.025)),
                ci_high=float(np.quantile(samples, 0.975)),
                worst_heldout_value=float(values.min()),
                materially_harmed_families=0,
                nonreference_selections=int((~frame["used_reference"]).sum()),
                families=len(values),
            )
        )
    return pd.DataFrame(rows)


def _plot_primary(primary: pd.DataFrame, path: Path, dpi: int) -> None:
    families = sorted(primary["system_family"].unique())
    models = list(primary["epidemic_model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=True)
    axes = np.atleast_1d(axes)
    offsets = {"pooled_mean": -0.16, "maximin_reference_anchored": 0.0, "minimax_regret": 0.16}
    colors = {"pooled_mean": "#d95f02", "maximin_reference_anchored": "#1b9e77", "minimax_regret": "#7570b3"}
    labels = {"pooled_mean": "Mean optimizer", "maximin_reference_anchored": "Reference-anchored maximin", "minimax_regret": "Minimax regret"}
    for ax, model in zip(axes, models):
        frame = primary.loc[primary["epidemic_model"].eq(model)]
        for selector, group in frame.groupby("selector", observed=True):
            values = group.set_index("system_family")["value"].reindex(families)
            y = np.arange(len(families)) + offsets[selector]
            ax.scatter(100 * values, y, s=55, color=colors[selector], label=labels[selector])
        ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
        ax.axvline(-0.5, color="#b22222", linewidth=1, linestyle=":")
        ax.set_title(model.replace("temporal_", "").replace("_", " ").upper())
        ax.set_xlabel("Held-out gain over fixed reference (attack-rate points)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(np.arange(len(families)), [name.replace("_", " ") for name in families])
    handles, labels_out = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Cross-system allocation: improvement and abstention on held-out animal systems", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][profile_name]
    decision = config["decision"]
    source_path = _resolve(config["inputs"]["family_policy"])
    source_audit_path = _resolve(config["inputs"]["source_audit"])
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    family_policy = pd.read_csv(source_path)
    family_policy = family_policy.loc[
        family_policy["epidemic_model"].isin(profile["epidemic_models"])
        & family_policy["recognition_sensitivity"].isin(profile["recognition_sensitivities"])
    ].copy()
    allocations, scores = select_allocations(
        family_policy,
        list(decision["selectors"]),
        list(map(float, profile["cost_ratios"])),
        float(decision["reference_sentinel_fraction"]),
        float(decision["reference_response_fraction"]),
        profile["heldout_families"],
    )
    threshold = float(decision["material_harm_threshold_attack_rate"])
    allocations["material_harm"] = allocations["value"].lt(-threshold)
    summary = _bootstrap_summary(
        allocations, int(profile["bootstrap_replicates"]), int(config["evaluation"]["seed"])
    )
    summary["materially_harmed_families"] = [
        int(
            allocations.loc[
                allocations["epidemic_model"].eq(row.epidemic_model)
                & allocations["recognition_sensitivity"].eq(row.recognition_sensitivity)
                & allocations["cost_ratio"].eq(row.cost_ratio)
                & allocations["selector"].eq(row.selector),
                "material_harm",
            ].sum()
        )
        for row in summary.itertuples(index=False)
    ]
    primary = allocations.loc[
        allocations["recognition_sensitivity"].eq(float(decision["primary_recognition_sensitivity"]))
        & allocations["cost_ratio"].eq(float(decision["primary_cost_ratio"]))
    ].copy()
    primary_selector = str(decision["primary_selector"])
    comparator = str(decision["comparator_selector"])
    paired = primary.pivot(index=["epidemic_model", "system_family"], columns="selector", values="value").reset_index()
    paired["safety_improvement"] = paired[primary_selector] - paired[comparator]
    selected_primary = primary.loc[primary["selector"].eq(primary_selector)]
    comparator_primary = primary.loc[primary["selector"].eq(comparator)]
    gate = decision["primary_gate"]
    lower_worst_harm = selected_primary["value"].min() > comparator_primary["value"].min() + 1e-12
    harm_pass = int(selected_primary["material_harm"].sum()) <= int(gate["maximum_materially_harmed_families"])
    nontrivial = int((~selected_primary["used_reference"]).sum()) >= int(gate["minimum_nonreference_selections_across_models"])
    decision_value = "safer_nontrivial" if lower_worst_harm and harm_pass and nontrivial else ("safe_fallback_only" if lower_worst_harm and harm_pass else "unsupported")
    families = sorted(family_policy["system_family"].unique())
    checks = {
        "source_audit_passed": source_audit.get("status") == "pass",
        "family_policy_nonempty": not family_policy.empty,
        "complete_heldout_exclusion": bool(allocations.apply(lambda row: row.system_family not in str(row.training_families).split("|"), axis=1).all()),
        "all_selectors_present": set(allocations["selector"]) == set(decision["selectors"]),
        "reference_reproduces_zero_gain": bool(allocations.loc[allocations["used_reference"], "value"].abs().lt(1e-12).all()),
        "finite_values": bool(np.isfinite(allocations["value"]).all()),
        "full_has_five_families": profile_name != "full" or set(allocations["system_family"]) == set(families),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "datasets": int(source_audit.get("datasets", 0)),
        "families": int(allocations["system_family"].nunique()),
        "allocation_decisions": len(allocations),
        "primary_decision": decision_value,
        "primary_worst_value": float(selected_primary["value"].min()),
        "comparator_worst_value": float(comparator_primary["value"].min()),
        "primary_material_harms": int(selected_primary["material_harm"].sum()),
        "primary_nonreference_selections": int((~selected_primary["used_reference"]).sum()),
        "scope": "reference_anchored_cross_system_allocation_reanalysis",
    }
    if audit["status"] != "pass":
        raise ValueError(f"safe transfer allocation audit failed: {audit}")
    experiment_id = config["experiment_id"]
    results_dir = _resolve(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = _resolve(config["outputs"]["reports_root"]) / experiment_id / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    allocations.to_csv(results_dir / "heldout_allocations.csv", index=False)
    scores.to_csv(results_dir / "training_policy_scores.csv", index=False)
    summary.to_csv(results_dir / "selector_summary.csv", index=False)
    paired.to_csv(results_dir / "primary_paired_safety.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "source_path": str(source_path.relative_to(ROOT)),
        "source_sha256": _sha256(source_path),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot_primary(primary, report_dir / "heldout_safety_profile.png", int(profile["render_dpi"]))
    report = f"""# Reference-anchored safe transfer allocation

This locked reanalysis compares a pooled mean optimizer with a reference-anchored
maximin selector and a minimax-regret selector. A complete animal-system family is
excluded from every selection calculation. The fixed 10% monitoring + 5% response
allocation remains available as the fallback.

- Technical audit: **{audit['status']}**
- Primary decision: **{decision_value}**
- Primary worst held-out gain, maximin: {100 * audit['primary_worst_value']:.3f} attack-rate points
- Primary worst held-out gain, pooled mean: {100 * audit['comparator_worst_value']:.3f} attack-rate points
- Materially harmed held-out families, maximin: {audit['primary_material_harms']}
- Non-reference primary selections, maximin: {audit['primary_nonreference_selections']}

This is an empirical complete-system stress test, not a formal safety guarantee.
Returning the reference is an abstention decision, not evidence of improvement.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260819-003_safe_transfer_allocation.yaml"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
