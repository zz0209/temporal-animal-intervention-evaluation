"""Test whether a small strictly prior local probe can veto harmful zero-shot transfer."""

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
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _policy(sentinel_fraction: float, response_fraction: float) -> str:
    return f"s{round(100 * sentinel_fraction):02d}_r{round(100 * response_fraction):02d}"


def _zero_shot_candidates(
    family_policy: pd.DataFrame,
    cost_ratios: list[float],
    reference_sentinel_fraction: float,
    reference_response_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = sorted(family_policy["system_family"].unique())
    policies = family_policy[["sentinel_fraction", "response_fraction", "policy"]].drop_duplicates()
    for (model, sensitivity), context in family_policy.groupby(
        ["epidemic_model", "recognition_sensitivity"], observed=True, sort=True
    ):
        for cost_ratio in cost_ratios:
            budget = reference_sentinel_fraction + cost_ratio * reference_response_fraction
            feasible = policies.loc[
                policies["sentinel_fraction"] + cost_ratio * policies["response_fraction"] <= budget + 1e-12
            ].copy()
            feasible["nominal_cost"] = feasible["sentinel_fraction"] + cost_ratio * feasible["response_fraction"]
            for heldout in families:
                scores = (
                    context.loc[context["system_family"].ne(heldout)]
                    .merge(feasible, on=["sentinel_fraction", "response_fraction", "policy"])
                    .groupby(["policy", "sentinel_fraction", "response_fraction", "nominal_cost"], observed=True)["mean_value"]
                    .mean()
                    .reset_index(name="training_attack_rate")
                    .sort_values(
                        ["training_attack_rate", "nominal_cost", "response_fraction", "policy"],
                        ascending=[True, True, False, True],
                        kind="stable",
                    )
                )
                selected = scores.iloc[0]
                rows.append(
                    {
                        "epidemic_model": model,
                        "recognition_sensitivity": float(sensitivity),
                        "cost_ratio": float(cost_ratio),
                        "system_family": heldout,
                        "zero_shot_policy": selected["policy"],
                        "training_families": "|".join(name for name in families if name != heldout),
                    }
                )
    return pd.DataFrame(rows)


def build_forward_evaluations(
    worlds: pd.DataFrame,
    candidates: pd.DataFrame,
    prior_requirements: list[int],
    reference_policy: str,
) -> pd.DataFrame:
    anchor_keys = [
        "epidemic_model",
        "recognition_sensitivity",
        "system_family",
        "dataset_id",
        "network_id",
        "anchor_id",
        "anchor_time",
        "policy",
    ]
    anchor_policy = worlds.groupby(anchor_keys, observed=True, as_index=False)["final_attack_rate"].mean()
    rows: list[dict[str, Any]] = []
    unit_keys = ["epidemic_model", "recognition_sensitivity", "system_family", "dataset_id", "network_id"]
    candidate_index = candidates.set_index(
        ["epidemic_model", "recognition_sensitivity", "cost_ratio", "system_family"]
    )
    for unit_key, unit in anchor_policy.groupby(unit_keys, observed=True, sort=True):
        model, sensitivity, family, dataset_id, network_id = unit_key
        times = sorted(unit["anchor_time"].unique())
        for cost_ratio in sorted(candidates["cost_ratio"].unique()):
            candidate_key = (model, float(sensitivity), float(cost_ratio), family)
            if candidate_key not in candidate_index.index:
                continue
            candidate = candidate_index.loc[candidate_key]
            zero_shot_policy = str(candidate["zero_shot_policy"])
            for current_time in times:
                prior_times = [value for value in times if value < current_time]
                for required in prior_requirements:
                    if len(prior_times) < required:
                        continue
                    selected_prior_times = prior_times[-required:]
                    prior = unit.loc[unit["anchor_time"].isin(selected_prior_times)]
                    prior_wide = prior.pivot_table(
                        index=["anchor_id", "anchor_time"], columns="policy", values="final_attack_rate"
                    )
                    if reference_policy not in prior_wide or zero_shot_policy not in prior_wide:
                        raise ValueError("prior reference or zero-shot candidate is missing")
                    prior_gain = float((prior_wide[reference_policy] - prior_wide[zero_shot_policy]).mean())
                    current = unit.loc[unit["anchor_time"].eq(current_time)].set_index("policy")["final_attack_rate"]
                    if reference_policy not in current or zero_shot_policy not in current:
                        raise ValueError("current reference or zero-shot candidate is missing")
                    probe_policy = zero_shot_policy if prior_gain >= 0 else reference_policy
                    for selector, selected_policy in [
                        ("fixed_reference", reference_policy),
                        ("zero_shot_mean", zero_shot_policy),
                        ("local_probe_veto", probe_policy),
                    ]:
                        rows.append(
                            {
                                "epidemic_model": model,
                                "recognition_sensitivity": float(sensitivity),
                                "cost_ratio": float(cost_ratio),
                                "system_family": family,
                                "dataset_id": dataset_id,
                                "network_id": str(network_id),
                                "anchor_id": unit.loc[unit["anchor_time"].eq(current_time), "anchor_id"].iloc[0],
                                "anchor_time": current_time,
                                "prior_times_required": int(required),
                                "prior_times_available": len(prior_times),
                                "latest_training_time": max(selected_prior_times),
                                "selector": selector,
                                "selected_policy": selected_policy,
                                "zero_shot_policy": zero_shot_policy,
                                "prior_local_gain_for_zero_shot": prior_gain,
                                "used_reference": selected_policy == reference_policy,
                                "final_attack_rate": float(current.loc[selected_policy]),
                                "reference_attack_rate": float(current.loc[reference_policy]),
                                "value": float(current.loc[reference_policy] - current.loc[selected_policy]),
                                "training_families": candidate["training_families"],
                            }
                        )
    return pd.DataFrame(rows)


def _family_summary(evaluations: pd.DataFrame) -> pd.DataFrame:
    keys = ["epidemic_model", "recognition_sensitivity", "cost_ratio", "prior_times_required", "selector", "system_family"]
    return evaluations.groupby(keys, observed=True, as_index=False).agg(
        mean_value=("value", "mean"),
        evaluation_anchors=("anchor_id", "size"),
        nonreference_evaluations=("used_reference", lambda values: int((~values).sum())),
    )


def _summary(family: pd.DataFrame, repetitions: int, seed: int, harm_threshold: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = ["epidemic_model", "recognition_sensitivity", "cost_ratio", "prior_times_required", "selector"]
    rows = []
    for key, frame in family.groupby(keys, observed=True, sort=True):
        values = frame["mean_value"].to_numpy(float)
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        rows.append(
            dict(
                zip(keys, key),
                family_equal_mean=float(values.mean()),
                ci_low=float(np.quantile(samples, 0.025)),
                ci_high=float(np.quantile(samples, 0.975)),
                worst_family_value=float(values.min()),
                materially_harmed_families=int((values < -harm_threshold).sum()),
                positive_families=int((values > 0).sum()),
                nonreference_evaluations=int(frame["nonreference_evaluations"].sum()),
                families=len(values),
            )
        )
    return pd.DataFrame(rows)


def _plot_primary(family: pd.DataFrame, path: Path, dpi: int) -> None:
    models = list(family["epidemic_model"].unique())
    families = sorted(family["system_family"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=True)
    axes = np.atleast_1d(axes)
    selectors = ["zero_shot_mean", "local_probe_veto"]
    offsets = {"zero_shot_mean": -0.1, "local_probe_veto": 0.1}
    colors = {"zero_shot_mean": "#d95f02", "local_probe_veto": "#1b9e77"}
    labels = {"zero_shot_mean": "Zero-shot mean optimizer", "local_probe_veto": "Two-window local probe + veto"}
    for ax, model in zip(axes, models):
        frame = family.loc[family["epidemic_model"].eq(model)]
        for selector in selectors:
            values = frame.loc[frame["selector"].eq(selector)].set_index("system_family")["mean_value"].reindex(families)
            y = np.arange(len(families)) + offsets[selector]
            ax.scatter(100 * values, y, s=65, color=colors[selector], label=labels[selector])
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
        ax.axvline(-0.5, color="#b22222", linestyle=":", linewidth=1)
        ax.set_title(model.replace("temporal_", "").replace("_", " ").upper())
        ax.set_xlabel("Forward gain over fixed reference (attack-rate points)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(np.arange(len(families)), [value.replace("_", " ") for value in families])
    handles, labels_out = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Can two strictly prior local windows prevent harmful cross-system transfer?", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][profile_name]
    decision = config["decision"]
    worlds_path = _resolve(config["inputs"]["policy_worlds"])
    family_path = _resolve(config["inputs"]["family_policy"])
    audit_path = _resolve(config["inputs"]["source_audit"])
    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    worlds = pd.read_csv(worlds_path, dtype={"network_id": str, "initial_infected": str})
    worlds["anchor_time"] = pd.to_datetime(worlds["anchor_time"], format="mixed")
    family_policy = pd.read_csv(family_path)
    models = set(profile["epidemic_models"])
    sensitivities = set(map(float, profile["recognition_sensitivities"]))
    worlds = worlds.loc[worlds["epidemic_model"].isin(models) & worlds["recognition_sensitivity"].isin(sensitivities)].copy()
    family_policy = family_policy.loc[family_policy["epidemic_model"].isin(models) & family_policy["recognition_sensitivity"].isin(sensitivities)].copy()
    if profile["datasets"] != "all":
        worlds = worlds.loc[worlds["dataset_id"].isin(profile["datasets"])].copy()
    candidates = _zero_shot_candidates(
        family_policy,
        list(map(float, profile["cost_ratios"])),
        float(decision["reference_sentinel_fraction"]),
        float(decision["reference_response_fraction"]),
    )
    evaluations = build_forward_evaluations(
        worlds,
        candidates,
        list(map(int, profile["prior_time_requirements"])),
        _policy(float(decision["reference_sentinel_fraction"]), float(decision["reference_response_fraction"])),
    )
    if evaluations.empty:
        raise ValueError("no forward local-probe evaluations were estimable")
    family = _family_summary(evaluations)
    threshold = float(decision["material_harm_threshold_attack_rate"])
    summary = _summary(family, int(profile["bootstrap_replicates"]), int(config["evaluation"]["seed"]), threshold)
    primary_family = family.loc[
        family["recognition_sensitivity"].eq(float(decision["primary_recognition_sensitivity"]))
        & family["cost_ratio"].eq(float(decision["primary_cost_ratio"]))
        & family["prior_times_required"].eq(int(decision["primary_prior_times"]))
    ].copy()
    wide = primary_family.pivot(index=["epidemic_model", "system_family"], columns="selector", values="mean_value").reset_index()
    wide["probe_minus_zero_shot"] = wide["local_probe_veto"] - wide["zero_shot_mean"]
    probe = primary_family.loc[primary_family["selector"].eq("local_probe_veto")]
    zero = primary_family.loc[primary_family["selector"].eq("zero_shot_mean")]
    gate = decision["primary_gate"]
    better_worst = probe["mean_value"].min() > zero["mean_value"].min() + 1e-12
    harmed = int(probe["mean_value"].lt(-threshold).sum())
    nonreference = int(probe["nonreference_evaluations"].sum())
    gate_pass = better_worst and harmed <= int(gate["maximum_materially_harmed_families"]) and nonreference >= int(gate["minimum_nonreference_evaluations"])
    primary_decision = "supported" if gate_pass else "unsupported"
    checks = {
        "source_audit_passed": source_audit.get("status") == "pass",
        "strictly_prior_training": bool((evaluations["latest_training_time"] < evaluations["anchor_time"]).all()),
        "current_anchor_excluded": bool(evaluations["prior_times_available"].ge(evaluations["prior_times_required"]).all()),
        "complete_family_exclusion_for_zero_shot": bool(evaluations.apply(lambda row: row.system_family not in str(row.training_families).split("|"), axis=1).all()),
        "reference_zero_gain": bool(evaluations.loc[evaluations["selector"].eq("fixed_reference"), "value"].abs().lt(1e-12).all()),
        "finite_values": bool(np.isfinite(evaluations["value"]).all()),
        "selector_rows_balanced": bool(evaluations.groupby(["epidemic_model", "recognition_sensitivity", "cost_ratio", "dataset_id", "network_id", "anchor_id", "prior_times_required"], observed=True).size().eq(3).all()),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "datasets": int(evaluations["dataset_id"].nunique()),
        "families": int(evaluations["system_family"].nunique()),
        "forward_anchor_evaluations": int(evaluations[["dataset_id", "network_id", "anchor_id", "prior_times_required"]].drop_duplicates().shape[0]),
        "policy_evaluations": len(evaluations),
        "primary_decision": primary_decision,
        "primary_probe_worst_family_value": float(probe["mean_value"].min()),
        "primary_zero_shot_worst_family_value": float(zero["mean_value"].min()),
        "primary_probe_material_harms": harmed,
        "primary_probe_nonreference_evaluations": nonreference,
        "scope": "strictly_forward_local_simulation_probe_for_resource_transfer",
    }
    if audit["status"] != "pass":
        raise ValueError(f"local probe veto audit failed: {audit}")
    experiment_id = config["experiment_id"]
    results_dir = _resolve(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = _resolve(config["outputs"]["reports_root"]) / experiment_id / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    evaluations.to_csv(results_dir / "forward_evaluations.csv.gz", index=False, compression="gzip")
    family.to_csv(results_dir / "family_forward_values.csv", index=False)
    summary.to_csv(results_dir / "selector_summary.csv", index=False)
    wide.to_csv(results_dir / "primary_probe_contrasts.csv", index=False)
    candidates.to_csv(results_dir / "zero_shot_candidates.csv", index=False)
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
        "worlds_sha256": _sha256(worlds_path),
        "family_policy_sha256": _sha256(family_path),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot_primary(primary_family, report_dir / "local_probe_safety.png", int(profile["render_dpi"]))
    report = f"""# Strictly forward local probe and veto

This frozen reanalysis asks whether a zero-shot cross-system allocation can be
screened using only the most recent strictly earlier local temporal windows.
The probe uses paired simulator outcomes from completed historical windows; it
does not use the current or future evaluation anchor.

- Technical audit: **{audit['status']}**
- Primary decision: **{primary_decision}**
- Evaluated families: {audit['families']}
- Primary probe worst-family gain: {100 * audit['primary_probe_worst_family_value']:.3f} attack-rate points
- Zero-shot worst-family gain: {100 * audit['primary_zero_shot_worst_family_value']:.3f} attack-rate points
- Materially harmed families after probe: {audit['primary_probe_material_harms']}
- Non-reference forward evaluations after probe: {audit['primary_probe_nonreference_evaluations']}

The local probe is a model-based calibration procedure. It requires a locally
credible simulator and completed historical windows; it is not field-observed
policy efficacy and does not apply to a system with no prior temporal support.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260819-004_local_probe_veto.yaml"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
