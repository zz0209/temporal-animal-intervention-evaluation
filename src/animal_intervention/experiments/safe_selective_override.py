from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from .set_value_model import (
    DISPLAY_NAMES,
    SET_KEY,
    _cpu_device_consistency,
    _member_tensors,
    _prepare,
    _sha256,
)
from .set_value_ranking_diagnostic import (
    _decision_table,
    _stratified_context_limit,
    _train_pairwise_model,
)


def _git_value(arguments: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _apply_threshold(decisions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = decisions.copy()
    changed = frame["selected_set_signature"].ne(frame["reference_set_signature"])
    override = changed & frame["normalized_margin"].ge(threshold)
    frame["override"] = override
    frame["policy_gain"] = np.where(override, frame["gain"], 0.0)
    frame["policy_value"] = np.where(
        override, frame["ranking_value"], frame["reference_value"]
    )
    frame["policy_set_signature"] = np.where(
        override, frame["selected_set_signature"], frame["reference_set_signature"]
    )
    return frame


def _bootstrap_probability(
    decisions: pd.DataFrame, repetitions: int, seed: int
) -> tuple[float, float, float]:
    anchor = (
        decisions.groupby(
            ["system_family", "dataset_id", "network_id", "anchor_id"],
            observed=True,
        )["policy_gain"]
        .mean()
        .reset_index()
    )
    families = sorted(anchor["system_family"].unique())
    arrays = {
        str(name): group["policy_gain"].to_numpy(float)
        for name, group in anchor.groupby("system_family", observed=True, sort=True)
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        family_means = []
        for name in rng.choice(families, size=len(families), replace=True):
            values = arrays[str(name)]
            selected = rng.integers(0, len(values), size=len(values))
            family_means.append(float(values[selected].mean()))
        samples[repetition] = float(np.mean(family_means))
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        float((samples > 0).mean()),
    )


def _policy_summary(
    decisions: pd.DataFrame, repetitions: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    tolerance = 1e-12
    for family, group in decisions.groupby("system_family", observed=True, sort=True):
        overrides = group.loc[group["override"]]
        rows.append(
            {
                "system_family": family,
                "contexts": len(group),
                "overrides": int(group["override"].sum()),
                "coverage": float(group["override"].mean()),
                "mean_policy_gain": float(group["policy_gain"].mean()),
                "mean_unrestricted_gain": float(group["gain"].mean()),
                "helpful_override_fraction": float(
                    overrides["gain"].gt(tolerance).mean() if len(overrides) else 0.0
                ),
                "harmful_override_fraction": float(
                    overrides["gain"].lt(-tolerance).mean() if len(overrides) else 0.0
                ),
            }
        )
    family = pd.DataFrame(rows)
    low, high, probability = _bootstrap_probability(decisions, repetitions, seed)
    active = family.loc[family["overrides"].gt(0)]
    overall = {
        "families": len(family),
        "contexts": len(decisions),
        "overrides": int(decisions["override"].sum()),
        "families_with_overrides": int(family["overrides"].gt(0).sum()),
        "families_with_positive_gain": int(family["mean_policy_gain"].gt(0).sum()),
        "family_equal_coverage": float(family["coverage"].mean()),
        "family_equal_mean_gain": float(family["mean_policy_gain"].mean()),
        "family_equal_mean_unrestricted_gain": float(
            family["mean_unrestricted_gain"].mean()
        ),
        "family_equal_helpful_override_fraction": float(
            active["helpful_override_fraction"].mean() if len(active) else 0.0
        ),
        "family_equal_harmful_override_fraction": float(
            active["harmful_override_fraction"].mean() if len(active) else 0.0
        ),
        "gain_ci_low": low,
        "gain_ci_high": high,
        "bootstrap_probability_positive": probability,
    }
    return family, overall


def _calibrate_threshold(
    decisions: pd.DataFrame,
    thresholds: list[float],
    settings: dict[str, Any],
    repetitions: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for index, threshold in enumerate(sorted(thresholds)):
        policy = _apply_threshold(decisions, threshold)
        _, summary = _policy_summary(policy, repetitions, seed + index)
        feasible = bool(
            summary["overrides"] >= int(settings["minimum_contexts_overridden"])
            and summary["families_with_overrides"]
            >= int(settings["minimum_families_with_overrides"])
            and summary["family_equal_coverage"]
            >= float(settings["minimum_family_equal_coverage"])
            and summary["family_equal_mean_gain"] > 0
            and summary["families_with_positive_gain"]
            >= int(settings["minimum_positive_families"])
            and summary["bootstrap_probability_positive"]
            >= float(settings["minimum_bootstrap_probability_positive"])
            and summary["family_equal_harmful_override_fraction"]
            <= float(settings["maximum_family_equal_harmful_override_fraction"])
        )
        rows.append({"threshold": threshold, "feasible": feasible, **summary})
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    feasible = table.loc[table["feasible"]]
    selected = float(feasible.iloc[0]["threshold"]) if len(feasible) else float("inf")
    table["selected"] = table["threshold"].eq(selected)
    return selected, table


def _plots(
    thresholds: pd.DataFrame,
    family: pd.DataFrame,
    decisions: pd.DataFrame,
    overall: dict[str, Any],
    report_dir: Path,
) -> None:
    display = {
        name: DISPLAY_NAMES.get(name, name.replace("_", " "))
        for name in sorted(thresholds["held_out_family"].unique())
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, (family_name, group) in zip(axes.flat, thresholds.groupby("held_out_family", sort=True)):
        axis.plot(100 * group["family_equal_coverage"], 100 * group["family_equal_mean_gain"], "o-", color="#4C78A8")
        selected = group.loc[group["selected"]]
        if len(selected):
            axis.scatter(100 * selected["family_equal_coverage"], 100 * selected["family_equal_mean_gain"], marker="*", s=180, color="#F58518", zorder=3)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(f"Held out: {display[family_name]}")
        axis.grid(alpha=0.18)
    fig.suptitle("Nested calibration gain–coverage frontier", fontsize=19, fontweight="bold", y=0.985)
    fig.text(0.5, 0.94, "Stars mark the threshold selected without using the held-out animal system", ha="center", color="#555555")
    fig.supxlabel("Calibration contexts overridden (%)", y=0.035)
    fig.supylabel("Calibration gain over stable + tracing (percentage points)", x=0.02)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.10, hspace=0.28, wspace=0.18)
    fig.savefig(report_dir / "nested_calibration_frontier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ordered = family.sort_values("mean_policy_gain")
    names = [display[name] for name in ordered["system_family"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    axes[0].barh(names, 100 * ordered["coverage"], color="#72B7B2")
    axes[0].set_xlabel("Held-out contexts overridden (%)")
    axes[1].barh(names, 100 * ordered["mean_policy_gain"], color="#4C78A8")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Gain over stable + tracing\n(attack-rate percentage points)")
    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
    fig.suptitle("Selective policy transfer to unseen animal systems", fontsize=19, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, "A zero bar means nested calibration abstained and retained the reference policy", ha="center", color="#555555")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.83, bottom=0.15, wspace=0.30)
    fig.savefig(report_dir / "held_out_policy_by_family.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    tolerance = 1e-12
    categories = [
        ("Reference retained", ~decisions["override"]),
        ("Override: same outcome", decisions["override"] & decisions["gain"].abs().le(tolerance)),
        ("Override: helpful", decisions["override"] & decisions["gain"].gt(tolerance)),
        ("Override: harmful", decisions["override"] & decisions["gain"].lt(-tolerance)),
    ]
    counts = [int(mask.sum()) for _, mask in categories]
    values = [count / len(decisions) for count in counts]
    fig, axis = plt.subplots(figsize=(9.5, 5.8))
    bars = axis.bar([label for label, _ in categories], values, color=["#777777", "#D9D9D9", "#4C78A8", "#F58518"])
    for bar, count in zip(bars, counts):
        axis.annotate(
            f"{count:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    axis.set_ylabel("Fraction of all held-out decision contexts")
    axis.set_ylim(0, max(values + [0.1]) * 1.18)
    axis.grid(axis="y", alpha=0.18)
    fig.suptitle("What the calibrated policy actually changed", fontsize=19, fontweight="bold", y=0.985)
    fig.text(0.5, 0.92, f"Family-equal mean gain: {100 * overall['family_equal_mean_gain']:.3f} percentage points", ha="center", color="#555555")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.18)
    fig.savefig(report_dir / "selective_decision_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    model_dir = results_dir / "models"
    for directory in (results_dir, report_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    data_paths = {key: Path(value) for key, value in config["data"].items()}
    precision_audit = json.loads(data_paths["precision_audit"].read_text(encoding="utf-8"))
    ranking_audit = json.loads(data_paths["ranking_audit"].read_text(encoding="utf-8"))
    if precision_audit.get("status") != "pass":
        raise ValueError("EXP-013 artifact audit must pass")
    if ranking_audit.get("status") != "pass":
        raise ValueError("EXP-015 artifact audit must pass")
    labels = pd.read_csv(data_paths["labels"], dtype={"initial_infected": str})
    members = pd.read_csv(data_paths["members"], dtype={"initial_infected": str, "candidate_id": str})
    reliability = pd.read_csv(data_paths["reliability"], dtype={"initial_infected": str})
    profile = dict(config["profiles"][profile_name])
    prepare_profile = dict(profile)
    prepare_profile["maximum_contexts_per_family"] = None
    labels, members = _prepare(labels, members, reliability, prepare_profile)
    labels, members = _stratified_context_limit(labels, members, profile.get("maximum_contexts_per_family"))
    labels = labels.reset_index(drop=True)
    member_features = list(config["features"]["member"])
    context_features = list(config["features"]["context"])
    member_values, member_mask = _member_tensors(labels, members, member_features)
    settings = dict(config["model"])
    settings.update({key: value for key, value in profile.items() if key in settings})
    seed = int(config["evaluation"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(config["execution"]["deterministic"])
    torch.use_deterministic_algorithms(deterministic)
    preferred = str(config["execution"]["preferred_device"])
    device = torch.device("cuda" if preferred == "cuda" and torch.cuda.is_available() else "cpu")
    consistency_ok, consistency_difference = _cpu_device_consistency(
        len(member_features), len(context_features), int(settings["hidden_features"]), seed, device
    )
    families = sorted(labels["system_family"].unique())
    outer_decisions = []
    calibration_decisions = []
    threshold_tables = []
    histories = []
    metadata_rows = []
    calibration_repetitions = int(profile.get("calibration_bootstrap_repetitions", config["calibration"]["bootstrap_repetitions"]))
    final_repetitions = int(profile.get("bootstrap_repetitions", config["evaluation"]["bootstrap_repetitions"]))
    progress = tqdm(total=len(families) * len(families), desc="Nested selective models", unit="model")
    try:
        for outer_index, held_out in enumerate(families):
            other_families = [name for name in families if name != held_out]
            inner_fold_decisions = []
            for inner_index, calibration_family in enumerate(other_families):
                fit_families = [name for name in other_families if name != calibration_family]
                train_mask = labels["system_family"].isin(fit_families).to_numpy()
                test_mask = labels["system_family"].eq(calibration_family).to_numpy()
                prediction, history, metadata, state, _ = _train_pairwise_model(
                    labels, member_values, member_mask, train_mask, test_mask, context_features,
                    len(member_features), settings, seed + 100 * outer_index + inner_index, device,
                )
                predicted = labels.loc[test_mask].copy()
                predicted["ranking_prediction"] = prediction
                decisions = _decision_table(predicted, "ranking_prediction")
                decisions["outer_held_out_family"] = held_out
                decisions["calibration_family"] = calibration_family
                inner_fold_decisions.append(decisions)
                history["fold_label"] = f"calibrate {held_out} via {calibration_family}"
                histories.append(history)
                metadata.update({"role": "inner_calibration", "outer_held_out_family": held_out, "held_out_family": calibration_family, "fit_families": fit_families})
                metadata_rows.append(metadata)
                torch.save({"state_dict": state, "metadata": metadata}, model_dir / f"inner_{held_out}_{calibration_family}.pt")
                progress.update(1)
            calibration_frame = pd.concat(inner_fold_decisions, ignore_index=True)
            selected_threshold, threshold_table = _calibrate_threshold(
                calibration_frame,
                list(config["calibration"]["normalized_margin_thresholds"]),
                config["calibration"],
                calibration_repetitions,
                seed + 1000 + outer_index,
            )
            threshold_table["held_out_family"] = held_out
            threshold_table["selected_threshold"] = selected_threshold
            threshold_tables.append(threshold_table)
            calibration_frame["selected_threshold"] = selected_threshold
            calibration_decisions.append(calibration_frame)

            train_mask = labels["system_family"].ne(held_out).to_numpy()
            test_mask = labels["system_family"].eq(held_out).to_numpy()
            prediction, history, metadata, state, _ = _train_pairwise_model(
                labels, member_values, member_mask, train_mask, test_mask, context_features,
                len(member_features), settings, seed + 10000 + outer_index, device,
            )
            predicted = labels.loc[test_mask].copy()
            predicted["ranking_prediction"] = prediction
            decisions = _apply_threshold(_decision_table(predicted, "ranking_prediction"), selected_threshold)
            decisions["selected_threshold"] = selected_threshold
            decisions["calibration_abstained"] = not np.isfinite(selected_threshold)
            outer_decisions.append(decisions)
            history["fold_label"] = f"outer {held_out}"
            histories.append(history)
            metadata.update(
                {
                    "role": "outer_test",
                    "held_out_family": held_out,
                    "selected_threshold": (
                        selected_threshold if np.isfinite(selected_threshold) else None
                    ),
                }
            )
            metadata_rows.append(metadata)
            torch.save({"state_dict": state, "metadata": metadata}, model_dir / f"outer_{held_out}.pt")
            progress.update(1)
    finally:
        progress.close()
    outer = pd.concat(outer_decisions, ignore_index=True)
    calibration = pd.concat(calibration_decisions, ignore_index=True)
    thresholds = pd.concat(threshold_tables, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    family_summary, overall = _policy_summary(outer, final_repetitions, seed + 20000)
    selected_rows = thresholds.loc[thresholds["selected"]]
    no_selection = set(families) - set(selected_rows["held_out_family"])
    threshold_by_family = thresholds.groupby("held_out_family")["selected"].any()
    all_folds_accounted = set(threshold_by_family.index) == set(families)
    scientific_pass = bool(
        overall["family_equal_mean_gain"] > 0
        and overall["family_equal_coverage"] >= float(config["evaluation"]["minimum_test_coverage"])
        and overall["families_with_positive_gain"] >= int(config["evaluation"]["minimum_positive_families"])
        and overall["bootstrap_probability_positive"] >= float(config["evaluation"]["minimum_bootstrap_probability_positive"])
        and overall["family_equal_harmful_override_fraction"] <= float(config["evaluation"]["maximum_harmful_override_fraction"])
    )
    interpretation = (
        "safe_selective_override_supported" if scientific_pass
        else "nested_calibration_abstains" if overall["overrides"] == 0
        else "selective_override_not_supported"
    )
    checks = {
        "precision_artifact_passed": precision_audit.get("status") == "pass",
        "ranking_artifact_passed": ranking_audit.get("status") == "pass",
        "four_independent_system_families": len(families) == 4,
        "all_outer_folds_accounted": all_folds_accounted,
        "calibration_excludes_outer_family": bool(
            calibration["outer_held_out_family"].ne(calibration["system_family"]).all()
        ),
        "three_inner_calibration_families_per_outer_fold": bool(
            calibration.groupby("outer_held_out_family", observed=True)["calibration_family"]
            .nunique()
            .eq(3)
            .all()
        ),
        "one_outer_decision_per_context": not outer.duplicated("context_id").any(),
        "finite_outer_values": bool(np.isfinite(outer[["ranking_value", "reference_value", "gain", "policy_gain", "normalized_margin"]]).all().all()),
        "nonnegative_normalized_margins": bool(outer["normalized_margin"].ge(-1e-12).all()),
        "abstaining_folds_retain_reference": bool((outer.loc[outer["calibration_abstained"], "policy_gain"] == 0).all()),
        "cpu_device_forward_consistency": consistency_ok,
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scientific_gate": {"status": "pass" if scientific_pass else "fail", "interpretation": interpretation, **overall},
        "calibration_folds_without_feasible_threshold": sorted(no_selection),
    }
    if audit["status"] != "pass":
        raise ValueError(f"safe selective override artifact audit failed: {audit}")
    outer.to_csv(results_dir / "outer_selective_decisions.csv.gz", index=False, compression="gzip")
    calibration.to_csv(results_dir / "nested_calibration_decisions.csv.gz", index=False, compression="gzip")
    thresholds.to_csv(results_dir / "threshold_selection.csv", index=False)
    family_summary.to_csv(results_dir / "family_policy_summary.csv", index=False)
    pd.DataFrame([overall]).to_csv(results_dir / "overall_policy_summary.csv", index=False)
    history.to_csv(results_dir / "training_history.csv", index=False)
    (results_dir / "fold_metadata.json").write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _plots(thresholds, family_summary, outer, overall, report_dir)
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "numeric_precision": "float32",
        "deterministic_algorithms": deterministic,
        "cpu_device_max_abs_difference": consistency_difference,
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "input_hashes": {key: _sha256(path) for key, path in data_paths.items()},
        "artifact_audit": audit["status"],
        "scientific_gate": audit["scientific_gate"]["status"],
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "README.md").write_text(
        "# Safe selective ranking override\n\n"
        f"Profile: **{profile_name}**. Artifact audit: **{audit['status']}**. Scientific gate: **{audit['scientific_gate']['status']}**. "
        f"Interpretation: **{interpretation}**.\n\n"
        "Stable plus tracing is the immutable default. Each held-out animal system uses a threshold selected only from nested out-of-family predictions among the remaining systems. "
        "A fold with no feasible threshold abstains completely. Held-out outcomes are never used to select its threshold.\n",
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run safe selective ranking override evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260816-016_safe_selective_override.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
