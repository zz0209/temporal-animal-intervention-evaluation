from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
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

from .history_baseline_substitution import _markdown_table
from .intervention_delivery_sensitivity import (
    NATURAL_KEYS,
    POLICY_KEYS,
    _hierarchical_summary,
    _parameter_pool,
    _run_task,
    _select_parameter_regimes,
)
from .outbreak_response_pilot import (
    _git_value,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _sha256,
)


METHODS = ["history_weight", "contact_to_detected"]
VALUE_METRICS = [
    "information_gain",
    "delay_cost",
    "net_wait_value",
    "reactive_absolute_value",
    "history_absolute_value",
]
CROSS_TIME_KEYS = NATURAL_KEYS + [
    "epidemic_model",
    "action_delay_fraction",
    "residual_contact_multiplier",
    "secondary_case_sensitivity",
    "false_positive_rate",
    "rewiring_fraction",
    "rewiring_mode",
    "budget_fraction",
]


def decompose_policy_value(worlds: pd.DataFrame, earliest_profile: str) -> pd.DataFrame:
    """Decompose waiting-and-updating value in paired natural epidemic worlds."""

    keys = POLICY_KEYS + ["budget_fraction", "epidemic_model", "detection_fraction"]
    history = worlds.loc[
        worlds["method"].eq("history_weight"),
        keys
        + [
            "system_family",
            "analysis_cluster_id",
            "population_size",
            "augmented_final_size",
            "standard_final_size",
            "natural_final_size",
            "detected_cases",
            "case_contact_evidence_mass",
            "case_contact_evidence_nodes",
            "case_contact_evidence_node_fraction",
        ],
    ].rename(columns={"augmented_final_size": "history_final_size"})
    reactive = worlds.loc[
        worlds["method"].eq("contact_to_detected"),
        keys + ["augmented_final_size"],
    ].rename(columns={"augmented_final_size": "reactive_final_size"})
    paired = history.merge(reactive, on=keys, validate="one_to_one")
    earliest = paired.loc[
        paired["detection_profile"].eq(earliest_profile),
        CROSS_TIME_KEYS + ["history_final_size"],
    ].rename(columns={"history_final_size": "earliest_history_final_size"})
    paired = paired.merge(earliest, on=CROSS_TIME_KEYS, validate="many_to_one")
    denominator = paired["population_size"].astype(float)
    paired["information_gain"] = (
        paired["history_final_size"] - paired["reactive_final_size"]
    ) / denominator
    paired["delay_cost"] = (
        paired["history_final_size"] - paired["earliest_history_final_size"]
    ) / denominator
    paired["net_wait_value"] = (
        paired["earliest_history_final_size"] - paired["reactive_final_size"]
    ) / denominator
    paired["reactive_absolute_value"] = (
        paired["standard_final_size"] - paired["reactive_final_size"]
    ) / denominator
    paired["history_absolute_value"] = (
        paired["standard_final_size"] - paired["history_final_size"]
    ) / denominator
    return paired


def _summarize_values(
    decomposition: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    families = []
    for index, metric in enumerate(VALUE_METRICS):
        summary, family = _hierarchical_summary(
            decomposition,
            value_column=metric,
            group_columns=["epidemic_model", "detection_fraction"],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index * 100,
        )
        summary["metric"] = metric
        family["metric"] = metric
        summaries.append(summary)
        families.append(family)
    return pd.concat(summaries, ignore_index=True), pd.concat(families, ignore_index=True)


def _cluster_slopes(decomposition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["system_family", "analysis_cluster_id", "epidemic_model"]
    for keys, frame in decomposition.groupby(group_columns, observed=True, sort=True):
        curve = frame.groupby("detection_fraction", observed=True)[VALUE_METRICS].mean().reset_index()
        x = curve["detection_fraction"].to_numpy(float)
        if len(x) < 2 or np.std(x) == 0:
            continue
        for metric in ("information_gain", "delay_cost", "net_wait_value"):
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "metric": metric,
                    "slope": float(np.polyfit(x, curve[metric].to_numpy(float), 1)[0]),
                }
            )
    return pd.DataFrame(rows)


def _summarize_slopes(
    slopes: pd.DataFrame, *, bootstrap_replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, family = _hierarchical_summary(
        slopes,
        value_column="slope",
        group_columns=["epidemic_model", "metric"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    return summary, family


def _classify(summary: pd.DataFrame, latest: float) -> pd.DataFrame:
    rows = []
    for model, frame in summary.loc[summary["detection_fraction"].eq(latest)].groupby(
        "epidemic_model", observed=True, sort=True
    ):
        metrics = frame.set_index("metric")
        families = int(metrics.loc["net_wait_value", "families"])
        required = int(math.ceil(0.8 * families))
        net = metrics.loc["net_wait_value"]
        absolute = metrics.loc["reactive_absolute_value"]
        information = metrics.loc["information_gain"]
        operational = (
            float(net.ci_low) > 0
            and int(net.positive_families) >= required
            and float(absolute.ci_low) > 0
            and int(absolute.positive_families) >= required
        )
        informative = (
            float(information.ci_low) > 0
            and int(information.positive_families) >= required
        )
        decision = (
            "operational_upgrade_supported"
            if operational
            else "informative_but_too_late"
            if informative
            else "retain_early_history_or_abstain"
        )
        rows.append(
            {
                "epidemic_model": model,
                "latest_detection_fraction": latest,
                "families": families,
                "required_positive_families": required,
                "information_gain": float(information.family_equal_mean),
                "information_gain_ci_low": float(information.ci_low),
                "information_gain_ci_high": float(information.ci_high),
                "net_wait_value": float(net.family_equal_mean),
                "net_wait_value_ci_low": float(net.ci_low),
                "net_wait_value_ci_high": float(net.ci_high),
                "reactive_absolute_value": float(absolute.family_equal_mean),
                "reactive_absolute_ci_low": float(absolute.ci_low),
                "reactive_absolute_ci_high": float(absolute.ci_high),
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_family_out(
    decomposition: pd.DataFrame,
    *,
    latest: float,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for index, held_out in enumerate(sorted(decomposition["system_family"].unique())):
        subset = decomposition.loc[~decomposition["system_family"].eq(held_out)]
        summary, _ = _summarize_values(
            subset,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index * 1000,
        )
        decisions = _classify(summary, latest)
        decisions.insert(0, "held_out_family", held_out)
        rows.append(decisions)
    return pd.concat(rows, ignore_index=True)


def _evidence_summary(worlds: pd.DataFrame) -> pd.DataFrame:
    unique = worlds.loc[worlds["method"].eq("contact_to_detected")].copy()
    cluster = (
        unique.groupby(
            ["epidemic_model", "detection_fraction", "system_family", "analysis_cluster_id"],
            observed=True,
        )[
            [
                "detected_cases",
                "case_contact_evidence_mass",
                "case_contact_evidence_node_fraction",
            ]
        ]
        .mean()
        .reset_index()
    )
    family = (
        cluster.groupby(["epidemic_model", "detection_fraction", "system_family"], observed=True)[
            [
                "detected_cases",
                "case_contact_evidence_mass",
                "case_contact_evidence_node_fraction",
            ]
        ]
        .mean()
        .reset_index()
    )
    return (
        family.groupby(["epidemic_model", "detection_fraction"], observed=True)[
            [
                "detected_cases",
                "case_contact_evidence_mass",
                "case_contact_evidence_node_fraction",
            ]
        ]
        .mean()
        .reset_index()
    )


def _plot_curves(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    models = ["temporal_sir", "temporal_seir_erlang"]
    labels = {
        "information_gain": "Same-time information gain",
        "delay_cost": "Delay opportunity cost",
        "net_wait_value": "Net value of waiting + update",
    }
    colors = {"information_gain": "#4C78A8", "delay_cost": "#E45756", "net_wait_value": "#59A14F"}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), sharey=True)
    for axis, model in zip(axes, models):
        frame = summary.loc[summary["epidemic_model"].eq(model)]
        for metric in labels:
            selected = frame.loc[frame["metric"].eq(metric)].sort_values("detection_fraction")
            x = selected["detection_fraction"].to_numpy(float)
            y = 100 * selected["family_equal_mean"].to_numpy(float)
            low = 100 * selected["ci_low"].to_numpy(float)
            high = 100 * selected["ci_high"].to_numpy(float)
            axis.plot(x, y, marker="o", linewidth=2, color=colors[metric], label=labels[metric])
            axis.fill_between(x, low, high, color=colors[metric], alpha=0.14)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title("Temporal SIR" if model == "temporal_sir" else "Staged SEIR/Erlang", fontweight="bold")
        axis.set_xlabel("Evidence time (mean infectious periods after outbreak start)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Family-equal avoided attack-rate difference (percentage points)")
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Information can improve target choice yet still arrive too late", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.15, wspace=0.10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_family_frontier(family: pd.DataFrame, latest: float, path: Path, dpi: int) -> None:
    selected = family.loc[
        family["detection_fraction"].eq(latest)
        & family["metric"].isin(["information_gain", "delay_cost"])
    ]
    wide = selected.pivot_table(
        index=["epidemic_model", "system_family"], columns="metric", values="mean_value"
    ).reset_index()
    values = 100 * wide[["information_gain", "delay_cost"]].to_numpy(float)
    lower = min(float(values.min()), 0.0) - 0.5
    upper = max(float(values.max()), 0.0) + 0.5
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), sharex=True, sharey=True)
    families = sorted(wide["system_family"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(families)))
    family_colors = dict(zip(families, colors))
    for axis, model in zip(axes, ["temporal_sir", "temporal_seir_erlang"]):
        frame = wide.loc[wide["epidemic_model"].eq(model)]
        axis.plot([lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=1)
        for row in frame.itertuples(index=False):
            axis.scatter(
                100 * row.delay_cost,
                100 * row.information_gain,
                s=70,
                color=family_colors[str(row.system_family)],
                label=str(row.system_family).replace("_", " "),
            )
        axis.set_title("Temporal SIR" if model == "temporal_sir" else "Staged SEIR/Erlang", fontweight="bold")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Delay cost at 0.75 (percentage points)")
    axes[0].set_ylabel("Same-time information gain at 0.75 (percentage points)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Above the diagonal: information gain pays for waiting", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.15, wspace=0.10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_evidence(evidence: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    for model, label, color in [
        ("temporal_sir", "Temporal SIR", "#4C78A8"),
        ("temporal_seir_erlang", "Staged SEIR/Erlang", "#F58518"),
    ]:
        frame = evidence.loc[evidence["epidemic_model"].eq(model)].sort_values("detection_fraction")
        axes[0].plot(frame["detection_fraction"], frame["detected_cases"], marker="o", label=label, color=color)
        axes[1].plot(frame["detection_fraction"], 100 * frame["case_contact_evidence_node_fraction"], marker="o", label=label, color=color)
    axes[0].set_ylabel("Family-equal detected animals")
    axes[1].set_ylabel("Animals with observed contact to a detected case (%)")
    for axis in axes:
        axis.set_xlabel("Evidence time (mean infectious periods)")
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("What additional outbreak evidence is actually available?", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16, wspace=0.20)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile = dict(config["profiles"][profile_name])
    decision = dict(config["decision"])
    evaluation = dict(config["evaluation"])
    prerequisite_path = Path(config["data"]["prerequisite_audit"])
    prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    if prerequisite.get("status") != "pass":
        raise ValueError("prerequisite audit must pass")
    stable_path = Path(config["data"]["stable_prediction_path"])
    stable_predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    stable_predictions["anchor_time"] = pd.to_datetime(stable_predictions["anchor_time"], format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    detections = [DetectionProfile(**item) for item in decision["detection_profiles"]]
    detection_fractions = {item.name: float(item.delay_fraction_of_mean_infectious_period) for item in detections}
    models = [dict(item) for item in decision["epidemic_models"]]
    task_specs = []
    support_rows = []
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        available = set(
            stable_predictions.loc[stable_predictions["dataset_id"].eq(dataset_id), ["network_id", "anchor_time"]]
            .itertuples(index=False, name=None)
        )
        for window in windows:
            window.setdefault("network_id", default_network_id)
        windows = [
            window
            for window in windows
            if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time)) in available
        ]
        maximum = profile.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        parameters = _parameter_pool(
            Path(specification["source_results"]) / "parameter_selection.csv",
            str(evaluation["parameter_pool"]),
        )
        for window in windows:
            anchor = window["anchor"]
            compatible = []
            for parameter in parameters.itertuples(index=False):
                mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
                supported = all(
                    detection_time_from_seed(anchor.anchor_time, anchor.horizon_end, mean_period, detection)
                    + mean_period * float(decision["action_delay_fraction_of_mean_infectious_period"])
                    < anchor.horizon_end
                    for detection in detections
                )
                support_rows.append({"dataset_id": dataset_id, "network_id": str(window["network_id"]), "anchor_id": anchor.anchor_id, "parameter_id": parameter.parameter_id, "supported": supported})
                if supported:
                    compatible.append(parameter)
            selected = _select_parameter_regimes(compatible, str(evaluation["parameter_selection_mode"]))
            if len(selected) != 1:
                continue
            _, parameter = selected[0]
            network_id = str(window["network_id"])
            cluster = (
                f"{dataset_id}::{network_id}"
                if specification.get("analysis_cluster") == "network"
                else f"{dataset_id}::{network_id}::{anchor.anchor_id}"
            )
            stable = _matching_stable_scores(stable_predictions, dataset_id, network_id, anchor.anchor_time, window["eligible"])
            seeds = stable_hash_order(
                list(map(str, window["eligible"])),
                int(evaluation["seed"]),
                dataset_id,
                anchor.anchor_id,
                "information_accrual_seeds",
            )[: int(profile["seeds_per_anchor"])]
            common = {
                "dataset_id": dataset_id,
                "network_id": network_id,
                "system_family": str(specification["system_family"]),
                "analysis_cluster_id": cluster,
                "window": window,
                "parameter": parameter,
                "stable_scores": stable,
                "seed_nodes": seeds,
            }
            for model in models:
                for detection in detections:
                    task_specs.append({**common, "epidemic_model": model, "detection_profile": detection})
    fingerprint = hashlib.sha256(
        config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()[:12]
    frames = []
    progress = tqdm(task_specs, desc="Information-accrual worlds", unit="task")
    for task in progress:
        model = task["epidemic_model"]
        detection = task["detection_profile"]
        identity = "|".join(
            [fingerprint, task["dataset_id"], task["network_id"], task["window"]["anchor"].anchor_id, str(task["parameter"].parameter_id), str(model["name"]), detection.name]
        )
        checkpoint = checkpoint_dir / f"worlds_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
        frame = pd.DataFrame()
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str})
        if frame.empty:
            frame = _run_task(
                dataset_id=task["dataset_id"],
                network_id=task["network_id"],
                system_family=task["system_family"],
                analysis_cluster_id=task["analysis_cluster_id"],
                window=task["window"],
                parameter=task["parameter"],
                detection_profile=detection,
                action_delay_fraction=float(decision["action_delay_fraction_of_mean_infectious_period"]),
                residual_contact_multiplier=float(decision["residual_contact_multiplier"]),
                stable_scores=task["stable_scores"],
                methods=list(decision["methods"]),
                seed_nodes=task["seed_nodes"],
                random_blocks=int(profile["random_blocks"]),
                minimum_budget=int(decision["minimum_additional_budget"]),
                budget_fraction=float(decision["additional_budget_fraction"]),
                secondary_case_sensitivity=float(decision["secondary_case_sensitivity"]),
                false_positive_rate=float(decision["false_positive_rate"]),
                rewiring_fraction=float(decision["rewiring_fraction"]),
                rewiring_mode=str(decision["rewiring_mode"]),
                tracing_half_life_fraction=float(decision["tracing_half_life_fraction_of_mean_infectious_period"]),
                experiment_seed=int(evaluation["seed"]),
                epidemic_model=model,
            )
            frame["detection_fraction"] = detection_fractions[detection.name]
            frame.to_csv(checkpoint, index=False, compression="gzip")
        elif "detection_fraction" not in frame:
            frame["detection_fraction"] = detection_fractions[detection.name]
        frames.append(frame)
        progress.set_postfix_str(f"{task['dataset_id']} {model['name']} {detection.name}")
    worlds = pd.concat(frames, ignore_index=True)
    for column in ("anchor_time", "horizon_end", "detection_time", "action_start"):
        worlds[column] = pd.to_datetime(worlds[column], format="mixed")
    decomposition = decompose_policy_value(worlds, str(evaluation["earliest_profile"]))
    repetitions = int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"]))
    deletion_repetitions = int(profile.get("deletion_bootstrap_replicates", evaluation["deletion_bootstrap_replicates"]))
    summary, family = _summarize_values(decomposition, bootstrap_replicates=repetitions, seed=int(evaluation["seed"]))
    slopes = _cluster_slopes(decomposition)
    slope_summary, slope_family = _summarize_slopes(slopes, bootstrap_replicates=repetitions, seed=int(evaluation["seed"]) + 800)
    latest = max(detection_fractions.values())
    decisions = _classify(summary, latest)
    deletion = _leave_one_family_out(
        decomposition,
        latest=latest,
        bootstrap_replicates=deletion_repetitions,
        seed=int(evaluation["seed"]) + 1600,
    )
    evidence = _evidence_summary(worlds)
    method_counts = worlds.groupby(POLICY_KEYS + ["epidemic_model", "detection_fraction"], observed=True)["method"].nunique()
    natural_consistency = worlds.groupby(NATURAL_KEYS + ["epidemic_model"], observed=True)["natural_final_size"].nunique()
    standard_consistency = worlds.groupby(POLICY_KEYS + ["epidemic_model", "detection_fraction"], observed=True)["standard_final_size"].nunique()
    earliest_mask = decomposition["detection_profile"].eq(str(evaluation["earliest_profile"]))
    checks = {
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "all_requested_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "five_independent_families_full": profile_name != "full" or worlds["system_family"].nunique() == 5,
        "all_detection_profiles_present": set(worlds["detection_profile"]) == set(detection_fractions),
        "paired_methods_complete": bool(method_counts.eq(2).all()),
        "natural_world_shared_across_times": bool(natural_consistency.eq(1).all()),
        "standard_care_shared_within_time": bool(standard_consistency.eq(1).all()),
        "decomposition_identity": bool(np.allclose(decomposition["net_wait_value"], decomposition["information_gain"] - decomposition["delay_cost"])),
        "earliest_delay_cost_zero": bool(np.allclose(decomposition.loc[earliest_mask, "delay_cost"], 0)),
        "finite_values": bool(np.isfinite(decomposition[VALUE_METRICS].to_numpy(float)).all()),
        "whole_family_deletion_complete": len(deletion) == worlds["system_family"].nunique() * worlds["epidemic_model"].nunique(),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "datasets": int(worlds["dataset_id"].nunique()),
        "families": int(worlds["system_family"].nunique()),
        "anchors": int(worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0]),
        "natural_worlds": int(worlds[NATURAL_KEYS + ["epidemic_model"]].drop_duplicates().shape[0]),
        "policy_evaluations": len(worlds),
        "decision_counts": decisions["decision"].value_counts().to_dict(),
        "scope": "model_based_value_of_information_not_field_causal_validation",
    }
    if audit["status"] != "pass":
        raise ValueError(f"information-accrual audit failed: {audit}")
    outputs = {
        "policy_worlds.csv.gz": (worlds, {"index": False, "compression": "gzip"}),
        "value_decomposition.csv.gz": (decomposition, {"index": False, "compression": "gzip"}),
        "value_curve_summary.csv": (summary, {"index": False}),
        "family_value_curves.csv": (family, {"index": False}),
        "cluster_slopes.csv": (slopes, {"index": False}),
        "slope_summary.csv": (slope_summary, {"index": False}),
        "family_slopes.csv": (slope_family, {"index": False}),
        "policy_decisions.csv": (decisions, {"index": False}),
        "leave_one_family_out_decisions.csv": (deletion, {"index": False}),
        "evidence_maturity_summary.csv": (evidence, {"index": False}),
        "parameter_time_support.csv": (pd.DataFrame(support_rows), {"index": False}),
    }
    for name, (frame, options) in outputs.items():
        frame.to_csv(results_dir / name, **options)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [config_path, stable_path, prerequisite_path, Path(__file__)]
    pd.DataFrame([{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in source_paths]).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dpi = int(profile["render_dpi"])
    _plot_curves(summary, report_dir / "information_delay_decomposition.png", dpi)
    _plot_family_frontier(family, latest, report_dir / "family_break_even_frontier.png", dpi)
    _plot_evidence(evidence, report_dir / "evidence_maturity.png", dpi)
    display = decisions.copy()
    for column in ["information_gain", "information_gain_ci_low", "information_gain_ci_high", "net_wait_value", "net_wait_value_ci_low", "net_wait_value_ci_high", "reactive_absolute_value", "reactive_absolute_ci_low", "reactive_absolute_ci_high"]:
        display[column] = 100 * display[column]
    report = f"""# Outbreak information-accrual value

This preregistered repeated-measures experiment separates same-time target-selection
information from the opportunity cost of waiting. It uses unchanged epidemic
parameters, policy budget, isolation delivery, observation sensitivity, and
history-weight baseline.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Paired natural worlds: {audit['natural_worlds']}
- Policy evaluations: {audit['policy_evaluations']}
- Technical audit: **{audit['status']}**

## Preregistered latest-versus-earliest decision

All numeric values below are attack-rate percentage points.

{_markdown_table(display)}

`information_gain` compares reactive and history targeting at the same action
time. `delay_cost` measures what history targeting loses by moving from the
earliest to the later action time. Their difference is `net_wait_value`. An
operational upgrade additionally requires reactive targeting to beat same-time
standard care under the frozen absolute-safety gate. Monte Carlo worlds and
anchors are nested evidence; animal-system family is the top-level replication
unit. These are model-based intervention results, not field causal effects.
"""
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the outbreak information-accrual value experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260818-006_information_accrual_value.yaml"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
