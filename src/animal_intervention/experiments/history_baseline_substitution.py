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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.evaluation import stable_hash_order
from animal_intervention.simulation import DetectionProfile, detection_time_from_seed

from .contact_observation_robustness import WORLD_KEYS as OBSERVATION_WORLD_KEYS
from .contact_observation_robustness import _run_window as _run_observation_window
from .intervention_delivery_sensitivity import (
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


METHOD_LABELS = {
    "history_weight": "History weight",
    "contact_to_detected": "Detected-case contacts",
}
MODEL_LABELS = {
    "temporal_sir": "Temporal SIR",
    "temporal_seir_erlang": "Staged SEIR/Erlang",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for item in frame.itertuples(index=False, name=None):
        values = []
        for value in item:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _pair_factorial(worlds: pd.DataFrame, baseline: str, method: str) -> pd.DataFrame:
    parts = []
    for model, frame in worlds.groupby("epidemic_model", observed=True, sort=True):
        keys = POLICY_KEYS
        reference = frame.loc[
            frame["method"].eq(baseline), keys + ["attack_rate_reduction"]
        ]
        challenger = frame.loc[
            frame["method"].eq(method),
            keys + ["system_family", "analysis_cluster_id", "attack_rate_reduction"],
        ]
        paired = challenger.merge(
            reference,
            on=keys,
            suffixes=("_method", "_baseline"),
            validate="one_to_one",
        )
        paired["epidemic_model"] = str(model)
        paired["method"] = method
        paired["baseline"] = baseline
        paired["increment"] = (
            paired["attack_rate_reduction_method"]
            - paired["attack_rate_reduction_baseline"]
        )
        parts.append(paired)
    return pd.concat(parts, ignore_index=True)


def _pair_observation(
    worlds: pd.DataFrame, baseline: str, method: str
) -> pd.DataFrame:
    keys = OBSERVATION_WORLD_KEYS + ["observation_profile"]
    reference = worlds.loc[
        worlds["method"].eq(baseline), keys + ["attack_rate_reduction"]
    ]
    challenger = worlds.loc[
        worlds["method"].eq(method),
        keys + ["system_family", "analysis_cluster_id", "attack_rate_reduction"],
    ]
    paired = challenger.merge(
        reference,
        on=keys,
        suffixes=("_method", "_baseline"),
        validate="one_to_one",
    )
    paired["method"] = method
    paired["baseline"] = baseline
    paired["increment"] = (
        paired["attack_rate_reduction_method"]
        - paired["attack_rate_reduction_baseline"]
    )
    return paired


def _decision_map(
    relative: pd.DataFrame,
    absolute: pd.DataFrame,
    *,
    method: str,
    baseline: str,
) -> pd.DataFrame:
    keys = ["epidemic_model", "detection_profile", "rewiring_fraction"]
    direct = absolute.loc[absolute["method"].eq(method)].copy()
    history = absolute.loc[absolute["method"].eq(baseline)].copy()
    relative_columns = keys + [
        "family_equal_mean",
        "ci_low",
        "ci_high",
        "positive_families",
        "families",
    ]
    result = relative[relative_columns].rename(
        columns={
            "family_equal_mean": "direct_minus_history",
            "ci_low": "direct_minus_history_ci_low",
            "ci_high": "direct_minus_history_ci_high",
            "positive_families": "direct_minus_history_positive_families",
            "families": "independent_families",
        }
    )
    for prefix, frame in (("direct_absolute", direct), ("history_absolute", history)):
        result = result.merge(
            frame[
                keys
                + ["family_equal_mean", "ci_low", "ci_high", "positive_families"]
            ].rename(
                columns={
                    "family_equal_mean": prefix,
                    "ci_low": f"{prefix}_ci_low",
                    "ci_high": f"{prefix}_ci_high",
                    "positive_families": f"{prefix}_positive_families",
                }
            ),
            on=keys,
            validate="one_to_one",
        )
    required = np.ceil(0.8 * result["independent_families"]).astype(int)
    override = (
        result["direct_minus_history_ci_low"].gt(0)
        & result["direct_absolute_ci_low"].gt(0)
        & result["direct_minus_history_positive_families"].ge(required)
        & result["direct_absolute_positive_families"].ge(required)
    )
    history_supported = (
        ~override
        & result["history_absolute_ci_low"].gt(0)
        & result["history_absolute_positive_families"].ge(required)
    )
    result["decision"] = np.select(
        [override, history_supported],
        ["override_with_detected_case_contacts", "retain_history_weight"],
        default="abstain_or_unresolved",
    )
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _plot_decision_map(frame: pd.DataFrame, path: Path) -> None:
    models = list(MODEL_LABELS)
    columns = [
        ("early_detection", 0.0, "Early\nNo rewiring"),
        ("early_detection", 1.0, "Early\nFull rewiring"),
        ("delayed_detection", 0.0, "Delayed\nNo rewiring"),
        ("delayed_detection", 1.0, "Delayed\nFull rewiring"),
    ]
    values = 100 * frame["direct_minus_history"].to_numpy(float)
    limit = max(float(np.max(np.abs(values))), 0.01)
    matrix = np.full((len(models), len(columns)), np.nan)
    decisions: list[list[str]] = [["" for _ in columns] for _ in models]
    direct_absolute = np.full_like(matrix, np.nan)
    for row_index, model in enumerate(models):
        for column_index, (detection, rewiring, _) in enumerate(columns):
            match = frame.loc[
                frame["epidemic_model"].eq(model)
                & frame["detection_profile"].eq(detection)
                & frame["rewiring_fraction"].eq(rewiring)
            ]
            if len(match) == 1:
                row = match.iloc[0]
                matrix[row_index, column_index] = 100 * float(row.direct_minus_history)
                direct_absolute[row_index, column_index] = 100 * float(row.direct_absolute)
                decisions[row_index][column_index] = str(row.decision)
    fig, axis = plt.subplots(figsize=(12.8, 6.0))
    image = axis.imshow(
        matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    axis.set_xticks(range(len(columns)), [item[2] for item in columns])
    axis.set_yticks(range(len(models)), [MODEL_LABELS[item] for item in models])
    for row in range(len(models)):
        for column in range(len(columns)):
            decision = decisions[row][column]
            short = {
                "override_with_detected_case_contacts": "Override",
                "retain_history_weight": "Retain history",
                "abstain_or_unresolved": "Abstain/unresolved",
            }.get(decision, decision)
            axis.text(
                column,
                row,
                f"Δ direct−history\n{matrix[row, column]:+.2f} pp\n"
                f"Direct absolute\n{direct_absolute[row, column]:+.2f} pp\n{short}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )
    colorbar = fig.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label("Direct minus history weight (percentage points)")
    fig.suptitle(
        "Final policy gate: override history only when relative and absolute benefit agree",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    axis.set_title(
        "Cells are blocked by independent animal-system family; grey decision text follows frozen two-gate rule",
        fontsize=11,
        pad=14,
    )
    fig.subplots_adjust(left=0.18, right=0.90, top=0.80, bottom=0.16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_family_boundary(family: pd.DataFrame, path: Path) -> None:
    primary = family.loc[
        family["detection_profile"].eq("early_detection")
        & family["rewiring_fraction"].eq(1.0)
    ].copy()
    families = sorted(primary["system_family"].unique())
    models = list(MODEL_LABELS)
    matrix = np.full((len(families), len(models)), np.nan)
    for row, family_name in enumerate(families):
        for column, model in enumerate(models):
            values = primary.loc[
                primary["system_family"].eq(family_name)
                & primary["epidemic_model"].eq(model),
                "mean_value",
            ]
            if len(values):
                matrix[row, column] = 100 * float(values.iloc[0])
    limit = max(float(np.nanmax(np.abs(matrix))), 0.01)
    fig, axis = plt.subplots(figsize=(9.5, 6.6))
    image = axis.imshow(
        matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    axis.set_xticks(range(len(models)), [MODEL_LABELS[item] for item in models])
    axis.set_yticks(range(len(families)), [item.replace("_", " ") for item in families])
    for row in range(len(families)):
        for column in range(len(models)):
            axis.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center")
    colorbar = fig.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label("Direct minus history weight (percentage points)")
    fig.suptitle(
        "Independent-family directions in the early/full-rewiring sentinel",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    axis.set_title("Positive favors detected-case contact targeting", fontsize=11, pad=12)
    fig.subplots_adjust(left=0.33, right=0.88, top=0.82, bottom=0.12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_observation_guardrail(
    relative: pd.DataFrame, absolute: pd.DataFrame, path: Path
) -> None:
    rows = []
    for item in relative.itertuples(index=False):
        rows.append(
            {
                "label": f"Direct − history | {item.observation_profile}",
                "estimate": item.family_equal_mean,
                "low": item.ci_low,
                "high": item.ci_high,
                "color": "#E45756",
            }
        )
    for method in ("contact_to_detected", "history_weight"):
        for item in absolute.loc[absolute["method"].eq(method)].itertuples(index=False):
            rows.append(
                {
                    "label": f"{METHOD_LABELS[method]} absolute | {item.observation_profile}",
                    "estimate": item.family_equal_mean,
                    "low": item.ci_low,
                    "high": item.ci_high,
                    "color": "#4C78A8" if method == "history_weight" else "#F58518",
                }
            )
    frame = pd.DataFrame(rows)
    positions = np.arange(len(frame))[::-1]
    fig, axis = plt.subplots(figsize=(11.5, 7.0))
    for position, item in zip(positions, frame.itertuples(index=False)):
        axis.errorbar(
            100 * item.estimate,
            position,
            xerr=[[100 * (item.estimate - item.low)], [100 * (item.high - item.estimate)]],
            fmt="o",
            color=item.color,
            capsize=4,
        )
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1.2)
    axis.set_yticks(positions, frame["label"])
    axis.set_xlabel("Family-equal attack-rate difference (percentage points)")
    axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Observation-loss guardrail for the substituted history baseline",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    axis.set_title(
        "SIR, early detection, full rewiring; intervals preserve animal-system blocks",
        fontsize=11,
        pad=14,
    )
    fig.subplots_adjust(left=0.36, right=0.97, top=0.82, bottom=0.13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile = dict(config["profiles"][profile_name])
    decision = dict(config["decision"])
    evaluation = dict(config["evaluation"])
    stable_path = Path(config["data"]["stable_prediction_path"])
    prerequisite_path = Path(config["data"]["prerequisite_audit"])
    prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    if prerequisite.get("status") != "pass":
        raise ValueError("prerequisite artifact audit must pass")
    stable_predictions = pd.read_csv(
        stable_path, dtype={"candidate_id": str, "network_id": str}
    )
    stable_predictions["anchor_time"] = pd.to_datetime(
        stable_predictions["anchor_time"], format="mixed"
    )
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    detections = [DetectionProfile(**item) for item in decision["detection_profiles"]]
    models = [dict(item) for item in decision["epidemic_models"]]
    stable_by_window: dict[tuple[str, str, str], pd.DataFrame] = {}
    task_specs: list[dict[str, Any]] = []
    observation_specs: list[dict[str, Any]] = []
    support_rows = []
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        available = set(
            stable_predictions.loc[
                stable_predictions["dataset_id"].eq(dataset_id),
                ["network_id", "anchor_time"],
            ].itertuples(index=False, name=None)
        )
        for window in windows:
            window.setdefault("network_id", default_network_id)
        windows = [
            window
            for window in windows
            if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time))
            in available
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
                    (
                        detection_time_from_seed(
                            anchor.anchor_time, anchor.horizon_end, mean_period, detection
                        )
                        + mean_period
                        * float(decision["action_delay_fraction_of_mean_infectious_period"])
                        < anchor.horizon_end
                    )
                    for detection in detections
                )
                support_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "network_id": str(window["network_id"]),
                        "anchor_id": anchor.anchor_id,
                        "parameter_id": parameter.parameter_id,
                        "supported": supported,
                    }
                )
                if supported:
                    compatible.append(parameter)
            selected = _select_parameter_regimes(
                compatible, str(evaluation["parameter_selection_mode"])
            )
            if len(selected) != 1:
                continue
            _, parameter = selected[0]
            network_id = str(window["network_id"])
            cluster = (
                f"{dataset_id}::{network_id}"
                if specification.get("analysis_cluster") == "network"
                else f"{dataset_id}::{network_id}::{anchor.anchor_id}"
            )
            stable = _matching_stable_scores(
                stable_predictions,
                dataset_id,
                network_id,
                anchor.anchor_time,
                window["eligible"],
            )
            stable_by_window[(dataset_id, network_id, anchor.anchor_id)] = stable
            seeds = stable_hash_order(
                list(map(str, window["eligible"])),
                int(evaluation["seed"]),
                dataset_id,
                anchor.anchor_id,
                "history_baseline_substitution_seeds",
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
                    for rewiring in map(float, decision["rewiring_fractions"]):
                        task_specs.append(
                            {
                                **common,
                                "epidemic_model": model,
                                "detection_profile": detection,
                                "rewiring_fraction": rewiring,
                            }
                        )
            if bool(decision["observation_sentinel"]["enabled"]):
                observation_specs.append(common)

    fingerprint = hashlib.sha256(
        config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()[:12]
    factorial_frames = []
    progress = tqdm(task_specs, desc="Baseline-substitution factorial", unit="task")
    for task in progress:
        model = task["epidemic_model"]
        detection = task["detection_profile"]
        identity = "|".join(
            [
                fingerprint,
                task["dataset_id"],
                task["network_id"],
                task["window"]["anchor"].anchor_id,
                str(task["parameter"].parameter_id),
                str(model["name"]),
                detection.name,
                str(task["rewiring_fraction"]),
            ]
        )
        checkpoint = checkpoint_dir / f"factorial_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
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
                action_delay_fraction=float(
                    decision["action_delay_fraction_of_mean_infectious_period"]
                ),
                residual_contact_multiplier=float(decision["residual_contact_multiplier"]),
                stable_scores=task["stable_scores"],
                methods=list(decision["methods"]),
                seed_nodes=task["seed_nodes"],
                random_blocks=int(profile["random_blocks"]),
                minimum_budget=int(decision["minimum_additional_budget"]),
                budget_fraction=float(decision["additional_budget_fraction"]),
                secondary_case_sensitivity=float(decision["secondary_case_sensitivity"]),
                false_positive_rate=float(decision["false_positive_rate"]),
                rewiring_fraction=float(task["rewiring_fraction"]),
                rewiring_mode=(
                    str(decision["rewiring_mode"])
                    if float(task["rewiring_fraction"]) > 0
                    else "none"
                ),
                tracing_half_life_fraction=float(
                    decision["tracing_half_life_fraction_of_mean_infectious_period"]
                ),
                experiment_seed=int(evaluation["seed"]),
                epidemic_model=model,
            )
            frame.to_csv(checkpoint, index=False, compression="gzip")
        factorial_frames.append(frame)
        progress.set_postfix_str(f"{task['dataset_id']} {model['name']} completed")
    factorial = pd.concat(factorial_frames, ignore_index=True)

    observation_frames = []
    sentinel = decision["observation_sentinel"]
    progress = tqdm(observation_specs, desc="Observation sentinel", unit="task")
    observation_decision = {
        "minimum_additional_budget": decision["minimum_additional_budget"],
        "additional_budget_fraction": decision["additional_budget_fraction"],
        "tracing_half_life_fraction_of_mean_infectious_period": decision[
            "tracing_half_life_fraction_of_mean_infectious_period"
        ],
        "secondary_case_sensitivity": decision["secondary_case_sensitivity"],
        "detection_profile": sentinel["detection_profile"],
        "action_delay_fraction_of_mean_infectious_period": decision[
            "action_delay_fraction_of_mean_infectious_period"
        ],
        "residual_contact_multiplier": decision["residual_contact_multiplier"],
        "rewiring_fraction": sentinel["rewiring_fraction"],
        "rewiring_mode": decision["rewiring_mode"],
    }
    for task in progress:
        identity = "|".join(
            [
                fingerprint,
                "observation",
                task["dataset_id"],
                task["network_id"],
                task["window"]["anchor"].anchor_id,
                str(task["parameter"].parameter_id),
            ]
        )
        checkpoint = checkpoint_dir / f"observation_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
        frame = pd.DataFrame()
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str})
        if frame.empty:
            frame = _run_observation_window(
                dataset_id=task["dataset_id"],
                network_id=task["network_id"],
                system_family=task["system_family"],
                analysis_cluster_id=task["analysis_cluster_id"],
                window=task["window"],
                parameter=task["parameter"],
                stable_scores=task["stable_scores"],
                profile_specs=list(sentinel["profiles"]),
                methods=list(decision["methods"]),
                seed_nodes=task["seed_nodes"],
                random_blocks=int(profile["random_blocks"]),
                decision=observation_decision,
                experiment_seed=int(evaluation["seed"]),
            )
            frame.to_csv(checkpoint, index=False, compression="gzip")
        observation_frames.append(frame)
        progress.set_postfix_str(f"{task['dataset_id']} completed")
    observation = pd.concat(observation_frames, ignore_index=True)

    baseline = str(evaluation["primary_baseline"])
    method = str(evaluation["primary_method"])
    paired = _pair_factorial(factorial, baseline, method)
    observation_paired = _pair_observation(observation, baseline, method)
    repetitions = int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"]))
    factorial_groups = ["epidemic_model", "detection_profile", "rewiring_fraction"]
    absolute, absolute_family = _hierarchical_summary(
        factorial,
        value_column="attack_rate_reduction",
        group_columns=factorial_groups + ["method"],
        bootstrap_replicates=repetitions,
        seed=int(evaluation["seed"]),
    )
    relative, relative_family = _hierarchical_summary(
        paired,
        value_column="increment",
        group_columns=factorial_groups,
        bootstrap_replicates=repetitions,
        seed=int(evaluation["seed"]) + 1,
    )
    observation_absolute, observation_absolute_family = _hierarchical_summary(
        observation,
        value_column="attack_rate_reduction",
        group_columns=["observation_profile", "method"],
        bootstrap_replicates=repetitions,
        seed=int(evaluation["seed"]) + 2,
    )
    observation_relative, observation_relative_family = _hierarchical_summary(
        observation_paired,
        value_column="increment",
        group_columns=["observation_profile"],
        bootstrap_replicates=repetitions,
        seed=int(evaluation["seed"]) + 3,
    )
    decisions = _decision_map(relative, absolute, method=method, baseline=baseline)

    factorial_keys = POLICY_KEYS + ["epidemic_model", "method"]
    method_counts = factorial.groupby(
        POLICY_KEYS + ["epidemic_model"], observed=True
    )["method"].nunique()
    cell_counts = (
        factorial[[
            "dataset_id",
            "network_id",
            "anchor_id",
            "epidemic_model",
            "detection_profile",
            "rewiring_fraction",
        ]]
        .drop_duplicates()
        .groupby(["dataset_id", "network_id", "anchor_id"], observed=True)
        .size()
    )
    observation_counts = observation.groupby(
        OBSERVATION_WORLD_KEYS + ["observation_profile"], observed=True
    )["method"].nunique()
    expected_families = {
        str(config["data"]["datasets"][item]["system_family"])
        for item in profile["datasets"]
    }
    audit = {
        "status": "pass",
        "checks": {
            "prerequisite_artifact_passed": prerequisite.get("status") == "pass",
            "factorial_keys_unique": not factorial.duplicated(factorial_keys).any(),
            "all_factorial_methods_complete": bool(method_counts.eq(len(decision["methods"])).all()),
            "eight_core_cells_per_anchor": bool(cell_counts.eq(8).all()),
            "observation_methods_complete": bool(observation_counts.eq(len(decision["methods"])).all()),
            "two_observation_profiles_complete": bool(
                observation.groupby(OBSERVATION_WORLD_KEYS, observed=True)[
                    "observation_profile"
                ].nunique().eq(2).all()
            ),
            "all_configured_datasets_retained": set(factorial["dataset_id"].unique())
            == set(profile["datasets"]),
            "all_configured_families_retained": set(factorial["system_family"].unique())
            == expected_families,
            "paired_rows_reconcile": len(paired)
            == len(factorial.loc[factorial["method"].eq(method)]),
            "fixed_factorial_budget": bool(
                factorial["additional_targets"].fillna("").map(
                    lambda value: 0 if not value else len(str(value).split("|"))
                ).eq(factorial["additional_budget"]).all()
            ),
            "fixed_observation_budget": bool(
                observation["additional_targets"].fillna("").map(
                    lambda value: 0 if not value else len(str(value).split("|"))
                ).eq(observation["realized_budget"]).all()
            ),
            "finite_outcomes": bool(
                np.isfinite(factorial["attack_rate_reduction"].to_numpy(float)).all()
                and np.isfinite(observation["attack_rate_reduction"].to_numpy(float)).all()
            ),
            "decision_map_complete": len(decisions) == 8,
        },
        "datasets": int(factorial["dataset_id"].nunique()),
        "system_families": int(factorial["system_family"].nunique()),
        "anchors": int(factorial[["dataset_id", "anchor_id"]].drop_duplicates().shape[0]),
        "core_cells": 8,
        "factorial_base_world_cells": int(
            factorial[POLICY_KEYS + ["epidemic_model"]].drop_duplicates().shape[0]
        ),
        "factorial_policy_evaluations": len(factorial),
        "observation_policy_evaluations": len(observation),
        "decision_counts": decisions["decision"].value_counts().to_dict(),
        "scientific_gate": {
            "status": "pass"
            if decisions["decision"].eq("override_with_detected_case_contacts").any()
            else "fail",
            "interpretation": "at_least_one_deployment_cell_supports_override"
            if decisions["decision"].eq("override_with_detected_case_contacts").any()
            else "no_cell_supports_overriding_history_weight_under_the_frozen_rule",
        },
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"history-baseline substitution audit failed: {audit}")

    factorial.to_csv(results_dir / "factorial_policy_worlds.csv.gz", index=False, compression="gzip")
    paired.to_csv(results_dir / "factorial_direct_over_history.csv.gz", index=False, compression="gzip")
    absolute.to_csv(results_dir / "factorial_absolute_summary.csv", index=False)
    absolute_family.to_csv(results_dir / "factorial_absolute_family.csv", index=False)
    relative.to_csv(results_dir / "factorial_relative_summary.csv", index=False)
    relative_family.to_csv(results_dir / "factorial_relative_family.csv", index=False)
    observation.to_csv(results_dir / "observation_sentinel_worlds.csv.gz", index=False, compression="gzip")
    observation_absolute.to_csv(results_dir / "observation_absolute_summary.csv", index=False)
    observation_absolute_family.to_csv(results_dir / "observation_absolute_family.csv", index=False)
    observation_relative.to_csv(results_dir / "observation_relative_summary.csv", index=False)
    observation_relative_family.to_csv(results_dir / "observation_relative_family.csv", index=False)
    decisions.to_csv(results_dir / "policy_decision_map.csv", index=False)
    pd.DataFrame(support_rows).to_csv(results_dir / "parameter_support.csv", index=False)
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
    _plot_decision_map(decisions, report_dir / "policy_decision_map.png")
    _plot_family_boundary(relative_family, report_dir / "primary_family_boundary.png")
    _plot_observation_guardrail(
        observation_relative,
        observation_absolute,
        report_dir / "observation_guardrail.png",
    )
    decision_table = decisions.copy()
    for column in [
        "direct_minus_history",
        "direct_minus_history_ci_low",
        "direct_minus_history_ci_high",
        "direct_absolute",
        "direct_absolute_ci_low",
        "direct_absolute_ci_high",
        "history_absolute",
        "history_absolute_ci_low",
        "history_absolute_ci_high",
    ]:
        decision_table[column] = 100 * decision_table[column]
    readme = f"""# Final history-baseline substitution gate

This experiment compares detected-case contact targeting with a transparent history-weight preparedness list and with no additional targeting. The eight core cells cross two epidemic models, two detection timings, and no versus full compensatory rewiring. The observation sentinel is a guardrail, not an extra pass opportunity.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['system_families']}
- Anchors: {audit['anchors']}
- Factorial policy evaluations: {audit['factorial_policy_evaluations']}
- Observation-sentinel policy evaluations: {audit['observation_policy_evaluations']}
- Technical audit: **{audit['status']}**
- Scientific override gate: **{audit['scientific_gate']['status']}**

The scientific gate is intentionally stricter than choosing the largest point estimate. An override requires a positive interval lower bound and at least four of five positive family means for both direct-minus-history and direct-minus-no-extra comparisons. Otherwise the cell retains history only when its absolute benefit passes the analogous rule; all remaining cells are abstain/unresolved.

The outcomes remain model-based simulation targets, not field estimates for a named pathogen.

## Decision map

{_markdown_table(decision_table)}
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final history-baseline substitution gate")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260817-006_history_baseline_substitution.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
