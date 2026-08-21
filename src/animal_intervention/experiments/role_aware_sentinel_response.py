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
from animal_intervention.simulation import (
    InterventionAction,
    pre_detection_event_signature,
    select_additional_targets,
)
from animal_intervention.surveillance import greedy_history_coverage

from .history_baseline_substitution import _markdown_table
from .intervention_delivery_sensitivity import (
    SYSTEM_FAMILY_LABELS,
    _hierarchical_summary,
    _parameter_pool,
    _select_parameter_regimes,
)
from .outbreak_response_pilot import (
    _git_value,
    _keyed_seed,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _sha256,
)
from .sequential_preparedness_update import _budget, _history_score_table, _parameters


SENTINEL_METHODS = ["random", "history_weight", "history_coverage"]
RESPONSE_METHODS = ["case_only", "history_weight"]
POLICIES = [
    f"{sentinel}__{response}"
    for sentinel in SENTINEL_METHODS
    for response in RESPONSE_METHODS
] + ["full_surveillance__history_weight"]
CONTRASTS = {
    "history_sentinel_value": (
        "random__history_weight",
        "history_weight__history_weight",
    ),
    "role_separation_value": (
        "history_weight__history_weight",
        "history_coverage__history_weight",
    ),
    "response_value_under_coverage": (
        "history_coverage__case_only",
        "history_coverage__history_weight",
    ),
    "response_value_under_random": (
        "random__case_only",
        "random__history_weight",
    ),
    "response_value_under_history": (
        "history_weight__case_only",
        "history_weight__history_weight",
    ),
    "full_surveillance_regret": (
        "history_coverage__history_weight",
        "full_surveillance__history_weight",
    ),
}
WORLD_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "epidemic_model",
    "random_block",
    "initial_infected",
    "world_seed",
]


def _top_history(
    stable_scores: pd.DataFrame,
    eligible: set[str],
    budget: int,
    seed: int,
    excluded: set[str] | None = None,
) -> set[str]:
    table = _history_score_table(stable_scores)
    return set(
        select_additional_targets(
            table.loc[table["candidate_id"].isin(eligible)],
            method="history_weight",
            budget=budget,
            detected_nodes=excluded or set(),
            world_seed=seed,
        )
    )


def _infection_events(result: Any) -> pd.DataFrame:
    events = result.event_log.loc[
        result.event_log["event"].isin(["initial_infection", "infection"]),
        ["time", "node_id"],
    ].copy()
    events["time"] = pd.to_datetime(events["time"], format="mixed")
    events["node_id"] = events["node_id"].astype(str)
    return events.sort_values(["time", "node_id"], kind="stable", ignore_index=True)


def _detection_metrics(
    natural: Any,
    sentinels: set[str],
    population_size: int,
    threshold_fraction: float,
) -> dict[str, Any]:
    infections = _infection_events(natural)
    sentinel_events = infections.loc[infections["node_id"].isin(sentinels)]
    if sentinel_events.empty:
        burden = int(natural.final_size)
        return {
            "detected": False,
            "detection_time": pd.NaT,
            "detected_nodes": set(),
            "detection_burden": burden,
            "detection_burden_rate": burden / population_size,
            "early_detection": False,
        }
    detection_time = pd.Timestamp(sentinel_events.iloc[0]["time"])
    detected_nodes = set(
        sentinel_events.loc[sentinel_events["time"].le(detection_time), "node_id"]
    )
    burden = int(infections.loc[infections["time"].le(detection_time), "node_id"].nunique())
    threshold = max(1, int(math.ceil(population_size * threshold_fraction)))
    return {
        "detected": True,
        "detection_time": detection_time,
        "detected_nodes": detected_nodes,
        "detection_burden": burden,
        "detection_burden_rate": burden / population_size,
        "early_detection": burden <= threshold,
    }


def _replay_response(
    *,
    engine: Any,
    parameters: Any,
    future: Any,
    natural: Any,
    initial: str,
    world_seed: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    detection_time: pd.Timestamp | None,
    action_delay: pd.Timedelta,
    targets: set[str],
    residual: float,
) -> tuple[Any, pd.Timestamp | None]:
    if detection_time is None or pd.isna(detection_time):
        return natural, None
    action_start = pd.Timestamp(detection_time) + action_delay
    if action_start >= end_time or not targets:
        return natural, action_start
    action = InterventionAction(
        name="sentinel_triggered_isolation",
        action_type="isolation",
        target_nodes=tuple(sorted(targets)),
        start_time=action_start,
        end_time=end_time,
        contact_multiplier=residual,
    )
    result = engine.simulate(
        future,
        parameters,
        initial_infected=(initial,),
        start_time=start_time,
        end_time=end_time,
        world_seed=world_seed,
        action=action,
    )
    if pre_detection_event_signature(result, pd.Timestamp(detection_time)) != pre_detection_event_signature(
        natural, pd.Timestamp(detection_time)
    ):
        raise AssertionError("sentinel response changed events before its trigger")
    return result, action_start


def _run_world(
    *,
    dataset_id: str,
    network_id: str,
    system_family: str,
    analysis_cluster_id: str,
    window: dict[str, Any],
    parameter: Any,
    model: dict[str, Any],
    stable_scores: pd.DataFrame,
    initial: str,
    random_block: int,
    experiment_seed: int,
    sentinel_fraction: float,
    response_fraction: float,
    minimum_budget: int,
    action_delay_fraction: float,
    residual: float,
    threshold_fraction: float,
) -> pd.DataFrame:
    anchor = window["anchor"]
    start_time = pd.Timestamp(anchor.anchor_time)
    end_time = pd.Timestamp(anchor.horizon_end)
    mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
    action_delay = mean_period * action_delay_fraction
    engine, parameters = _parameters(parameter, model, mean_period)
    world_seed = _keyed_seed(
        experiment_seed,
        dataset_id,
        anchor.anchor_id,
        parameter.parameter_id,
        random_block,
        initial,
    )
    natural = engine.simulate(
        window["future"],
        parameters,
        initial_infected=(initial,),
        start_time=start_time,
        end_time=end_time,
        world_seed=world_seed,
    )
    eligible = set(map(str, window["eligible"]))
    population_size = len(window["future"].nodes())
    sentinel_budget = _budget(len(eligible), minimum_budget, sentinel_fraction)
    sentinel_seed = _keyed_seed(
        experiment_seed,
        dataset_id,
        anchor.anchor_id,
        "sentinel_set",
        random_block,
    )
    sentinel_sets = {
        "random": set(stable_hash_order(sorted(eligible), sentinel_seed, "random_sentinels")[:sentinel_budget]),
        "history_weight": _top_history(stable_scores, eligible, sentinel_budget, sentinel_seed),
        "history_coverage": set(
            greedy_history_coverage(
                window["history"], eligible, sentinel_budget, seed=sentinel_seed
            )
        ),
    }
    if any(len(nodes) != sentinel_budget for nodes in sentinel_sets.values()):
        raise AssertionError("sentinel methods must use equal capacity")

    response_capacity = _budget(len(eligible), minimum_budget, response_fraction)
    rows = []
    for sentinel_method, sentinels in sentinel_sets.items():
        detection = _detection_metrics(
            natural, sentinels, population_size, threshold_fraction
        )
        detected_nodes = set(detection["detected_nodes"])
        response_budget = min(response_capacity, len(eligible - detected_nodes))
        additional = _top_history(
            stable_scores,
            eligible,
            response_budget,
            sentinel_seed,
            excluded=detected_nodes,
        )
        for response_method in RESPONSE_METHODS:
            response_targets = detected_nodes | (
                additional if response_method == "history_weight" else set()
            )
            result, action_start = _replay_response(
                engine=engine,
                parameters=parameters,
                future=window["future"],
                natural=natural,
                initial=initial,
                world_seed=world_seed,
                start_time=start_time,
                end_time=end_time,
                detection_time=detection["detection_time"],
                action_delay=action_delay,
                targets=response_targets,
                residual=residual,
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "network_id": network_id,
                    "system_family": system_family,
                    "analysis_cluster_id": analysis_cluster_id,
                    "anchor_id": anchor.anchor_id,
                    "anchor_time": start_time,
                    "horizon_end": end_time,
                    "parameter_id": parameter.parameter_id,
                    "beta": float(parameter.beta),
                    "mean_infectious_period_days": float(parameter.mean_infectious_period_days),
                    "epidemic_model": str(model["name"]),
                    "random_block": random_block,
                    "initial_infected": str(initial),
                    "world_seed": world_seed,
                    "population_size": population_size,
                    "policy": f"{sentinel_method}__{response_method}",
                    "sentinel_method": sentinel_method,
                    "response_method": response_method,
                    "sentinel_budget": sentinel_budget,
                    "response_budget": response_budget if response_method == "history_weight" else 0,
                    "response_capacity": response_capacity,
                    "sentinel_nodes": "|".join(sorted(sentinels)),
                    "response_nodes": "|".join(sorted(response_targets)),
                    "detected": bool(detection["detected"]),
                    "detection_time": detection["detection_time"],
                    "action_start": action_start,
                    "detection_burden": int(detection["detection_burden"]),
                    "detection_burden_rate": float(detection["detection_burden_rate"]),
                    "early_detection": bool(detection["early_detection"]),
                    "natural_final_size": natural.final_size,
                    "final_size": result.final_size,
                    "final_attack_rate": result.final_size / population_size,
                }
            )

    full_detection = {
        "detected": True,
        "detection_time": start_time,
        "detected_nodes": {str(initial)},
        "detection_burden": 1,
        "detection_burden_rate": 1 / population_size,
        "early_detection": True,
    }
    full_budget = min(response_capacity, len(eligible - {str(initial)}))
    full_additional = _top_history(
        stable_scores,
        eligible,
        full_budget,
        sentinel_seed,
        excluded={str(initial)},
    )
    full_result, full_action = _replay_response(
        engine=engine,
        parameters=parameters,
        future=window["future"],
        natural=natural,
        initial=initial,
        world_seed=world_seed,
        start_time=start_time,
        end_time=end_time,
        detection_time=start_time,
        action_delay=action_delay,
        targets={str(initial)} | full_additional,
        residual=residual,
    )
    rows.append(
        {
            "dataset_id": dataset_id,
            "network_id": network_id,
            "system_family": system_family,
            "analysis_cluster_id": analysis_cluster_id,
            "anchor_id": anchor.anchor_id,
            "anchor_time": start_time,
            "horizon_end": end_time,
            "parameter_id": parameter.parameter_id,
            "beta": float(parameter.beta),
            "mean_infectious_period_days": float(parameter.mean_infectious_period_days),
            "epidemic_model": str(model["name"]),
            "random_block": random_block,
            "initial_infected": str(initial),
            "world_seed": world_seed,
            "population_size": population_size,
            "policy": "full_surveillance__history_weight",
            "sentinel_method": "full_surveillance",
            "response_method": "history_weight",
            "sentinel_budget": population_size,
            "response_budget": full_budget,
            "response_capacity": response_capacity,
            "sentinel_nodes": "ALL",
            "response_nodes": "|".join(sorted({str(initial)} | full_additional)),
            "detected": True,
            "detection_time": start_time,
            "action_start": full_action,
            "detection_burden": 1,
            "detection_burden_rate": 1 / population_size,
            "early_detection": True,
            "natural_final_size": natural.final_size,
            "final_size": full_result.final_size,
            "final_attack_rate": full_result.final_size / population_size,
        }
    )
    return pd.DataFrame(rows)


def compute_contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    metadata = worlds.drop_duplicates(WORLD_KEYS)[
        WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]
    ]
    wide = worlds.pivot(index=WORLD_KEYS, columns="policy", values="final_size").reset_index()
    paired = metadata.merge(wide, on=WORLD_KEYS, validate="one_to_one")
    rows = []
    for name, (reference, challenger) in CONTRASTS.items():
        frame = paired[WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]].copy()
        frame["contrast"] = name
        frame["value"] = (paired[reference] - paired[challenger]) / paired["population_size"].astype(float)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def detection_contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    metrics = worlds.loc[
        worlds["response_method"].eq("case_only"),
        WORLD_KEYS + ["system_family", "analysis_cluster_id", "sentinel_method", "detection_burden_rate"],
    ]
    wide = metrics.pivot(index=WORLD_KEYS, columns="sentinel_method", values="detection_burden_rate").reset_index()
    metadata = metrics.drop_duplicates(WORLD_KEYS)[WORLD_KEYS + ["system_family", "analysis_cluster_id"]]
    paired = metadata.merge(wide, on=WORLD_KEYS, validate="one_to_one")
    return pd.concat(
        [
            paired[WORLD_KEYS + ["system_family", "analysis_cluster_id"]].assign(
                contrast="history_detection_value",
                value=paired["random"] - paired["history_weight"],
            ),
            paired[WORLD_KEYS + ["system_family", "analysis_cluster_id"]].assign(
                contrast="coverage_detection_value",
                value=paired["history_weight"] - paired["history_coverage"],
            ),
        ],
        ignore_index=True,
    )


def _summaries(worlds: pd.DataFrame, contrasts: pd.DataFrame, detection: pd.DataFrame, *, bootstrap_replicates: int, seed: int) -> tuple[pd.DataFrame, ...]:
    summary, family = _hierarchical_summary(
        contrasts,
        value_column="value",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    detection_summary, detection_family = _hierarchical_summary(
        detection,
        value_column="value",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 300,
    )
    rates = worlds.copy()
    rates["value"] = rates["final_attack_rate"]
    policy_summary, policy_family = _hierarchical_summary(
        rates,
        value_column="value",
        group_columns=["epidemic_model", "policy"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 600,
    )
    return summary, family, detection_summary, detection_family, policy_summary, policy_family


def _classify(summary: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = ["history_sentinel_value", "role_separation_value", "response_value_under_coverage"]
    for model, frame in summary.groupby("epidemic_model", observed=True, sort=True):
        metrics = frame.set_index("contrast")
        family_frame = family.loc[family["epidemic_model"].eq(model)]
        families = int(frame["families"].max())
        required = int(math.ceil(0.8 * families))
        passes = {}
        directions = {}
        for metric in primary:
            directions[metric] = int(
                family_frame.loc[family_frame["contrast"].eq(metric), "mean_value"].gt(0).sum()
            )
            passes[metric] = float(metrics.loc[metric, "ci_low"]) > 0 and directions[metric] >= required
        if all(passes.values()):
            decision = "role_separated"
        elif passes["history_sentinel_value"] and passes["response_value_under_coverage"]:
            decision = "shared_history_list_sufficient"
        elif passes["history_sentinel_value"]:
            decision = "surveillance_helpful_response_uncertain"
        else:
            decision = "retain_exogenous_detection_scope_or_abstain"
        row = {"epidemic_model": model, "families": families, "required_families": required, "decision": decision}
        for metric in primary:
            result = metrics.loc[metric]
            row.update(
                {
                    metric: float(result.family_equal_mean),
                    f"{metric}_ci_low": float(result.ci_low),
                    f"{metric}_ci_high": float(result.ci_high),
                    f"{metric}_positive_families": directions[metric],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _leave_one_family_out(worlds: pd.DataFrame, contrasts: pd.DataFrame, detection: pd.DataFrame, *, bootstrap_replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for index, held_out in enumerate(sorted(worlds["system_family"].unique())):
        subset_worlds = worlds.loc[~worlds["system_family"].eq(held_out)]
        subset_contrasts = contrasts.loc[~contrasts["system_family"].eq(held_out)]
        subset_detection = detection.loc[~detection["system_family"].eq(held_out)]
        summary, family, detection_summary, detection_family, *_ = _summaries(
            subset_worlds,
            subset_contrasts,
            subset_detection,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index * 1000,
        )
        decision = _classify(summary, family)
        for record in detection_summary.itertuples(index=False):
            metric = str(record.contrast)
            positive_families = int(
                detection_family.loc[
                    (detection_family["epidemic_model"].eq(record.epidemic_model))
                    & (detection_family["contrast"].eq(metric)),
                    "mean_value",
                ].gt(0).sum()
            )
            mask = decision["epidemic_model"].eq(record.epidemic_model)
            decision.loc[mask, metric] = float(record.family_equal_mean)
            decision.loc[mask, f"{metric}_ci_low"] = float(record.ci_low)
            decision.loc[mask, f"{metric}_ci_high"] = float(record.ci_high)
            decision.loc[mask, f"{metric}_positive_families"] = positive_families
        decision.insert(0, "held_out_family", held_out)
        rows.append(decision)
    return pd.concat(rows, ignore_index=True)


def _plot_contrasts(summary: pd.DataFrame, detection_summary: pd.DataFrame, path: Path, dpi: int) -> None:
    combined = pd.concat(
        [
            summary.loc[summary["contrast"].isin(list(CONTRASTS)[:3])],
            detection_summary,
        ],
        ignore_index=True,
    )
    order = ["history_detection_value", "coverage_detection_value", "history_sentinel_value", "role_separation_value", "response_value_under_coverage"]
    labels = {
        "history_detection_value": "History sentinels: detection burden",
        "coverage_detection_value": "Coverage sentinels: detection burden",
        "history_sentinel_value": "History sentinels: final infections",
        "role_separation_value": "Coverage vs top-history sentinels",
        "response_value_under_coverage": "Post-detection response value",
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    for axis, model, title in zip(axes, ["temporal_sir", "temporal_seir_erlang"], ["Temporal SIR", "Staged SEIR/Erlang"]):
        frame = combined.loc[combined["epidemic_model"].eq(model)].set_index("contrast").loc[order]
        y = np.arange(len(order))
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        axis.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="o", capsize=4, color="#4C78A8")
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_yticks(y, [labels[item] for item in order])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Do surveillance and intervention require different animal lists?", fontsize=21, weight="bold")
    fig.supxlabel("Family-equal improvement (percentage points)", fontsize=14)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.84, bottom=0.12, wspace=0.10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_policy_rates(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    labels = {
        "random__case_only": "Random sentinels; cases only",
        "random__history_weight": "Random sentinels; history response",
        "history_weight__case_only": "Top-history sentinels; cases only",
        "history_weight__history_weight": "Top-history sentinels; history response",
        "history_coverage__case_only": "Coverage sentinels; cases only",
        "history_coverage__history_weight": "Coverage sentinels; history response",
        "full_surveillance__history_weight": "Full-surveillance ceiling",
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    for axis, model, title in zip(axes, ["temporal_sir", "temporal_seir_erlang"], ["Temporal SIR", "Staged SEIR/Erlang"]):
        frame = summary.loc[summary["epidemic_model"].eq(model)].set_index("policy").loc[POLICIES]
        y = np.arange(len(POLICIES))
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        axis.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="o", capsize=4, color="#F58518")
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_yticks(y, [labels[item] for item in POLICIES])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Endogenous detection and response outcomes", fontsize=22, weight="bold")
    fig.supxlabel("Final attack rate (%)", fontsize=14)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.82, bottom=0.12, wspace=0.28)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_family_frontier(family: pd.DataFrame, detection_family: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    families = sorted(family["system_family"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(families)))
    for axis, model, title in zip(axes, ["temporal_sir", "temporal_seir_erlang"], ["Temporal SIR", "Staged SEIR/Erlang"]):
        effects = family.loc[(family["epidemic_model"].eq(model)) & (family["contrast"].eq("history_sentinel_value"))].set_index("system_family")["mean_value"]
        detections = detection_family.loc[(detection_family["epidemic_model"].eq(model)) & (detection_family["contrast"].eq("history_detection_value"))].set_index("system_family")["mean_value"]
        for color, system in zip(colors, families):
            axis.scatter(100 * detections.loc[system], 100 * effects.loc[system], s=85, color=color, label=SYSTEM_FAMILY_LABELS.get(system, system.replace("_", " ")))
        axis.axhline(0, color="#666666", linestyle="--", linewidth=1)
        axis.axvline(0, color="#666666", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Final attack-rate improvement (points)", fontsize=13)
    fig.supxlabel("Detection-burden improvement (points)", fontsize=13)
    axes[1].legend(loc="best", fontsize=9, frameon=False)
    fig.suptitle("Does earlier sentinel detection translate into epidemic control?", fontsize=20, weight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.15, wspace=0.12)
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
        raise ValueError("role-aware surveillance prerequisite audit must pass")
    stable_path = Path(config["data"]["stable_prediction_path"])
    stable_predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    stable_predictions["anchor_time"] = pd.to_datetime(stable_predictions["anchor_time"], format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = []
    support_rows = []
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        available = set(stable_predictions.loc[stable_predictions["dataset_id"].eq(dataset_id), ["network_id", "anchor_time"]].itertuples(index=False, name=None))
        for window in windows:
            window.setdefault("network_id", default_network_id)
        windows = [window for window in windows if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time)) in available]
        maximum = profile.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        parameters = _parameter_pool(Path(specification["source_results"]) / "parameter_selection.csv", str(evaluation["parameter_pool"]))
        for window in windows:
            anchor = window["anchor"]
            compatible = []
            for parameter in parameters.itertuples(index=False):
                mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
                supported = pd.Timestamp(anchor.anchor_time) + mean_period * float(decision["action_delay_fraction_of_mean_infectious_period"]) < pd.Timestamp(anchor.horizon_end)
                support_rows.append({"dataset_id": dataset_id, "network_id": str(window["network_id"]), "anchor_id": anchor.anchor_id, "parameter_id": parameter.parameter_id, "supported": supported})
                if supported:
                    compatible.append(parameter)
            selected = _select_parameter_regimes(compatible, str(evaluation["parameter_selection_mode"]))
            if len(selected) != 1:
                continue
            _, parameter = selected[0]
            network_id = str(window["network_id"])
            cluster = f"{dataset_id}::{network_id}" if specification.get("analysis_cluster") == "network" else f"{dataset_id}::{network_id}::{anchor.anchor_id}"
            stable = _matching_stable_scores(stable_predictions, dataset_id, network_id, anchor.anchor_time, window["eligible"])
            seeds = stable_hash_order(list(map(str, window["eligible"])), int(evaluation["seed"]), dataset_id, anchor.anchor_id, "role_aware_seeds")[: int(profile["seeds_per_anchor"])]
            for model in decision["epidemic_models"]:
                tasks.append({"dataset_id": dataset_id, "network_id": network_id, "system_family": str(specification["system_family"]), "analysis_cluster_id": cluster, "window": window, "parameter": parameter, "model": dict(model), "stable_scores": stable, "seeds": seeds})

    helper_path = Path(__file__).parents[1] / "surveillance" / "sentinels.py"
    fingerprint = hashlib.sha256(config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes() + helper_path.read_bytes()).hexdigest()[:12]
    frames = []
    progress = tqdm(tasks, desc="Role-aware sentinel worlds", unit="task")
    for task in progress:
        identity = "|".join([fingerprint, task["dataset_id"], task["network_id"], task["window"]["anchor"].anchor_id, str(task["parameter"].parameter_id), str(task["model"]["name"])])
        checkpoint = checkpoint_dir / f"worlds_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
        frame = pd.DataFrame()
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str})
        if frame.empty:
            task_frames = []
            for block in range(int(profile["random_blocks"])):
                for initial in task["seeds"]:
                    task_frames.append(
                        _run_world(
                            dataset_id=task["dataset_id"], network_id=task["network_id"], system_family=task["system_family"], analysis_cluster_id=task["analysis_cluster_id"], window=task["window"], parameter=task["parameter"], model=task["model"], stable_scores=task["stable_scores"], initial=str(initial), random_block=block, experiment_seed=int(evaluation["seed"]), sentinel_fraction=float(decision["sentinel_budget_fraction"]), response_fraction=float(decision["response_budget_fraction"]), minimum_budget=int(decision["minimum_budget"]), action_delay_fraction=float(decision["action_delay_fraction_of_mean_infectious_period"]), residual=float(decision["residual_contact_multiplier"]), threshold_fraction=float(decision["early_detection_threshold_fraction"]),
                        )
                    )
            frame = pd.concat(task_frames, ignore_index=True)
            frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
        progress.set_postfix_str(f"{task['dataset_id']} {task['model']['name']}")

    worlds = pd.concat(frames, ignore_index=True)
    contrasts = compute_contrasts(worlds)
    detection = detection_contrasts(worlds)
    repetitions = int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"]))
    deletion_repetitions = int(profile.get("deletion_bootstrap_replicates", evaluation["deletion_bootstrap_replicates"]))
    summary, family, detection_summary, detection_family, policy_summary, policy_family = _summaries(worlds, contrasts, detection, bootstrap_replicates=repetitions, seed=int(evaluation["seed"]))
    decisions = _classify(summary, family)
    deletion = _leave_one_family_out(worlds, contrasts, detection, bootstrap_replicates=deletion_repetitions, seed=int(evaluation["seed"]) + 2000)

    deployable = worlds.loc[~worlds["sentinel_method"].eq("full_surveillance")]
    policy_counts = worlds.groupby(WORLD_KEYS, observed=True)["policy"].nunique()
    natural_counts = worlds.groupby(WORLD_KEYS, observed=True)["natural_final_size"].nunique()
    detection_shared = deployable.groupby(WORLD_KEYS + ["sentinel_method"], observed=True)[["detection_time", "detection_burden"]].nunique(dropna=False)
    deployable_history_response = deployable.loc[deployable["response_method"].eq("history_weight")]
    response_capacities = deployable_history_response.groupby(WORLD_KEYS, observed=True)["response_capacity"].nunique()
    response_budgets = deployable_history_response.groupby(WORLD_KEYS, observed=True)["response_budget"].nunique()
    sentinel_sizes = deployable.groupby(WORLD_KEYS + ["sentinel_method"], observed=True)["sentinel_nodes"].first().map(lambda value: 0 if pd.isna(value) or value == "" else len(str(value).split("|")))
    expected_sentinel = deployable.groupby(WORLD_KEYS + ["sentinel_method"], observed=True)["sentinel_budget"].first()
    checks = {
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "all_requested_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "five_independent_families_full": profile_name != "full" or worlds["system_family"].nunique() == 5,
        "all_policies_present": set(worlds["policy"]) == set(POLICIES),
        "policy_worlds_complete": bool(policy_counts.eq(len(POLICIES)).all()),
        "natural_world_shared": bool(natural_counts.eq(1).all()),
        "detection_shared_across_response_arms": bool(detection_shared.eq(1).all().all()),
        "resource_matched_sentinels": bool(sentinel_sizes.eq(expected_sentinel).all()),
        "resource_matched_response_capacity": bool(response_capacities.eq(1).all()),
        "resource_matched_response_budgets": bool(response_budgets.eq(1).all()),
        "perfect_sentinel_detection_is_explicit": float(decision["sentinel_detection_sensitivity"]) == 1.0,
        "finite_contrasts": bool(np.isfinite(contrasts["value"].to_numpy(float)).all() and np.isfinite(detection["value"].to_numpy(float)).all()),
        "whole_family_deletion_complete": len(deletion) == worlds["system_family"].nunique() * worlds["epidemic_model"].nunique(),
        "full_surveillance_is_ceiling_only": bool(worlds.loc[worlds["sentinel_method"].eq("full_surveillance"), "detection_burden"].eq(1).all()),
        "no_rewiring": float(decision["rewiring_fraction"]) == 0.0,
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "datasets": int(worlds["dataset_id"].nunique()),
        "families": int(worlds["system_family"].nunique()),
        "anchors": int(worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0]),
        "natural_worlds": int(worlds[WORLD_KEYS].drop_duplicates().shape[0]),
        "policy_evaluations": len(worlds),
        "decision_counts": decisions["decision"].value_counts().to_dict(),
        "scope": "idealized_monitored_sentinel_detection_and_model_based_response",
    }
    if audit["status"] != "pass":
        raise ValueError(f"role-aware sentinel audit failed: {audit}")

    outputs = {
        "policy_worlds.csv.gz": (worlds, {"index": False, "compression": "gzip"}),
        "paired_contrasts.csv.gz": (contrasts, {"index": False, "compression": "gzip"}),
        "detection_contrasts.csv.gz": (detection, {"index": False, "compression": "gzip"}),
        "contrast_summary.csv": (summary, {"index": False}),
        "family_contrasts.csv": (family, {"index": False}),
        "detection_summary.csv": (detection_summary, {"index": False}),
        "family_detection_contrasts.csv": (detection_family, {"index": False}),
        "policy_attack_rate_summary.csv": (policy_summary, {"index": False}),
        "family_policy_attack_rates.csv": (policy_family, {"index": False}),
        "role_decisions.csv": (decisions, {"index": False}),
        "leave_one_family_out_decisions.csv": (deletion, {"index": False}),
        "parameter_time_support.csv": (pd.DataFrame(support_rows), {"index": False}),
    }
    for name, (frame, options) in outputs.items():
        frame.to_csv(results_dir / name, **options)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [config_path, stable_path, prerequisite_path, Path(__file__), helper_path]
    pd.DataFrame([{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in source_paths]).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {"experiment_id": experiment_id, "profile": profile_name, "created_at_utc": datetime.now(UTC).isoformat(), "elapsed_seconds": round(time.perf_counter() - started, 3), "python": platform.python_version(), "platform": platform.platform(), "git_commit": _git_value(["rev-parse", "HEAD"]), "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])), "config_path": str(config_path), "config_sha256": _sha256(config_path)}
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dpi = int(profile["render_dpi"])
    _plot_contrasts(summary, detection_summary, report_dir / "role_contrasts.png", dpi)
    _plot_policy_rates(policy_summary, report_dir / "policy_attack_rates.png", dpi)
    _plot_family_frontier(family, detection_family, report_dir / "detection_control_frontier.png", dpi)
    display = decisions.copy()
    effect_columns = [column for column in display if column not in {"epidemic_model", "families", "required_families", "decision"} and "families" not in column]
    display[effect_columns] = display[effect_columns] * 100
    report = f"""# Role-aware sentinel surveillance and response

This preregistered experiment endogenizes outbreak detection through a fixed sentinel
set and then evaluates a resource-matched response on the same temporal epidemic
world. Perfect monitored-animal recognition is an idealized assumption.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Paired natural worlds: {audit['natural_worlds']}
- Policy evaluations: {audit['policy_evaluations']}
- Technical audit: **{audit['status']}**

All effect values below are attack-rate percentage points.

{_markdown_table(display)}

The primary question is whether past contact history improves sentinel placement,
whether a coverage-diverse sentinel list improves on simply reusing the intervention
ranking, and whether a post-detection history response adds absolute value. The full
surveillance arm is a non-resource-matched ceiling. Results are model-based and do
not validate field diagnostics or a named pathogen.
"""
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run role-aware sentinel surveillance and response.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
