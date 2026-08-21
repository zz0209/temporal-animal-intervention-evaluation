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
    ContactReductionPhase,
    PairedTemporalSEIREngine,
    PairedTemporalSIREngine,
    SEIRParameters,
    SIRParameters,
    apply_contact_reduction_schedule,
    observe_detected_cases,
    pre_detection_event_signature,
    pre_detection_scores,
    segment_exposure_stream,
    select_additional_targets,
    states_at,
)

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


POLICIES = [
    "case_only",
    "early_history",
    "two_stage_history",
    "two_stage_reactive",
    "history_upfront",
]
CONTRASTS = {
    "preparedness_absolute": ("case_only", "early_history"),
    "second_history_value": ("early_history", "two_stage_history"),
    "second_reactive_value": ("early_history", "two_stage_reactive"),
    "update_information_gain": ("two_stage_history", "two_stage_reactive"),
    "staged_history_cost": ("two_stage_history", "history_upfront"),
    "sequential_recovery": ("history_upfront", "two_stage_reactive"),
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


def _parameters(parameter: Any, model: dict[str, Any], mean_period: pd.Timedelta) -> tuple[Any, Any]:
    name = str(model["name"])
    if name == "temporal_sir":
        return (
            PairedTemporalSIREngine(),
            SIRParameters(
                beta=float(parameter.beta),
                recovery_rate=float(parameter.recovery_rate_per_second),
            ),
        )
    if name == "temporal_seir_erlang":
        latent_fraction = float(model["latent_period_fraction_of_mean_infectious_period"])
        return (
            PairedTemporalSEIREngine(),
            SEIRParameters(
                beta=float(parameter.beta),
                latent_rate=1.0 / (mean_period.total_seconds() * latent_fraction),
                recovery_rate=float(parameter.recovery_rate_per_second),
                latent_stages=int(model.get("latent_stages", 2)),
                infectious_stages=int(model.get("infectious_stages", 3)),
            ),
        )
    raise ValueError(f"unsupported epidemic model: {name}")


def _budget(remaining: int, minimum: int, fraction: float) -> int:
    if remaining <= 0:
        return 0
    return min(remaining, max(minimum, int(math.ceil(remaining * fraction))))


def _scheduled_stream(
    segmented: Any,
    *,
    early_start: pd.Timestamp,
    update_start: pd.Timestamp,
    end_time: pd.Timestamp,
    early_targets: set[str],
    late_targets: set[str],
    residual: float,
) -> Any:
    return apply_contact_reduction_schedule(
        segmented,
        [
            ContactReductionPhase(
                early_start,
                update_start,
                tuple(sorted(early_targets)),
                residual,
            ),
            ContactReductionPhase(
                update_start,
                end_time,
                tuple(sorted(late_targets)),
                residual,
            ),
        ],
    )


def _history_score_table(stable: pd.DataFrame) -> pd.DataFrame:
    scores = stable.copy()
    scores["candidate_id"] = scores["candidate_id"].astype(str)
    scores["current_activity"] = 0.0
    scores["contact_to_detected"] = 0.0
    return scores


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
    early_fraction: float,
    update_fraction: float,
    action_delay_fraction: float,
    residual: float,
    sensitivity: float,
    false_positive_rate: float,
    minimum_budget: int,
    budget_fraction: float,
    tracing_half_life_fraction: float,
) -> pd.DataFrame:
    anchor = window["anchor"]
    end_time = pd.Timestamp(anchor.horizon_end)
    mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
    early_detection = pd.Timestamp(anchor.anchor_time) + mean_period * early_fraction
    update_detection = pd.Timestamp(anchor.anchor_time) + mean_period * update_fraction
    early_start = early_detection + mean_period * action_delay_fraction
    update_start = update_detection + mean_period * action_delay_fraction
    if not early_detection < early_start < update_detection < update_start < end_time:
        return pd.DataFrame()

    engine, parameters = _parameters(parameter, model, mean_period)
    segmented = segment_exposure_stream(window["future"], [early_start, update_start])
    world_seed = _keyed_seed(
        experiment_seed,
        dataset_id,
        anchor.anchor_id,
        parameter.parameter_id,
        random_block,
        initial,
    )
    natural = engine.simulate(
        segmented,
        parameters,
        initial_infected=(initial,),
        start_time=anchor.anchor_time,
        end_time=end_time,
        world_seed=world_seed,
    )
    early_states = states_at(natural, early_detection)
    detected_early = set(
        observe_detected_cases(
            early_states,
            trigger_node=initial,
            secondary_case_sensitivity=sensitivity,
            false_positive_rate=false_positive_rate,
            world_seed=world_seed,
        )
    )
    eligible = set(map(str, window["eligible"]))
    history_table = _history_score_table(stable_scores)
    first_budget = _budget(
        len(eligible - detected_early), minimum_budget, budget_fraction
    )
    first_history = set(
        select_additional_targets(
            history_table,
            method="history_weight",
            budget=first_budget,
            detected_nodes=detected_early,
            world_seed=world_seed,
        )
    )
    upfront_history = set(
        select_additional_targets(
            history_table,
            method="history_weight",
            budget=min(len(eligible - detected_early), 2 * first_budget),
            detected_nodes=detected_early,
            world_seed=world_seed,
        )
    )

    branch_early_targets = {
        "case": detected_early,
        "prepared": detected_early | first_history,
        "upfront": detected_early | upfront_history,
    }
    branch_results: dict[str, Any] = {}
    branch_streams: dict[str, Any] = {}
    branch_detected: dict[str, set[str]] = {}
    for name, targets in branch_early_targets.items():
        stream = _scheduled_stream(
            segmented,
            early_start=early_start,
            update_start=update_start,
            end_time=end_time,
            early_targets=targets,
            late_targets=targets,
            residual=residual,
        )
        result = engine.simulate(
            stream,
            parameters,
            initial_infected=(initial,),
            start_time=anchor.anchor_time,
            end_time=end_time,
            world_seed=world_seed,
        )
        later = set(
            observe_detected_cases(
                states_at(result, update_detection),
                trigger_node=initial,
                secondary_case_sensitivity=sensitivity,
                false_positive_rate=false_positive_rate,
                world_seed=world_seed,
            )
        )
        branch_results[name] = result
        branch_streams[name] = stream
        branch_detected[name] = detected_early | later

    prepared_blocked = branch_early_targets["prepared"] | branch_detected["prepared"]
    second_budget = min(first_budget, len(eligible - prepared_blocked))
    second_history = set(
        select_additional_targets(
            history_table,
            method="history_weight",
            budget=second_budget,
            detected_nodes=prepared_blocked,
            world_seed=world_seed,
        )
    )
    contact_scores = pre_detection_scores(
        branch_streams["prepared"],
        detected_nodes=branch_detected["prepared"],
        start_time=anchor.anchor_time,
        detection_time=update_detection,
        half_life=mean_period * tracing_half_life_fraction,
    )
    contact_scores["candidate_id"] = contact_scores["candidate_id"].astype(str)
    score_table = contact_scores.merge(
        stable_scores.assign(candidate_id=stable_scores["candidate_id"].astype(str)),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    score_table = score_table.loc[score_table["candidate_id"].isin(eligible)].copy()
    if score_table["history_weight"].isna().any():
        raise ValueError("sequential score table has missing history values")
    second_reactive = set(
        select_additional_targets(
            score_table,
            method="contact_to_detected",
            budget=second_budget,
            detected_nodes=prepared_blocked,
            world_seed=world_seed,
        )
    )
    if len(second_history) != len(second_reactive):
        raise AssertionError("second-tranche policies must use equal capacity")

    policy_targets = {
        "case_only": (
            branch_early_targets["case"],
            branch_early_targets["case"] | branch_detected["case"],
            set(),
            set(),
        ),
        "early_history": (
            branch_early_targets["prepared"],
            branch_early_targets["prepared"] | branch_detected["prepared"],
            first_history,
            set(),
        ),
        "two_stage_history": (
            branch_early_targets["prepared"],
            branch_early_targets["prepared"]
            | branch_detected["prepared"]
            | second_history,
            first_history,
            second_history,
        ),
        "two_stage_reactive": (
            branch_early_targets["prepared"],
            branch_early_targets["prepared"]
            | branch_detected["prepared"]
            | second_reactive,
            first_history,
            second_reactive,
        ),
        "history_upfront": (
            branch_early_targets["upfront"],
            branch_early_targets["upfront"] | branch_detected["upfront"],
            upfront_history,
            upfront_history - first_history,
        ),
    }

    rows = []
    prepared_signature = pre_detection_event_signature(
        branch_results["prepared"], update_detection
    )
    for policy, (
        early_targets,
        late_targets,
        first_additional,
        second_additional,
    ) in policy_targets.items():
        stream = _scheduled_stream(
            segmented,
            early_start=early_start,
            update_start=update_start,
            end_time=end_time,
            early_targets=early_targets,
            late_targets=late_targets,
            residual=residual,
        )
        result = engine.simulate(
            stream,
            parameters,
            initial_infected=(initial,),
            start_time=anchor.anchor_time,
            end_time=end_time,
            world_seed=world_seed,
        )
        if policy in {"early_history", "two_stage_history", "two_stage_reactive"}:
            if pre_detection_event_signature(result, update_detection) != prepared_signature:
                raise AssertionError("prepared policies diverged before the update decision")
        rows.append(
            {
                "dataset_id": dataset_id,
                "network_id": network_id,
                "system_family": system_family,
                "analysis_cluster_id": analysis_cluster_id,
                "anchor_id": anchor.anchor_id,
                "anchor_time": anchor.anchor_time,
                "horizon_end": end_time,
                "parameter_id": parameter.parameter_id,
                "beta": float(parameter.beta),
                "mean_infectious_period_days": float(parameter.mean_infectious_period_days),
                "epidemic_model": str(model["name"]),
                "random_block": random_block,
                "initial_infected": str(initial),
                "world_seed": world_seed,
                "population_size": len(segmented.nodes()),
                "policy": policy,
                "natural_final_size": natural.final_size,
                "final_size": result.final_size,
                "final_attack_rate": result.final_size / len(segmented.nodes()),
                "early_detection_time": early_detection,
                "early_action_start": early_start,
                "update_detection_time": update_detection,
                "update_action_start": update_start,
                "detected_early_count": len(detected_early),
                "detected_update_count": len(
                    branch_detected[
                        "case"
                        if policy == "case_only"
                        else "upfront"
                        if policy == "history_upfront"
                        else "prepared"
                    ]
                ),
                "tranche_capacity": first_budget,
                "update_available_capacity": second_budget,
                "first_additional_targets": "|".join(sorted(first_additional)),
                "second_additional_targets": "|".join(sorted(second_additional)),
                "first_additional_count": len(first_additional),
                "second_additional_count": len(second_additional),
                "active_targets_after_update": len(late_targets),
                "contact_evidence_mass": float(score_table["contact_to_detected"].sum()),
                "contact_evidence_nodes": int(score_table["contact_to_detected"].gt(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    """Create normalized paired policy contrasts in each natural world."""

    metadata = worlds.drop_duplicates(WORLD_KEYS)[
        WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]
    ]
    wide = worlds.pivot(index=WORLD_KEYS, columns="policy", values="final_size").reset_index()
    paired = metadata.merge(wide, on=WORLD_KEYS, validate="one_to_one")
    rows = []
    for name, (reference, challenger) in CONTRASTS.items():
        selected = paired[
            WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]
        ].copy()
        selected["contrast"] = name
        selected["value"] = (
            paired[reference] - paired[challenger]
        ) / paired["population_size"].astype(float)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _summaries(
    worlds: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrast_summary, contrast_family = _hierarchical_summary(
        contrasts,
        value_column="value",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    rates = worlds.copy()
    rates["value"] = rates["final_attack_rate"]
    policy_summary, policy_family = _hierarchical_summary(
        rates,
        value_column="value",
        group_columns=["epidemic_model", "policy"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 500,
    )
    return contrast_summary, contrast_family, policy_summary, policy_family


def _classify(summary: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in summary.groupby("epidemic_model", observed=True, sort=True):
        metrics = frame.set_index("contrast")
        family_frame = family.loc[family["epidemic_model"].eq(model)]
        families = int(frame["families"].max())
        required = int(math.ceil(0.8 * families))

        def direction(metric: str, nonnegative: bool = False) -> int:
            values = family_frame.loc[family_frame["contrast"].eq(metric), "mean_value"]
            return int(values.ge(0).sum() if nonnegative else values.gt(0).sum())

        update = metrics.loc["update_information_gain"]
        absolute = metrics.loc["second_reactive_value"]
        recovery = metrics.loc["sequential_recovery"]
        update_pass = float(update.ci_low) > 0 and direction("update_information_gain") >= required
        absolute_pass = float(absolute.ci_low) > 0 and direction("second_reactive_value") >= required
        recovery_pass = float(recovery.ci_low) >= 0 and direction("sequential_recovery", True) >= required
        if update_pass and absolute_pass and recovery_pass:
            decision = "sequential_update_supported"
        elif update_pass and absolute_pass:
            decision = "useful_update_but_not_timing_recovery"
        elif absolute_pass:
            decision = "second_tranche_helpful_but_history_sufficient"
        else:
            decision = "retain_early_preparedness_only_or_abstain"
        rows.append(
            {
                "epidemic_model": model,
                "families": families,
                "required_families": required,
                "update_information_gain": float(update.family_equal_mean),
                "update_ci_low": float(update.ci_low),
                "update_ci_high": float(update.ci_high),
                "update_positive_families": direction("update_information_gain"),
                "second_reactive_value": float(absolute.family_equal_mean),
                "second_reactive_ci_low": float(absolute.ci_low),
                "second_reactive_ci_high": float(absolute.ci_high),
                "second_reactive_positive_families": direction("second_reactive_value"),
                "sequential_recovery": float(recovery.family_equal_mean),
                "recovery_ci_low": float(recovery.ci_low),
                "recovery_ci_high": float(recovery.ci_high),
                "recovery_nonnegative_families": direction("sequential_recovery", True),
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_family_out(
    worlds: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for index, held_out in enumerate(sorted(contrasts["system_family"].unique())):
        subset_contrasts = contrasts.loc[~contrasts["system_family"].eq(held_out)]
        subset_worlds = worlds.loc[~worlds["system_family"].eq(held_out)]
        summary, family, _, _ = _summaries(
            subset_worlds,
            subset_contrasts,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index * 1000,
        )
        decisions = _classify(summary, family)
        decisions.insert(0, "held_out_family", held_out)
        rows.append(decisions)
    return pd.concat(rows, ignore_index=True)


def _plot_contrasts(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    order = list(CONTRASTS)
    labels = {
        "preparedness_absolute": "Early history vs cases only",
        "second_history_value": "Second history tranche",
        "second_reactive_value": "Second reactive tranche",
        "update_information_gain": "Reactive vs history update",
        "staged_history_cost": "Cost of staging history",
        "sequential_recovery": "Reactive sequence vs upfront",
    }
    models = ["temporal_sir", "temporal_seir_erlang"]
    titles = ["Temporal SIR", "Staged SEIR/Erlang"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for axis, model, title in zip(axes, models, titles):
        frame = summary.loc[summary["epidemic_model"].eq(model)].set_index("contrast").loc[order]
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
    fig.suptitle("What does each stage of the closed-loop policy add?", fontsize=22, weight="bold")
    fig.supxlabel("Family-equal avoided attack-rate difference (percentage points)", fontsize=14)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.84, bottom=0.12, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_policy_rates(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    labels = {
        "case_only": "Cases only",
        "early_history": "Early history",
        "two_stage_history": "History + history",
        "two_stage_reactive": "History + outbreak update",
        "history_upfront": "Two history tranches upfront",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = summary.loc[summary["epidemic_model"].eq(model)].set_index("policy").loc[POLICIES]
        x = np.arange(len(POLICIES))
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        axis.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="o", capsize=4, color="#F58518")
        axis.set_xticks(x, [labels[item] for item in POLICIES], rotation=25, ha="right")
        axis.set_title(title, fontsize=16, weight="bold")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final attack rate (%)", fontsize=13)
    fig.suptitle("Resource-matched two-stage policy outcomes", fontsize=22, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.26, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_family_frontier(family: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    families = sorted(family["system_family"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(families)))
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = family.loc[family["epidemic_model"].eq(model)]
        wide = frame.pivot(index="system_family", columns="contrast", values="mean_value")
        for color, system in zip(colors, families):
            axis.scatter(
                100 * wide.loc[system, "update_information_gain"],
                100 * wide.loc[system, "sequential_recovery"],
                s=85,
                color=color,
                label=SYSTEM_FAMILY_LABELS.get(system, system.replace("_", " ")),
            )
        axis.axhline(0, color="#666666", linestyle="--", linewidth=1)
        axis.axvline(0, color="#666666", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Sequence vs equal capacity upfront (points)", fontsize=13)
    fig.supxlabel("Reactive vs history second tranche (points)", fontsize=13)
    axes[1].legend(loc="best", fontsize=9, frameon=False)
    fig.suptitle("Does information improve selection and recover staging cost?", fontsize=21, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.15, wspace=0.12)
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
        raise ValueError("sequential policy prerequisite audit must pass")
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
        available = set(
            stable_predictions.loc[
                stable_predictions["dataset_id"].eq(dataset_id), ["network_id", "anchor_time"]
            ].itertuples(index=False, name=None)
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
                update_start = pd.Timestamp(anchor.anchor_time) + mean_period * (
                    float(decision["update_detection_fraction"])
                    + float(decision["action_delay_fraction_of_mean_infectious_period"])
                )
                supported = update_start < pd.Timestamp(anchor.horizon_end)
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
            seeds = stable_hash_order(
                list(map(str, window["eligible"])),
                int(evaluation["seed"]),
                dataset_id,
                anchor.anchor_id,
                "sequential_policy_seeds",
            )[: int(profile["seeds_per_anchor"])]
            for model in decision["epidemic_models"]:
                tasks.append(
                    {
                        "dataset_id": dataset_id,
                        "network_id": network_id,
                        "system_family": str(specification["system_family"]),
                        "analysis_cluster_id": cluster,
                        "window": window,
                        "parameter": parameter,
                        "model": dict(model),
                        "stable_scores": stable,
                        "seeds": seeds,
                    }
                )

    fingerprint = hashlib.sha256(
        config_path.read_bytes()
        + stable_path.read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()[:12]
    frames = []
    progress = tqdm(tasks, desc="Sequential policy worlds", unit="task")
    for task in progress:
        identity = "|".join(
            [
                fingerprint,
                task["dataset_id"],
                task["network_id"],
                task["window"]["anchor"].anchor_id,
                str(task["parameter"].parameter_id),
                str(task["model"]["name"]),
            ]
        )
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
                            dataset_id=task["dataset_id"],
                            network_id=task["network_id"],
                            system_family=task["system_family"],
                            analysis_cluster_id=task["analysis_cluster_id"],
                            window=task["window"],
                            parameter=task["parameter"],
                            model=task["model"],
                            stable_scores=task["stable_scores"],
                            initial=str(initial),
                            random_block=block,
                            experiment_seed=int(evaluation["seed"]),
                            early_fraction=float(decision["early_detection_fraction"]),
                            update_fraction=float(decision["update_detection_fraction"]),
                            action_delay_fraction=float(decision["action_delay_fraction_of_mean_infectious_period"]),
                            residual=float(decision["residual_contact_multiplier"]),
                            sensitivity=float(decision["secondary_case_sensitivity"]),
                            false_positive_rate=float(decision["false_positive_rate"]),
                            minimum_budget=int(decision["minimum_tranche_budget"]),
                            budget_fraction=float(decision["tranche_budget_fraction"]),
                            tracing_half_life_fraction=float(decision["tracing_half_life_fraction_of_mean_infectious_period"]),
                        )
                    )
            frame = pd.concat(task_frames, ignore_index=True)
            frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
        progress.set_postfix_str(
            f"{task['dataset_id']} {task['model']['name']}"
        )
    worlds = pd.concat(frames, ignore_index=True)
    contrasts = compute_contrasts(worlds)
    repetitions = int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"]))
    deletion_repetitions = int(
        profile.get("deletion_bootstrap_replicates", evaluation["deletion_bootstrap_replicates"])
    )
    summary, family, policy_summary, policy_family = _summaries(
        worlds,
        contrasts,
        bootstrap_replicates=repetitions,
        seed=int(evaluation["seed"]),
    )
    decisions = _classify(summary, family)
    deletion = _leave_one_family_out(
        worlds,
        contrasts,
        bootstrap_replicates=deletion_repetitions,
        seed=int(evaluation["seed"]) + 2000,
    )

    policy_counts = worlds.groupby(WORLD_KEYS, observed=True)["policy"].nunique()
    natural_counts = worlds.groupby(WORLD_KEYS, observed=True)["natural_final_size"].nunique()
    second_counts = worlds.loc[
        worlds["policy"].isin(["two_stage_history", "two_stage_reactive"])
    ].groupby(WORLD_KEYS, observed=True)["second_additional_count"].apply(list)
    checks = {
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "all_requested_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "five_independent_families_full": profile_name != "full" or worlds["system_family"].nunique() == 5,
        "all_policies_present": set(worlds["policy"]) == set(POLICIES),
        "policy_worlds_complete": bool(policy_counts.eq(len(POLICIES)).all()),
        "natural_world_shared": bool(natural_counts.eq(1).all()),
        "resource_matched_second_tranches": all(len(set(counts)) == 1 for counts in second_counts),
        "finite_contrasts": bool(np.isfinite(contrasts["value"].to_numpy(float)).all()),
        "whole_family_deletion_complete": len(deletion) == worlds["system_family"].nunique() * worlds["epidemic_model"].nunique(),
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
        "scope": "model_based_sequential_policy_not_field_causal_validation",
    }
    if audit["status"] != "pass":
        raise ValueError(f"sequential policy audit failed: {audit}")

    outputs = {
        "policy_worlds.csv.gz": (worlds, {"index": False, "compression": "gzip"}),
        "paired_contrasts.csv.gz": (contrasts, {"index": False, "compression": "gzip"}),
        "contrast_summary.csv": (summary, {"index": False}),
        "family_contrasts.csv": (family, {"index": False}),
        "policy_attack_rate_summary.csv": (policy_summary, {"index": False}),
        "family_policy_attack_rates.csv": (policy_family, {"index": False}),
        "policy_decisions.csv": (decisions, {"index": False}),
        "leave_one_family_out_decisions.csv": (deletion, {"index": False}),
        "parameter_time_support.csv": (pd.DataFrame(support_rows), {"index": False}),
    }
    for name, (frame, options) in outputs.items():
        frame.to_csv(results_dir / name, **options)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {
        "profile": profile_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    source_paths = [config_path, stable_path, prerequisite_path, Path(__file__)]
    pd.DataFrame(
        [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ]
    ).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
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
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    dpi = int(profile["render_dpi"])
    _plot_contrasts(summary, report_dir / "closed_loop_contrasts.png", dpi)
    _plot_policy_rates(policy_summary, report_dir / "policy_attack_rates.png", dpi)
    _plot_family_frontier(family, report_dir / "family_update_frontier.png", dpi)

    display = decisions.copy()
    effect_columns = [
        "update_information_gain",
        "update_ci_low",
        "update_ci_high",
        "second_reactive_value",
        "second_reactive_ci_low",
        "second_reactive_ci_high",
        "sequential_recovery",
        "recovery_ci_low",
        "recovery_ci_high",
    ]
    display[effect_columns] = display[effect_columns] * 100
    report = f"""# Sequential preparedness-to-update policy

This preregistered experiment tests a closed-loop policy without delaying the first
history-based action. Later surveillance is generated under each arm's own early
intervention trajectory, and contact scores use intervention-adjusted exposures.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Paired natural worlds: {audit['natural_worlds']}
- Policy evaluations: {audit['policy_evaluations']}
- Technical audit: **{audit['status']}**

All effect values below are attack-rate percentage points.

{_markdown_table(display)}

`update_information_gain` compares reactive and history targeting for the same
second tranche. `second_reactive_value` compares adding that reactive tranche with
stopping after early preparedness. `sequential_recovery` compares the full reactive
sequence with placing the same two-tranche history capacity upfront. Animal-system
family is the top-level replication unit. Results are model-based counterfactuals,
not field causal effects.
"""
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sequential preparedness-to-update policy experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
