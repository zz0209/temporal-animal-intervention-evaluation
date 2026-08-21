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
from animal_intervention.simulation import select_additional_targets

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
from .role_aware_sentinel_response import (
    _detection_metrics,
    _replay_response,
)
from .sequential_preparedness_update import _budget, _history_score_table, _parameters


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


def _policy(sentinel_fraction: float, response_fraction: float) -> str:
    return f"s{round(100 * sentinel_fraction):02d}_r{round(100 * response_fraction):02d}"


def _ordered_history_targets(
    stable_scores: pd.DataFrame,
    eligible: set[str],
    budget: int,
    seed: int,
    excluded: set[str] | None = None,
) -> list[str]:
    table = _history_score_table(stable_scores)
    return list(
        select_additional_targets(
            table.loc[table["candidate_id"].isin(eligible)],
            method="history_weight",
            budget=budget,
            detected_nodes=excluded or set(),
            world_seed=seed,
        )
    )


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
    sentinel_fractions: list[float],
    response_fractions: list[float],
    minimum_budget: int,
    action_delay_fraction: float,
    residual: float,
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
    sentinel_seed = _keyed_seed(
        experiment_seed,
        dataset_id,
        anchor.anchor_id,
        "capacity_frontier_sentinels",
        random_block,
    )
    maximum_sentinel_budget = max(
        _budget(len(eligible), minimum_budget, fraction) for fraction in sentinel_fractions
    )
    sentinel_order = _ordered_history_targets(
        stable_scores,
        eligible,
        maximum_sentinel_budget,
        sentinel_seed,
    )
    rows: list[dict[str, Any]] = []
    for sentinel_fraction in sentinel_fractions:
        sentinel_budget = _budget(len(eligible), minimum_budget, sentinel_fraction)
        sentinels = set(sentinel_order[:sentinel_budget])
        detection = _detection_metrics(natural, sentinels, population_size, 1.0)
        detected_nodes = set(detection["detected_nodes"])
        for response_fraction in response_fractions:
            response_capacity = (
                0
                if response_fraction <= 0
                else _budget(len(eligible), minimum_budget, response_fraction)
            )
            response_budget = min(response_capacity, len(eligible - detected_nodes))
            additional = set(
                _ordered_history_targets(
                    stable_scores,
                    eligible,
                    response_budget,
                    sentinel_seed,
                    excluded=detected_nodes,
                )
            )
            targets = detected_nodes | additional
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
                targets=targets,
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
                    "parameter_id": parameter.parameter_id,
                    "epidemic_model": model["name"],
                    "random_block": random_block,
                    "initial_infected": str(initial),
                    "world_seed": world_seed,
                    "population_size": population_size,
                    "eligible_size": len(eligible),
                    "policy": _policy(sentinel_fraction, response_fraction),
                    "sentinel_fraction": sentinel_fraction,
                    "response_fraction": response_fraction,
                    "sentinel_budget": sentinel_budget,
                    "response_capacity": response_capacity,
                    "response_budget": response_budget,
                    "sentinel_nodes": "|".join(sorted(sentinels)),
                    "response_nodes": "|".join(sorted(targets)),
                    "detected": bool(detection["detected"]),
                    "detection_time": detection["detection_time"],
                    "action_start": action_start,
                    "detection_burden": int(detection["detection_burden"]),
                    "detection_burden_rate": float(detection["detection_burden_rate"]),
                    "natural_final_size": natural.final_size,
                    "final_size": result.final_size,
                    "final_attack_rate": result.final_size / population_size,
                }
            )
    return pd.DataFrame(rows)


def _primary_contrasts(worlds: pd.DataFrame, decision: dict[str, Any]) -> pd.DataFrame:
    s0 = float(decision["reference_sentinel_fraction"])
    s1 = float(decision["expanded_sentinel_fraction"])
    r0 = float(decision["reference_response_fraction"])
    r1 = float(decision["expanded_response_fraction"])
    columns = ["final_size", "sentinel_budget", "response_capacity"]
    wide_parts = []
    for column in columns:
        wide = worlds.pivot(index=WORLD_KEYS, columns="policy", values=column)
        wide.columns = [f"{column}__{item}" for item in wide.columns]
        wide_parts.append(wide)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    metadata = worlds.drop_duplicates(WORLD_KEYS)[
        WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]
    ]
    paired = metadata.merge(wide, on=WORLD_KEYS, validate="one_to_one")

    p00 = _policy(s0, r0)
    p10 = _policy(s1, r0)
    p01 = _policy(s0, r1)
    p11 = _policy(s1, r1)
    specifications = {
        "surveillance_doubling_value": (
            paired[f"final_size__{p00}"] - paired[f"final_size__{p10}"],
            paired[f"sentinel_budget__{p10}"] - paired[f"sentinel_budget__{p00}"],
        ),
        "response_doubling_value": (
            paired[f"final_size__{p00}"] - paired[f"final_size__{p01}"],
            paired[f"response_capacity__{p01}"] - paired[f"response_capacity__{p00}"],
        ),
        "capacity_complementarity": (
            (paired[f"final_size__{p10}"] - paired[f"final_size__{p11}"])
            - (paired[f"final_size__{p00}"] - paired[f"final_size__{p01}"]),
            (paired[f"sentinel_budget__{p10}"] - paired[f"sentinel_budget__{p00}"])
            * (paired[f"response_capacity__{p01}"] - paired[f"response_capacity__{p00}"]),
        ),
    }
    rows = []
    base = paired[WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]]
    for name, (difference, capacity_difference) in specifications.items():
        frame = base.copy()
        frame["contrast"] = name
        frame["value"] = difference / paired["population_size"].astype(float)
        frame["capacity_difference"] = capacity_difference.astype(float)
        frame["estimable"] = frame["capacity_difference"].gt(0)
        frame["value_per_added_animal"] = np.where(
            frame["estimable"], frame["value"] / frame["capacity_difference"], np.nan
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _summarize(
    worlds: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, ...]:
    estimable = contrasts.loc[contrasts["estimable"]].copy()
    contrast_summary, family_contrasts = _hierarchical_summary(
        estimable,
        value_column="value",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    efficiency_summary, family_efficiency = _hierarchical_summary(
        estimable,
        value_column="value_per_added_animal",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 300,
    )
    rates = worlds.assign(value=worlds["final_attack_rate"])
    policy_summary, family_policy = _hierarchical_summary(
        rates,
        value_column="value",
        group_columns=["epidemic_model", "sentinel_fraction", "response_fraction"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 600,
    )
    detection = worlds.loc[worlds["response_fraction"].eq(0)].assign(
        value=lambda frame: frame["detection_burden_rate"]
    )
    detection_summary, family_detection = _hierarchical_summary(
        detection,
        value_column="value",
        group_columns=["epidemic_model", "sentinel_fraction"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 900,
    )
    return (
        contrast_summary,
        family_contrasts,
        efficiency_summary,
        family_efficiency,
        policy_summary,
        family_policy,
        detection_summary,
        family_detection,
    )


def _decisions(summary: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in summary.groupby("epidemic_model", observed=True, sort=True):
        metrics = frame.set_index("contrast")
        family_frame = family.loc[family["epidemic_model"].eq(model)]
        record: dict[str, Any] = {"epidemic_model": model}
        passes: dict[str, bool] = {}
        for contrast in [
            "surveillance_doubling_value",
            "response_doubling_value",
            "capacity_complementarity",
        ]:
            result = metrics.loc[contrast]
            values = family_frame.loc[family_frame["contrast"].eq(contrast), "mean_value"]
            positive = int(values.gt(0).sum())
            negative = int(values.lt(0).sum())
            if contrast == "capacity_complementarity":
                interval_excludes_zero = float(result.ci_low) > 0 or float(result.ci_high) < 0
                directional = max(positive, negative) >= 3
                passes[contrast] = interval_excludes_zero and directional
            else:
                passes[contrast] = float(result.ci_low) > 0 and positive >= 3
            record.update(
                {
                    contrast: float(result.family_equal_mean),
                    f"{contrast}_ci_low": float(result.ci_low),
                    f"{contrast}_ci_high": float(result.ci_high),
                    f"{contrast}_positive_families": positive,
                    f"{contrast}_negative_families": negative,
                    f"{contrast}_pass": passes[contrast],
                }
            )
        if passes["surveillance_doubling_value"] and passes["response_doubling_value"]:
            conclusion = "both_capacities_valuable"
        elif passes["surveillance_doubling_value"]:
            conclusion = "surveillance_capacity_priority"
        elif passes["response_doubling_value"]:
            conclusion = "response_capacity_priority"
        else:
            conclusion = "no_transportable_local_capacity_gain"
        if passes["capacity_complementarity"]:
            sign = record["capacity_complementarity"]
            conclusion += "__complements" if sign > 0 else "__substitutes"
        else:
            conclusion += "__interaction_uncertain"
        record["decision"] = conclusion
        rows.append(record)
    return pd.DataFrame(rows)


def _leave_one_family_out(
    contrasts: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for index, held_out in enumerate(sorted(contrasts["system_family"].unique())):
        subset = contrasts.loc[
            (~contrasts["system_family"].eq(held_out)) & contrasts["estimable"]
        ]
        summary, family = _hierarchical_summary(
            subset,
            value_column="value",
            group_columns=["epidemic_model", "contrast"],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index * 1000,
        )
        decision = _decisions(summary, family)
        decision.insert(0, "held_out_family", held_out)
        rows.append(decision)
    return pd.concat(rows, ignore_index=True)


def _plot_frontier(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    models = ["temporal_sir", "temporal_seir_erlang"]
    titles = ["Temporal SIR", "Staged SEIR/Erlang"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    values = 100 * summary["family_equal_mean"].to_numpy(float)
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    image = None
    for axis, model, title in zip(axes, models, titles):
        frame = summary.loc[summary["epidemic_model"].eq(model)]
        pivot = 100 * frame.pivot(
            index="sentinel_fraction", columns="response_fraction", values="family_equal_mean"
        ).sort_index(ascending=False)
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        axis.set_xticks(range(len(pivot.columns)), [f"{100*x:.0f}%" for x in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), [f"{100*x:.0f}%" for x in pivot.index])
        for row in range(len(pivot.index)):
            for column in range(len(pivot.columns)):
                value = float(pivot.iloc[row, column])
                midpoint = (vmin + vmax) / 2
                axis.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white" if value > midpoint else "black",
                )
        axis.set_title(title, fontsize=15, weight="bold")
        axis.set_xlabel("Additional response capacity")
    axes[0].set_ylabel("Monitored sentinel capacity")
    fig.suptitle("Surveillance-response capacity frontier", fontsize=20, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.86, top=0.82, bottom=0.13, wspace=0.20)
    color_axis = fig.add_axes([0.90, 0.19, 0.018, 0.57])
    fig.colorbar(image, cax=color_axis, label="Final attack rate (%)")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_primary(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    order = ["surveillance_doubling_value", "response_doubling_value", "capacity_complementarity"]
    labels = ["Double monitored capacity", "Double response capacity", "Capacity interaction"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = summary.loc[summary["epidemic_model"].eq(model)].set_index("contrast").loc[order]
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        y = np.arange(len(order))
        axis.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="o", capsize=4, color="#4C78A8")
        axis.axvline(0, linestyle="--", color="#666666", linewidth=1)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=15, weight="bold")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Marginal value and interaction of limited capacities", fontsize=19, weight="bold")
    fig.supxlabel("Family-equal attack-rate improvement (percentage points)")
    endpoints = 100 * summary[["ci_low", "ci_high"]].to_numpy(float)
    if np.nanmax(np.abs(endpoints)) < 1e-6:
        axes[0].set_xlim(-0.1, 0.1)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.80, bottom=0.16, wspace=0.15)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_efficiency(family: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    families = sorted(family["system_family"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(families)))
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = family.loc[family["epidemic_model"].eq(model)]
        surveillance = frame.loc[frame["contrast"].eq("surveillance_doubling_value")].set_index("system_family")["mean_value"]
        response = frame.loc[frame["contrast"].eq("response_doubling_value")].set_index("system_family")["mean_value"]
        common = surveillance.index.intersection(response.index)
        for color, system in zip(colors, families):
            if system not in common:
                continue
            axis.scatter(100 * surveillance.loc[system], 100 * response.loc[system], s=90, color=color, label=SYSTEM_FAMILY_LABELS.get(system, system.replace("_", " ")))
        axis.axhline(0, linestyle="--", color="#666666", linewidth=1)
        axis.axvline(0, linestyle="--", color="#666666", linewidth=1)
        axis.set_title(title, fontsize=15, weight="bold")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Response gain per added response animal (points)")
    fig.supxlabel("Surveillance gain per added monitored animal (points)")
    axes[1].legend(loc="best", frameon=False, fontsize=9)
    plotted = 100 * family["mean_value"].dropna().to_numpy(float)
    if len(plotted) and np.nanmax(np.abs(plotted)) < 1e-6:
        axes[0].set_xlim(-0.1, 0.1)
        axes[0].set_ylim(-0.1, 0.1)
    fig.suptitle("Which added capacity is more efficient within each animal system?", fontsize=19, weight="bold")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.81, bottom=0.15, wspace=0.14)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_detection(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 6))
    for model, label, color, marker in [
        ("temporal_sir", "Temporal SIR", "#4C78A8", "o"),
        ("temporal_seir_erlang", "Staged SEIR/Erlang", "#F58518", "s"),
    ]:
        frame = summary.loc[summary["epidemic_model"].eq(model)].sort_values(
            "sentinel_fraction"
        )
        x = 100 * frame["sentinel_fraction"].to_numpy(float)
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        axis.errorbar(
            x,
            mean,
            yerr=[mean - low, high - mean],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            capsize=4,
        )
    axis.set_xlabel("Monitored sentinel capacity (% of eligible animals)")
    axis.set_ylabel("Epidemic burden accrued before detection (%)")
    axis.set_title("How monitoring capacity changes detection burden", fontsize=18, weight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _nested_sentinels(worlds: pd.DataFrame) -> bool:
    first = worlds.drop_duplicates(WORLD_KEYS + ["sentinel_fraction"])
    for _, frame in first.groupby(WORLD_KEYS, observed=True):
        previous: set[str] = set()
        for record in frame.sort_values("sentinel_fraction").itertuples(index=False):
            current = set() if not record.sentinel_nodes else set(str(record.sentinel_nodes).split("|"))
            if not previous.issubset(current):
                return False
            previous = current
    return True


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
        raise ValueError("capacity-frontier prerequisite audit must pass")
    stable_path = Path(config["data"]["stable_prediction_path"])
    stable_predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    stable_predictions["anchor_time"] = pd.to_datetime(stable_predictions["anchor_time"], format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
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
                supported = pd.Timestamp(anchor.anchor_time) + mean_period * float(
                    decision["action_delay_fraction_of_mean_infectious_period"]
                ) < pd.Timestamp(anchor.horizon_end)
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
                "capacity_frontier_seeds",
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
        config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()[:12]
    frames = []
    progress = tqdm(tasks, desc="Capacity-frontier worlds", unit="task")
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
                            sentinel_fractions=list(map(float, decision["sentinel_budget_fractions"])),
                            response_fractions=list(map(float, decision["response_budget_fractions"])),
                            minimum_budget=int(decision["minimum_positive_budget"]),
                            action_delay_fraction=float(decision["action_delay_fraction_of_mean_infectious_period"]),
                            residual=float(decision["residual_contact_multiplier"]),
                        )
                    )
            frame = pd.concat(task_frames, ignore_index=True)
            frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
        progress.set_postfix_str(f"{task['dataset_id']} {task['model']['name']}")

    worlds = pd.concat(frames, ignore_index=True)
    contrasts = _primary_contrasts(worlds, decision)
    repetitions = int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"]))
    deletion_repetitions = int(
        profile.get("deletion_bootstrap_replicates", evaluation["deletion_bootstrap_replicates"])
    )
    summaries = _summarize(
        worlds, contrasts, bootstrap_replicates=repetitions, seed=int(evaluation["seed"])
    )
    (
        contrast_summary,
        family_contrasts,
        efficiency_summary,
        family_efficiency,
        policy_summary,
        family_policy,
        detection_summary,
        family_detection,
    ) = summaries
    decisions = _decisions(contrast_summary, family_contrasts)
    deletion = _leave_one_family_out(
        contrasts,
        bootstrap_replicates=deletion_repetitions,
        seed=int(evaluation["seed"]) + 3000,
    )

    policy_count = len(decision["sentinel_budget_fractions"]) * len(decision["response_budget_fractions"])
    counts = worlds.groupby(WORLD_KEYS, observed=True)["policy"].nunique()
    natural_counts = worlds.groupby(WORLD_KEYS, observed=True)["natural_final_size"].nunique()
    detection_counts = worlds.groupby(WORLD_KEYS + ["sentinel_fraction"], observed=True)[
        ["detection_time", "detection_burden"]
    ].nunique(dropna=False)
    finite = np.isfinite(contrasts["value"].to_numpy(float)).all()
    estimable_families = contrasts.loc[contrasts["estimable"]].groupby(
        ["epidemic_model", "contrast"], observed=True
    )["system_family"].nunique()
    checks = {
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "all_requested_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "five_independent_families_full": profile_name != "full" or worlds["system_family"].nunique() == 5,
        "all_factorial_policies_present": worlds["policy"].nunique() == policy_count,
        "policy_worlds_complete": bool(counts.eq(policy_count).all()),
        "natural_world_shared": bool(natural_counts.eq(1).all()),
        "detection_shared_across_response_capacity": bool(detection_counts.eq(1).all().all()),
        "nested_sentinel_sets": _nested_sentinels(worlds),
        "nondecreasing_realized_budgets": bool(
            worlds.groupby(WORLD_KEYS + ["sentinel_fraction"], observed=True)["sentinel_budget"].first().groupby(level=WORLD_KEYS).apply(lambda series: series.sort_index(level="sentinel_fraction").is_monotonic_increasing).all()
        ),
        "finite_primary_contrasts": bool(finite),
        "nonestimable_integer_ties_flagged": bool((~contrasts["estimable"]).any()),
        "at_least_four_estimable_families_per_main_effect": bool(
            profile_name != "full" or estimable_families.min() >= 4
        ),
        "whole_family_deletion_complete": len(deletion) == worlds["system_family"].nunique() * worlds["epidemic_model"].nunique(),
        "perfect_detection_assumption_explicit": float(decision["sentinel_detection_sensitivity"]) == 1.0,
        "no_rewiring": float(decision["rewiring_fraction"]) == 0.0,
        "attack_rates_bounded": bool(worlds["final_attack_rate"].between(0, 1).all()),
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
        "scope": "idealized_detection_capacity_and_post_detection_response_frontier",
    }
    if audit["status"] != "pass":
        raise ValueError(f"capacity-frontier audit failed: {audit}")

    outputs = {
        "policy_worlds.csv.gz": (worlds, {"index": False, "compression": "gzip"}),
        "primary_contrasts.csv.gz": (contrasts, {"index": False, "compression": "gzip"}),
        "contrast_summary.csv": (contrast_summary, {"index": False}),
        "family_contrasts.csv": (family_contrasts, {"index": False}),
        "efficiency_summary.csv": (efficiency_summary, {"index": False}),
        "family_efficiency.csv": (family_efficiency, {"index": False}),
        "policy_summary.csv": (policy_summary, {"index": False}),
        "family_policy.csv": (family_policy, {"index": False}),
        "detection_summary.csv": (detection_summary, {"index": False}),
        "family_detection.csv": (family_detection, {"index": False}),
        "decisions.csv": (decisions, {"index": False}),
        "leave_one_family_out.csv": (deletion, {"index": False}),
        "parameter_time_support.csv": (pd.DataFrame(support_rows), {"index": False}),
    }
    for name, (frame, options) in outputs.items():
        frame.to_csv(results_dir / name, **options)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
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
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dpi = int(profile["render_dpi"])
    _plot_frontier(policy_summary, report_dir / "capacity_frontier.png", dpi)
    _plot_primary(contrast_summary, report_dir / "primary_capacity_effects.png", dpi)
    _plot_efficiency(family_efficiency, report_dir / "family_capacity_efficiency.png", dpi)
    _plot_detection(detection_summary, report_dir / "detection_capacity_curve.png", dpi)
    display = decisions.copy()
    numeric = [column for column in display if column not in {"epidemic_model", "decision"} and not column.endswith("_pass") and display[column].dtype.kind in "f"]
    display[numeric] = 100 * display[numeric]
    report = f"""# Surveillance-response capacity frontier

This frozen 3 x 3 factorial experiment estimates the separate and joint value of
monitored-animal capacity and post-detection response capacity. It does not assume
equal monetary costs for the two resources.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Paired natural worlds: {audit['natural_worlds']}
- Policy evaluations: {audit['policy_evaluations']}
- Technical audit: **{audit['status']}**

All continuous effects in the table are attack-rate percentage points.

{_markdown_table(display)}

Families with unchanged integer budgets are marked non-estimable for the relevant
contrast and are not counted as zero biological effects. Perfect recognition of an
infected monitored animal is an idealized assumption. Results are model-based and
do not define a universal monetary allocation without local monitoring and response costs.
"""
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
