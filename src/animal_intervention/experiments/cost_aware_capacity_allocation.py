"""Evaluate transferable surveillance-response allocation with imperfect recognition."""

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
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.evaluation import stable_hash_order

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
from .role_aware_sentinel_response import _infection_events, _replay_response
from .sequential_preparedness_update import _budget, _parameters
from .surveillance_response_frontier import WORLD_KEYS, _ordered_history_targets, _policy


PAIR_KEYS = WORLD_KEYS + ["recognition_sensitivity"]


def _recognition_uniform(seed: int, node_id: str, time_value: Any) -> float:
    """Return a keyed uniform variate shared across recognition sensitivities."""
    payload = f"{seed}|{node_id}|{pd.Timestamp(time_value).isoformat()}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def _recognized_detection_metrics(
    natural: Any,
    sentinels: set[str],
    population_size: int,
    sensitivity: float,
    recognition_seed: int,
) -> dict[str, Any]:
    """Detect the first infected sentinel whose infection is recognized."""
    infections = _infection_events(natural)
    sentinel_events = infections.loc[infections["node_id"].isin(sentinels)].copy()
    if not sentinel_events.empty:
        sentinel_events["recognition_uniform"] = [
            _recognition_uniform(recognition_seed, row.node_id, row.time)
            for row in sentinel_events.itertuples(index=False)
        ]
        sentinel_events = sentinel_events.loc[
            sentinel_events["recognition_uniform"].lt(float(sensitivity))
        ]
    if sentinel_events.empty:
        burden = int(natural.final_size)
        return {
            "detected": False,
            "detection_time": pd.NaT,
            "detected_nodes": set(),
            "detection_burden": burden,
            "detection_burden_rate": burden / population_size,
        }
    detection_time = pd.Timestamp(sentinel_events.iloc[0]["time"])
    detected_nodes = set(
        sentinel_events.loc[sentinel_events["time"].le(detection_time), "node_id"]
    )
    burden = int(infections.loc[infections["time"].le(detection_time), "node_id"].nunique())
    return {
        "detected": True,
        "detection_time": detection_time,
        "detected_nodes": detected_nodes,
        "detection_burden": burden,
        "detection_burden_rate": burden / population_size,
    }


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
    sensitivities: list[float],
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
    recognition_seed = _keyed_seed(
        experiment_seed,
        dataset_id,
        anchor.anchor_id,
        parameter.parameter_id,
        random_block,
        initial,
        "sentinel_recognition",
    )
    maximum_budget = max(
        _budget(len(eligible), minimum_budget, value) for value in sentinel_fractions
    )
    sentinel_order = _ordered_history_targets(
        stable_scores, eligible, maximum_budget, sentinel_seed
    )
    rows: list[dict[str, Any]] = []
    for sentinel_fraction in sentinel_fractions:
        sentinel_budget = _budget(len(eligible), minimum_budget, sentinel_fraction)
        sentinels = set(sentinel_order[:sentinel_budget])
        for sensitivity in sensitivities:
            detection = _recognized_detection_metrics(
                natural, sentinels, population_size, sensitivity, recognition_seed
            )
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
                        "recognition_sensitivity": sensitivity,
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


def _loso_allocations(
    family_policy: pd.DataFrame,
    cost_ratios: list[float],
    reference_sentinel_fraction: float,
    reference_response_fraction: float,
) -> pd.DataFrame:
    """Select an allocation without using outcomes from its held-out family."""
    rows: list[dict[str, Any]] = []
    families = sorted(family_policy["system_family"].unique())
    group_columns = ["epidemic_model", "recognition_sensitivity"]
    for key, context in family_policy.groupby(group_columns, observed=True, sort=True):
        model, sensitivity = key
        policies = context[["sentinel_fraction", "response_fraction", "policy"]].drop_duplicates()
        for cost_ratio in cost_ratios:
            nominal_budget = reference_sentinel_fraction + cost_ratio * reference_response_fraction
            feasible = policies.loc[
                policies["sentinel_fraction"]
                + cost_ratio * policies["response_fraction"]
                <= nominal_budget + 1e-12
            ].copy()
            feasible["nominal_cost"] = (
                feasible["sentinel_fraction"]
                + cost_ratio * feasible["response_fraction"]
            )
            for heldout in families:
                training = context.loc[context["system_family"].ne(heldout)].merge(
                    feasible, on=["sentinel_fraction", "response_fraction", "policy"], how="inner"
                )
                scores = (
                    training.groupby(
                        ["sentinel_fraction", "response_fraction", "policy", "nominal_cost"],
                        observed=True,
                    )["mean_value"]
                    .mean()
                    .reset_index(name="training_attack_rate")
                    .sort_values(
                        ["training_attack_rate", "nominal_cost", "response_fraction", "policy"],
                        ascending=[True, True, False, True],
                        kind="stable",
                    )
                )
                if scores.empty:
                    raise ValueError("no feasible allocation available")
                selected = scores.iloc[0]
                held = context.loc[context["system_family"].eq(heldout)]
                chosen = held.loc[held["policy"].eq(selected["policy"]), "mean_value"]
                reference_policy = _policy(
                    reference_sentinel_fraction, reference_response_fraction
                )
                reference = held.loc[held["policy"].eq(reference_policy), "mean_value"]
                if len(chosen) != 1 or len(reference) != 1:
                    raise ValueError("held-out or reference policy is incomplete")
                chosen_value = float(chosen.iloc[0])
                reference_value = float(reference.iloc[0])
                rows.append(
                    {
                        "epidemic_model": model,
                        "recognition_sensitivity": float(sensitivity),
                        "cost_ratio": float(cost_ratio),
                        "system_family": heldout,
                        "analysis_cluster_id": heldout,
                        "selected_policy": selected["policy"],
                        "selected_sentinel_fraction": float(selected["sentinel_fraction"]),
                        "selected_response_fraction": float(selected["response_fraction"]),
                        "selected_nominal_cost": float(selected["nominal_cost"]),
                        "nominal_budget": float(nominal_budget),
                        "training_attack_rate": float(selected["training_attack_rate"]),
                        "heldout_attack_rate": chosen_value,
                        "reference_attack_rate": reference_value,
                        "value": reference_value - chosen_value,
                        "feasible_policies": len(feasible),
                        "training_families": "|".join(
                            name for name in families if name != heldout
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _local_oracle_allocations(
    family_policy: pd.DataFrame,
    cost_ratios: list[float],
    reference_sentinel_fraction: float,
    reference_response_fraction: float,
) -> pd.DataFrame:
    """Compute the retrospective local-calibration ceiling, not a deployable predictor."""
    rows: list[dict[str, Any]] = []
    reference_policy = _policy(reference_sentinel_fraction, reference_response_fraction)
    group_columns = ["epidemic_model", "recognition_sensitivity", "system_family"]
    for key, context in family_policy.groupby(group_columns, observed=True, sort=True):
        model, sensitivity, family = key
        reference = context.loc[context["policy"].eq(reference_policy), "mean_value"]
        if len(reference) != 1:
            raise ValueError("local reference policy is incomplete")
        for cost_ratio in cost_ratios:
            nominal_budget = reference_sentinel_fraction + cost_ratio * reference_response_fraction
            feasible = context.loc[
                context["sentinel_fraction"]
                + cost_ratio * context["response_fraction"]
                <= nominal_budget + 1e-12
            ].copy()
            feasible["nominal_cost"] = (
                feasible["sentinel_fraction"]
                + cost_ratio * feasible["response_fraction"]
            )
            selected = feasible.sort_values(
                ["mean_value", "nominal_cost", "response_fraction", "policy"],
                ascending=[True, True, False, True],
                kind="stable",
            ).iloc[0]
            rows.append(
                {
                    "epidemic_model": model,
                    "recognition_sensitivity": float(sensitivity),
                    "cost_ratio": float(cost_ratio),
                    "system_family": family,
                    "analysis_cluster_id": family,
                    "selected_policy": selected["policy"],
                    "reference_attack_rate": float(reference.iloc[0]),
                    "local_best_attack_rate": float(selected["mean_value"]),
                    "value": float(reference.iloc[0] - selected["mean_value"]),
                }
            )
    return pd.DataFrame(rows)


def _detection_lever_contrasts(family_detection: pd.DataFrame) -> pd.DataFrame:
    """Contrast recognition quality and monitoring coverage at the family level."""
    rows: list[dict[str, Any]] = []
    for (model, family), context in family_detection.groupby(
        ["epidemic_model", "system_family"], observed=True, sort=True
    ):
        values = context.set_index(["recognition_sensitivity", "sentinel_fraction"])[
            "mean_value"
        ]
        contrast_values = {
            "recognition_gain_at_5pct_monitoring": values.loc[(0.5, 0.05)] - values.loc[(1.0, 0.05)],
            "recognition_gain_at_20pct_monitoring": values.loc[(0.5, 0.20)] - values.loc[(1.0, 0.20)],
            "monitoring_gain_at_50pct_recognition": values.loc[(0.5, 0.05)] - values.loc[(0.5, 0.20)],
            "monitoring_gain_at_perfect_recognition": values.loc[(1.0, 0.05)] - values.loc[(1.0, 0.20)],
        }
        contrast_values["recognition_by_monitoring_substitution"] = (
            contrast_values["recognition_gain_at_5pct_monitoring"]
            - contrast_values["recognition_gain_at_20pct_monitoring"]
        )
        for contrast, value in contrast_values.items():
            rows.append(
                {
                    "epidemic_model": model,
                    "contrast": contrast,
                    "system_family": family,
                    "analysis_cluster_id": family,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def _perfect_detection_match(worlds: pd.DataFrame, perfect_path: Path) -> bool:
    perfect = pd.read_csv(
        perfect_path,
        dtype={
            "initial_infected": str,
            "network_id": str,
            "sentinel_nodes": str,
            "response_nodes": str,
        },
    )
    current = worlds.loc[worlds["recognition_sensitivity"].eq(1.0)].copy()
    keys = WORLD_KEYS + ["policy"]
    shared = perfect.merge(current, on=keys, suffixes=("_old", "_new"))
    if len(shared) != len(current):
        return False
    integer_columns = ["final_size", "natural_final_size", "detection_burden"]
    if not all(
        np.array_equal(
            shared[f"{column}_old"].to_numpy(),
            shared[f"{column}_new"].to_numpy(),
        )
        for column in integer_columns
    ):
        return False
    for column in ["sentinel_nodes", "response_nodes"]:
        old = shared[f"{column}_old"].fillna("").astype(str)
        new = shared[f"{column}_new"].fillna("").astype(str).str.removesuffix(".0")
        if not old.equals(new):
            return False
    old_time = pd.to_datetime(shared["detection_time_old"], errors="coerce", format="mixed")
    new_time = pd.to_datetime(shared["detection_time_new"], errors="coerce", format="mixed")
    return bool(old_time.equals(new_time))


def _nested_recognition(worlds: pd.DataFrame) -> bool:
    first = worlds.drop_duplicates(WORLD_KEYS + ["sentinel_fraction", "recognition_sensitivity"])
    for _, frame in first.groupby(WORLD_KEYS + ["sentinel_fraction"], observed=True):
        previous: float | None = None
        for row in frame.sort_values("recognition_sensitivity", ascending=False).itertuples(index=False):
            current = (
                pd.Timestamp(row.detection_time).value
                if pd.notna(row.detection_time)
                else float("inf")
            )
            if previous is not None and current < previous:
                return False
            previous = current
    return True


def _plot_detection(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    colors = {0.5: "#D55E00", 0.75: "#E69F00", 1.0: "#0072B2"}
    for axis, model in zip(axes, sorted(summary["epidemic_model"].unique())):
        subset = summary.loc[summary["epidemic_model"].eq(model)]
        for sensitivity, frame in subset.groupby("recognition_sensitivity", observed=True):
            frame = frame.sort_values("sentinel_fraction")
            x = 100 * frame["sentinel_fraction"].to_numpy(float)
            mean = 100 * frame["family_equal_mean"].to_numpy(float)
            low = 100 * frame["ci_low"].to_numpy(float)
            high = 100 * frame["ci_high"].to_numpy(float)
            axis.errorbar(
                x, mean, yerr=[mean - low, high - mean], marker="o", capsize=3,
                linewidth=2, color=colors[float(sensitivity)], label=f"Sensitivity {sensitivity:.2g}"
            )
        axis.set_title(model.replace("_", " ").upper(), weight="bold")
        axis.set_xlabel("Animals monitored (%)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Population infected before recognized detection (%)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Imperfect recognition delays outbreak detection", fontsize=18, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.14, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_allocation_map(loso: pd.DataFrame, path: Path, dpi: int) -> None:
    models = sorted(loso["epidemic_model"].unique())
    sensitivities = sorted(loso["recognition_sensitivity"].unique(), reverse=True)
    ratios = sorted(loso["cost_ratio"].unique())
    policies = sorted(loso["selected_policy"].unique())
    colors = plt.get_cmap("tab10", max(len(policies), 1))
    mapping = {name: index for index, name in enumerate(policies)}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    for axis, model in zip(axes, models):
        table = loso.loc[loso["epidemic_model"].eq(model)]
        matrix = np.zeros((len(sensitivities), len(ratios)))
        labels: dict[tuple[int, int], str] = {}
        for row, sensitivity in enumerate(sensitivities):
            for column, ratio in enumerate(ratios):
                counts = table.loc[
                    table["recognition_sensitivity"].eq(sensitivity)
                    & table["cost_ratio"].eq(ratio), "selected_policy"
                ].value_counts()
                modal = str(counts.index[0])
                matrix[row, column] = mapping[modal]
                labels[(row, column)] = f"{modal}\n({counts.iloc[0]}/5)"
        axis.imshow(matrix, cmap=colors, vmin=-0.5, vmax=max(len(policies) - 0.5, 0.5), aspect="auto")
        axis.set_xticks(range(len(ratios)), [f"{value:g}" for value in ratios])
        axis.set_yticks(range(len(sensitivities)), [f"{value:.2g}" for value in sensitivities])
        axis.set_xlabel("Response cost / monitoring cost")
        axis.set_title(model.replace("_", " ").upper(), weight="bold")
        for (row, column), label in labels.items():
            axis.text(column, row, label, ha="center", va="center", fontsize=9, color="black")
    axes[0].set_ylabel("Recognition sensitivity")
    fig.suptitle("Policies selected without seeing the held-out animal system", fontsize=18, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.14, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_loso_gain(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    colors = {0.5: "#D55E00", 0.75: "#E69F00", 1.0: "#0072B2"}
    for axis, model in zip(axes, sorted(summary["epidemic_model"].unique())):
        subset = summary.loc[summary["epidemic_model"].eq(model)]
        for sensitivity, frame in subset.groupby("recognition_sensitivity", observed=True):
            frame = frame.sort_values("cost_ratio")
            x = np.arange(len(frame))
            mean = 100 * frame["family_equal_mean"].to_numpy(float)
            low = 100 * frame["ci_low"].to_numpy(float)
            high = 100 * frame["ci_high"].to_numpy(float)
            axis.errorbar(
                x, mean, yerr=[mean - low, high - mean], marker="o", capsize=3,
                linewidth=2, color=colors[float(sensitivity)], label=f"Sensitivity {sensitivity:.2g}"
            )
        ratios = sorted(subset["cost_ratio"].unique())
        axis.set_xticks(range(len(ratios)), [f"{value:g}" for value in ratios])
        axis.axhline(0, color="#555555", linewidth=1)
        axis.set_xlabel("Response cost / monitoring cost")
        axis.set_title(model.replace("_", " ").upper(), weight="bold")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Held-out gain over fixed reference (attack-rate points)")
    axes[1].legend(frameon=False)
    fig.suptitle("Does cost-aware allocation transfer to a new animal system?", fontsize=18, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.14, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_primary_family(loso: pd.DataFrame, decision: dict[str, Any], path: Path, dpi: int) -> None:
    primary = loso.loc[
        loso["cost_ratio"].eq(float(decision["primary_cost_ratio"]))
        & loso["recognition_sensitivity"].eq(float(decision["primary_detection_sensitivity"]))
    ].copy()
    models = sorted(primary["epidemic_model"].unique())
    families = sorted(primary["system_family"].unique())
    y = np.arange(len(families))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    for axis, model in zip(axes, models):
        table = primary.loc[primary["epidemic_model"].eq(model)].set_index("system_family")
        values = np.array([100 * table.loc[name, "value"] for name in families])
        colors = np.where(values >= 0, "#0072B2", "#D55E00")
        axis.axvline(0, color="#555555", linewidth=1)
        axis.scatter(values, y, c=colors, s=55, zorder=3)
        for index, value in enumerate(values):
            axis.plot([0, value], [index, index], color=colors[index], linewidth=2)
        axis.set_xlabel("Gain over fixed reference (attack-rate points)")
        axis.set_title(model.replace("_", " ").upper(), weight="bold")
        axis.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(y, [SYSTEM_FAMILY_LABELS.get(name, name) for name in families])
    fig.suptitle("Pre-registered primary result by held-out animal system", fontsize=18, weight="bold")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.84, bottom=0.14, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_transfer_vs_local(
    loso_summary: pd.DataFrame,
    local_summary: pd.DataFrame,
    decision: dict[str, Any],
    path: Path,
    dpi: int,
) -> None:
    sensitivity = float(decision["primary_detection_sensitivity"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for axis, model in zip(axes, sorted(loso_summary["epidemic_model"].unique())):
        for label, source, color in [
            ("Cross-system selector", loso_summary, "#D55E00"),
            ("Retrospective local ceiling", local_summary, "#0072B2"),
        ]:
            frame = source.loc[
                source["epidemic_model"].eq(model)
                & source["recognition_sensitivity"].eq(sensitivity)
            ].sort_values("cost_ratio")
            x = np.arange(len(frame))
            mean = 100 * frame["family_equal_mean"].to_numpy(float)
            low = 100 * frame["ci_low"].to_numpy(float)
            high = 100 * frame["ci_high"].to_numpy(float)
            axis.errorbar(
                x, mean, yerr=[mean - low, high - mean], marker="o",
                capsize=3, linewidth=2, color=color, label=label
            )
        ratios = sorted(frame["cost_ratio"].unique())
        axis.set_xticks(range(len(ratios)), [f"{value:g}" for value in ratios])
        axis.axhline(0, color="#555555", linewidth=1)
        axis.set_xlabel("Response cost / monitoring cost")
        axis.set_title(model.replace("_", " ").upper(), weight="bold")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Gain over fixed reference (attack-rate points)")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Cross-system transfer versus the local-calibration ceiling",
        fontsize=18,
        weight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.14, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_detection_levers(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    labels = {
        "recognition_gain_at_5pct_monitoring": "Recognition: 50% to 100%\nat 5% monitoring",
        "recognition_gain_at_20pct_monitoring": "Recognition: 50% to 100%\nat 20% monitoring",
        "monitoring_gain_at_50pct_recognition": "Monitoring: 5% to 20%\nat 50% recognition",
        "monitoring_gain_at_perfect_recognition": "Monitoring: 5% to 20%\nat perfect recognition",
        "recognition_by_monitoring_substitution": "Substitution interaction",
    }
    order = list(labels)
    models = sorted(summary["epidemic_model"].unique())
    x = np.arange(len(order))
    width = 0.34
    fig, axis = plt.subplots(figsize=(13.5, 6.2))
    for index, (model, color) in enumerate(zip(models, ["#0072B2", "#E69F00"])):
        table = summary.loc[summary["epidemic_model"].eq(model)].set_index("contrast").loc[order]
        mean = 100 * table["family_equal_mean"].to_numpy(float)
        low = 100 * table["ci_low"].to_numpy(float)
        high = 100 * table["ci_high"].to_numpy(float)
        offset = (index - 0.5) * width
        axis.errorbar(
            x + offset, mean, yerr=[mean - low, high - mean], fmt="o",
            capsize=4, markersize=7, linewidth=2, color=color,
            label=model.replace("_", " ").upper(),
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(x, [labels[name] for name in order])
    axis.set_ylabel("Reduction in infection burden before detection (points)")
    axis.set_title(
        "Two robust levers reduce pre-detection burden",
        fontsize=18,
        weight="bold",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.25)
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
        raise ValueError("capacity-frontier prerequisite audit must pass")
    stable_path = Path(config["data"]["stable_prediction_path"])
    perfect_path = Path(config["data"]["perfect_detection_worlds"])
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
            window for window in windows
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
                support_rows.append({
                    "dataset_id": dataset_id,
                    "network_id": str(window["network_id"]),
                    "anchor_id": anchor.anchor_id,
                    "parameter_id": parameter.parameter_id,
                    "supported": supported,
                })
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
            stable = _matching_stable_scores(
                stable_predictions, dataset_id, network_id, anchor.anchor_time, window["eligible"]
            )
            seeds = stable_hash_order(
                list(map(str, window["eligible"])), int(evaluation["seed"]),
                dataset_id, anchor.anchor_id, "capacity_frontier_seeds"
            )[: int(profile["seeds_per_anchor"])]
            for model in decision["epidemic_models"]:
                tasks.append({
                    "dataset_id": dataset_id,
                    "network_id": network_id,
                    "system_family": str(specification["system_family"]),
                    "analysis_cluster_id": cluster,
                    "window": window,
                    "parameter": parameter,
                    "model": dict(model),
                    "stable_scores": stable,
                    "seeds": seeds,
                })

    worlds_path = results_dir / "policy_worlds.csv.gz"
    if bool(config["execution"].get("resume", True)) and worlds_path.exists():
        worlds = pd.read_csv(
            worlds_path,
            dtype={
                "initial_infected": str,
                "network_id": str,
                "sentinel_nodes": str,
                "response_nodes": str,
            },
        )
    else:
        fingerprint = hashlib.sha256(
            config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes()
        ).hexdigest()[:12]
        frames = []
        progress = tqdm(tasks, desc="Imperfect-recognition worlds", unit="task")
        for task in progress:
            identity = "|".join([
                fingerprint, task["dataset_id"], task["network_id"],
                task["window"]["anchor"].anchor_id, str(task["parameter"].parameter_id),
                str(task["model"]["name"]),
            ])
            checkpoint = checkpoint_dir / f"worlds_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
            frame = pd.DataFrame()
            if bool(config["execution"].get("resume", True)) and checkpoint.exists():
                frame = pd.read_csv(
                    checkpoint,
                    dtype={
                        "initial_infected": str,
                        "network_id": str,
                        "sentinel_nodes": str,
                        "response_nodes": str,
                    },
                )
            if frame.empty:
                task_frames = []
                for block in range(int(profile["random_blocks"])):
                    for initial in task["seeds"]:
                        task_frames.append(_run_world(
                            dataset_id=task["dataset_id"], network_id=task["network_id"],
                            system_family=task["system_family"], analysis_cluster_id=task["analysis_cluster_id"],
                            window=task["window"], parameter=task["parameter"], model=task["model"],
                            stable_scores=task["stable_scores"], initial=str(initial), random_block=block,
                            experiment_seed=int(evaluation["seed"]),
                            sentinel_fractions=list(map(float, decision["sentinel_budget_fractions"])),
                            response_fractions=list(map(float, decision["response_budget_fractions"])),
                            sensitivities=list(map(float, decision["sentinel_recognition_sensitivities"])),
                            minimum_budget=int(decision["minimum_positive_budget"]),
                            action_delay_fraction=float(decision["action_delay_fraction_of_mean_infectious_period"]),
                            residual=float(decision["residual_contact_multiplier"]),
                        ))
                frame = pd.concat(task_frames, ignore_index=True)
                frame.to_csv(checkpoint, index=False, compression="gzip")
            frames.append(frame)
            progress.set_postfix_str(f"{task['dataset_id']} {task['model']['name']}")
        worlds = pd.concat(frames, ignore_index=True)
    repetitions = int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"]))
    policy_summary, family_policy = _hierarchical_summary(
        worlds.assign(value=worlds["final_attack_rate"]), value_column="value",
        group_columns=["epidemic_model", "recognition_sensitivity", "sentinel_fraction", "response_fraction", "policy"],
        bootstrap_replicates=repetitions, seed=int(evaluation["seed"]),
    )
    detection_source = worlds.loc[worlds["response_fraction"].eq(0)].drop_duplicates(
        WORLD_KEYS + ["recognition_sensitivity", "sentinel_fraction"]
    )
    detection_summary, family_detection = _hierarchical_summary(
        detection_source.assign(value=detection_source["detection_burden_rate"]), value_column="value",
        group_columns=["epidemic_model", "recognition_sensitivity", "sentinel_fraction"],
        bootstrap_replicates=repetitions, seed=int(evaluation["seed"]) + 1000,
    )
    loso = _loso_allocations(
        family_policy,
        list(map(float, decision["response_to_monitoring_unit_cost_ratios"])),
        float(decision["reference_sentinel_fraction"]),
        float(decision["reference_response_fraction"]),
    )
    loso_summary, family_loso = _hierarchical_summary(
        loso, value_column="value",
        group_columns=["epidemic_model", "recognition_sensitivity", "cost_ratio"],
        bootstrap_replicates=repetitions, seed=int(evaluation["seed"]) + 2000,
    )
    local = _local_oracle_allocations(
        family_policy,
        list(map(float, decision["response_to_monitoring_unit_cost_ratios"])),
        float(decision["reference_sentinel_fraction"]),
        float(decision["reference_response_fraction"]),
    )
    local_summary, family_local = _hierarchical_summary(
        local, value_column="value",
        group_columns=["epidemic_model", "recognition_sensitivity", "cost_ratio"],
        bootstrap_replicates=repetitions, seed=int(evaluation["seed"]) + 3000,
    )
    detection_contrasts = _detection_lever_contrasts(family_detection)
    detection_contrast_summary, family_detection_contrasts = _hierarchical_summary(
        detection_contrasts, value_column="value",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=repetitions, seed=int(evaluation["seed"]) + 4000,
    )
    primary = loso_summary.loc[
        loso_summary["cost_ratio"].eq(float(decision["primary_cost_ratio"]))
        & loso_summary["recognition_sensitivity"].eq(float(decision["primary_detection_sensitivity"]))
    ].copy()
    primary["interval_pass"] = primary["ci_low"].gt(0)
    primary["family_count_pass"] = primary["positive_families"].ge(3)
    primary["decision"] = np.where(
        primary["interval_pass"] & primary["family_count_pass"], "pass", "fail"
    )

    policy_count = (
        len(decision["sentinel_budget_fractions"])
        * len(decision["response_budget_fractions"])
        * len(decision["sentinel_recognition_sensitivities"])
    )
    counts = worlds.groupby(WORLD_KEYS, observed=True).size()
    natural_counts = worlds.groupby(WORLD_KEYS, observed=True)["natural_final_size"].nunique()
    recognition_shared = worlds.groupby(
        WORLD_KEYS + ["sentinel_fraction", "recognition_sensitivity"], observed=True
    )[["detection_time", "detection_burden"]].nunique(dropna=False)
    expected_folds = worlds["system_family"].nunique()
    checks = {
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "all_requested_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "five_independent_families_full": profile_name != "full" or expected_folds == 5,
        "factorial_worlds_complete": bool(counts.eq(policy_count).all()),
        "natural_world_shared": bool(natural_counts.eq(1).all()),
        "detection_shared_across_response_capacity": bool(recognition_shared.eq(1).all().all()),
        "recognition_is_nested": _nested_recognition(worlds),
        "perfect_detection_reproduces_prerequisite": _perfect_detection_match(worlds, perfect_path),
        "loso_has_all_folds": bool(
            loso.groupby(["epidemic_model", "recognition_sensitivity", "cost_ratio"], observed=True)["system_family"].nunique().eq(expected_folds).all()
        ),
        "no_heldout_leakage": bool(
            loso.apply(lambda row: row.system_family not in str(row.training_families).split("|"), axis=1).all()
        ),
        "selected_allocations_feasible": bool(loso["selected_nominal_cost"].le(loso["nominal_budget"] + 1e-12).all()),
        "attack_rates_bounded": bool(worlds["final_attack_rate"].between(0, 1).all()),
        "no_rewiring": float(decision["rewiring_fraction"]) == 0.0,
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "datasets": int(worlds["dataset_id"].nunique()),
        "families": int(expected_folds),
        "anchors": int(worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0]),
        "natural_worlds": int(worlds[WORLD_KEYS].drop_duplicates().shape[0]),
        "policy_evaluations": len(worlds),
        "primary_decisions": primary.set_index("epidemic_model")["decision"].to_dict(),
        "scope": "cost_aware_imperfect_recognition_loso_allocation",
    }
    if audit["status"] != "pass":
        raise ValueError(f"cost-aware allocation audit failed: {audit}")

    outputs = {
        "policy_worlds.csv.gz": (worlds, {"index": False, "compression": "gzip"}),
        "policy_summary.csv": (policy_summary, {"index": False}),
        "family_policy.csv": (family_policy, {"index": False}),
        "detection_summary.csv": (detection_summary, {"index": False}),
        "family_detection.csv": (family_detection, {"index": False}),
        "loso_allocations.csv": (loso, {"index": False}),
        "loso_summary.csv": (loso_summary, {"index": False}),
        "family_loso.csv": (family_loso, {"index": False}),
        "local_calibration_ceiling.csv": (local, {"index": False}),
        "local_calibration_ceiling_summary.csv": (local_summary, {"index": False}),
        "family_local_calibration_ceiling.csv": (family_local, {"index": False}),
        "detection_lever_contrasts.csv": (detection_contrasts, {"index": False}),
        "detection_lever_contrast_summary.csv": (detection_contrast_summary, {"index": False}),
        "family_detection_lever_contrasts.csv": (family_detection_contrasts, {"index": False}),
        "primary_decisions.csv": (primary, {"index": False}),
        "parameter_time_support.csv": (pd.DataFrame(support_rows), {"index": False}),
    }
    for name, (frame, options) in outputs.items():
        frame.to_csv(results_dir / name, **options)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [config_path, stable_path, prerequisite_path, perfect_path, Path(__file__)]
    pd.DataFrame([
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in source_paths
    ]).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {
        "experiment_id": experiment_id, "profile": profile_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(), "platform": platform.platform(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dpi = int(profile["render_dpi"])
    _plot_detection(detection_summary, report_dir / "detection_sensitivity_curve.png", dpi)
    _plot_allocation_map(loso, report_dir / "cost_allocation_map.png", dpi)
    _plot_loso_gain(loso_summary, report_dir / "loso_allocation_gain.png", dpi)
    _plot_primary_family(loso, decision, report_dir / "heldout_primary_gain.png", dpi)
    _plot_transfer_vs_local(
        loso_summary, local_summary, decision,
        report_dir / "transfer_vs_local_ceiling.png", dpi,
    )
    _plot_detection_levers(
        detection_contrast_summary,
        report_dir / "detection_resource_substitution.png",
        dpi,
    )
    display = primary.copy()
    for column in ["family_equal_mean", "ci_low", "ci_high"]:
        display[column] = 100 * display[column]
    secondary_display = detection_contrast_summary.copy()
    for column in ["family_equal_mean", "ci_low", "ci_high"]:
        secondary_display[column] = 100 * secondary_display[column]
    report = f"""# Cost-aware allocation under imperfect sentinel recognition

This frozen capstone asks whether a monitoring-response allocation learned without
one animal-system family improves on a fixed 10% monitoring + 5% response reference
when applied to that unseen family. Recognition sensitivity and relative unit costs
are explicit decision inputs; they are not fitted to held-out outcomes.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Paired natural worlds: {audit['natural_worlds']}
- Policy evaluations: {audit['policy_evaluations']}
- Technical audit: **{audit['status']}**

The pre-registered primary gate uses recognition sensitivity 0.50 and equal unit costs.
Effects below are held-out attack-rate percentage-point gains over the fixed reference.

{_markdown_table(display)}

The following mechanism contrasts were declared after the primary reveal and are
secondary. Positive values mean fewer infections accrued before recognized detection.

{_markdown_table(secondary_display)}

This experiment supports a transferable allocation claim only where held-out effects
are positive. The retrospective local optimum is only a calibration ceiling and did
not show a large universal improvement opportunity. The experiment does not estimate
field monetary costs, animal welfare costs, or pathogen-specific diagnostic
sensitivity. Those remain deployment inputs.
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
