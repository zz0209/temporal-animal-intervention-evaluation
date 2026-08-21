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
from .role_aware_sentinel_response import (
    WORLD_KEYS,
    _detection_metrics,
    _replay_response,
)
from .sequential_preparedness_update import _budget, _parameters


RESPONSE_METHODS = ["case_only", "random", "history_weight"]
CONTRASTS = ["capacity_increment", "targeting_increment", "total_history_increment"]


def _random_response_targets(
    eligible: set[str],
    budget: int,
    seed: int,
    excluded: set[str] | None = None,
) -> set[str]:
    available = sorted(eligible - (excluded or set()))
    return set(stable_hash_order(available, seed, "matched_random_response")[:budget])


def _compute_decomposition(worlds: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = WORLD_KEYS + [
        "system_family",
        "analysis_cluster_id",
        "population_size",
    ]
    metadata = worlds.drop_duplicates(WORLD_KEYS)[metadata_columns]
    wide = worlds.pivot(
        index=WORLD_KEYS, columns="response_method", values="final_size"
    ).reset_index()
    paired = metadata.merge(wide, on=WORLD_KEYS, validate="one_to_one")
    definitions = {
        "capacity_increment": ("case_only", "random"),
        "targeting_increment": ("random", "history_weight"),
        "total_history_increment": ("case_only", "history_weight"),
    }
    rows = []
    for name, (reference, challenger) in definitions.items():
        frame = paired[metadata_columns].copy()
        frame["contrast"] = name
        frame["value"] = (
            paired[reference] - paired[challenger]
        ) / paired["population_size"].astype(float)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _run_random_arm(
    *,
    dataset_id: str,
    network_id: str,
    system_family: str,
    analysis_cluster_id: str,
    window: dict[str, Any],
    parameter: Any,
    model: dict[str, Any],
    initial: str,
    random_block: int,
    target_replicates: int,
    experiment_seed: int,
    sentinel_fraction: float,
    response_fraction: float,
    minimum_budget: int,
    action_delay_fraction: float,
    residual: float,
    threshold_fraction: float,
) -> list[dict[str, Any]]:
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
    sentinels = set(
        greedy_history_coverage(
            window["history"], eligible, sentinel_budget, seed=sentinel_seed
        )
    )
    if len(sentinels) != sentinel_budget:
        raise AssertionError("coverage sentinels must use the configured capacity")
    detection = _detection_metrics(
        natural, sentinels, population_size, threshold_fraction
    )
    detected_nodes = set(detection["detected_nodes"])
    response_capacity = _budget(len(eligible), minimum_budget, response_fraction)
    response_budget = min(response_capacity, len(eligible - detected_nodes))
    rows = []
    for target_replicate in range(target_replicates):
        response_seed = _keyed_seed(
            experiment_seed,
            dataset_id,
            network_id,
            anchor.anchor_id,
            parameter.parameter_id,
            model["name"],
            random_block,
            initial,
            "random_response",
            target_replicate,
        )
        additional = _random_response_targets(
            eligible, response_budget, response_seed, excluded=detected_nodes
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
        rows.append({
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
        "target_replicate": target_replicate,
        "initial_infected": str(initial),
        "world_seed": world_seed,
        "population_size": population_size,
        "policy": "history_coverage__random",
        "sentinel_method": "history_coverage",
        "response_method": "random",
        "sentinel_budget": sentinel_budget,
        "response_budget": response_budget,
        "response_capacity": response_capacity,
        "sentinel_nodes": "|".join(sorted(sentinels)),
        "response_nodes": "|".join(sorted(targets)),
        "detected": bool(detection["detected"]),
        "detection_time": detection["detection_time"],
        "action_start": action_start,
        "detection_burden": int(detection["detection_burden"]),
        "detection_burden_rate": float(detection["detection_burden_rate"]),
        "early_detection": bool(detection["early_detection"]),
        "natural_final_size": natural.final_size,
        "final_size": result.final_size,
        "final_attack_rate": result.final_size / population_size,
        })
    return rows


def _summaries(
    contrasts: pd.DataFrame, *, bootstrap_replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _hierarchical_summary(
        contrasts,
        value_column="value",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def _classify(summary: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in summary.groupby("epidemic_model", observed=True, sort=True):
        result = frame.loc[frame["contrast"].eq("targeting_increment")].iloc[0]
        family_frame = family.loc[
            family["epidemic_model"].eq(model)
            & family["contrast"].eq("targeting_increment")
        ]
        families = int(result.families)
        required = int(math.ceil(0.8 * families))
        positive = int(family_frame["mean_value"].gt(0).sum())
        if float(result.ci_low) > 0 and positive >= required:
            decision = "strong_targeting_increment"
        elif float(result.family_equal_mean) > 0 and positive >= required:
            decision = "directional_targeting_increment"
        else:
            decision = "targeting_increment_unsupported"
        rows.append(
            {
                "epidemic_model": model,
                "decision": decision,
                "targeting_increment": float(result.family_equal_mean),
                "ci_low": float(result.ci_low),
                "ci_high": float(result.ci_high),
                "positive_families": positive,
                "families": families,
                "required_families": required,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_family_out(
    contrasts: pd.DataFrame, *, bootstrap_replicates: int, seed: int
) -> pd.DataFrame:
    rows = []
    families = sorted(contrasts["system_family"].unique())
    for index, held_out in enumerate(families):
        subset = contrasts.loc[~contrasts["system_family"].eq(held_out)]
        summary, family = _summaries(
            subset,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 1000 * index,
        )
        decision = _classify(summary, family)
        decision.insert(0, "held_out_family", held_out)
        rows.append(decision)
    return pd.concat(rows, ignore_index=True)


def _plot_decomposition(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    labels = {
        "capacity_increment": "Extra capacity: random vs cases only",
        "targeting_increment": "History targeting vs random targets",
        "total_history_increment": "Total: history targets vs cases only",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharex=True, sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = summary.loc[summary["epidemic_model"].eq(model)].set_index("contrast").loc[CONTRASTS]
        y = np.arange(len(CONTRASTS))
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        colors = ["#72B7B2", "#4C78A8", "#F58518"]
        axis.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="none", ecolor="#777777", capsize=4)
        axis.scatter(mean, y, s=85, color=colors, zorder=3)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_yticks(y, [labels[item] for item in CONTRASTS])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("What does history-ranked outbreak response add?", fontsize=21, weight="bold")
    fig.supxlabel("Family-equal reduction in final attack rate (percentage points)", fontsize=14)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.14, wspace=0.12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_family_increment(family: pd.DataFrame, path: Path, dpi: int) -> None:
    frame = family.loc[family["contrast"].eq("targeting_increment")].copy()
    families = sorted(frame["system_family"].unique())
    labels = [SYSTEM_FAMILY_LABELS.get(item, item.replace("_", " ")) for item in families]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        values = frame.loc[frame["epidemic_model"].eq(model)].set_index("system_family").loc[families, "mean_value"]
        y = np.arange(len(families))
        colors = np.where(values.to_numpy(float) > 0, "#4C78A8", "#E45756")
        axis.barh(y, 100 * values.to_numpy(float), color=colors)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("History-ranked targets versus equal-capacity random targets", fontsize=21, weight="bold")
    fig.supxlabel("Mean reduction in final attack rate (percentage points)", fontsize=14)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.82, bottom=0.14, wspace=0.10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _random_distribution_diagnostics(
    random_replicates: pd.DataFrame,
    history_worlds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history_worlds.loc[
        history_worlds["response_method"].eq("history_weight"),
        WORLD_KEYS
        + [
            "system_family",
            "analysis_cluster_id",
            "population_size",
            "final_size",
        ],
    ].rename(columns={"final_size": "history_final_size"})
    paired = random_replicates.merge(
        history,
        on=WORLD_KEYS,
        how="inner",
        validate="many_to_one",
        suffixes=("", "_history"),
    )
    paired["targeting_increment"] = (
        paired["final_size"] - paired["history_final_size"]
    ) / paired["population_size"].astype(float)
    paired["history_win"] = (
        paired["final_size"].gt(paired["history_final_size"]).astype(float)
        + 0.5 * paired["final_size"].eq(paired["history_final_size"]).astype(float)
    )
    world = (
        paired.groupby(WORLD_KEYS, observed=True)
        .agg(
            system_family=("system_family", "first"),
            analysis_cluster_id=("analysis_cluster_id", "first"),
            history_percentile=("history_win", "mean"),
            mean_targeting_increment=("targeting_increment", "mean"),
            random_target_sd=("targeting_increment", "std"),
            distinct_random_outcomes=("final_size", "nunique"),
        )
        .reset_index()
    )
    return paired, world


def _plot_random_distribution(
    diagnostics: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    families = sorted(diagnostics["system_family"].unique())
    labels = [SYSTEM_FAMILY_LABELS.get(item, item.replace("_", " ")) for item in families]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        values = [
            diagnostics.loc[
                diagnostics["epidemic_model"].eq(model)
                & diagnostics["system_family"].eq(family),
                "history_percentile",
            ].to_numpy(float)
            for family in families
        ]
        axis.boxplot(
            values,
            orientation="horizontal",
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#9ECAE1", "edgecolor": "#4C78A8"},
            medianprops={"color": "#1F4E79", "linewidth": 2},
        )
        axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1.2)
        axis.set_xlim(-0.02, 1.02)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Where does the history-ranked response fall among random lists?",
        fontsize=21,
        weight="bold",
    )
    fig.supxlabel(
        "History policy percentile within equal-capacity random outcomes (1 = best)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.84, bottom=0.14, wspace=0.10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile = dict(config["profiles"][profile_name])
    decision = dict(config["decision"])
    evaluation = dict(config["evaluation"])
    source_config = yaml.safe_load(Path(config["data"]["source_role_config"]).read_text(encoding="utf-8"))
    if int(source_config["evaluation"]["seed"]) != int(evaluation["seed"]):
        raise ValueError("source and matched-response experiments must share the experiment seed")
    source_path = Path(config["data"]["source_policy_worlds"])
    source_worlds = pd.read_csv(
        source_path,
        dtype={"initial_infected": str, "network_id": str},
        parse_dates=["anchor_time", "horizon_end", "detection_time", "action_start"],
    )
    source_worlds = source_worlds.loc[
        source_worlds["sentinel_method"].eq(decision["sentinel_method"])
        & source_worlds["response_method"].isin(["case_only", "history_weight"])
        & source_worlds["dataset_id"].isin(profile["datasets"])
    ].copy()
    stable_path = Path(config["data"]["stable_prediction_path"])
    stable_predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    stable_predictions["anchor_time"] = pd.to_datetime(stable_predictions["anchor_time"], format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        dataset_source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, dataset_source_config)
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
                if pd.Timestamp(anchor.anchor_time) + mean_period * float(decision["action_delay_fraction_of_mean_infectious_period"]) < pd.Timestamp(anchor.horizon_end):
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
                "role_aware_seeds",
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
        config_path.read_bytes() + source_path.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()[:12]
    frames = []
    progress = tqdm(tasks, desc="Matched random response worlds", unit="task")
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
        checkpoint = checkpoint_dir / f"random_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
        frame = pd.DataFrame()
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str})
        if frame.empty:
            rows = []
            for block in range(int(profile["random_blocks"])):
                for initial in task["seeds"]:
                    rows.extend(
                        _run_random_arm(
                            dataset_id=task["dataset_id"],
                            network_id=task["network_id"],
                            system_family=task["system_family"],
                            analysis_cluster_id=task["analysis_cluster_id"],
                            window=task["window"],
                            parameter=task["parameter"],
                            model=task["model"],
                            initial=str(initial),
                            random_block=block,
                            target_replicates=int(
                                profile.get("random_target_replicates", 1)
                            ),
                            experiment_seed=int(evaluation["seed"]),
                            sentinel_fraction=float(decision["sentinel_budget_fraction"]),
                            response_fraction=float(decision["response_budget_fraction"]),
                            minimum_budget=int(decision["minimum_budget"]),
                            action_delay_fraction=float(
                                decision[
                                    "action_delay_fraction_of_mean_infectious_period"
                                ]
                            ),
                            residual=float(decision["residual_contact_multiplier"]),
                            threshold_fraction=float(
                                decision["early_detection_threshold_fraction"]
                            ),
                        )
                    )
            frame = pd.DataFrame(rows)
            frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
        progress.set_postfix_str(f"{task['dataset_id']} {task['model']['name']}")

    random_replicates = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True
    )
    random_replicates.to_csv(
        results_dir / "random_target_replicates.csv.gz",
        index=False,
        compression="gzip",
    )
    expected_target_replicates = int(profile.get("random_target_replicates", 1))
    replicate_counts = random_replicates.groupby(WORLD_KEYS, observed=True)[
        "target_replicate"
    ].nunique()
    aggregation = {
        column: "first"
        for column in random_replicates.columns
        if column not in set(WORLD_KEYS + ["target_replicate", "final_size", "final_attack_rate"])
    }
    aggregation.update({"final_size": "mean", "final_attack_rate": "mean"})
    random_worlds = (
        random_replicates.groupby(WORLD_KEYS, observed=True, as_index=False)
        .agg(aggregation)
    )
    random_keys = set(map(tuple, random_worlds[WORLD_KEYS].itertuples(index=False, name=None)))
    available_source_keys = set(map(tuple, source_worlds[WORLD_KEYS].itertuples(index=False, name=None)))
    if not random_keys.issubset(available_source_keys):
        raise AssertionError("matched random arm contains worlds absent from the frozen source experiment")
    source_worlds = source_worlds.merge(
        random_worlds[WORLD_KEYS].drop_duplicates(),
        on=WORLD_KEYS,
        how="inner",
        validate="many_to_one",
    )
    source_keys = set(map(tuple, source_worlds[WORLD_KEYS].itertuples(index=False, name=None)))
    worlds = pd.concat([source_worlds, random_worlds], ignore_index=True)
    contrasts = _compute_decomposition(worlds)
    random_pairs, random_diagnostics = _random_distribution_diagnostics(
        random_replicates,
        source_worlds,
    )
    repetitions = int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"]))
    deletion_repetitions = int(profile.get("deletion_bootstrap_replicates", evaluation["deletion_bootstrap_replicates"]))
    summary, family = _summaries(
        contrasts,
        bootstrap_replicates=repetitions,
        seed=int(evaluation["seed"]),
    )
    decisions = _classify(summary, family)
    deletion = _leave_one_family_out(
        contrasts,
        bootstrap_replicates=deletion_repetitions,
        seed=int(evaluation["seed"]) + 2000,
    )

    counts = worlds.groupby(WORLD_KEYS, observed=True)["response_method"].nunique()
    natural_counts = worlds.groupby(WORLD_KEYS, observed=True)["natural_final_size"].nunique()
    detection_counts = worlds.groupby(WORLD_KEYS, observed=True)[["detection_time", "detection_burden"]].nunique(dropna=False)
    extra = worlds.loc[worlds["response_method"].isin(["random", "history_weight"])].copy()
    extra["response_node_count"] = extra["response_nodes"].fillna("").map(
        lambda value: 0 if value == "" else len(str(value).split("|"))
    )
    budget_pairs = extra.pivot(index=WORLD_KEYS, columns="response_method", values="response_budget")
    target_count_pairs = extra.pivot(
        index=WORLD_KEYS, columns="response_method", values="response_node_count"
    )
    source_natural = source_worlds.drop_duplicates(WORLD_KEYS).set_index(WORLD_KEYS)["natural_final_size"]
    random_natural = random_worlds.set_index(WORLD_KEYS)["natural_final_size"]
    checks = {
        "all_requested_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "five_independent_families_full": profile_name != "full" or worlds["system_family"].nunique() == 5,
        "three_response_arms_complete": bool(counts.eq(3).all()),
        "source_world_keys_exactly_reproduced": source_keys == random_keys,
        "source_natural_outcomes_exactly_reproduced": bool(source_natural.sort_index().equals(random_natural.sort_index())),
        "natural_world_shared": bool(natural_counts.eq(1).all()),
        "detection_shared_across_response_arms": bool(detection_counts.eq(1).all().all()),
        "equal_additional_response_budgets": bool((budget_pairs["random"] == budget_pairs["history_weight"]).all()),
        "equal_total_response_target_counts": bool(
            (target_count_pairs["random"] == target_count_pairs["history_weight"]).all()
        ),
        "finite_decomposition": bool(np.isfinite(contrasts["value"].to_numpy(float)).all()),
        "whole_family_deletion_complete": len(deletion) == worlds["system_family"].nunique() * worlds["epidemic_model"].nunique(),
        "random_target_replicates_complete": bool(
            replicate_counts.eq(expected_target_replicates).all()
        ),
        "random_replicates_share_natural_world": bool(
            random_replicates.groupby(WORLD_KEYS, observed=True)["natural_final_size"]
            .nunique()
            .eq(1)
            .all()
        ),
        "random_replicates_share_detection": bool(
            random_replicates.groupby(WORLD_KEYS, observed=True)[
                ["detection_time", "detection_burden"]
            ]
            .nunique(dropna=False)
            .eq(1)
            .all()
            .all()
        ),
        "random_distribution_worlds_complete": len(random_diagnostics)
        == len(random_worlds),
        "random_distribution_percentiles_bounded": bool(
            random_diagnostics["history_percentile"].between(0, 1).all()
        ),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "datasets": int(worlds["dataset_id"].nunique()),
        "families": int(worlds["system_family"].nunique()),
        "anchors": int(worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0]),
        "paired_worlds": int(worlds[WORLD_KEYS].drop_duplicates().shape[0]),
        "policy_evaluations": len(worlds),
        "random_target_policy_evaluations": len(random_replicates),
        "random_target_replicates_per_world": expected_target_replicates,
        "decisions": decisions.set_index("epidemic_model")["decision"].to_dict(),
        "scope": "matched_capacity_history_versus_random_outbreak_response",
    }
    if audit["status"] != "pass":
        raise ValueError(f"matched response audit failed: {audit}")

    worlds.to_csv(results_dir / "response_worlds.csv.gz", index=False, compression="gzip")
    random_pairs.to_csv(
        results_dir / "random_target_pairwise_comparisons.csv.gz",
        index=False,
        compression="gzip",
    )
    random_diagnostics.to_csv(
        results_dir / "random_target_distribution_diagnostics.csv",
        index=False,
    )
    contrasts.to_csv(results_dir / "paired_decomposition.csv.gz", index=False, compression="gzip")
    summary.to_csv(results_dir / "contrast_summary.csv", index=False)
    family.to_csv(results_dir / "family_contrasts.csv", index=False)
    decisions.to_csv(results_dir / "targeting_decisions.csv", index=False)
    deletion.to_csv(results_dir / "leave_one_family_out_decisions.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [config_path, source_path, stable_path, Path(__file__)]
    pd.DataFrame(
        [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in source_paths]
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
    _plot_decomposition(summary, report_dir / "response_value_decomposition.png", dpi)
    _plot_family_increment(family, report_dir / "family_targeting_increment.png", dpi)
    _plot_random_distribution(
        random_diagnostics,
        report_dir / "history_policy_random_percentile.png",
        dpi,
    )
    display = decisions.copy()
    for column in ["targeting_increment", "ci_low", "ci_high"]:
        display[column] = 100 * display[column]
    report = f"""# Matched-capacity response targeting increment

This experiment preserves the history-coverage sentinel set, detection time, action
delay, response capacity, disease parameters, index case, future contacts, and keyed
epidemic randomness from EXP-20260818-008. It adds one equal-capacity random-response
arm so that response-capacity benefit and history-targeting benefit are separately
identified.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Paired epidemic worlds: {audit['paired_worlds']}
- Policy evaluations: {audit['policy_evaluations']}
- Random target-list evaluations: {audit['random_target_policy_evaluations']}
- Random target lists per epidemic world: {audit['random_target_replicates_per_world']}
- Technical audit: **{audit['status']}**

All effect values below are attack-rate percentage points.

{_markdown_table(display)}

The primary estimand is random-response attack rate minus history-ranked-response
attack rate. Positive values favor historical targeting. These are model-based
effects under the frozen response contract, not field causal effects.
"""
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched-capacity response targeting experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
