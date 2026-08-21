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
    ContactObservationProfile,
    DetectionProfile,
    PairedTemporalSIREngine,
    SIRParameters,
    detection_time_from_seed,
    observe_detected_cases,
    perturbed_pre_detection_scores,
    pre_detection_event_signature,
    select_additional_targets,
    states_at,
)

from .intervention_delivery_sensitivity import (
    _hierarchical_summary,
    _operational_isolation_action,
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


WORLD_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "random_block",
    "initial_infected",
    "world_seed",
]
PROFILE_ORDER = [
    "reference",
    "event_loss_30",
    "tag_failure_10",
    "roster_missing_10",
    "coarse_time_025",
    "binary_intensity",
    "joint_moderate",
    "joint_severe",
]


def _keyed_uniform(seed: int, kind: str, identifier: str) -> float:
    payload = f"{seed}\x1f{kind}\x1f{identifier}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / 2**64


def _profile(specification: dict[str, Any], mean_period: pd.Timedelta) -> ContactObservationProfile:
    fraction = specification.get("time_bin_fraction_of_mean_infectious_period")
    return ContactObservationProfile(
        name=str(specification["name"]),
        event_retention_probability=float(
            specification.get("event_retention_probability", 1.0)
        ),
        tag_retention_probability=float(
            specification.get("tag_retention_probability", 1.0)
        ),
        time_bin=(mean_period * float(fraction) if fraction is not None else None),
        binary_intensity=bool(specification.get("binary_intensity", False)),
    )


def _retained_roster(
    eligible: list[str], probability: float, world_seed: int
) -> set[str]:
    return {
        node
        for node in eligible
        if _keyed_uniform(world_seed, "roster", node) < probability
    }


def _run_window(
    *,
    dataset_id: str,
    network_id: str,
    system_family: str,
    analysis_cluster_id: str,
    window: dict[str, Any],
    parameter: Any,
    stable_scores: pd.DataFrame,
    profile_specs: list[dict[str, Any]],
    methods: list[str],
    seed_nodes: list[str],
    random_blocks: int,
    decision: dict[str, Any],
    experiment_seed: int,
) -> pd.DataFrame:
    anchor = window["anchor"]
    stream = window["future"]
    eligible = list(map(str, window["eligible"]))
    population_size = len(stream.nodes())
    mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
    detection_profile = DetectionProfile(**decision["detection_profile"])
    detection_time = detection_time_from_seed(
        anchor.anchor_time, anchor.horizon_end, mean_period, detection_profile
    )
    if detection_time is None:
        return pd.DataFrame()
    action_start = detection_time + mean_period * float(
        decision["action_delay_fraction_of_mean_infectious_period"]
    )
    if action_start >= anchor.horizon_end:
        return pd.DataFrame()
    parameters = SIRParameters(
        beta=float(parameter.beta),
        recovery_rate=float(parameter.recovery_rate_per_second),
    )
    stable = stable_scores.copy()
    stable["candidate_id"] = stable["candidate_id"].astype(str)
    engine = PairedTemporalSIREngine()
    rows: list[dict[str, Any]] = []
    for block in range(random_blocks):
        for initial in seed_nodes:
            world_seed = _keyed_seed(
                experiment_seed,
                dataset_id,
                anchor.anchor_id,
                parameter.parameter_id,
                block,
                initial,
            )
            natural = engine.simulate(
                stream,
                parameters,
                initial_infected=(initial,),
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
            )
            decision_states = states_at(natural, detection_time)
            detected = observe_detected_cases(
                decision_states,
                trigger_node=str(initial),
                secondary_case_sensitivity=float(decision["secondary_case_sensitivity"]),
                world_seed=world_seed,
            )
            standard_action = _operational_isolation_action(
                "standard_care",
                detected,
                action_start,
                anchor.horizon_end,
                float(decision["residual_contact_multiplier"]),
                float(decision["rewiring_fraction"]),
                str(decision["rewiring_mode"]),
            )
            standard = engine.simulate(
                stream,
                parameters,
                initial_infected=(initial,),
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
                action=standard_action,
            )
            natural_signature = pre_detection_event_signature(natural, action_start)
            if pre_detection_event_signature(standard, action_start) != natural_signature:
                raise AssertionError("standard care diverged before action delivery")
            remaining = max(0, len(eligible) - len(set(eligible) & set(detected)))
            nominal_budget = min(
                remaining,
                max(
                    int(decision["minimum_additional_budget"]),
                    int(math.ceil(remaining * float(decision["additional_budget_fraction"]))),
                )
                if remaining
                else 0,
            )
            for profile_spec in profile_specs:
                contact_profile = _profile(profile_spec, mean_period)
                roster_probability = float(
                    profile_spec.get("roster_retention_probability", 1.0)
                )
                roster = _retained_roster(eligible, roster_probability, world_seed)
                available = roster - set(detected)
                budget = min(nominal_budget, len(available))
                contact_scores, diagnostics = perturbed_pre_detection_scores(
                    stream,
                    detected_nodes=detected,
                    start_time=anchor.anchor_time,
                    detection_time=detection_time,
                    half_life=mean_period
                    * float(decision["tracing_half_life_fraction_of_mean_infectious_period"]),
                    profile=contact_profile,
                    observation_seed=world_seed,
                )
                contact_scores["candidate_id"] = contact_scores["candidate_id"].astype(str)
                score_table = contact_scores.loc[
                    contact_scores["candidate_id"].isin(available)
                ].merge(stable, on="candidate_id", how="left", validate="one_to_one")
                if score_table["stable_score"].isna().any():
                    raise ValueError(
                        f"missing stable scores for {dataset_id}/{network_id}/{anchor.anchor_id}"
                    )
                score_table["infected_at_detection"] = score_table["candidate_id"].map(
                    decision_states
                ).eq("I")
                nonzero_contact_fraction = float(
                    score_table["contact_to_detected"].gt(0).mean()
                    if len(score_table)
                    else 0.0
                )
                for method in methods:
                    targets = select_additional_targets(
                        score_table,
                        method=method,
                        budget=budget,
                        detected_nodes=detected,
                        world_seed=world_seed,
                    )
                    action = _operational_isolation_action(
                        method,
                        tuple(sorted(set(detected) | set(targets))),
                        action_start,
                        anchor.horizon_end,
                        float(decision["residual_contact_multiplier"]),
                        float(decision["rewiring_fraction"]),
                        str(decision["rewiring_mode"]),
                    )
                    augmented = engine.simulate(
                        stream,
                        parameters,
                        initial_infected=(initial,),
                        start_time=anchor.anchor_time,
                        end_time=anchor.horizon_end,
                        world_seed=world_seed,
                        action=action,
                    )
                    if pre_detection_event_signature(augmented, action_start) != natural_signature:
                        raise AssertionError(f"{method} diverged before action delivery")
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "network_id": network_id,
                            "system_family": system_family,
                            "analysis_cluster_id": analysis_cluster_id,
                            "anchor_id": anchor.anchor_id,
                            "anchor_time": anchor.anchor_time,
                            "parameter_id": parameter.parameter_id,
                            "random_block": block,
                            "initial_infected": str(initial),
                            "world_seed": world_seed,
                            "population_size": population_size,
                            "eligible_nodes": len(eligible),
                            "natural_final_size": natural.final_size,
                            "standard_final_size": standard.final_size,
                            "observation_profile": contact_profile.name,
                            "observation_label": str(profile_spec["label"]),
                            "event_retention_probability": contact_profile.event_retention_probability,
                            "tag_retention_probability": contact_profile.tag_retention_probability,
                            "roster_retention_probability": roster_probability,
                            "time_bin_fraction": float(
                                profile_spec.get(
                                    "time_bin_fraction_of_mean_infectious_period", 0.0
                                )
                            ),
                            "binary_intensity": contact_profile.binary_intensity,
                            "detected_nodes": "|".join(detected),
                            "nominal_budget": nominal_budget,
                            "realized_budget": budget,
                            "retained_roster_nodes": len(roster),
                            "retained_tag_nodes": diagnostics["retained_tag_nodes"],
                            "eligible_dyadic_events": diagnostics["eligible_dyadic_events"],
                            "retained_dyadic_events": diagnostics["retained_dyadic_events"],
                            "eligible_group_events": diagnostics["eligible_group_events"],
                            "retained_group_events": diagnostics["retained_group_events"],
                            "nonzero_contact_candidate_fraction": nonzero_contact_fraction,
                            "method": method,
                            "additional_targets": "|".join(targets),
                            "augmented_final_size": augmented.final_size,
                            "attack_rate_reduction": (
                                standard.final_size - augmented.final_size
                            )
                            / population_size,
                        }
                    )
    return pd.DataFrame(rows)


def _pair_methods(worlds: pd.DataFrame) -> pd.DataFrame:
    keys = WORLD_KEYS + ["observation_profile"]
    stable = worlds.loc[
        worlds["method"].eq("stable_watchlist"), keys + ["attack_rate_reduction"]
    ]
    direct = worlds.loc[
        worlds["method"].eq("contact_to_detected"),
        keys + ["system_family", "analysis_cluster_id", "attack_rate_reduction"],
    ]
    paired = direct.merge(
        stable, on=keys, suffixes=("_direct", "_stable"), validate="one_to_one"
    )
    paired["method"] = "contact_to_detected"
    paired["increment"] = (
        paired["attack_rate_reduction_direct"]
        - paired["attack_rate_reduction_stable"]
    )
    return paired


def _pair_reference(worlds: pd.DataFrame) -> pd.DataFrame:
    reference = worlds.loc[
        worlds["observation_profile"].eq("reference"),
        WORLD_KEYS + ["method", "additional_targets", "attack_rate_reduction"],
    ]
    stressed = worlds.loc[
        worlds["observation_profile"].ne("reference"),
        WORLD_KEYS
        + [
            "system_family",
            "analysis_cluster_id",
            "observation_profile",
            "method",
            "additional_targets",
            "attack_rate_reduction",
        ],
    ]
    paired = stressed.merge(
        reference,
        on=WORLD_KEYS + ["method"],
        suffixes=("_stress", "_reference"),
        validate="many_to_one",
    )
    paired["value_degradation"] = (
        paired["attack_rate_reduction_stress"]
        - paired["attack_rate_reduction_reference"]
    )

    def overlap(row: pd.Series) -> float:
        stress = set(filter(None, str(row["additional_targets_stress"]).split("|")))
        reference_set = set(
            filter(None, str(row["additional_targets_reference"]).split("|"))
        )
        union = stress | reference_set
        return len(stress & reference_set) / len(union) if union else 1.0

    paired["selection_jaccard"] = paired.apply(overlap, axis=1)
    return paired


def _pair_boundary_reference(paired: pd.DataFrame) -> pd.DataFrame:
    reference = paired.loc[
        paired["observation_profile"].eq("reference"),
        WORLD_KEYS + ["increment"],
    ]
    stressed = paired.loc[
        paired["observation_profile"].ne("reference"),
        WORLD_KEYS
        + [
            "system_family",
            "analysis_cluster_id",
            "observation_profile",
            "method",
            "increment",
        ],
    ]
    contrast = stressed.merge(
        reference,
        on=WORLD_KEYS,
        suffixes=("_stress", "_reference"),
        validate="many_to_one",
    )
    contrast["boundary_degradation"] = (
        contrast["increment_stress"] - contrast["increment_reference"]
    )
    return contrast


def _plot_policy_value(summary: pd.DataFrame, labels: dict[str, str], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.5, 7.2))
    positions = np.arange(len(PROFILE_ORDER))
    colors = {"stable_watchlist": "#4C78A8", "contact_to_detected": "#E45756"}
    names = {"stable_watchlist": "Stable history", "contact_to_detected": "Live contact"}
    for method, offset in (("stable_watchlist", -0.12), ("contact_to_detected", 0.12)):
        frame = summary.loc[summary["method"].eq(method)].set_index(
            "observation_profile"
        ).reindex(PROFILE_ORDER)
        axis.errorbar(
            positions + offset,
            100 * frame["family_equal_mean"],
            yerr=[
                100 * (frame["family_equal_mean"] - frame["ci_low"]),
                100 * (frame["ci_high"] - frame["family_equal_mean"]),
            ],
            marker="o",
            linewidth=2,
            capsize=3,
            color=colors[method],
            label=names[method],
        )
    axis.axhline(0, color="#444444", linewidth=1)
    axis.set_xticks(positions, [labels[item] for item in PROFILE_ORDER], rotation=24, ha="right")
    axis.set_ylabel("Benefit over detected-case isolation (percentage points)")
    axis.set_title("Policy value under degraded contact observation", fontsize=18, fontweight="bold", pad=16)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_family_boundary(
    family: pd.DataFrame, labels: dict[str, str], path: Path
) -> None:
    pivot = family.pivot(
        index="system_family", columns="observation_profile", values="mean_value"
    ).reindex(columns=PROFILE_ORDER)
    values = 100 * pivot.to_numpy(float)
    limit = max(0.5, float(np.nanmax(np.abs(values))))
    fig, axis = plt.subplots(figsize=(13.2, 6.4))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(PROFILE_ORDER)), [labels[item] for item in PROFILE_ORDER], rotation=24, ha="right")
    axis.set_yticks(range(len(pivot)), [item.replace("_", " ").title() for item in pivot.index])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.1f}", ha="center", va="center", fontsize=9)
    axis.set_title("Live-contact advantage by independent animal-system family", fontsize=18, fontweight="bold", pad=16)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Direct over stable (percentage points)")
    fig.subplots_adjust(left=0.22, right=0.93, top=0.87, bottom=0.27)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_information_guardrail(
    information: pd.DataFrame,
    degradation: pd.DataFrame,
    labels: dict[str, str],
    path: Path,
) -> None:
    direct = degradation.loc[degradation["method"].eq("contact_to_detected")].copy()
    merged = information.merge(
        direct[["observation_profile", "family_equal_mean", "ci_low", "ci_high"]],
        on="observation_profile",
        validate="one_to_one",
    )
    fig, axis = plt.subplots(figsize=(10.5, 7.0))
    offsets = {
        "binary_intensity": (-8, 12),
        "coarse_time_025": (-8, -2),
        "tag_failure_10": (8, 12),
        "roster_missing_10": (8, -2),
    }
    for row in merged.itertuples(index=False):
        axis.scatter(
            100 * row.mean_selection_jaccard,
            100 * row.family_equal_mean,
            s=85,
            color="#E45756",
        )
        x_offset, y_offset = offsets.get(row.observation_profile, (6, 6))
        axis.annotate(
            labels[row.observation_profile],
            (100 * row.mean_selection_jaccard, 100 * row.family_equal_mean),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            ha="right" if x_offset < 0 else "left",
        )
    axis.axhline(0, color="#444444", linewidth=1)
    axis.set_xlabel("Target-set overlap with complete records (%)")
    axis.set_ylabel("Change in live-contact benefit (percentage points)")
    axis.set_title("When observation loss changes decisions and value", fontsize=18, fontweight="bold", pad=16)
    axis.grid(alpha=0.2)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.87, bottom=0.13)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile_config = dict(config["profiles"][profile_name])
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
    source_paths = [
        Path(__file__),
        Path(__file__).parents[1] / "simulation" / "outbreak_response.py",
        Path(__file__).parents[1] / "simulation" / "paired.py",
    ]
    fingerprint = hashlib.sha256(
        config_path.read_bytes()
        + stable_path.read_bytes()
        + b"".join(path.read_bytes() for path in source_paths)
    ).hexdigest()[:12]
    tasks: list[tuple[Any, ...]] = []
    considered: dict[str, int] = {}
    retained: dict[str, int] = {}
    decision = dict(config["decision"])
    detection = DetectionProfile(**decision["detection_profile"])
    for dataset_id in profile_config["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        for window in windows:
            window.setdefault("network_id", default_network_id)
        available = set(
            stable_predictions.loc[
                stable_predictions["dataset_id"].eq(dataset_id),
                ["network_id", "anchor_time"],
            ].itertuples(index=False, name=None)
        )
        windows = [
            window
            for window in windows
            if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time))
            in available
        ]
        maximum = profile_config.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        considered[dataset_id] = len(windows)
        retained[dataset_id] = 0
        parameters = _parameter_pool(
            Path(specification["source_results"]) / "parameter_selection.csv",
            str(config["evaluation"]["parameter_pool"]),
        )
        for window in windows:
            anchor = window["anchor"]
            compatible = []
            for parameter in parameters.itertuples(index=False):
                mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
                decision_time = detection_time_from_seed(
                    anchor.anchor_time, anchor.horizon_end, mean_period, detection
                )
                if decision_time is not None and (
                    decision_time
                    + mean_period
                    * float(decision["action_delay_fraction_of_mean_infectious_period"])
                    < anchor.horizon_end
                ):
                    compatible.append(parameter)
            regimes = _select_parameter_regimes(compatible, "median")
            if not regimes:
                continue
            retained[dataset_id] += 1
            cluster = (
                f"{dataset_id}::{window['network_id']}"
                if specification.get("analysis_cluster") == "network"
                else f"{dataset_id}::{window['network_id']}::{anchor.anchor_id}"
            )
            tasks.append((dataset_id, specification, window, regimes[0][1], cluster))
    frames = []
    progress = tqdm(tasks, desc="Observation-robustness anchors", unit="anchor")
    for dataset_id, specification, window, parameter, cluster in progress:
        anchor = window["anchor"]
        identity = (
            f"{fingerprint}|{dataset_id}|{window['network_id']}|"
            f"{anchor.anchor_id}|{parameter.parameter_id}"
        )
        checkpoint = checkpoint_dir / (
            f"{dataset_id}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.csv.gz"
        )
        expected_profiles = {item["name"] for item in decision["observation_profiles"]}
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(
                checkpoint, dtype={"initial_infected": str}, keep_default_na=False
            )
            if (
                set(frame["method"]) == set(decision["methods"])
                and set(frame["observation_profile"]) == expected_profiles
            ):
                frames.append(frame)
                progress.set_postfix_str(f"{dataset_id} cached")
                continue
        stable = _matching_stable_scores(
            stable_predictions,
            dataset_id,
            str(window["network_id"]),
            anchor.anchor_time,
            window["eligible"],
        )
        seeds = stable_hash_order(
            list(map(str, window["eligible"])),
            int(config["evaluation"]["seed"]),
            dataset_id,
            anchor.anchor_id,
            "contact_observation_seeds",
        )[: int(profile_config["seeds_per_anchor"])]
        frame = _run_window(
            dataset_id=dataset_id,
            network_id=str(window["network_id"]),
            system_family=str(specification["system_family"]),
            analysis_cluster_id=cluster,
            window=window,
            parameter=parameter,
            stable_scores=stable,
            profile_specs=list(decision["observation_profiles"]),
            methods=list(decision["methods"]),
            seed_nodes=seeds,
            random_blocks=int(profile_config["random_blocks"]),
            decision=decision,
            experiment_seed=int(config["evaluation"]["seed"]),
        )
        if frame.empty:
            raise ValueError(f"unsupported observation task: {identity}")
        frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
        progress.set_postfix_str(f"{dataset_id} completed")
    worlds = pd.concat(frames, ignore_index=True)
    paired = _pair_methods(worlds)
    degradation_pairs = _pair_reference(worlds)
    boundary_degradation_pairs = _pair_boundary_reference(paired)
    repetitions = int(
        profile_config.get(
            "bootstrap_replicates", config["evaluation"]["bootstrap_replicates"]
        )
    )
    absolute, absolute_family = _hierarchical_summary(
        worlds,
        value_column="attack_rate_reduction",
        group_columns=["observation_profile", "method"],
        bootstrap_replicates=repetitions,
        seed=int(config["evaluation"]["seed"]),
    )
    relative, relative_family = _hierarchical_summary(
        paired,
        value_column="increment",
        group_columns=["observation_profile", "method"],
        bootstrap_replicates=repetitions,
        seed=int(config["evaluation"]["seed"]) + 1,
    )
    degradation, degradation_family = _hierarchical_summary(
        degradation_pairs,
        value_column="value_degradation",
        group_columns=["observation_profile", "method"],
        bootstrap_replicates=repetitions,
        seed=int(config["evaluation"]["seed"]) + 2,
    )
    boundary_degradation, boundary_degradation_family = _hierarchical_summary(
        boundary_degradation_pairs,
        value_column="boundary_degradation",
        group_columns=["observation_profile", "method"],
        bootstrap_replicates=repetitions,
        seed=int(config["evaluation"]["seed"]) + 3,
    )
    overlap = (
        degradation_pairs.groupby(
            ["observation_profile", "method"], observed=True, as_index=False
        )["selection_jaccard"]
        .mean()
        .rename(columns={"selection_jaccard": "mean_selection_jaccard"})
    )
    information = (
        worlds.groupby("observation_profile", observed=True, as_index=False)
        .agg(
            mean_roster_fraction=(
                "retained_roster_nodes",
                lambda values: float(
                    np.mean(values / worlds.loc[values.index, "eligible_nodes"])
                ),
            ),
            mean_tag_fraction=(
                "retained_tag_nodes",
                lambda values: float(np.mean(values / worlds.loc[values.index, "population_size"])),
            ),
            mean_nonzero_contact_fraction=("nonzero_contact_candidate_fraction", "mean"),
            mean_realized_budget_fraction=(
                "realized_budget",
                lambda values: float(np.mean(values / worlds.loc[values.index, "nominal_budget"].replace(0, np.nan))),
            ),
        )
        .merge(overlap.loc[overlap["method"].eq("contact_to_detected")], on="observation_profile")
    )
    family_gate = relative_family.loc[
        relative_family["observation_profile"].eq("joint_moderate")
    ]
    moderate = relative.loc[
        relative["observation_profile"].eq("joint_moderate")
    ].iloc[0]
    positive_families = int(family_gate["mean_value"].gt(0).sum())
    gate_passed = bool(moderate["family_equal_mean"] > 0 and positive_families >= 4)
    recommendation_rows = []
    for observation_name, frame in absolute.groupby(
        "observation_profile", observed=True
    ):
        best = frame.sort_values("family_equal_mean", ascending=False).iloc[0]
        recommendation_rows.append(
            {
                "observation_profile": observation_name,
                "recommended_action": (
                    str(best["method"]) if best["ci_low"] > 0 else "abstain_from_additional_targeting"
                ),
                "best_method_point_estimate": float(best["family_equal_mean"]),
                "best_method_ci_low": float(best["ci_low"]),
                "best_method_ci_high": float(best["ci_high"]),
            }
        )
    recommendations = pd.DataFrame(recommendation_rows)
    moderate_boundary_degradation = boundary_degradation.loc[
        boundary_degradation["observation_profile"].eq("joint_moderate")
    ].iloc[0]
    profile_counts = worlds.groupby(WORLD_KEYS, observed=True)["observation_profile"].nunique()
    method_counts = worlds.groupby(WORLD_KEYS + ["observation_profile"], observed=True)["method"].nunique()
    natural_counts = worlds.groupby(WORLD_KEYS, observed=True)["natural_final_size"].nunique()
    standard_counts = worlds.groupby(WORLD_KEYS, observed=True)["standard_final_size"].nunique()
    target_sets = worlds.apply(
        lambda row: set(filter(None, str(row.additional_targets).split("|"))), axis=1
    )
    budget_counts = target_sets.map(len)
    moderate_worlds = worlds.loc[worlds["observation_profile"].eq("joint_moderate")]
    severe_worlds = worlds.loc[worlds["observation_profile"].eq("joint_severe")]
    nesting = moderate_worlds[WORLD_KEYS + ["method", "retained_tag_nodes", "retained_roster_nodes", "retained_dyadic_events", "retained_group_events"]].merge(
        severe_worlds[WORLD_KEYS + ["method", "retained_tag_nodes", "retained_roster_nodes", "retained_dyadic_events", "retained_group_events"]],
        on=WORLD_KEYS + ["method"], suffixes=("_moderate", "_severe"), validate="one_to_one"
    )
    nested_columns = ["retained_tag_nodes", "retained_roster_nodes", "retained_dyadic_events", "retained_group_events"]
    expected_families = {
        str(config["data"]["datasets"][dataset_id]["system_family"])
        for dataset_id in profile_config["datasets"]
    }
    audit = {
        "status": "pass",
        "checks": {
            "prerequisite_artifact_passed": prerequisite.get("status") == "pass",
            "all_configured_datasets_retained": set(worlds["dataset_id"]) == set(profile_config["datasets"]),
            "all_configured_independent_system_families_retained": set(
                worlds["system_family"].unique()
            )
            == expected_families,
            "all_windows_accounted": all(0 < retained[key] <= considered[key] for key in considered),
            "all_observation_profiles_complete": bool(profile_counts.eq(len(PROFILE_ORDER)).all()),
            "all_methods_complete": bool(method_counts.eq(len(decision["methods"])).all()),
            "natural_world_is_fixed_across_observation_profiles": bool(natural_counts.eq(1).all()),
            "standard_care_is_fixed_across_observation_profiles": bool(standard_counts.eq(1).all()),
            "selected_targets_respect_realized_budget": bool(budget_counts.eq(worlds["realized_budget"]).all()),
            "joint_severe_is_nested_within_joint_moderate": bool(all(nesting[f"{column}_severe"].le(nesting[f"{column}_moderate"]).all() for column in nested_columns)),
            "finite_outcomes": bool(np.isfinite(worlds["attack_rate_reduction"]).all()),
            "paired_method_rows_reconcile": len(paired) * 2 == len(worlds),
            "reference_degradation_rows_reconcile": len(degradation_pairs) == len(worlds) - len(worlds.loc[worlds["observation_profile"].eq("reference")]),
            "boundary_degradation_rows_reconcile": len(boundary_degradation_pairs) == len(paired) - len(paired.loc[paired["observation_profile"].eq("reference")]),
        },
        "datasets": worlds["dataset_id"].nunique(),
        "system_families": worlds["system_family"].nunique(),
        "anchors": worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0],
        "natural_worlds": worlds[WORLD_KEYS].drop_duplicates().shape[0],
        "policy_evaluations": len(worlds),
        "scientific_result": {
            "moderate_observation_gate_passed": gate_passed,
            "joint_moderate_direct_over_stable": float(moderate["family_equal_mean"]),
            "joint_moderate_ci_low": float(moderate["ci_low"]),
            "joint_moderate_ci_high": float(moderate["ci_high"]),
            "joint_moderate_positive_families": positive_families,
            "joint_moderate_families": int(family_gate["system_family"].nunique()),
            "joint_moderate_boundary_change_from_reference": float(
                moderate_boundary_degradation["family_equal_mean"]
            ),
            "joint_moderate_boundary_change_ci_low": float(
                moderate_boundary_degradation["ci_low"]
            ),
            "joint_moderate_boundary_change_ci_high": float(
                moderate_boundary_degradation["ci_high"]
            ),
            "stress_levels_are_not_field_failure_rate_estimates": True,
        },
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"contact-observation audit failed: {audit}")
    worlds.to_csv(results_dir / "response_worlds.csv.gz", index=False, compression="gzip")
    paired.to_csv(results_dir / "paired_policy_increments.csv.gz", index=False, compression="gzip")
    degradation_pairs.to_csv(results_dir / "paired_profile_degradation.csv.gz", index=False, compression="gzip")
    boundary_degradation_pairs.to_csv(
        results_dir / "paired_boundary_degradation.csv.gz",
        index=False,
        compression="gzip",
    )
    absolute.to_csv(results_dir / "absolute_policy_summary.csv", index=False)
    absolute_family.to_csv(results_dir / "absolute_family_summary.csv", index=False)
    relative.to_csv(results_dir / "relative_policy_summary.csv", index=False)
    relative_family.to_csv(results_dir / "relative_family_summary.csv", index=False)
    degradation.to_csv(results_dir / "profile_degradation_summary.csv", index=False)
    degradation_family.to_csv(results_dir / "profile_degradation_family_summary.csv", index=False)
    boundary_degradation.to_csv(
        results_dir / "boundary_degradation_summary.csv", index=False
    )
    boundary_degradation_family.to_csv(
        results_dir / "boundary_degradation_family_summary.csv", index=False
    )
    information.to_csv(results_dir / "observation_information_summary.csv", index=False)
    recommendations.to_csv(results_dir / "deployment_guardrail.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    labels = {item["name"]: item["label"] for item in decision["observation_profiles"]}
    _plot_policy_value(absolute, labels, report_dir / "observation_policy_value.png")
    _plot_family_boundary(relative_family, labels, report_dir / "observation_family_boundary.png")
    _plot_information_guardrail(information, degradation, labels, report_dir / "information_value_guardrail.png")
    (report_dir / "README.md").write_text(
        "# Contact Observation Robustness\n\n"
        f"Profile: **{profile_name}**. Artifact audit: **{audit['status']}**. "
        f"Moderate observation gate: **{'pass' if gate_passed else 'fail'}**.\n\n"
        "The epidemic world and intervention semantics are fixed across profiles; only information available to target selection is perturbed. Stress levels are generic sensitivity scenarios, not estimated device failure rates.\n\n"
        f"Joint-moderate direct over stable: {100 * float(moderate['family_equal_mean']):+.2f} percentage points "
        f"({100 * float(moderate['ci_low']):+.2f} to {100 * float(moderate['ci_high']):+.2f}); "
        f"{positive_families}/{int(family_gate['system_family'].nunique())} families strictly positive. "
        f"Paired change from complete records: {100 * float(moderate_boundary_degradation['family_equal_mean']):+.2f} points "
        f"({100 * float(moderate_boundary_degradation['ci_low']):+.2f} to {100 * float(moderate_boundary_degradation['ci_high']):+.2f}).\n",
        encoding="utf-8",
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
        "input_hashes": {
            "stable_predictions": _sha256(stable_path),
            "prerequisite_audit": _sha256(prerequisite_path),
        },
        "audit_status": audit["status"],
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contact-observation robustness")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260817-004_contact_observation_robustness.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
