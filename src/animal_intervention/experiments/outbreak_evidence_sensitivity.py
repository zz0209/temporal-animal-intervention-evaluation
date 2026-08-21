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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.evaluation import stable_hash_order
from animal_intervention.simulation import (
    DetectionProfile,
    PairedTemporalSIREngine,
    SIRParameters,
    detection_time_from_seed,
    observe_detected_cases,
    pre_detection_event_signature,
    pre_detection_scores,
    select_additional_targets,
    states_at,
)

from .outbreak_response_pilot import (
    DATASET_LABELS,
    _git_value,
    _isolation_action,
    _keyed_seed,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _selected_parameters,
    _sha256,
)


POLICY_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "detection_profile",
    "evidence_profile",
    "secondary_case_sensitivity",
    "budget_fraction",
    "random_block",
    "initial_infected",
    "world_seed",
]
NATURAL_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "random_block",
    "initial_infected",
    "world_seed",
]


def _run_task(
    *,
    dataset_id: str,
    network_id: str,
    system_family: str,
    analysis_cluster_id: str,
    window: dict[str, Any],
    parameter: Any,
    detection_profile: DetectionProfile,
    evidence_profile: str,
    secondary_case_sensitivity: float,
    budget_fraction: float,
    stable_scores: pd.DataFrame,
    methods: list[str],
    seed_nodes: list[str],
    random_blocks: int,
    minimum_budget: int,
    tracing_half_life_fraction: float,
    experiment_seed: int,
) -> pd.DataFrame:
    anchor = window["anchor"]
    stream = window["future"]
    eligible = list(map(str, window["eligible"]))
    population_size = len(stream.nodes())
    parameters = SIRParameters(
        beta=float(parameter.beta),
        recovery_rate=float(parameter.recovery_rate_per_second),
    )
    mean_infectious_period = pd.Timedelta(
        days=float(parameter.mean_infectious_period_days)
    )
    detection_time = detection_time_from_seed(
        anchor.anchor_time,
        anchor.horizon_end,
        mean_infectious_period,
        detection_profile,
    )
    if detection_time is None:
        return pd.DataFrame()
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
                secondary_case_sensitivity=secondary_case_sensitivity,
                world_seed=world_seed,
            )
            contact_scores = pre_detection_scores(
                stream,
                detected_nodes=detected,
                start_time=anchor.anchor_time,
                detection_time=detection_time,
                half_life=mean_infectious_period * tracing_half_life_fraction,
            )
            contact_scores["candidate_id"] = contact_scores["candidate_id"].astype(str)
            contact_scores = contact_scores.loc[
                contact_scores["candidate_id"].isin(set(eligible))
            ].copy()
            score_table = contact_scores.merge(
                stable,
                on="candidate_id",
                how="left",
                validate="one_to_one",
            )
            if score_table["stable_score"].isna().any():
                raise ValueError(
                    f"missing stable scores for {dataset_id}/{network_id}/{anchor.anchor_id}"
                )
            score_table["infected_at_detection"] = score_table["candidate_id"].map(
                decision_states
            ).eq("I")
            remaining = max(0, len(eligible) - len(set(eligible) & set(detected)))
            budget = min(
                remaining,
                max(minimum_budget, int(math.ceil(remaining * budget_fraction)))
                if remaining
                else 0,
            )
            standard = engine.simulate(
                stream,
                parameters,
                initial_infected=(initial,),
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
                action=_isolation_action(
                    "standard_care",
                    detected,
                    detection_time,
                    anchor.horizon_end,
                ),
            )
            natural_signature = pre_detection_event_signature(natural, detection_time)
            if pre_detection_event_signature(standard, detection_time) != natural_signature:
                raise AssertionError("standard care diverged before detection")
            infectious_nodes = {
                node for node, state in decision_states.items() if state == "I"
            }
            secondary_infectious = infectious_nodes - {str(initial)}
            observed_secondary = set(detected) - {str(initial)}
            for method in methods:
                targets = select_additional_targets(
                    score_table,
                    method=method,
                    budget=budget,
                    detected_nodes=detected,
                    world_seed=world_seed,
                )
                augmented = engine.simulate(
                    stream,
                    parameters,
                    initial_infected=(initial,),
                    start_time=anchor.anchor_time,
                    end_time=anchor.horizon_end,
                    world_seed=world_seed,
                    action=_isolation_action(
                        method,
                        tuple(sorted(set(detected) | set(targets))),
                        detection_time,
                        anchor.horizon_end,
                    ),
                )
                if pre_detection_event_signature(augmented, detection_time) != natural_signature:
                    raise AssertionError(f"{method} diverged before detection")
                selected = score_table.set_index("candidate_id").loc[list(targets)]
                selected_fraction = (
                    float(selected["infected_at_detection"].mean())
                    if len(selected)
                    else 0.0
                )
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "network_id": network_id,
                        "system_family": system_family,
                        "analysis_cluster_id": analysis_cluster_id,
                        "anchor_id": anchor.anchor_id,
                        "anchor_time": anchor.anchor_time,
                        "horizon_end": anchor.horizon_end,
                        "parameter_id": parameter.parameter_id,
                        "beta": float(parameter.beta),
                        "mean_infectious_period_days": float(
                            parameter.mean_infectious_period_days
                        ),
                        "detection_profile": detection_profile.name,
                        "detection_time": detection_time,
                        "evidence_profile": evidence_profile,
                        "secondary_case_sensitivity": secondary_case_sensitivity,
                        "budget_fraction": budget_fraction,
                        "random_block": block,
                        "initial_infected": str(initial),
                        "trigger_state_at_detection": decision_states[str(initial)],
                        "trigger_infectious_at_detection": decision_states[str(initial)] == "I",
                        "world_seed": world_seed,
                        "population_size": population_size,
                        "detected_nodes": "|".join(detected),
                        "detected_cases": len(detected),
                        "secondary_infectious_cases": len(secondary_infectious),
                        "observed_secondary_cases": len(observed_secondary),
                        "additional_budget": budget,
                        "method": method,
                        "additional_targets": "|".join(targets),
                        "selected_infected_fraction": selected_fraction,
                        "natural_final_size": natural.final_size,
                        "standard_final_size": standard.final_size,
                        "augmented_final_size": augmented.final_size,
                        "avoided_infections": standard.final_size - augmented.final_size,
                        "attack_rate_reduction": (
                            standard.final_size - augmented.final_size
                        )
                        / population_size,
                    }
                )
    return pd.DataFrame(rows)


def _paired_increments(worlds: pd.DataFrame, baseline: str) -> pd.DataFrame:
    reference_columns = POLICY_KEYS + ["attack_rate_reduction"]
    reference = worlds.loc[worlds["method"].eq(baseline), reference_columns]
    parts = []
    for method in sorted(set(worlds["method"]) - {baseline}):
        challenger = worlds.loc[
            worlds["method"].eq(method),
            POLICY_KEYS
            + ["system_family", "analysis_cluster_id", "attack_rate_reduction"],
        ]
        paired = challenger.merge(
            reference,
            on=POLICY_KEYS,
            suffixes=("_method", "_baseline"),
            validate="one_to_one",
        )
        paired["method"] = method
        paired["baseline"] = baseline
        paired["increment"] = (
            paired["attack_rate_reduction_method"]
            - paired["attack_rate_reduction_baseline"]
        )
        parts.append(paired)
    return pd.concat(parts, ignore_index=True)


def _hierarchical_summary(
    paired: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "detection_profile",
        "evidence_profile",
        "secondary_case_sensitivity",
        "budget_fraction",
        "method",
    ]
    for key, frame in paired.groupby(group_columns, observed=True, sort=True):
        family_names = sorted(frame["system_family"].unique())
        family_units: dict[str, list[np.ndarray]] = {}
        family_means = []
        for family in family_names:
            family_frame = frame.loc[frame["system_family"].eq(family)]
            units = [
                unit.groupby("anchor_id", observed=True)["increment"].mean().to_numpy(float)
                for _, unit in family_frame.groupby(
                    ["dataset_id", "network_id"], observed=True, sort=True
                )
            ]
            family_units[family] = units
            family_means.append(float(np.mean([values.mean() for values in units])))
        rng = np.random.default_rng(_keyed_seed(seed, *key, "family_bootstrap"))
        draws = np.empty(bootstrap_replicates)
        for draw in range(bootstrap_replicates):
            sampled_families = rng.choice(family_names, len(family_names), replace=True)
            sampled_family_means = []
            for family in sampled_families:
                units = family_units[str(family)]
                sampled_units = rng.integers(0, len(units), len(units))
                unit_means = []
                for unit_index in sampled_units:
                    values = units[int(unit_index)]
                    sampled_anchors = rng.integers(0, len(values), len(values))
                    unit_means.append(float(values[sampled_anchors].mean()))
                sampled_family_means.append(float(np.mean(unit_means)))
            draws[draw] = float(np.mean(sampled_family_means))
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "family_equal_mean": float(np.mean(family_means)),
                "blocked_ci_low": float(np.quantile(draws, 0.025)),
                "blocked_ci_high": float(np.quantile(draws, 0.975)),
                "positive_family_fraction": float(np.mean(np.asarray(family_means) > 0)),
                "families": len(family_names),
                "contexts": frame[["dataset_id", "network_id", "anchor_id"]]
                .drop_duplicates()
                .shape[0],
            }
        )
    return pd.DataFrame(rows)


def _dataset_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(
            [
                "dataset_id",
                "detection_profile",
                "evidence_profile",
                "secondary_case_sensitivity",
                "budget_fraction",
                "method",
            ],
            observed=True,
            sort=True,
        )
        .agg(
            mean_increment=("increment", "mean"),
            positive_world_fraction=("increment", lambda values: float(values.gt(0).mean())),
            worlds=("world_seed", "size"),
        )
        .reset_index()
    )


def _evidence_summary(worlds: pd.DataFrame) -> pd.DataFrame:
    evidence_keys = [column for column in POLICY_KEYS if column != "budget_fraction"]
    unique = worlds.drop_duplicates(evidence_keys).copy()
    unique["secondary_detection_fraction"] = np.where(
        unique["secondary_infectious_cases"].gt(0),
        unique["observed_secondary_cases"] / unique["secondary_infectious_cases"],
        np.nan,
    )
    return (
        unique.groupby(
            ["dataset_id", "detection_profile", "evidence_profile", "secondary_case_sensitivity"],
            observed=True,
            sort=True,
        )
        .agg(
            mean_detected_cases=("detected_cases", "mean"),
            mean_secondary_infectious_cases=("secondary_infectious_cases", "mean"),
            mean_observed_secondary_cases=("observed_secondary_cases", "mean"),
            empirical_secondary_detection_fraction=("secondary_detection_fraction", "mean"),
            worlds=("world_seed", "size"),
        )
        .reset_index()
    )


def _realized_budget_summary(worlds: pd.DataFrame) -> pd.DataFrame:
    unique = worlds.drop_duplicates(POLICY_KEYS).copy()
    return (
        unique.groupby(
            ["dataset_id", "budget_fraction"], observed=True, sort=True
        )
        .agg(
            mean_population_size=("population_size", "mean"),
            mean_additional_budget=("additional_budget", "mean"),
            minimum_additional_budget=("additional_budget", "min"),
            maximum_additional_budget=("additional_budget", "max"),
            distinct_realized_budgets=("additional_budget", "nunique"),
            worlds=("world_seed", "size"),
        )
        .reset_index()
    )


def _plot_robustness(summary: pd.DataFrame, path: Path) -> None:
    methods = ["contact_to_detected", "stable_plus_tracing"]
    profiles = ["early_detection", "delayed_detection"]
    colors = {0.0: "#9E9E9E", 0.5: "#4C78A8", 1.0: "#D95F02"}
    markers = {0.0: "o", 0.5: "s", 1.0: "^"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for row, method in enumerate(methods):
        for column, profile in enumerate(profiles):
            axis = axes[row, column]
            frame = summary.loc[
                summary["method"].eq(method)
                & summary["detection_profile"].eq(profile)
            ]
            for sensitivity, group in frame.groupby(
                "secondary_case_sensitivity", observed=True, sort=True
            ):
                group = group.sort_values("budget_fraction")
                x = 100 * group["budget_fraction"].to_numpy(float)
                y = 100 * group["family_equal_mean"].to_numpy(float)
                low = 100 * group["blocked_ci_low"].to_numpy(float)
                high = 100 * group["blocked_ci_high"].to_numpy(float)
                axis.errorbar(
                    x,
                    y,
                    yerr=[y - low, high - y],
                    color=colors[float(sensitivity)],
                    marker=markers[float(sensitivity)],
                    capsize=3,
                    label=f"{int(100 * sensitivity)}% secondary-case detection",
                )
            axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.18)
            axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
            if column == 0:
                label = "Contact to detected" if method == "contact_to_detected" else "Stable + tracing"
                axis.set_ylabel(f"{label} over stable\nattack-rate difference (percentage points)")
            if row == 1:
                axis.set_xlabel("Additional isolation budget (% of eligible animals)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Response-policy robustness across case ascertainment and budgets", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.945, "Family-equal paired increments over the strictly-prior stable watchlist", ha="center", color="#555555")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.09, hspace=0.30, wspace=0.16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heterogeneity(dataset_summary: pd.DataFrame, path: Path) -> None:
    frame = dataset_summary.loc[dataset_summary["method"].eq("stable_plus_tracing")].copy()
    profiles = ["early_detection", "delayed_detection"]
    datasets = sorted(frame["dataset_id"].unique())
    columns = sorted(
        frame[["secondary_case_sensitivity", "budget_fraction"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    maximum = max(abs(frame["mean_increment"].min()), abs(frame["mean_increment"].max()), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    image = None
    for axis, profile in zip(axes, profiles):
        subset = frame.loc[frame["detection_profile"].eq(profile)]
        matrix = np.full((len(datasets), len(columns)), np.nan)
        for row, dataset_id in enumerate(datasets):
            for column, (sensitivity, budget) in enumerate(columns):
                selected = subset.loc[
                    subset["dataset_id"].eq(dataset_id)
                    & subset["secondary_case_sensitivity"].eq(sensitivity)
                    & subset["budget_fraction"].eq(budget),
                    "mean_increment",
                ]
                if len(selected):
                    matrix[row, column] = 100 * float(selected.iloc[0])
        image = axis.imshow(matrix, cmap="PuOr", norm=TwoSlopeNorm(vmin=-100 * maximum, vcenter=0, vmax=100 * maximum), aspect="auto")
        axis.set_xticks(
            range(len(columns)),
            [f"{int(100*s)}% cases\n{100*b:g}% budget" for s, b in columns],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(range(len(datasets)), [DATASET_LABELS.get(item, item) for item in datasets])
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        for row in range(len(datasets)):
            for column in range(len(columns)):
                if np.isfinite(matrix[row, column]):
                    axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    assert image is not None
    color_axis = fig.add_axes([0.925, 0.25, 0.015, 0.52])
    fig.colorbar(image, cax=color_axis, label="Stable + tracing increment (percentage points)")
    fig.suptitle("Dataset heterogeneity of stable-plus-tracing increments", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.925, "Signed means are shown without suppressing locally negative settings", ha="center", color="#555555")
    fig.subplots_adjust(left=0.19, right=0.89, top=0.86, bottom=0.24, wspace=0.08)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_evidence_mechanism(worlds: pd.DataFrame, path: Path) -> None:
    evidence_keys = [column for column in POLICY_KEYS if column != "budget_fraction"]
    evidence_worlds = worlds.drop_duplicates(evidence_keys).copy()
    evidence_worlds["secondary_detection_fraction"] = np.where(
        evidence_worlds["secondary_infectious_cases"].gt(0),
        evidence_worlds["observed_secondary_cases"]
        / evidence_worlds["secondary_infectious_cases"],
        np.nan,
    )
    case_units = (
        evidence_worlds.groupby(
            [
                "system_family",
                "dataset_id",
                "network_id",
                "anchor_id",
                "secondary_case_sensitivity",
                "detection_profile",
            ],
            observed=True,
        )["secondary_detection_fraction"]
        .mean()
        .reset_index()
    )
    case_families = (
        case_units.groupby(
            ["system_family", "secondary_case_sensitivity", "detection_profile"],
            observed=True,
        )["secondary_detection_fraction"]
        .mean()
        .reset_index()
    )
    case_frame = (
        case_families.groupby(
            ["secondary_case_sensitivity", "detection_profile"], observed=True
        )["secondary_detection_fraction"]
        .mean()
        .reset_index()
    )
    target_units = (
        worlds.loc[
            worlds["method"].isin(
                ["stable_watchlist", "contact_to_detected", "stable_plus_tracing"]
            )
        ]
        .groupby(
            [
                "system_family",
                "dataset_id",
                "network_id",
                "anchor_id",
                "secondary_case_sensitivity",
                "detection_profile",
                "method",
            ],
            observed=True,
        )["selected_infected_fraction"]
        .mean()
        .reset_index()
    )
    target_families = (
        target_units.groupby(
            [
                "system_family",
                "secondary_case_sensitivity",
                "detection_profile",
                "method",
            ],
            observed=True,
        )["selected_infected_fraction"]
        .mean()
        .reset_index()
    )
    target_frame = (
        target_families.groupby(
            ["secondary_case_sensitivity", "detection_profile", "method"],
            observed=True,
        )["selected_infected_fraction"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for profile, group in case_frame.groupby("detection_profile", observed=True):
        axes[0].plot(100 * group["secondary_case_sensitivity"], group["secondary_detection_fraction"], marker="o", label=profile.replace("_", " "))
    axes[0].plot([0, 100], [0, 1], color="#777777", linestyle="--", label="nominal sensitivity")
    axes[0].set_xlabel("Secondary-case detection sensitivity (%)")
    axes[0].set_ylabel("Observed fraction of secondary infectious cases")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Case-ascertainment check", fontweight="bold")
    axes[0].legend(frameon=False)
    method_colors = {"stable_watchlist": "#4C78A8", "contact_to_detected": "#F58518", "stable_plus_tracing": "#D95F02"}
    for method, group in target_frame.groupby("method", observed=True):
        collapsed = group.groupby("secondary_case_sensitivity", observed=True)["selected_infected_fraction"].mean().reset_index()
        axes[1].plot(100 * collapsed["secondary_case_sensitivity"], collapsed["selected_infected_fraction"], marker="o", color=method_colors[str(method)], label=str(method).replace("_", " "))
    axes[1].set_xlabel("Secondary-case detection sensitivity (%)")
    axes[1].set_ylabel("Fraction of additional targets infectious at detection")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Immediate target yield", fontweight="bold")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.suptitle("Observed outbreak evidence and policy target yield", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.88, "Animal systems are equally weighted; target yield is averaged across budgets and detection times", ha="center", color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.16, wspace=0.24)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile_config = config["profiles"][profile_name]
    stable_path = Path(config["data"]["stable_prediction_path"])
    stable_predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    stable_predictions["anchor_time"] = pd.to_datetime(stable_predictions["anchor_time"], format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    fingerprint_payload = config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes()
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:12]
    payloads = []
    for dataset_id in profile_config["datasets"]:
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
            if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time)) in available
        ]
        maximum = profile_config.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        parameters = _selected_parameters(
            Path(specification["source_results"]) / "parameter_selection.csv",
            None,
        )
        if not windows:
            raise ValueError(f"no matched sensitivity windows for {dataset_id}")
        payloads.append((dataset_id, specification, windows, parameters))
    detection_profiles = [DetectionProfile(**item) for item in config["decision"]["detection_profiles"]]
    evidence_profiles = list(config["decision"]["evidence_profiles"])
    budget_fractions = list(map(float, config["decision"]["additional_budget_fractions"]))
    tasks = []
    support_rows = []
    for dataset_id, specification, windows, parameters in payloads:
        for window in windows:
            anchor = window["anchor"]
            compatible_parameters = []
            window_support_rows = []
            for parameter in parameters.itertuples(index=False):
                mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
                supported = all(
                    detection_time_from_seed(anchor.anchor_time, anchor.horizon_end, mean_period, profile) is not None
                    for profile in detection_profiles
                )
                window_support_rows.append({"dataset_id": dataset_id, "network_id": str(window["network_id"]), "anchor_id": anchor.anchor_id, "parameter_id": parameter.parameter_id, "mean_attack_rate": float(parameter.mean_attack_rate), "supported": supported, "selected_for_sensitivity": False})
                if supported:
                    compatible_parameters.append(parameter)
            if not compatible_parameters:
                support_rows.extend(window_support_rows)
                continue
            compatible_parameters.sort(key=lambda item: float(item.mean_attack_rate))
            parameter = compatible_parameters[len(compatible_parameters) // 2]
            for row in window_support_rows:
                if row["parameter_id"] == parameter.parameter_id:
                    row["selected_for_sensitivity"] = True
            support_rows.extend(window_support_rows)
            cluster = (
                f"{dataset_id}::{window['network_id']}"
                if specification.get("analysis_cluster") == "network"
                else f"{dataset_id}::{window['network_id']}::{anchor.anchor_id}"
            )
            for detection in detection_profiles:
                for evidence in evidence_profiles:
                    for budget in budget_fractions:
                        tasks.append((dataset_id, specification, window, parameter, detection, evidence, budget, cluster))
    output_frames = []
    progress = tqdm(tasks, desc="Evidence-sensitivity tasks", unit="task")
    for dataset_id, specification, window, parameter, detection, evidence, budget, cluster in progress:
        anchor = window["anchor"]
        identity = f"{fingerprint}|{dataset_id}|{window['network_id']}|{anchor.anchor_id}|{parameter.parameter_id}|{detection.name}|{evidence['name']}|{budget}"
        checkpoint = checkpoint_dir / f"{dataset_id}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.csv.gz"
        expected_methods = set(config["decision"]["methods"])
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str})
            if not frame.empty and set(frame["method"]) == expected_methods:
                output_frames.append(frame)
                progress.set_postfix_str(f"{dataset_id} cached")
                continue
        stable_scores = _matching_stable_scores(stable_predictions, dataset_id, str(window["network_id"]), anchor.anchor_time, window["eligible"])
        seeds = stable_hash_order(list(map(str, window["eligible"])), int(config["evaluation"]["seed"]), dataset_id, anchor.anchor_id, "sensitivity_seeds")[: int(profile_config["seeds_per_anchor"])]
        frame = _run_task(
            dataset_id=dataset_id,
            network_id=str(window["network_id"]),
            system_family=str(specification["system_family"]),
            analysis_cluster_id=cluster,
            window=window,
            parameter=parameter,
            detection_profile=detection,
            evidence_profile=str(evidence["name"]),
            secondary_case_sensitivity=float(evidence["secondary_case_sensitivity"]),
            budget_fraction=budget,
            stable_scores=stable_scores,
            methods=list(config["decision"]["methods"]),
            seed_nodes=seeds,
            random_blocks=int(profile_config["random_blocks"]),
            minimum_budget=int(config["decision"]["minimum_additional_budget"]),
            tracing_half_life_fraction=float(config["decision"]["tracing_half_life_fraction_of_mean_infectious_period"]),
            experiment_seed=int(config["evaluation"]["seed"]),
        )
        frame.to_csv(checkpoint, index=False, compression="gzip")
        output_frames.append(frame)
        progress.set_postfix_str(f"{dataset_id} completed")
    worlds = pd.concat(output_frames, ignore_index=True)
    for column in ("anchor_time", "horizon_end", "detection_time"):
        worlds[column] = pd.to_datetime(worlds[column], format="mixed")
    paired = _paired_increments(worlds, str(config["evaluation"]["primary_baseline"]))
    family_summary = _hierarchical_summary(paired, bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]), seed=int(config["evaluation"]["seed"]))
    dataset_summary = _dataset_summary(paired)
    evidence_summary = _evidence_summary(worlds)
    realized_budget_summary = _realized_budget_summary(worlds)
    worlds.to_csv(results_dir / "response_worlds.csv.gz", index=False, compression="gzip")
    paired.to_csv(results_dir / "paired_policy_increments.csv.gz", index=False, compression="gzip")
    family_summary.to_csv(results_dir / "family_increment_summary.csv", index=False)
    dataset_summary.to_csv(results_dir / "dataset_increment_summary.csv", index=False)
    evidence_summary.to_csv(results_dir / "evidence_summary.csv", index=False)
    realized_budget_summary.to_csv(results_dir / "realized_budget_summary.csv", index=False)
    pd.DataFrame(support_rows).to_csv(results_dir / "parameter_detection_support.csv", index=False)
    policy_groups = POLICY_KEYS
    method_counts = worlds.groupby(policy_groups, observed=True)["method"].nunique()
    budget_counts = worlds["additional_targets"].fillna("").map(lambda value: 0 if not value else len(str(value).split("|")))
    detected_disjoint = worlds.apply(lambda row: not bool(set(str(row.detected_nodes).split("|")) & set(str(row.additional_targets).split("|")) if str(row.additional_targets) else set()), axis=1)
    natural_matrix = worlds.drop_duplicates(POLICY_KEYS).groupby(NATURAL_KEYS, observed=True).agg(outcomes=("natural_final_size", "nunique"), conditions=("evidence_profile", "nunique"))
    standard_budget = worlds.drop_duplicates(POLICY_KEYS).groupby(NATURAL_KEYS + ["detection_profile", "evidence_profile"], observed=True)["standard_final_size"].nunique()
    detection_counts = worlds.drop_duplicates(POLICY_KEYS).groupby(NATURAL_KEYS + ["detection_profile", "secondary_case_sensitivity"], observed=True)["detected_cases"].first().unstack("secondary_case_sensitivity")
    monotone_detection = bool((detection_counts.diff(axis=1).iloc[:, 1:].fillna(0) >= 0).all().all())
    zero_only_trigger = worlds.loc[worlds["secondary_case_sensitivity"].eq(0), "detected_cases"].eq(1).all()
    complete_rows = worlds.loc[worlds["secondary_case_sensitivity"].eq(1)].drop_duplicates(POLICY_KEYS)
    complete_detection = complete_rows["observed_secondary_cases"].eq(
        complete_rows["secondary_infectious_cases"]
    ).all()
    audit = {
        "status": "pass",
        "checks": {
            "policy_keys_unique_per_method": not worlds.duplicated(POLICY_KEYS + ["method"]).any(),
            "all_methods_complete": bool(method_counts.eq(len(config["decision"]["methods"])).all()),
            "fixed_budget": bool(budget_counts.eq(worlds["additional_budget"]).all()),
            "detected_excluded_from_targets": bool(detected_disjoint.all()),
            "natural_world_shared_across_matrix": bool(natural_matrix["outcomes"].eq(1).all()),
            "standard_care_shared_across_budgets": bool(standard_budget.eq(1).all()),
            "detected_cases_monotone_with_sensitivity": monotone_detection,
            "zero_sensitivity_is_trigger_only": bool(zero_only_trigger),
            "complete_sensitivity_detects_all_secondary_infectious_cases": bool(complete_detection),
            "finite_outcomes": bool(np.isfinite(worlds[["attack_rate_reduction", "selected_infected_fraction"]].to_numpy(float)).all()),
            "paired_rows_reconcile": len(paired) == len(worlds.loc[worlds["method"].ne(config["evaluation"]["primary_baseline"])]),
        },
        "datasets": worlds["dataset_id"].nunique(),
        "system_families": worlds["system_family"].nunique(),
        "anchors": worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0],
        "natural_worlds": worlds[NATURAL_KEYS].drop_duplicates().shape[0],
        "policy_worlds": len(worlds),
        "matrix_cells": len(detection_profiles) * len(evidence_profiles) * len(budget_fractions),
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"evidence-sensitivity audit failed: {audit}")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _plot_robustness(family_summary, report_dir / "policy_robustness.png")
    _plot_heterogeneity(dataset_summary, report_dir / "dataset_heterogeneity.png")
    _plot_evidence_mechanism(worlds, report_dir / "evidence_mechanism.png")
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
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    primary = family_summary.loc[family_summary["method"].eq(config["evaluation"]["primary_method"])]
    readme = "# Outbreak-evidence and budget sensitivity\n\n" + f"Datasets: {audit['datasets']}; independent families: {audit['system_families']}; anchors: {audit['anchors']}; natural worlds: {audit['natural_worlds']}; policy evaluations: {audit['policy_worlds']}. Audit: **{audit['status']}**.\n\nThe index case is always the observed trigger. Secondary-case sensitivity applies only to other animals infectious at the decision time; specificity is fixed at one. The median compatible previously selected disease scenario is used for each dataset and window.\n\nPrimary family-equal results are stored in `family_increment_summary.csv`. Realized integer budgets, including settings where multiple percentages map to the same animal count, are explicit in `realized_budget_summary.csv`.\n"
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run outbreak-evidence and budget sensitivity experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260816-009_outbreak_evidence_sensitivity.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
