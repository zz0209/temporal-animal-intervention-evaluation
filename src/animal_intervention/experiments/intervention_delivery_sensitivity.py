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
    InterventionAction,
    PairedTemporalSEIREngine,
    PairedTemporalSIREngine,
    SEIRParameters,
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
    _keyed_seed,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _selected_parameters,
    _sha256,
)


SYSTEM_FAMILY_LABELS = {
    "domestic_sheep_sirtrack": "Domestic sheep",
    "guinea_baboons_sociopatterns": "Guinea baboons",
    "linked_wytham_songbird_family": "Linked Wytham songbirds",
    "oxford_wildbird_network": "Oxford wild birds",
    "radolfzell_great_tits_ontogeny": "Radolfzell great tits",
    "wild_vampire_bats_proximity": "Vampire bats",
    "free_ranging_sheep_fission_fusion": "Free-ranging sheep",
}


POLICY_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "detection_profile",
    "action_delay_fraction",
    "residual_contact_multiplier",
    "secondary_case_sensitivity",
    "false_positive_rate",
    "rewiring_fraction",
    "rewiring_mode",
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


def _spearman_correlation(first: pd.Series, second: pd.Series) -> float:
    """Compute Spearman correlation without an optional SciPy dependency."""

    if len(first) < 2 or first.nunique() < 2 or second.nunique() < 2:
        return float("nan")
    return float(first.rank(method="average").corr(second.rank(method="average")))


def _parameter_pool(path: Path, pool: str) -> pd.DataFrame:
    """Load a preregistered epidemic-parameter pool from an existing calibration."""

    if pool == "selected":
        return _selected_parameters(path, None)
    if pool != "informative":
        raise ValueError(f"unsupported parameter pool: {pool}")
    parameters = pd.read_csv(path)
    flag = parameters["informative"]
    keep = flag if flag.dtype == bool else flag.astype(str).str.strip().str.lower().eq("true")
    parameters = parameters.loc[keep].copy()
    parameters = parameters.sort_values("mean_attack_rate", kind="stable").reset_index(drop=True)
    parameters["recovery_rate_per_second"] = (
        1.0 / parameters["mean_infectious_period_days"] / 86_400
    )
    return parameters


def _select_parameter_regimes(
    compatible: list[Any], mode: str
) -> list[tuple[str, Any]]:
    """Select fixed within-window disease regimes without using policy outcomes."""

    ordered = sorted(compatible, key=lambda item: float(item.mean_attack_rate))
    if mode == "median":
        return [("reference", ordered[len(ordered) // 2])] if ordered else []
    if mode != "attack_rate_triplet":
        raise ValueError(f"unsupported parameter selection mode: {mode}")
    if len(ordered) < 3:
        return []
    indices = np.linspace(0, len(ordered) - 1, num=3).round().astype(int)
    if len(np.unique(indices)) != 3:
        return []
    return list(zip(("low", "middle", "high"), (ordered[index] for index in indices)))


def _operational_isolation_action(
    name: str,
    targets: tuple[str, ...],
    action_start: pd.Timestamp,
    end_time: pd.Timestamp,
    residual_contact_multiplier: float,
    rewiring_fraction: float = 0.0,
    rewiring_mode: str = "none",
) -> InterventionAction:
    if not 0 <= residual_contact_multiplier <= 1:
        raise ValueError("residual contact multiplier must be between zero and one")
    return InterventionAction(
        name=name,
        action_type="isolation",
        target_nodes=tuple(sorted(set(targets))),
        start_time=pd.Timestamp(action_start),
        end_time=pd.Timestamp(end_time),
        contact_multiplier=float(residual_contact_multiplier),
        rewiring_fraction=float(rewiring_fraction),
        rewiring_mode=str(rewiring_mode),
    )


def _run_task(
    *,
    dataset_id: str,
    network_id: str,
    system_family: str,
    analysis_cluster_id: str,
    window: dict[str, Any],
    parameter: Any,
    detection_profile: DetectionProfile,
    action_delay_fraction: float,
    residual_contact_multiplier: float,
    stable_scores: pd.DataFrame,
    methods: list[str],
    seed_nodes: list[str],
    random_blocks: int,
    minimum_budget: int,
    budget_fraction: float,
    secondary_case_sensitivity: float,
    false_positive_rate: float,
    rewiring_fraction: float,
    rewiring_mode: str,
    tracing_half_life_fraction: float,
    experiment_seed: int,
    epidemic_model: dict[str, Any] | None = None,
) -> pd.DataFrame:
    anchor = window["anchor"]
    stream = window["future"]
    eligible = list(map(str, window["eligible"]))
    population_size = len(stream.nodes())
    mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
    model = epidemic_model or {"name": "temporal_sir"}
    model_name = str(model.get("name", "temporal_sir"))
    if model_name == "temporal_sir":
        parameters = SIRParameters(
            beta=float(parameter.beta),
            recovery_rate=float(parameter.recovery_rate_per_second),
        )
        engine = PairedTemporalSIREngine()
    elif model_name == "temporal_seir_erlang":
        latent_fraction = float(model["latent_period_fraction_of_mean_infectious_period"])
        if latent_fraction <= 0:
            raise ValueError("SEIR latent-period fraction must be positive")
        parameters = SEIRParameters(
            beta=float(parameter.beta),
            latent_rate=1.0 / (mean_period.total_seconds() * latent_fraction),
            recovery_rate=float(parameter.recovery_rate_per_second),
            latent_stages=int(model.get("latent_stages", 2)),
            infectious_stages=int(model.get("infectious_stages", 3)),
        )
        engine = PairedTemporalSEIREngine()
    else:
        raise ValueError(f"unsupported epidemic model: {model_name}")
    detection_time = detection_time_from_seed(
        anchor.anchor_time, anchor.horizon_end, mean_period, detection_profile
    )
    if detection_time is None:
        return pd.DataFrame()
    action_start = detection_time + mean_period * action_delay_fraction
    if action_start >= anchor.horizon_end:
        return pd.DataFrame()
    stable = stable_scores.copy()
    stable["candidate_id"] = stable["candidate_id"].astype(str)
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
                false_positive_rate=false_positive_rate,
                world_seed=world_seed,
            )
            infectious_nodes = {
                node for node, state in decision_states.items() if state == "I"
            }
            secondary_infectious = infectious_nodes - {str(initial)}
            detected_set = set(detected)
            infectious_detected_count = len(detected_set & infectious_nodes)
            noninfectious_detected_count = len(detected_set - infectious_nodes)
            generated_false_positive_count = len(
                (detected_set - {str(initial)}) - infectious_nodes
            )
            detected_secondary_count = len(detected_set & secondary_infectious)
            contact_scores = pre_detection_scores(
                stream,
                detected_nodes=detected,
                start_time=anchor.anchor_time,
                detection_time=detection_time,
                half_life=mean_period * tracing_half_life_fraction,
            )
            contact_scores["candidate_id"] = contact_scores["candidate_id"].astype(str)
            contact_scores = contact_scores.loc[
                contact_scores["candidate_id"].isin(set(eligible))
            ].copy()
            score_table = contact_scores.merge(
                stable, on="candidate_id", how="left", validate="one_to_one"
            )
            if score_table["stable_score"].isna().any():
                raise ValueError(
                    f"missing stable scores for {dataset_id}/{network_id}/{anchor.anchor_id}"
                )
            score_table["infected_at_detection"] = score_table["candidate_id"].map(
                decision_states
            ).eq("I")
            case_contact_evidence_mass = float(
                score_table["contact_to_detected"].sum()
            )
            case_contact_evidence_nodes = int(
                score_table["contact_to_detected"].gt(0).sum()
            )
            remaining = max(0, len(eligible) - len(set(eligible) & set(detected)))
            budget = min(
                remaining,
                max(minimum_budget, int(math.ceil(remaining * budget_fraction)))
                if remaining
                else 0,
            )
            standard_action = _operational_isolation_action(
                "standard_care",
                detected,
                action_start,
                anchor.horizon_end,
                residual_contact_multiplier,
                rewiring_fraction,
                rewiring_mode,
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
            standard_hazard = engine.intervention_hazard_accounting(
                stream,
                start_time=action_start,
                end_time=anchor.horizon_end,
                action=standard_action,
            )
            natural_signature = pre_detection_event_signature(natural, action_start)
            if pre_detection_event_signature(standard, action_start) != natural_signature:
                raise AssertionError("standard care diverged before action delivery")
            for method in methods:
                targets = select_additional_targets(
                    score_table,
                    method=method,
                    budget=budget,
                    detected_nodes=detected,
                    world_seed=world_seed,
                )
                augmented_action = _operational_isolation_action(
                    method,
                    tuple(sorted(set(detected) | set(targets))),
                    action_start,
                    anchor.horizon_end,
                    residual_contact_multiplier,
                    rewiring_fraction,
                    rewiring_mode,
                )
                augmented = engine.simulate(
                    stream,
                    parameters,
                    initial_infected=(initial,),
                    start_time=anchor.anchor_time,
                    end_time=anchor.horizon_end,
                    world_seed=world_seed,
                    action=augmented_action,
                )
                augmented_hazard = engine.intervention_hazard_accounting(
                    stream,
                    start_time=action_start,
                    end_time=anchor.horizon_end,
                    action=augmented_action,
                )
                if pre_detection_event_signature(augmented, action_start) != natural_signature:
                    raise AssertionError(f"{method} diverged before action delivery")
                selected = score_table.set_index("candidate_id").loc[list(targets)]
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
                        "mean_infectious_period_days": float(parameter.mean_infectious_period_days),
                        "epidemic_model": model_name,
                        "latent_period_fraction": float(
                            model.get("latent_period_fraction_of_mean_infectious_period", 0.0)
                        ),
                        "latent_stages": int(model.get("latent_stages", 0)),
                        "infectious_stages": int(model.get("infectious_stages", 1)),
                        "detection_profile": detection_profile.name,
                        "detection_time": detection_time,
                        "action_start": action_start,
                        "action_delay_fraction": action_delay_fraction,
                        "residual_contact_multiplier": residual_contact_multiplier,
                        "isolation_contact_reduction": 1.0 - residual_contact_multiplier,
                        "secondary_case_sensitivity": secondary_case_sensitivity,
                        "false_positive_rate": false_positive_rate,
                        "rewiring_fraction": rewiring_fraction,
                        "rewiring_mode": rewiring_mode,
                        "budget_fraction": budget_fraction,
                        "random_block": block,
                        "initial_infected": str(initial),
                        "world_seed": world_seed,
                        "population_size": population_size,
                        "detected_nodes": "|".join(detected),
                        "detected_cases": len(detected),
                        "infectious_cases_at_detection": len(infectious_nodes),
                        "secondary_infectious_cases_at_detection": len(secondary_infectious),
                        "detected_infectious_cases": infectious_detected_count,
                        "detected_noninfectious_animals": noninfectious_detected_count,
                        "generated_false_positives": generated_false_positive_count,
                        "detected_actionable_infectious_fraction": (
                            infectious_detected_count / len(detected) if detected else 0.0
                        ),
                        "detected_secondary_recall": (
                            detected_secondary_count / len(secondary_infectious)
                            if secondary_infectious
                            else 1.0
                        ),
                        "case_contact_evidence_mass": case_contact_evidence_mass,
                        "case_contact_evidence_nodes": case_contact_evidence_nodes,
                        "case_contact_evidence_node_fraction": (
                            case_contact_evidence_nodes / population_size
                            if population_size
                            else 0.0
                        ),
                        "additional_budget": budget,
                        "method": method,
                        "additional_targets": "|".join(targets),
                        "selected_infected_fraction": float(
                            selected["infected_at_detection"].mean() if len(selected) else 0.0
                        ),
                        "original_hazard_mass": augmented_hazard.original_hazard_mass,
                        "standard_removed_hazard_mass": standard_hazard.removed_original_hazard_mass,
                        "standard_rewired_hazard_mass": standard_hazard.rewired_hazard_mass,
                        "augmented_removed_hazard_mass": augmented_hazard.removed_original_hazard_mass,
                        "augmented_rewired_hazard_mass": augmented_hazard.rewired_hazard_mass,
                        "additional_removed_hazard_mass": (
                            augmented_hazard.removed_original_hazard_mass
                            - standard_hazard.removed_original_hazard_mass
                        ),
                        "additional_rewired_hazard_mass": (
                            augmented_hazard.rewired_hazard_mass
                            - standard_hazard.rewired_hazard_mass
                        ),
                        "additional_removed_hazard_fraction": (
                            (
                                augmented_hazard.removed_original_hazard_mass
                                - standard_hazard.removed_original_hazard_mass
                            )
                            / augmented_hazard.original_hazard_mass
                            if augmented_hazard.original_hazard_mass > 0
                            else 0.0
                        ),
                        "additional_rewired_hazard_fraction": (
                            (
                                augmented_hazard.rewired_hazard_mass
                                - standard_hazard.rewired_hazard_mass
                            )
                            / augmented_hazard.original_hazard_mass
                            if augmented_hazard.original_hazard_mass > 0
                            else 0.0
                        ),
                        "natural_final_size": natural.final_size,
                        "standard_final_size": standard.final_size,
                        "augmented_final_size": augmented.final_size,
                        "avoided_infections": standard.final_size - augmented.final_size,
                        "attack_rate_reduction": (
                            standard.final_size - augmented.final_size
                        ) / population_size,
                    }
                )
    return pd.DataFrame(rows)


def _paired_increments(worlds: pd.DataFrame, baseline: str) -> pd.DataFrame:
    regime_columns = ["disease_regime"] if "disease_regime" in worlds.columns else []
    reference = worlds.loc[
        worlds["method"].eq(baseline), POLICY_KEYS + regime_columns + ["attack_rate_reduction"]
    ]
    parts = []
    for method in sorted(set(worlds["method"]) - {baseline}):
        challenger = worlds.loc[
            worlds["method"].eq(method),
            POLICY_KEYS
            + regime_columns
            + ["system_family", "analysis_cluster_id", "attack_rate_reduction"],
        ]
        paired = challenger.merge(
            reference,
            on=POLICY_KEYS + regime_columns,
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


def _detection_timing_contrasts(paired: pd.DataFrame) -> pd.DataFrame:
    """Pair delayed- versus early-detection policy increments in each natural world."""

    regime_columns = ["disease_regime"] if "disease_regime" in paired.columns else []
    keys = NATURAL_KEYS + regime_columns + [
        "action_delay_fraction",
        "residual_contact_multiplier",
        "secondary_case_sensitivity",
        "false_positive_rate",
        "rewiring_fraction",
        "rewiring_mode",
        "method",
    ]
    early = paired.loc[
        paired["detection_profile"].eq("early_detection"),
        keys + ["increment"],
    ]
    delayed = paired.loc[
        paired["detection_profile"].eq("delayed_detection"),
        keys + ["system_family", "analysis_cluster_id", "increment"],
    ]
    contrast = delayed.merge(
        early,
        on=keys,
        suffixes=("_delayed", "_early"),
        validate="one_to_one",
    )
    contrast["timing_contrast"] = (
        contrast["increment_delayed"] - contrast["increment_early"]
    )
    return contrast


def _rewiring_mechanism_contrasts(worlds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contrast outcome and redistributed hazard for direct versus stable targeting."""

    metric_columns = [
        "attack_rate_reduction",
        "additional_removed_hazard_fraction",
        "additional_rewired_hazard_fraction",
    ]
    stable = worlds.loc[
        worlds["method"].eq("stable_watchlist"), POLICY_KEYS + metric_columns
    ]
    direct = worlds.loc[
        worlds["method"].eq("contact_to_detected"),
        POLICY_KEYS + ["system_family", "analysis_cluster_id"] + metric_columns,
    ]
    paired = direct.merge(
        stable,
        on=POLICY_KEYS,
        suffixes=("_direct", "_stable"),
        validate="one_to_one",
    )
    paired["direct_gain_over_stable"] = (
        paired["attack_rate_reduction_direct"]
        - paired["attack_rate_reduction_stable"]
    )
    paired["stable_minus_direct_removed_hazard_fraction"] = (
        paired["additional_removed_hazard_fraction_stable"]
        - paired["additional_removed_hazard_fraction_direct"]
    )
    paired["stable_minus_direct_rewired_hazard_fraction"] = (
        paired["additional_rewired_hazard_fraction_stable"]
        - paired["additional_rewired_hazard_fraction_direct"]
    )
    cell_columns = [
        "detection_profile",
        "rewiring_fraction",
        "system_family",
        "analysis_cluster_id",
    ]
    cluster = (
        paired.groupby(cell_columns, observed=True)[
            [
                "direct_gain_over_stable",
                "stable_minus_direct_removed_hazard_fraction",
                "stable_minus_direct_rewired_hazard_fraction",
            ]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for key, group in cluster.groupby(
        ["detection_profile", "rewiring_fraction"], observed=True, sort=True
    ):
        family = (
            group.groupby("system_family", observed=True)[
                [
                    "direct_gain_over_stable",
                    "stable_minus_direct_removed_hazard_fraction",
                    "stable_minus_direct_rewired_hazard_fraction",
                ]
            ]
            .mean()
            .reset_index()
        )
        rows.append(
            {
                "detection_profile": key[0],
                "rewiring_fraction": key[1],
                "families": family["system_family"].nunique(),
                "clusters": len(group),
                "family_equal_direct_gain": family["direct_gain_over_stable"].mean(),
                "family_equal_stable_minus_direct_removed_hazard_fraction": family[
                    "stable_minus_direct_removed_hazard_fraction"
                ].mean(),
                "family_equal_stable_minus_direct_rewired_hazard_fraction": family[
                    "stable_minus_direct_rewired_hazard_fraction"
                ].mean(),
                "positive_gain_families": int(
                    family["direct_gain_over_stable"].gt(0).sum()
                ),
                "stable_rewires_more_families": int(
                    family["stable_minus_direct_rewired_hazard_fraction"].gt(0).sum()
                ),
                "cluster_spearman_gain_vs_rewired_difference": _spearman_correlation(
                    group["direct_gain_over_stable"],
                    group["stable_minus_direct_rewired_hazard_fraction"],
                ),
                "family_spearman_gain_vs_rewired_difference": _spearman_correlation(
                    family["direct_gain_over_stable"],
                    family["stable_minus_direct_rewired_hazard_fraction"],
                ),
            }
        )
    return paired, pd.DataFrame(rows)


def _random_block_family_summary(
    paired: pd.DataFrame, *, method: str
) -> pd.DataFrame:
    """Summarize each independent Monte Carlo block with equal family weight."""

    subset = paired.loc[paired["method"].eq(method)].copy()
    regime_columns = ["disease_regime"] if "disease_regime" in subset.columns else []
    cluster = (
        subset.groupby(
            regime_columns + [
                "detection_profile",
                "rewiring_fraction",
                "random_block",
                "system_family",
                "analysis_cluster_id",
            ],
            observed=True,
        )["increment"]
        .mean()
        .reset_index()
    )
    family = (
        cluster.groupby(
            regime_columns + [
                "detection_profile",
                "rewiring_fraction",
                "random_block",
                "system_family",
            ],
            observed=True,
        )["increment"]
        .mean()
        .reset_index()
    )
    return (
        family.groupby(
            regime_columns + ["detection_profile", "rewiring_fraction", "random_block"],
            observed=True,
        )
        .agg(
            family_equal_increment=("increment", "mean"),
            positive_families=("increment", lambda values: int((values > 0).sum())),
            families=("system_family", "nunique"),
        )
        .reset_index()
    )


def _hierarchical_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    family_rows = []
    for group_index, (key, group) in enumerate(
        frame.groupby(group_columns, observed=True, sort=True)
    ):
        if not isinstance(key, tuple):
            key = (key,)
        identifiers = dict(zip(group_columns, key))
        units = (
            group.groupby(
                ["system_family", "analysis_cluster_id"], observed=True
            )[value_column]
            .mean()
            .reset_index()
        )
        family = (
            units.groupby("system_family", observed=True)[value_column]
            .mean()
            .reset_index()
        )
        for item in family.itertuples(index=False):
            family_rows.append(
                {
                    **identifiers,
                    "system_family": item.system_family,
                    "mean_value": float(getattr(item, value_column)),
                }
            )
        arrays = {
            str(name): subset[value_column].to_numpy(float)
            for name, subset in units.groupby("system_family", observed=True, sort=True)
        }
        family_names = sorted(arrays)
        rng = np.random.default_rng(seed + group_index)
        samples = np.empty(bootstrap_replicates, dtype=float)
        for repetition in range(bootstrap_replicates):
            means = []
            for name in rng.choice(family_names, size=len(family_names), replace=True):
                values = arrays[str(name)]
                selected = rng.integers(0, len(values), size=len(values))
                means.append(float(values[selected].mean()))
            samples[repetition] = float(np.mean(means))
        estimate = float(family[value_column].mean())
        rows.append(
            {
                **identifiers,
                "families": len(family_names),
                "contexts": group[
                    [column for column in POLICY_KEYS if column in group.columns]
                ].drop_duplicates().shape[0],
                "family_equal_mean": estimate,
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_positive": float((samples > 0).mean()),
                "positive_families": int(family[value_column].gt(0).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(family_rows)


def _plot_heatmaps(
    summary: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    subtitle: str,
    colorbar_label: str,
    path: Path,
) -> None:
    profiles = ["early_detection", "delayed_detection"]
    delays = sorted(summary["action_delay_fraction"].unique())
    residuals = sorted(summary["residual_contact_multiplier"].unique())
    values = 100 * summary[value_column].to_numpy(float)
    limit = max(abs(values.min()), abs(values.max()), 0.001)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    image = None
    for axis, profile in zip(axes, profiles):
        subset = summary.loc[summary["detection_profile"].eq(profile)]
        matrix = np.full((len(delays), len(residuals)), np.nan)
        for row, delay in enumerate(delays):
            for column, residual in enumerate(residuals):
                match = subset.loc[
                    subset["action_delay_fraction"].eq(delay)
                    & subset["residual_contact_multiplier"].eq(residual),
                    value_column,
                ]
                if len(match):
                    matrix[row, column] = 100 * float(match.iloc[0])
        image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        axis.set_xticks(
            range(len(residuals)),
            [f"{100 * (1 - value):.0f}%" for value in residuals],
        )
        axis.set_yticks(
            range(len(delays)),
            [f"{value:.2g}" for value in delays],
        )
        axis.set_xlabel("Contact reduction after action starts")
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        for row in range(len(delays)):
            for column in range(len(residuals)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    axes[0].set_ylabel("Action delay (mean infectious periods)")
    assert image is not None
    color_axis = fig.add_axes([0.92, 0.21, 0.018, 0.56])
    fig.colorbar(image, cax=color_axis, label=colorbar_label)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, subtitle, ha="center", color="#555555")
    fig.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.16, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_family_operational(
    family: pd.DataFrame, path: Path
) -> None:
    subset = family.loc[
        family["action_delay_fraction"].eq(0.1)
        & family["residual_contact_multiplier"].eq(0.25)
    ].copy()
    families = sorted(subset["system_family"].unique())
    y = np.arange(len(families))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharey=True)
    for axis, profile, color in zip(
        axes, ["early_detection", "delayed_detection"], ["#4C78A8", "#F58518"]
    ):
        values = (
            subset.loc[subset["detection_profile"].eq(profile)]
            .set_index("system_family")
            .reindex(families)["mean_value"]
        )
        axis.barh(y, 100 * values, color=color)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        axis.set_xlabel("Gain over stable watchlist\n(attack-rate percentage points)")
        axis.grid(axis="x", alpha=0.18)
    axes[0].set_yticks(
        y,
        [
            SYSTEM_FAMILY_LABELS.get(
                name, DATASET_LABELS.get(name, name.replace("_", " "))
            )
            for name in families
        ],
    )
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(
        "Stable + tracing increment under an operational isolation scenario",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Action delay = 0.1 infectious periods; contact reduction = 75%; systems are not pooled",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.81, bottom=0.17, wspace=0.16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_observation_heatmaps(
    summary: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    subtitle: str,
    colorbar_label: str,
    path: Path,
) -> None:
    profiles = ["early_detection", "delayed_detection"]
    sensitivities = sorted(summary["secondary_case_sensitivity"].unique())
    false_positive_rates = sorted(summary["false_positive_rate"].unique())
    values = 100 * summary[value_column].to_numpy(float)
    limit = max(abs(values.min()), abs(values.max()), 0.001)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.1), sharey=True)
    image = None
    for axis, profile in zip(axes, profiles):
        subset = summary.loc[summary["detection_profile"].eq(profile)]
        matrix = np.full((len(sensitivities), len(false_positive_rates)), np.nan)
        for row, sensitivity in enumerate(sensitivities):
            for column, false_positive_rate in enumerate(false_positive_rates):
                match = subset.loc[
                    subset["secondary_case_sensitivity"].eq(sensitivity)
                    & subset["false_positive_rate"].eq(false_positive_rate),
                    value_column,
                ]
                if len(match):
                    matrix[row, column] = 100 * float(match.iloc[0])
        image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        axis.set_xticks(
            range(len(false_positive_rates)),
            [f"{100 * value:.0f}%" for value in false_positive_rates],
        )
        axis.set_yticks(
            range(len(sensitivities)),
            [f"{100 * value:.0f}%" for value in sensitivities],
        )
        axis.set_xlabel("False-positive rate among non-infectious animals")
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        for row in range(len(sensitivities)):
            for column in range(len(false_positive_rates)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    axes[0].set_ylabel("Secondary infectious-case sensitivity")
    assert image is not None
    color_axis = fig.add_axes([0.92, 0.21, 0.018, 0.56])
    fig.colorbar(image, cax=color_axis, label=colorbar_label)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, subtitle, ha="center", color="#555555")
    fig.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.18, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_observation_quality(worlds: pd.DataFrame, path: Path) -> None:
    unique = worlds.drop_duplicates(POLICY_KEYS).copy()
    profiles = ["early_detection", "delayed_detection"]
    sensitivities = sorted(unique["secondary_case_sensitivity"].unique())
    false_positive_rates = sorted(unique["false_positive_rate"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharex=True, sharey=True)
    image = None
    for column, profile in enumerate(profiles):
        subset = unique.loc[unique["detection_profile"].eq(profile)]
        for row, (metric, label) in enumerate(
            [
                (
                    "detected_actionable_infectious_fraction",
                    "Currently infectious fraction",
                ),
                ("detected_secondary_recall", "Secondary-case recall"),
            ]
        ):
            axis = axes[row, column]
            matrix = np.full((len(sensitivities), len(false_positive_rates)), np.nan)
            for sensitivity_index, sensitivity in enumerate(sensitivities):
                for rate_index, false_positive_rate in enumerate(false_positive_rates):
                    cells = subset.loc[
                        subset["secondary_case_sensitivity"].eq(sensitivity)
                        & subset["false_positive_rate"].eq(false_positive_rate)
                    ]
                    if metric == "detected_secondary_recall":
                        cells = cells.loc[
                            cells["secondary_infectious_cases_at_detection"].gt(0)
                        ]
                    if len(cells):
                        matrix[sensitivity_index, rate_index] = float(cells[metric].mean())
            image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            for sensitivity_index in range(len(sensitivities)):
                for rate_index in range(len(false_positive_rates)):
                    value = matrix[sensitivity_index, rate_index]
                    axis.text(
                        rate_index,
                        sensitivity_index,
                        "NA" if np.isnan(value) else f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if not np.isnan(value) and value < 0.55 else "black",
                        fontsize=9,
                    )
            axis.set_title(
                f"{profile.replace('_', ' ').title()}\n{label}",
                fontweight="bold",
            )
            axis.set_xticks(
                range(len(false_positive_rates)),
                [f"{100 * value:.0f}%" for value in false_positive_rates],
            )
            axis.set_yticks(
                range(len(sensitivities)),
                [f"{100 * value:.0f}%" for value in sensitivities],
            )
    for axis in axes[-1, :]:
        axis.set_xlabel("False-positive rate")
    for axis in axes[:, 0]:
        axis.set_ylabel("Configured secondary-case sensitivity")
    assert image is not None
    color_axis = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    fig.colorbar(image, cax=color_axis, label="Observed fraction")
    fig.suptitle("Realized case-observation quality", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Recall excludes worlds with no secondary infectious case; the confirmed trigger remains listed even if recovered",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.11, right=0.88, top=0.84, bottom=0.10, hspace=0.30, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_observation_family(
    family: pd.DataFrame,
    *,
    sensitivity: float,
    false_positive_rate: float,
    path: Path,
) -> None:
    subset = family.loc[
        family["secondary_case_sensitivity"].eq(sensitivity)
        & family["false_positive_rate"].eq(false_positive_rate)
    ].copy()
    families = sorted(subset["system_family"].unique())
    y = np.arange(len(families))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharey=True)
    for axis, profile, color in zip(
        axes, ["early_detection", "delayed_detection"], ["#4C78A8", "#F58518"]
    ):
        values = (
            subset.loc[subset["detection_profile"].eq(profile)]
            .set_index("system_family")
            .reindex(families)["mean_value"]
        )
        axis.barh(y, 100 * values, color=color)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        axis.set_xlabel("Gain over stable watchlist\n(attack-rate percentage points)")
        axis.grid(axis="x", alpha=0.18)
    axes[0].set_yticks(
        y,
        [SYSTEM_FAMILY_LABELS.get(name, name.replace("_", " ")) for name in families],
    )
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(
        "Stable + tracing increment under observation error",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        f"Secondary sensitivity = {100 * sensitivity:.0f}%; false-positive rate = {100 * false_positive_rate:.0f}%; systems are not pooled",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.81, bottom=0.17, wspace=0.16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_detection_timing_contrast(summary: pd.DataFrame, path: Path) -> None:
    sensitivities = sorted(summary["secondary_case_sensitivity"].unique())
    false_positive_rates = sorted(summary["false_positive_rate"].unique())
    matrix = np.full((len(sensitivities), len(false_positive_rates)), np.nan)
    for row, sensitivity in enumerate(sensitivities):
        for column, false_positive_rate in enumerate(false_positive_rates):
            match = summary.loc[
                summary["secondary_case_sensitivity"].eq(sensitivity)
                & summary["false_positive_rate"].eq(false_positive_rate),
                "family_equal_mean",
            ]
            if len(match):
                matrix[row, column] = 100 * float(match.iloc[0])
    limit = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.001)
    fig, axis = plt.subplots(figsize=(7.4, 6.2))
    image = axis.imshow(
        matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    axis.set_xticks(
        range(len(false_positive_rates)),
        [f"{100 * value:.0f}%" for value in false_positive_rates],
    )
    axis.set_yticks(
        range(len(sensitivities)),
        [f"{100 * value:.0f}%" for value in sensitivities],
    )
    axis.set_xlabel("False-positive rate among non-infectious animals")
    axis.set_ylabel("Secondary infectious-case sensitivity")
    for row in range(len(sensitivities)):
        for column in range(len(false_positive_rates)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=10,
            )
    fig.colorbar(
        image,
        ax=axis,
        label="Delayed minus early tracing increment (percentage points)",
        fraction=0.046,
        pad=0.05,
    )
    fig.suptitle(
        "Paired detection-timing contrast",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Positive values mean tracing adds more value at delayed than early detection",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.16, right=0.90, top=0.82, bottom=0.14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rewiring_sensitivity(
    absolute: pd.DataFrame,
    relative: pd.DataFrame,
    path: Path,
) -> None:
    fractions = sorted(relative["rewiring_fraction"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for axis, frame, value_label in [
        (axes[0], absolute, "Benefit over standard care"),
        (axes[1], relative, "Increment over stable watchlist"),
    ]:
        for profile, color, marker in [
            ("early_detection", "#4C78A8", "o"),
            ("delayed_detection", "#F58518", "s"),
        ]:
            subset = (
                frame.loc[frame["detection_profile"].eq(profile)]
                .set_index("rewiring_fraction")
                .reindex(fractions)
            )
            means = 100 * subset["family_equal_mean"].to_numpy(float)
            lower = means - 100 * subset["ci_low"].to_numpy(float)
            upper = 100 * subset["ci_high"].to_numpy(float) - means
            axis.errorbar(
                100 * np.asarray(fractions),
                means,
                yerr=np.vstack([lower, upper]),
                color=color,
                marker=marker,
                linewidth=2,
                capsize=4,
                label=profile.replace("_", " ").title(),
            )
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xlabel("Compensated lost contact (%)")
        axis.set_ylabel(f"{value_label}\n(attack-rate percentage points)")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False)
    fig.suptitle(
        "Targeted-intervention value under compensatory rewiring",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Points are five-family-equal estimates; bars are hierarchical bootstrap 95% intervals",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.18, wspace=0.28)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rewiring_family(
    relative_family: pd.DataFrame,
    path: Path,
    *,
    policy_label: str,
) -> None:
    fractions = sorted(relative_family["rewiring_fraction"].unique())
    families = sorted(relative_family["system_family"].unique())
    values = 100 * relative_family["mean_value"].to_numpy(float)
    limit = max(abs(values.min()), abs(values.max()), 0.001)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    image = None
    for axis, profile in zip(axes, ["early_detection", "delayed_detection"]):
        subset = relative_family.loc[
            relative_family["detection_profile"].eq(profile)
        ]
        matrix = np.full((len(families), len(fractions)), np.nan)
        for row, family in enumerate(families):
            for column, fraction in enumerate(fractions):
                match = subset.loc[
                    subset["system_family"].eq(family)
                    & subset["rewiring_fraction"].eq(fraction),
                    "mean_value",
                ]
                if len(match):
                    matrix[row, column] = 100 * float(match.iloc[0])
        image = axis.imshow(
            matrix,
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            aspect="auto",
        )
        axis.set_xticks(
            range(len(fractions)), [f"{100 * value:.0f}%" for value in fractions]
        )
        axis.set_yticks(
            range(len(families)),
            [SYSTEM_FAMILY_LABELS.get(name, name.replace("_", " ")) for name in families],
        )
        axis.set_xlabel("Compensated lost contact")
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        for row in range(len(families)):
            for column in range(len(fractions)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    axes[1].tick_params(axis="y", labelleft=False)
    assert image is not None
    color_axis = fig.add_axes([0.92, 0.21, 0.018, 0.56])
    fig.colorbar(image, cax=color_axis, label="Increment over stable (percentage points)")
    fig.suptitle(
        f"{policy_label} heterogeneity under rewiring",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Uniform partner substitution is a structural stress test, not a fitted behavior model",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.25, right=0.88, top=0.82, bottom=0.16, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rewiring_policy_comparison(absolute: pd.DataFrame, path: Path) -> None:
    fractions = sorted(absolute["rewiring_fraction"].unique())
    styles = {
        "random": ("#9E9E9E", "o"),
        "stable_watchlist": ("#4C78A8", "s"),
        "stable_plus_tracing": ("#F58518", "D"),
        "contact_to_detected": ("#54A24B", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9), sharey=True)
    for axis, profile in zip(axes, ["early_detection", "delayed_detection"]):
        profile_frame = absolute.loc[absolute["detection_profile"].eq(profile)]
        for method, (color, marker) in styles.items():
            if method not in set(profile_frame["method"]):
                continue
            subset = (
                profile_frame.loc[profile_frame["method"].eq(method)]
                .set_index("rewiring_fraction")
                .reindex(fractions)
            )
            means = 100 * subset["family_equal_mean"].to_numpy(float)
            lower = means - 100 * subset["ci_low"].to_numpy(float)
            upper = 100 * subset["ci_high"].to_numpy(float) - means
            axis.errorbar(
                100 * np.asarray(fractions),
                means,
                yerr=np.vstack([lower, upper]),
                color=color,
                marker=marker,
                linewidth=2,
                capsize=3,
                label=method.replace("_", " ").title(),
            )
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        axis.set_xlabel("Compensated lost contact (%)")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Benefit over standard care\n(attack-rate percentage points)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Transparent policies under compensatory rewiring",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Positive values mean fewer infections than isolating detected cases alone",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.81, bottom=0.23, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rewiring_mechanism(paired: pd.DataFrame, path: Path) -> None:
    cluster = (
        paired.groupby(
            [
                "detection_profile",
                "rewiring_fraction",
                "system_family",
                "analysis_cluster_id",
            ],
            observed=True,
        )[
            [
                "direct_gain_over_stable",
                "stable_minus_direct_rewired_hazard_fraction",
            ]
        ]
        .mean()
        .reset_index()
    )
    fractions = sorted(cluster["rewiring_fraction"].unique())
    profiles = ["early_detection", "delayed_detection"]
    colors = {
        family: plt.get_cmap("tab10")(index)
        for index, family in enumerate(sorted(cluster["system_family"].unique()))
    }
    fig, axes = plt.subplots(
        len(profiles), len(fractions), figsize=(5.3 * len(fractions), 8.2), squeeze=False
    )
    for row, profile in enumerate(profiles):
        for column, fraction in enumerate(fractions):
            axis = axes[row, column]
            subset = cluster.loc[
                cluster["detection_profile"].eq(profile)
                & cluster["rewiring_fraction"].eq(fraction)
            ]
            for family, group in subset.groupby("system_family", observed=True):
                axis.scatter(
                    100 * group["stable_minus_direct_rewired_hazard_fraction"],
                    100 * group["direct_gain_over_stable"],
                    s=38,
                    alpha=0.72,
                    color=colors[str(family)],
                    label=SYSTEM_FAMILY_LABELS.get(str(family), str(family)),
                )
            axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
            axis.axvline(0, color="#666666", linewidth=1, linestyle="--")
            correlation = _spearman_correlation(
                subset["direct_gain_over_stable"],
                subset["stable_minus_direct_rewired_hazard_fraction"],
            )
            correlation_label = (
                "NA (constant axis)" if math.isnan(correlation) else f"{correlation:+.2f}"
            )
            axis.set_title(
                f"{profile.replace('_', ' ').title()} | {100 * fraction:.0f}% compensation\n"
                f"cluster Spearman = {correlation_label}",
                fontweight="bold",
            )
            axis.grid(alpha=0.16)
    for axis in axes[-1, :]:
        axis.set_xlabel("Stable minus direct redistributed hazard\n(% of original opportunity mass)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Direct gain over stable\n(attack-rate percentage points)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Is the apparent policy switch explained by redistributed hazard?",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        "Each point is one independent analysis cluster; association is diagnostic, not causal mediation",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.17, hspace=0.34, wspace=0.24)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_disease_regime_phase(
    summary: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    subtitle: str,
    colorbar_label: str,
    path: Path,
) -> None:
    regimes = [item for item in ("low", "middle", "high") if item in set(summary["disease_regime"])]
    rewiring = sorted(summary["rewiring_fraction"].unique())
    profiles = [item for item in ("early_detection", "delayed_detection") if item in set(summary["detection_profile"])]
    values = 100 * summary[value_column].to_numpy(float)
    limit = max(float(np.max(np.abs(values))), 0.001)
    fig, axes = plt.subplots(1, len(profiles), figsize=(12.8, 5.7), squeeze=False)
    image = None
    for axis, profile in zip(axes[0], profiles):
        selected = summary.loc[summary["detection_profile"].eq(profile)]
        matrix = (
            selected.pivot(index="disease_regime", columns="rewiring_fraction", values=value_column)
            .reindex(index=regimes, columns=rewiring)
            .to_numpy(float)
            * 100
        )
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        for row in range(len(regimes)):
            for column in range(len(rewiring)):
                axis.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=12)
        axis.set_xticks(range(len(rewiring)), [f"{100 * value:.0f}%" for value in rewiring])
        axis.set_yticks(range(len(regimes)), [name.title() for name in regimes])
        axis.set_xlabel("Compensatory rewiring")
        axis.set_title(profile.replace("_", " ").title())
    axes[0, 0].set_ylabel("Within-dataset calibrated disease regime")
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, subtitle, ha="center", color="#555555", fontsize=11)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04)
        colorbar.set_label(colorbar_label)
    fig.subplots_adjust(left=0.11, right=0.89, top=0.82, bottom=0.14, wspace=0.28)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_disease_regime_family_support(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[summary["method"].eq("contact_to_detected")].copy()
    support = (
        selected.groupby(
            ["disease_regime", "detection_profile", "rewiring_fraction"], observed=True
        )
        .agg(
            positive_families=("mean_value", lambda values: int((values > 0).sum())),
            families=("system_family", "nunique"),
        )
        .reset_index()
    )
    support["support_fraction"] = support["positive_families"] / support["families"]
    regimes = [item for item in ("low", "middle", "high") if item in set(support["disease_regime"])]
    rewiring = sorted(support["rewiring_fraction"].unique())
    profiles = [item for item in ("early_detection", "delayed_detection") if item in set(support["detection_profile"])]
    fig, axes = plt.subplots(1, len(profiles), figsize=(12.8, 5.7), squeeze=False)
    image = None
    for axis, profile in zip(axes[0], profiles):
        selected = support.loc[support["detection_profile"].eq(profile)]
        fractions = selected.pivot(index="disease_regime", columns="rewiring_fraction", values="support_fraction").reindex(index=regimes, columns=rewiring)
        positive = selected.pivot(index="disease_regime", columns="rewiring_fraction", values="positive_families").reindex(index=regimes, columns=rewiring)
        families = selected.pivot(index="disease_regime", columns="rewiring_fraction", values="families").reindex(index=regimes, columns=rewiring)
        image = axis.imshow(fractions.to_numpy(float), cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
        for row in range(len(regimes)):
            for column in range(len(rewiring)):
                axis.text(column, row, f"{int(positive.iloc[row, column])}/{int(families.iloc[row, column])}", ha="center", va="center", fontsize=13)
        axis.set_xticks(range(len(rewiring)), [f"{100 * value:.0f}%" for value in rewiring])
        axis.set_yticks(range(len(regimes)), [name.title() for name in regimes])
        axis.set_xlabel("Compensatory rewiring")
        axis.set_title(profile.replace("_", " ").title())
    axes[0, 0].set_ylabel("Within-dataset calibrated disease regime")
    fig.suptitle("Cross-family support for direct-contact targeting", fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, "Cells show independent animal-system families with a positive direct-over-stable increment", ha="center", color="#555555", fontsize=11)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04)
        colorbar.set_label("Fraction of families with positive increment")
    fig.subplots_adjust(left=0.11, right=0.89, top=0.82, bottom=0.14, wspace=0.28)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _natural_attack_rate_support(worlds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = NATURAL_KEYS + (["disease_regime"] if "disease_regime" in worlds else [])
    natural = worlds.drop_duplicates(keys).copy()
    natural["natural_attack_rate"] = (
        natural["natural_final_size"] / natural["population_size"]
    )
    family = (
        natural.groupby(
            ["epidemic_model", "disease_regime", "system_family"], observed=True
        )["natural_attack_rate"]
        .mean()
        .reset_index(name="family_mean_attack_rate")
    )
    summary = (
        family.groupby(["epidemic_model", "disease_regime"], observed=True)[
            "family_mean_attack_rate"
        ]
        .mean()
        .reset_index(name="family_equal_mean_attack_rate")
    )
    return family, summary


def _plot_cross_model_attack_rates(summary: pd.DataFrame, path: Path) -> None:
    regimes = [
        item for item in ("low", "middle", "high")
        if item in set(summary["disease_regime"])
    ]
    positions = np.arange(len(regimes))
    labels = {
        "temporal_sir": "Markov SIR",
        "temporal_seir_erlang": "Staged SEIR/Erlang",
    }
    colors = {"temporal_sir": "#4C78A8", "temporal_seir_erlang": "#E45756"}
    fig, axis = plt.subplots(figsize=(9.2, 5.7))
    for model, group in summary.groupby("epidemic_model", observed=True, sort=True):
        ordered = group.set_index("disease_regime").reindex(regimes)
        axis.plot(
            positions,
            100 * ordered["family_equal_mean_attack_rate"],
            marker="o",
            linewidth=2.4,
            color=colors.get(model, "#777777"),
            label=labels.get(model, model),
        )
    axis.set_xticks(positions, [item.title() for item in regimes])
    axis.set_ylabel("Natural final attack rate (%)")
    axis.set_xlabel("Frozen within-dataset disease regime")
    axis.set_title(
        "Alternative epidemic model retains a discriminating severity gradient",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.18)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_model_policy_boundary(summary: pd.DataFrame, path: Path) -> None:
    regimes = [
        item for item in ("low", "middle", "high")
        if item in set(summary["disease_regime"])
    ]
    positions = np.arange(len(regimes))
    labels = {
        "temporal_sir": "Markov SIR",
        "temporal_seir_erlang": "Staged SEIR/Erlang",
    }
    colors = {"temporal_sir": "#4C78A8", "temporal_seir_erlang": "#E45756"}
    fig, axis = plt.subplots(figsize=(9.2, 5.7))
    for offset, (model, group) in zip(
        (-0.06, 0.06), summary.groupby("epidemic_model", observed=True, sort=True)
    ):
        ordered = group.set_index("disease_regime").reindex(regimes)
        axis.errorbar(
            positions + offset,
            100 * ordered["family_equal_mean"],
            yerr=[
                100 * (ordered["family_equal_mean"] - ordered["ci_low"]),
                100 * (ordered["ci_high"] - ordered["family_equal_mean"]),
            ],
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=colors.get(model, "#777777"),
            label=labels.get(model, model),
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(positions, [item.title() for item in regimes])
    axis.set_ylabel("Direct over stable (attack-rate percentage points)")
    axis.set_xlabel("Frozen within-dataset disease regime")
    axis.set_title(
        "Primary policy boundary across epidemic state models",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0.5,
        1.01,
        "Early detection with full compensatory rewiring; family-first hierarchical intervals",
        transform=axis.transAxes,
        ha="center",
        color="#555555",
    )
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.18)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile = dict(config["profiles"][profile_name])
    stable_path = Path(config["data"]["stable_prediction_path"])
    prerequisite_path = Path(config["data"]["prerequisite_audit"])
    zero_rewiring_reference_path = (
        Path(config["data"]["zero_rewiring_reference"])
        if config["data"].get("zero_rewiring_reference")
        else None
    )
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
    simulation_sources = [
        Path(__file__),
        Path(__file__).parents[1] / "simulation" / "paired.py",
        Path(__file__).parents[1] / "simulation" / "interventions.py",
        Path(__file__).parents[1] / "simulation" / "seir.py",
    ]
    fingerprint = hashlib.sha256(
        config_path.read_bytes()
        + stable_path.read_bytes()
        + b"".join(path.read_bytes() for path in simulation_sources)
    ).hexdigest()[:12]
    detections = [DetectionProfile(**item) for item in config["decision"]["detection_profiles"]]
    action_delays = list(map(float, config["decision"]["action_delay_fractions_of_mean_infectious_period"]))
    residuals = list(map(float, config["decision"]["residual_contact_multipliers"]))
    sensitivities = list(
        map(
            float,
            config["decision"].get(
                "secondary_case_sensitivities",
                [config["decision"]["secondary_case_sensitivity"]],
            ),
        )
    )
    false_positive_rates = list(
        map(float, config["decision"].get("false_positive_rates", [0.0]))
    )
    rewiring_fractions = list(
        map(float, config["decision"].get("rewiring_fractions", [0.0]))
    )
    rewiring_mode = str(config["decision"].get("rewiring_mode", "none"))
    parameter_pool = str(config["evaluation"].get("parameter_pool", "selected"))
    parameter_selection_mode = str(
        config["evaluation"].get("parameter_selection_mode", "median")
    )
    epidemic_model = dict(
        config["evaluation"].get("epidemic_model", {"name": "temporal_sir"})
    )
    stratify_by_disease_regime = parameter_selection_mode != "median"
    tasks = []
    support_rows = []
    considered_windows: dict[str, int] = {}
    retained_windows: dict[str, int] = {}
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
            parameter_pool,
        )
        if not windows:
            raise ValueError(f"no matched intervention-delivery windows for {dataset_id}")
        considered_windows[dataset_id] = len(windows)
        retained_windows[dataset_id] = 0
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
                        + mean_period * max(action_delays)
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
                        "mean_attack_rate": float(parameter.mean_attack_rate),
                        "supports_full_delivery_matrix": supported,
                    }
                )
                if supported:
                    compatible.append(parameter)
            if not compatible:
                continue
            parameter_regimes = _select_parameter_regimes(
                compatible, parameter_selection_mode
            )
            if not parameter_regimes:
                continue
            retained_windows[dataset_id] += 1
            cluster = (
                f"{dataset_id}::{window['network_id']}"
                if specification.get("analysis_cluster") == "network"
                else f"{dataset_id}::{window['network_id']}::{anchor.anchor_id}"
            )
            for disease_regime, parameter in parameter_regimes:
                for detection in detections:
                    for delay in action_delays:
                        for residual in residuals:
                            for sensitivity in sensitivities:
                                for false_positive_rate in false_positive_rates:
                                    for rewiring_fraction in rewiring_fractions:
                                        tasks.append(
                                            (
                                                dataset_id,
                                                specification,
                                                window,
                                                disease_regime,
                                                parameter,
                                                detection,
                                                delay,
                                                residual,
                                                sensitivity,
                                                false_positive_rate,
                                                rewiring_fraction,
                                                cluster,
                                            )
                                        )
    output_frames = []
    progress = tqdm(tasks, desc="Delivery-sensitivity tasks", unit="task")
    for (
        dataset_id,
        specification,
        window,
        disease_regime,
        parameter,
        detection,
        delay,
        residual,
        sensitivity,
        false_positive_rate,
        rewiring_fraction,
        cluster,
    ) in progress:
        anchor = window["anchor"]
        identity = f"{fingerprint}|{dataset_id}|{window['network_id']}|{anchor.anchor_id}|{disease_regime}|{parameter.parameter_id}|{detection.name}|{delay}|{residual}|{sensitivity}|{false_positive_rate}|{rewiring_fraction}|{rewiring_mode}"
        checkpoint = checkpoint_dir / f"{dataset_id}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.csv.gz"
        expected_methods = set(config["decision"]["methods"])
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(
                checkpoint,
                dtype={"initial_infected": str},
                keep_default_na=False,
            )
            if not frame.empty and set(frame["method"]) == expected_methods:
                frame["disease_regime"] = disease_regime
                output_frames.append(frame)
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
            "delivery_sensitivity_seeds",
        )[: int(profile["seeds_per_anchor"])]
        frame = _run_task(
            dataset_id=dataset_id,
            network_id=str(window["network_id"]),
            system_family=str(specification["system_family"]),
            analysis_cluster_id=cluster,
            window=window,
            parameter=parameter,
            detection_profile=detection,
            action_delay_fraction=delay,
            residual_contact_multiplier=residual,
            stable_scores=stable,
            methods=list(config["decision"]["methods"]),
            seed_nodes=seeds,
            random_blocks=int(profile["random_blocks"]),
            minimum_budget=int(config["decision"]["minimum_additional_budget"]),
            budget_fraction=float(config["decision"]["additional_budget_fraction"]),
            secondary_case_sensitivity=sensitivity,
            false_positive_rate=false_positive_rate,
            rewiring_fraction=rewiring_fraction,
            rewiring_mode=(rewiring_mode if rewiring_fraction > 0 else "none"),
            tracing_half_life_fraction=float(config["decision"]["tracing_half_life_fraction_of_mean_infectious_period"]),
            experiment_seed=int(config["evaluation"]["seed"]),
            epidemic_model=epidemic_model,
        )
        if frame.empty:
            raise ValueError(f"unsupported task unexpectedly reached execution: {identity}")
        frame["disease_regime"] = disease_regime
        frame.to_csv(checkpoint, index=False, compression="gzip")
        output_frames.append(frame)
        progress.set_postfix_str(f"{dataset_id} completed")
    worlds = pd.concat(output_frames, ignore_index=True)
    paired = _paired_increments(worlds, str(config["evaluation"]["primary_baseline"]))
    mechanism_paired = pd.DataFrame()
    mechanism_summary = pd.DataFrame()
    if bool(config["evaluation"].get("mechanism_audit", True)) and {"stable_watchlist", "contact_to_detected"}.issubset(
        set(worlds["method"].unique())
    ) and "additional_rewired_hazard_fraction" in worlds:
        mechanism_paired, mechanism_summary = _rewiring_mechanism_contrasts(worlds)
    repetitions = int(profile.get("bootstrap_replicates", config["evaluation"]["bootstrap_replicates"]))
    regime_groups = ["disease_regime"] if stratify_by_disease_regime else []
    action_groups = regime_groups + [
        "detection_profile",
        "action_delay_fraction",
        "residual_contact_multiplier",
        "secondary_case_sensitivity",
        "false_positive_rate",
        "rewiring_fraction",
        "rewiring_mode",
        "method",
    ]
    absolute, absolute_family = _hierarchical_summary(
        worlds,
        value_column="attack_rate_reduction",
        group_columns=action_groups,
        bootstrap_replicates=repetitions,
        seed=int(config["evaluation"]["seed"]),
    )
    relative, relative_family = _hierarchical_summary(
        paired,
        value_column="increment",
        group_columns=action_groups,
        bootstrap_replicates=repetitions,
        seed=int(config["evaluation"]["seed"]) + 1,
    )
    natural_attack_family = pd.DataFrame()
    natural_attack_summary = pd.DataFrame()
    cross_model_policy = pd.DataFrame()
    if stratify_by_disease_regime:
        natural_attack_family, natural_attack_summary = _natural_attack_rate_support(worlds)
        reference_root_value = config["data"].get("reference_model_results")
        if reference_root_value:
            reference_root = Path(reference_root_value)
            reference_worlds = pd.read_csv(
                reference_root / "response_worlds.csv.gz",
                dtype={"initial_infected": str, "network_id": str},
            )
            reference_worlds["epidemic_model"] = "temporal_sir"
            reference_family, reference_attack = _natural_attack_rate_support(
                reference_worlds
            )
            natural_attack_family = pd.concat(
                [reference_family, natural_attack_family], ignore_index=True
            )
            natural_attack_summary = pd.concat(
                [reference_attack, natural_attack_summary], ignore_index=True
            )
            reference_relative = pd.read_csv(
                reference_root / "relative_policy_summary.csv"
            )
            reference_relative["epidemic_model"] = "temporal_sir"
            alternative_relative = relative.copy()
            alternative_relative["epidemic_model"] = str(
                epidemic_model.get("name", "temporal_sir")
            )
            cross_model_policy = pd.concat(
                [reference_relative, alternative_relative], ignore_index=True
            )
            cross_model_policy = cross_model_policy.loc[
                cross_model_policy["method"].eq(config["evaluation"]["primary_method"])
                & cross_model_policy["detection_profile"].eq("early_detection")
                & cross_model_policy["rewiring_fraction"].eq(max(rewiring_fractions))
            ].copy()
    timing_contrasts = _detection_timing_contrasts(paired)
    timing_groups = regime_groups + [
        "action_delay_fraction",
        "residual_contact_multiplier",
        "secondary_case_sensitivity",
        "false_positive_rate",
        "rewiring_fraction",
        "rewiring_mode",
        "method",
    ]
    timing_summary, timing_family = _hierarchical_summary(
        timing_contrasts,
        value_column="timing_contrast",
        group_columns=timing_groups,
        bootstrap_replicates=repetitions,
        seed=int(config["evaluation"]["seed"]) + 2,
    )
    primary = relative.loc[
        relative["method"].eq(config["evaluation"]["primary_method"])
    ].copy()
    block_summary = _random_block_family_summary(
        paired, method=str(config["evaluation"]["primary_method"])
    )
    absolute_primary = absolute.loc[
        absolute["method"].eq(config["evaluation"]["primary_method"])
    ].copy()
    timing_primary = timing_summary.loc[
        timing_summary["method"].eq(config["evaluation"]["primary_method"])
    ].copy()
    direct_contact = relative.loc[relative["method"].eq("contact_to_detected")].copy()
    high_rewiring_early_direct = direct_contact.loc[
        direct_contact["detection_profile"].eq("early_detection")
        & direct_contact["rewiring_fraction"].gt(0)
    ]
    scientific = {
        "primary_increment_classification": (
            "directionally_robust"
            if primary["family_equal_mean"].ge(0).all()
            and not primary["ci_high"].lt(0).any()
            else f"{config['evaluation'].get('sensitivity_axis', 'delivery')}_sensitive"
        ),
        "absolute_benefit_classification": (
            f"beneficial_in_all_{config['evaluation'].get('sensitivity_axis', 'delivery')}_cells"
            if absolute_primary["ci_low"].gt(0).all()
            else f"not_resolved_in_all_{config['evaluation'].get('sensitivity_axis', 'delivery')}_cells"
        ),
        "cells": len(primary),
        "positive_point_estimate_cells": int(primary["family_equal_mean"].gt(0).sum()),
        "positive_lower_interval_cells": int(primary["ci_low"].gt(0).sum()),
        "negative_upper_interval_cells": int(primary["ci_high"].lt(0).sum()),
        "minimum_family_equal_increment": float(primary["family_equal_mean"].min()),
        "maximum_family_equal_increment": float(primary["family_equal_mean"].max()),
        "minimum_family_equal_absolute_benefit": float(
            absolute_primary["family_equal_mean"].min()
        ),
        "maximum_family_equal_absolute_benefit": float(
            absolute_primary["family_equal_mean"].max()
        ),
        "detection_timing_contrast_classification": (
            "delayed_increment_consistently_higher"
            if timing_primary["family_equal_mean"].gt(0).all()
            and not timing_primary["ci_high"].lt(0).any()
            else "detection_timing_contrast_mixed"
        ),
        "positive_detection_timing_contrast_cells": int(
            timing_primary["family_equal_mean"].gt(0).sum()
        ),
        "positive_detection_timing_contrast_lower_interval_cells": int(
            timing_primary["ci_low"].gt(0).sum()
        ),
        "minimum_detection_timing_contrast": float(
            timing_primary["family_equal_mean"].min()
        ),
        "maximum_detection_timing_contrast": float(
            timing_primary["family_equal_mean"].max()
        ),
        "direct_contact_positive_cells": int(
            direct_contact["family_equal_mean"].gt(0).sum()
        ),
        "direct_contact_positive_lower_interval_cells": int(
            direct_contact["ci_low"].gt(0).sum()
        ),
        "early_high_rewiring_direct_contact_lower_intervals_positive": bool(
            len(high_rewiring_early_direct) > 0
            and high_rewiring_early_direct["ci_low"].gt(0).all()
        ),
    }
    attack_rate_support_passed = True
    if len(natural_attack_summary):
        current_attack = natural_attack_summary.loc[
            natural_attack_summary["epidemic_model"].eq(
                str(epidemic_model.get("name", "temporal_sir"))
            )
        ].set_index("disease_regime")["family_equal_mean_attack_rate"]
        ordered_attack = current_attack.reindex(["low", "middle", "high"])
        attack_rate_support_passed = bool(
            ordered_attack.notna().all()
            and bool((np.diff(ordered_attack.to_numpy(float)) > 0).all())
            and ordered_attack["high"] - ordered_attack["low"] >= 0.10
        )
        scientific.update(
            {
                "alternative_model_family_equal_attack_rates": {
                    regime: float(value)
                    for regime, value in ordered_attack.items()
                },
                "alternative_model_high_minus_low_attack_rate": float(
                    ordered_attack["high"] - ordered_attack["low"]
                ),
                "attack_rate_support_gate_passed": attack_rate_support_passed,
            }
        )
    if stratify_by_disease_regime:
        maximum_rewiring = max(rewiring_fractions)
        regime_family = relative_family.loc[
            relative_family["method"].eq("contact_to_detected")
            & relative_family["detection_profile"].eq("early_detection")
            & relative_family["rewiring_fraction"].eq(maximum_rewiring)
        ]
        regime_gate = (
            regime_family.groupby("disease_regime", observed=True)
            .agg(
                family_equal_increment=("mean_value", "mean"),
                positive_families=("mean_value", lambda values: int((values > 0).sum())),
                families=("system_family", "nunique"),
            )
            .reset_index()
        )
        regime_gate["passes_family_direction_gate"] = (
            regime_gate["family_equal_increment"].gt(0)
            & regime_gate["positive_families"].ge(
                np.ceil(0.8 * regime_gate["families"]).astype(int)
            )
        )
        scientific.update(
            {
                "disease_regime_gate": regime_gate.to_dict(orient="records"),
                "disease_regimes_passing_family_direction_gate": int(
                    regime_gate["passes_family_direction_gate"].sum()
                ),
                "policy_boundary_retention_gate_passed": bool(
                    regime_gate["passes_family_direction_gate"].sum() >= 2
                ),
            }
        )
    if len(mechanism_summary):
        early_mechanism = mechanism_summary.loc[
            mechanism_summary["detection_profile"].eq("early_detection")
            & mechanism_summary["rewiring_fraction"].gt(0)
        ]
        scientific.update(
            {
                "early_high_rewiring_stable_rewires_more_in_all_families": bool(
                    len(early_mechanism) > 0
                    and early_mechanism["stable_rewires_more_families"]
                    .eq(early_mechanism["families"])
                    .all()
                ),
                "early_high_rewiring_cluster_spearman_range": [
                    float(early_mechanism[
                        "cluster_spearman_gain_vs_rewired_difference"
                    ].min()),
                    float(early_mechanism[
                        "cluster_spearman_gain_vs_rewired_difference"
                    ].max()),
                ],
            }
        )
    early_block_summary = block_summary.loc[
        block_summary["detection_profile"].eq("early_detection")
        & block_summary["rewiring_fraction"].gt(0)
    ]
    if len(early_block_summary):
        positive_blocks = (
            early_block_summary.groupby("rewiring_fraction", observed=True)[
                "family_equal_increment"
            ]
            .agg(
                positive_blocks=lambda values: int((values > 0).sum()),
                blocks="size",
            )
            .reset_index()
        )
        minimum_positive_fraction = float(
            (positive_blocks["positive_blocks"] / positive_blocks["blocks"]).min()
        )
        scientific["early_high_rewiring_positive_random_block_fraction_minimum"] = (
            minimum_positive_fraction
        )
        required_fraction = float(
            config["evaluation"].get("minimum_positive_random_block_fraction", 0.75)
        )
        scientific["random_block_replication_gate_passed"] = bool(
            minimum_positive_fraction >= required_fraction
        )
    standard_counts = worlds.groupby(POLICY_KEYS, observed=True)["standard_final_size"].nunique()
    method_counts = worlds.groupby(POLICY_KEYS, observed=True)["method"].nunique()
    block_counts = worlds.groupby(
        [
            "dataset_id",
            "network_id",
            "anchor_id",
            "parameter_id",
            "detection_profile",
            "action_delay_fraction",
            "residual_contact_multiplier",
            "secondary_case_sensitivity",
            "false_positive_rate",
            "rewiring_fraction",
            "rewiring_mode",
        ],
        observed=True,
    )["random_block"].nunique()
    natural_counts = worlds.groupby(NATURAL_KEYS, observed=True)["natural_final_size"].nunique()
    budget_counts = worlds["additional_targets"].fillna("").map(
        lambda value: 0 if not value else len(str(value).split("|"))
    )
    target_disjoint = worlds.apply(
        lambda row: not bool(
            set(str(row.detected_nodes).split("|"))
            & (set(str(row.additional_targets).split("|")) if str(row.additional_targets) else set())
        ),
        axis=1,
    )
    trigger_detected = worlds.apply(
        lambda row: str(row.initial_infected) in set(str(row.detected_nodes).split("|")),
        axis=1,
    )
    zero_false_positive_rows = worlds.loc[worlds["false_positive_rate"].eq(0)]
    detection_counts_reconcile = worlds["detected_cases"].eq(
        worlds["detected_infectious_cases"]
        + worlds["detected_noninfectious_animals"]
    )
    observation_keys = [
        "detection_profile",
        "secondary_case_sensitivity",
        "false_positive_rate",
    ]
    selection_keys = NATURAL_KEYS + observation_keys + ["method"]
    selected_set_counts = worlds.groupby(selection_keys, observed=True)[
        "additional_targets"
    ].nunique()
    detected_set_counts = worlds.groupby(
        NATURAL_KEYS + observation_keys, observed=True
    )["detected_nodes"].nunique()
    expected_regimes = 3 if stratify_by_disease_regime else 1
    matrix_size = expected_regimes * (
        len(detections)
        * len(action_delays)
        * len(residuals)
        * len(sensitivities)
        * len(false_positive_rates)
        * len(rewiring_fractions)
    )
    anchor_matrix_counts = (
        worlds[
            [
                "dataset_id",
                "network_id",
                "anchor_id",
                *(["disease_regime"] if stratify_by_disease_regime else []),
                "detection_profile",
                "action_delay_fraction",
                "residual_contact_multiplier",
                "secondary_case_sensitivity",
                "false_positive_rate",
                "rewiring_fraction",
                "rewiring_mode",
            ]
        ]
        .drop_duplicates()
        .groupby(["dataset_id", "network_id", "anchor_id"], observed=True)
        .size()
    )
    expected_families = {
        str(config["data"]["datasets"][dataset_id]["system_family"])
        for dataset_id in profile["datasets"]
    }
    zero_rewiring_matches_reference = True
    if zero_rewiring_reference_path is not None:
        reference = pd.read_csv(
            zero_rewiring_reference_path,
            dtype={"initial_infected": str, "network_id": str},
        )
        reference = reference.loc[
            reference["dataset_id"].isin(profile["datasets"])
            & reference["action_delay_fraction"].eq(action_delays[0])
            & reference["residual_contact_multiplier"].eq(residuals[0])
            & reference["secondary_case_sensitivity"].eq(sensitivities[0])
            & reference["false_positive_rate"].eq(false_positive_rates[0])
        ].copy()
        zero = worlds.loc[worlds["rewiring_fraction"].eq(0)].copy()
        comparison_keys = [
            "dataset_id",
            "network_id",
            "anchor_id",
            "parameter_id",
            "detection_profile",
            "action_delay_fraction",
            "residual_contact_multiplier",
            "secondary_case_sensitivity",
            "false_positive_rate",
            "random_block",
            "initial_infected",
            "world_seed",
            "method",
        ]
        comparison_values = [
            "detected_nodes",
            "additional_targets",
            "natural_final_size",
            "standard_final_size",
            "augmented_final_size",
        ]
        merged = zero[comparison_keys + comparison_values].merge(
            reference[comparison_keys + comparison_values],
            on=comparison_keys,
            suffixes=("_rewiring", "_reference"),
            validate="one_to_one",
        )
        zero_rewiring_matches_reference = (
            len(merged) == len(zero)
            and all(
                merged[f"{column}_rewiring"].fillna("").equals(
                    merged[f"{column}_reference"].fillna("")
                )
                for column in comparison_values
            )
        )
    hazard_roundoff_tolerance = np.maximum(
        1e-9,
        worlds["original_hazard_mass"].abs() * 1e-12,
    )
    audit = {
        "status": "pass",
        "checks": {
            "prerequisite_artifact_passed": prerequisite.get("status") == "pass",
            "policy_keys_unique_per_method": not worlds.duplicated(POLICY_KEYS + ["method"]).any(),
            "all_methods_complete": bool(method_counts.eq(len(config["decision"]["methods"])).all()),
            "all_random_blocks_complete": bool(
                block_counts.eq(int(profile["random_blocks"])).all()
            ),
            "standard_care_shared_across_methods": bool(standard_counts.eq(1).all()),
            "natural_world_shared_across_factorial_matrix": bool(natural_counts.eq(1).all()),
            "fixed_budget": bool(budget_counts.eq(worlds["additional_budget"]).all()),
            "detected_cases_excluded_from_additional_targets": bool(target_disjoint.all()),
            "trigger_case_always_observed": bool(trigger_detected.all()),
            "zero_rate_generates_no_false_positives": bool(
                zero_false_positive_rows["generated_false_positives"].eq(0).all()
            ),
            "detection_counts_reconcile": bool(detection_counts_reconcile.all()),
            "detected_evidence_fixed_across_delivery_scenarios": bool(detected_set_counts.eq(1).all()),
            "selected_targets_fixed_across_delivery_scenarios": bool(selected_set_counts.eq(1).all()),
            "action_never_precedes_detection": bool(pd.to_datetime(worlds["action_start"]).ge(pd.to_datetime(worlds["detection_time"])).all()),
            "complete_factorial_matrix_per_anchor": bool(anchor_matrix_counts.eq(matrix_size).all()),
            "all_disease_regimes_retained_per_anchor": bool(
                not stratify_by_disease_regime
                or worlds.groupby(
                    ["dataset_id", "network_id", "anchor_id"], observed=True
                )["disease_regime"].nunique().eq(expected_regimes).all()
            ),
            "all_configured_datasets_retained": set(worlds["dataset_id"].unique()) == set(profile["datasets"]),
            "window_coverage_accounted": bool(
                set(considered_windows) == set(profile["datasets"])
                and all(
                    0 < retained_windows[dataset_id] <= considered_windows[dataset_id]
                    for dataset_id in profile["datasets"]
                )
            ),
            "all_configured_system_families_retained": set(worlds["system_family"].unique()) == expected_families,
            "single_declared_epidemic_model": bool(
                worlds["epidemic_model"].nunique() == 1
                and worlds["epidemic_model"].iloc[0]
                == str(epidemic_model.get("name", "temporal_sir"))
            ),
            "disease_regimes_retain_attack_rate_support": bool(
                not stratify_by_disease_regime or attack_rate_support_passed
            ),
            "paired_rows_reconcile": len(paired) == len(worlds.loc[worlds["method"].ne(config["evaluation"]["primary_baseline"])]),
            "detection_timing_contrasts_reconcile": len(timing_contrasts) * 2 == len(paired),
            "finite_outcomes": bool(np.isfinite(worlds[["attack_rate_reduction", "selected_infected_fraction"]].to_numpy(float)).all()),
            "finite_hazard_accounting": bool(
                np.isfinite(
                    worlds[
                        [
                            "original_hazard_mass",
                            "standard_removed_hazard_mass",
                            "standard_rewired_hazard_mass",
                            "augmented_removed_hazard_mass",
                            "augmented_rewired_hazard_mass",
                            "additional_removed_hazard_fraction",
                            "additional_rewired_hazard_fraction",
                        ]
                    ].to_numpy(float)
                ).all()
            ),
            "removed_hazard_nonnegative_within_roundoff": bool(
                worlds["standard_removed_hazard_mass"]
                .ge(-hazard_roundoff_tolerance)
                .all()
                and worlds["augmented_removed_hazard_mass"]
                .ge(-hazard_roundoff_tolerance)
                .all()
            ),
            "rewired_hazard_does_not_exceed_removed_hazard": bool(
                worlds["standard_rewired_hazard_mass"]
                .le(
                    worlds["standard_removed_hazard_mass"].clip(lower=0)
                    + hazard_roundoff_tolerance
                )
                .all()
                and worlds["augmented_rewired_hazard_mass"]
                .le(
                    worlds["augmented_removed_hazard_mass"].clip(lower=0)
                    + hazard_roundoff_tolerance
                )
                .all()
            ),
            "zero_rewiring_reference_consistent_or_not_requested": bool(
                zero_rewiring_reference_path is None or zero_rewiring_matches_reference
            ),
        },
        "datasets": worlds["dataset_id"].nunique(),
        "system_families": worlds["system_family"].nunique(),
        "anchors": worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0],
        "window_coverage": {
            dataset_id: {
                "considered": considered_windows[dataset_id],
                "retained": retained_windows[dataset_id],
                "excluded_for_incomplete_parameter_triplet": (
                    considered_windows[dataset_id] - retained_windows[dataset_id]
                ),
            }
            for dataset_id in profile["datasets"]
        },
        "natural_worlds": worlds[NATURAL_KEYS].drop_duplicates().shape[0],
        "policy_evaluations": len(worlds),
        "scientific_result": scientific,
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"delivery-sensitivity audit failed: {audit}")
    worlds.to_csv(results_dir / "response_worlds.csv.gz", index=False, compression="gzip")
    paired.to_csv(results_dir / "paired_policy_increments.csv.gz", index=False, compression="gzip")
    absolute.to_csv(results_dir / "absolute_policy_summary.csv", index=False)
    absolute_family.to_csv(results_dir / "absolute_family_summary.csv", index=False)
    relative.to_csv(results_dir / "relative_policy_summary.csv", index=False)
    relative_family.to_csv(results_dir / "relative_family_summary.csv", index=False)
    block_summary.to_csv(results_dir / "random_block_family_summary.csv", index=False)
    timing_contrasts.to_csv(
        results_dir / "paired_detection_timing_contrasts.csv.gz",
        index=False,
        compression="gzip",
    )
    timing_summary.to_csv(results_dir / "detection_timing_contrast_summary.csv", index=False)
    timing_family.to_csv(results_dir / "detection_timing_contrast_family_summary.csv", index=False)
    if len(natural_attack_summary):
        natural_attack_family.to_csv(
            results_dir / "natural_attack_rate_family_summary.csv", index=False
        )
        natural_attack_summary.to_csv(
            results_dir / "natural_attack_rate_model_summary.csv", index=False
        )
    if len(cross_model_policy):
        cross_model_policy.to_csv(
            results_dir / "cross_model_policy_boundary.csv", index=False
        )
    if len(mechanism_paired):
        mechanism_paired.to_csv(
            results_dir / "rewiring_mechanism_pairs.csv.gz",
            index=False,
            compression="gzip",
        )
        mechanism_summary.to_csv(
            results_dir / "rewiring_mechanism_summary.csv", index=False
        )
    support_frame = pd.DataFrame(support_rows)
    used_parameters = set(
        worlds[["dataset_id", "network_id", "anchor_id", "parameter_id"]]
        .drop_duplicates()
        .astype(str)
        .agg("||".join, axis=1)
    )
    support_frame["selected_for_delivery"] = (
        support_frame[["dataset_id", "network_id", "anchor_id", "parameter_id"]]
        .astype(str)
        .agg("||".join, axis=1)
        .isin(used_parameters)
    )
    if stratify_by_disease_regime:
        regime_lookup = worlds[
            ["dataset_id", "network_id", "anchor_id", "parameter_id", "disease_regime"]
        ].drop_duplicates()
        support_frame = support_frame.merge(
            regime_lookup,
            on=["dataset_id", "network_id", "anchor_id", "parameter_id"],
            how="left",
            validate="many_to_one",
        )
    support_frame.to_csv(results_dir / "parameter_delivery_support.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    sensitivity_axis = str(config["evaluation"].get("sensitivity_axis", "delivery"))
    if sensitivity_axis == "observation":
        subtitle = (
            "Five animal-system families equally weighted; action delay = 0.1 infectious "
            "periods and contact reduction = 75%"
        )
        _plot_observation_heatmaps(
            absolute_primary,
            value_column="family_equal_mean",
            title="Absolute benefit under case-observation error",
            subtitle=subtitle,
            colorbar_label="Attack-rate reduction (percentage points)",
            path=report_dir / "absolute_benefit_observation_matrix.png",
        )
        _plot_observation_heatmaps(
            primary,
            value_column="family_equal_mean",
            title="Tracing increment under case-observation error",
            subtitle="Positive values favor adding case-linked temporal contact evidence",
            colorbar_label="Increment (percentage points)",
            path=report_dir / "relative_increment_observation_matrix.png",
        )
        _plot_observation_quality(
            worlds,
            report_dir / "realized_observation_quality.png",
        )
        _plot_detection_timing_contrast(
            timing_primary,
            report_dir / "detection_timing_contrast.png",
        )
        _plot_observation_family(
            relative_family.loc[
                relative_family["method"].eq(config["evaluation"]["primary_method"])
            ],
            sensitivity=float(config["evaluation"]["family_panel_sensitivity"]),
            false_positive_rate=float(
                config["evaluation"]["family_panel_false_positive_rate"]
            ),
            path=report_dir / "observation_family_heterogeneity.png",
        )
    elif sensitivity_axis == "disease_regime_rewiring":
        _plot_disease_regime_phase(
            primary,
            value_column="family_equal_mean",
            title="Policy phase diagram across calibrated disease regimes",
            subtitle="Positive values favor direct contact-to-detected targeting over the stable watchlist; animal-system families are equally weighted",
            colorbar_label="Direct-over-stable increment (percentage points)",
            path=report_dir / "policy_phase_diagram.png",
        )
        _plot_disease_regime_phase(
            absolute_primary,
            value_column="family_equal_mean",
            title="Absolute benefit of direct-contact targeting",
            subtitle="Benefit relative to detected-case isolation alone; animal-system families are equally weighted",
            colorbar_label="Attack-rate reduction (percentage points)",
            path=report_dir / "absolute_policy_phase_diagram.png",
        )
        _plot_disease_regime_family_support(
            relative_family,
            report_dir / "policy_phase_family_support.png",
        )
        if len(natural_attack_summary):
            _plot_cross_model_attack_rates(
                natural_attack_summary,
                report_dir / "epidemic_model_attack_rate_support.png",
            )
        if len(cross_model_policy):
            _plot_cross_model_policy_boundary(
                cross_model_policy,
                report_dir / "cross_model_policy_boundary.png",
            )
        winner_columns = ["disease_regime", "detection_profile", "rewiring_fraction"]
        winners = (
            absolute.sort_values(
                winner_columns + ["family_equal_mean"],
                ascending=[True, True, True, False],
            )
            .groupby(winner_columns, observed=True, as_index=False)
            .first()
        )
        winners.to_csv(results_dir / "disease_regime_policy_winners.csv", index=False)
    elif sensitivity_axis == "rewiring":
        _plot_rewiring_sensitivity(
            absolute_primary,
            primary,
            report_dir / "rewiring_sensitivity.png",
        )
        _plot_rewiring_family(
            relative_family.loc[
                relative_family["method"].eq(config["evaluation"]["primary_method"])
            ],
            report_dir / "rewiring_family_heterogeneity.png",
            policy_label=str(config["evaluation"]["primary_method"])
            .replace("_", " ")
            .title(),
        )
        if config["evaluation"]["primary_method"] != "contact_to_detected":
            _plot_rewiring_family(
                relative_family.loc[
                    relative_family["method"].eq("contact_to_detected")
                ],
                report_dir / "direct_contact_rewiring_family_heterogeneity.png",
                policy_label="Contact-to-detected",
            )
        _plot_rewiring_policy_comparison(
            absolute,
            report_dir / "rewiring_policy_comparison.png",
        )
        winner_columns = ["detection_profile", "rewiring_fraction"]
        winners = (
            absolute.sort_values(
                winner_columns + ["family_equal_mean"],
                ascending=[True, True, False],
            )
            .groupby(winner_columns, observed=True, as_index=False)
            .first()
        )
        winners.to_csv(results_dir / "rewiring_policy_winners.csv", index=False)
        if len(mechanism_paired):
            _plot_rewiring_mechanism(
                mechanism_paired,
                report_dir / "rewiring_mechanism_diagnostic.png",
            )
    else:
        _plot_heatmaps(
            absolute_primary,
            value_column="family_equal_mean",
            title="Absolute benefit of stable + tracing",
            subtitle="Five animal-system families equally weighted; 50% secondary-case sensitivity and 5% additional budget",
            colorbar_label="Attack-rate reduction (percentage points)",
            path=report_dir / "absolute_benefit_matrix.png",
        )
        _plot_heatmaps(
            primary,
            value_column="family_equal_mean",
            title="Stable + tracing increment over stable watchlist",
            subtitle="Positive values favor adding case-linked temporal contact evidence",
            colorbar_label="Increment (percentage points)",
            path=report_dir / "relative_increment_matrix.png",
        )
        _plot_family_operational(
            relative_family.loc[
                relative_family["method"].eq(config["evaluation"]["primary_method"])
            ],
            report_dir / "operational_family_heterogeneity.png",
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
            **(
                {"zero_rewiring_reference": _sha256(zero_rewiring_reference_path)}
                if zero_rewiring_reference_path is not None
                else {}
            ),
        },
        "audit_status": audit["status"],
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    observation_grid = (
        len(set(float(value) for value in sensitivities)) > 1
        or len(set(float(value) for value in false_positive_rates)) > 1
    )
    rewiring_grid = len(set(float(value) for value in rewiring_fractions)) > 1
    if sensitivity_axis == "disease_regime_rewiring":
        scope_note = (
            "Low, middle, and high disease regimes are fixed within each eligible window from the lower, median, and upper calibrated attack-rate profiles that support the full action timeline. "
            "These labels are within-dataset stress levels, not claims that beta values are biologically interchangeable across species. Rewiring is a compensatory partner-substitution stress test, not a fitted animal behavior model.\n"
        )
        result_note = (
            "\nThe preregistered retention gate and all family-level phase cells are recorded in `audit.json` and the result tables. "
            "Figures include `policy_phase_diagram.png`, `absolute_policy_phase_diagram.png`, and `policy_phase_family_support.png`.\n"
        )
    elif rewiring_grid:
        scope_note = (
            "After action starts, the configured fraction of contact hazard removed from a target-partner opportunity is redirected to a deterministic uniformly selected non-target while retaining the original non-target partner, time interval, and direction. "
            "This is a compensatory partner-substitution stress test, not a fitted animal behavior model.\n"
        )
        result_note = (
            "\nThe scientific classification and mechanism diagnostics are recorded in `audit.json` and the result tables. "
            "Any policy-switch signal remains conditional on this hazard-redistribution stress test and is not evidence that one rule is universally optimal.\n\n"
            "Figures include `rewiring_sensitivity.png`, `rewiring_policy_comparison.png`, "
            "`rewiring_family_heterogeneity.png`, and `rewiring_mechanism_diagnostic.png`.\n"
        )
    elif observation_grid:
        scope_note = (
            "The confirmed trigger case is always observed. Secondary infectious cases are sampled at the configured sensitivity, "
            "and noninfectious animals are sampled independently at the configured false-positive stress rate. "
            "The selected response set is fixed at detection time; these observation-error levels are generic sensitivity scenarios, not estimates of a particular field program.\n"
        )
        result_note = ""
    else:
        scope_note = (
            "The selected response set is fixed at detection time. Action delay represents implementation lag; no new evidence is incorporated during that lag. "
            "Residual contact is a node-level multiplier, so an exposure between two isolated animals receives the multiplier twice. These are generic operational scenarios, not estimates of a particular field program.\n"
        )
        result_note = ""
    (report_dir / "README.md").write_text(
        f"# {config['experiment']['name'].replace('_', ' ').title()}\n\n"
        f"Profile: **{profile_name}**. Artifact audit: **{audit['status']}**. Primary-increment classification: **{scientific['primary_increment_classification']}**.\n\n"
        + scope_note
        + result_note,
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intervention-delivery sensitivity")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260816-017_intervention_delivery_sensitivity.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
