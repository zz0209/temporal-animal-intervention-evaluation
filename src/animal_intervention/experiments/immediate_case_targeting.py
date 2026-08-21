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
from animal_intervention.simulation.outbreak_response import pre_detection_scores
from animal_intervention.surveillance import greedy_history_coverage, history_pair_weights

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
from .response_targeting_increment import (
    WORLD_KEYS,
    _classify,
    _compute_decomposition,
    _random_response_targets,
)
from .role_aware_sentinel_response import _replay_response, _top_history
from .sequential_preparedness_update import _budget, _parameters


def _ring_ranked_targets(
    *,
    contact_scores: dict[str, float],
    stable_scores: pd.DataFrame,
    eligible: set[str],
    initial: str,
    budget: int,
    seed: int,
    method: str,
) -> set[str]:
    candidates = stable_scores.loc[
        stable_scores["candidate_id"].astype(str).isin(eligible - {initial}),
        ["candidate_id", "stable_score"],
    ].copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    candidates["contact_score"] = (
        candidates["candidate_id"].map(contact_scores).fillna(0.0).astype(float)
    )
    candidates["has_contact"] = candidates["contact_score"].gt(0)
    tie_order = stable_hash_order(
        candidates["candidate_id"].tolist(), seed, method, "ties"
    )
    tie_rank = {node: index for index, node in enumerate(tie_order)}
    candidates["tie_rank"] = candidates["candidate_id"].map(tie_rank)
    selected = candidates.sort_values(
        ["has_contact", "contact_score", "stable_score", "tie_rank", "candidate_id"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).head(min(int(budget), len(candidates)))
    return set(selected["candidate_id"])


def _case_conditioned_history_sets(
    *,
    history_stream: Any,
    stable_scores: pd.DataFrame,
    eligible: set[str],
    initial: str,
    budget: int,
    history_start: pd.Timestamp,
    anchor_time: pd.Timestamp,
    recency_half_life: pd.Timedelta,
    seed: int,
) -> list[tuple[str, set[str]]]:
    """Build source-conditioned rings from pre-anchor contacts only.

    Both arms prioritize observed contacts of the confirmed index case and use the
    frozen stable watchlist only to fill unused capacity or break equal-contact ties.
    The static arm aggregates all past exposure mass; the temporal arm exponentially
    discounts older contacts using the pathogen's mean infectious period.
    """

    if budget <= 0:
        return []
    pair_weights = history_pair_weights(history_stream, eligible)
    static_scores = {
        str(node): float(value)
        for node, value in pair_weights.get(str(initial), {}).items()
    }
    recent = pre_detection_scores(
        history_stream,
        detected_nodes=(initial,),
        start_time=history_start,
        detection_time=anchor_time,
        half_life=recency_half_life,
    )
    recent_scores = dict(
        zip(
            recent["candidate_id"].astype(str),
            recent["contact_to_detected"].astype(float),
            strict=True,
        )
    )
    static_ring = _ring_ranked_targets(
        contact_scores=static_scores,
        stable_scores=stable_scores,
        eligible=eligible,
        initial=initial,
        budget=budget,
        seed=seed,
        method="past_weight_ring",
    )
    recent_ring = _ring_ranked_targets(
        contact_scores=recent_scores,
        stable_scores=stable_scores,
        eligible=eligible,
        initial=initial,
        budget=budget,
        seed=seed,
        method="past_recent_ring",
    )
    return [
        ("past_weight_ring", static_ring),
        ("past_recent_ring", recent_ring),
    ]


def _history_only_policy_scores(history_stream: Any, eligible: set[str]) -> pd.DataFrame:
    """Create a cumulative-contact policy score without intervention-value labels."""

    pair_weights = history_pair_weights(history_stream, eligible)
    totals = {
        str(node): float(sum(pair_weights.get(str(node), {}).values()))
        for node in eligible
    }
    frame = pd.DataFrame({"candidate_id": sorted(map(str, eligible))})
    frame["history_weight"] = frame["candidate_id"].map(totals).fillna(0.0)
    frame["stable_score"] = frame["history_weight"].rank(
        method="average", pct=True
    )
    frame["history_recency"] = frame["history_weight"]
    return frame


def _serialize_random_final_sizes(outcomes: list[int | float]) -> str:
    """Serialize individual random-policy outcomes for distributional auditing."""
    return "|".join(str(int(outcome)) for outcome in outcomes)


def _checkpoint_fingerprint(config_path: Path, source_path: Path) -> str:
    """Bind checkpoint identity to the exact configuration and experiment source."""
    return hashlib.sha256(config_path.read_bytes() + source_path.read_bytes()).hexdigest()[:12]


def _run_task(task: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    decision = config["decision"]
    evaluation = config["evaluation"]
    window = task["window"]
    anchor = window["anchor"]
    start_time = pd.Timestamp(anchor.anchor_time)
    end_time = pd.Timestamp(anchor.horizon_end)
    parameter = task["parameter"]
    mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
    engine, parameters = _parameters(parameter, task["model"], mean_period)
    eligible = set(map(str, window["eligible"]))
    population_size = len(window["future"].nodes())
    rows = []
    for block in range(task["random_blocks"]):
        for initial in task["seeds"]:
            initial = str(initial)
            world_seed = _keyed_seed(
                int(evaluation["seed"]),
                task["dataset_id"],
                anchor.anchor_id,
                parameter.parameter_id,
                block,
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
            response_capacity = _budget(
                len(eligible),
                int(decision["minimum_budget"]),
                float(decision["response_budget_fraction"]),
            )
            response_budget = min(response_capacity, len(eligible - {initial}))
            history = _top_history(
                task["stable_scores"],
                eligible,
                response_budget,
                world_seed,
                excluded={initial},
            )
            diversity_config = decision.get("diversity_response", {})
            diversity_enabled = bool(diversity_config.get("enabled", False))
            diversity_sets: list[tuple[str, set[str]]] = []
            if diversity_enabled and response_budget:
                remaining = eligible - {initial}
                multiplier = float(diversity_config["shortlist_multiplier"])
                shortlist_budget = min(
                    len(remaining),
                    max(response_budget, int(math.ceil(multiplier * response_budget))),
                )
                shortlist = _top_history(
                    task["stable_scores"],
                    eligible,
                    shortlist_budget,
                    world_seed,
                    excluded={initial},
                )
                shortlist_diverse = set(
                    greedy_history_coverage(
                        window["history"],
                        shortlist,
                        response_budget,
                        seed=_keyed_seed(world_seed, "shortlist_diversity"),
                    )
                )
                pure_coverage = set(
                    greedy_history_coverage(
                        window["history"],
                        remaining,
                        response_budget,
                        seed=_keyed_seed(world_seed, "pure_history_coverage"),
                    )
                )
                diversity_sets = [
                    ("history_shortlist_diverse", shortlist_diverse),
                    ("pure_history_coverage", pure_coverage),
                ]
                if any(len(nodes) != response_budget for _, nodes in diversity_sets):
                    raise AssertionError("diversity response sets must use equal capacity")
            case_history_config = decision.get("case_conditioned_history", {})
            case_history_enabled = bool(case_history_config.get("enabled", False))
            case_history_sets: list[tuple[str, set[str]]] = []
            if case_history_enabled and response_budget:
                ring_seed = _keyed_seed(
                    int(evaluation["seed"]),
                    task["dataset_id"],
                    task["network_id"],
                    anchor.anchor_id,
                    initial,
                    "case_conditioned_history",
                )
                case_history_sets = _case_conditioned_history_sets(
                    history_stream=window["history"],
                    stable_scores=task["stable_scores"],
                    eligible=eligible,
                    initial=initial,
                    budget=response_budget,
                    history_start=pd.Timestamp(anchor.history_start),
                    anchor_time=start_time,
                    recency_half_life=mean_period
                    * float(case_history_config["recency_half_life_fraction"]),
                    seed=ring_seed,
                )
                if any(len(nodes) != response_budget for _, nodes in case_history_sets):
                    raise AssertionError("case-conditioned response sets must use equal capacity")
            common = {
                "dataset_id": task["dataset_id"],
                "network_id": task["network_id"],
                "system_family": task["system_family"],
                "analysis_cluster_id": task["analysis_cluster_id"],
                "anchor_id": anchor.anchor_id,
                "anchor_time": start_time,
                "horizon_end": end_time,
                "parameter_id": parameter.parameter_id,
                "beta": float(parameter.beta),
                "mean_infectious_period_days": float(
                    parameter.mean_infectious_period_days
                ),
                "epidemic_model": task["model"]["name"],
                "random_block": block,
                "initial_infected": initial,
                "world_seed": world_seed,
                "population_size": population_size,
                "sentinel_method": "immediate_confirmed_index",
                "sentinel_budget": 1,
                "response_budget": response_budget,
                "response_capacity": response_capacity,
                "sentinel_nodes": initial,
                "detected": True,
                "detection_time": start_time,
                "action_start": start_time,
                "detection_burden": 1,
                "detection_burden_rate": 1 / population_size,
                "early_detection": True,
                "natural_final_size": natural.final_size,
            }
            policy_rows = []
            for method, additional in [
                ("case_only", set()),
                ("history_weight", history),
                *diversity_sets,
                *case_history_sets,
            ]:
                targets = {initial} | additional
                result, action_start = _replay_response(
                    engine=engine,
                    parameters=parameters,
                    future=window["future"],
                    natural=natural,
                    initial=initial,
                    world_seed=world_seed,
                    start_time=start_time,
                    end_time=end_time,
                    detection_time=start_time,
                    action_delay=pd.Timedelta(0),
                    targets=targets,
                    residual=float(decision["residual_contact_multiplier"]),
                )
                policy_rows.append(
                    {
                        **common,
                        "policy": f"immediate_index__{method}",
                        "response_method": method,
                        "response_nodes": "|".join(sorted(targets)),
                        "action_start": action_start,
                        "final_size": result.final_size,
                        "final_attack_rate": result.final_size / population_size,
                    }
                )
            random_outcomes = []
            random_nodes = []
            for target_replicate in range(task["random_target_replicates"]):
                target_seed = _keyed_seed(
                    int(evaluation["seed"]),
                    task["dataset_id"],
                    task["network_id"],
                    anchor.anchor_id,
                    parameter.parameter_id,
                    task["model"]["name"],
                    block,
                    initial,
                    "immediate_random_response",
                    target_replicate,
                )
                additional = _random_response_targets(
                    eligible,
                    response_budget,
                    target_seed,
                    excluded={initial},
                )
                targets = {initial} | additional
                result, _ = _replay_response(
                    engine=engine,
                    parameters=parameters,
                    future=window["future"],
                    natural=natural,
                    initial=initial,
                    world_seed=world_seed,
                    start_time=start_time,
                    end_time=end_time,
                    detection_time=start_time,
                    action_delay=pd.Timedelta(0),
                    targets=targets,
                    residual=float(decision["residual_contact_multiplier"]),
                )
                random_outcomes.append(result.final_size)
                random_nodes.append("|".join(sorted(targets)))
            policy_rows.append(
                {
                    **common,
                    "policy": "immediate_index__random",
                    "response_method": "random",
                    "response_nodes": " || ".join(random_nodes),
                    "final_size": float(np.mean(random_outcomes)),
                    "final_attack_rate": float(np.mean(random_outcomes))
                    / population_size,
                    "random_target_replicates": len(random_outcomes),
                    "random_final_sizes": _serialize_random_final_sizes(
                        random_outcomes
                    ),
                }
            )
            rows.append(pd.DataFrame(policy_rows))
    return pd.concat(
        [frame.dropna(axis=1, how="all") for frame in rows], ignore_index=True
    )


def _diversity_contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    required = {"history_shortlist_diverse", "pure_history_coverage"}
    if not required.issubset(set(worlds["response_method"])):
        return pd.DataFrame()
    metadata_columns = WORLD_KEYS + [
        "system_family",
        "analysis_cluster_id",
        "population_size",
        "response_budget",
    ]
    metadata = worlds.drop_duplicates(WORLD_KEYS)[metadata_columns]
    wide = worlds.pivot(
        index=WORLD_KEYS,
        columns="response_method",
        values="final_size",
    ).reset_index()
    paired = metadata.merge(wide, on=WORLD_KEYS, validate="one_to_one")
    definitions = {
        "shortlist_diversity_vs_random": ("random", "history_shortlist_diverse"),
        "shortlist_diversity_vs_top_history": (
            "history_weight",
            "history_shortlist_diverse",
        ),
        "pure_coverage_vs_random": ("random", "pure_history_coverage"),
    }
    rows = []
    for contrast, (reference, challenger) in definitions.items():
        frame = paired[metadata_columns].copy()
        frame["contrast"] = contrast
        frame["value"] = (
            paired[reference] - paired[challenger]
        ) / paired["population_size"].astype(float)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _ring_contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    required = {"past_weight_ring", "past_recent_ring"}
    if not required.issubset(set(worlds["response_method"])):
        return pd.DataFrame()
    metadata_columns = WORLD_KEYS + [
        "system_family",
        "analysis_cluster_id",
        "population_size",
        "response_budget",
    ]
    metadata = worlds.drop_duplicates(WORLD_KEYS)[metadata_columns]
    wide = worlds.pivot(
        index=WORLD_KEYS,
        columns="response_method",
        values="final_size",
    ).reset_index()
    paired = metadata.merge(wide, on=WORLD_KEYS, validate="one_to_one")
    definitions = {
        "recent_ring_vs_random": ("random", "past_recent_ring"),
        "recent_ring_vs_stable": ("history_weight", "past_recent_ring"),
        "recent_ring_vs_static_ring": ("past_weight_ring", "past_recent_ring"),
        "static_ring_vs_random": ("random", "past_weight_ring"),
    }
    rows = []
    for contrast, (reference, challenger) in definitions.items():
        frame = paired[metadata_columns].copy()
        frame["contrast"] = contrast
        frame["value"] = (
            paired[reference] - paired[challenger]
        ) / paired["population_size"].astype(float)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _selection_geometry(worlds: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "history_weight",
        "history_shortlist_diverse",
        "pure_history_coverage",
    ]
    if not set(methods).issubset(set(worlds["response_method"])):
        return pd.DataFrame()
    index = WORLD_KEYS + ["response_budget", "system_family"]
    wide = worlds.pivot(
        index=index,
        columns="response_method",
        values="response_nodes",
    ).reset_index()
    rows = []
    for row in wide.itertuples(index=False):
        initial = str(row.initial_infected)

        def parse(value: str) -> set[str]:
            return set(str(value).split("|")) - {initial}

        top = parse(row.history_weight)
        shortlist = parse(row.history_shortlist_diverse)
        pure = parse(row.pure_history_coverage)
        rows.append(
            {
                **{column: getattr(row, column) for column in index},
                "multi_node_budget": int(row.response_budget) > 1,
                "shortlist_equals_top": shortlist == top,
                "shortlist_top_jaccard": (
                    len(shortlist & top) / len(shortlist | top)
                    if shortlist | top
                    else 1.0
                ),
                "pure_top_jaccard": (
                    len(pure & top) / len(pure | top) if pure | top else 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _ring_geometry(worlds: pd.DataFrame) -> pd.DataFrame:
    methods = ["history_weight", "past_weight_ring", "past_recent_ring"]
    if not set(methods).issubset(set(worlds["response_method"])):
        return pd.DataFrame()
    index = WORLD_KEYS + ["response_budget", "system_family"]
    wide = worlds.pivot(
        index=index,
        columns="response_method",
        values="response_nodes",
    ).reset_index()
    rows = []
    for row in wide.itertuples(index=False):
        initial = str(row.initial_infected)

        def parse(value: str) -> set[str]:
            return set(str(value).split("|")) - {initial}

        stable = parse(row.history_weight)
        static = parse(row.past_weight_ring)
        recent = parse(row.past_recent_ring)

        def jaccard(left: set[str], right: set[str]) -> float:
            return len(left & right) / len(left | right) if left | right else 1.0

        rows.append(
            {
                **{column: getattr(row, column) for column in index},
                "recent_equals_stable": recent == stable,
                "recent_equals_static_ring": recent == static,
                "recent_stable_jaccard": jaccard(recent, stable),
                "recent_static_jaccard": jaccard(recent, static),
            }
        )
    return pd.DataFrame(rows)


def _classify_diversity(
    summary: pd.DataFrame,
    family: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    diversity = summary.loc[
        summary["contrast"].isin(
            [
                "shortlist_diversity_vs_random",
                "shortlist_diversity_vs_top_history",
                "pure_coverage_vs_random",
            ]
        )
    ]
    for result in diversity.itertuples(index=False):
        family_frame = family.loc[
            family["epidemic_model"].eq(result.epidemic_model)
            & family["contrast"].eq(result.contrast)
        ]
        required = int(math.ceil(0.8 * int(result.families)))
        positive = int(family_frame["mean_value"].gt(0).sum())
        if float(result.ci_low) > 0 and positive >= required:
            decision = "strong"
        elif float(result.family_equal_mean) > 0 and positive >= required:
            decision = "directional"
        else:
            decision = "unsupported"
        rows.append(
            {
                "epidemic_model": result.epidemic_model,
                "contrast": result.contrast,
                "decision": decision,
                "estimate": float(result.family_equal_mean),
                "ci_low": float(result.ci_low),
                "ci_high": float(result.ci_high),
                "positive_families": positive,
                "families": int(result.families),
            }
        )
    return pd.DataFrame(rows)


def _diversity_repair_gate(decisions: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {
            "overall": "not_applicable",
            "per_model": {},
            "rule": "Diversity response was not enabled for this experiment.",
        }
    required = {
        "shortlist_diversity_vs_random",
        "shortlist_diversity_vs_top_history",
    }
    per_model: dict[str, str] = {}
    for model, frame in decisions.groupby("epidemic_model", observed=True):
        selected = frame.loc[frame["contrast"].isin(required)]
        if set(selected["contrast"]) != required:
            per_model[str(model)] = "incomplete"
            continue
        model_decisions = set(selected["decision"])
        if model_decisions == {"strong"}:
            per_model[str(model)] = "strong"
        elif model_decisions.issubset({"strong", "directional"}):
            per_model[str(model)] = "directional"
        else:
            per_model[str(model)] = "unsupported"
    expected_models = {"temporal_sir", "temporal_seir_erlang"}
    if set(per_model) != expected_models:
        overall = "incomplete"
    elif set(per_model.values()) == {"strong"}:
        overall = "strong"
    elif set(per_model.values()).issubset({"strong", "directional"}):
        overall = "directional"
    else:
        overall = "unsupported"
    return {
        "overall": overall,
        "per_model": per_model,
        "rule": (
            "Both primary shortlist-diversity contrasts must meet the same "
            "decision level within both epidemic models."
        ),
    }


def _classify_ring(
    summary: pd.DataFrame,
    family: pd.DataFrame,
) -> pd.DataFrame:
    contrasts = {
        "recent_ring_vs_random",
        "recent_ring_vs_stable",
        "recent_ring_vs_static_ring",
        "static_ring_vs_random",
    }
    rows = []
    for result in summary.loc[summary["contrast"].isin(contrasts)].itertuples(
        index=False
    ):
        family_frame = family.loc[
            family["epidemic_model"].eq(result.epidemic_model)
            & family["contrast"].eq(result.contrast)
        ]
        required = int(math.ceil(0.8 * int(result.families)))
        positive = int(family_frame["mean_value"].gt(0).sum())
        if float(result.ci_low) > 0 and positive >= required:
            decision = "strong"
        elif float(result.family_equal_mean) > 0 and positive >= required:
            decision = "directional"
        else:
            decision = "unsupported"
        rows.append(
            {
                "epidemic_model": result.epidemic_model,
                "contrast": result.contrast,
                "decision": decision,
                "estimate": float(result.family_equal_mean),
                "ci_low": float(result.ci_low),
                "ci_high": float(result.ci_high),
                "positive_families": positive,
                "families": int(result.families),
            }
        )
    return pd.DataFrame(rows)


def _joint_decision_gate(
    decisions: pd.DataFrame,
    required_contrasts: set[str],
) -> dict[str, Any]:
    per_model: dict[str, str] = {}
    for model, frame in decisions.groupby("epidemic_model", observed=True):
        selected = frame.loc[frame["contrast"].isin(required_contrasts)]
        if set(selected["contrast"]) != required_contrasts:
            per_model[str(model)] = "incomplete"
            continue
        levels = set(selected["decision"])
        if levels == {"strong"}:
            per_model[str(model)] = "strong"
        elif levels.issubset({"strong", "directional"}):
            per_model[str(model)] = "directional"
        else:
            per_model[str(model)] = "unsupported"
    expected_models = {"temporal_sir", "temporal_seir_erlang"}
    if set(per_model) != expected_models:
        overall = "incomplete"
    elif set(per_model.values()) == {"strong"}:
        overall = "strong"
    elif set(per_model.values()).issubset({"strong", "directional"}):
        overall = "directional"
    else:
        overall = "unsupported"
    return {"overall": overall, "per_model": per_model}


def _ring_gates(decisions: pd.DataFrame) -> dict[str, Any]:
    return {
        "operational_case_conditioning": _joint_decision_gate(
            decisions,
            {"recent_ring_vs_random", "recent_ring_vs_stable"},
        ),
        "temporal_recency_increment": _joint_decision_gate(
            decisions,
            {"recent_ring_vs_static_ring"},
        ),
        "rules": {
            "strong": (
                "Every required contrast has a positive family-bootstrap interval "
                "and at least four of five positive family means in both models."
            ),
            "directional": (
                "Every required contrast has a positive mean and at least four of "
                "five positive family means in both models, but an interval crosses zero."
            ),
            "unsupported": "Otherwise.",
        },
    }


def _plot_diversity(
    summary: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    order = [
        "shortlist_diversity_vs_random",
        "shortlist_diversity_vs_top_history",
        "pure_coverage_vs_random",
    ]
    labels = [
        "Shortlist diversity vs random",
        "Shortlist diversity vs top history",
        "Pure coverage vs random",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = (
            summary.loc[
                summary["epidemic_model"].eq(model)
                & summary["contrast"].isin(order)
            ]
            .set_index("contrast")
            .loc[order]
        )
        y = np.arange(len(order))
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        axis.errorbar(
            mean,
            y,
            xerr=np.vstack((mean - low, high - mean)),
            fmt="o",
            markersize=8,
            capsize=5,
            color="#4C78A8",
        )
        axis.axvline(0, color="#555555", linestyle="--")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_xlabel("Avoided attack-rate percentage points")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Can history-only set diversification close the policy gap?",
        fontsize=20,
        weight="bold",
        y=0.94,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.78, bottom=0.14, wspace=0.18)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_ring(
    summary: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    order = [
        "recent_ring_vs_random",
        "recent_ring_vs_stable",
        "recent_ring_vs_static_ring",
        "static_ring_vs_random",
    ]
    labels = [
        "Recent ring vs random",
        "Recent ring vs stable list",
        "Recent ring vs static ring",
        "Static ring vs random",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = (
            summary.loc[
                summary["epidemic_model"].eq(model)
                & summary["contrast"].isin(order)
            ]
            .set_index("contrast")
            .loc[order]
        )
        y = np.arange(len(order))
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        axis.errorbar(
            mean,
            y,
            xerr=np.vstack((mean - low, high - mean)),
            fmt="o",
            markersize=8,
            capsize=5,
            color="#4C78A8",
        )
        axis.axvline(0, color="#555555", linestyle="--")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_xlabel("Avoided attack-rate percentage points")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Does a confirmed case make past contacts actionable?",
        fontsize=20,
        weight="bold",
        y=0.94,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.78, bottom=0.14, wspace=0.18)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot(summary: pd.DataFrame, family: pd.DataFrame, path: Path, dpi: int) -> None:
    target = summary.loc[summary["contrast"].eq("targeting_increment")]
    family_target = family.loc[family["contrast"].eq("targeting_increment")]
    families = sorted(family_target["system_family"].unique())
    labels = [SYSTEM_FAMILY_LABELS.get(item, item.replace("_", " ")) for item in families]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), sharey=False)
    models = ["temporal_sir", "temporal_seir_erlang"]
    estimates = target.set_index("epidemic_model").loc[models]
    x = np.arange(2)
    mean = 100 * estimates["family_equal_mean"].to_numpy(float)
    low = 100 * estimates["ci_low"].to_numpy(float)
    high = 100 * estimates["ci_high"].to_numpy(float)
    axes[0].errorbar(
        x,
        mean,
        yerr=[mean - low, high - mean],
        fmt="o",
        markersize=9,
        capsize=5,
        color="#4C78A8",
    )
    axes[0].axhline(0, color="#555555", linestyle="--")
    axes[0].set_xticks(x, ["Temporal SIR", "Staged SEIR/Erlang"])
    axes[0].set_ylabel("History minus random benefit (attack-rate points)")
    axes[0].set_title("Family-equal targeting increment", fontsize=16, weight="bold")
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.36
    for offset, model, color, label in [
        (-width / 2, "temporal_sir", "#4C78A8", "SIR"),
        (width / 2, "temporal_seir_erlang", "#F58518", "SEIR/Erlang"),
    ]:
        values = (
            family_target.loc[family_target["epidemic_model"].eq(model)]
            .set_index("system_family")
            .loc[families, "mean_value"]
            .to_numpy(float)
        )
        axes[1].barh(
            np.arange(len(families)) + offset,
            100 * values,
            height=width,
            color=color,
            label=label,
        )
    axes[1].axvline(0, color="#555555", linestyle="--")
    axes[1].set_yticks(np.arange(len(families)), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("History minus random benefit (attack-rate points)")
    axes[1].set_title("Independent-system direction", fontsize=16, weight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Does a precomputed history list help at the first confirmed case?",
        fontsize=20,
        weight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.12, wspace=0.34)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = dict(config["profiles"][profile_name])
    evaluation = config["evaluation"]
    stable = pd.read_csv(
        config["data"]["stable_prediction_path"],
        dtype={"candidate_id": str, "network_id": str},
    )
    stable["anchor_time"] = pd.to_datetime(stable["anchor_time"], format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / config["experiment"]["id"] / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / config["experiment"]["id"] / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        for window in windows:
            window.setdefault("network_id", default_network_id)
        available_keys = set(
            stable.loc[
                stable["dataset_id"].eq(dataset_id),
                ["network_id", "anchor_time"],
            ].itertuples(index=False, name=None)
        )
        fallback_datasets = set(
            config["data"].get("history_score_fallback_datasets", [])
        )
        if dataset_id not in fallback_datasets:
            windows = [
                window
                for window in windows
                if (
                    str(window["network_id"]),
                    pd.Timestamp(window["anchor"].anchor_time),
                )
                in available_keys
            ]
        maximum = profile.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        parameters = _parameter_pool(
            Path(specification["source_results"]) / "parameter_selection.csv",
            str(evaluation["parameter_pool"]),
        )
        for window in windows:
            selected = _select_parameter_regimes(
                list(parameters.itertuples(index=False)),
                str(evaluation["parameter_selection_mode"]),
            )
            if len(selected) != 1:
                continue
            _, parameter = selected[0]
            network_id = str(window["network_id"])
            if (
                network_id,
                pd.Timestamp(window["anchor"].anchor_time),
            ) in available_keys:
                score = _matching_stable_scores(
                    stable,
                    dataset_id,
                    network_id,
                    window["anchor"].anchor_time,
                    window["eligible"],
                )
            else:
                score = _history_only_policy_scores(
                    window["history"], set(map(str, window["eligible"]))
                )
            seeds = stable_hash_order(
                list(map(str, window["eligible"])),
                int(evaluation["seed"]),
                dataset_id,
                window["anchor"].anchor_id,
                "role_aware_seeds",
            )[: int(profile["seeds_per_anchor"])]
            cluster = (
                f"{dataset_id}::{network_id}"
                if specification.get("analysis_cluster") == "network"
                else f"{dataset_id}::{network_id}::{window['anchor'].anchor_id}"
            )
            for model in config["decision"]["epidemic_models"]:
                tasks.append(
                    {
                        "dataset_id": dataset_id,
                        "network_id": network_id,
                        "system_family": specification["system_family"],
                        "analysis_cluster_id": cluster,
                        "window": window,
                        "parameter": parameter,
                        "model": dict(model),
                        "stable_scores": score,
                        "seeds": seeds,
                        "random_blocks": int(profile["random_blocks"]),
                        "random_target_replicates": int(
                            profile["random_target_replicates"]
                        ),
                    }
                )
    fingerprint = _checkpoint_fingerprint(config_path, Path(__file__))
    frames = []
    for task in tqdm(tasks, desc="Immediate case-triggered worlds", unit="task"):
        identity = "|".join(
            [
                fingerprint,
                task["dataset_id"],
                task["network_id"],
                task["window"]["anchor"].anchor_id,
                task["model"]["name"],
            ]
        )
        checkpoint = checkpoint_dir / f"world_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
        if checkpoint.exists() and config["execution"].get("resume", True):
            frame = pd.read_csv(
                checkpoint, dtype={"initial_infected": str, "network_id": str}
            )
        else:
            frame = _run_task(task, config)
            frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
    worlds = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True
    )
    reference_reconciles = True
    reference_worlds_path = config["data"].get("reference_worlds_path")
    if reference_worlds_path:
        reference = pd.read_csv(
            reference_worlds_path,
            dtype={"initial_infected": str, "network_id": str},
        )
        base_methods = {"case_only", "history_weight", "random"}
        key = [*WORLD_KEYS, "response_method"]
        current_base = worlds.loc[
            worlds["response_method"].isin(base_methods),
            [*key, "final_size", "response_nodes"],
        ]
        reference_base = reference.loc[
            reference["response_method"].isin(base_methods),
            [*key, "final_size", "response_nodes"],
        ].merge(
            current_base[WORLD_KEYS].drop_duplicates(),
            on=WORLD_KEYS,
            how="inner",
            validate="many_to_one",
        )
        aligned = current_base.merge(
            reference_base,
            on=key,
            how="outer",
            suffixes=("_current", "_reference"),
            indicator=True,
            validate="one_to_one",
        )
        reference_reconciles = bool(
            len(aligned) == len(current_base) == len(reference_base)
            and aligned["_merge"].eq("both").all()
            and np.allclose(
                aligned["final_size_current"],
                aligned["final_size_reference"],
            )
            and aligned["response_nodes_current"]
            .fillna("")
            .eq(aligned["response_nodes_reference"].fillna(""))
            .all()
        )
    contrasts = _compute_decomposition(worlds)
    diversity_contrasts = _diversity_contrasts(worlds)
    if not diversity_contrasts.empty:
        contrasts = pd.concat([contrasts, diversity_contrasts], ignore_index=True)
    ring_contrasts = _ring_contrasts(worlds)
    if not ring_contrasts.empty:
        contrasts = pd.concat([contrasts, ring_contrasts], ignore_index=True)
    summary, family = _hierarchical_summary(
        contrasts,
        value_column="value",
        group_columns=["epidemic_model", "contrast"],
        bootstrap_replicates=int(
            profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"])
        ),
        seed=int(evaluation["seed"]),
    )
    decisions = _classify(summary, family)
    diversity_decisions = _classify_diversity(summary, family)
    diversity_repair_gate = _diversity_repair_gate(diversity_decisions)
    ring_decisions = _classify_ring(summary, family)
    ring_gates = _ring_gates(ring_decisions)
    diversity_enabled = bool(
        config["decision"].get("diversity_response", {}).get("enabled", False)
    )
    ring_enabled = bool(
        config["decision"].get("case_conditioned_history", {}).get("enabled", False)
    )
    expected_policy_arms = 3 + (2 if diversity_enabled else 0) + (2 if ring_enabled else 0)
    geometry = _selection_geometry(worlds)
    ring_geometry = _ring_geometry(worlds)
    multi_summary = pd.DataFrame()
    multi_family = pd.DataFrame()
    if diversity_enabled:
        multi_contrasts = diversity_contrasts.loc[
            diversity_contrasts["response_budget"].gt(1)
        ]
        if not multi_contrasts.empty:
            multi_summary, multi_family = _hierarchical_summary(
                multi_contrasts,
                value_column="value",
                group_columns=["epidemic_model", "contrast"],
                bootstrap_replicates=int(
                    profile.get(
                        "bootstrap_replicates",
                        evaluation["bootstrap_replicates"],
                    )
                ),
                seed=int(evaluation["seed"]) + 700_000,
            )
    checks = {
        "all_requested_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "expected_families_full": profile_name != "full" or worlds["system_family"].nunique()
        == int(profile.get("expected_system_families", 5)),
        "expected_policy_arms": bool(
            worlds.groupby(WORLD_KEYS, observed=True)["response_method"]
            .nunique()
            .eq(expected_policy_arms)
            .all()
        ),
        "immediate_detection": bool(
            pd.to_datetime(worlds["detection_time"], format="mixed").eq(
                pd.to_datetime(worlds["anchor_time"], format="mixed")
            ).all()
        ),
        "immediate_action": bool(
            pd.to_datetime(worlds["action_start"], format="mixed").eq(
                pd.to_datetime(worlds["anchor_time"], format="mixed")
            ).all()
        ),
        "finite_contrasts": bool(np.isfinite(contrasts["value"]).all()),
        "eight_random_lists_full": profile_name != "full" or bool(
            worlds.loc[worlds["response_method"].eq("random"), "random_target_replicates"]
            .eq(8)
            .all()
        ),
        "diversity_contrasts_complete": bool(
            not diversity_enabled
            or set(diversity_decisions["contrast"])
            == {
                "shortlist_diversity_vs_random",
                "shortlist_diversity_vs_top_history",
                "pure_coverage_vs_random",
            }
        ),
        "diversity_sets_materially_distinct": bool(
            not diversity_enabled
            or (
                geometry.loc[geometry["multi_node_budget"], "shortlist_equals_top"]
                .mean()
                < 0.5
            )
        ),
        "three_multinode_families": bool(
            not diversity_enabled
            or geometry.loc[geometry["multi_node_budget"], "system_family"].nunique()
            == 3
        ),
        "ring_contrasts_complete": bool(
            not ring_enabled
            or set(ring_decisions["contrast"])
            == {
                "recent_ring_vs_random",
                "recent_ring_vs_stable",
                "recent_ring_vs_static_ring",
                "static_ring_vs_random",
            }
        ),
        "ring_sets_materially_distinct": bool(
            not ring_enabled
            or profile_name != "full"
            or ring_geometry["recent_equals_stable"].mean() < 0.5
        ),
        "reference_base_worlds_reconcile": bool(
            profile_name != "full" or reference_reconciles
        ),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "datasets": int(worlds["dataset_id"].nunique()),
        "families": int(worlds["system_family"].nunique()),
        "anchors": int(worlds[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0]),
        "paired_worlds": int(worlds[WORLD_KEYS].drop_duplicates().shape[0]),
        "policy_evaluations": len(worlds),
        "decisions": decisions.set_index("epidemic_model")["decision"].to_dict(),
        "diversity_decisions": diversity_decisions.to_dict(orient="records"),
        "diversity_repair_gate": diversity_repair_gate,
        "ring_decisions": ring_decisions.to_dict(orient="records"),
        "ring_gates": ring_gates,
        "scope": "immediate_confirmed_index_equal_capacity_targeting",
    }
    if audit["status"] != "pass":
        raise ValueError(f"immediate targeting audit failed: {audit}")
    worlds.to_csv(results_dir / "policy_worlds.csv.gz", index=False, compression="gzip")
    contrasts.to_csv(results_dir / "paired_decomposition.csv.gz", index=False, compression="gzip")
    summary.to_csv(results_dir / "contrast_summary.csv", index=False)
    family.to_csv(results_dir / "family_contrasts.csv", index=False)
    decisions.to_csv(results_dir / "targeting_decisions.csv", index=False)
    if diversity_enabled:
        diversity_decisions.to_csv(
            results_dir / "diversity_decisions.csv",
            index=False,
        )
        (results_dir / "diversity_repair_gate.json").write_text(
            json.dumps(diversity_repair_gate, indent=2),
            encoding="utf-8",
        )
        geometry.to_csv(results_dir / "selection_geometry.csv.gz", index=False, compression="gzip")
        multi_summary.to_csv(results_dir / "multinode_contrast_summary.csv", index=False)
        multi_family.to_csv(results_dir / "multinode_family_contrasts.csv", index=False)
    if ring_enabled:
        ring_decisions.to_csv(results_dir / "ring_decisions.csv", index=False)
        ring_geometry.to_csv(
            results_dir / "ring_selection_geometry.csv.gz",
            index=False,
            compression="gzip",
        )
        (results_dir / "ring_gates.json").write_text(
            json.dumps(ring_gates, indent=2),
            encoding="utf-8",
        )
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [
        config_path,
        Path(__file__),
        Path(config["data"]["stable_prediction_path"]),
    ]
    if reference_worlds_path:
        source_paths.append(Path(reference_worlds_path))
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_paths.extend(
            [
                Path(specification["source_config"]),
                Path(specification["source_results"]) / "parameter_selection.csv",
            ]
        )
    pd.DataFrame(
        [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        ]
    ).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {
        "experiment_id": config["experiment"]["id"],
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
    _plot(summary, family, report_dir / "immediate_targeting_increment.png", int(profile["render_dpi"]))
    if diversity_enabled:
        _plot_diversity(
            summary,
            report_dir / "diversity_targeting_increment.png",
            int(profile["render_dpi"]),
        )
        _plot_diversity(
            multi_summary,
            report_dir / "multinode_diversity_targeting_increment.png",
            int(profile["render_dpi"]),
        )
    if ring_enabled:
        _plot_ring(
            summary,
            report_dir / "case_conditioned_history.png",
            int(profile["render_dpi"]),
        )
    display = decisions.copy()
    for column in ["targeting_increment", "ci_low", "ci_high"]:
        display[column] = 100 * display[column]
    report = "# Immediate case-triggered targeting\n\n"
    report += "This experiment isolates the best-timing boundary for a precomputed history list: the first index case is confirmed at the forecast anchor, the index is handled immediately, and an equal additional capacity is assigned either by frozen history scores or by eight independent random lists.\n\n"
    report += _markdown_table(display)
    report += "\n\nThis is an idealized timing boundary, not a field estimate of diagnostic delay or intervention efficacy.\n"
    if diversity_enabled:
        report += "\n## History-only set diversification\n\n"
        diversity_display = diversity_decisions.copy()
        for column in ["estimate", "ci_low", "ci_high"]:
            diversity_display[column] = 100 * diversity_display[column]
        report += _markdown_table(diversity_display)
        report += (
            "\n\nThe primary diversity selector first restricts candidates to the "
            "top history-score shortlist and then greedily maximizes nonredundant "
            "past-neighbor coverage. It uses no future contacts or epidemic outcomes.\n"
        )
        multi_geometry = geometry.loc[geometry["multi_node_budget"]]
        report += (
            "\nAcross multi-node worlds, the shortlist-diversified set exactly matched "
            f"top-history in {multi_geometry['shortlist_equals_top'].mean():.1%} of worlds "
            f"and had mean Jaccard overlap {multi_geometry['shortlist_top_jaccard'].mean():.3f}. "
            "The multi-node-only summary is a post-primary mechanism diagnostic over "
            "three independent families, not a replacement confirmatory estimand.\n"
        )
    if ring_enabled:
        ring_display = ring_decisions.copy()
        for column in ["estimate", "ci_low", "ci_high"]:
            ring_display[column] = 100 * ring_display[column]
        report += "\n## Case-conditioned history rings\n\n"
        report += _markdown_table(ring_display)
        report += (
            "\n\nThe recent ring prioritizes pre-anchor contacts of the confirmed "
            "index case with exponential recency decay and fills any unused slots "
            "from the frozen stable list. The static ring uses the same mapper and "
            "capacity without recency decay. Both are deployment-valid at the anchor.\n"
        )
        report += (
            "\nJoint gates: `"
            f"operational={ring_gates['operational_case_conditioning']['overall']}`; `"
            f"temporal_recency={ring_gates['temporal_recency_increment']['overall']}`.\n"
        )
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run immediate confirmed-case targeting experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
