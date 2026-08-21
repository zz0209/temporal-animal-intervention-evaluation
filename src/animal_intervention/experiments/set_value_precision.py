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

from animal_intervention.simulation import PairedTemporalSIREngine, SIRParameters, states_at
from animal_intervention.simulation.outbreak_response import (
    observe_detected_cases,
    pre_detection_scores,
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
from .set_value_pilot import FEATURE_COLUMNS, _percentile, _set_features


OBSERVATION_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "detection_profile",
    "evidence_profile",
    "budget_fraction",
    "initial_infected",
    "observation_random_block",
    "observation_world_seed",
]


def _nodes(value: object) -> tuple[str, ...]:
    if pd.isna(value) or str(value) == "":
        return ()
    return tuple(sorted(item for item in str(value).split("|") if item))


def _rename_observation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "random_block": "observation_random_block",
            "world_seed": "observation_world_seed",
        }
    )


def _select_contexts(values: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    selected = values.loc[
        values["dataset_id"].isin(profile["datasets"]) & values["budget"].gt(1)
    ].copy()
    maximum = profile.get("max_contexts_per_dataset")
    if maximum is None:
        return selected
    context_frames = []
    for _, group in selected.groupby("dataset_id", observed=True, sort=True):
        contexts = group[OBSERVATION_KEYS].drop_duplicates().sort_values(OBSERVATION_KEYS)
        context_frames.append(contexts.head(int(maximum)))
    keep = pd.concat(context_frames, ignore_index=True)
    return selected.merge(keep, on=OBSERVATION_KEYS, how="inner", validate="many_to_one")


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() <= 1 or right.nunique() <= 1:
        return float("nan")
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _ordered_signatures(values: pd.Series) -> list[str]:
    frame = values.rename("value").reset_index()
    return frame.sort_values(
        ["value", "set_signature"], ascending=[False, True], kind="stable"
    )["set_signature"].astype(str).tolist()


def _context_reliability(
    replicate_values: pd.DataFrame,
    evaluation: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in replicate_values.groupby(OBSERVATION_KEYS, observed=True, sort=True):
        pivot = group.pivot(
            index="set_signature", columns="continuation_block", values="set_attack_rate_value"
        ).sort_index()
        blocks = list(pivot.columns)
        midpoint = len(blocks) // 2
        if midpoint == 0 or midpoint == len(blocks):
            raise ValueError("at least two continuation blocks are required")
        first = pivot[blocks[:midpoint]].mean(axis=1)
        second = pivot[blocks[midpoint:]].mean(axis=1)
        full = pivot.mean(axis=1)
        count = len(full)
        top_count = max(1, int(math.ceil(count * float(evaluation["top_fraction"]))))
        first_order = _ordered_signatures(first)
        second_order = _ordered_signatures(second)
        first_top = set(first_order[:top_count])
        second_top = set(second_order[:top_count])
        overlap = len(first_top & second_top) / top_count
        chance = top_count / count
        first_best, first_worst = first_order[0], first_order[-1]
        second_best, second_worst = second_order[0], second_order[-1]
        regret_second = float(second.max() - second.loc[first_best])
        regret_first = float(first.max() - first.loc[second_best])
        cross_half_regret = (regret_first + regret_second) / 2
        spread = (float(first.max() - first.min()) + float(second.max() - second.min())) / 2
        normalized_regret = cross_half_regret / spread if spread > 0 else 1.0
        cross_half_gap = (
            float(second.loc[first_best] - second.loc[first_worst])
            + float(first.loc[second_best] - first.loc[second_worst])
        ) / 2
        top_excess = overlap - chance
        reproducible = bool(
            float(full.max() - full.min()) > 0
            and cross_half_gap > 0
            and normalized_regret
            <= float(evaluation["maximum_normalized_cross_half_regret"])
            and top_excess
            >= float(evaluation["minimum_top_overlap_above_chance"])
        )
        rows.append(
            {
                **dict(zip(OBSERVATION_KEYS, key)),
                "system_family": str(group["system_family"].iloc[0]),
                "budget": int(group["budget"].iloc[0]),
                "sets": count,
                "split_half_spearman": _rank_correlation(first, second),
                "top_overlap_fraction": overlap,
                "chance_top_overlap_fraction": chance,
                "top_overlap_above_chance": top_excess,
                "cross_half_gap": cross_half_gap,
                "cross_half_regret": cross_half_regret,
                "normalized_cross_half_regret": normalized_regret,
                "full_mean_spread": float(full.max() - full.min()),
                "reproducible": reproducible,
            }
        )
    return pd.DataFrame(rows)


def _family_summary(
    contexts: pd.DataFrame,
    evaluation: dict[str, Any],
) -> pd.DataFrame:
    def safe_median(values: pd.Series) -> float:
        finite = values.dropna()
        return float(finite.median()) if len(finite) else float("nan")

    rows = []
    for family, group in contexts.groupby("system_family", observed=True, sort=True):
        reproducible = group.loc[group["reproducible"]]
        rows.append(
            {
                "system_family": family,
                "contexts": len(group),
                "reproducible_context_fraction": float(group["reproducible"].mean()),
                "reproducible_anchors": int(
                    reproducible[["dataset_id", "network_id", "anchor_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "median_split_half_spearman": safe_median(
                    group["split_half_spearman"]
                ),
                "median_top_overlap_above_chance": float(
                    group["top_overlap_above_chance"].median()
                ),
                "median_normalized_cross_half_regret": float(
                    group["normalized_cross_half_regret"].median()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["qualifies"] = (
        summary["reproducible_context_fraction"].ge(
            float(evaluation["minimum_reproducible_context_fraction"])
        )
        & summary["reproducible_anchors"].ge(
            int(evaluation["minimum_reproducible_anchors_per_family"])
        )
    )
    return summary


def _convergence_summary(replicates: pd.DataFrame) -> pd.DataFrame:
    final = replicates.groupby(
        OBSERVATION_KEYS + ["set_signature"], observed=True
    )["set_attack_rate_value"].mean().rename("final_mean")
    rows = []
    blocks = sorted(replicates["continuation_block"].unique())
    checkpoints = sorted(set([1, 2, 4, len(blocks)]))
    for count in checkpoints:
        subset = replicates.loc[replicates["continuation_block"].isin(blocks[:count])]
        partial = subset.groupby(
            OBSERVATION_KEYS + ["set_signature", "system_family"], observed=True
        )["set_attack_rate_value"].mean().rename("partial_mean").reset_index()
        merged = partial.merge(final.reset_index(), on=OBSERVATION_KEYS + ["set_signature"])
        merged["absolute_error"] = (merged["partial_mean"] - merged["final_mean"]).abs()
        for family, group in merged.groupby("system_family", observed=True):
            rows.append(
                {
                    "system_family": family,
                    "continuation_blocks": count,
                    "median_absolute_change_from_final": float(group["absolute_error"].median()),
                    "mean_absolute_change_from_final": float(group["absolute_error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot_family_gate(summary: pd.DataFrame, evaluation: dict[str, Any], path: Path) -> None:
    frame = summary.sort_values("reproducible_context_fraction")
    labels = [item.replace("_", " ") for item in frame["system_family"]]
    colors = ["#4C78A8" if value else "#B8B8B8" for value in frame["qualifies"]]
    fig, axis = plt.subplots(figsize=(11, 6))
    bars = axis.barh(labels, frame["reproducible_context_fraction"], color=colors)
    threshold = float(evaluation["minimum_reproducible_context_fraction"])
    axis.axvline(threshold, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlim(0, max(0.5, 1.15 * float(frame["reproducible_context_fraction"].max())))
    axis.set_xlabel("Fraction of contexts with cross-half reproducible set choice")
    axis.grid(axis="x", alpha=0.18)
    for bar, anchors in zip(bars, frame["reproducible_anchors"]):
        x = max(bar.get_width() + 0.01, threshold + 0.02 if bar.get_width() < threshold else 0)
        axis.text(x, bar.get_y() + bar.get_height() / 2, f"{int(anchors)} anchors", va="center")
    fig.suptitle("Conditional set-label precision gate", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.90,
        "Detection-time state and candidate sets are frozen; only future epidemic randomness changes",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.31, right=0.97, top=0.80, bottom=0.14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(contexts: pd.DataFrame, path: Path) -> None:
    families = sorted(contexts["system_family"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    positions = np.arange(len(families))
    data_overlap = [
        contexts.loc[contexts["system_family"].eq(family), "top_overlap_above_chance"]
        for family in families
    ]
    data_regret = [
        contexts.loc[contexts["system_family"].eq(family), "normalized_cross_half_regret"]
        .clip(upper=2)
        for family in families
    ]
    axes[0].boxplot(data_overlap, tick_labels=[item.replace("_", " ") for item in families], showfliers=False)
    axes[1].boxplot(data_regret, tick_labels=[item.replace("_", " ") for item in families], showfliers=False)
    axes[0].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Top-quartile overlap above chance")
    axes[1].set_ylabel("Cross-half regret / within-half spread (clipped at 2)")
    axes[0].set_title("Top-set agreement", fontweight="bold")
    axes[1].set_title("Selection regret", fontweight="bold")
    for axis in axes:
        axis.set_xticks(positions + 1, [item.replace("_", " ") for item in families], rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Independent continuation-block reliability", fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.28, wspace=0.30)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(summary: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 6))
    for family, group in summary.groupby("system_family", observed=True, sort=True):
        axis.plot(
            group["continuation_blocks"],
            100 * group["mean_absolute_change_from_final"],
            marker="o",
            label=family.replace("_", " "),
        )
    axis.set_xlabel("Post-detection continuation blocks used")
    axis.set_ylabel("Mean absolute label change from final estimate (percentage points)")
    axis.set_title("Conditional set-value Monte Carlo convergence", fontsize=18, fontweight="bold")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    config_path: Path,
    profile_name: str,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    compute_only: bool = False,
) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][profile_name]
    coverage_config_path = Path(config["data"]["coverage_config"])
    coverage_config = yaml.safe_load(coverage_config_path.read_text(encoding="utf-8"))
    source_path = Path(config["data"]["coverage_values"])
    source_values = _rename_observation_columns(
        pd.read_csv(source_path, dtype={"network_id": str, "initial_infected": str})
    )
    source_values["anchor_time"] = pd.to_datetime(source_values["anchor_time"], format="mixed")
    source_values = _select_contexts(source_values, profile)
    if source_values.empty:
        raise ValueError("no multi-node source contexts were selected")

    stable_path = Path(config["data"]["stable_prediction_path"])
    predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    predictions["anchor_time"] = pd.to_datetime(predictions["anchor_time"], format="mixed")
    resources: dict[tuple[str, str, str], tuple[dict[str, Any], Any]] = {}
    parameter_lookup: dict[tuple[str, str], Any] = {}
    for dataset_id in profile["datasets"]:
        if dataset_id not in set(source_values["dataset_id"]):
            continue
        specification = coverage_config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network = str(specification.get("network_id", "all"))
        for window in windows:
            window.setdefault("network_id", default_network)
            resources[(dataset_id, str(window["network_id"]), window["anchor"].anchor_id)] = (
                window,
                specification,
            )
        parameters = _selected_parameters(
            Path(specification["source_results"]) / "parameter_selection.csv", None
        )
        for parameter in parameters.itertuples(index=False):
            parameter_lookup[(dataset_id, str(parameter.parameter_id))] = parameter

    experiment_id = str(config["experiment"]["id"])
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    fingerprint = hashlib.sha256(
        config_path.read_bytes() + source_path.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()[:12]
    checkpoint_dir = results_dir / "checkpoints" / fingerprint
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be within [0, shard_count)")

    replicate_frames: list[pd.DataFrame] = []
    member_frames: list[pd.DataFrame] = []
    all_grouped = list(source_values.groupby(OBSERVATION_KEYS, observed=True, sort=True))
    grouped = [
        item for index, item in enumerate(all_grouped) if index % shard_count == shard_index
    ]
    progress = tqdm(
        grouped,
        desc=f"Conditional precision {profile_name} shard {shard_index + 1}/{shard_count}",
        unit="context",
    )
    for key, candidates in progress:
        identity = fingerprint + "|" + "|".join(map(str, key))
        token = hashlib.sha256(identity.encode()).hexdigest()[:20]
        value_checkpoint = checkpoint_dir / f"{token}_values.csv.gz"
        member_checkpoint = checkpoint_dir / f"{token}_members.csv.gz"
        if bool(config["execution"].get("resume", True)) and value_checkpoint.exists() and member_checkpoint.exists():
            replicate_frames.append(pd.read_csv(value_checkpoint, dtype={"network_id": str, "initial_infected": str}))
            member_frames.append(pd.read_csv(member_checkpoint, dtype={"network_id": str, "initial_infected": str, "candidate_id": str}))
            progress.set_postfix_str(f"{key[0]} cached")
            continue
        first = candidates.iloc[0]
        resource_key = (str(first.dataset_id), str(first.network_id), str(first.anchor_id))
        if resource_key not in resources:
            raise ValueError(f"missing source window for {resource_key}")
        window, specification = resources[resource_key]
        parameter = parameter_lookup[(str(first.dataset_id), str(first.parameter_id))]
        anchor = window["anchor"]
        stream = window["future"]
        mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
        detection_time = anchor.anchor_time + mean_period * float(first.detection_delay_fraction)
        parameters = SIRParameters(
            beta=float(parameter.beta), recovery_rate=float(parameter.recovery_rate_per_second)
        )
        engine = PairedTemporalSIREngine()
        natural = engine.simulate(
            stream,
            parameters,
            initial_infected=(str(first.initial_infected),),
            start_time=anchor.anchor_time,
            end_time=anchor.horizon_end,
            world_seed=int(first.observation_world_seed),
        )
        decision_states = states_at(natural, detection_time)
        detected = observe_detected_cases(
            decision_states,
            trigger_node=str(first.initial_infected),
            secondary_case_sensitivity=float(first.secondary_case_sensitivity),
            world_seed=int(first.observation_world_seed),
        )
        if detected != _nodes(first.detected_nodes):
            raise AssertionError("replayed detected cases differ from the frozen context")

        stable = _matching_stable_scores(
            predictions,
            str(first.dataset_id),
            str(first.network_id),
            anchor.anchor_time,
            window["eligible"],
        )
        scores = pre_detection_scores(
            stream,
            detected_nodes=detected,
            start_time=anchor.anchor_time,
            detection_time=detection_time,
            half_life=mean_period
            * float(coverage_config["decision"]["tracing_half_life_fraction_of_mean_infectious_period"]),
        )
        scores["candidate_id"] = scores["candidate_id"].astype(str)
        stable["candidate_id"] = stable["candidate_id"].astype(str)
        scores = scores.loc[scores["candidate_id"].isin(window["eligible"])].merge(
            stable, on="candidate_id", how="left", validate="one_to_one"
        )
        remaining = scores.loc[~scores["candidate_id"].isin(detected)].copy()
        remaining["stable_percentile"] = _percentile(remaining["stable_score"])
        remaining["activity_percentile"] = _percentile(remaining["current_activity"])
        remaining["tracing_percentile"] = _percentile(remaining["contact_to_detected"])
        remaining = remaining.set_index("candidate_id", drop=False)

        member_rows = []
        for candidate in candidates.itertuples(index=False):
            selected = _nodes(candidate.selected_nodes)
            computed = _set_features(remaining.reset_index(drop=True), selected)
            if not np.allclose(
                [computed[column] for column in FEATURE_COLUMNS if column.startswith("set_")],
                [float(getattr(candidate, column)) for column in FEATURE_COLUMNS if column.startswith("set_")],
            ):
                raise AssertionError("replayed set features differ from the frozen context")
            for node in selected:
                row = remaining.loc[node]
                member_rows.append(
                    {
                        **dict(zip(OBSERVATION_KEYS, key)),
                        "system_family": str(first.system_family),
                        "set_signature": str(candidate.set_signature),
                        "candidate_id": node,
                        "stable_percentile": float(row.stable_percentile),
                        "tracing_percentile": float(row.tracing_percentile),
                        "activity_percentile": float(row.activity_percentile),
                        "has_positive_tracing": float(row.contact_to_detected > 0),
                        "stable_tracing_product": float(row.stable_percentile * row.tracing_percentile),
                    }
                )
        members = pd.DataFrame(member_rows)

        current_infected = tuple(sorted(node for node, state in decision_states.items() if state == "I"))
        current_recovered = tuple(sorted(node for node, state in decision_states.items() if state == "R"))
        rows = []
        for continuation_block in range(int(profile["continuation_blocks"])):
            continuation_seed = _keyed_seed(
                int(config["evaluation"]["seed"]), *key, "post_detection", continuation_block
            )
            if current_infected:
                standard = engine.simulate(
                    stream,
                    parameters,
                    initial_infected=current_infected,
                    initial_recovered=current_recovered,
                    start_time=detection_time,
                    end_time=anchor.horizon_end,
                    world_seed=continuation_seed,
                    action=_isolation_action(
                        "conditional_standard_care", detected, detection_time, anchor.horizon_end
                    ),
                )
                standard_final_size = standard.final_size
            else:
                standard_final_size = len(current_recovered)
            for candidate in candidates.itertuples(index=False):
                selected = _nodes(candidate.selected_nodes)
                if current_infected:
                    augmented = engine.simulate(
                        stream,
                        parameters,
                        initial_infected=current_infected,
                        initial_recovered=current_recovered,
                        start_time=detection_time,
                        end_time=anchor.horizon_end,
                        world_seed=continuation_seed,
                        action=_isolation_action(
                            "conditional_standard_plus_set",
                            tuple(sorted(set(detected) | set(selected))),
                            detection_time,
                            anchor.horizon_end,
                        ),
                    )
                    set_final_size = augmented.final_size
                else:
                    set_final_size = len(current_recovered)
                rows.append(
                    {
                        **dict(zip(OBSERVATION_KEYS, key)),
                        "system_family": str(first.system_family),
                        "anchor_time": anchor.anchor_time,
                        "detection_time": detection_time,
                        "continuation_block": continuation_block,
                        "continuation_seed": continuation_seed,
                        "population_size": int(first.population_size),
                        "budget": int(first.budget),
                        "detected_nodes": "|".join(detected),
                        "infectious_at_detection": len(current_infected),
                        "recovered_at_detection": len(current_recovered),
                        "source_methods": str(candidate.source_methods),
                        "selected_nodes": str(candidate.selected_nodes),
                        "set_signature": str(candidate.set_signature),
                        "standard_final_size": standard_final_size,
                        "set_final_size": set_final_size,
                        "set_attack_rate_value": (standard_final_size - set_final_size)
                        / int(first.population_size),
                        **{column: float(getattr(candidate, column)) for column in FEATURE_COLUMNS},
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(value_checkpoint, index=False, compression="gzip")
        members.to_csv(member_checkpoint, index=False, compression="gzip")
        replicate_frames.append(frame)
        member_frames.append(members)
        progress.set_postfix_str(f"{first.dataset_id} complete")

    if compute_only:
        return results_dir, report_dir

    if shard_count != 1:
        raise ValueError("aggregation must run without sharding")

    replicates = pd.concat(replicate_frames, ignore_index=True)
    members = pd.concat(member_frames, ignore_index=True)
    aggregate_keys = OBSERVATION_KEYS + ["set_signature"]
    aggregate_feature_columns = [
        column for column in FEATURE_COLUMNS if column not in {"population_size", "budget"}
    ]
    labels = replicates.groupby(aggregate_keys, observed=True, sort=True).agg(
        system_family=("system_family", "first"),
        anchor_time=("anchor_time", "first"),
        population_size=("population_size", "first"),
        budget=("budget", "first"),
        source_methods=("source_methods", "first"),
        selected_nodes=("selected_nodes", "first"),
        repetitions=("continuation_block", "nunique"),
        mean_set_value=("set_attack_rate_value", "mean"),
        set_value_std=("set_attack_rate_value", "std"),
        **{column: (column, "first") for column in aggregate_feature_columns},
    ).reset_index()
    labels["set_value_se"] = labels["set_value_std"].fillna(0) / np.sqrt(labels["repetitions"])
    labels["set_value_ci_low"] = labels["mean_set_value"] - 1.96 * labels["set_value_se"]
    labels["set_value_ci_high"] = labels["mean_set_value"] + 1.96 * labels["set_value_se"]
    context_reliability = _context_reliability(replicates, config["evaluation"])
    family_summary = _family_summary(context_reliability, config["evaluation"])
    convergence = _convergence_summary(replicates)
    family_equal_fraction = float(family_summary["reproducible_context_fraction"].mean())
    qualifying_families = int(family_summary["qualifies"].sum())
    scientific_pass = bool(
        qualifying_families >= int(config["evaluation"]["minimum_qualifying_families"])
        and family_equal_fraction
        >= float(config["evaluation"]["minimum_family_equal_reproducible_fraction"])
    )
    gate_status = (
        "pass" if scientific_pass else "fail"
    ) if bool(profile.get("enforce_scientific_gate", False)) else "not_evaluated"

    context_block_counts = replicates.groupby(OBSERVATION_KEYS + ["set_signature"], observed=True)[
        "continuation_block"
    ].nunique()
    standard_counts = replicates.groupby(
        OBSERVATION_KEYS + ["continuation_block"], observed=True
    )["standard_final_size"].nunique()
    checks = {
        "source_contexts_are_multinode": bool(replicates["budget"].gt(1).all()),
        "replicate_keys_unique": not replicates.duplicated(
            OBSERVATION_KEYS + ["continuation_block", "set_signature"]
        ).any(),
        "all_sets_have_configured_blocks": bool(
            context_block_counts.eq(int(profile["continuation_blocks"])).all()
        ),
        "standard_care_shared_within_continuation": bool(standard_counts.eq(1).all()),
        "paired_arithmetic": bool(
            np.allclose(
                replicates["set_attack_rate_value"],
                (replicates["standard_final_size"] - replicates["set_final_size"])
                / replicates["population_size"],
            )
        ),
        "frozen_set_membership_across_continuations": bool(
            replicates.groupby(OBSERVATION_KEYS + ["set_signature"], observed=True)[
                "selected_nodes"
            ].nunique().eq(1).all()
        ),
        "member_rows_unique": not members.duplicated(
            OBSERVATION_KEYS + ["set_signature", "candidate_id"]
        ).any(),
        "finite_outputs": bool(
            np.isfinite(
                replicates[FEATURE_COLUMNS + ["set_attack_rate_value"]].to_numpy(float)
            ).all()
        ),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scientific_gate": {
            "status": gate_status,
            "qualifying_families": qualifying_families,
            "minimum_qualifying_families": int(config["evaluation"]["minimum_qualifying_families"]),
            "family_equal_reproducible_context_fraction": family_equal_fraction,
            "minimum_family_equal_reproducible_fraction": float(
                config["evaluation"]["minimum_family_equal_reproducible_fraction"]
            ),
        },
        "observation_contexts": int(context_reliability.shape[0]),
        "candidate_sets": int(labels.shape[0]),
        "conditional_replicates": int(replicates.shape[0]),
        "families": int(family_summary.shape[0]),
    }
    if audit["status"] != "pass":
        raise ValueError(f"conditional precision artifact audit failed: {audit}")

    replicates.to_csv(results_dir / "conditional_set_replicates.csv.gz", index=False, compression="gzip")
    labels.to_csv(results_dir / "conditional_set_labels.csv.gz", index=False, compression="gzip")
    members.to_csv(results_dir / "set_member_features.csv.gz", index=False, compression="gzip")
    context_reliability.to_csv(results_dir / "context_reliability.csv", index=False)
    family_summary.to_csv(results_dir / "family_precision_summary.csv", index=False)
    convergence.to_csv(results_dir / "convergence_summary.csv", index=False)
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
        "source_values": str(source_path),
        "source_values_sha256": _sha256(source_path),
        "artifact_audit": audit["status"],
        "scientific_gate": gate_status,
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _plot_family_gate(family_summary, config["evaluation"], report_dir / "precision_gate.png")
    _plot_reliability(context_reliability, report_dir / "split_half_reliability.png")
    _plot_convergence(convergence, report_dir / "monte_carlo_convergence.png")
    lines = "\n".join(
        f"- {row.system_family}: {row.reproducible_context_fraction:.1%} reproducible contexts, "
        f"{int(row.reproducible_anchors)} anchors, qualifies={bool(row.qualifies)}."
        for row in family_summary.itertuples(index=False)
    )
    (report_dir / "README.md").write_text(
        "# Conditional set-label precision and repeatability\n\n"
        f"Profile: **{profile_name}**. Frozen observation contexts: {len(context_reliability):,}; "
        f"candidate sets: {len(labels):,}; conditional continuations: {len(replicates):,}. "
        f"Artifact audit: **{audit['status']}**. Scientific gate: **{gate_status}**.\n\n"
        f"Family-equal reproducible-context fraction: {family_equal_fraction:.1%}; "
        f"qualifying families: {qualifying_families}.\n\n{lines}\n\n"
        "This experiment freezes the detection-time state, observed evidence, and candidate sets. "
        "Only post-detection epidemic randomness is resampled. It does not train a planner.\n",
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run conditional set-label precision and repeatability audit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260816-013_set_value_precision.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--compute-only", action="store_true")
    args = parser.parse_args()
    results, reports = run(
        args.config,
        args.profile,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        compute_only=args.compute_only,
    )
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
