from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import slice_stream
from animal_intervention.evaluation import stable_hash_order
from animal_intervention.simulation import (
    DetectionProfile,
    InterventionAction,
    PairedTemporalSIREngine,
    SIRParameters,
    detection_time_from_seed,
    pre_detection_event_signature,
    pre_detection_scores,
    select_additional_targets,
    states_at,
)
from animal_intervention.transmission.mappers import (
    CoalescedDurationContactMapper,
    compile_primary_exposure,
)

from .experimental_songbirds_validation import (
    _observed_group_stream as _observed_songbird_stream,
    _prepare_windows as _prepare_songbird_windows,
)
from .additional_system_validation import (
    _bat_stream,
    _bat_windows,
    _sheep_stream as _free_sheep_stream,
    _sheep_windows as _free_sheep_windows,
)
from .oxford_predefense import _keyed_seed, _prepare_windows as _prepare_dyadic_windows
from .radolfzell_validation import (
    _observed_group_stream as _observed_radolfzell_stream,
    _prepare_windows as _prepare_radolfzell_windows,
)
from .sheep_validation import _load_phase_by_date, _prepare_network_windows
from .wytham_validation import (
    _host_group_stream,
    _prepare_windows as _prepare_wytham_windows,
)


METHOD_LABELS = {
    "random": "Random",
    "stable_watchlist": "Stable watchlist",
    "history_weight": "History weight",
    "history_recency": "History recency",
    "current_activity": "Current activity",
    "contact_to_detected": "Contact to detected",
    "stable_plus_tracing": "Stable + tracing",
    "perfect_state_diagnostic": "Perfect-state diagnostic",
}
METHOD_COLORS = {
    "random": "#B8B8B8",
    "stable_watchlist": "#4C78A8",
    "history_weight": "#59A14F",
    "history_recency": "#B279A2",
    "current_activity": "#72B7B2",
    "contact_to_detected": "#F58518",
    "stable_plus_tracing": "#D95F02",
    "perfect_state_diagnostic": "#555555",
}
DATASET_LABELS = {
    "oxford_wildbird_network": "Oxford wild birds",
    "guinea_baboons_sociopatterns": "Guinea baboons",
    "domestic_sheep_sirtrack": "Domestic sheep",
    "wytham_great_tits_divorce": "Wytham great tits",
    "radolfzell_great_tits_ontogeny": "Radolfzell great tits",
    "experimental_wild_songbirds": "Experimental wild songbirds",
    "wild_vampire_bats_proximity": "Wild vampire bats",
    "free_ranging_sheep_fission_fusion": "Free-ranging sheep",
}
COMPARISON_LABELS = {
    "stable_over_random": "Stable watchlist over random",
    "history_weight_over_stable": "History weight over stable",
    "history_recency_over_stable": "History recency over stable",
    "current_activity_over_stable": "Current activity over stable",
    "contact_to_detected_over_stable": "Contact to detected over stable",
    "stable_plus_tracing_over_stable": "Stable + tracing over stable",
    "perfect_state_diagnostic_over_stable": "Perfect-state diagnostic over stable",
}
WORLD_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "detection_profile",
    "random_block",
    "initial_infected",
    "world_seed",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _load_source_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_windows(
    dataset_id: str,
    source_config: dict[str, Any],
) -> list[dict[str, Any]]:
    def attach_history(prepared: list[dict[str, Any]], stream: Any) -> list[dict[str, Any]]:
        for item in prepared:
            anchor = item["anchor"]
            item["history"] = slice_stream(
                stream,
                pd.Timestamp(anchor.history_start),
                pd.Timestamp(anchor.anchor_time),
            )
        return prepared

    dataset = CanonicalDataset.read(Path(source_config["data"]["canonical_path"]))
    if dataset_id == "experimental_wild_songbirds":
        stream = _observed_songbird_stream(dataset, source_config)
        prepared, _ = _prepare_songbird_windows(
            stream,
            source_config["windows"],
            int(source_config["windows"]["max_anchors"]),
        )
        return attach_history(prepared, stream)
    if dataset_id == "domestic_sheep_sirtrack":
        stream = CoalescedDurationContactMapper().compile(dataset)
        phase_by_date = _load_phase_by_date(
            Path(source_config["data"]["raw_measurements_path"])
        )
        prepared, _ = _prepare_network_windows(
            stream,
            dataset,
            source_config["windows"],
            int(source_config["windows"]["max_anchors"]),
            phase_by_date,
        )
        return attach_history(prepared, stream)
    if dataset_id == "wytham_great_tits_divorce":
        stream = _host_group_stream(
            dataset,
            str(source_config["data"]["host_species_code"]),
        )
        prepared, _ = _prepare_wytham_windows(
            stream,
            source_config["windows"],
            int(source_config["windows"]["max_anchors"]),
        )
        return attach_history(prepared, stream)
    if dataset_id == "radolfzell_great_tits_ontogeny":
        stream = _observed_radolfzell_stream(dataset, source_config)
        prepared, _ = _prepare_radolfzell_windows(
            stream,
            source_config["windows"],
            int(source_config["windows"]["max_anchors"]),
        )
        return attach_history(prepared, stream)
    if dataset_id == "wild_vampire_bats_proximity":
        stream = _bat_stream(dataset, source_config)
        prepared, _ = _bat_windows(
            stream,
            source_config["windows"],
            int(source_config["windows"]["max_anchors"]),
        )
        return attach_history(prepared, stream)
    if dataset_id == "free_ranging_sheep_fission_fusion":
        stream = _free_sheep_stream(dataset, source_config)
        prepared, _ = _free_sheep_windows(
            stream,
            source_config["windows"],
            int(source_config["windows"]["max_anchors"]),
        )
        return attach_history(prepared, stream)
    stream = compile_primary_exposure(dataset)
    prepared = _prepare_dyadic_windows(
        stream,
        source_config["windows"],
        int(source_config["windows"]["max_anchors"]),
    )
    default_network_id = str(source_config["data"].get("network_id", "all"))
    for window in prepared:
        window.setdefault("network_id", default_network_id)
    return attach_history(prepared, stream)


def _selected_parameters(path: Path, limit: int | None) -> pd.DataFrame:
    selected = pd.read_csv(path)
    flag = selected["selected"]
    if flag.dtype == bool:
        keep = flag
    else:
        keep = flag.astype(str).str.strip().str.lower().eq("true")
    selected = selected.loc[keep].copy()
    selected = selected.sort_values("mean_attack_rate", kind="stable").reset_index(drop=True)
    if limit is not None and len(selected) > limit:
        indices = np.linspace(0, len(selected) - 1, num=limit).round().astype(int)
        selected = selected.iloc[np.unique(indices)].copy()
    selected["recovery_rate_per_second"] = (
        1.0 / selected["mean_infectious_period_days"] / 86_400
    )
    return selected.reset_index(drop=True)


def _matching_stable_scores(
    predictions: pd.DataFrame,
    dataset_id: str,
    network_id: str,
    anchor_time: pd.Timestamp,
    eligible: list[str],
) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["dataset_id"].eq(dataset_id)
        & predictions["network_id"].astype(str).eq(str(network_id))
        & predictions["anchor_time"].eq(pd.Timestamp(anchor_time)),
        [
            "candidate_id",
            "score_stable_watchlist",
            "weighted_exposure_rate",
            "recency_score",
        ],
    ].copy()
    selected = selected.rename(
        columns={
            "score_stable_watchlist": "stable_score",
            "weighted_exposure_rate": "history_weight",
            "recency_score": "history_recency",
        }
    )
    expected = set(map(str, eligible))
    if set(selected["candidate_id"].astype(str)) != expected:
        missing = expected - set(selected["candidate_id"].astype(str))
        raise ValueError(
            f"stable-score candidate mismatch for {dataset_id} at {anchor_time}: {len(missing)} missing"
        )
    return selected


def _isolation_action(
    name: str,
    targets: tuple[str, ...],
    detection_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> InterventionAction:
    return InterventionAction(
        name=name,
        action_type="isolation",
        target_nodes=tuple(sorted(set(targets))),
        start_time=detection_time,
        end_time=end_time,
        contact_multiplier=0.0,
    )


def _post_detection_infections(result: Any, detection_time: pd.Timestamp) -> int:
    events = result.event_log
    return int(
        events["event"].isin(["initial_infection", "infection"])
        .loc[pd.to_datetime(events["time"]).ge(detection_time)]
        .sum()
    )


def _run_task(
    *,
    dataset_id: str,
    network_id: str,
    system_family: str,
    analysis_cluster_id: str,
    window: dict[str, Any],
    parameter: Any,
    profile: DetectionProfile,
    stable_scores: pd.DataFrame,
    methods: list[str],
    seed_nodes: list[str],
    random_blocks: int,
    budget_fraction: float,
    minimum_budget: int,
    tracing_half_life_fraction: float,
    experiment_seed: int,
) -> pd.DataFrame:
    anchor = window["anchor"]
    stream = window["future"]
    eligible = list(map(str, window["eligible"]))
    population_size = len(stream.nodes())
    budget = max(minimum_budget, int(math.ceil((len(eligible) - 1) * budget_fraction)))
    sir = SIRParameters(
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
        profile,
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
                sir,
                initial_infected=(initial,),
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
            )
            detected = (str(initial),)
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
                missing_candidates = score_table.loc[
                    score_table["stable_score"].isna(), "candidate_id"
                ].astype(str).head(5).tolist()
                raise ValueError(
                    "missing stable scores in response task for "
                    f"{dataset_id}/{network_id}/{anchor.anchor_id}: "
                    f"{missing_candidates}"
                )
            decision_states = states_at(natural, detection_time)
            score_table["infected_at_detection"] = score_table["candidate_id"].map(
                decision_states
            ).eq("I")
            standard = engine.simulate(
                stream,
                sir,
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
            standard_signature = pre_detection_event_signature(standard, detection_time)
            if natural_signature != standard_signature:
                raise AssertionError("standard care diverged before detection")
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
                    sir,
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
                standard_post = _post_detection_infections(standard, detection_time)
                augmented_post = _post_detection_infections(augmented, detection_time)
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
                        "detection_profile": profile.name,
                        "detection_time": detection_time,
                        "detection_delay_hours": (detection_time - anchor.anchor_time).total_seconds() / 3600,
                        "random_block": block,
                        "initial_infected": str(initial),
                        "world_seed": world_seed,
                        "population_size": population_size,
                        "additional_budget": budget,
                        "method": method,
                        "additional_targets": "|".join(targets),
                        "infectious_animals_at_detection": int(
                            sum(state == "I" for state in decision_states.values())
                        ),
                        "selected_infected_at_detection": int(
                            selected["infected_at_detection"].sum()
                        ),
                        "selected_infected_fraction": float(
                            selected["infected_at_detection"].mean()
                        ),
                        "mean_selected_trace_score": float(
                            selected["contact_to_detected"].mean()
                        ),
                        "natural_final_size": natural.final_size,
                        "standard_final_size": standard.final_size,
                        "augmented_final_size": augmented.final_size,
                        "standard_post_detection_infections": standard_post,
                        "augmented_post_detection_infections": augmented_post,
                        "avoided_infections": standard.final_size - augmented.final_size,
                        "avoided_post_detection_infections": standard_post - augmented_post,
                        "attack_rate_reduction": (
                            standard.final_size - augmented.final_size
                        ) / population_size,
                        "standard_peak_infectious": standard.peak_infectious,
                        "augmented_peak_infectious": augmented.peak_infectious,
                    }
                )
    return pd.DataFrame(rows)


def _cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    cluster_means = frame.groupby("analysis_cluster_id", observed=True)[value].mean().to_numpy(float)
    point = float(cluster_means.mean())
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, len(cluster_means), size=len(cluster_means))
        draws[index] = float(cluster_means[sampled].mean())
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summarize_policies(
    worlds: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for (dataset_id, profile, method), frame in worlds.groupby(
        ["dataset_id", "detection_profile", "method"], sort=True, observed=True
    ):
        mean, low, high = _cluster_bootstrap_mean(
            frame,
            "attack_rate_reduction",
            replicates=bootstrap_replicates,
            seed=_keyed_seed(seed, dataset_id, profile, method),
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "detection_profile": profile,
                "method": method,
                "mean_attack_rate_reduction": mean,
                "blocked_ci_low": low,
                "blocked_ci_high": high,
                "mean_avoided_infections": float(frame["avoided_infections"].mean()),
                "positive_world_fraction": float(frame["avoided_infections"].gt(0).mean()),
                "negative_world_fraction": float(frame["avoided_infections"].lt(0).mean()),
                "mean_selected_infected_fraction": float(
                    frame["selected_infected_fraction"].mean()
                ),
                "anchors": frame["anchor_id"].nunique(),
                "worlds": frame[WORLD_KEYS].drop_duplicates().shape[0],
            }
        )
    return pd.DataFrame(rows)


def _summarize_increments(
    worlds: pd.DataFrame,
    *,
    baseline: str,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = worlds.loc[worlds["method"].eq(baseline), WORLD_KEYS + ["attack_rate_reduction"]]
    rows = []
    paired_parts = []
    for method in sorted(set(worlds["method"]) - {baseline}):
        challenger = worlds.loc[
            worlds["method"].eq(method),
            WORLD_KEYS
            + ["system_family", "analysis_cluster_id", "attack_rate_reduction"],
        ]
        paired = challenger.merge(reference, on=WORLD_KEYS, suffixes=("_method", "_baseline"))
        paired["method"] = method
        paired["baseline"] = baseline
        paired["increment"] = (
            paired["attack_rate_reduction_method"]
            - paired["attack_rate_reduction_baseline"]
        )
        paired_parts.append(paired)
        for (dataset_id, profile), frame in paired.groupby(
            ["dataset_id", "detection_profile"], sort=True, observed=True
        ):
            mean, low, high = _cluster_bootstrap_mean(
                frame,
                "increment",
                replicates=bootstrap_replicates,
                seed=_keyed_seed(seed, dataset_id, profile, method, baseline),
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "detection_profile": profile,
                    "method": method,
                    "baseline": baseline,
                    "mean_increment": mean,
                    "blocked_ci_low": low,
                    "blocked_ci_high": high,
                    "positive_anchor_fraction": float(
                        frame.groupby("anchor_id", observed=True)["increment"].mean().gt(0).mean()
                    ),
                    "anchors": frame["anchor_id"].nunique(),
                    "paired_worlds": len(frame),
                }
            )
    return pd.concat(paired_parts, ignore_index=True), pd.DataFrame(rows)


def _family_equal_increment_summary(
    paired: pd.DataFrame,
    *,
    baseline: str,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = paired.copy()
    comparisons["comparison"] = comparisons["method"] + f"_over_{baseline}"
    random_mask = comparisons["method"].eq("random") & (baseline == "stable_watchlist")
    comparisons.loc[random_mask, "comparison"] = "stable_over_random"
    comparisons["difference"] = comparisons["increment"]
    comparisons.loc[random_mask, "difference"] *= -1
    rows: list[dict[str, Any]] = []
    for (profile, comparison), frame in comparisons.groupby(
        ["detection_profile", "comparison"], sort=True, observed=True
    ):
        rng = np.random.default_rng(_keyed_seed(seed, profile, comparison, "family"))
        family_names = sorted(frame["system_family"].unique())
        family_units: dict[str, list[np.ndarray]] = {}
        family_means: list[float] = []
        for family in family_names:
            family_frame = frame.loc[frame["system_family"].eq(family)]
            unit_arrays = [
                unit.groupby("anchor_id", observed=True)["difference"]
                .mean()
                .to_numpy(float)
                for _, unit in family_frame.groupby(
                    ["dataset_id", "network_id"], sort=True, observed=True
                )
            ]
            family_units[family] = unit_arrays
            family_means.append(float(np.mean([values.mean() for values in unit_arrays])))
        draws = np.empty(bootstrap_replicates, dtype=float)
        for draw_index in range(bootstrap_replicates):
            sampled_families = rng.choice(
                family_names, size=len(family_names), replace=True
            )
            sampled_family_means = []
            for family in sampled_families:
                units = family_units[str(family)]
                sampled_units = rng.integers(0, len(units), size=len(units))
                unit_means = []
                for unit_index in sampled_units:
                    values = units[int(unit_index)]
                    sampled_contexts = rng.integers(0, len(values), size=len(values))
                    unit_means.append(float(values[sampled_contexts].mean()))
                sampled_family_means.append(float(np.mean(unit_means)))
            draws[draw_index] = float(np.mean(sampled_family_means))
        rows.append(
            {
                "detection_profile": profile,
                "comparison": comparison,
                "family_equal_mean": float(np.mean(family_means)),
                "blocked_ci_low": float(np.quantile(draws, 0.025)),
                "blocked_ci_high": float(np.quantile(draws, 0.975)),
                "positive_family_fraction": float(
                    np.mean(np.asarray(family_means) > 0)
                ),
                "families": len(family_names),
                "contexts": frame[
                    ["dataset_id", "network_id", "anchor_id"]
                ].drop_duplicates().shape[0],
            }
        )
    return pd.DataFrame(rows)


def _standard_care_summary(worlds: pd.DataFrame) -> pd.DataFrame:
    base = worlds.drop_duplicates(WORLD_KEYS).copy()
    base["standard_avoided_infections"] = (
        base["natural_final_size"] - base["standard_final_size"]
    )
    base["standard_attack_rate_reduction"] = (
        base["standard_avoided_infections"] / base["population_size"]
    )
    return (
        base.groupby(["dataset_id", "detection_profile"], observed=True, sort=True)
        .agg(
            mean_natural_final_size=("natural_final_size", "mean"),
            mean_standard_final_size=("standard_final_size", "mean"),
            mean_standard_avoided_infections=("standard_avoided_infections", "mean"),
            mean_standard_attack_rate_reduction=(
                "standard_attack_rate_reduction",
                "mean",
            ),
            positive_world_fraction=(
                "standard_avoided_infections",
                lambda values: float(values.gt(0).mean()),
            ),
            worlds=("world_seed", "size"),
        )
        .reset_index()
    )


def _plot_policy_effects(summary: pd.DataFrame, path: Path) -> None:
    frame = summary.loc[summary["method"].eq("stable_watchlist")].copy()
    frame["label"] = frame["dataset_id"].map(DATASET_LABELS).fillna(frame["dataset_id"])
    frame["label"] += " | " + frame["detection_profile"].str.replace("_", " ")
    frame = frame.sort_values(["dataset_id", "detection_profile"], kind="stable")
    y = np.arange(len(frame))
    values = 100 * frame["mean_attack_rate_reduction"].to_numpy(float)
    low = 100 * frame["blocked_ci_low"].to_numpy(float)
    high = 100 * frame["blocked_ci_high"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(11, max(6.0, 0.55 * len(frame) + 2.0)))
    axis.errorbar(
        values,
        y,
        xerr=[values - low, high - values],
        fmt="o",
        color=METHOD_COLORS["stable_watchlist"],
        ecolor="#9ECAE1",
        capsize=3,
    )
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(y, frame["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Additional avoided attack rate after standard care (percentage points)")
    fig.suptitle(
        "Stable-watchlist benefit by animal system and detection timing",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        "Points are blocked means; intervals resample independent sheep groups or temporal anchors in single-population datasets",
        ha="center",
        color="#555555",
    )
    axis.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_increment(summary: pd.DataFrame, path: Path) -> None:
    preferred_profiles = ["early_detection", "delayed_detection"]
    profiles = [item for item in preferred_profiles if item in set(summary["detection_profile"])]
    profiles.extend(sorted(set(summary["detection_profile"]) - set(profiles)))
    comparisons = list(dict.fromkeys(summary["comparison"].tolist()))
    fig, axes = plt.subplots(
        1,
        len(profiles),
        figsize=(max(9.5, 7.2 * len(profiles)), max(5.5, 0.55 * len(comparisons) + 2.6)),
        sharey=True,
    )
    for axis, profile in zip(np.atleast_1d(axes), profiles):
        frame = (
            summary.loc[summary["detection_profile"].eq(profile)]
            .set_index("comparison")
            .reindex(comparisons)
        )
        y = np.arange(len(frame))
        values = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["blocked_ci_low"].to_numpy(float)
        high = 100 * frame["blocked_ci_high"].to_numpy(float)
        axis.errorbar(
            values,
            y,
            xerr=[values - low, high - values],
            fmt="o",
            color="#D95F02",
            ecolor="#F6B27A",
            capsize=3,
        )
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_yticks(
            y,
            [COMPARISON_LABELS.get(item, item.replace("_", " ").title()) for item in comparisons],
        )
        axis.invert_yaxis()
        axis.set_title(profile.replace("_", " ").title(), fontweight="bold")
        axis.grid(axis="x", alpha=0.18)
        axis.set_xlabel("Family-equal attack-rate difference (percentage points)")
    fig.suptitle("Cross-system response-policy increments", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.93,
        "Positive values favor the first policy named; hierarchical bootstrap respects family, network, and anchor blocks",
        ha="center",
        color="#555555",
    )
    fig.subplots_adjust(left=0.31, right=0.98, top=0.86, bottom=0.13, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _closest_prior_gate(
    worlds: pd.DataFrame,
    *,
    method: str,
    baselines: list[str],
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for baseline in baselines:
        paired, _ = _summarize_increments(
            worlds,
            baseline=baseline,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        summary = _family_equal_increment_summary(
            paired,
            baseline=baseline,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        comparison = f"{method}_over_{baseline}"
        selected = summary.loc[summary["comparison"].eq(comparison)]
        if len(selected) != 1:
            raise ValueError(f"missing closest-prior comparison: {comparison}")
        rows.append(selected.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _plot_closest_prior_gate(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["comparison"].str.replace("_", " ").str.title()
    values = 100 * frame["family_equal_mean"].to_numpy(float)
    low = 100 * frame["blocked_ci_low"].to_numpy(float)
    high = 100 * frame["blocked_ci_high"].to_numpy(float)
    y = np.arange(len(frame))
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.errorbar(
        values,
        y,
        xerr=[values - low, high - values],
        fmt="o",
        color="#4C78A8",
        ecolor="#9ECAE1",
        capsize=4,
        markersize=8,
    )
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Family-equal attack-rate difference (percentage points)")
    axis.set_title("Frozen watchlist versus closest history-only comparators", fontsize=17, fontweight="bold", pad=18)
    axis.text(
        0.5,
        1.01,
        "Positive favors the frozen stable watchlist; intervals preserve independent animal-system blocks",
        transform=axis.transAxes,
        ha="center",
        color="#555555",
    )
    axis.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.82, bottom=0.16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_target_yield(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[summary["method"].ne("random")].copy()
    selected["panel"] = selected["dataset_id"].map(DATASET_LABELS).fillna(selected["dataset_id"])
    selected["panel"] += "\n" + selected["detection_profile"].str.replace("_", " ")
    matrix = selected.pivot(index="panel", columns="method", values="mean_selected_infected_fraction")
    methods = [method for method in METHOD_LABELS if method in matrix.columns and method != "random"]
    matrix = matrix.reindex(columns=methods)
    fig, axis = plt.subplots(figsize=(11, max(6.0, 0.55 * len(matrix) + 2.0)))
    image = axis.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods], rotation=35, ha="right")
    axis.set_yticks(range(len(matrix)), matrix.index)
    for row in range(len(matrix)):
        for column in range(len(methods)):
            axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Fraction of additional targets infectious at detection")
    axis.set_title("Immediate case-finding yield of each response policy", fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.27, right=0.95, top=0.88, bottom=0.26)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_detection_design(worlds: pd.DataFrame, path: Path) -> None:
    design = worlds[[
        "dataset_id",
        "parameter_id",
        "mean_infectious_period_days",
        "detection_profile",
        "detection_delay_hours",
    ]].drop_duplicates()
    design["delay_days"] = design["detection_delay_hours"] / 24
    labels = design["dataset_id"].map(DATASET_LABELS).fillna(design["dataset_id"])
    labels += " | " + design["detection_profile"].str.replace("_", " ")
    design = design.assign(panel=labels)
    order = list(dict.fromkeys(design["panel"]))
    fig, axis = plt.subplots(figsize=(11, max(6.0, 0.52 * len(order) + 2.0)))
    for position, panel in enumerate(order):
        values = design.loc[design["panel"].eq(panel), "delay_days"].sort_values()
        axis.scatter(values, np.full(len(values), position), color="#4C78A8", s=65)
        axis.hlines(position, values.min(), values.max(), color="#9ECAE1", linewidth=3, zorder=0)
    axis.set_yticks(range(len(order)), order)
    axis.set_xlabel("Detection delay after epidemic introduction (days)")
    axis.set_title("Predeclared outbreak-detection timing across disease scenarios", fontsize=18, fontweight="bold")
    axis.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_increment_table(frame: pd.DataFrame) -> str:
    columns = [
        "dataset_id",
        "detection_profile",
        "mean_increment",
        "blocked_ci_low",
        "blocked_ci_high",
        "anchors",
        "paired_worlds",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        values = [
            f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _markdown_family_table(frame: pd.DataFrame) -> str:
    columns = [
        "detection_profile",
        "comparison",
        "family_equal_mean",
        "blocked_ci_low",
        "blocked_ci_high",
        "positive_family_fraction",
        "families",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        values = [
            f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile_config = config["profiles"][profile_name]
    stable_path = Path(config["data"]["stable_prediction_path"])
    stable_predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    stable_predictions["anchor_time"] = pd.to_datetime(
        stable_predictions["anchor_time"], format="mixed"
    )
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    implementation_paths = [
        Path(__file__),
        Path(__file__).parents[1] / "simulation" / "outbreak_response.py",
    ]
    fingerprint_payload = config_path.read_bytes() + stable_path.read_bytes()
    for implementation_path in implementation_paths:
        fingerprint_payload += implementation_path.read_bytes()
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:12]
    dataset_payloads = []
    for dataset_id in profile_config["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config_path = Path(specification["source_config"])
        source_config = _load_source_config(source_config_path)
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        available_keys = set(
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
            if (
                str(window["network_id"]),
                pd.Timestamp(window["anchor"].anchor_time),
            )
            in available_keys
        ]
        maximum = profile_config.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        parameters = _selected_parameters(
            Path(specification["source_results"]) / "parameter_selection.csv",
            profile_config.get("selected_parameter_limit"),
        )
        if not windows:
            raise ValueError(f"no forward-response windows matched stable predictions for {dataset_id}")
        dataset_payloads.append((dataset_id, specification, windows, parameters))
    profiles = [DetectionProfile(**item) for item in config["decision"]["detection_profiles"]]
    tasks = []
    support_rows = []
    for dataset_id, specification, windows, parameters in dataset_payloads:
        for window in windows:
            anchor = window["anchor"]
            for parameter in parameters.itertuples(index=False):
                mean_infectious_period = pd.Timedelta(
                    days=float(parameter.mean_infectious_period_days)
                )
                supported = all(
                    detection_time_from_seed(
                        anchor.anchor_time,
                        anchor.horizon_end,
                        mean_infectious_period,
                        detection_profile,
                    )
                    is not None
                    for detection_profile in profiles
                )
                support_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "network_id": str(window["network_id"]),
                        "anchor_id": anchor.anchor_id,
                        "parameter_id": parameter.parameter_id,
                        "mean_infectious_period_days": float(
                            parameter.mean_infectious_period_days
                        ),
                        "all_detection_profiles_observed_within_horizon": supported,
                    }
                )
                if not supported:
                    continue
                analysis_cluster_id = (
                    f"{dataset_id}::{window['network_id']}"
                    if specification.get("analysis_cluster") == "network"
                    else f"{dataset_id}::{window['network_id']}::{anchor.anchor_id}"
                )
                for detection_profile in profiles:
                    tasks.append(
                        (
                            dataset_id,
                            str(window["network_id"]),
                            str(specification["system_family"]),
                            analysis_cluster_id,
                            window,
                            parameter,
                            detection_profile,
                        )
                    )
    parameter_support = pd.DataFrame(support_rows)
    if not tasks:
        raise ValueError("no parameter/window combinations support every detection profile")
    output_frames = []
    progress = tqdm(tasks, desc="Outbreak-response tasks", unit="task")
    for (
        dataset_id,
        network_id,
        system_family,
        analysis_cluster_id,
        window,
        parameter,
        detection_profile,
    ) in progress:
        anchor = window["anchor"]
        task_name = hashlib.sha256(
            f"{fingerprint}|{dataset_id}|{anchor.anchor_id}|{parameter.parameter_id}|{detection_profile.name}".encode()
        ).hexdigest()[:16]
        checkpoint = checkpoint_dir / f"{dataset_id}_{task_name}.csv.gz"
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str})
            expected_methods = set(config["decision"]["methods"])
            if not frame.empty and set(frame["method"]) == expected_methods:
                output_frames.append(frame)
                progress.set_postfix_str(f"{dataset_id} cached")
                continue
        stable_scores = _matching_stable_scores(
            stable_predictions,
            dataset_id,
            network_id,
            anchor.anchor_time,
            window["eligible"],
        )
        seeds = stable_hash_order(
            list(map(str, window["eligible"])),
            int(config["evaluation"]["seed"]),
            dataset_id,
            anchor.anchor_id,
            "response_seeds",
        )[: int(profile_config["seeds_per_anchor"])]
        frame = _run_task(
            dataset_id=dataset_id,
            network_id=network_id,
            system_family=system_family,
            analysis_cluster_id=analysis_cluster_id,
            window=window,
            parameter=parameter,
            profile=detection_profile,
            stable_scores=stable_scores,
            methods=list(config["decision"]["methods"]),
            seed_nodes=seeds,
            random_blocks=int(profile_config["random_blocks"]),
            budget_fraction=float(config["decision"]["additional_budget_fraction"]),
            minimum_budget=int(config["decision"]["minimum_additional_budget"]),
            tracing_half_life_fraction=float(
                config["decision"]["tracing_half_life_fraction_of_mean_infectious_period"]
            ),
            experiment_seed=int(config["evaluation"]["seed"]),
        )
        frame.to_csv(checkpoint, index=False, compression="gzip")
        output_frames.append(frame)
        progress.set_postfix_str(f"{dataset_id} completed")
    worlds = pd.concat(output_frames, ignore_index=True)
    for column in ("anchor_time", "horizon_end", "detection_time"):
        worlds[column] = pd.to_datetime(worlds[column], format="mixed")
    summary = _summarize_policies(
        worlds,
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["seed"]),
    )
    paired, increments = _summarize_increments(
        worlds,
        baseline=str(config["evaluation"]["primary_baseline"]),
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["seed"]),
    )
    family_summary = _family_equal_increment_summary(
        paired,
        baseline=str(config["evaluation"]["primary_baseline"]),
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["seed"]),
    )
    standard_summary = _standard_care_summary(worlds)
    closest_prior_gate = pd.DataFrame()
    if "comparator_scope" in config["evaluation"]:
        closest_prior_gate = _closest_prior_gate(
            worlds,
            method=str(config["evaluation"]["primary_method"]),
            baselines=sorted(config["evaluation"]["comparator_scope"]),
            bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
            seed=int(config["evaluation"]["seed"]),
        )
    worlds.to_csv(results_dir / "response_worlds.csv.gz", index=False, compression="gzip")
    summary.to_csv(results_dir / "policy_summary.csv", index=False)
    paired.to_csv(results_dir / "paired_policy_increments.csv.gz", index=False, compression="gzip")
    increments.to_csv(results_dir / "increment_summary.csv", index=False)
    parameter_support.to_csv(results_dir / "parameter_detection_support.csv", index=False)
    family_summary.to_csv(results_dir / "family_increment_summary.csv", index=False)
    standard_summary.to_csv(results_dir / "standard_care_summary.csv", index=False)
    if not closest_prior_gate.empty:
        closest_prior_gate.to_csv(results_dir / "closest_prior_gate.csv", index=False)
    base_worlds = worlds.drop_duplicates(WORLD_KEYS)
    matched_natural = base_worlds.groupby(
        [
            "dataset_id",
            "network_id",
            "anchor_id",
            "parameter_id",
            "random_block",
            "initial_infected",
            "world_seed",
        ],
        observed=True,
    ).agg(
        detection_profiles=("detection_profile", "nunique"),
        natural_outcomes=("natural_final_size", "nunique"),
    )
    audit = {
        "status": "pass",
        "checks": {
            "world_keys_unique_per_method": not worlds.duplicated(WORLD_KEYS + ["method"]).any(),
            "all_methods_complete": bool(
                worlds.groupby(WORLD_KEYS, observed=True)["method"].nunique().eq(
                    len(config["decision"]["methods"])
                ).all()
            ),
            "fixed_budget": bool(
                worlds["additional_targets"].fillna("").map(
                    lambda value: 0 if not value else len(str(value).split("|"))
                ).eq(worlds["additional_budget"]).all()
            ),
            "detected_excluded_from_targets": bool(
                worlds.apply(
                    lambda row: str(row.initial_infected)
                    not in str(row.additional_targets).split("|"),
                    axis=1,
                ).all()
            ),
            "finite_outcomes": bool(
                np.isfinite(
                    worlds[["attack_rate_reduction", "selected_infected_fraction"]].to_numpy(float)
                ).all()
            ),
            "paired_rows_reconcile": len(paired)
            == len(worlds.loc[worlds["method"].ne(config["evaluation"]["primary_baseline"])]),
            "natural_world_shared_across_detection_profiles": bool(
                matched_natural["detection_profiles"].eq(len(profiles)).all()
                and matched_natural["natural_outcomes"].eq(1).all()
            ),
        },
        "datasets": worlds["dataset_id"].nunique(),
        "system_families": worlds["system_family"].nunique(),
        "analysis_clusters": worlds["analysis_cluster_id"].nunique(),
        "supported_parameter_windows": int(
            parameter_support["all_detection_profiles_observed_within_horizon"].sum()
        ),
        "excluded_parameter_windows": int(
            (~parameter_support["all_detection_profiles_observed_within_horizon"]).sum()
        ),
        "anchors": worlds[["dataset_id", "anchor_id"]].drop_duplicates().shape[0],
        "base_worlds": worlds[WORLD_KEYS].drop_duplicates().shape[0],
        "policy_worlds": len(worlds),
        "methods": worlds["method"].nunique(),
    }
    if not closest_prior_gate.empty:
        mean_positive = closest_prior_gate["family_equal_mean"].gt(0)
        direction_supported = closest_prior_gate["positive_family_fraction"].ge(0.8)
        audit["scientific_gate"] = {
            "status": "pass" if bool((mean_positive & direction_supported).all()) else "fail",
            "interpretation": (
                "stable_watchlist_outperforms_closest_history_comparators"
                if bool((mean_positive & direction_supported).all())
                else "stable_watchlist_not_supported_over_closest_history_comparators"
            ),
            "comparisons": closest_prior_gate.to_dict(orient="records"),
        }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"outbreak-response audit failed: {audit}")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _plot_policy_effects(summary, report_dir / "policy_effects.png")
    comparison_figure = (
        "increment_over_stable.png"
        if str(config["evaluation"]["primary_baseline"]) == "stable_watchlist"
        else "policy_comparisons.png"
    )
    _plot_increment(family_summary, report_dir / comparison_figure)
    _plot_target_yield(summary, report_dir / "target_state_yield.png")
    _plot_detection_design(worlds, report_dir / "detection_design.png")
    if not closest_prior_gate.empty:
        _plot_closest_prior_gate(closest_prior_gate, report_dir / "closest_prior_gate.png")
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
        "stable_predictions_sha256": _sha256(stable_path),
        "audit_status": audit["status"],
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    primary = increments.loc[
        increments["method"].eq(config["evaluation"]["primary_method"])
    ]
    diagnostic_note = (
        "The perfect-state diagnostic can see which animals are infectious at the decision time. "
        "It is not deployable and is not an optimal future oracle.\n\n"
        if "perfect_state_diagnostic" in set(config["decision"]["methods"])
        else ""
    )
    scientific_gate_note = (
        f"Closest-prior scientific gate: **{audit['scientific_gate']['status']}** "
        f"({audit['scientific_gate']['interpretation']}).\n\n"
        if "scientific_gate" in audit
        else ""
    )
    readme = f"""# Detection-triggered outbreak-response pilot

This experiment conditions on a seeded infection being detected after a
predeclared fraction of the mean infectious period. The detected case is
isolated under standard care. Every policy receives the same additional budget
and is evaluated by paired replay against standard care in the same random
world. Contact-based scores use only interactions observed before detection.

- Datasets: {audit['datasets']}
- Independent system families: {audit['system_families']}
- Evaluated anchors: {audit['anchors']}
- Base epidemic worlds: {audit['base_worlds']}
- Policy evaluations: {audit['policy_worlds']}
- Audit status: **{audit['status']}**

{diagnostic_note}{scientific_gate_note}This pilot uses generic detection-delay scenarios and simulation-derived outcomes;
it is not field validation for a named pathogen.

Primary method increments over the configured baseline:
{_markdown_increment_table(primary)}

Family-equal policy comparisons:
{_markdown_family_table(family_summary)}
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detection-triggered outbreak-response pilot")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260816-007_outbreak_response_pilot.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
