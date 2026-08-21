from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from animal_intervention.simulation import (
    InterventionAction,
    PairedTemporalSIREngine,
    SIRParameters,
)
from animal_intervention.transmission.contract import ExposureStream


@dataclass(frozen=True, slots=True)
class AnchorWindow:
    anchor_id: str
    history_start: pd.Timestamp
    anchor_time: pd.Timestamp
    horizon_end: pd.Timestamp


def stream_extent(stream: ExposureStream) -> tuple[pd.Timestamp, pd.Timestamp]:
    temporal_frames = [
        frame
        for frame in (stream.dyadic_exposures, stream.group_exposures)
        if not frame.empty
    ]
    starts = pd.concat(
        [frame["start_time"] for frame in temporal_frames], ignore_index=True
    ).dropna()
    ends = pd.concat(
        [frame["end_time"] for frame in temporal_frames], ignore_index=True
    ).dropna()
    if starts.empty or ends.empty:
        raise ValueError("exposure stream has no temporal extent")
    return pd.Timestamp(pd.to_datetime(starts).min()), pd.Timestamp(pd.to_datetime(ends).max())


def rolling_anchors(
    stream: ExposureStream,
    *,
    lookback: pd.Timedelta,
    horizon: pd.Timedelta,
    step: pd.Timedelta,
    max_anchors: int | None = None,
) -> list[AnchorWindow]:
    if min(lookback, horizon, step) <= pd.Timedelta(0):
        raise ValueError("lookback, horizon, and step must be positive")
    start, end = stream_extent(stream)
    anchors: list[AnchorWindow] = []
    anchor = start + lookback
    while anchor + horizon <= end:
        anchors.append(
            AnchorWindow(
                anchor_id=f"anchor_{len(anchors) + 1:03d}",
                history_start=anchor - lookback,
                anchor_time=anchor,
                horizon_end=anchor + horizon,
            )
        )
        if max_anchors is not None and len(anchors) >= max_anchors:
            break
        anchor += step
    if not anchors:
        raise ValueError("no complete rolling anchor fits within the exposure stream")
    return anchors


def slice_stream(
    stream: ExposureStream,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> ExposureStream:
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    def clipped(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        starts = pd.to_datetime(frame["start_time"])
        ends = pd.to_datetime(frame["end_time"])
        selected = frame.loc[starts.lt(end_time) & ends.gt(start_time)].copy()
        selected["start_time"] = pd.to_datetime(selected["start_time"]).clip(lower=start_time)
        selected["end_time"] = pd.to_datetime(selected["end_time"]).clip(upper=end_time)
        return selected

    dyadic = clipped(stream.dyadic_exposures)
    groups = clipped(stream.group_exposures)
    group_ids = set(groups["group_event_id"].astype(str))
    memberships = stream.group_memberships.loc[
        stream.group_memberships["group_event_id"].astype(str).isin(group_ids)
    ].copy()
    result = ExposureStream(
        dataset_id=stream.dataset_id,
        population_nodes=stream.population_nodes,
        dyadic_exposures=dyadic,
        group_exposures=groups,
        group_memberships=memberships,
        metadata={**stream.metadata, "slice_start": str(start_time), "slice_end": str(end_time)},
    )
    result.validate()
    return result


def node_support(stream: ExposureStream) -> pd.Series:
    counts: dict[str, int] = {node: 0 for node in stream.nodes()}
    for row in stream.dyadic_exposures.itertuples(index=False):
        counts[str(row.source_id)] = counts.get(str(row.source_id), 0) + 1
        counts[str(row.target_id)] = counts.get(str(row.target_id), 0) + 1
    for row in stream.group_memberships.itertuples(index=False):
        counts[str(row.node_id)] = counts.get(str(row.node_id), 0) + 1
    return pd.Series(counts, dtype="int64", name="event_support")


def _stable_index(seed: int, anchor_id: str, world_index: int, size: int) -> int:
    digest = hashlib.sha256(
        f"{seed}|{anchor_id}|{world_index}|initial_seed".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % size


def _world_seed(seed: int, anchor_id: str, world_index: int) -> int:
    digest = hashlib.sha256(
        f"{seed}|{anchor_id}|{world_index}|world".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _stratified_world_seed(
    seed: int,
    anchor_id: str,
    stratum: str,
    initial_node: str,
    replicate: int,
) -> int:
    digest = hashlib.sha256(
        f"{seed}|{anchor_id}|{stratum}|{initial_node}|{replicate}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _candidate_action(
    candidate: str,
    anchor: AnchorWindow,
    action_config: dict[str, Any],
) -> InterventionAction:
    action_start = anchor.anchor_time + pd.Timedelta(action_config["delay"])
    action_end = min(
        anchor.horizon_end,
        action_start + pd.Timedelta(action_config["duration"]),
    )
    return InterventionAction(
        name=str(action_config["name"]),
        action_type=str(action_config["action_type"]),
        target_nodes=(candidate,),
        start_time=action_start,
        end_time=action_end,
        contact_multiplier=float(action_config["contact_multiplier"]),
        susceptibility_multiplier=float(action_config.get("susceptibility_multiplier", 1.0)),
        infectivity_multiplier=float(action_config.get("infectivity_multiplier", 1.0)),
        recovery_rate_multiplier=float(action_config.get("recovery_rate_multiplier", 1.0)),
    )


def estimate_stratified_singleton_values(
    stream: ExposureStream,
    anchors: list[AnchorWindow],
    parameters: SIRParameters,
    *,
    action_config: dict[str, Any],
    self_seed_replicates: int,
    seed: int,
    min_history_events: int = 1,
    candidate_limit: int | None = None,
    bootstrap_replicates: int = 1_000,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate balanced intervention values over index-case strata.

    Every eligible node appears exactly once as the index case in the shared
    non-index block. Each candidate also receives independent self-index
    replicates. The unconditional estimand combines both strata using their
    population probabilities, preventing rare self-index outcomes from being
    over-weighted by a small Monte Carlo sample.
    """
    if self_seed_replicates < 2:
        raise ValueError("at least two self-seed replicates are required")
    if bootstrap_replicates < 200:
        raise ValueError("at least 200 bootstrap replicates are required")
    engine = PairedTemporalSIREngine()
    world_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    prepared: list[
        tuple[AnchorWindow, ExposureStream, list[str], list[str], pd.Series, pd.Series]
    ] = []
    total = 0
    for anchor in anchors:
        history = slice_stream(stream, anchor.history_start, anchor.anchor_time)
        future = slice_stream(stream, anchor.anchor_time, anchor.horizon_end)
        history_support = node_support(history)
        future_support = node_support(future)
        eligible = sorted(
            node for node, count in history_support.items() if count >= min_history_events
        )
        if len(eligible) < 2:
            raise ValueError(f"{anchor.anchor_id} needs at least two eligible nodes")
        candidates = eligible
        if candidate_limit is not None:
            candidates = sorted(
                eligible,
                key=lambda node: (-int(history_support.get(node, 0)), node),
            )[:candidate_limit]
        prepared.append(
            (anchor, future, eligible, candidates, history_support, future_support)
        )
        total += len(candidates) * (len(eligible) - 1 + self_seed_replicates)

    progress = tqdm(
        total=total,
        desc="Balanced candidate-index simulations",
        disable=not show_progress,
    )
    try:
        for anchor, future, eligible, candidates, history_support, future_support in prepared:
            population_size = len(future.nodes() | set(eligible))
            shared_baselines: dict[str, tuple[int, Any]] = {}
            for initial in eligible:
                current_seed = _stratified_world_seed(
                    seed, anchor.anchor_id, "non_index", initial, 0
                )
                shared_baselines[initial] = (
                    current_seed,
                    engine.simulate(
                        future,
                        parameters,
                        initial_infected=[initial],
                        start_time=anchor.anchor_time,
                        end_time=anchor.horizon_end,
                        world_seed=current_seed,
                    ),
                )
            anchor_rows.append(
                {
                    "anchor_id": anchor.anchor_id,
                    "history_start": anchor.history_start,
                    "anchor_time": anchor.anchor_time,
                    "horizon_end": anchor.horizon_end,
                    "eligible_count": len(eligible),
                    "evaluated_candidate_count": len(candidates),
                    "future_population_size": population_size,
                    "non_index_seeds_per_candidate": len(eligible) - 1,
                    "self_seed_replicates": self_seed_replicates,
                }
            )
            for candidate in candidates:
                action = _candidate_action(candidate, anchor, action_config)
                for initial_index, initial in enumerate(eligible):
                    if initial == candidate:
                        continue
                    current_seed, baseline = shared_baselines[initial]
                    intervention = engine.simulate(
                        future,
                        parameters,
                        initial_infected=[initial],
                        start_time=anchor.anchor_time,
                        end_time=anchor.horizon_end,
                        world_seed=current_seed,
                        action=action,
                    )
                    delta = baseline.final_size - intervention.final_size
                    world_rows.append(
                        {
                            "anchor_id": anchor.anchor_id,
                            "candidate_id": candidate,
                            "introduction_stratum": "non_index",
                            "introduction_replicate": initial_index,
                            "world_seed": current_seed,
                            "initial_infected": initial,
                            "baseline_final_size": baseline.final_size,
                            "intervention_final_size": intervention.final_size,
                            "avoided_infections": delta,
                            "avoided_attack_rate": delta / population_size,
                        }
                    )
                    progress.update(1)
                for replicate in range(self_seed_replicates):
                    current_seed = _stratified_world_seed(
                        seed, anchor.anchor_id, "self_index", candidate, replicate
                    )
                    baseline = engine.simulate(
                        future,
                        parameters,
                        initial_infected=[candidate],
                        start_time=anchor.anchor_time,
                        end_time=anchor.horizon_end,
                        world_seed=current_seed,
                    )
                    intervention = engine.simulate(
                        future,
                        parameters,
                        initial_infected=[candidate],
                        start_time=anchor.anchor_time,
                        end_time=anchor.horizon_end,
                        world_seed=current_seed,
                        action=action,
                    )
                    delta = baseline.final_size - intervention.final_size
                    world_rows.append(
                        {
                            "anchor_id": anchor.anchor_id,
                            "candidate_id": candidate,
                            "introduction_stratum": "self_index",
                            "introduction_replicate": replicate,
                            "world_seed": current_seed,
                            "initial_infected": candidate,
                            "baseline_final_size": baseline.final_size,
                            "intervention_final_size": intervention.final_size,
                            "avoided_infections": delta,
                            "avoided_attack_rate": delta / population_size,
                        }
                    )
                    progress.update(1)
    finally:
        progress.close()

    worlds_frame = pd.DataFrame(world_rows)
    anchor_frame = pd.DataFrame(anchor_rows)
    anchor_sizes = anchor_frame.set_index("anchor_id")["eligible_count"].to_dict()
    support_lookup = {
        item[0].anchor_id: (item[4], item[5])
        for item in prepared
    }
    estimate_rows: list[dict[str, Any]] = []
    for (anchor_id, candidate), group in worlds_frame.groupby(
        ["anchor_id", "candidate_id"], observed=True, sort=False
    ):
        self_values = group.loc[
            group["introduction_stratum"].eq("self_index"), "avoided_attack_rate"
        ].to_numpy(dtype=float)
        non_values = group.loc[
            group["introduction_stratum"].eq("non_index"), "avoided_attack_rate"
        ].to_numpy(dtype=float)
        population = int(anchor_sizes[anchor_id])
        self_weight = 1.0 / population
        non_weight = 1.0 - self_weight
        self_mean = float(self_values.mean())
        non_mean = float(non_values.mean())
        unconditional = self_weight * self_mean + non_weight * non_mean
        self_se = float(self_values.std(ddof=1) / math.sqrt(len(self_values)))
        non_se = float(non_values.std(ddof=1) / math.sqrt(len(non_values)))
        unconditional_se = math.sqrt(
            (self_weight * self_se) ** 2 + (non_weight * non_se) ** 2
        )
        bootstrap_seed = _stratified_world_seed(
            seed, anchor_id, "bootstrap", candidate, 0
        )
        rng = np.random.default_rng(bootstrap_seed)
        self_boot = rng.choice(
            self_values, size=(bootstrap_replicates, len(self_values)), replace=True
        ).mean(axis=1)
        non_boot = rng.choice(
            non_values, size=(bootstrap_replicates, len(non_values)), replace=True
        ).mean(axis=1)
        unconditional_boot = self_weight * self_boot + non_weight * non_boot
        history_support, future_support = support_lookup[anchor_id]
        estimate_rows.append(
            {
                "anchor_id": anchor_id,
                "candidate_id": candidate,
                "eligible_population": population,
                "self_index_worlds": len(self_values),
                "non_index_worlds": len(non_values),
                "known_index_value": self_mean,
                "known_index_ci95_lower": float(np.quantile(self_boot, 0.025)),
                "known_index_ci95_upper": float(np.quantile(self_boot, 0.975)),
                "non_index_value": non_mean,
                "non_index_ci95_lower": float(np.quantile(non_boot, 0.025)),
                "non_index_ci95_upper": float(np.quantile(non_boot, 0.975)),
                "unconditional_value": unconditional,
                "mean_avoided_attack_rate": unconditional,
                "mc_standard_error": unconditional_se,
                "ci95_lower": float(np.quantile(unconditional_boot, 0.025)),
                "ci95_upper": float(np.quantile(unconditional_boot, 0.975)),
                "probability_beneficial": float(
                    self_weight * (self_values > 0).mean()
                    + non_weight * (non_values > 0).mean()
                ),
                "history_event_support": int(history_support.get(candidate, 0)),
                "future_event_support": int(future_support.get(candidate, 0)),
            }
        )
    estimates = pd.DataFrame(estimate_rows)
    estimates["rank"] = estimates.groupby("anchor_id")["unconditional_value"].rank(
        method="min", ascending=False
    ).astype(int)
    estimates = estimates.sort_values(
        ["anchor_id", "rank", "candidate_id"], ignore_index=True
    )
    return estimates, worlds_frame, anchor_frame


def estimate_singleton_values(
    stream: ExposureStream,
    anchors: list[AnchorWindow],
    parameters: SIRParameters,
    *,
    action_config: dict[str, Any],
    worlds: int,
    seed: int,
    min_history_events: int = 1,
    candidate_limit: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int], Any]]:
    if worlds < 2:
        raise ValueError("at least two Monte Carlo worlds are required")
    engine = PairedTemporalSIREngine()
    world_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    baseline_results: dict[tuple[str, int], Any] = {}

    prepared: list[tuple[AnchorWindow, ExposureStream, list[str], pd.Series, pd.Series]] = []
    total = 0
    for anchor in anchors:
        history = slice_stream(stream, anchor.history_start, anchor.anchor_time)
        future = slice_stream(stream, anchor.anchor_time, anchor.horizon_end)
        history_support = node_support(history)
        future_support = node_support(future)
        eligible = sorted(
            node for node, count in history_support.items() if count >= min_history_events
        )
        if candidate_limit is not None:
            eligible = sorted(
                eligible,
                key=lambda node: (-int(history_support.get(node, 0)), node),
            )[:candidate_limit]
        if not eligible:
            raise ValueError(f"{anchor.anchor_id} has no eligible candidates")
        prepared.append((anchor, future, eligible, history_support, future_support))
        total += worlds * len(eligible)

    progress = tqdm(total=total, desc="Paired candidate-world simulations", disable=not show_progress)
    try:
        for anchor, future, eligible, history_support, future_support in prepared:
            population_size = len(future.nodes() | set(eligible))
            if population_size == 0:
                raise ValueError(f"{anchor.anchor_id} future exposure window has no nodes")
            baselines: dict[int, Any] = {}
            initial_by_world: dict[int, str] = {}
            for world_index in range(worlds):
                initial = eligible[_stable_index(seed, anchor.anchor_id, world_index, len(eligible))]
                current_seed = _world_seed(seed, anchor.anchor_id, world_index)
                baseline = engine.simulate(
                    future,
                    parameters,
                    initial_infected=[initial],
                    start_time=anchor.anchor_time,
                    end_time=anchor.horizon_end,
                    world_seed=current_seed,
                )
                baselines[world_index] = baseline
                initial_by_world[world_index] = initial
                baseline_results[(anchor.anchor_id, world_index)] = baseline

            anchor_rows.append(
                {
                    "anchor_id": anchor.anchor_id,
                    "history_start": anchor.history_start,
                    "anchor_time": anchor.anchor_time,
                    "horizon_end": anchor.horizon_end,
                    "eligible_count": len(eligible),
                    "future_population_size": population_size,
                }
            )
            for candidate in eligible:
                for world_index in range(worlds):
                    baseline = baselines[world_index]
                    current_seed = _world_seed(seed, anchor.anchor_id, world_index)
                    action_start = anchor.anchor_time + pd.Timedelta(action_config["delay"])
                    action_end = min(
                        anchor.horizon_end,
                        action_start + pd.Timedelta(action_config["duration"]),
                    )
                    action = InterventionAction(
                        name=str(action_config["name"]),
                        action_type=str(action_config["action_type"]),
                        target_nodes=(candidate,),
                        start_time=action_start,
                        end_time=action_end,
                        contact_multiplier=float(action_config["contact_multiplier"]),
                        susceptibility_multiplier=float(
                            action_config.get("susceptibility_multiplier", 1.0)
                        ),
                        infectivity_multiplier=float(
                            action_config.get("infectivity_multiplier", 1.0)
                        ),
                        recovery_rate_multiplier=float(
                            action_config.get("recovery_rate_multiplier", 1.0)
                        ),
                    )
                    intervention = engine.simulate(
                        future,
                        parameters,
                        initial_infected=[initial_by_world[world_index]],
                        start_time=anchor.anchor_time,
                        end_time=anchor.horizon_end,
                        world_seed=current_seed,
                        action=action,
                    )
                    delta = baseline.final_size - intervention.final_size
                    world_rows.append(
                        {
                            "anchor_id": anchor.anchor_id,
                            "candidate_id": candidate,
                            "world_index": world_index,
                            "world_seed": current_seed,
                            "initial_infected": initial_by_world[world_index],
                            "baseline_final_size": baseline.final_size,
                            "intervention_final_size": intervention.final_size,
                            "avoided_infections": delta,
                            "avoided_attack_rate": delta / population_size,
                        }
                    )
                    progress.update(1)
    finally:
        progress.close()

    worlds_frame = pd.DataFrame(world_rows)
    for (anchor_id, candidate), group in worlds_frame.groupby(
        ["anchor_id", "candidate_id"], observed=True, sort=False
    ):
        values = group["avoided_attack_rate"].astype(float)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(len(values)))
        candidate_rows.append(
            {
                "anchor_id": anchor_id,
                "candidate_id": candidate,
                "worlds": len(values),
                "mean_avoided_infections": float(group["avoided_infections"].mean()),
                "mean_avoided_attack_rate": mean,
                "mc_standard_error": se,
                "ci95_lower": mean - 1.96 * se,
                "ci95_upper": mean + 1.96 * se,
                "probability_beneficial": float((group["avoided_infections"] > 0).mean()),
                "history_event_support": int(
                    next(item[3] for item in prepared if item[0].anchor_id == anchor_id).get(
                        candidate, 0
                    )
                ),
                "future_event_support": int(
                    next(item[4] for item in prepared if item[0].anchor_id == anchor_id).get(
                        candidate, 0
                    )
                ),
            }
        )
    estimates = pd.DataFrame(candidate_rows)
    estimates["rank"] = estimates.groupby("anchor_id")[
        "mean_avoided_attack_rate"
    ].rank(method="min", ascending=False).astype(int)
    estimates = estimates.sort_values(
        ["anchor_id", "rank", "candidate_id"], ignore_index=True
    )
    return estimates, worlds_frame, pd.DataFrame(anchor_rows), baseline_results
