from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np
import pandas as pd

from animal_intervention.transmission.contract import ExposureStream

from .interventions import InterventionAction
from .paired import PairedTemporalSIREngine
from .sir import SIRParameters, SimulationResult


@dataclass(frozen=True, slots=True)
class DetectionProfile:
    """Operational detection timing expressed relative to the infectious period."""

    name: str
    delay_fraction_of_mean_infectious_period: float

    def __post_init__(self) -> None:
        if self.delay_fraction_of_mean_infectious_period < 0:
            raise ValueError("detection delay fraction must be non-negative")


@dataclass(frozen=True, slots=True)
class ContactObservationProfile:
    """A preregistered stress test for contact evidence available at decision time."""

    name: str
    event_retention_probability: float = 1.0
    tag_retention_probability: float = 1.0
    time_bin: pd.Timedelta | None = None
    binary_intensity: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.event_retention_probability, "event retention probability"),
            (self.tag_retention_probability, "tag retention probability"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be between zero and one")
        if self.time_bin is not None and self.time_bin <= pd.Timedelta(0):
            raise ValueError("time bin must be positive")


@dataclass(slots=True)
class ResponsePair:
    """Paired standard-care and augmented-response outcomes in one random world."""

    natural_history: SimulationResult
    standard_care: SimulationResult
    augmented_response: SimulationResult
    detection_time: pd.Timestamp
    detected_nodes: tuple[str, ...]
    additional_targets: tuple[str, ...]

    @property
    def avoided_infections(self) -> int:
        return self.standard_care.final_size - self.augmented_response.final_size


def _keyed_uniform(seed: int, *parts: object) -> float:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / 2**64


def detection_time_from_seed(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    mean_infectious_period: pd.Timedelta,
    profile: DetectionProfile,
) -> pd.Timestamp | None:
    """Return the first seeded-case detection time, or None beyond the horizon."""

    detection_time = pd.Timestamp(start_time) + (
        mean_infectious_period * profile.delay_fraction_of_mean_infectious_period
    )
    return detection_time if detection_time < pd.Timestamp(end_time) else None


def states_at(result: SimulationResult, time: pd.Timestamp) -> dict[str, str]:
    """Reconstruct epidemic states immediately before a decision timestamp."""

    states = {str(node): "S" for node in result.final_states}
    events = result.event_log.loc[pd.to_datetime(result.event_log["time"]).lt(time)]
    staged_progression = "become_infectious" in set(result.event_log["event"])
    for row in events.sort_values(["time", "event", "node_id"], kind="stable").itertuples():
        node = str(row.node_id)
        if row.event == "initial_infection":
            states[node] = "I"
        elif row.event == "infection":
            states[node] = "E" if staged_progression else "I"
        elif row.event == "become_infectious":
            states[node] = "I"
        elif row.event == "recovery":
            states[node] = "R"
    return states


def observe_detected_cases(
    states: dict[str, str],
    *,
    trigger_node: str,
    secondary_case_sensitivity: float,
    false_positive_rate: float = 0.0,
    world_seed: int,
) -> tuple[str, ...]:
    """Observe a trigger case plus keyed true- and false-positive detections.

    The trigger case represents the case that initiated outbreak response. The
    sensitivity applies to other animals that are infectious immediately before
    the decision time. The false-positive rate applies independently to animals
    that are not infectious. Both samples are nested under keyed uniforms.
    """

    if not 0 <= secondary_case_sensitivity <= 1:
        raise ValueError("secondary-case sensitivity must be between zero and one")
    if not 0 <= false_positive_rate <= 1:
        raise ValueError("false-positive rate must be between zero and one")
    trigger = str(trigger_node)
    detected = {trigger}
    for node, state in sorted(states.items()):
        node = str(node)
        if node == trigger:
            continue
        if state == "I" and (
            _keyed_uniform(world_seed, "secondary_case_detection", node)
            < secondary_case_sensitivity
        ):
            detected.add(node)
        elif state != "I" and (
            _keyed_uniform(world_seed, "secondary_case_false_positive", node)
            < false_positive_rate
        ):
            detected.add(node)
    return tuple(sorted(detected))


def _overlap_weight(
    start: pd.Timestamp,
    end: pd.Timestamp,
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
    half_life: pd.Timedelta,
) -> float:
    left = max(pd.Timestamp(start), pd.Timestamp(observation_start))
    right = min(pd.Timestamp(end), pd.Timestamp(observation_end))
    if right <= left:
        return 0.0
    duration = (right - left).total_seconds()
    midpoint = left + (right - left) / 2
    age = max(0.0, (pd.Timestamp(observation_end) - midpoint).total_seconds())
    decay = 0.5 ** (age / half_life.total_seconds())
    return duration * decay


def _observed_overlap_weight(
    start: pd.Timestamp,
    end: pd.Timestamp,
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
    half_life: pd.Timedelta,
    time_bin: pd.Timedelta | None,
) -> float:
    """Return a causal recency weight after optional timestamp coarsening.

    Coarsening rounds contact age away from the decision time. It preserves the
    observed overlap duration and never makes an old contact appear more recent.
    """

    left = max(pd.Timestamp(start), pd.Timestamp(observation_start))
    right = min(pd.Timestamp(end), pd.Timestamp(observation_end))
    if right <= left:
        return 0.0
    duration = (right - left).total_seconds()
    midpoint = left + (right - left) / 2
    age = max(0.0, (pd.Timestamp(observation_end) - midpoint).total_seconds())
    if time_bin is not None:
        bin_seconds = time_bin.total_seconds()
        age = math.ceil(age / bin_seconds) * bin_seconds
    decay = 0.5 ** (age / half_life.total_seconds())
    return duration * decay


def _observation_kept(seed: int, kind: str, identifier: object, probability: float) -> bool:
    return _keyed_uniform(seed, "contact_observation", kind, identifier) < probability


def perturbed_pre_detection_scores(
    stream: ExposureStream,
    *,
    detected_nodes: Iterable[str],
    start_time: pd.Timestamp,
    detection_time: pd.Timestamp,
    half_life: pd.Timedelta,
    profile: ContactObservationProfile,
    observation_seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Score pre-detection contacts after a paired observation-process perturbation.

    The epidemic replay is not modified. Event and tag loss use keyed uniforms so
    stronger retention stress is nested within milder stress for the same world.
    """

    if half_life <= pd.Timedelta(0):
        raise ValueError("half_life must be positive")
    detected = set(map(str, detected_nodes))
    nodes = sorted(stream.nodes())
    retained_tags = {
        node
        for node in nodes
        if _observation_kept(
            observation_seed, "tag", node, profile.tag_retention_probability
        )
    }
    rows = {
        node: {"candidate_id": node, "current_activity": 0.0, "contact_to_detected": 0.0}
        for node in nodes
    }
    eligible_dyadic = 0
    retained_dyadic = 0
    for exposure in stream.dyadic_exposures.itertuples(index=False):
        source = str(exposure.source_id)
        target = str(exposure.target_id)
        if source not in retained_tags or target not in retained_tags:
            continue
        base = _observed_overlap_weight(
            exposure.start_time,
            exposure.end_time,
            start_time,
            detection_time,
            half_life,
            profile.time_bin,
        )
        if base <= 0:
            continue
        eligible_dyadic += 1
        identifier = (
            str(exposure.exposure_id)
            if pd.notna(exposure.exposure_id)
            else f"{source}|{target}|{exposure.start_time}|{exposure.end_time}"
        )
        if not _observation_kept(
            observation_seed,
            "dyadic_event",
            identifier,
            profile.event_retention_probability,
        ):
            continue
        retained_dyadic += 1
        multiplier = 1.0 if profile.binary_intensity else float(
            exposure.hazard_rate_multiplier
        )
        weight = base * multiplier
        rows[source]["current_activity"] += weight
        rows[target]["current_activity"] += weight
        if source in detected and target not in detected:
            rows[target]["contact_to_detected"] += weight
        if not bool(exposure.directed) and target in detected and source not in detected:
            rows[source]["contact_to_detected"] += weight

    memberships = {
        str(group_id): [
            (str(row.node_id), float(row.membership_weight))
            for row in frame.itertuples(index=False)
            if str(row.node_id) in retained_tags
        ]
        for group_id, frame in stream.group_memberships.groupby(
            "group_event_id", observed=True
        )
    }
    eligible_groups = 0
    retained_groups = 0
    for exposure in stream.group_exposures.itertuples(index=False):
        members = memberships.get(str(exposure.group_event_id), [])
        base = _observed_overlap_weight(
            exposure.start_time,
            exposure.end_time,
            start_time,
            detection_time,
            half_life,
            profile.time_bin,
        )
        if base <= 0 or len(members) < 2:
            continue
        eligible_groups += 1
        if not _observation_kept(
            observation_seed,
            "group_event",
            str(exposure.group_event_id),
            profile.event_retention_probability,
        ):
            continue
        retained_groups += 1
        multiplier = 1.0 if profile.binary_intensity else float(
            exposure.hazard_rate_multiplier
        )
        base *= multiplier
        denominator = (
            max(1, len(members) - 1)
            if exposure.group_mixing_mode == "frequency_dependent"
            else 1
        )
        detected_members = [(node, weight) for node, weight in members if node in detected]
        for node, node_weight in members:
            partner_weight = sum(weight for partner, weight in members if partner != node)
            rows[node]["current_activity"] += base * node_weight * partner_weight / denominator
            if node not in detected:
                rows[node]["contact_to_detected"] += (
                    base
                    * node_weight
                    * sum(weight for _, weight in detected_members)
                    / denominator
                )
    diagnostics: dict[str, float | int] = {
        "population_nodes": len(nodes),
        "retained_tag_nodes": len(retained_tags),
        "eligible_dyadic_events": eligible_dyadic,
        "retained_dyadic_events": retained_dyadic,
        "eligible_group_events": eligible_groups,
        "retained_group_events": retained_groups,
    }
    return pd.DataFrame(rows.values()), diagnostics


def pre_detection_scores(
    stream: ExposureStream,
    *,
    detected_nodes: Iterable[str],
    start_time: pd.Timestamp,
    detection_time: pd.Timestamp,
    half_life: pd.Timedelta,
) -> pd.DataFrame:
    """Compute information available from contacts observed before detection."""
    scores, _ = perturbed_pre_detection_scores(
        stream,
        detected_nodes=detected_nodes,
        start_time=start_time,
        detection_time=detection_time,
        half_life=half_life,
        profile=ContactObservationProfile(name="complete_observation"),
        observation_seed=0,
    )
    return scores


def _percentile_scores(values: pd.Series) -> pd.Series:
    if len(values) <= 1 or values.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    return values.rank(method="average", pct=True)


def select_additional_targets(
    score_table: pd.DataFrame,
    *,
    method: str,
    budget: int,
    detected_nodes: Iterable[str],
    world_seed: int,
) -> tuple[str, ...]:
    """Select a fixed-budget response set using only declared score columns."""

    detected = set(map(str, detected_nodes))
    candidates = score_table.loc[
        ~score_table["candidate_id"].astype(str).isin(detected)
    ].copy()
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if budget == 0 or candidates.empty:
        return ()
    budget = min(int(budget), len(candidates))
    if method == "random":
        candidates["policy_score"] = candidates["candidate_id"].map(
            lambda node: _keyed_uniform(world_seed, "response_random", node)
        )
    elif method == "stable_watchlist":
        candidates["policy_score"] = candidates["stable_score"]
    elif method == "history_weight":
        if "history_weight" not in candidates:
            raise ValueError("history_weight requires history_weight")
        candidates["policy_score"] = candidates["history_weight"]
    elif method == "history_recency":
        if "history_recency" not in candidates:
            raise ValueError("history_recency requires history_recency")
        candidates["policy_score"] = candidates["history_recency"]
    elif method == "current_activity":
        candidates["policy_score"] = candidates["current_activity"]
    elif method == "contact_to_detected":
        candidates["policy_score"] = candidates["contact_to_detected"]
    elif method == "stable_plus_tracing":
        candidates["policy_score"] = (
            _percentile_scores(candidates["stable_score"])
            + _percentile_scores(candidates["contact_to_detected"])
        ) / 2
    elif method == "perfect_state_diagnostic":
        if "infected_at_detection" not in candidates:
            raise ValueError("perfect_state_diagnostic requires infected_at_detection")
        candidates["policy_score"] = (
            candidates["infected_at_detection"].astype(float) * 2
            + _percentile_scores(candidates["current_activity"])
        )
    else:
        raise ValueError(f"unknown response method: {method}")
    candidates["tie_break"] = candidates["candidate_id"].map(
        lambda node: _keyed_uniform(world_seed, "response_tie", method, node)
    )
    return tuple(
        candidates.sort_values(
            ["policy_score", "tie_break", "candidate_id"],
            ascending=[False, False, True],
            kind="stable",
        )["candidate_id"]
        .astype(str)
        .head(budget)
    )


def run_response_pair(
    engine: PairedTemporalSIREngine,
    stream: ExposureStream,
    parameters: SIRParameters,
    *,
    initial_infected: Iterable[str],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    detection_time: pd.Timestamp,
    detected_nodes: Iterable[str],
    additional_targets: Iterable[str],
    world_seed: int,
) -> ResponsePair:
    """Replay standard care and augmented quarantine with identical randomness."""

    detected = tuple(sorted(set(map(str, detected_nodes))))
    additional = tuple(sorted(set(map(str, additional_targets)) - set(detected)))
    natural = engine.simulate(
        stream,
        parameters,
        initial_infected=initial_infected,
        start_time=start_time,
        end_time=end_time,
        world_seed=world_seed,
    )
    standard_action = InterventionAction(
        name="isolate_detected_cases",
        action_type="isolation",
        target_nodes=detected,
        start_time=detection_time,
        end_time=end_time,
        contact_multiplier=0.0,
    )
    augmented_action = InterventionAction(
        name="isolate_detected_and_additional_targets",
        action_type="isolation",
        target_nodes=tuple(sorted(set(detected) | set(additional))),
        start_time=detection_time,
        end_time=end_time,
        contact_multiplier=0.0,
    )
    standard = engine.simulate(
        stream,
        parameters,
        initial_infected=initial_infected,
        start_time=start_time,
        end_time=end_time,
        world_seed=world_seed,
        action=standard_action,
    )
    augmented = engine.simulate(
        stream,
        parameters,
        initial_infected=initial_infected,
        start_time=start_time,
        end_time=end_time,
        world_seed=world_seed,
        action=augmented_action,
    )
    return ResponsePair(
        natural_history=natural,
        standard_care=standard,
        augmented_response=augmented,
        detection_time=pd.Timestamp(detection_time),
        detected_nodes=detected,
        additional_targets=additional,
    )


def pre_detection_event_signature(
    result: SimulationResult, detection_time: pd.Timestamp
) -> list[tuple[object, ...]]:
    """Return a comparison-safe event signature before intervention starts."""

    frame = result.event_log.loc[
        pd.to_datetime(result.event_log["time"]).lt(detection_time)
    ]
    return list(frame[["time", "event", "node_id", "source_id"]].itertuples(index=False, name=None))
