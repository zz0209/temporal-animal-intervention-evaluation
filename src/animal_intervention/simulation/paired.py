from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math
from typing import Any, Iterable

import pandas as pd

from animal_intervention.transmission.contract import ExposureStream

from .interventions import InterventionAction, neutral_action
from .sir import SIRParameters, SimulationResult


def _keyed_exponential(seed: int, *parts: object) -> float:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], "big")
    uniform = (integer + 0.5) / 2**64
    return -math.log(uniform)


def _segments(
    start: pd.Timestamp,
    end: pd.Timestamp,
    action: InterventionAction,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    boundaries = {pd.Timestamp(start), pd.Timestamp(end)}
    for boundary in (pd.Timestamp(action.start_time), pd.Timestamp(action.end_time)):
        if start < boundary < end:
            boundaries.add(boundary)
    ordered = sorted(boundaries)
    return list(zip(ordered[:-1], ordered[1:]))


def _transmission_multiplier(
    action: InterventionAction,
    source: str,
    target: str,
    time: pd.Timestamp,
) -> float:
    if not action.active_at(time):
        return 1.0
    multiplier = 1.0
    if action.is_target(source):
        multiplier *= action.contact_multiplier * action.infectivity_multiplier
    if action.is_target(target):
        multiplier *= action.contact_multiplier * action.susceptibility_multiplier
    return multiplier


@dataclass(frozen=True, slots=True)
class _Opportunity:
    opportunity_id: str
    source_id: str
    target_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    rate_multiplier: float


@dataclass(frozen=True, slots=True)
class InterventionHazardAccounting:
    """Integrated transmission-opportunity mass during an intervention."""

    original_hazard_mass: float
    residual_original_hazard_mass: float
    removed_original_hazard_mass: float
    rewired_hazard_mass: float
    rewired_opportunities: int


class PairedTemporalSIREngine:
    """Replay temporal exposures with addressable common random primitives.

    Recovery clocks and every directed transmission opportunity use hash-keyed
    exponential thresholds. An intervention changes hazards but never changes
    the random primitive assigned to a biological opportunity.
    """

    def __init__(self) -> None:
        self._opportunity_cache: dict[int, dict[str, list[_Opportunity]]] = {}
        self._validated_streams: dict[int, ExposureStream] = {}
        self._hazard_pair_cache: dict[
            tuple[int, int, int],
            tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]],
        ] = {}

    def intervention_hazard_accounting(
        self,
        stream: ExposureStream,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        action: InterventionAction,
    ) -> InterventionHazardAccounting:
        """Summarize hazard removal and redistribution without disease states.

        The quantities describe directed transmission opportunities, not physical
        contact duration. They are intended for within-context mechanism audits.
        """

        start = max(pd.Timestamp(start_time), pd.Timestamp(action.start_time))
        end = min(pd.Timestamp(end_time), pd.Timestamp(action.end_time))
        if end <= start:
            return InterventionHazardAccounting(0.0, 0.0, 0.0, 0.0, 0)
        outgoing = self._opportunity_cache.get(id(stream))
        if outgoing is None:
            outgoing = self._build_opportunities(stream, start, end)
        cache_key = (id(stream), start.value, end.value)
        cached_pairs = self._hazard_pair_cache.get(cache_key)
        if cached_pairs is None:
            pair_mass: dict[tuple[str, str], float] = {}
            pair_count: dict[tuple[str, str], int] = {}
            for opportunities in outgoing.values():
                for opportunity in opportunities:
                    overlap_start = max(start, opportunity.start_time)
                    overlap_end = min(end, opportunity.end_time)
                    if overlap_end <= overlap_start:
                        continue
                    duration = (overlap_end - overlap_start).total_seconds()
                    key = (opportunity.source_id, opportunity.target_id)
                    pair_mass[key] = pair_mass.get(key, 0.0) + (
                        opportunity.rate_multiplier * duration
                    )
                    pair_count[key] = pair_count.get(key, 0) + 1
            cached_pairs = (pair_mass, pair_count)
            self._hazard_pair_cache[cache_key] = cached_pairs
        pair_mass, pair_count = cached_pairs
        nodes = stream.nodes()
        available = nodes - set(action.target_nodes)
        original_mass = float(sum(pair_mass.values()))
        residual_mass = 0.0
        rewired_mass = 0.0
        rewired_opportunities = 0
        lost_fraction = (
            (1.0 - action.contact_multiplier) * action.rewiring_fraction
            if action.action_type == "isolation"
            and action.rewiring_mode != "none"
            else 0.0
        )
        midpoint = start + (end - start) / 2
        for (source, target), mass in pair_mass.items():
            residual_mass += mass * _transmission_multiplier(
                action, source, target, midpoint
            )
            source_targeted = action.is_target(source)
            target_targeted = action.is_target(target)
            if lost_fraction > 0 and source_targeted != target_targeted:
                survivor = target if source_targeted else source
                if available - {survivor}:
                    rewired_mass += mass * lost_fraction
                    rewired_opportunities += pair_count[(source, target)]
        return InterventionHazardAccounting(
            original_hazard_mass=float(original_mass),
            residual_original_hazard_mass=float(residual_mass),
            removed_original_hazard_mass=float(original_mass - residual_mass),
            rewired_hazard_mass=float(rewired_mass),
            rewired_opportunities=rewired_opportunities,
        )

    def simulate(
        self,
        stream: ExposureStream,
        parameters: SIRParameters,
        *,
        initial_infected: Iterable[str],
        initial_recovered: Iterable[str] = (),
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        world_seed: int,
        action: InterventionAction | None = None,
    ) -> SimulationResult:
        cache_key = id(stream)
        cached_stream = self._validated_streams.get(cache_key)
        if cached_stream is not stream:
            stream.validate()
            self._validated_streams[cache_key] = stream
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        if end_time <= start_time:
            raise ValueError("simulation end_time must follow start_time")
        action = action or neutral_action(start_time, end_time)
        nodes = stream.nodes()
        initial = tuple(sorted(set(map(str, initial_infected))))
        recovered_initial = tuple(sorted(set(map(str, initial_recovered))))
        if not initial:
            raise ValueError("at least one initial infected node is required")
        if set(initial) & set(recovered_initial):
            raise ValueError("initial infected and recovered nodes must be disjoint")
        nodes.update(initial)
        nodes.update(recovered_initial)

        outgoing = self._opportunity_cache.get(cache_key)
        if outgoing is None:
            outgoing = self._build_opportunities(stream, start_time, end_time)
            self._opportunity_cache[cache_key] = outgoing
        rewired_outgoing = self._build_rewired_opportunities(outgoing, nodes, action)
        states = {node: "S" for node in nodes}
        for node in recovered_initial:
            states[node] = "R"
        ever_infected = set(initial) | set(recovered_initial)
        infection_times: dict[str, pd.Timestamp] = {}
        event_rows: list[dict[str, Any]] = []
        queue: list[tuple[int, int, str, str, str | None]] = []
        sequence = 0

        def push(time: pd.Timestamp, kind: str, node: str, source: str | None) -> None:
            nonlocal sequence
            if time <= end_time:
                sequence += 1
                heapq.heappush(queue, (time.value, sequence, kind, node, source))

        def schedule_from(source: str, infected_at: pd.Timestamp) -> None:
            recovery = self._recovery_time(
                source, infected_at, end_time, parameters, action, world_seed
            )
            if recovery is not None:
                push(recovery, "recovery", source, None)
            for opportunity in (*outgoing.get(source, []), *rewired_outgoing.get(source, [])):
                candidate = self._transmission_time(
                    opportunity,
                    max(infected_at, opportunity.start_time),
                    parameters,
                    action,
                    world_seed,
                )
                if candidate is not None:
                    push(candidate, "infection", opportunity.target_id, source)

        for node in initial:
            states[node] = "I"
            infection_times[node] = start_time
            event_rows.append(
                {
                    "time": start_time,
                    "event": "initial_infection",
                    "node_id": node,
                    "source_id": None,
                }
            )
            schedule_from(node, start_time)

        peak_infectious = len(initial)
        while queue:
            time_value, _, kind, node, source = heapq.heappop(queue)
            event_time = pd.Timestamp(time_value)
            if kind == "recovery":
                if states.get(node) != "I":
                    continue
                states[node] = "R"
            else:
                if states.get(node) != "S" or source is None or states.get(source) != "I":
                    continue
                states[node] = "I"
                ever_infected.add(node)
                infection_times[node] = event_time
                schedule_from(node, event_time)
            event_rows.append(
                {
                    "time": event_time,
                    "event": kind,
                    "node_id": node,
                    "source_id": source,
                }
            )
            peak_infectious = max(peak_infectious, sum(value == "I" for value in states.values()))

        event_log = pd.DataFrame(
            event_rows, columns=["time", "event", "node_id", "source_id"]
        ).sort_values(["time", "event", "node_id"], kind="stable", ignore_index=True)
        return SimulationResult(
            dataset_id=stream.dataset_id,
            start_time=start_time,
            end_time=end_time,
            initial_infected=initial,
            isolated_nodes=(
                tuple(sorted(action.target_nodes))
                if action.action_type == "isolation"
                else ()
            ),
            final_states=states,
            event_log=event_log,
            final_size=len(ever_infected),
            peak_infectious=peak_infectious,
            extinct=not any(value == "I" for value in states.values()),
            seed=world_seed,
        )

    @staticmethod
    def _build_rewired_opportunities(
        outgoing: dict[str, list[_Opportunity]],
        nodes: set[str],
        action: InterventionAction,
    ) -> dict[str, list[_Opportunity]]:
        """Redirect a bounded fraction of target-partner contact to non-targets.

        Each original directed opportunity with exactly one targeted endpoint keeps
        its non-target endpoint and replaces the targeted endpoint with a
        deterministic uniformly selected non-target. The added hazard equals the
        configured fraction of contact hazard removed by the target's contact
        multiplier. Target-target opportunities are not redistributed.
        """

        if (
            action.action_type != "isolation"
            or action.rewiring_fraction <= 0
            or action.rewiring_mode == "none"
        ):
            return {}
        targets = set(action.target_nodes)
        node_order = sorted(nodes)
        available = set(node_order) - targets
        additions: dict[str, list[_Opportunity]] = {}
        lost_fraction = (1.0 - action.contact_multiplier) * action.rewiring_fraction
        if lost_fraction <= 0:
            return additions
        for opportunities in outgoing.values():
            for opportunity in opportunities:
                source_targeted = opportunity.source_id in targets
                target_targeted = opportunity.target_id in targets
                if source_targeted == target_targeted:
                    continue
                survivor = (
                    opportunity.target_id if source_targeted else opportunity.source_id
                )
                if not (available - {survivor}):
                    continue
                payload = "\x1f".join(
                    [
                        str(action.start_time.value),
                        opportunity.opportunity_id,
                        opportunity.source_id,
                        opportunity.target_id,
                    ]
                )
                start_index = int.from_bytes(
                    hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
                ) % len(node_order)
                replacement = None
                for offset in range(len(node_order)):
                    candidate = node_order[(start_index + offset) % len(node_order)]
                    if candidate in available and candidate != survivor:
                        replacement = candidate
                        break
                if replacement is None:
                    continue
                source = replacement if source_targeted else survivor
                target = survivor if source_targeted else replacement
                start = max(opportunity.start_time, pd.Timestamp(action.start_time))
                end = min(opportunity.end_time, pd.Timestamp(action.end_time))
                if end <= start:
                    continue
                rewired = _Opportunity(
                    opportunity_id=(
                        f"rewired:{opportunity.opportunity_id}:{source}:{target}"
                    ),
                    source_id=source,
                    target_id=target,
                    start_time=start,
                    end_time=end,
                    rate_multiplier=opportunity.rate_multiplier * lost_fraction,
                )
                additions.setdefault(source, []).append(rewired)
        for opportunities in additions.values():
            opportunities.sort(key=lambda item: (item.start_time, item.opportunity_id))
        return additions

    @staticmethod
    def _build_opportunities(
        stream: ExposureStream,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> dict[str, list[_Opportunity]]:
        outgoing: dict[str, list[_Opportunity]] = {}

        def add(opportunity: _Opportunity) -> None:
            if opportunity.end_time > opportunity.start_time and opportunity.rate_multiplier > 0:
                outgoing.setdefault(opportunity.source_id, []).append(opportunity)

        for row in stream.dyadic_exposures.itertuples(index=False):
            exposure_start = max(start_time, pd.Timestamp(row.start_time))
            exposure_end = min(end_time, pd.Timestamp(row.end_time))
            add(
                _Opportunity(
                    str(row.exposure_id), str(row.source_id), str(row.target_id),
                    exposure_start, exposure_end, float(row.hazard_rate_multiplier)
                )
            )
            if not bool(row.directed):
                add(
                    _Opportunity(
                        f"{row.exposure_id}:reverse", str(row.target_id), str(row.source_id),
                        exposure_start, exposure_end, float(row.hazard_rate_multiplier)
                    )
                )

        members = {
            str(group_id): [
                (str(row.node_id), float(row.membership_weight))
                for row in frame.itertuples(index=False)
            ]
            for group_id, frame in stream.group_memberships.groupby(
                "group_event_id", observed=True
            )
        }
        for row in stream.group_exposures.itertuples(index=False):
            group_id = str(row.group_event_id)
            group_members = members.get(group_id, [])
            denominator = (
                max(1, len(group_members) - 1)
                if row.group_mixing_mode == "frequency_dependent"
                else 1
            )
            exposure_start = max(start_time, pd.Timestamp(row.start_time))
            exposure_end = min(end_time, pd.Timestamp(row.end_time))
            for source, source_weight in group_members:
                for target, target_weight in group_members:
                    if source == target:
                        continue
                    add(
                        _Opportunity(
                            f"{group_id}:{source}:{target}", source, target,
                            exposure_start, exposure_end,
                            float(row.hazard_rate_multiplier)
                            * source_weight * target_weight / denominator,
                        )
                    )
        for opportunities in outgoing.values():
            opportunities.sort(key=lambda item: (item.start_time, item.opportunity_id))
        return outgoing

    @staticmethod
    def _transmission_time(
        opportunity: _Opportunity,
        risk_start: pd.Timestamp,
        parameters: SIRParameters,
        action: InterventionAction,
        world_seed: int,
    ) -> pd.Timestamp | None:
        if risk_start >= opportunity.end_time or parameters.beta <= 0:
            return None
        budget = _keyed_exponential(
            world_seed,
            "transmission",
            opportunity.opportunity_id,
            opportunity.source_id,
            opportunity.target_id,
        )
        for segment_start, segment_end in _segments(risk_start, opportunity.end_time, action):
            midpoint = segment_start + (segment_end - segment_start) / 2
            rate = (
                parameters.beta
                * opportunity.rate_multiplier
                * _transmission_multiplier(
                    action, opportunity.source_id, opportunity.target_id, midpoint
                )
            )
            if rate <= 0:
                continue
            duration = (segment_end - segment_start).total_seconds()
            integrated = rate * duration
            if budget <= integrated:
                return segment_start + pd.to_timedelta(budget / rate, unit="s")
            budget -= integrated
        return None

    @staticmethod
    def _recovery_time(
        node: str,
        infected_at: pd.Timestamp,
        end_time: pd.Timestamp,
        parameters: SIRParameters,
        action: InterventionAction,
        world_seed: int,
    ) -> pd.Timestamp | None:
        if parameters.recovery_rate <= 0:
            return None
        budget = _keyed_exponential(world_seed, "recovery", node)
        for segment_start, segment_end in _segments(infected_at, end_time, action):
            midpoint = segment_start + (segment_end - segment_start) / 2
            modifier = (
                action.recovery_rate_multiplier
                if action.is_target(node) and action.active_at(midpoint)
                else 1.0
            )
            rate = parameters.recovery_rate * modifier
            if rate <= 0:
                continue
            duration = (segment_end - segment_start).total_seconds()
            integrated = rate * duration
            if budget <= integrated:
                return segment_start + pd.to_timedelta(budget / rate, unit="s")
            budget -= integrated
        return None
