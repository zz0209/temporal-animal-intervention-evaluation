from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Iterable

import pandas as pd

from animal_intervention.transmission.contract import ExposureStream

from .interventions import InterventionAction, neutral_action
from .paired import PairedTemporalSIREngine, _keyed_exponential
from .sir import SimulationResult


@dataclass(frozen=True, slots=True)
class SEIRParameters:
    beta: float
    latent_rate: float
    recovery_rate: float
    latent_stages: int = 2
    infectious_stages: int = 3

    def __post_init__(self) -> None:
        if self.beta < 0 or self.latent_rate < 0 or self.recovery_rate < 0:
            raise ValueError("epidemic rates must be non-negative")
        if self.latent_stages < 1 or self.infectious_stages < 1:
            raise ValueError("Erlang stage counts must be positive integers")


def _erlang_duration_seconds(
    world_seed: int,
    clock: str,
    node: str,
    rate: float,
    stages: int,
) -> float | None:
    if rate <= 0:
        return None
    stage_rate = stages * rate
    return sum(
        _keyed_exponential(world_seed, clock, node, stage) / stage_rate
        for stage in range(stages)
    )


class PairedTemporalSEIREngine(PairedTemporalSIREngine):
    """Paired temporal SEIR replay with addressable Erlang progression clocks."""

    def simulate(
        self,
        stream: ExposureStream,
        parameters: SEIRParameters,
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
        if action.recovery_rate_multiplier != 1.0:
            raise ValueError(
                "staged SEIR currently requires recovery_rate_multiplier=1"
            )

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
        event_rows: list[dict[str, Any]] = []
        queue: list[tuple[int, int, str, str, str | None]] = []
        sequence = 0

        def push(time: pd.Timestamp, kind: str, node: str, source: str | None) -> None:
            nonlocal sequence
            if time <= end_time:
                sequence += 1
                heapq.heappush(queue, (time.value, sequence, kind, node, source))

        def schedule_infectious_period(node: str, infectious_at: pd.Timestamp) -> None:
            duration = _erlang_duration_seconds(
                world_seed,
                "infectious_stage",
                node,
                parameters.recovery_rate,
                parameters.infectious_stages,
            )
            if duration is not None:
                push(
                    infectious_at + pd.to_timedelta(duration, unit="s"),
                    "recovery",
                    node,
                    None,
                )
            for opportunity in (
                *outgoing.get(node, []),
                *rewired_outgoing.get(node, []),
            ):
                candidate = self._transmission_time(
                    opportunity,
                    max(infectious_at, opportunity.start_time),
                    parameters,
                    action,
                    world_seed,
                )
                if candidate is not None:
                    push(candidate, "infection", opportunity.target_id, node)

        def schedule_progression(node: str, exposed_at: pd.Timestamp) -> None:
            duration = _erlang_duration_seconds(
                world_seed,
                "latent_stage",
                node,
                parameters.latent_rate,
                parameters.latent_stages,
            )
            if duration is None:
                push(exposed_at, "become_infectious", node, None)
            else:
                push(
                    exposed_at + pd.to_timedelta(duration, unit="s"),
                    "become_infectious",
                    node,
                    None,
                )

        for node in initial:
            states[node] = "I"
            event_rows.append(
                {
                    "time": start_time,
                    "event": "initial_infection",
                    "node_id": node,
                    "source_id": None,
                }
            )
            schedule_infectious_period(node, start_time)

        peak_infectious = len(initial)
        while queue:
            time_value, _, kind, node, source = heapq.heappop(queue)
            event_time = pd.Timestamp(time_value)
            if kind == "recovery":
                if states.get(node) != "I":
                    continue
                states[node] = "R"
            elif kind == "become_infectious":
                if states.get(node) != "E":
                    continue
                states[node] = "I"
                schedule_infectious_period(node, event_time)
            else:
                if states.get(node) != "S" or source is None or states.get(source) != "I":
                    continue
                states[node] = "E"
                ever_infected.add(node)
                schedule_progression(node, event_time)
            event_rows.append(
                {
                    "time": event_time,
                    "event": kind,
                    "node_id": node,
                    "source_id": source,
                }
            )
            peak_infectious = max(
                peak_infectious,
                sum(value == "I" for value in states.values()),
            )

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
            extinct=not any(value in {"E", "I"} for value in states.values()),
            seed=world_seed,
        )
