from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from animal_intervention.transmission.contract import ExposureStream


@dataclass(frozen=True, slots=True)
class SIRParameters:
    beta: float
    recovery_rate: float

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.recovery_rate < 0:
            raise ValueError("recovery_rate must be non-negative")


@dataclass(slots=True)
class SimulationResult:
    dataset_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    initial_infected: tuple[str, ...]
    isolated_nodes: tuple[str, ...]
    final_states: dict[str, str]
    event_log: pd.DataFrame
    final_size: int
    peak_infectious: int
    extinct: bool
    seed: int


def _seconds(timestamp: pd.Timestamp) -> float:
    return float(timestamp.value / 1_000_000_000)


def _timestamp(seconds: float) -> pd.Timestamp:
    return pd.Timestamp(int(round(seconds * 1_000_000_000)))


class TemporalSIREngine:
    """Exact Markov SIR simulation between piecewise-constant network boundaries.

    Active dyadic and group exposures define infection transition rates. Network
    start/end boundaries are deterministic; infection and recovery transitions
    are sampled with a Gillespie step inside each constant-rate interval.
    """

    def simulate(
        self,
        stream: ExposureStream,
        parameters: SIRParameters,
        *,
        initial_infected: Iterable[str],
        isolated_nodes: Iterable[str] = (),
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        seed: int = 0,
        max_transitions: int = 1_000_000,
    ) -> SimulationResult:
        stream.validate()
        nodes = stream.nodes()
        infected_initial = tuple(sorted(set(map(str, initial_infected))))
        isolated = set(map(str, isolated_nodes))
        if not infected_initial:
            raise ValueError("at least one initial infected node is required")
        if set(infected_initial) - nodes:
            raise ValueError("initial infected nodes are absent from exposure stream")
        if set(infected_initial) & isolated:
            raise ValueError("an initially infected node cannot also be isolated at start")

        start_candidates = pd.concat(
            [
                pd.to_datetime(stream.dyadic_exposures["start_time"], errors="coerce"),
                pd.to_datetime(stream.group_exposures["start_time"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        end_candidates = pd.concat(
            [
                pd.to_datetime(stream.dyadic_exposures["end_time"], errors="coerce"),
                pd.to_datetime(stream.group_exposures["end_time"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        if start_candidates.empty or end_candidates.empty:
            raise ValueError("exposure stream has no temporal extent")
        start_timestamp = pd.Timestamp(start_time or start_candidates.min())
        end_timestamp = pd.Timestamp(end_time or end_candidates.max())
        if end_timestamp <= start_timestamp:
            raise ValueError("simulation end_time must follow start_time")

        states = {node: "S" for node in nodes}
        for node in infected_initial:
            states[node] = "I"
        ever_infected = set(infected_initial)
        peak_infectious = len(infected_initial)
        event_rows: list[dict[str, Any]] = [
            {"time": start_timestamp, "event": "initial_infection", "node_id": node, "source_id": None}
            for node in infected_initial
        ]

        dyadic_records = stream.dyadic_exposures.to_dict("records")
        group_records = stream.group_exposures.to_dict("records")
        members_by_group = {
            str(group_id): {
                str(row.node_id): float(row.membership_weight)
                for row in frame.itertuples(index=False)
            }
            for group_id, frame in stream.group_memberships.groupby("group_event_id", observed=True)
        }
        boundaries: dict[float, dict[str, list[tuple[str, dict[str, Any]]]]] = {}
        active_dyadic: dict[str, dict[str, Any]] = {}
        active_groups: dict[str, dict[str, Any]] = {}
        start_seconds = _seconds(start_timestamp)
        end_seconds = _seconds(end_timestamp)

        def register(record: dict[str, Any], kind: str, identifier_field: str) -> None:
            exposure_start = _seconds(pd.Timestamp(record["start_time"]))
            exposure_end = _seconds(pd.Timestamp(record["end_time"]))
            identifier = str(record[identifier_field])
            if exposure_start <= start_seconds < exposure_end:
                (active_dyadic if kind == "dyadic" else active_groups)[identifier] = record
            if start_seconds < exposure_start < end_seconds:
                boundaries.setdefault(exposure_start, {"end": [], "start": []})["start"].append(
                    (kind, record)
                )
            if start_seconds < exposure_end <= end_seconds:
                boundaries.setdefault(exposure_end, {"end": [], "start": []})["end"].append(
                    (kind, record)
                )

        for record in dyadic_records:
            register(record, "dyadic", "exposure_id")
        for record in group_records:
            register(record, "group", "group_event_id")
        boundary_times = sorted(boundaries)
        boundary_index = 0
        current_time = start_seconds
        rng = np.random.default_rng(seed)
        transitions = 0

        while current_time < end_seconds and transitions < max_transitions:
            next_boundary = (
                boundary_times[boundary_index]
                if boundary_index < len(boundary_times)
                else end_seconds
            )
            transition_options: list[tuple[str, str, str | None, float]] = []
            if parameters.recovery_rate > 0:
                transition_options.extend(
                    ("recovery", node, None, parameters.recovery_rate)
                    for node, state in states.items()
                    if state == "I"
                )

            for record in active_dyadic.values():
                source = str(record["source_id"])
                target = str(record["target_id"])
                if source in isolated or target in isolated:
                    continue
                rate = parameters.beta * float(record["hazard_rate_multiplier"])
                if rate <= 0:
                    continue
                if states.get(source) == "I" and states.get(target) == "S":
                    transition_options.append(("infection", target, source, rate))
                if (
                    not bool(record.get("directed", False))
                    and states.get(target) == "I"
                    and states.get(source) == "S"
                ):
                    transition_options.append(("infection", source, target, rate))

            for record in active_groups.values():
                group_id = str(record["group_event_id"])
                member_weights = {
                    node: weight
                    for node, weight in members_by_group.get(group_id, {}).items()
                    if node not in isolated
                }
                group_size = len(member_weights)
                if group_size < 2:
                    continue
                infected_members = [node for node in member_weights if states.get(node) == "I"]
                susceptible_members = [node for node in member_weights if states.get(node) == "S"]
                denominator = group_size - 1 if record["group_mixing_mode"] == "frequency_dependent" else 1
                base_rate = parameters.beta * float(record["hazard_rate_multiplier"]) / denominator
                for infector in infected_members:
                    for target in susceptible_members:
                        rate = base_rate * member_weights[infector] * member_weights[target]
                        if rate > 0:
                            transition_options.append(("infection", target, infector, rate))

            total_rate = sum(option[3] for option in transition_options)
            if total_rate > 0:
                waiting_time = float(rng.exponential(1.0 / total_rate))
            else:
                waiting_time = float("inf")
            if current_time + waiting_time < next_boundary:
                current_time += waiting_time
                threshold = float(rng.random() * total_rate)
                cumulative = 0.0
                selected = transition_options[-1]
                for option in transition_options:
                    cumulative += option[3]
                    if cumulative >= threshold:
                        selected = option
                        break
                event, node, source, _ = selected
                if event == "recovery" and states.get(node) == "I":
                    states[node] = "R"
                elif event == "infection" and states.get(node) == "S":
                    states[node] = "I"
                    ever_infected.add(node)
                else:
                    continue
                transitions += 1
                infectious_count = sum(state == "I" for state in states.values())
                peak_infectious = max(peak_infectious, infectious_count)
                event_rows.append(
                    {
                        "time": _timestamp(current_time),
                        "event": event,
                        "node_id": node,
                        "source_id": source,
                    }
                )
                continue

            current_time = next_boundary
            if boundary_index >= len(boundary_times):
                break
            actions = boundaries[next_boundary]
            for kind, record in actions["end"]:
                identifier = str(
                    record["exposure_id"] if kind == "dyadic" else record["group_event_id"]
                )
                (active_dyadic if kind == "dyadic" else active_groups).pop(identifier, None)
            for kind, record in actions["start"]:
                identifier = str(
                    record["exposure_id"] if kind == "dyadic" else record["group_event_id"]
                )
                (active_dyadic if kind == "dyadic" else active_groups)[identifier] = record
            boundary_index += 1

        if transitions >= max_transitions:
            raise RuntimeError("maximum transition count reached")
        return SimulationResult(
            dataset_id=stream.dataset_id,
            start_time=start_timestamp,
            end_time=end_timestamp,
            initial_infected=infected_initial,
            isolated_nodes=tuple(sorted(isolated)),
            final_states=states,
            event_log=pd.DataFrame(
                event_rows, columns=["time", "event", "node_id", "source_id"]
            ),
            final_size=len(ever_infected),
            peak_infectious=peak_infectious,
            extinct=not any(state == "I" for state in states.values()),
            seed=seed,
        )
