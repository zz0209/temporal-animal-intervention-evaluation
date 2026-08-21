from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from animal_intervention.transmission.contract import ExposureStream


@dataclass(frozen=True, slots=True)
class ContactReductionPhase:
    """One phase of a node-targeted contact-reduction schedule."""

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    target_nodes: tuple[str, ...]
    residual_contact_multiplier: float

    def __post_init__(self) -> None:
        if pd.Timestamp(self.end_time) <= pd.Timestamp(self.start_time):
            raise ValueError("contact-reduction phase end must follow start")
        if not 0 <= self.residual_contact_multiplier <= 1:
            raise ValueError("residual contact multiplier must be between zero and one")


def _segment_bounds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    boundaries: tuple[pd.Timestamp, ...],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    points = {pd.Timestamp(start), pd.Timestamp(end)}
    points.update(boundary for boundary in boundaries if start < boundary < end)
    ordered = sorted(points)
    return list(zip(ordered[:-1], ordered[1:]))


def segment_exposure_stream(
    stream: ExposureStream,
    boundaries: Iterable[pd.Timestamp],
) -> ExposureStream:
    """Split every exposure at shared policy times without changing total hazard."""

    cuts = tuple(sorted({pd.Timestamp(value) for value in boundaries}))
    dyadic_rows: list[dict[str, object]] = []
    for row in stream.dyadic_exposures.to_dict("records"):
        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])
        for left, right in _segment_bounds(start, end, cuts):
            item = dict(row)
            item["start_time"] = left
            item["end_time"] = right
            item["exposure_id"] = (
                f"{row['exposure_id']}::segment::{left.value}::{right.value}"
            )
            dyadic_rows.append(item)

    memberships = {
        str(group_id): frame.to_dict("records")
        for group_id, frame in stream.group_memberships.groupby(
            "group_event_id", observed=True, sort=False
        )
    }
    group_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for row in stream.group_exposures.to_dict("records"):
        original_id = str(row["group_event_id"])
        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])
        for left, right in _segment_bounds(start, end, cuts):
            segment_id = f"{original_id}::segment::{left.value}::{right.value}"
            item = dict(row)
            item["group_event_id"] = segment_id
            item["start_time"] = left
            item["end_time"] = right
            group_rows.append(item)
            for membership in memberships.get(original_id, []):
                copied = dict(membership)
                copied["group_event_id"] = segment_id
                membership_rows.append(copied)

    segmented = ExposureStream(
        dataset_id=stream.dataset_id,
        population_nodes=stream.population_nodes,
        dyadic_exposures=pd.DataFrame(dyadic_rows),
        group_exposures=pd.DataFrame(group_rows),
        group_memberships=pd.DataFrame(membership_rows),
        metadata={
            **stream.metadata,
            "segmentation_boundaries": [value.isoformat() for value in cuts],
        },
    )
    segmented.validate()
    return segmented


def _active_phase(
    time: pd.Timestamp,
    phases: tuple[ContactReductionPhase, ...],
) -> ContactReductionPhase | None:
    active = [
        phase
        for phase in phases
        if pd.Timestamp(phase.start_time) <= time < pd.Timestamp(phase.end_time)
    ]
    if len(active) > 1:
        raise ValueError("contact-reduction phases must not overlap")
    return active[0] if active else None


def apply_contact_reduction_schedule(
    segmented_stream: ExposureStream,
    phases: Iterable[ContactReductionPhase],
) -> ExposureStream:
    """Apply phased node contact multipliers to an already segmented stream.

    Dyadic hazards receive one multiplier per targeted endpoint. Group-event
    membership weights receive the endpoint multiplier, yielding the same pairwise
    product when the transmission engine expands a group into directed opportunities.
    """

    ordered = tuple(sorted(phases, key=lambda item: pd.Timestamp(item.start_time)))
    for first, second in zip(ordered[:-1], ordered[1:]):
        if pd.Timestamp(first.end_time) > pd.Timestamp(second.start_time):
            raise ValueError("contact-reduction phases must not overlap")

    dyadic = segmented_stream.dyadic_exposures.copy()
    for index, row in dyadic.iterrows():
        midpoint = pd.Timestamp(row["start_time"]) + (
            pd.Timestamp(row["end_time"]) - pd.Timestamp(row["start_time"])
        ) / 2
        phase = _active_phase(midpoint, ordered)
        if phase is None:
            continue
        targets = set(map(str, phase.target_nodes))
        endpoint_count = int(str(row["source_id"]) in targets) + int(
            str(row["target_id"]) in targets
        )
        dyadic.at[index, "hazard_rate_multiplier"] = float(
            row["hazard_rate_multiplier"]
        ) * phase.residual_contact_multiplier**endpoint_count

    groups = segmented_stream.group_exposures.copy()
    memberships = segmented_stream.group_memberships.copy()
    group_phase: dict[str, ContactReductionPhase | None] = {}
    for row in groups.itertuples(index=False):
        midpoint = pd.Timestamp(row.start_time) + (
            pd.Timestamp(row.end_time) - pd.Timestamp(row.start_time)
        ) / 2
        group_phase[str(row.group_event_id)] = _active_phase(midpoint, ordered)
    for index, row in memberships.iterrows():
        phase = group_phase.get(str(row["group_event_id"]))
        if phase is None or str(row["node_id"]) not in set(map(str, phase.target_nodes)):
            continue
        memberships.at[index, "membership_weight"] = float(
            row["membership_weight"]
        ) * phase.residual_contact_multiplier

    adjusted = ExposureStream(
        dataset_id=segmented_stream.dataset_id,
        population_nodes=segmented_stream.population_nodes,
        dyadic_exposures=dyadic,
        group_exposures=groups,
        group_memberships=memberships,
        metadata={
            **segmented_stream.metadata,
            "contact_reduction_schedule": [
                {
                    "start_time": pd.Timestamp(phase.start_time).isoformat(),
                    "end_time": pd.Timestamp(phase.end_time).isoformat(),
                    "target_nodes": list(map(str, phase.target_nodes)),
                    "residual_contact_multiplier": phase.residual_contact_multiplier,
                }
                for phase in ordered
            ],
        },
    )
    adjusted.validate()
    return adjusted
