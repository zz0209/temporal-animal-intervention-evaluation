from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.simulation import (
    ContactReductionPhase,
    apply_contact_reduction_schedule,
    segment_exposure_stream,
)
from animal_intervention.transmission.contract import ExposureStream


def _stream() -> ExposureStream:
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(hours=3)
    return ExposureStream(
        dataset_id="fixture",
        population_nodes=("a", "b", "c"),
        dyadic_exposures=pd.DataFrame(
            [
                {
                    "dataset_id": "fixture",
                    "exposure_id": "edge",
                    "source_id": "a",
                    "target_id": "b",
                    "start_time": start,
                    "end_time": end,
                    "hazard_rate_multiplier": 2.0,
                    "directed": False,
                    "transmission_route": "fixture",
                    "mapper_name": "fixture",
                    "origin_event_id": "edge",
                    "location_id": "site",
                }
            ]
        ),
        group_exposures=pd.DataFrame(
            [
                {
                    "dataset_id": "fixture",
                    "group_event_id": "group",
                    "start_time": start,
                    "end_time": end,
                    "hazard_rate_multiplier": 3.0,
                    "transmission_route": "fixture",
                    "mapper_name": "fixture",
                    "group_mixing_mode": "frequency_dependent",
                    "location_id": "site",
                }
            ]
        ),
        group_memberships=pd.DataFrame(
            [
                {
                    "dataset_id": "fixture",
                    "group_event_id": "group",
                    "node_id": node,
                    "membership_weight": 1.0,
                }
                for node in ("a", "b", "c")
            ]
        ),
    )


def test_segmentation_preserves_integrated_dyadic_hazard() -> None:
    stream = _stream()
    cut = pd.Timestamp("2020-01-01 01:00:00")
    segmented = segment_exposure_stream(stream, [cut])
    duration = (
        pd.to_datetime(segmented.dyadic_exposures["end_time"])
        - pd.to_datetime(segmented.dyadic_exposures["start_time"])
    ).dt.total_seconds()
    integrated = (
        duration * segmented.dyadic_exposures["hazard_rate_multiplier"].astype(float)
    ).sum()
    assert len(segmented.dyadic_exposures) == 2
    assert integrated == pytest.approx(2.0 * 3 * 3600)


def test_schedule_applies_one_multiplier_per_targeted_endpoint() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    first = start + pd.Timedelta(hours=1)
    second = start + pd.Timedelta(hours=2)
    segmented = segment_exposure_stream(stream, [first, second])
    adjusted = apply_contact_reduction_schedule(
        segmented,
        [
            ContactReductionPhase(first, second, ("a",), 0.25),
            ContactReductionPhase(second, start + pd.Timedelta(hours=3), ("a", "b"), 0.25),
        ],
    )
    assert adjusted.dyadic_exposures["exposure_id"].tolist() == segmented.dyadic_exposures[
        "exposure_id"
    ].tolist()
    assert adjusted.dyadic_exposures["hazard_rate_multiplier"].astype(float).tolist() == pytest.approx(
        [2.0, 0.5, 0.125]
    )


def test_schedule_scales_group_membership_weights() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    cut = start + pd.Timedelta(hours=1)
    segmented = segment_exposure_stream(stream, [cut])
    adjusted = apply_contact_reduction_schedule(
        segmented,
        [ContactReductionPhase(cut, start + pd.Timedelta(hours=3), ("a",), 0.2)],
    )
    after_group = adjusted.group_exposures.sort_values("start_time").iloc[-1]["group_event_id"]
    after = adjusted.group_memberships.loc[
        adjusted.group_memberships["group_event_id"].eq(after_group)
    ].set_index("node_id")["membership_weight"]
    assert float(after.loc["a"]) == pytest.approx(0.2)
    assert float(after.loc["b"]) == pytest.approx(1.0)


def test_schedule_rejects_overlapping_phases() -> None:
    stream = segment_exposure_stream(_stream(), [])
    start = pd.Timestamp("2020-01-01")
    phases = [
        ContactReductionPhase(start, start + pd.Timedelta(hours=2), ("a",), 0.5),
        ContactReductionPhase(
            start + pd.Timedelta(hours=1),
            start + pd.Timedelta(hours=3),
            ("b",),
            0.5,
        ),
    ]
    with pytest.raises(ValueError, match="must not overlap"):
        apply_contact_reduction_schedule(stream, phases)
