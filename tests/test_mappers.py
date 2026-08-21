from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.data.contract import CanonicalDataset, DatasetMetadata
from animal_intervention.transmission.mappers import (
    AggregatedAssociationMapper,
    CoalescedDurationContactMapper,
    DurationContactMapper,
    GroupMixingMapper,
    compile_primary_exposure,
)


def test_duration_mapper_selects_only_compatible_events(dyadic_dataset):
    stream = DurationContactMapper().compile(dyadic_dataset)
    assert stream.dyadic_exposures["origin_event_id"].tolist() == ["duration"]
    assert stream.dyadic_exposures["hazard_rate_multiplier"].tolist() == [1.0]


def test_association_mapper_preserves_integrated_weight(dyadic_dataset):
    stream = AggregatedAssociationMapper().compile(dyadic_dataset)
    row = stream.dyadic_exposures.iloc[0]
    duration = (row.end_time - row.start_time).total_seconds()
    assert row.hazard_rate_multiplier * duration == pytest.approx(0.4)


def test_group_mapper_preserves_group_representation(group_dataset):
    stream = GroupMixingMapper(mode="frequency_dependent").compile(group_dataset)
    assert len(stream.group_exposures) == 1
    assert len(stream.group_memberships) == 3
    assert stream.group_exposures.iloc[0].group_mixing_mode == "frequency_dependent"


def test_primary_mapper_selection_uses_contract_not_dataset_name(group_dataset):
    stream = compile_primary_exposure(group_dataset)
    assert stream.metadata["mapper"] == "GroupMixingMapper"


def test_mapper_excludes_zero_length_observation(dyadic_dataset):
    dyadic_dataset.dyadic_events.loc[
        dyadic_dataset.dyadic_events["event_id"].eq("duration"), "end_time"
    ] = dyadic_dataset.dyadic_events.loc[
        dyadic_dataset.dyadic_events["event_id"].eq("duration"), "start_time"
    ].to_numpy()
    with pytest.raises(ValueError, match="no positive"):
        DurationContactMapper().compile(dyadic_dataset)


def test_mapper_excludes_flagged_exact_source_duplicate(dyadic_dataset):
    duplicate = dyadic_dataset.dyadic_events.iloc[[0]].copy()
    duplicate["event_id"] = "duplicate"
    duplicate["quality_flag"] = "exact_duplicate_source_record"
    dyadic_dataset.dyadic_events = pd.concat(
        [dyadic_dataset.dyadic_events, duplicate], ignore_index=True
    )
    stream = DurationContactMapper().compile(dyadic_dataset)
    assert stream.dyadic_exposures["origin_event_id"].tolist() == ["duration"]
    assert stream.metadata["excluded_exact_source_duplicates"] == 1


def test_coalesced_duration_mapper_uses_interval_union(dyadic_dataset):
    base = dyadic_dataset.dyadic_events.loc[
        dyadic_dataset.dyadic_events["event_id"].eq("duration")
    ].iloc[0]
    overlap = base.copy()
    overlap["event_id"] = "duration-overlap"
    overlap["start_time"] = pd.Timestamp(base["start_time"]) + pd.Timedelta(seconds=5)
    overlap["end_time"] = pd.Timestamp(base["end_time"]) + pd.Timedelta(seconds=5)
    overlap["duration_seconds"] = (
        pd.Timestamp(overlap["end_time"]) - pd.Timestamp(overlap["start_time"])
    ).total_seconds()
    dyadic_dataset.dyadic_events = pd.concat(
        [dyadic_dataset.dyadic_events, overlap.to_frame().T], ignore_index=True
    )
    stream = CoalescedDurationContactMapper().compile(dyadic_dataset)
    assert len(stream.dyadic_exposures) == 1
    row = stream.dyadic_exposures.iloc[0]
    assert row.start_time == pd.Timestamp(base["start_time"])
    assert row.end_time == pd.Timestamp(overlap["end_time"])
    assert stream.metadata["merged_source_exposure_count"] == 1


def test_mapper_rejects_non_temporal_dataset():
    dataset = CanonicalDataset(
        metadata=DatasetMetadata(
            dataset_id="static",
            title="static",
            adapter_name="test",
            has_temporal_order=False,
        )
    )
    with pytest.raises(ValueError, match="no recoverable temporal order"):
        DurationContactMapper().compile(dataset)
