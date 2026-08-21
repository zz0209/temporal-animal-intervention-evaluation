from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import pandas as pd
import pytest

from animal_intervention.data.contract import CanonicalDataset, DatasetMetadata


@pytest.fixture
def dyadic_dataset() -> CanonicalDataset:
    start = pd.Timestamp("2020-01-01T00:00:00")
    return CanonicalDataset(
        metadata=DatasetMetadata(
            dataset_id="test_dyadic",
            title="Synthetic dyadic fixture",
            adapter_name="test",
            edge_semantics=["sensor_proximity"],
            measurement_types=["duration", "association_index"],
        ),
        individuals=pd.DataFrame(
            {
                "dataset_id": "test_dyadic",
                "node_id": ["A", "B", "C"],
                "species": "test species",
            }
        ),
        dyadic_events=pd.DataFrame(
            {
                "dataset_id": "test_dyadic",
                "event_id": ["duration", "association"],
                "source_id": ["A", "B"],
                "target_id": ["B", "C"],
                "start_time": [start, start + pd.Timedelta("10s")],
                "end_time": [start + pd.Timedelta("10s"), start + pd.Timedelta("20s")],
                "duration_seconds": [10.0, 10.0],
                "event_representation": ["interval", "aggregated_interval"],
                "edge_semantics": ["sensor_proximity", "association"],
                "measurement_type": ["duration", "association_index"],
                "measurement_value": [10.0, 0.4],
                "measurement_unit": ["seconds", "unitless"],
                "directed": [False, False],
            }
        ),
    )


@pytest.fixture
def group_dataset() -> CanonicalDataset:
    start = pd.Timestamp("2020-01-01T00:00:00")
    return CanonicalDataset(
        metadata=DatasetMetadata(
            dataset_id="test_group",
            title="Synthetic group fixture",
            adapter_name="test",
            primary_event_mode="group",
            edge_semantics=["group_coattendance"],
        ),
        individuals=pd.DataFrame(
            {
                "dataset_id": "test_group",
                "node_id": ["A", "B", "C"],
            }
        ),
        group_events=pd.DataFrame(
            {
                "dataset_id": "test_group",
                "group_event_id": ["g1"],
                "start_time": [start],
                "end_time": [start + pd.Timedelta("10s")],
                "duration_seconds": [10.0],
                "event_semantics": ["group_coattendance"],
            }
        ),
        group_memberships=pd.DataFrame(
            {
                "dataset_id": "test_group",
                "group_event_id": ["g1", "g1", "g1"],
                "node_id": ["A", "B", "C"],
                "membership_weight": [1.0, 1.0, 1.0],
            }
        ),
    )
