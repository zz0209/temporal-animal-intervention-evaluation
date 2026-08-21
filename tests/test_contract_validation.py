from __future__ import annotations

import pandas as pd

from animal_intervention.data.contract import CanonicalDataset, DatasetMetadata
from animal_intervention.data.validation import validate_dataset


def test_contract_adds_stable_columns_and_validates(dyadic_dataset):
    report = validate_dataset(dyadic_dataset)
    assert not report.has_errors
    assert report.metrics["individuals"] == 3
    assert "attributes_json" in dyadic_dataset.individuals


def test_validation_rejects_self_loop():
    dataset = CanonicalDataset(
        metadata=DatasetMetadata("bad", "bad", "test"),
        individuals=pd.DataFrame({"dataset_id": ["bad"], "node_id": ["A"]}),
        dyadic_events=pd.DataFrame(
            {
                "dataset_id": ["bad"],
                "event_id": ["e1"],
                "source_id": ["A"],
                "target_id": ["A"],
                "start_time": [pd.Timestamp("2020-01-01")],
                "end_time": [pd.Timestamp("2020-01-01 00:00:01")],
                "duration_seconds": [1.0],
                "measurement_value": [1.0],
            }
        ),
    )
    report = validate_dataset(dataset)
    assert report.has_errors
    assert "self_loops" in {issue.code for issue in report.issues}


def test_parquet_round_trip(tmp_path, dyadic_dataset):
    dyadic_dataset.write(tmp_path)
    restored = CanonicalDataset.read(tmp_path)
    assert restored.summary() == dyadic_dataset.summary()
    assert restored.dyadic_events["event_id"].tolist() == ["duration", "association"]

