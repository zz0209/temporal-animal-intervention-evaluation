from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from animal_intervention.data.adapters import ADAPTERS
from animal_intervention.data.validation import validate_dataset


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("dataset_id", list(ADAPTERS))
def test_real_adapter_smoke(dataset_id):
    raw_directory = DATA_ROOT / dataset_id / "raw"
    if not raw_directory.exists() or not any(raw_directory.iterdir()):
        pytest.skip(
            f"Raw third-party payload for {dataset_id} is not redistributed; "
            "see DATA_SOURCES.md."
        )
    adapter = ADAPTERS[dataset_id]()
    dataset = adapter.load(
        raw_directory, sample=True, progress=False
    )
    report = validate_dataset(dataset)
    assert not report.has_errors, report.to_dict()
    assert len(dataset.individuals) > 0
    assert len(dataset.dyadic_events) + len(dataset.group_events) > 0
    if dataset.metadata.has_temporal_order:
        assert dataset.summary()["time_start"] is not None

    rebuilt = adapter.load(raw_directory, sample=True, progress=False)
    assert rebuilt.summary() == dataset.summary()
    for table_name, frame in dataset.tables().items():
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            rebuilt.tables()[table_name].reset_index(drop=True),
            check_dtype=True,
        )
