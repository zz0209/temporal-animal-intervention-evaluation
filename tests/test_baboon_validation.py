from __future__ import annotations

from pathlib import Path

import pandas as pd

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.evaluation import aggregate_label_precision
from animal_intervention.experiments.baboon_validation import _data_quality_audit
from animal_intervention.experiments.oxford_predefense import _parameter_grid
from animal_intervention.transmission.mappers import compile_primary_exposure


def test_parameter_grid_accepts_modality_neutral_beta_values():
    grid = _parameter_grid(
        {
            "beta_values": [0.001, 0.002],
            "mean_infectious_period_days": [1.0, 2.0],
        }
    )
    assert len(grid) == 4
    assert set(grid["beta"]) == {0.001, 0.002}
    assert set(grid["mean_infectious_period_days"]) == {1.0, 2.0}
    assert pd.api.types.is_numeric_dtype(grid["recovery_rate_per_day"])


def test_baboon_raw_and_canonical_timestamps_reconcile():
    repository_root = Path(__file__).resolve().parents[1]
    canonical_path = repository_root / "data/guinea_baboons_sociopatterns/processed"
    raw_path = (
        repository_root
        / "data/guinea_baboons_sociopatterns/raw/baboons_proximity_data.txt.gz"
    )
    if not canonical_path.exists() or not raw_path.exists():
        return
    dataset = CanonicalDataset.read(canonical_path)
    audit, _, _ = _data_quality_audit(
        dataset, compile_primary_exposure(dataset), raw_path
    )
    assert audit["raw_timezone_offset_seconds_values"] == [7200.0]
    assert audit["canonical_timestamp_mismatch_rows"] == 0


def test_aggregate_label_precision_targets_delivered_label():
    rows = []
    for block_id, values in enumerate(
        ([0.9, 0.5, 0.1], [0.8, 0.6, 0.2], [0.85, 0.55, 0.15])
    ):
        for parameter_id in ("p1", "p2"):
            for candidate_id, value in zip(("A", "B", "C"), values):
                rows.append(
                    {
                        "anchor_id": "anchor_001",
                        "parameter_id": parameter_id,
                        "block_id": block_id,
                        "candidate_id": candidate_id,
                        "unconditional_value": value,
                    }
                )
    stability, separation, metrics = aggregate_label_precision(
        pd.DataFrame(rows), top_k=1
    )
    assert len(stability) == 3
    assert len(separation) == 1
    assert metrics["aggregate_label_block_count"] == 3
    assert metrics["aggregate_label_single_block_spearman"] == 1.0
    assert metrics["aggregate_label_spearman_brown_reliability"] == 1.0
    assert metrics["minimum_anchor_spearman_brown_reliability"] == 1.0
