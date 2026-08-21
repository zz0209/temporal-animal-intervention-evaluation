from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.estimands.label_contract import (
    LABEL_CONTRACT_VERSION,
    build_model_ready_labels,
    validate_model_ready_labels,
)
from animal_intervention.evaluation import aggregate_label_precision, spearman_brown_reliability
from animal_intervention.transmission.contract import ExposureStream


def _stream() -> ExposureStream:
    return ExposureStream(
        dataset_id="fixture",
        dyadic_exposures=pd.DataFrame(
            [
                {
                    "dataset_id": "fixture",
                    "exposure_id": "h1",
                    "source_id": "a",
                    "target_id": "b",
                    "start_time": pd.Timestamp("2020-01-01"),
                    "end_time": pd.Timestamp("2020-01-02"),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                },
                {
                    "dataset_id": "fixture",
                    "exposure_id": "f1",
                    "source_id": "a",
                    "target_id": "b",
                    "start_time": pd.Timestamp("2020-01-02"),
                    "end_time": pd.Timestamp("2020-01-03"),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                },
            ]
        ),
        metadata={"mapper": "FixtureMapper", "beta_unit": "per_fixture_unit"},
    )


def _block_estimates() -> pd.DataFrame:
    rows = []
    for parameter_id, parameter_offset in (("p1", 0.0), ("p2", 0.1)):
        for block_id, block_offset in ((0, 0.0), (1, 0.01), (2, -0.01)):
            for candidate_id, base in (("a", 0.3), ("b", 0.1)):
                rows.append(
                    {
                        "anchor_id": "anchor_001",
                        "parameter_id": parameter_id,
                        "block_id": block_id,
                        "candidate_id": candidate_id,
                        "eligible_population": 2,
                        "outcome_population": 2,
                        "self_index_worlds": 4,
                        "non_index_worlds": 1,
                        "unconditional_value": base + parameter_offset + block_offset,
                    }
                )
    return pd.DataFrame(rows)


def _labels(block_estimates: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        block_estimates.groupby(
            ["anchor_id", "parameter_id", "candidate_id"], as_index=False
        )["unconditional_value"]
        .mean()
    )
    aggregate["priority"] = aggregate.groupby(["anchor_id", "parameter_id"])[
        "unconditional_value"
    ].rank(pct=True)
    return (
        aggregate.groupby(["anchor_id", "candidate_id"], as_index=False)
        .agg(
            parameter_contexts=("parameter_id", "nunique"),
            robust_intervention_value=("unconditional_value", "mean"),
            minimum_scenario_value=("unconditional_value", "min"),
            maximum_scenario_value=("unconditional_value", "max"),
            disease_scenario_sd=("unconditional_value", "std"),
            robust_priority_percentile=("priority", "mean"),
            minimum_priority_percentile=("priority", "min"),
            maximum_priority_percentile=("priority", "max"),
        )
        .assign(mean_random_block_sd=0.01, robust_rank=[1.0, 2.0])
    )


def test_model_ready_contract_preserves_dataset_specific_provenance() -> None:
    blocks = _block_estimates()
    config = {
        "experiment": {"id": "EXP-FIXTURE"},
        "windows": {"lookback": "1d", "horizon": "1d", "step": "1d"},
        "intervention": {"action_type": "isolation", "duration": "1d"},
        "profiles": {"full": {"max_anchors": 1}},
    }
    model = build_model_ready_labels(
        labels=_labels(blocks),
        block_estimates=blocks,
        stream=_stream(),
        config=config,
        profile="full",
    )
    assert len(model) == 2
    assert model["label_contract_version"].eq(LABEL_CONTRACT_VERSION).all()
    assert model["primary_mapper"].eq("FixtureMapper").all()
    assert model["network_id"].eq("all").all()
    assert model["introduction_sampling"].eq("exhaustive").all()
    assert model["random_block_count"].eq(3).all()
    assert validate_model_ready_labels(model)["status"] == "passed"


def test_aggregate_precision_reports_anchor_range() -> None:
    _, separation, metrics = aggregate_label_precision(_block_estimates(), top_k=1)
    assert len(separation) == 1
    assert metrics["aggregate_label_block_count"] == 3
    assert metrics["minimum_anchor_spearman_brown_reliability"] == pytest.approx(1.0)
    assert metrics["maximum_anchor_spearman_brown_reliability"] == pytest.approx(1.0)


def test_spearman_brown_reports_negative_repeat_signal_as_zero() -> None:
    assert spearman_brown_reliability(-0.5, 3) == 0.0
