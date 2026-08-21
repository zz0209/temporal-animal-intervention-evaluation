from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from animal_intervention.centrality.history_features import build_history_features
from animal_intervention.data.contract import CanonicalDataset, DatasetMetadata, empty_table
from animal_intervention.evaluation.baseline_ranking import (
    evaluate_baseline_scores,
    fit_baseline_scores,
    fit_feature_ablation_scores,
)
from animal_intervention.experiments.history_baseline_pilot import _write_csv


def _fixture_dataset() -> CanonicalDataset:
    events = empty_table("dyadic_events")
    events = pd.DataFrame(
        [
            {
                "dataset_id": "fixture",
                "event_id": "e1",
                "source_id": "a",
                "target_id": "b",
                "start_time": pd.Timestamp("2020-01-01 12:00:00"),
                "end_time": pd.Timestamp("2020-01-01 12:01:00"),
                "duration_seconds": 60.0,
                "event_representation": "interval",
                "edge_semantics": "sensor_proximity",
                "measurement_type": "duration",
                "measurement_value": 60.0,
                "measurement_unit": "second",
                "location_id": "x",
                "directed": False,
                "native_time_resolution_seconds": 1.0,
                "source_record_id": "1",
                "quality_flag": "ok",
                "attributes_json": "{}",
            },
            {
                "dataset_id": "fixture",
                "event_id": "future",
                "source_id": "a",
                "target_id": "c",
                "start_time": pd.Timestamp("2020-01-03"),
                "end_time": pd.Timestamp("2020-01-03 00:01:00"),
                "duration_seconds": 60.0,
                "event_representation": "interval",
                "edge_semantics": "sensor_proximity",
                "measurement_type": "duration",
                "measurement_value": 60.0,
                "measurement_unit": "second",
                "location_id": "x",
                "directed": False,
                "native_time_resolution_seconds": 1.0,
                "source_record_id": "2",
                "quality_flag": "ok",
                "attributes_json": "{}",
            },
            {
                "dataset_id": "fixture",
                "event_id": "behavior",
                "source_id": "a",
                "target_id": "c",
                "start_time": pd.Timestamp("2020-01-01 13:00:00"),
                "end_time": pd.Timestamp("2020-01-01 13:01:00"),
                "duration_seconds": 60.0,
                "event_representation": "interval",
                "edge_semantics": "direct_behavior",
                "measurement_type": "duration",
                "measurement_value": 60.0,
                "measurement_unit": "second",
                "location_id": "x",
                "directed": False,
                "native_time_resolution_seconds": 1.0,
                "source_record_id": "3",
                "quality_flag": "ok",
                "attributes_json": "{}",
            },
            {
                "dataset_id": "fixture",
                "event_id": "duplicate",
                "source_id": "a",
                "target_id": "b",
                "start_time": pd.Timestamp("2020-01-01 12:00:00"),
                "end_time": pd.Timestamp("2020-01-01 12:01:00"),
                "duration_seconds": 60.0,
                "event_representation": "interval",
                "edge_semantics": "sensor_proximity",
                "measurement_type": "duration",
                "measurement_value": 60.0,
                "measurement_unit": "second",
                "location_id": "x",
                "directed": False,
                "native_time_resolution_seconds": 1.0,
                "source_record_id": "4",
                "quality_flag": "exact_duplicate_source_record",
                "attributes_json": "{}",
            },
        ]
    )
    return CanonicalDataset(
        metadata=DatasetMetadata(
            dataset_id="fixture", title="fixture", adapter_name="FixtureAdapter"
        ),
        dyadic_events=events,
    )


def test_history_features_exclude_post_anchor_events() -> None:
    labels = pd.DataFrame(
        {
            "dataset_id": ["fixture"] * 3,
            "network_id": ["all"] * 3,
            "anchor_time": [pd.Timestamp("2020-01-02")] * 3,
            "history_start": [pd.Timestamp("2020-01-01")] * 3,
            "candidate_id": ["a", "b", "c"],
            "eligible_population": [3] * 3,
        }
    )
    features = build_history_features(_fixture_dataset(), labels).set_index("candidate_id")
    assert features.loc["a", "activity_count"] == 1
    assert features.loc["a", "eligible_partner_fraction"] == 0.5
    assert features.loc["a", "observed_partner_count"] == 1
    assert features.loc["c", "activity_count"] == 0


def test_partner_features_separate_eligible_and_all_observed_partners() -> None:
    labels = pd.DataFrame(
        {
            "dataset_id": ["fixture"] * 2,
            "network_id": ["all"] * 2,
            "anchor_time": [pd.Timestamp("2020-01-04")] * 2,
            "history_start": [pd.Timestamp("2020-01-03")] * 2,
            "candidate_id": ["a", "b"],
            "eligible_population": [2] * 2,
        }
    )
    features = build_history_features(_fixture_dataset(), labels).set_index("candidate_id")
    assert features.loc["a", "eligible_partner_fraction"] == 0.0
    assert features.loc["a", "observed_partner_count"] == 1.0
    assert features["eligible_partner_fraction"].between(0.0, 1.0).all()


def test_history_features_reject_mismatched_eligible_population() -> None:
    labels = pd.DataFrame(
        {
            "dataset_id": ["fixture"] * 2,
            "network_id": ["all"] * 2,
            "anchor_time": [pd.Timestamp("2020-01-02")] * 2,
            "history_start": [pd.Timestamp("2020-01-01")] * 2,
            "candidate_id": ["a", "b"],
            "eligible_population": [3] * 2,
        }
    )
    with pytest.raises(ValueError, match="eligible population"):
        build_history_features(_fixture_dataset(), labels)


def test_leave_one_system_scores_and_metrics_are_complete() -> None:
    rows = []
    for dataset_id in ("system_a", "system_b"):
        for candidate, value in zip(("a", "b", "c"), (0.1, 0.2, 0.3)):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "network_id": "all",
                    "anchor_time": pd.Timestamp("2020-01-02"),
                    "candidate_id": candidate,
                    "label_id": f"{dataset_id}:{candidate}",
                    "robust_intervention_value": value,
                    "robust_priority_percentile": value / 0.3,
                    "event_rate_per_day": value,
                    "contact_opportunity_rate": value,
                    "weighted_exposure_rate": value,
                    "eligible_partner_fraction": value,
                    "observed_partner_count": value,
                    "location_count": value,
                    "mean_group_size": value,
                    "recency_score": value,
                    "recent_activity_fraction": value,
                    "active_span_fraction": value,
                }
            )
    scored = fit_baseline_scores(pd.DataFrame(rows))
    assert scored["score_ridge_loso"].notna().all()
    context, family = evaluate_baseline_scores(scored, top_fraction=1 / 3)
    assert len(context) == 14
    assert len(family) == 14
    assert context.loc[context["method"].eq("activity"), "spearman"].eq(1.0).all()
    ablation = fit_feature_ablation_scores(pd.DataFrame(rows))
    assert ablation["score_ridge_static_summary_loso"].notna().all()
    assert ablation["score_ridge_temporal_summary_loso"].notna().all()


def test_constant_scores_are_reported_as_no_information_with_tie_aware_value() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["system"] * 4,
            "network_id": ["all"] * 4,
            "anchor_time": [pd.Timestamp("2020-01-02")] * 4,
            "candidate_id": list("abcd"),
            "robust_intervention_value": [0.0, 1.0, 2.0, 3.0],
            "robust_priority_percentile": [0.25, 0.5, 0.75, 1.0],
            "score_constant": [1.0] * 4,
        }
    )
    context, _ = evaluate_baseline_scores(frame, top_fraction=0.5)
    assert context.loc[0, "spearman"] == 0.0
    assert not context.loc[0, "score_has_variation"]
    assert context.loc[0, "selected_mean_value"] == 1.5


def test_random_baseline_uses_exact_selection_expectation() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["system"] * 4,
            "network_id": ["all"] * 4,
            "anchor_time": [pd.Timestamp("2020-01-02")] * 4,
            "candidate_id": list("abcd"),
            "robust_intervention_value": [0.0, 1.0, 2.0, 3.0],
            "robust_priority_percentile": [0.25, 0.5, 0.75, 1.0],
            "score_random": [0.1, 0.2, 0.3, 0.4],
        }
    )
    context, _ = evaluate_baseline_scores(frame, top_fraction=0.5)
    row = context.iloc[0]
    assert row["selection_evaluation"] == "analytic_random_expectation"
    assert row["spearman"] == 0.0
    assert row["selected_mean_value"] == 1.5
    assert row["value_capture_above_random"] == 0.0
    assert row["top_set_overlap"] == 0.5


def test_csv_writer_normalizes_equivalent_timestamp_representations(tmp_path) -> None:
    path = tmp_path / "timestamps.csv"
    frame = pd.DataFrame(
        {
            "anchor_time": [pd.Timestamp("2020-01-02"), date(2020, 1, 2)],
            "value": [1, 2],
        }
    )
    _write_csv(frame, path)
    restored = pd.read_csv(path)
    assert restored["anchor_time"].nunique() == 1
    assert restored.loc[0, "anchor_time"] == "2020-01-02T00:00:00"
