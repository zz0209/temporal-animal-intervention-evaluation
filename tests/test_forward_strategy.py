from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.evaluation.forward_strategy import (
    add_strictly_prior_candidate_history,
    balanced_variance_decomposition,
    build_forward_predictions,
    evaluate_raw_predictions,
)
from animal_intervention.evaluation.baseline_ranking import FEATURE_COLUMNS


def _forward_fixture() -> pd.DataFrame:
    rows = []
    values = {
        "2020-01-01": {"a": 0.2, "b": 0.8},
        "2020-01-02": {"a": 0.4, "b": 0.6},
        "2020-01-03": {"a": 0.9, "b": 0.1},
    }
    for time_index, (anchor_time, candidates) in enumerate(values.items()):
        for candidate_index, (candidate, target) in enumerate(candidates.items()):
            row = {
                "dataset_id": "fixture",
                "network_id": "cohort",
                "anchor_time": anchor_time,
                "candidate_id": candidate,
                "label_id": f"{anchor_time}|{candidate}",
                "robust_priority_percentile": target,
                "robust_intervention_value": target / 10,
            }
            for feature_index, column in enumerate(FEATURE_COLUMNS):
                row[column] = float(time_index + candidate_index + feature_index / 10)
            rows.append(row)
    return pd.DataFrame(rows)


def test_candidate_history_uses_only_strictly_earlier_anchors() -> None:
    frame = _forward_fixture()
    enriched = add_strictly_prior_candidate_history(frame)
    third_a = enriched.loc[
        enriched["anchor_time"].eq(pd.Timestamp("2020-01-03"))
        & enriched["candidate_id"].eq("a")
    ].iloc[0]
    assert third_a["stable_prior_score"] == pytest.approx(0.3)
    assert third_a["last_prior_score"] == pytest.approx(0.4)
    assert third_a["prior_observations"] == 2
    changed = frame.copy()
    changed.loc[changed["anchor_time"].eq("2020-01-03"), "robust_priority_percentile"] = 0.0
    changed_prior = add_strictly_prior_candidate_history(changed)
    changed_third_a = changed_prior.loc[
        changed_prior["anchor_time"].eq(pd.Timestamp("2020-01-03"))
        & changed_prior["candidate_id"].eq("a")
    ].iloc[0]
    assert changed_third_a["stable_prior_score"] == third_a["stable_prior_score"]


def test_forward_predictions_emit_only_eligible_test_anchors() -> None:
    predictions = build_forward_predictions(
        _forward_fixture(), min_prior_anchor_times=2, ridge_alpha=1.0
    )
    assert predictions["anchor_time"].nunique() == 1
    assert predictions["anchor_time"].iloc[0] == pd.Timestamp("2020-01-03")
    assert predictions["prior_anchor_times"].eq(2).all()
    assert predictions.filter(like="score_").notna().all().all()


def test_balanced_variance_decomposition_recovers_individual_signal() -> None:
    frame = pd.DataFrame(
        [
            ("a", "2020-01-01", 1.0),
            ("a", "2020-01-02", 1.0),
            ("b", "2020-01-01", 0.0),
            ("b", "2020-01-02", 0.0),
        ],
        columns=["candidate_id", "anchor_time", "robust_priority_percentile"],
    )
    frame["dataset_id"] = "fixture"
    frame["network_id"] = "cohort"
    result = balanced_variance_decomposition(frame).iloc[0]
    assert result["individual_fraction"] == 1.0
    assert result["anchor_fraction"] == 0.0
    assert result["individual_anchor_residual_fraction"] == 0.0


def test_oracle_raw_prediction_has_zero_error() -> None:
    predictions = build_forward_predictions(_forward_fixture())
    assert predictions["score_future_oracle"].equals(
        predictions["robust_intervention_value"]
    )
    metrics = evaluate_raw_predictions(predictions)
    oracle = metrics.loc[metrics["method"].eq("future_oracle")].iloc[0]
    assert oracle["mae"] == 0.0
    assert oracle["rmse"] == 0.0
