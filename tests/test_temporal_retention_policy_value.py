from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.experiments.temporal_retention_policy_value import _retention_metrics


def test_retention_is_one_for_identical_dyads() -> None:
    weights = {("a", "b"): 2.0, ("b", "c"): 1.0}
    result = _retention_metrics(weights, weights, ["a", "b", "c"], 0.34)
    assert result["adjusted_dyad_retention"] == 1.0
    assert result["weighted_cosine"] == pytest.approx(1.0)
    assert result["top_node_overlap"] == 1.0


def test_retention_reports_turnover_against_chance() -> None:
    left = {("a", "b"): 1.0}
    right = {("c", "d"): 1.0}
    result = _retention_metrics(left, right, ["a", "b", "c", "d"], 0.25)
    assert result["adjusted_dyad_retention"] < 0
    assert result["dyad_jaccard"] == 0
    assert result["weighted_cosine"] == 0


def test_binary_retention_flags_saturated_support_as_not_estimable() -> None:
    complete = {("a", "b"): 1.0, ("a", "c"): 2.0, ("b", "c"): 3.0}
    result = _retention_metrics(complete, complete, ["a", "b", "c"], 0.34)
    assert result["adjusted_dyad_retention_estimable"] == 0.0
    assert result["weighted_cosine"] == pytest.approx(1.0)


def test_retention_config_exists() -> None:
    assert pd.notna("configs/EXP-20260818-004_temporal_retention_policy_value.yaml")
