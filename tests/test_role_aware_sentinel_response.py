from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from animal_intervention.experiments.role_aware_sentinel_response import _detection_metrics


def _result(rows: list[tuple[str, str]], final_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        event_log=pd.DataFrame(rows, columns=["time", "node_id"]).assign(event="infection"),
        final_size=final_size,
    )


def test_detection_burden_counts_infections_through_first_sentinel() -> None:
    result = _result(
        [("2020-01-01 00:00", "a"), ("2020-01-01 01:00", "b"), ("2020-01-01 01:00", "c")],
        final_size=3,
    )
    metrics = _detection_metrics(result, {"b"}, population_size=10, threshold_fraction=0.2)
    assert metrics["detected"]
    assert metrics["detection_burden"] == 3
    assert not metrics["early_detection"]


def test_undetected_outbreak_is_censored_at_final_burden() -> None:
    result = _result([("2020-01-01 00:00", "a"), ("2020-01-01 01:00", "b")], final_size=2)
    metrics = _detection_metrics(result, {"z"}, population_size=10, threshold_fraction=0.2)
    assert not metrics["detected"]
    assert pd.isna(metrics["detection_time"])
    assert metrics["detection_burden"] == 2
    assert metrics["detection_burden_rate"] == 0.2
