import pandas as pd

from animal_intervention.experiments.historical_set_planning import _all_subsets, _structure_metrics


def test_all_subsets_respects_budget() -> None:
    subsets = _all_subsets(("a", "b", "c"), 2)
    assert len(subsets) == 7
    assert all(len(item) <= 2 for item in subsets)


def test_structure_metric_detects_replicated_complementarity() -> None:
    rows = []
    values = {"": 0.0, "a": 0.1, "b": 0.1, "a|b": 0.5, "c": 0.0, "a|c": 0.1, "b|c": 0.1}
    for block in range(4):
        for signature, value in values.items():
            rows.append({"history_block": block, "set_signature": signature, "value": value})
    metrics = _structure_metrics(pd.DataFrame(rows), ("a", "b", "c"), 2)
    assert metrics["eligible_inequalities"] > 0
    assert metrics["replicated_violations"] > 0
    assert metrics["replicated_violation_rate"] > 0
