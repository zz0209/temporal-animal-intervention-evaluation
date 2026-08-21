from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.model_shift_robustness import _parse_set, _robust_selections


def test_parse_set_handles_empty_and_sorted_signatures() -> None:
    assert _parse_set("") == set()
    assert _parse_set("b|a") == {"a", "b"}


def test_robust_selection_minimizes_worst_normalized_regret() -> None:
    context = {
        "dataset_id": "example",
        "network_id": "all",
        "system_family": "family",
        "anchor_id": "anchor_001",
        "anchor_time": pd.Timestamp("2020-01-01"),
        "initial_infected": "seed",
    }
    rows = []
    values = {
        "temporal_sir": {"a": 1.0, "b": 0.0, "c": 0.6},
        "temporal_seir_erlang": {"a": 0.0, "b": 1.0, "c": 0.6},
    }
    for model, mapping in values.items():
        for signature, value in mapping.items():
            rows.append({**context, "epidemic_model": model, "set_signature": signature, "set_size": 1, "value": value})
    scores = pd.DataFrame(rows)
    selections = pd.DataFrame(
        [
            {**context, "epidemic_model": "temporal_sir", "budget": 1, "history_exact": "a", "stable": "c"},
            {**context, "epidemic_model": "temporal_seir_erlang", "budget": 1, "history_exact": "b", "stable": "c"},
        ]
    )
    result = _robust_selections(scores, selections).iloc[0]
    assert result.robust_plan == "c"
    assert result.model_specific_agreement is False or not bool(result.model_specific_agreement)
