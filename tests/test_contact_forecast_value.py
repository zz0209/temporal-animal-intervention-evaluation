import pandas as pd

from animal_intervention.experiments.contact_forecast_value import (
    _model_interaction,
    _leave_one_family_out,
    _planning_budget,
    _select_signature,
)


def test_future_selection_uses_only_declared_blocks() -> None:
    frame = pd.DataFrame({
        "future_block": [0, 0, 1, 1, 2, 2],
        "set_signature": ["a|b", "a|c"] * 3,
        "value": [1.0, 0.0, 0.8, 0.1, 0.0, 2.0],
    })
    assert _select_signature(frame, [0, 1]) == "a|b"


def test_pair_budget_is_structural_and_preserves_small_groups() -> None:
    decision = {
        "minimum_budget": 1,
        "response_budget_fraction": 0.05,
        "minimum_population_for_pair": 10,
        "maximum_planning_budget": 2,
    }
    assert _planning_budget(26, decision) == 2
    assert _planning_budget(13, decision) == 2
    assert _planning_budget(9, decision) == 1


def test_model_interaction_preserves_family_pairing() -> None:
    family = pd.DataFrame({
        "contrast": ["test"] * 4,
        "system_family": ["a", "b", "a", "b"],
        "epidemic_model": ["temporal_sir", "temporal_sir", "temporal_seir_erlang", "temporal_seir_erlang"],
        "mean_value": [0.1, 0.2, 0.4, 0.6],
    })
    result = _model_interaction(family, bootstrap_replicates=100, seed=7)
    assert abs(result.loc[0, "family_equal_mean"] - 0.35) < 1e-12
    assert result.loc[0, "positive_families"] == 2


def test_leave_one_family_out_uses_independent_family_means() -> None:
    family = pd.DataFrame({
        "epidemic_model": ["temporal_sir"] * 3,
        "contrast": ["test"] * 3,
        "system_family": ["a", "b", "c"],
        "mean_value": [0.1, 0.2, 0.6],
    })
    detail, summary = _leave_one_family_out(family)
    assert len(detail) == 3
    assert abs(summary.loc[0, "minimum_leave_one_out_mean"] - 0.15) < 1e-12
    assert abs(summary.loc[0, "maximum_leave_one_out_mean"] - 0.4) < 1e-12
