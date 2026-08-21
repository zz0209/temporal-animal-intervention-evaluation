from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.surveillance_response_frontier import (
    WORLD_KEYS,
    _policy,
    _primary_contrasts,
)


def _worlds() -> pd.DataFrame:
    rows = []
    sizes = {
        _policy(0.05, 0.05): (30, 1, 1),
        _policy(0.10, 0.05): (25, 2, 1),
        _policy(0.05, 0.10): (27, 1, 2),
        _policy(0.10, 0.10): (20, 2, 2),
    }
    for policy, (final_size, sentinel_budget, response_budget) in sizes.items():
        rows.append(
            {
                "dataset_id": "toy",
                "network_id": "all",
                "anchor_id": "a1",
                "parameter_id": "p1",
                "epidemic_model": "temporal_sir",
                "random_block": 0,
                "initial_infected": "x",
                "world_seed": 7,
                "system_family": "toy_family",
                "analysis_cluster_id": "toy::a1",
                "population_size": 100,
                "policy": policy,
                "final_size": final_size,
                "sentinel_budget": sentinel_budget,
                "response_budget": response_budget,
                "response_capacity": response_budget,
            }
        )
    return pd.DataFrame(rows)


def test_policy_names_are_stable_percent_codes() -> None:
    assert _policy(0.05, 0.10) == "s05_r10"


def test_primary_contrasts_have_expected_sign_and_interaction() -> None:
    decision = {
        "reference_sentinel_fraction": 0.05,
        "expanded_sentinel_fraction": 0.10,
        "reference_response_fraction": 0.05,
        "expanded_response_fraction": 0.10,
    }
    contrasts = _primary_contrasts(_worlds(), decision).set_index("contrast")
    assert contrasts.loc["surveillance_doubling_value", "value"] == 0.05
    assert contrasts.loc["response_doubling_value", "value"] == 0.03
    assert contrasts.loc["capacity_complementarity", "value"] == 0.02
    assert contrasts["estimable"].all()
    assert set(WORLD_KEYS).issubset(contrasts.columns)


def test_equal_realized_integer_budget_is_nonestimable_not_zero_effect() -> None:
    decision = {
        "reference_sentinel_fraction": 0.05,
        "expanded_sentinel_fraction": 0.10,
        "reference_response_fraction": 0.05,
        "expanded_response_fraction": 0.10,
    }
    worlds = _worlds()
    expanded = worlds["policy"].eq(_policy(0.10, 0.05))
    worlds.loc[expanded, "sentinel_budget"] = 1
    contrasts = _primary_contrasts(worlds, decision).set_index("contrast")
    surveillance = contrasts.loc["surveillance_doubling_value"]
    assert not bool(surveillance["estimable"])
    assert pd.isna(surveillance["value_per_added_animal"])


def test_untriggered_response_world_remains_in_unconditional_capacity_estimand() -> None:
    decision = {
        "reference_sentinel_fraction": 0.05,
        "expanded_sentinel_fraction": 0.10,
        "reference_response_fraction": 0.05,
        "expanded_response_fraction": 0.10,
    }
    worlds = _worlds()
    worlds["final_size"] = 30
    worlds["response_budget"] = 0
    contrasts = _primary_contrasts(worlds, decision).set_index("contrast")
    response = contrasts.loc["response_doubling_value"]
    assert bool(response["estimable"])
    assert response["value"] == 0
    assert response["value_per_added_animal"] == 0
