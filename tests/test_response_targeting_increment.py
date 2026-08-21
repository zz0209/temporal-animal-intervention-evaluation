from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.response_targeting_increment import (
    _compute_decomposition,
    _random_distribution_diagnostics,
    _random_response_targets,
)


def test_random_response_targets_are_deterministic_and_exclude_detected_nodes() -> None:
    eligible = {"a", "b", "c", "d"}
    first = _random_response_targets(eligible, budget=2, seed=17, excluded={"a"})
    second = _random_response_targets(eligible, budget=2, seed=17, excluded={"a"})
    assert first == second
    assert len(first) == 2
    assert "a" not in first


def test_decomposition_separates_capacity_and_targeting() -> None:
    rows = []
    for method, final_size in [("case_only", 8), ("random", 6), ("history_weight", 5)]:
        rows.append(
            {
                "dataset_id": "d",
                "network_id": "n",
                "anchor_id": "a",
                "parameter_id": "p",
                "epidemic_model": "temporal_sir",
                "random_block": 0,
                "initial_infected": "x",
                "world_seed": 1,
                "system_family": "f",
                "analysis_cluster_id": "c",
                "population_size": 10,
                "response_method": method,
                "final_size": final_size,
            }
        )
    result = _compute_decomposition(pd.DataFrame(rows)).set_index("contrast")["value"]
    assert result["capacity_increment"] == 0.2
    assert result["targeting_increment"] == 0.1
    assert result["total_history_increment"] == 0.3


def test_random_distribution_uses_midrank_ties_and_world_level_average() -> None:
    keys = {
        "dataset_id": "d",
        "network_id": "n",
        "anchor_id": "a",
        "parameter_id": "p",
        "epidemic_model": "temporal_sir",
        "random_block": 0,
        "initial_infected": "x",
        "world_seed": 1,
    }
    history = pd.DataFrame(
        [
            {
                **keys,
                "system_family": "f",
                "analysis_cluster_id": "c",
                "population_size": 10,
                "response_method": "history_weight",
                "final_size": 4,
            }
        ]
    )
    random = pd.DataFrame(
        [
            {
                **keys,
                "system_family": "f",
                "analysis_cluster_id": "c",
                "population_size": 10,
                "target_replicate": replicate,
                "final_size": final_size,
            }
            for replicate, final_size in enumerate([3, 4, 7, 8])
        ]
    )
    pairwise, world = _random_distribution_diagnostics(random, history)
    assert len(pairwise) == 4
    assert world.loc[0, "history_percentile"] == 0.625
    assert world.loc[0, "mean_targeting_increment"] == 0.15
