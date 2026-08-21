from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.immediate_case_targeting import (
    _diversity_repair_gate,
    _diversity_contrasts,
    _selection_geometry,
)


def test_diversity_repair_gate_requires_both_contrasts_and_models() -> None:
    rows = []
    for model in ["temporal_sir", "temporal_seir_erlang"]:
        rows.extend(
            [
                {
                    "epidemic_model": model,
                    "contrast": "shortlist_diversity_vs_random",
                    "decision": "strong",
                },
                {
                    "epidemic_model": model,
                    "contrast": "shortlist_diversity_vs_top_history",
                    "decision": "directional",
                },
            ]
        )
    gate = _diversity_repair_gate(pd.DataFrame(rows))
    assert gate["overall"] == "directional"
    rows[-1]["decision"] = "unsupported"
    gate = _diversity_repair_gate(pd.DataFrame(rows))
    assert gate["overall"] == "unsupported"
from animal_intervention.experiments.response_targeting_increment import WORLD_KEYS


def test_diversity_contrasts_preserve_reference_direction() -> None:
    common = {
        "dataset_id": "d",
        "network_id": "n",
        "anchor_id": "a",
        "parameter_id": "p",
        "epidemic_model": "temporal_sir",
        "random_block": 0,
        "initial_infected": "i",
        "world_seed": 1,
        "system_family": "f",
        "analysis_cluster_id": "c",
        "population_size": 100,
        "response_budget": 2,
    }
    final_sizes = {
        "case_only": 30.0,
        "history_weight": 20.0,
        "random": 24.0,
        "history_shortlist_diverse": 18.0,
        "pure_history_coverage": 22.0,
    }
    worlds = pd.DataFrame(
        [
            {**common, "response_method": method, "final_size": value}
            for method, value in final_sizes.items()
        ]
    )
    assert not worlds[WORLD_KEYS].isna().any().any()
    contrasts = _diversity_contrasts(worlds).set_index("contrast")["value"]
    assert contrasts["shortlist_diversity_vs_random"] == 0.06
    assert contrasts["shortlist_diversity_vs_top_history"] == 0.02
    assert contrasts["pure_coverage_vs_random"] == 0.02


def test_diversity_contrasts_are_empty_without_diversity_arms() -> None:
    worlds = pd.DataFrame({"response_method": ["random", "history_weight"]})
    assert _diversity_contrasts(worlds).empty


def test_selection_geometry_excludes_confirmed_index() -> None:
    common = {
        "dataset_id": "d",
        "network_id": "n",
        "anchor_id": "a",
        "parameter_id": "p",
        "epidemic_model": "temporal_sir",
        "random_block": 0,
        "initial_infected": "i",
        "world_seed": 1,
        "response_budget": 2,
        "system_family": "f",
    }
    nodes = {
        "history_weight": "i|a|b",
        "history_shortlist_diverse": "i|a|c",
        "pure_history_coverage": "i|c|d",
    }
    worlds = pd.DataFrame(
        [
            {**common, "response_method": method, "response_nodes": value}
            for method, value in nodes.items()
        ]
    )
    row = _selection_geometry(worlds).iloc[0]
    assert bool(row.multi_node_budget)
    assert not bool(row.shortlist_equals_top)
    assert row.shortlist_top_jaccard == 1 / 3
