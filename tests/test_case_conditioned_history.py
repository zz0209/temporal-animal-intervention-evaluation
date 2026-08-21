from __future__ import annotations

import pandas as pd
from pathlib import Path

from animal_intervention.experiments.immediate_case_targeting import (
    _case_conditioned_history_sets,
    _checkpoint_fingerprint,
    _ring_contrasts,
    _ring_gates,
    _serialize_random_final_sizes,
)


def test_random_final_sizes_are_serialized_without_averaging() -> None:
    assert _serialize_random_final_sizes([1, 7.0, 3]) == "1|7|3"


def test_checkpoint_fingerprint_changes_with_source(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    source = tmp_path / "experiment.py"
    config.write_text("seed: 1", encoding="utf-8")
    source.write_text("value = 1", encoding="utf-8")
    first = _checkpoint_fingerprint(config, source)
    source.write_text("value = 2", encoding="utf-8")

    assert _checkpoint_fingerprint(config, source) != first
from animal_intervention.transmission import ExposureStream


def test_recent_ring_uses_time_while_static_ring_uses_total_mass() -> None:
    anchor = pd.Timestamp("2020-01-02T00:00:00")
    history = ExposureStream(
        dataset_id="fixture",
        population_nodes=("i", "a", "b", "c"),
        dyadic_exposures=pd.DataFrame(
            {
                "exposure_id": ["old_long", "recent_short"],
                "source_id": ["i", "i"],
                "target_id": ["a", "b"],
                "start_time": [
                    anchor - pd.Timedelta(hours=23),
                    anchor - pd.Timedelta(minutes=10),
                ],
                "end_time": [
                    anchor - pd.Timedelta(hours=22, minutes=40),
                    anchor - pd.Timedelta(minutes=5),
                ],
                "hazard_rate_multiplier": [1.0, 1.0],
                "directed": [False, False],
            }
        ),
    )
    stable = pd.DataFrame(
        {
            "candidate_id": ["i", "a", "b", "c"],
            "stable_score": [0.0, 0.2, 0.1, 0.9],
        }
    )
    sets = dict(
        _case_conditioned_history_sets(
            history_stream=history,
            stable_scores=stable,
            eligible={"i", "a", "b", "c"},
            initial="i",
            budget=1,
            history_start=anchor - pd.Timedelta(days=1),
            anchor_time=anchor,
            recency_half_life=pd.Timedelta(hours=1),
            seed=17,
        )
    )
    assert sets["past_weight_ring"] == {"a"}
    assert sets["past_recent_ring"] == {"b"}


def test_ring_contrasts_use_avoided_attack_rate_direction() -> None:
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
    sizes = {
        "case_only": 30.0,
        "history_weight": 22.0,
        "random": 24.0,
        "past_weight_ring": 21.0,
        "past_recent_ring": 18.0,
    }
    worlds = pd.DataFrame(
        [
            {**common, "response_method": method, "final_size": value}
            for method, value in sizes.items()
        ]
    )
    values = _ring_contrasts(worlds).set_index("contrast")["value"]
    assert values["recent_ring_vs_random"] == 0.06
    assert values["recent_ring_vs_stable"] == 0.04
    assert values["recent_ring_vs_static_ring"] == 0.03
    assert values["static_ring_vs_random"] == 0.03


def test_ring_gate_requires_both_models_and_operational_comparators() -> None:
    rows = []
    for model in ["temporal_sir", "temporal_seir_erlang"]:
        for contrast in [
            "recent_ring_vs_random",
            "recent_ring_vs_stable",
            "recent_ring_vs_static_ring",
            "static_ring_vs_random",
        ]:
            rows.append(
                {
                    "epidemic_model": model,
                    "contrast": contrast,
                    "decision": "strong",
                }
            )
    decisions = pd.DataFrame(rows)
    gates = _ring_gates(decisions)
    assert gates["operational_case_conditioning"]["overall"] == "strong"
    assert gates["temporal_recency_increment"]["overall"] == "strong"
    decisions.loc[
        decisions["contrast"].eq("recent_ring_vs_stable")
        & decisions["epidemic_model"].eq("temporal_sir"),
        "decision",
    ] = "unsupported"
    gates = _ring_gates(decisions)
    assert gates["operational_case_conditioning"]["overall"] == "unsupported"
