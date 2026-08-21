import pandas as pd

from animal_intervention.experiments.set_value_coverage import _family_gate_summary
from animal_intervention.experiments.set_value_pilot import _perturb_set
from animal_intervention.experiments.set_value_precision import _context_reliability


def test_perturb_set_preserves_budget_when_replacements_exist() -> None:
    base = ("a", "b")
    perturbed = _perturb_set(base, ["a", "b", "c"], 0.5, 7, "test")

    assert len(perturbed) == len(base)
    assert set(perturbed).issubset({"a", "b", "c"})
    assert set(perturbed) != set(base)


def test_perturb_set_returns_base_when_no_replacement_exists() -> None:
    base = ("a", "b")

    assert _perturb_set(base, ["a", "b"], 0.5, 7, "test") == base


def test_family_gate_excludes_singletons_and_requires_multiple_anchors() -> None:
    contexts = pd.DataFrame(
        [
            {
                "dataset_id": "a",
                "network_id": "all",
                "anchor_id": "one",
                "system_family": "family_a",
                "budget": 2,
                "distinct_values": 2,
                "value_spread": 0.1,
            },
            {
                "dataset_id": "a",
                "network_id": "all",
                "anchor_id": "two",
                "system_family": "family_a",
                "budget": 2,
                "distinct_values": 2,
                "value_spread": 0.2,
            },
            {
                "dataset_id": "b",
                "network_id": "all",
                "anchor_id": "one",
                "system_family": "family_b",
                "budget": 1,
                "distinct_values": 3,
                "value_spread": 0.3,
            },
        ]
    )

    summary = _family_gate_summary(
        contexts,
        minimum_variable_fraction=0.1,
        minimum_variable_anchors=2,
    ).set_index("system_family")

    assert bool(summary.loc["family_a", "qualifies"])
    assert not bool(summary.loc["family_b", "qualifies"])
    assert summary.loc["family_b", "multinode_contexts"] == 0


def test_conditional_reliability_recovers_repeated_set_order() -> None:
    rows = []
    values = {"a": 0.3, "b": 0.2, "c": 0.0}
    for block in range(4):
        for signature, value in values.items():
            rows.append(
                {
                    "dataset_id": "dataset",
                    "network_id": "all",
                    "anchor_id": "anchor",
                    "parameter_id": "parameter",
                    "detection_profile": "early",
                    "evidence_profile": "trigger_only",
                    "budget_fraction": 0.1,
                    "initial_infected": "seed",
                    "observation_random_block": 0,
                    "observation_world_seed": 7,
                    "continuation_block": block,
                    "set_signature": signature,
                    "set_attack_rate_value": value,
                    "system_family": "family",
                    "budget": 2,
                }
            )
    evaluation = {
        "top_fraction": 0.25,
        "maximum_normalized_cross_half_regret": 0.5,
        "minimum_top_overlap_above_chance": 0.05,
    }

    result = _context_reliability(pd.DataFrame(rows), evaluation).iloc[0]

    assert result.split_half_spearman == 1.0
    assert result.normalized_cross_half_regret == 0.0
    assert bool(result.reproducible)
