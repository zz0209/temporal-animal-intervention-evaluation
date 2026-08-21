import pandas as pd

from animal_intervention.experiments.policy_mechanism_taxonomy import (
    _spearman,
    build_paired_worlds,
    classify_cells,
    variance_decomposition,
)


def test_spearman_without_optional_scipy_dependency() -> None:
    assert _spearman(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0])) == -1.0


def test_variance_decomposition_closes_for_balanced_design() -> None:
    rows = []
    for family_index, family in enumerate(["a", "b"]):
        for model in ["temporal_sir", "temporal_seir_erlang"]:
            for timing in ["early_detection", "delayed_detection"]:
                for rewiring in [0.0, 1.0]:
                    rows.append(
                        {
                            "system_family": family,
                            "epidemic_model": model,
                            "detection_profile": timing,
                            "rewiring_fraction": rewiring,
                            "value": family_index + rewiring + (model == "temporal_seir_erlang"),
                        }
                    )
    result = variance_decomposition(pd.DataFrame(rows), "value")
    assert abs(result["variance_fraction"].sum() - 1.0) < 1e-10


def test_classification_uses_whole_family_deletions() -> None:
    decisions = pd.DataFrame(
        [
            {"epidemic_model": "temporal_sir", "detection_profile": "early_detection", "rewiring_fraction": 0.0, "decision": "retain_history_weight"},
            {"epidemic_model": "temporal_sir", "detection_profile": "early_detection", "rewiring_fraction": 1.0, "decision": "override_with_detected_case_contacts"},
        ]
    )
    resilience = decisions[["epidemic_model", "detection_profile", "rewiring_fraction"]].copy()
    resilience["same_decision_folds"] = [5, 2]
    resilience["override_folds"] = [0, 2]
    resilience["omission_folds"] = 5
    resilience["resilience_status"] = ["decision_invariant", "fragile"]
    result = classify_cells(decisions, resilience)
    assert result["evidence_class"].tolist() == ["robust_history", "pooled_override_only"]


def test_world_pairing_builds_target_disagreement() -> None:
    base = {
        "dataset_id": "d", "network_id": "n", "system_family": "f", "analysis_cluster_id": "c",
        "anchor_id": "a", "anchor_time": "2020-01-01", "horizon_end": "2020-01-02", "parameter_id": "p",
        "epidemic_model": "temporal_sir", "detection_profile": "early_detection", "rewiring_fraction": 0.0,
        "random_block": 0, "initial_infected": "i", "world_seed": 1, "population_size": 10,
        "natural_final_size": 5, "standard_final_size": 4, "additional_budget": 2,
        "detected_actionable_infectious_fraction": 1.0, "detected_secondary_recall": 1.0,
        "selected_infected_fraction": 0.5, "additional_removed_hazard_fraction": 0.2,
        "additional_rewired_hazard_fraction": 0.0, "augmented_final_size": 3, "attack_rate_reduction": 0.1,
    }
    direct = {**base, "method": "contact_to_detected", "additional_targets": "a|b"}
    history = {**base, "method": "history_weight", "additional_targets": "b|c", "attack_rate_reduction": 0.05}
    paired = build_paired_worlds(pd.DataFrame([direct, history]))
    assert paired.loc[0, "target_jaccard"] == 1 / 3
    assert paired.loc[0, "direct_minus_history"] == 0.05
