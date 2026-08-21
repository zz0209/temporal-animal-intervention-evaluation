from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.experiments.information_accrual_value import decompose_policy_value


def _row(profile: str, method: str, final_size: int, standard_size: int = 8) -> dict[str, object]:
    return {
        "dataset_id": "demo",
        "network_id": "all",
        "system_family": "demo_family",
        "analysis_cluster_id": "demo::a",
        "anchor_id": "a",
        "parameter_id": "p",
        "epidemic_model": "temporal_sir",
        "detection_profile": profile,
        "detection_fraction": 0.25 if profile == "evidence_025" else 0.75,
        "action_delay_fraction": 0.1,
        "residual_contact_multiplier": 0.25,
        "secondary_case_sensitivity": 0.5,
        "false_positive_rate": 0.0,
        "rewiring_fraction": 0.0,
        "rewiring_mode": "none",
        "budget_fraction": 0.05,
        "random_block": 0,
        "initial_infected": "x",
        "world_seed": 7,
        "population_size": 10,
        "method": method,
        "augmented_final_size": final_size,
        "standard_final_size": standard_size,
        "natural_final_size": 9,
        "detected_cases": 1,
        "case_contact_evidence_mass": 2.0,
        "case_contact_evidence_nodes": 2,
        "case_contact_evidence_node_fraction": 0.2,
    }


def test_value_decomposition_identity() -> None:
    rows = [
        _row("evidence_025", "history_weight", 6),
        _row("evidence_025", "contact_to_detected", 5),
        _row("evidence_075", "history_weight", 8),
        _row("evidence_075", "contact_to_detected", 4),
    ]
    result = decompose_policy_value(pd.DataFrame(rows), "evidence_025")
    late = result.loc[result["detection_profile"].eq("evidence_075")].iloc[0]
    assert late["information_gain"] == pytest.approx(0.4)
    assert late["delay_cost"] == pytest.approx(0.2)
    assert late["net_wait_value"] == pytest.approx(0.2)
    assert late["information_gain"] - late["delay_cost"] == pytest.approx(late["net_wait_value"])


def test_earliest_delay_cost_is_zero() -> None:
    rows = [
        _row("evidence_025", "history_weight", 6),
        _row("evidence_025", "contact_to_detected", 5),
    ]
    result = decompose_policy_value(pd.DataFrame(rows), "evidence_025")
    assert result.iloc[0]["delay_cost"] == pytest.approx(0.0)
