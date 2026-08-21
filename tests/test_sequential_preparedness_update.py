from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.experiments.sequential_preparedness_update import (
    POLICIES,
    _classify,
    compute_contrasts,
)


def _worlds() -> pd.DataFrame:
    rows = []
    sizes = {
        "case_only": 20,
        "early_history": 16,
        "two_stage_history": 14,
        "two_stage_reactive": 12,
        "history_upfront": 11,
    }
    for policy in POLICIES:
        rows.append(
            {
                "dataset_id": "dataset",
                "network_id": "network",
                "anchor_id": "anchor",
                "parameter_id": "parameter",
                "epidemic_model": "temporal_sir",
                "random_block": 0,
                "initial_infected": "a",
                "world_seed": 1,
                "system_family": "family",
                "analysis_cluster_id": "cluster",
                "population_size": 100,
                "policy": policy,
                "final_size": sizes[policy],
            }
        )
    return pd.DataFrame(rows)


def test_compute_contrasts_uses_resource_matched_policy_differences() -> None:
    contrasts = compute_contrasts(_worlds()).set_index("contrast")["value"]
    assert contrasts.loc["preparedness_absolute"] == pytest.approx(0.04)
    assert contrasts.loc["update_information_gain"] == pytest.approx(0.02)
    assert contrasts.loc["staged_history_cost"] == pytest.approx(0.03)
    assert contrasts.loc["sequential_recovery"] == pytest.approx(-0.01)


def test_classification_requires_update_absolute_and_recovery_gates() -> None:
    metrics = [
        "preparedness_absolute",
        "second_history_value",
        "second_reactive_value",
        "update_information_gain",
        "staged_history_cost",
        "sequential_recovery",
    ]
    summary_rows = []
    family_rows = []
    for metric in metrics:
        value = 0.02
        summary_rows.append(
            {
                "epidemic_model": "temporal_sir",
                "contrast": metric,
                "families": 5,
                "family_equal_mean": value,
                "ci_low": 0.01,
                "ci_high": 0.03,
            }
        )
        for family in range(5):
            family_rows.append(
                {
                    "epidemic_model": "temporal_sir",
                    "contrast": metric,
                    "system_family": f"family_{family}",
                    "mean_value": value,
                }
            )
    decision = _classify(pd.DataFrame(summary_rows), pd.DataFrame(family_rows))
    assert decision.loc[0, "decision"] == "sequential_update_supported"

    summary_rows[-1]["ci_low"] = -0.01
    decision = _classify(pd.DataFrame(summary_rows), pd.DataFrame(family_rows))
    assert decision.loc[0, "decision"] == "useful_update_but_not_timing_recovery"
