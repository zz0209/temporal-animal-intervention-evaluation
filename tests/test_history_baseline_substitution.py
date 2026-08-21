import numpy as np
import pandas as pd

from animal_intervention.experiments.contact_observation_robustness import WORLD_KEYS
from animal_intervention.experiments.history_baseline_substitution import (
    _decision_map,
    _pair_factorial,
    _pair_observation,
)
from animal_intervention.experiments.intervention_delivery_sensitivity import POLICY_KEYS


def _summary_row(method: str, mean: float, low: float, positive: int) -> dict:
    return {
        "epidemic_model": "temporal_sir",
        "detection_profile": "early_detection",
        "rewiring_fraction": 0.0,
        "method": method,
        "families": 5,
        "family_equal_mean": mean,
        "ci_low": low,
        "ci_high": mean + 0.02,
        "positive_families": positive,
    }


def test_decision_map_requires_relative_and_absolute_support() -> None:
    relative = pd.DataFrame(
        [_summary_row("contact_to_detected", 0.03, 0.01, 4)]
    ).drop(columns="method")
    absolute = pd.DataFrame(
        [
            _summary_row("contact_to_detected", 0.04, 0.01, 5),
            _summary_row("history_weight", 0.01, -0.01, 3),
        ]
    )
    result = _decision_map(
        relative,
        absolute,
        method="contact_to_detected",
        baseline="history_weight",
    )
    assert result.iloc[0]["decision"] == "override_with_detected_case_contacts"

    absolute.loc[absolute["method"].eq("contact_to_detected"), "ci_low"] = -0.001
    result = _decision_map(
        relative,
        absolute,
        method="contact_to_detected",
        baseline="history_weight",
    )
    assert result.iloc[0]["decision"] == "abstain_or_unresolved"


def test_decision_map_retains_supported_history_when_override_fails() -> None:
    relative = pd.DataFrame(
        [_summary_row("contact_to_detected", 0.01, -0.01, 3)]
    ).drop(columns="method")
    absolute = pd.DataFrame(
        [
            _summary_row("contact_to_detected", 0.02, 0.01, 5),
            _summary_row("history_weight", 0.02, 0.005, 4),
        ]
    )
    result = _decision_map(
        relative,
        absolute,
        method="contact_to_detected",
        baseline="history_weight",
    )
    assert result.iloc[0]["decision"] == "retain_history_weight"


def test_factorial_pairing_uses_identical_policy_world() -> None:
    key_values = {key: f"value_{key}" for key in POLICY_KEYS}
    key_values.update(
        {
            "action_delay_fraction": 0.1,
            "residual_contact_multiplier": 0.25,
            "secondary_case_sensitivity": 0.5,
            "false_positive_rate": 0.0,
            "rewiring_fraction": 1.0,
            "random_block": 0,
            "world_seed": 7,
        }
    )
    rows = []
    for method, value in (("history_weight", 0.02), ("contact_to_detected", 0.05)):
        rows.append(
            {
                **key_values,
                "epidemic_model": "temporal_sir",
                "system_family": "family",
                "analysis_cluster_id": "cluster",
                "method": method,
                "attack_rate_reduction": value,
            }
        )
    paired = _pair_factorial(
        pd.DataFrame(rows), "history_weight", "contact_to_detected"
    )
    assert len(paired) == 1
    assert np.isclose(paired.iloc[0]["increment"], 0.03)


def test_observation_pairing_keeps_profile_in_pairing_key() -> None:
    rows = []
    for profile, history, direct in (
        ("reference", 0.01, 0.04),
        ("joint_moderate", -0.01, 0.02),
    ):
        key_values = {key: f"value_{key}" for key in WORLD_KEYS}
        key_values.update({"random_block": 0, "world_seed": 11})
        for method, value in (
            ("history_weight", history),
            ("contact_to_detected", direct),
        ):
            rows.append(
                {
                    **key_values,
                    "observation_profile": profile,
                    "system_family": "family",
                    "analysis_cluster_id": "cluster",
                    "method": method,
                    "attack_rate_reduction": value,
                }
            )
    paired = _pair_observation(
        pd.DataFrame(rows), "history_weight", "contact_to_detected"
    )
    assert len(paired) == 2
    assert np.allclose(paired["increment"], [0.03, 0.03])
