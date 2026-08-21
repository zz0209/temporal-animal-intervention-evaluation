import pandas as pd

from animal_intervention.experiments.evidence_synthesis import (
    _apply_filters,
    _effect_status,
    _evaluate_claim,
    _safety_frontier,
)


def test_effect_status_separates_supported_and_possible_harm() -> None:
    assert _effect_status(0.1, 0.01, 0.2) == "supported_benefit"
    assert _effect_status(-0.1, -0.2, 0.01) == "possible_harm"
    assert _effect_status(-0.1, -0.2, -0.01) == "supported_harm"


def test_filters_accept_scalar_and_list_values() -> None:
    frame = pd.DataFrame({"profile": ["early", "late", "early"], "level": [0.0, 0.5, 1.0]})
    selected = _apply_filters(frame, {"profile": "early", "level": [0.0, 1.0]})
    assert selected["level"].tolist() == [0.0, 1.0]


def test_claim_gate_uses_cell_fraction_and_family_direction() -> None:
    ledger = pd.DataFrame(
        {
            "experiment_key": ["phase"] * 3,
            "experiment_id": ["EXP"] * 3,
            "domain": ["disease"] * 3,
            "estimand": ["relative"] * 3,
            "method": ["direct"] * 3,
            "family_equal_mean": [0.1, 0.2, 0.3],
            "ci_low": [-0.1, 0.01, 0.02],
            "ci_high": [0.2, 0.3, 0.4],
            "positive_families": [2, 4, 5],
        }
    )
    claim = {
        "claim_id": "c",
        "claim_text": "test",
        "source": "phase",
        "estimand": "relative",
        "method": "direct",
        "filters": {},
        "gate": {
            "require_point_positive": True,
            "require_ci_low_positive": False,
            "minimum_positive_families": 4,
            "minimum_passing_cell_fraction": 2 / 3,
        },
        "scope": "test",
    }
    result, cells = _evaluate_claim(claim, ledger)
    assert result["gate_passed"]
    assert result["passing_cells"] == 2
    assert cells["cell_passes"].tolist() == [False, True, True]


def test_safety_frontier_keeps_relative_and_absolute_axes_distinct() -> None:
    rows = []
    for estimand, mean, low, high in [
        ("relative", 0.2, 0.1, 0.3),
        ("absolute", -0.1, -0.2, 0.1),
    ]:
        rows.append(
            {
                "experiment_key": "phase",
                "estimand": estimand,
                "method": "direct",
                "disease_regime": "low",
                "detection_profile": "early_detection",
                "rewiring_fraction": 1.0,
                "family_equal_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "positive_families": 3,
            }
        )
    frontier = _safety_frontier(pd.DataFrame(rows), "phase", "direct")
    assert frontier.iloc[0]["decision"] == "abstain_or_require_external_calibration"
    assert frontier.iloc[0]["relative_status"] == "supported_benefit"
    assert frontier.iloc[0]["absolute_status"] == "possible_harm"
