import pandas as pd

from animal_intervention.experiments.policy_leave_one_family_out import _summarize_resilience


def _row(decision: str, omitted_family: str | None = None) -> dict:
    row = {
        "epidemic_model": "temporal_sir",
        "detection_profile": "early_detection",
        "rewiring_fraction": 1.0,
        "decision": decision,
        "direct_minus_history_ci_low": 0.01,
        "direct_absolute_ci_low": 0.01,
    }
    if omitted_family is not None:
        row["omitted_family"] = omitted_family
    return row


def test_resilience_marks_all_fold_override_as_deletion_robust() -> None:
    full = pd.DataFrame([_row("override_with_detected_case_contacts")])
    loo = pd.DataFrame(
        [
            _row("override_with_detected_case_contacts", f"family_{index}")
            for index in range(5)
        ]
    )
    result = _summarize_resilience(full, loo, total_omissions=5)
    assert result.iloc[0]["override_folds"] == 5
    assert result.iloc[0]["resilience_status"] == "deletion_robust"


def test_resilience_marks_two_failed_override_folds_as_fragile() -> None:
    full = pd.DataFrame([_row("override_with_detected_case_contacts")])
    decisions = [
        "override_with_detected_case_contacts",
        "override_with_detected_case_contacts",
        "override_with_detected_case_contacts",
        "retain_history_weight",
        "abstain_or_unresolved",
    ]
    loo = pd.DataFrame(
        [_row(decision, f"family_{index}") for index, decision in enumerate(decisions)]
    )
    result = _summarize_resilience(full, loo, total_omissions=5)
    assert result.iloc[0]["override_folds"] == 3
    assert result.iloc[0]["resilience_status"] == "fragile"


def test_resilience_tracks_non_override_decision_changes() -> None:
    full = pd.DataFrame([_row("retain_history_weight")])
    loo = pd.DataFrame(
        [
            _row("retain_history_weight", "family_0"),
            _row("abstain_or_unresolved", "family_1"),
        ]
    )
    result = _summarize_resilience(full, loo, total_omissions=2)
    assert result.iloc[0]["same_decision_folds"] == 1
    assert result.iloc[0]["resilience_status"] == "decision_changes"
