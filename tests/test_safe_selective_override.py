import numpy as np
import pandas as pd

from animal_intervention.experiments.safe_selective_override import (
    _apply_threshold,
    _calibrate_threshold,
)


def _decisions() -> pd.DataFrame:
    rows = []
    for family, gains in {"a": [0.1, 0.1, -0.1], "b": [0.2, 0.2, -0.1]}.items():
        for index, gain in enumerate(gains):
            rows.append(
                {
                    "context_id": f"{family}{index}",
                    "system_family": family,
                    "dataset_id": family,
                    "network_id": "n",
                    "anchor_id": f"x{index}",
                    "selected_set_signature": "learned",
                    "reference_set_signature": "reference",
                    "normalized_margin": [0.9, 0.8, 0.1][index],
                    "gain": gain,
                    "ranking_value": 0.5 + gain,
                    "reference_value": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_threshold_retains_reference_below_margin() -> None:
    policy = _apply_threshold(_decisions(), 0.5)
    assert policy["override"].sum() == 4
    assert np.isclose(policy.loc[~policy["override"], "policy_gain"].sum(), 0.0)
    assert policy.loc[~policy["override"], "policy_set_signature"].eq("reference").all()


def test_calibration_selects_highest_coverage_feasible_threshold() -> None:
    settings = {
        "minimum_contexts_overridden": 2,
        "minimum_families_with_overrides": 2,
        "minimum_family_equal_coverage": 0.1,
        "minimum_positive_families": 2,
        "minimum_bootstrap_probability_positive": 0.5,
        "maximum_family_equal_harmful_override_fraction": 0.25,
    }
    selected, table = _calibrate_threshold(
        _decisions(), [0.0, 0.5, 0.95], settings, repetitions=100, seed=7
    )
    assert selected == 0.5
    assert table.loc[table["selected"], "threshold"].tolist() == [0.5]


def test_calibration_abstains_when_no_threshold_is_safe() -> None:
    frame = _decisions()
    frame["gain"] = -frame["gain"].abs()
    settings = {
        "minimum_contexts_overridden": 2,
        "minimum_families_with_overrides": 2,
        "minimum_family_equal_coverage": 0.1,
        "minimum_positive_families": 2,
        "minimum_bootstrap_probability_positive": 0.5,
        "maximum_family_equal_harmful_override_fraction": 0.25,
    }
    selected, table = _calibrate_threshold(
        frame, [0.0, 0.5], settings, repetitions=50, seed=8
    )
    assert np.isinf(selected)
    assert not table["selected"].any()
