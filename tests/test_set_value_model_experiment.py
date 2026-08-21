import numpy as np
import pandas as pd

from animal_intervention.experiments.set_value_model import (
    _balanced_weights,
    _bootstrap_gain_intervals,
    _standardize,
)


def test_balanced_weights_give_each_family_equal_total_weight() -> None:
    frame = pd.DataFrame(
        {
            "system_family": ["a", "a", "a", "b"],
            "context_id": ["a1", "a1", "a2", "b1"],
            "set_signature": ["1", "2", "3", "4"],
        }
    )
    weights = _balanced_weights(frame)
    totals = pd.Series(weights).groupby(frame["system_family"]).sum()
    assert np.isclose(totals.loc["a"], totals.loc["b"])


def test_standardization_uses_training_statistics_only() -> None:
    train = np.array([[0.0], [2.0]])
    test = np.array([[101.0]])
    standardized_train, standardized_test = _standardize(train, test)
    np.testing.assert_allclose(standardized_train[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(standardized_test[:, 0], [100.0])


def test_hierarchical_bootstrap_preserves_zero_reference_gain() -> None:
    frame = pd.DataFrame(
        {
            "system_family": ["a", "a", "b", "b"],
            "dataset_id": ["a", "a", "b", "b"],
            "network_id": ["all"] * 4,
            "anchor_id": ["1", "2", "1", "2"],
            "reproducible": [True] * 4,
            "stable_plus_tracing_value": [0.1, 0.2, 0.3, 0.4],
            "stable_watchlist_value": [0.1, 0.2, 0.3, 0.4],
            "contact_to_detected_value": [0.1, 0.2, 0.3, 0.4],
            "ridge_value": [0.1, 0.2, 0.3, 0.4],
            "deep_sets_value": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = _bootstrap_gain_intervals(frame, repetitions=20, seed=3)
    reference = result.loc[result["method"].eq("stable_plus_tracing")]
    assert (reference["gain_ci_low"] == 0).all()
    assert (reference["gain_ci_high"] == 0).all()
