from __future__ import annotations

import numpy as np
import pandas as pd

from animal_intervention.experiments.effective_surveillance_dose import (
    _decision,
    _dose_curve,
    _fit_rate,
    _paired_improvements,
    _parse_nodes,
)


def test_parse_nodes_handles_empty_and_delimited_values() -> None:
    assert _parse_nodes(np.nan) == ()
    assert _parse_nodes("") == ()
    assert _parse_nodes("a|b") == ("a", "b")


def test_dose_curve_is_bounded_and_monotone() -> None:
    values = _dose_curve(np.array([0.0, 0.1, 0.3]), rate=4.0)
    assert values[0] == 0.0
    assert np.all(np.diff(values) > 0)
    assert np.all((values >= 0) & (values <= 1))


def test_fit_rate_recovers_monotone_detection_pattern() -> None:
    dose = np.linspace(0.02, 0.5, 100)
    truth = _dose_curve(dose, rate=5.0)
    rates = np.geomspace(0.01, 100.0, 5000)
    fitted = _fit_rate(dose, truth, np.ones_like(dose), rates, "absolute_error")
    assert abs(np.log(fitted / 5.0)) < 0.02


def test_paired_improvement_is_nominal_loss_minus_network_loss() -> None:
    metrics = pd.DataFrame(
        [
            {"epidemic_model": "sir", "heldout_family": "a", "endpoint": "brier", "dose_model": "nominal", "loss": 0.3},
            {"epidemic_model": "sir", "heldout_family": "a", "endpoint": "brier", "dose_model": "network", "loss": 0.2},
        ]
    )
    paired = _paired_improvements(metrics)
    assert np.isclose(paired.iloc[0]["improvement"], 0.1)


def test_decision_requires_all_model_endpoint_cells() -> None:
    summary = pd.DataFrame(
        {
            "ci_low": [0.01, 0.02, 0.01, 0.02],
            "positive_families": [4, 5, 4, 5],
            "family_equal_mean_improvement": [0.02] * 4,
        }
    )
    assert _decision(summary, expected_families=5) == "strong"
    summary.loc[0, "ci_low"] = -0.01
    assert _decision(summary, expected_families=5) == "directional"
    summary.loc[0, "family_equal_mean_improvement"] = -0.01
    assert _decision(summary, expected_families=5) == "unsupported"
