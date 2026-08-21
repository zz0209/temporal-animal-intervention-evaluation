from __future__ import annotations

import numpy as np
import pandas as pd

from animal_intervention.experiments.rolling_surveillance_calibration import (
    _comparison,
    _time_equal_downstream_weights,
    _time_equal_weights,
)


def test_time_equal_weights_give_each_anchor_equal_total_weight() -> None:
    frame = pd.DataFrame(
        {"anchor_time": ["a", "a", "b"], "downstream_cases": [1, 2, 4]}
    )
    weights = pd.Series(_time_equal_weights(frame), index=frame.index)
    totals = weights.groupby(frame["anchor_time"]).sum()
    assert np.allclose(totals.to_numpy(), [1.0, 1.0])


def test_downstream_weights_are_normalized_within_anchor() -> None:
    frame = pd.DataFrame(
        {"anchor_time": ["a", "a", "b"], "downstream_cases": [1, 3, 2]}
    )
    weights = pd.Series(_time_equal_downstream_weights(frame), index=frame.index)
    totals = weights.groupby(frame["anchor_time"]).sum()
    assert np.allclose(totals.to_numpy(), [1.0, 1.0])


def test_primary_comparison_favors_lower_rolling_error() -> None:
    rows = []
    for method, loss in {
        "universal_nominal": 0.3,
        "rolling_nominal": 0.2,
        "rolling_network": 0.15,
        "universal_network": 0.25,
    }.items():
        rows.append(
            {
                "epidemic_model": "sir",
                "system_family": "family",
                "endpoint": "detection_brier",
                "method": method,
                "loss": loss,
            }
        )
    metrics = pd.DataFrame(rows)
    primary = _comparison(metrics, "primary")
    secondary = _comparison(metrics, "secondary")
    assert np.isclose(primary.iloc[0]["improvement"], 0.1)
    assert np.isclose(secondary.iloc[0]["improvement"], 0.05)
