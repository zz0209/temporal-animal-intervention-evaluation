from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.prediction_policy_gap import _forest


def test_forest_plot_accepts_ordered_intervals() -> None:
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [{"estimate": 0.2, "ci_low": 0.1, "ci_high": 0.3}]
    )
    figure, axis = plt.subplots()
    _forest(axis, frame, ["metric"], "title", 1.0, "x")
    plt.close(figure)
