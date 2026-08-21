from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.immediate_case_targeting import _plot


def test_immediate_plot_accepts_complete_model_and_family_tables(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "epidemic_model": model,
                "contrast": "targeting_increment",
                "family_equal_mean": 0.01,
                "ci_low": 0.001,
                "ci_high": 0.02,
            }
            for model in ["temporal_sir", "temporal_seir_erlang"]
        ]
    )
    family = pd.DataFrame(
        [
            {
                "epidemic_model": model,
                "contrast": "targeting_increment",
                "system_family": system,
                "mean_value": value,
            }
            for model in ["temporal_sir", "temporal_seir_erlang"]
            for system, value in [("a", 0.01), ("b", 0.02)]
        ]
    )
    path = tmp_path / "figure.png"
    _plot(summary, family, path, 80)
    assert path.exists()
