from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.finite_family_robustness import (
    exact_one_sided_sign_probability,
    summarize_family_contrast,
)


def test_exact_sign_probability_respects_independent_family_count() -> None:
    assert exact_one_sided_sign_probability(pd.Series([1, 2, 3, 4, 5])) == 1 / 32
    assert exact_one_sided_sign_probability(pd.Series([1, 2, 3, 4])) == 1 / 16


def test_family_summary_does_not_count_contexts_as_families() -> None:
    rows = pd.DataFrame(
        {
            "system_family": ["a", "a", "b", "b"],
            "value": [1.0, 3.0, 4.0, 4.0],
            "population_size": [10, 10, 100, 100],
        }
    )
    summary, leave = summarize_family_contrast(rows)

    assert summary.loc[0, "families"] == 2
    assert summary.loc[0, "family_equal_mean"] == 3.0
    assert summary.loc[0, "population_context_weighted_mean"] > 3.8
    assert len(leave) == 2
