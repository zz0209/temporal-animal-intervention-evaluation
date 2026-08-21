import numpy as np
import pandas as pd

from animal_intervention.experiments.set_value_ranking_diagnostic import (
    _chronological_test_mask,
    _pair_table,
)


def test_pair_confidence_downweights_monte_carlo_ambiguity() -> None:
    frame = pd.DataFrame(
        {
            "context_id": ["c", "c", "c"],
            "system_family": ["f", "f", "f"],
            "mean_set_value": [0.0, 0.01, 0.10],
            "set_value_se": [0.05, 0.05, 0.01],
        },
        index=[0, 1, 2],
    )
    pairs = _pair_table(frame)
    assert pairs["confidence"].max() > pairs["confidence"].min()
    assert (pairs["weight"] > 0).all()


def test_pair_weights_equalize_family_totals() -> None:
    frame = pd.DataFrame(
        {
            "context_id": ["a", "a", "b", "b"],
            "system_family": ["first", "first", "second", "second"],
            "mean_set_value": [0.0, 0.1, 0.0, 0.2],
            "set_value_se": [0.01] * 4,
        }
    )
    pairs = _pair_table(frame)
    totals = pairs.groupby("system_family")["weight"].sum()
    assert np.isclose(totals.loc["first"], totals.loc["second"])


def test_chronological_split_holds_out_latest_complete_anchor() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["d"] * 6,
            "network_id": ["n"] * 6,
            "anchor_id": ["a", "a", "b", "b", "c", "c"],
            "anchor_time": ["2020-01-01"] * 2 + ["2020-01-02"] * 2 + ["2020-01-03"] * 2,
        }
    )
    mask = _chronological_test_mask(frame, 0.20)
    assert mask.tolist() == [False, False, False, False, True, True]
