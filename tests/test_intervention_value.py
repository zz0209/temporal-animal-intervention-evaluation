from __future__ import annotations

import pandas as pd

from animal_intervention.estimands.intervention_value import (
    estimate_singleton_values,
    estimate_stratified_singleton_values,
    node_support,
    rolling_anchors,
    slice_stream,
)
from animal_intervention.simulation import SIRParameters
from animal_intervention.transmission.contract import ExposureStream


def _event(exposure_id: str, source: str, target: str, start: str, end: str) -> dict:
    return {
        "dataset_id": "fixture",
        "exposure_id": exposure_id,
        "source_id": source,
        "target_id": target,
        "start_time": pd.Timestamp(start),
        "end_time": pd.Timestamp(end),
        "hazard_rate_multiplier": 1.0,
        "directed": False,
    }


def test_anchor_eligibility_uses_history_not_future() -> None:
    stream = ExposureStream(
        dataset_id="fixture",
        dyadic_exposures=pd.DataFrame(
            [
                _event("history", "a", "b", "2020-01-01", "2020-01-02"),
                _event("future", "c", "d", "2020-01-02", "2020-01-03"),
            ]
        ),
    )
    anchors = rolling_anchors(
        stream, lookback=pd.Timedelta(days=1), horizon=pd.Timedelta(days=1),
        step=pd.Timedelta(days=1),
    )
    history = slice_stream(stream, anchors[0].history_start, anchors[0].anchor_time)
    future = slice_stream(stream, anchors[0].anchor_time, anchors[0].horizon_end)
    assert set(node_support(history).index) == {"a", "b"}
    assert set(node_support(future).index) == {"c", "d"}


def test_estimator_returns_reproducible_paired_labels_and_intervals() -> None:
    stream = ExposureStream(
        dataset_id="fixture",
        dyadic_exposures=pd.DataFrame(
            [
                _event("h1", "a", "b", "2020-01-01", "2020-01-02"),
                _event("h2", "b", "c", "2020-01-01", "2020-01-02"),
                _event("f1", "a", "b", "2020-01-02", "2020-01-02 00:00:01"),
                _event("f2", "b", "c", "2020-01-02 00:00:01", "2020-01-03"),
            ]
        ),
    )
    anchors = rolling_anchors(
        stream, lookback=pd.Timedelta(days=1), horizon=pd.Timedelta(days=1),
        step=pd.Timedelta(days=1),
    )
    action = {
        "name": "isolation", "action_type": "isolation", "delay": "0s",
        "duration": "1d", "contact_multiplier": 0.0,
    }
    first = estimate_singleton_values(
        stream, anchors, SIRParameters(beta=100, recovery_rate=0),
        action_config=action, worlds=4, seed=9, show_progress=False,
    )
    second = estimate_singleton_values(
        stream, anchors, SIRParameters(beta=100, recovery_rate=0),
        action_config=action, worlds=4, seed=9, show_progress=False,
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert {"mean_avoided_attack_rate", "mc_standard_error", "ci95_lower", "ci95_upper", "rank"} <= set(first[0])
    assert len(first[1]) == 12


def test_stratified_estimator_balances_index_cases_and_weights_strata() -> None:
    stream = ExposureStream(
        dataset_id="fixture",
        dyadic_exposures=pd.DataFrame(
            [
                _event("h1", "a", "b", "2020-01-01", "2020-01-02"),
                _event("h2", "b", "c", "2020-01-01", "2020-01-02"),
                _event("f1", "a", "b", "2020-01-02", "2020-01-02 12:00:00"),
                _event("f2", "b", "c", "2020-01-02 12:00:00", "2020-01-03"),
            ]
        ),
    )
    anchors = rolling_anchors(
        stream, lookback=pd.Timedelta(days=1), horizon=pd.Timedelta(days=1),
        step=pd.Timedelta(days=1),
    )
    estimates, worlds, _ = estimate_stratified_singleton_values(
        stream,
        anchors,
        SIRParameters(beta=100, recovery_rate=0),
        action_config={
            "name": "isolation", "action_type": "isolation", "delay": "0s",
            "duration": "1d", "contact_multiplier": 0.0,
        },
        self_seed_replicates=2,
        bootstrap_replicates=200,
        seed=11,
        show_progress=False,
    )
    counts = worlds.groupby(["candidate_id", "introduction_stratum"]).size().unstack()
    assert counts["self_index"].eq(2).all()
    assert counts["non_index"].eq(2).all()
    assert set(worlds.loc[worlds["introduction_stratum"].eq("self_index"), "candidate_id"]) == {"a", "b", "c"}
    expected = estimates["known_index_value"] / 3 + estimates["non_index_value"] * 2 / 3
    assert (estimates["unconditional_value"] - expected).abs().max() < 1e-12
