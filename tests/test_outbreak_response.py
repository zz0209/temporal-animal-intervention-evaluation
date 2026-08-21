from __future__ import annotations

import pandas as pd

from animal_intervention.simulation import (
    ContactObservationProfile,
    DetectionProfile,
    PairedTemporalSIREngine,
    SIRParameters,
    detection_time_from_seed,
    observe_detected_cases,
    pre_detection_event_signature,
    pre_detection_scores,
    perturbed_pre_detection_scores,
    run_response_pair,
    select_additional_targets,
)
from animal_intervention.transmission.contract import ExposureStream


def _stream() -> ExposureStream:
    start = pd.Timestamp("2020-01-01")
    return ExposureStream(
        dataset_id="fixture",
        population_nodes=("a", "b", "c", "d"),
        dyadic_exposures=pd.DataFrame(
            [
                {
                    "dataset_id": "fixture",
                    "exposure_id": "ab",
                    "source_id": "a",
                    "target_id": "b",
                    "start_time": start,
                    "end_time": start + pd.Timedelta(hours=2),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                    "transmission_route": "fixture",
                    "mapper_name": "fixture",
                    "origin_event_id": "ab",
                    "location_id": pd.NA,
                },
                {
                    "dataset_id": "fixture",
                    "exposure_id": "bc",
                    "source_id": "b",
                    "target_id": "c",
                    "start_time": start + pd.Timedelta(hours=2),
                    "end_time": start + pd.Timedelta(hours=4),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                    "transmission_route": "fixture",
                    "mapper_name": "fixture",
                    "origin_event_id": "bc",
                    "location_id": pd.NA,
                },
            ]
        ),
    )


def test_detection_time_is_relative_to_infectious_period() -> None:
    time = detection_time_from_seed(
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-03"),
        pd.Timedelta(days=1),
        DetectionProfile("early", 0.25),
    )
    assert time == pd.Timestamp("2020-01-01 06:00:00")


def test_secondary_case_detection_is_keyed_and_respects_sensitivity() -> None:
    states = {"a": "I", "b": "I", "c": "I", "d": "S"}
    assert observe_detected_cases(
        states,
        trigger_node="a",
        secondary_case_sensitivity=0.0,
        world_seed=17,
    ) == ("a",)
    assert observe_detected_cases(
        states,
        trigger_node="a",
        secondary_case_sensitivity=1.0,
        world_seed=17,
    ) == ("a", "b", "c")
    first = observe_detected_cases(
        states,
        trigger_node="a",
        secondary_case_sensitivity=0.5,
        world_seed=17,
    )
    second = observe_detected_cases(
        states,
        trigger_node="a",
        secondary_case_sensitivity=0.5,
        world_seed=17,
    )
    assert first == second


def test_false_positive_detection_is_keyed_and_respects_rate() -> None:
    states = {"a": "I", "b": "I", "c": "S", "d": "R"}
    no_false_positives = observe_detected_cases(
        states,
        trigger_node="a",
        secondary_case_sensitivity=0.0,
        false_positive_rate=0.0,
        world_seed=31,
    )
    all_false_positives = observe_detected_cases(
        states,
        trigger_node="a",
        secondary_case_sensitivity=0.0,
        false_positive_rate=1.0,
        world_seed=31,
    )
    assert no_false_positives == ("a",)
    assert all_false_positives == ("a", "c", "d")


def test_case_observation_rejects_invalid_false_positive_rate() -> None:
    try:
        observe_detected_cases(
            {"a": "I"},
            trigger_node="a",
            secondary_case_sensitivity=0.5,
            false_positive_rate=1.1,
            world_seed=31,
        )
    except ValueError as error:
        assert "false-positive rate" in str(error)
    else:
        raise AssertionError("invalid false-positive rate was accepted")


def test_contact_to_detected_uses_only_pre_detection_contacts() -> None:
    stream = _stream()
    scores = pre_detection_scores(
        stream,
        detected_nodes=("a",),
        start_time=pd.Timestamp("2020-01-01"),
        detection_time=pd.Timestamp("2020-01-01 02:00:00"),
        half_life=pd.Timedelta(hours=1),
    ).set_index("candidate_id")
    assert scores.loc["b", "contact_to_detected"] > 0
    assert scores.loc["c", "contact_to_detected"] == 0


def test_complete_contact_observation_matches_reference_scores() -> None:
    arguments = {
        "detected_nodes": ("a",),
        "start_time": pd.Timestamp("2020-01-01"),
        "detection_time": pd.Timestamp("2020-01-01 04:00:00"),
        "half_life": pd.Timedelta(hours=1),
    }
    reference = pre_detection_scores(_stream(), **arguments).sort_values("candidate_id")
    perturbed, diagnostics = perturbed_pre_detection_scores(
        _stream(),
        **arguments,
        profile=ContactObservationProfile(name="reference"),
        observation_seed=19,
    )
    pd.testing.assert_frame_equal(
        reference.reset_index(drop=True),
        perturbed.sort_values("candidate_id").reset_index(drop=True),
    )
    assert diagnostics["retained_dyadic_events"] == 2


def test_stronger_event_loss_is_nested_and_reproducible() -> None:
    arguments = {
        "detected_nodes": ("a",),
        "start_time": pd.Timestamp("2020-01-01"),
        "detection_time": pd.Timestamp("2020-01-01 04:00:00"),
        "half_life": pd.Timedelta(hours=1),
        "observation_seed": 23,
    }
    moderate = perturbed_pre_detection_scores(
        _stream(),
        **arguments,
        profile=ContactObservationProfile(
            name="moderate", event_retention_probability=0.7
        ),
    )
    severe = perturbed_pre_detection_scores(
        _stream(),
        **arguments,
        profile=ContactObservationProfile(
            name="severe", event_retention_probability=0.3
        ),
    )
    repeated = perturbed_pre_detection_scores(
        _stream(),
        **arguments,
        profile=ContactObservationProfile(
            name="moderate_again", event_retention_probability=0.7
        ),
    )
    assert severe[1]["retained_dyadic_events"] <= moderate[1]["retained_dyadic_events"]
    pd.testing.assert_frame_equal(moderate[0], repeated[0])


def test_time_coarsening_never_makes_contact_more_recent() -> None:
    arguments = {
        "detected_nodes": ("a",),
        "start_time": pd.Timestamp("2020-01-01"),
        "detection_time": pd.Timestamp("2020-01-01 02:00:00"),
        "half_life": pd.Timedelta(hours=1),
        "observation_seed": 29,
    }
    reference, _ = perturbed_pre_detection_scores(
        _stream(), **arguments, profile=ContactObservationProfile(name="reference")
    )
    coarse, _ = perturbed_pre_detection_scores(
        _stream(),
        **arguments,
        profile=ContactObservationProfile(name="coarse", time_bin=pd.Timedelta(hours=2)),
    )
    merged = reference.merge(coarse, on="candidate_id", suffixes=("_reference", "_coarse"))
    assert merged["contact_to_detected_coarse"].le(
        merged["contact_to_detected_reference"]
    ).all()


def test_target_selection_excludes_detected_and_respects_budget() -> None:
    table = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "stable_score": [1.0, 0.9, 0.5, 0.1],
            "current_activity": [1.0, 2.0, 3.0, 4.0],
            "contact_to_detected": [0.0, 4.0, 2.0, 1.0],
            "infected_at_detection": [True, False, True, False],
        }
    )
    targets = select_additional_targets(
        table,
        method="stable_plus_tracing",
        budget=2,
        detected_nodes=("a",),
        world_seed=7,
    )
    assert len(targets) == 2
    assert "a" not in targets
    assert "b" in targets

    perfect_state = select_additional_targets(
        table,
        method="perfect_state_diagnostic",
        budget=1,
        detected_nodes=("a",),
        world_seed=7,
    )
    assert perfect_state == ("c",)


def test_history_comparators_use_their_declared_columns() -> None:
    table = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "stable_score": [0.1, 0.2, 0.3],
            "history_weight": [1.0, 5.0, 2.0],
            "history_recency": [4.0, 1.0, 3.0],
        }
    )

    assert select_additional_targets(
        table,
        method="history_weight",
        budget=1,
        detected_nodes=("a",),
        world_seed=7,
    ) == ("b",)
    assert select_additional_targets(
        table,
        method="history_recency",
        budget=1,
        detected_nodes=("a",),
        world_seed=7,
    ) == ("c",)


def test_paired_response_is_identical_before_detection() -> None:
    stream = _stream()
    engine = PairedTemporalSIREngine()
    start = pd.Timestamp("2020-01-01")
    detection = start + pd.Timedelta(hours=2)
    pair = run_response_pair(
        engine,
        stream,
        SIRParameters(beta=0.01, recovery_rate=0.0),
        initial_infected=("a",),
        start_time=start,
        end_time=start + pd.Timedelta(hours=5),
        detection_time=detection,
        detected_nodes=("a",),
        additional_targets=("b",),
        world_seed=11,
    )
    natural = pre_detection_event_signature(pair.natural_history, detection)
    standard = pre_detection_event_signature(pair.standard_care, detection)
    augmented = pre_detection_event_signature(pair.augmented_response, detection)
    assert natural == standard == augmented
    assert pair.avoided_infections >= 0
