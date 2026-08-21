from __future__ import annotations

import pandas as pd

from animal_intervention.simulation.sir import SIRParameters, TemporalSIREngine
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import GroupMixingMapper


def _dyadic_stream(
    rows: list[tuple[str, str, int, int]], *, directed: bool = False
) -> ExposureStream:
    origin = pd.Timestamp("2020-01-01")
    return ExposureStream(
        dataset_id="sir_test",
        dyadic_exposures=pd.DataFrame(
            {
                "dataset_id": "sir_test",
                "exposure_id": [f"e{i}" for i in range(len(rows))],
                "source_id": [row[0] for row in rows],
                "target_id": [row[1] for row in rows],
                "start_time": [origin + pd.Timedelta(f"{row[2]}s") for row in rows],
                "end_time": [origin + pd.Timedelta(f"{row[3]}s") for row in rows],
                "hazard_rate_multiplier": 1.0,
                "directed": directed,
            }
        ),
    )


def test_sir_transmits_over_active_contact():
    result = TemporalSIREngine().simulate(
        _dyadic_stream([("A", "B", 0, 10)]),
        SIRParameters(beta=100.0, recovery_rate=0.0),
        initial_infected=["A"],
        seed=4,
    )
    assert result.final_size == 2
    assert result.final_states["B"] == "I"


def test_sir_respects_temporal_order():
    # B meets C before B can be infected by A, so C must remain susceptible.
    result = TemporalSIREngine().simulate(
        _dyadic_stream([("B", "C", 0, 10), ("A", "B", 10, 20)]),
        SIRParameters(beta=100.0, recovery_rate=0.0),
        initial_infected=["A"],
        seed=4,
    )
    assert result.final_size == 2
    assert result.final_states["C"] == "S"


def test_sir_respects_preemptive_isolation():
    result = TemporalSIREngine().simulate(
        _dyadic_stream([("A", "B", 0, 10)]),
        SIRParameters(beta=100.0, recovery_rate=0.0),
        initial_infected=["A"],
        isolated_nodes=["B"],
        seed=4,
    )
    assert result.final_size == 1


def test_sir_respects_directed_exposure():
    result = TemporalSIREngine().simulate(
        _dyadic_stream([("A", "B", 0, 10)], directed=True),
        SIRParameters(beta=100.0, recovery_rate=0.0),
        initial_infected=["B"],
        seed=4,
    )
    assert result.final_size == 1
    assert result.final_states["A"] == "S"


def test_sir_recovery_transition_and_seed_reproducibility():
    stream = _dyadic_stream([("A", "B", 0, 10)])
    parameters = SIRParameters(beta=0.0, recovery_rate=100.0)
    first = TemporalSIREngine().simulate(
        stream, parameters, initial_infected=["A"], seed=8
    )
    second = TemporalSIREngine().simulate(
        stream, parameters, initial_infected=["A"], seed=8
    )
    assert first.extinct
    assert first.final_states["A"] == "R"
    pd.testing.assert_frame_equal(first.event_log, second.event_log)


def test_sir_runs_on_group_exposure(group_dataset):
    stream = GroupMixingMapper().compile(group_dataset)
    result = TemporalSIREngine().simulate(
        stream,
        SIRParameters(beta=100.0, recovery_rate=0.0),
        initial_infected=["A"],
        seed=4,
    )
    assert result.final_size == 3


def test_sir_retains_population_nodes_without_future_exposure():
    stream = _dyadic_stream([("A", "B", 0, 10)])
    stream.population_nodes = ("A", "B", "C")
    result = TemporalSIREngine().simulate(
        stream,
        SIRParameters(beta=0.0, recovery_rate=0.0),
        initial_infected=["C"],
        seed=4,
    )
    assert result.final_size == 1
    assert result.final_states == {"A": "S", "B": "S", "C": "I"}
