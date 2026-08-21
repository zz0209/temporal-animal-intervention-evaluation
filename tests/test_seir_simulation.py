from __future__ import annotations

import pandas as pd

from animal_intervention.simulation import PairedTemporalSEIREngine, SEIRParameters, states_at
from animal_intervention.simulation.sir import SimulationResult
from animal_intervention.transmission.contract import ExposureStream


def _chain_stream() -> ExposureStream:
    start = pd.Timestamp("2020-01-01")
    return ExposureStream(
        dataset_id="fixture",
        dyadic_exposures=pd.DataFrame(
            [
                {
                    "dataset_id": "fixture",
                    "exposure_id": "ab",
                    "source_id": "a",
                    "target_id": "b",
                    "start_time": start,
                    "end_time": start + pd.Timedelta(seconds=1),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                },
                {
                    "dataset_id": "fixture",
                    "exposure_id": "bc",
                    "source_id": "b",
                    "target_id": "c",
                    "start_time": start + pd.Timedelta(seconds=1),
                    "end_time": start + pd.Timedelta(seconds=2),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                },
            ]
        ),
    )


def test_latent_period_blocks_immediate_temporal_chain() -> None:
    start = pd.Timestamp("2020-01-01")
    result = PairedTemporalSEIREngine().simulate(
        _chain_stream(),
        SEIRParameters(
            beta=100.0,
            latent_rate=1e-9,
            recovery_rate=0.0,
            latent_stages=2,
            infectious_stages=3,
        ),
        initial_infected=("a",),
        start_time=start,
        end_time=start + pd.Timedelta(seconds=2),
        world_seed=17,
    )
    assert result.final_size == 2
    assert result.final_states["b"] == "E"
    assert result.final_states["c"] == "S"


def test_seir_replay_is_keyed_and_reproducible() -> None:
    start = pd.Timestamp("2020-01-01")
    parameters = SEIRParameters(
        beta=10.0,
        latent_rate=2.0,
        recovery_rate=0.5,
        latent_stages=2,
        infectious_stages=3,
    )
    engine = PairedTemporalSEIREngine()
    first = engine.simulate(
        _chain_stream(),
        parameters,
        initial_infected=("a",),
        start_time=start,
        end_time=start + pd.Timedelta(seconds=2),
        world_seed=41,
    )
    second = engine.simulate(
        _chain_stream(),
        parameters,
        initial_infected=("a",),
        start_time=start,
        end_time=start + pd.Timedelta(seconds=2),
        world_seed=41,
    )
    pd.testing.assert_frame_equal(first.event_log, second.event_log)
    assert first.final_states == second.final_states


def test_states_at_reconstructs_exposed_and_infectious_states() -> None:
    start = pd.Timestamp("2020-01-01")
    event_log = pd.DataFrame(
        [
            {"time": start, "event": "initial_infection", "node_id": "a", "source_id": None},
            {"time": start + pd.Timedelta(seconds=1), "event": "infection", "node_id": "b", "source_id": "a"},
            {"time": start + pd.Timedelta(seconds=3), "event": "become_infectious", "node_id": "b", "source_id": None},
        ]
    )
    result = SimulationResult(
        dataset_id="fixture",
        start_time=start,
        end_time=start + pd.Timedelta(seconds=5),
        initial_infected=("a",),
        isolated_nodes=(),
        final_states={"a": "I", "b": "I"},
        event_log=event_log,
        final_size=2,
        peak_infectious=2,
        extinct=False,
        seed=1,
    )
    assert states_at(result, start + pd.Timedelta(seconds=2))["b"] == "E"
    assert states_at(result, start + pd.Timedelta(seconds=4))["b"] == "I"
