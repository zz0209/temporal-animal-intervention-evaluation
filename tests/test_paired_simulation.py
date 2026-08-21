from __future__ import annotations

import pandas as pd

from animal_intervention.simulation import InterventionAction, PairedTemporalSIREngine, SIRParameters
from animal_intervention.transmission.contract import ExposureStream


def _stream() -> ExposureStream:
    base = pd.Timestamp("2020-01-01")
    return ExposureStream(
        dataset_id="fixture",
        dyadic_exposures=pd.DataFrame(
            [
                {
                    "dataset_id": "fixture",
                    "exposure_id": "ab",
                    "source_id": "a",
                    "target_id": "b",
                    "start_time": base,
                    "end_time": base + pd.Timedelta(seconds=1),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                },
                {
                    "dataset_id": "fixture",
                    "exposure_id": "bc",
                    "source_id": "b",
                    "target_id": "c",
                    "start_time": base + pd.Timedelta(seconds=1),
                    "end_time": base + pd.Timedelta(seconds=2),
                    "hazard_rate_multiplier": 1.0,
                    "directed": False,
                },
            ]
        ),
    )


def test_neutral_action_reproduces_baseline_exactly() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=2)
    engine = PairedTemporalSIREngine()
    baseline = engine.simulate(
        stream, SIRParameters(beta=100, recovery_rate=0), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=17,
    )
    neutral = InterventionAction(
        name="neutral", action_type="none", target_nodes=("b",),
        start_time=start, end_time=end,
    )
    repeated = engine.simulate(
        stream, SIRParameters(beta=100, recovery_rate=0), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=17, action=neutral,
    )
    pd.testing.assert_frame_equal(baseline.event_log, repeated.event_log)
    assert baseline.final_states == repeated.final_states


def test_complete_isolation_removes_transmission_opportunities() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=2)
    action = InterventionAction(
        name="isolate_b", action_type="isolation", target_nodes=("b",),
        start_time=start, end_time=end, contact_multiplier=0,
    )
    engine = PairedTemporalSIREngine()
    baseline = engine.simulate(
        stream, SIRParameters(beta=100, recovery_rate=0), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=17,
    )
    isolated = engine.simulate(
        stream, SIRParameters(beta=100, recovery_rate=0), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=17, action=action,
    )
    assert baseline.final_size == 3
    assert isolated.final_size == 1


def test_event_times_are_ordered_and_world_seed_is_reproducible() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=2)
    engine = PairedTemporalSIREngine()
    first = engine.simulate(
        stream, SIRParameters(beta=5, recovery_rate=0.2), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=991,
    )
    second = engine.simulate(
        stream, SIRParameters(beta=5, recovery_rate=0.2), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=991,
    )
    assert first.event_log["time"].is_monotonic_increasing
    pd.testing.assert_frame_equal(first.event_log, second.event_log)


def test_target_without_future_contact_has_zero_effect() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=1)
    action = InterventionAction(
        name="late_target", action_type="isolation", target_nodes=("c",),
        start_time=start, end_time=end, contact_multiplier=0,
    )
    engine = PairedTemporalSIREngine()
    baseline = engine.simulate(
        stream, SIRParameters(beta=5, recovery_rate=0), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=3,
    )
    isolated = engine.simulate(
        stream, SIRParameters(beta=5, recovery_rate=0), initial_infected=["a"],
        start_time=start, end_time=end, world_seed=3, action=action,
    )
    pd.testing.assert_frame_equal(baseline.event_log, isolated.event_log)


def test_initial_node_without_future_exposure_is_a_valid_isolate() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=1)
    result = PairedTemporalSIREngine().simulate(
        stream, SIRParameters(beta=5, recovery_rate=0), initial_infected=["history_only"],
        start_time=start, end_time=end, world_seed=5,
    )
    assert result.final_size == 1
    assert result.final_states["history_only"] == "I"


def test_reusing_engine_preserves_exact_simulation_output() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=2)
    parameters = SIRParameters(beta=5, recovery_rate=0.2)
    reused_engine = PairedTemporalSIREngine()
    first = reused_engine.simulate(
        stream, parameters, initial_infected=["a"], start_time=start,
        end_time=end, world_seed=991,
    )
    second = reused_engine.simulate(
        stream, parameters, initial_infected=["a"], start_time=start,
        end_time=end, world_seed=991,
    )
    fresh = PairedTemporalSIREngine().simulate(
        stream, parameters, initial_infected=["a"], start_time=start,
        end_time=end, world_seed=991,
    )
    pd.testing.assert_frame_equal(first.event_log, second.event_log)
    pd.testing.assert_frame_equal(first.event_log, fresh.event_log)
    assert first.final_states == second.final_states == fresh.final_states


def test_conditional_restart_keeps_recovered_nodes_immune() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=1)
    end = start + pd.Timedelta(seconds=1)
    result = PairedTemporalSIREngine().simulate(
        stream,
        SIRParameters(beta=100, recovery_rate=0),
        initial_infected=["b"],
        initial_recovered=["c"],
        start_time=start,
        end_time=end,
        world_seed=17,
    )

    assert result.final_states["c"] == "R"
    assert result.final_size == 2


def test_conditional_restart_rejects_conflicting_initial_states() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=1)

    try:
        PairedTemporalSIREngine().simulate(
            stream,
            SIRParameters(beta=1, recovery_rate=1),
            initial_infected=["a"],
            initial_recovered=["a"],
            start_time=start,
            end_time=end,
            world_seed=1,
        )
    except ValueError as error:
        assert "must be disjoint" in str(error)
    else:
        raise AssertionError("conflicting conditional states were accepted")


def test_partner_substitution_rewiring_redirects_removed_contact() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=1)
    isolated = InterventionAction(
        name="isolate_b",
        action_type="isolation",
        target_nodes=("b",),
        start_time=start,
        end_time=end,
        contact_multiplier=0,
    )
    rewired = InterventionAction(
        name="isolate_b_with_substitution",
        action_type="isolation",
        target_nodes=("b",),
        start_time=start,
        end_time=end,
        contact_multiplier=0,
        rewiring_fraction=1,
        rewiring_mode="uniform_partner_substitution",
    )
    engine = PairedTemporalSIREngine()
    without_rewiring = engine.simulate(
        stream,
        SIRParameters(beta=100, recovery_rate=0),
        initial_infected=["a"],
        start_time=start,
        end_time=end,
        world_seed=17,
        action=isolated,
    )
    with_rewiring = engine.simulate(
        stream,
        SIRParameters(beta=100, recovery_rate=0),
        initial_infected=["a"],
        start_time=start,
        end_time=end,
        world_seed=17,
        action=rewired,
    )

    assert without_rewiring.final_size == 1
    assert with_rewiring.final_size == 2
    assert with_rewiring.final_states["c"] == "I"


def test_rewiring_fraction_requires_supported_mode_and_bounds() -> None:
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(days=1)
    for fraction, mode in [(1.1, "uniform_partner_substitution"), (0.5, "none")]:
        try:
            InterventionAction(
                name="invalid_rewiring",
                action_type="isolation",
                target_nodes=("a",),
                start_time=start,
                end_time=end,
                rewiring_fraction=fraction,
                rewiring_mode=mode,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rewiring configuration was accepted")


def test_intervention_hazard_accounting_reconciles_removed_and_rewired_mass() -> None:
    stream = _stream()
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(seconds=1)
    action = InterventionAction(
        name="account_isolation",
        action_type="isolation",
        target_nodes=("b",),
        start_time=start,
        end_time=end,
        contact_multiplier=0.25,
        rewiring_fraction=0.5,
        rewiring_mode="uniform_partner_substitution",
    )

    accounting = PairedTemporalSIREngine().intervention_hazard_accounting(
        stream,
        start_time=start,
        end_time=end,
        action=action,
    )

    assert accounting.original_hazard_mass == 2.0
    assert accounting.residual_original_hazard_mass == 0.5
    assert accounting.removed_original_hazard_mass == 1.5
    assert accounting.rewired_hazard_mass == 0.75
    assert accounting.rewired_opportunities == 2
