from __future__ import annotations

from pathlib import Path

import pandas as pd

from animal_intervention.estimands.intervention_value import AnchorWindow
from animal_intervention.experiments.oxford_predefense import _run_stability
from animal_intervention.experiments.stability_parallel import run_checkpointed_stability
from animal_intervention.transmission.contract import ExposureStream


def _prepared_window() -> dict[str, object]:
    start = pd.Timestamp("2020-01-01")
    exposures = []
    for index, (source, target) in enumerate(
        [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]
    ):
        exposures.append(
            {
                "dataset_id": "fixture",
                "exposure_id": f"e{index}",
                "source_id": source,
                "target_id": target,
                "start_time": start + pd.Timedelta(seconds=index),
                "end_time": start + pd.Timedelta(seconds=index + 1),
                "hazard_rate_multiplier": 1.0,
                "directed": False,
            }
        )
    stream = ExposureStream(
        dataset_id="fixture",
        population_nodes=("A", "B", "C", "D"),
        dyadic_exposures=pd.DataFrame(exposures),
    )
    return {
        "anchor": AnchorWindow(
            anchor_id="anchor_001",
            history_start=start - pd.Timedelta(days=1),
            anchor_time=start,
            horizon_end=start + pd.Timedelta(seconds=4),
        ),
        "future": stream,
        "eligible": ["A", "B", "C", "D"],
        "history_support": pd.Series({node: 1 for node in "ABCD"}),
        "population_size": 4,
    }


def _parameters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parameter_id": "beta_5__ip_1d",
                "beta": 5.0,
                "mean_infectious_period_days": 1.0,
                "recovery_rate_per_day": 1.0,
            }
        ]
    )


def _action() -> dict[str, object]:
    return {
        "name": "complete_contact_isolation",
        "action_type": "isolation",
        "delay": "0s",
        "duration": "4s",
        "contact_multiplier": 0.0,
        "susceptibility_multiplier": 1.0,
        "infectivity_multiplier": 1.0,
        "recovery_rate_multiplier": 1.0,
    }


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "anchor_id",
        "parameter_id",
        "block_id",
        "candidate_id",
        "introduction_stratum",
        "initial_infected",
        "world_seed",
    ]
    return frame.sort_values(columns, ignore_index=True)


def test_checkpointed_runner_matches_legacy_serial_worlds(tmp_path: Path) -> None:
    prepared = [_prepared_window()]
    legacy_worlds, legacy_estimates = _run_stability(
        prepared,
        _parameters(),
        _action(),
        random_blocks=2,
        non_index_cases=2,
        self_replicates=1,
        candidate_limit=None,
        seed=91,
        progress_label="legacy test",
    )
    worlds, estimates = run_checkpointed_stability(
        prepared,
        _parameters(),
        _action(),
        random_blocks=2,
        non_index_cases=2,
        self_replicates=1,
        candidate_limit=None,
        seed=91,
        checkpoint_dir=tmp_path / "serial",
        max_workers=1,
        progress_label="checkpoint test",
    )
    shared_world_columns = list(legacy_worlds.columns)
    pd.testing.assert_frame_equal(
        _sort(legacy_worlds)[shared_world_columns],
        _sort(worlds)[shared_world_columns],
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        legacy_estimates.sort_values(
            ["anchor_id", "parameter_id", "block_id", "candidate_id"],
            ignore_index=True,
        ),
        estimates.sort_values(
            ["anchor_id", "parameter_id", "block_id", "candidate_id"],
            ignore_index=True,
        ),
        check_dtype=False,
    )


def test_parallel_runner_and_resume_match_serial(tmp_path: Path) -> None:
    arguments = {
        "prepared": [_prepared_window()],
        "parameters": _parameters(),
        "action_config": _action(),
        "random_blocks": 2,
        "non_index_cases": 2,
        "self_replicates": 1,
        "candidate_limit": None,
        "seed": 92,
    }
    serial, _ = run_checkpointed_stability(
        **arguments,
        checkpoint_dir=tmp_path / "serial",
        max_workers=1,
        progress_label="serial test",
    )
    parallel, _ = run_checkpointed_stability(
        **arguments,
        checkpoint_dir=tmp_path / "parallel",
        max_workers=2,
        progress_label="parallel test",
    )
    resumed, _ = run_checkpointed_stability(
        **arguments,
        checkpoint_dir=tmp_path / "parallel",
        max_workers=2,
        progress_label="resume test",
    )
    pd.testing.assert_frame_equal(_sort(serial), _sort(parallel), check_dtype=False)
    pd.testing.assert_frame_equal(_sort(parallel), _sort(resumed), check_dtype=False)
