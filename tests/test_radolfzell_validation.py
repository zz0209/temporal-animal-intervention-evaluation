from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.experiments.radolfzell_validation import (
    _data_quality_audit,
    _observed_group_stream,
    _prepare_windows,
)
from animal_intervention.experiments.wytham_validation import _label_precision_diagnostics


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "EXP-20260815-011_radolfzell_validation.yaml"


def _prepared_data():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = CanonicalDataset.read(ROOT / config["data"]["canonical_path"])
    stream = _observed_group_stream(dataset, config)
    prepared, metadata = _prepare_windows(stream, config["windows"], max_anchors=15)
    return dataset, stream, prepared, metadata


def test_radolfzell_windows_preserve_discontinuous_sampling_design() -> None:
    _, _, prepared, metadata = _prepared_data()

    assert len(prepared) == 15
    assert metadata["observation_season"].value_counts().to_dict() == {
        "summer": 12,
        "autumn": 1,
        "winter": 1,
        "spring": 1,
    }
    assert metadata["network_id"].nunique() == 1
    assert metadata["eligible_population"].between(16, 105).all()
    assert metadata["future_active_fraction"].ge(0.65).all()
    for window in prepared:
        support = window["history_period_support"].reindex(window["eligible"])
        assert support.ge(2).all()
        assert window["future"].nodes() == set(window["eligible"])


def test_radolfzell_quality_audit_reconciles_published_gmm_events() -> None:
    dataset, stream, prepared, _ = _prepared_data()

    audit, daily, group_sizes = _data_quality_audit(dataset, stream, prepared)

    assert audit["status"] == "passed"
    assert audit["excluded_nonpositive_all_species_group_events"] == 120
    assert audit["mapped_group_events"] == 6431
    assert audit["observed_host_species_individuals"] == 306
    assert audit["observed_individuals_without_roster_metadata"] == 107
    assert audit["recording_periods_by_season"] == {
        "autumn": 3,
        "spring": 3,
        "summer": 14,
        "winter": 3,
    }
    assert not daily.empty
    assert len(group_sizes) == 6431


def test_ready_precision_diagnostic_is_not_mislabeled_as_insufficient() -> None:
    worlds = pd.DataFrame(
        {
            "anchor_id": ["anchor_001"] * 4,
            "parameter_id": ["scenario_1"] * 4,
            "candidate_id": ["a", "a", "b", "b"],
            "introduction_stratum": ["non_index", "self_index"] * 2,
            "avoided_attack_rate": [0.1, 0.2, 0.0, 0.1],
        }
    )
    block_estimates = pd.DataFrame(
        {
            "anchor_id": ["anchor_001"] * 4,
            "block_id": [0, 1, 0, 1],
            "candidate_id": ["a", "a", "b", "b"],
            "known_index_value": [0.2, 0.2, 0.1, 0.1],
        }
    )
    separation = pd.DataFrame(
        {
            "anchor_id": ["anchor_001"],
            "averaged_block_rank_reliability": [0.9],
            "averaged_block_candidate_separation_icc": [0.8],
        }
    )

    _, diagnostics = _label_precision_diagnostics(
        worlds,
        block_estimates,
        separation,
        {
            "aggregate_label_reliability": 0.7,
            "aggregate_label_candidate_separation_icc": 0.4,
        },
    )

    assert bool(diagnostics.loc[0, "primary_label_ready"])
    assert diagnostics.loc[0, "diagnosis"] == "passed"
