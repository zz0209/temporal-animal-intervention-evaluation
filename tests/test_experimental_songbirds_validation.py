from __future__ import annotations

import json
from pathlib import Path

import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.experiments.experimental_songbirds_validation import (
    _data_quality_audit,
    _observed_group_stream,
    _prepare_windows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "EXP-20260816-001_experimental_songbirds_validation.yaml"


def _prepared_data():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = CanonicalDataset.read(ROOT / config["data"]["canonical_path"])
    stream = _observed_group_stream(dataset, config)
    prepared, metadata = _prepare_windows(stream, config["windows"], max_anchors=13)
    return dataset, stream, prepared, metadata


def test_songbird_windows_do_not_cross_manipulation_boundary() -> None:
    _, _, prepared, metadata = _prepared_data()

    assert len(prepared) == 13
    assert metadata["observation_phase"].value_counts().to_dict() == {
        "during": 10,
        "pre": 3,
    }
    assert metadata["network_id"].nunique() == 1
    assert metadata["future_active_fraction"].ge(0.70).all()
    for window in prepared:
        phase = window["observation_unit_id"]
        assert window["anchor"].anchor_id.startswith(f"{phase}::")
        assert window["history_period_support"].reindex(window["eligible"]).ge(2).all()
        assert window["future"].nodes() == set(window["eligible"])


def test_songbird_quality_audit_reconciles_published_phase_counts() -> None:
    dataset, stream, prepared, _ = _prepared_data()

    audit, daily, group_sizes = _data_quality_audit(dataset, stream, prepared)

    assert audit["status"] == "passed"
    assert audit["raw_group_events_by_phase"] == {"during": 52_483, "pre": 10_954}
    assert audit["raw_memberships_by_phase"] == {"during": 187_232, "pre": 50_201}
    assert audit["raw_individuals_by_phase"] == {"during": 339, "pre": 240}
    assert audit["excluded_nonpositive_group_events"] == 170
    assert audit["during_feeder_tag_match_fraction"] > 0.70
    assert audit["linked_wytham_observation_overlap"]["matched_songbird_events"] == 435
    assert not daily.empty
    assert len(group_sizes) == 63_267


def test_songbird_adapter_retains_random_tag_parity() -> None:
    dataset, _, _, _ = _prepared_data()
    parities = {
        json.loads(value)["tag_parity"] for value in dataset.individuals["attributes_json"]
    }
    assert parities == {"even", "odd"}
