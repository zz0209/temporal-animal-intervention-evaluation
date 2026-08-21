from __future__ import annotations

import json

from animal_intervention.data.release import build_release_manifest, verify_release_manifest


def test_canonical_release_detects_changed_processed_file(tmp_path) -> None:
    dataset = tmp_path / "example"
    processed = dataset / "processed"
    processed.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    (processed / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "example",
                "adapter_name": "ExampleAdapter",
                "adapter_version": "1",
                "has_temporal_order": True,
                "primary_event_mode": "dyadic",
            }
        ),
        encoding="utf-8",
    )
    (processed / "validation_report.json").write_text(
        json.dumps(
            {
                "has_errors": False,
                "metrics": {
                    "individuals": 2,
                    "dyadic_events": 1,
                    "group_events": 0,
                    "group_memberships": 0,
                    "observation_windows": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (processed / "manifest_verification.json").write_text(
        json.dumps({"status": "verified"}), encoding="utf-8"
    )
    manifest = build_release_manifest(tmp_path, "test")
    assert manifest["release_ready"] is True
    assert verify_release_manifest(tmp_path, manifest) == []
    (processed / "dataset_metadata.json").write_text("{}", encoding="utf-8")
    assert verify_release_manifest(tmp_path, manifest) == [
        "example: processed hash changed for dataset_metadata.json"
    ]
