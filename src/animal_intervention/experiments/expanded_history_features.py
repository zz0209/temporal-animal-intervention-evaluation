from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from animal_intervention.centrality import build_history_features
from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.transmission.mappers import compile_primary_exposure

from .additional_system_validation import _sheep_stream


KEYS = ["dataset_id", "network_id", "anchor_time", "candidate_id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    labels_path = Path(config["data"]["label_path"])
    existing_path = Path(config["data"]["existing_feature_path"])
    labels = pd.read_csv(labels_path, dtype={"candidate_id": str})
    existing = pd.read_csv(existing_path, dtype={"candidate_id": str})
    labels["anchor_time"] = pd.to_datetime(labels["anchor_time"], format="mixed")
    existing["anchor_time"] = pd.to_datetime(existing["anchor_time"], format="mixed")
    new_tables: list[pd.DataFrame] = []
    stream_audits: list[dict[str, Any]] = []
    for dataset_id in config["data"]["new_datasets"]:
        dataset_labels = labels.loc[labels["dataset_id"].eq(dataset_id)].copy()
        dataset = CanonicalDataset.read(
            Path(config["data"]["canonical_root"]) / dataset_id / "processed"
        )
        stream = (
            _sheep_stream(dataset, {"data": {"group_mixing_mode": "frequency_dependent"}})
            if dataset_id == "free_ranging_sheep_fission_fusion"
            else compile_primary_exposure(dataset)
        )
        features = build_history_features(dataset, dataset_labels, exposure_stream=stream)
        new_tables.append(features)
        stream_audits.append(
            {
                "dataset_id": dataset_id,
                "mapper": stream.metadata.get("mapper"),
                "label_rows": int(len(dataset_labels)),
                "feature_rows": int(len(features)),
            }
        )
    retained = existing.loc[
        ~existing["dataset_id"].isin(config["data"]["new_datasets"])
    ].copy()
    combined = pd.concat([retained, *new_tables], ignore_index=True)
    expected = labels[KEYS].copy()
    observed = combined[KEYS].copy()
    audit = {
        "status": "pass",
        "checks": {
            "feature_keys_unique": not combined.duplicated(KEYS).any(),
            "label_keys_unique": not labels.duplicated(KEYS).any(),
            "all_label_rows_covered": len(expected.merge(observed, on=KEYS, how="inner"))
            == len(expected)
            == len(observed),
            "new_datasets_present": set(config["data"]["new_datasets"])
            <= set(combined["dataset_id"]),
        },
        "datasets": int(combined["dataset_id"].nunique()),
        "rows": int(len(combined)),
        "new_streams": stream_audits,
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(audit)
    experiment_id = str(config["experiment"]["id"])
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / "full"
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / "full"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for directory in (results_dir, report_dir):
        combined.to_csv(directory / "history_features.csv", index=False)
        (directory / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    manifest = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": _sha256(config_path),
        "labels_sha256": _sha256(labels_path),
        "existing_features_sha256": _sha256(existing_path),
        "output_rows": len(combined),
        "audit_status": audit["status"],
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded past-only history features")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=2))


if __name__ == "__main__":
    main()
