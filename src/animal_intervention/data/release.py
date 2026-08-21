from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_manifest(data_root: Path, release_id: str) -> dict[str, Any]:
    """Freeze source-manifest and canonical-output hashes for every dataset."""

    datasets: list[dict[str, Any]] = []
    for dataset_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        source_manifest = dataset_dir / "manifest.json"
        processed_dir = dataset_dir / "processed"
        metadata_path = processed_dir / "dataset_metadata.json"
        validation_path = processed_dir / "validation_report.json"
        verification_path = processed_dir / "manifest_verification.json"
        required = [
            source_manifest,
            metadata_path,
            validation_path,
            verification_path,
        ]
        if not all(path.exists() for path in required):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        processed_files = {
            path.name: _sha256(path)
            for path in sorted(processed_dir.iterdir())
            if path.is_file()
        }
        datasets.append(
            {
                "dataset_id": metadata["dataset_id"],
                "adapter_name": metadata["adapter_name"],
                "adapter_version": metadata["adapter_version"],
                "has_temporal_order": bool(metadata["has_temporal_order"]),
                "primary_event_mode": metadata["primary_event_mode"],
                "source_manifest_sha256": _sha256(source_manifest),
                "source_payload_verification_status": verification["status"],
                "canonical_validation_has_errors": bool(validation["has_errors"]),
                "canonical_counts": {
                    key: int(validation["metrics"][key])
                    for key in [
                        "individuals",
                        "dyadic_events",
                        "group_events",
                        "group_memberships",
                        "observation_windows",
                    ]
                },
                "processed_files_sha256": processed_files,
            }
        )
    if not datasets:
        raise ValueError(f"No complete canonical datasets found under {data_root}")
    release_ready = all(
        item["source_payload_verification_status"] == "verified"
        and not item["canonical_validation_has_errors"]
        for item in datasets
    )
    return {
        "release_id": release_id,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hash_algorithm": "sha256",
        "dataset_count": len(datasets),
        "release_ready": release_ready,
        "datasets": datasets,
    }


def verify_release_manifest(data_root: Path, manifest: dict[str, Any]) -> list[str]:
    """Return human-readable mismatches against a frozen canonical release."""

    mismatches: list[str] = []
    for dataset in manifest["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        dataset_dir = data_root / dataset_id
        source_manifest = dataset_dir / "manifest.json"
        if not source_manifest.exists():
            mismatches.append(f"{dataset_id}: missing source manifest")
        elif _sha256(source_manifest) != dataset["source_manifest_sha256"]:
            mismatches.append(f"{dataset_id}: source manifest hash changed")
        processed_dir = dataset_dir / "processed"
        for name, expected in dataset["processed_files_sha256"].items():
            path = processed_dir / name
            if not path.exists():
                mismatches.append(f"{dataset_id}: missing processed file {name}")
            elif _sha256(path) != expected:
                mismatches.append(f"{dataset_id}: processed hash changed for {name}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify a canonical data release")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/_shared/canonical_release.json")
    )
    parser.add_argument("--release-id")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        mismatches = verify_release_manifest(args.data_root, manifest)
        if mismatches:
            raise SystemExit("\n".join(mismatches))
        print(f"Verified {len(manifest['datasets'])} canonical datasets")
        return
    if not args.release_id:
        parser.error("--release-id is required when creating a release")
    manifest = build_release_manifest(args.data_root, args.release_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen {manifest['dataset_count']} canonical datasets at {args.output}")


if __name__ == "__main__":
    main()
