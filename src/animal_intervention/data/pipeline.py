from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
from tqdm.auto import tqdm

from .adapters import ADAPTERS
from .reporting import save_quality_report
from .validation import validate_dataset


def verify_manifest(
    dataset_dir: Path,
    *,
    verify_sha256: bool,
    progress: bool,
) -> dict:
    """Verify immutable raw payloads against the reviewed local manifest."""
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_dir.name:
        raise ValueError(f"manifest dataset_id does not match directory {dataset_dir.name}")
    entries = list(manifest.get("files", []))
    if "archive" in manifest:
        entries.append(manifest["archive"])
    checks = []
    for entry in entries:
        path = (dataset_dir / entry["path"]).resolve()
        if not path.is_relative_to(dataset_dir.resolve()):
            raise ValueError(f"manifest path escapes dataset directory: {entry['path']}")
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(entry["bytes"]):
            raise ValueError(
                f"size mismatch for {entry['path']}: observed {size}, expected {entry['bytes']}"
            )
        observed_sha256 = None
        if verify_sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream, tqdm(
                total=size,
                desc=f"verify {path.name}",
                unit="B",
                unit_scale=True,
                disable=not progress,
            ) as bar:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    bar.update(len(chunk))
            observed_sha256 = digest.hexdigest()
            expected_sha256 = entry.get("sha256")
            if expected_sha256 and observed_sha256 != expected_sha256:
                raise ValueError(f"SHA-256 mismatch for {entry['path']}")
        checks.append(
            {
                "path": entry["path"],
                "bytes": size,
                "size_verified": True,
                "sha256_verified": bool(verify_sha256 and entry.get("sha256")),
                "observed_sha256": observed_sha256,
            }
        )
    return {
        "dataset_id": dataset_dir.name,
        "manifest": str(manifest_path),
        "status": "verified",
        "sha256_checked": verify_sha256,
        "files": checks,
    }


def process_dataset(
    dataset_id: str,
    *,
    data_root: Path,
    mode: str,
    reports_dir: Path,
    progress: bool = True,
) -> dict:
    started = time.perf_counter()
    adapter = ADAPTERS[dataset_id]()
    dataset_dir = data_root / dataset_id
    output_dir = dataset_dir / ("interim/smoke" if mode == "smoke" else "processed")
    provenance = verify_manifest(
        dataset_dir,
        verify_sha256=mode == "full",
        progress=progress,
    )
    dataset = adapter.load(dataset_dir / "raw", sample=mode == "smoke", progress=progress)
    validation = validate_dataset(dataset)
    dataset.write(output_dir)
    (output_dir / "manifest_verification.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if mode == "full":
        save_quality_report(dataset, validation, reports_dir)
    if validation.has_errors:
        raise ValueError(f"{dataset_id} failed high-severity validation; inspect validation_report.json")
    result = dataset.summary()
    result.update(
        {
            "status": "ok",
            "validation_issues": len(validation.issues),
            "validation_issue_codes": [issue.code for issue in validation.issues],
            "source_manifest_status": provenance["status"],
            "source_sha256_checked": provenance["sha256_checked"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "output_dir": str(output_dir),
        }
    )
    return result


def write_quality_index(data_root: Path, reports_dir: Path) -> None:
    """Rebuild the cross-dataset index from every available processed artifact."""
    rows = []
    for dataset_id in ADAPTERS:
        processed = data_root / dataset_id / "processed"
        required = [
            processed / "summary.json",
            processed / "validation_report.json",
            processed / "manifest_verification.json",
        ]
        if not all(path.is_file() for path in required):
            continue
        summary = json.loads(required[0].read_text(encoding="utf-8"))
        validation = json.loads(required[1].read_text(encoding="utf-8"))
        provenance = json.loads(required[2].read_text(encoding="utf-8"))
        summary.update(
            {
                "validation_issues": len(validation["issues"]),
                "validation_issue_codes": [issue["code"] for issue in validation["issues"]],
                "source_manifest_status": provenance["status"],
                "source_sha256_checked": provenance["sha256_checked"],
            }
        )
        rows.append(summary)
    if not rows:
        return
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(reports_dir / "dataset_summary.csv", index=False)
    summary_columns = [
        "dataset_id",
        "individuals",
        "dyadic_events",
        "group_events",
        "group_memberships",
        "time_start",
        "time_end",
        "validation_issues",
        "source_manifest_status",
    ]
    summary = pd.DataFrame(rows)[summary_columns]
    header = "| " + " | ".join(summary_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(summary_columns)) + " |"
    table_rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in summary.itertuples(index=False, name=None)
    ]
    (reports_dir / "README.md").write_text(
        "# Canonical dataset quality summary\n\n"
        + "\n".join([header, separator, *table_rows])
        + "\n\nEach dataset has a detailed Markdown audit and diagnostic figure in this directory.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonicalize and audit Temporal Animal Intervention Evaluation datasets")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="process every registered dataset")
    selection.add_argument("--dataset", action="append", choices=sorted(ADAPTERS))
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/data_quality"))
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_ids = list(ADAPTERS) if args.all else args.dataset
    results = []
    failures = []
    for dataset_id in tqdm(
        dataset_ids,
        desc=f"Temporal Animal Intervention Evaluation {args.mode} adapters",
        disable=args.no_progress,
        unit="dataset",
    ):
        try:
            results.append(
                process_dataset(
                    dataset_id,
                    data_root=args.data_root,
                    mode=args.mode,
                    reports_dir=args.reports_dir,
                    progress=not args.no_progress,
                )
            )
        except Exception as error:  # surface all dataset failures in one batch summary
            failures.append({"dataset_id": dataset_id, "error": f"{type(error).__name__}: {error}"})
    if args.mode == "full" and results:
        write_quality_index(args.data_root, args.reports_dir)
    print(json.dumps({"mode": args.mode, "successes": results, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
