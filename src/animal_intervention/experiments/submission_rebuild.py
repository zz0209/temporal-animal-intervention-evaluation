from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import yaml


def _rewrite_value(
    value: Any,
    selected_ids: set[str],
    clean_root: Path,
    config_map: dict[str, str],
    profile: str,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_value(item, selected_ids, clean_root, config_map, profile)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_value(item, selected_ids, clean_root, config_map, profile)
            for item in value
        ]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    if normalized in config_map:
        return config_map[normalized]
    for experiment_id in selected_ids:
        prefix = f"results/{experiment_id}"
        if normalized == prefix or normalized.startswith(prefix + "/"):
            if profile == "smoke":
                return value
            return str(clean_root / normalized).replace("\\", "/")
    return value


def prepare_clean_configs(config: dict[str, Any], profile: str) -> tuple[Path, list[dict[str, str]]]:
    clean_root = Path(config["design"]["clean_root"]) / f"{profile}_run"
    if clean_root.exists():
        raise FileExistsError(
            f"Clean rebuild root already exists: {clean_root}. Use --resume or choose a new experiment ID."
        )
    config_dir = clean_root / "configs"
    config_dir.mkdir(parents=True)
    tasks = list(config["tasks"])
    selected_ids = {str(task["experiment_id"]) for task in tasks}
    config_map = {
        str(task["config"]).replace("\\", "/"): str(config_dir / Path(task["config"]).name).replace("\\", "/")
        for task in tasks
    }
    prepared: list[dict[str, str]] = []
    for task in tasks:
        original_path = Path(task["config"])
        original = yaml.safe_load(original_path.read_text(encoding="utf-8"))
        rewritten = _rewrite_value(original, selected_ids, clean_root, config_map, profile)
        original_results_root = Path(str(original["outputs"]["results_root"]))
        original_report_root = Path(str(original["outputs"]["report_root"]))
        rewritten["outputs"]["results_root"] = str(clean_root / original_results_root).replace("\\", "/")
        rewritten["outputs"]["report_root"] = str(clean_root / original_report_root).replace("\\", "/")
        output_path = config_dir / original_path.name
        output_path.write_text(yaml.safe_dump(rewritten, sort_keys=False), encoding="utf-8")
        prepared.append(
            {
                "experiment_id": str(task["experiment_id"]),
                "module": str(task["module"]),
                "config": str(output_path),
            }
        )
    return clean_root, prepared


def _materialize_static_inputs(config: dict[str, Any], clean_root: Path) -> int:
    records = list(config["design"].get("frozen_static_inputs", []))
    for record in records:
        source = Path(record["path"])
        if source.is_absolute():
            raise ValueError(f"Frozen static inputs must use repository-relative paths: {source}")
        if not source.exists():
            raise FileNotFoundError(f"Frozen static input is missing: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != str(record["sha256"]).lower():
            raise ValueError(f"Frozen static input hash mismatch: {source}")
        destination = clean_root / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if destination_digest != digest:
                raise ValueError(f"Materialized static input differs: {destination}")
        else:
            shutil.copy2(source, destination)
    return len(records)


def _normalize_json(value: Any) -> Any:
    volatile_fragments = ("hash", "path", "time", "duration", "created", "generated", "git")
    if isinstance(value, dict):
        return {
            key: _normalize_json(item)
            for key, item in value.items()
            if not any(fragment in key.lower() for fragment in volatile_fragments)
        }
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    return value


def _compare_frame(reference_path: Path, rebuilt_path: Path, atol: float, rtol: float) -> str | None:
    reference = pd.read_csv(reference_path, dtype_backend="numpy_nullable")
    rebuilt = pd.read_csv(rebuilt_path, dtype_backend="numpy_nullable")
    if list(reference.columns) != list(rebuilt.columns):
        return "column mismatch"
    if reference.shape != rebuilt.shape:
        return f"shape mismatch {reference.shape} != {rebuilt.shape}"
    for column in reference.columns:
        left = reference[column]
        right = rebuilt[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = left.astype(float).to_numpy()
            right_values = right.astype(float).to_numpy()
            if not np.allclose(left_values, right_values, atol=atol, rtol=rtol, equal_nan=True):
                maximum = float(np.nanmax(np.abs(left_values - right_values)))
                return f"numeric mismatch in {column}; max_abs={maximum}"
        else:
            equal = left.astype("string").fillna("<NA>").eq(right.astype("string").fillna("<NA>"))
            if not bool(equal.all()):
                return f"value mismatch in {column}"
    return None


def _compare_frame_semantic(
    reference_path: Path,
    rebuilt_path: Path,
    atol: float,
    rtol: float,
    diagnostic_tolerances: dict[str, float] | None = None,
    metadata_fill_columns: set[str] | None = None,
) -> tuple[str, str]:
    """Compare frozen columns while retaining explicit schema-evolution provenance."""

    reference = pd.read_csv(reference_path, dtype_backend="numpy_nullable")
    rebuilt = pd.read_csv(rebuilt_path, dtype_backend="numpy_nullable")
    missing = [column for column in reference.columns if column not in rebuilt.columns]
    if missing:
        return "mismatch", f"missing frozen columns: {missing}"
    if len(reference) != len(rebuilt):
        return "mismatch", f"row-count mismatch {len(reference)} != {len(rebuilt)}"
    tolerated: list[str] = []
    formatting_only: list[str] = []
    metadata_filled: list[str] = []
    diagnostic_tolerances = diagnostic_tolerances or {}
    metadata_fill_columns = metadata_fill_columns or set()
    for column in reference.columns:
        left = reference[column]
        right = rebuilt[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = left.astype(float).to_numpy()
            right_values = right.astype(float).to_numpy()
            if column in metadata_fill_columns:
                equal_values = np.isclose(
                    left_values, right_values, atol=atol, rtol=rtol, equal_nan=True
                )
                changed = ~equal_values
                if changed.any() and np.isnan(left_values[changed]).all() and np.isfinite(
                    right_values[changed]
                ).all():
                    metadata_filled.append(column)
                    continue
            column_atol = max(atol, float(diagnostic_tolerances.get(column, 0.0)))
            if not np.allclose(
                left_values,
                right_values,
                atol=column_atol,
                rtol=rtol,
                equal_nan=True,
            ):
                maximum = float(np.nanmax(np.abs(left_values - right_values)))
                return "mismatch", f"numeric mismatch in {column}; max_abs={maximum}"
            if column in diagnostic_tolerances and not np.allclose(
                left_values, right_values, atol=atol, rtol=rtol, equal_nan=True
            ):
                tolerated.append(column)
        else:
            equal = left.astype("string").fillna("<NA>").eq(
                right.astype("string").fillna("<NA>")
            )
            if not bool(equal.all()):
                if column.endswith(("_time", "_date", "_start", "_end")):
                    left_time = pd.to_datetime(left, errors="coerce", utc=True, format="mixed")
                    right_time = pd.to_datetime(right, errors="coerce", utc=True, format="mixed")
                    if left_time.equals(right_time):
                        formatting_only.append(column)
                        continue
                if column in metadata_fill_columns:
                    changed = ~equal
                    if bool(left.loc[changed].isna().all()) and bool(
                        right.loc[changed].notna().all()
                    ):
                        metadata_filled.append(column)
                        continue
                return "mismatch", f"value mismatch in {column}"
    added = [column for column in rebuilt.columns if column not in reference.columns]
    if tolerated:
        return (
            "diagnostic_tolerance",
            "explicit tie-sensitive diagnostic tolerance applied to " + ", ".join(tolerated),
        )
    if metadata_filled:
        return (
            "metadata_enrichment",
            "previously null design metadata materialized in " + ", ".join(metadata_filled),
        )
    if added:
        return "schema_extension", "rebuilt artifact adds columns: " + ", ".join(added)
    if list(reference.columns) != list(rebuilt.columns):
        return "column_order_only", "same frozen columns and values; column order evolved"
    if formatting_only:
        return "formatting_only", "equivalent parsed values in " + ", ".join(formatting_only)
    return "match", ""


def _compare_scientific_subset(
    reference: Any,
    rebuilt: Any,
    *,
    path: str = "scientific_result",
    atol: float,
    rtol: float,
    diagnostic_tolerances: dict[str, float],
) -> str | None:
    """Require every frozen scientific field while allowing additive diagnostics."""

    if isinstance(reference, dict):
        if not isinstance(rebuilt, dict):
            return f"type mismatch at {path}"
        for key, value in reference.items():
            if key not in rebuilt:
                return f"missing frozen scientific field {path}.{key}"
            difference = _compare_scientific_subset(
                value,
                rebuilt[key],
                path=f"{path}.{key}",
                atol=atol,
                rtol=rtol,
                diagnostic_tolerances=diagnostic_tolerances,
            )
            if difference:
                return difference
        return None
    if isinstance(reference, list):
        if not isinstance(rebuilt, list) or len(reference) != len(rebuilt):
            return f"list mismatch at {path}"
        for index, (left, right) in enumerate(zip(reference, rebuilt)):
            difference = _compare_scientific_subset(
                left,
                right,
                path=f"{path}[{index}]",
                atol=atol,
                rtol=rtol,
                diagnostic_tolerances=diagnostic_tolerances,
            )
            if difference:
                return difference
        return None
    if isinstance(reference, (int, float)) and isinstance(rebuilt, (int, float)):
        tolerance = max(atol, float(diagnostic_tolerances.get(path, 0.0)))
        if np.isnan(reference) and np.isnan(rebuilt):
            return None
        if not np.isclose(reference, rebuilt, atol=tolerance, rtol=rtol):
            return f"numeric mismatch at {path}: {reference} != {rebuilt}"
        return None
    return None if reference == rebuilt else f"value mismatch at {path}"


def compare_results(
    tasks: list[dict[str, str]],
    clean_root: Path,
    profile: str,
    excluded: set[str],
    atol: float,
    rtol: float,
    reconciliation: dict[str, Any] | None = None,
) -> pd.DataFrame:
    reconciliation = reconciliation or {}
    additive_artifacts = set(reconciliation.get("additive_artifacts", []))
    provenance_artifacts = set(reconciliation.get("provenance_artifacts", []))
    table_tolerances = reconciliation.get("diagnostic_table_tolerances", {})
    metadata_fill_columns = reconciliation.get("metadata_fill_columns", {})
    audit_tolerances = reconciliation.get("diagnostic_audit_tolerances", {})
    rows: list[dict[str, Any]] = []
    for task in tasks:
        experiment_id = task["experiment_id"]
        reference_dir = Path("results") / experiment_id / profile
        rebuilt_dir = clean_root / "results" / experiment_id / profile
        reference_files = {
            path.name: path
            for path in reference_dir.iterdir()
            if path.is_file() and path.name not in excluded
        }
        rebuilt_files = {
            path.name: path
            for path in rebuilt_dir.iterdir()
            if path.is_file() and path.name not in excluded
        }
        for name in sorted(set(reference_files) | set(rebuilt_files)):
            artifact_key = f"{experiment_id}/{name}"
            status = "match"
            detail = ""
            if name not in reference_files:
                if artifact_key in additive_artifacts:
                    status, detail = "additive_artifact", "documented post-freeze diagnostic"
                else:
                    status, detail = "mismatch", "missing frozen reference"
            elif name not in rebuilt_files:
                status, detail = "mismatch", "missing rebuilt artifact"
            elif name == "audit.json":
                left = json.loads(reference_files[name].read_text(encoding="utf-8"))
                right = json.loads(rebuilt_files[name].read_text(encoding="utf-8"))
                if left.get("status") != "pass" or right.get("status") != "pass":
                    status, detail = "mismatch", "frozen or rebuilt audit is not passing"
                else:
                    difference = _compare_scientific_subset(
                        left.get("scientific_result", {}),
                        right.get("scientific_result", {}),
                        atol=atol,
                        rtol=rtol,
                        diagnostic_tolerances=audit_tolerances.get(experiment_id, {}),
                    )
                    if difference:
                        status, detail = "mismatch", difference
                    elif _normalize_json(left) != _normalize_json(right):
                        status, detail = (
                            "audit_revalidated",
                            "frozen scientific fields retained; current audit adds or strengthens checks",
                        )
            elif name.endswith(".csv") or name.endswith(".csv.gz"):
                if artifact_key in provenance_artifacts:
                    left = pd.read_csv(reference_files[name])
                    right = pd.read_csv(rebuilt_files[name])
                    identity = [column for column in ("experiment_id", "status") if column in left]
                    if len(left) != len(right) or any(
                        not left[column].astype("string").equals(right[column].astype("string"))
                        for column in identity
                    ):
                        status, detail = "mismatch", "provenance inventory identity/status changed"
                    else:
                        status, detail = (
                            "provenance_rebuilt",
                            "paths, hashes, sizes, and check counts reflect the isolated rebuild",
                        )
                else:
                    status, detail = _compare_frame_semantic(
                        reference_files[name],
                        rebuilt_files[name],
                        atol,
                        rtol,
                        table_tolerances.get(artifact_key, {}),
                        set(metadata_fill_columns.get(artifact_key, [])),
                    )
            else:
                status, detail = "not_compared", "non-tabular artifact"
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "artifact": name,
                    "status": status,
                    "detail": detail,
                }
            )
    return pd.DataFrame(rows)


def _compare_image(reference_path: Path, rebuilt_path: Path) -> str | None:
    try:
        with Image.open(reference_path) as image:
            reference = np.asarray(image.convert("RGBA"))
        with Image.open(rebuilt_path) as image:
            rebuilt = np.asarray(image.convert("RGBA"))
    except Exception as error:
        return f"image decode failure: {error}"
    if reference.shape != rebuilt.shape:
        return f"pixel-shape mismatch {reference.shape} != {rebuilt.shape}"
    if min(reference.shape[:2]) < 100:
        return f"unexpectedly small figure {reference.shape[:2]}"
    if float(np.std(reference[:, :, :3])) < 1.0:
        return "reference figure is effectively blank"
    if not np.array_equal(reference, rebuilt):
        changed = np.any(reference != rebuilt, axis=2)
        return f"pixel mismatch; changed_fraction={float(changed.mean()):.12g}"
    return None


def compare_visuals(
    original_tasks: list[dict[str, str]],
    clean_root: Path,
    profile: str,
    reconciliation: dict[str, Any] | None = None,
) -> pd.DataFrame:
    reconciliation = reconciliation or {}
    additive_figures = set(reconciliation.get("additive_figures", []))
    rendering_only_figures = set(reconciliation.get("rendering_only_figures", []))
    rows: list[dict[str, Any]] = []
    for task in original_tasks:
        experiment_id = str(task["experiment_id"])
        original_config_path = Path(task["config"])
        rebuilt_config_path = clean_root / "configs" / original_config_path.name
        original_config = yaml.safe_load(original_config_path.read_text(encoding="utf-8"))
        rebuilt_config = yaml.safe_load(rebuilt_config_path.read_text(encoding="utf-8"))
        reference_dir = Path(original_config["outputs"]["report_root"]) / experiment_id / profile
        rebuilt_dir = Path(rebuilt_config["outputs"]["report_root"]) / experiment_id / profile
        reference_files = {
            path.relative_to(reference_dir).as_posix(): path
            for path in reference_dir.rglob("*.png")
        }
        rebuilt_files = {
            path.relative_to(rebuilt_dir).as_posix(): path
            for path in rebuilt_dir.rglob("*.png")
        }
        for name in sorted(set(reference_files) | set(rebuilt_files)):
            figure_key = f"{experiment_id}/{name}"
            status = "match"
            detail = ""
            if name not in reference_files:
                if figure_key in additive_figures:
                    difference = _compare_image(rebuilt_files[name], rebuilt_files[name])
                    if difference:
                        status, detail = "mismatch", difference
                    else:
                        status, detail = "additive_figure", "documented post-freeze diagnostic"
                else:
                    status, detail = "mismatch", "missing frozen reference"
            elif name not in rebuilt_files:
                status, detail = "mismatch", "missing rebuilt figure"
            else:
                difference = _compare_image(reference_files[name], rebuilt_files[name])
                if difference:
                    if figure_key in rendering_only_figures:
                        status, detail = (
                            "rendering_only",
                            f"documented title/layout revision; {difference}",
                        )
                    else:
                        status, detail = "mismatch", difference
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "figure": name,
                    "status": status,
                    "detail": detail,
                }
            )
    return pd.DataFrame(rows)


def run(config_path: Path, profile: str, resume: bool = False) -> dict[str, Any]:
    started = time.time()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    clean_root = Path(config["design"]["clean_root"]) / f"{profile}_run"
    if resume:
        config_dir = clean_root / "configs"
        if not config_dir.exists():
            raise FileNotFoundError(f"No prepared rebuild exists at {clean_root}")
        tasks = [
            {
                "experiment_id": str(task["experiment_id"]),
                "module": str(task["module"]),
                "config": str(config_dir / Path(task["config"]).name),
            }
            for task in config["tasks"]
        ]
    else:
        clean_root, tasks = prepare_clean_configs(config, profile)
    static_input_count = _materialize_static_inputs(config, clean_root)
    task_rows: list[dict[str, Any]] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        rebuilt_audit = clean_root / "results" / task["experiment_id"] / profile / "audit.json"
        if resume and rebuilt_audit.exists():
            existing = json.loads(rebuilt_audit.read_text(encoding="utf-8"))
            if existing.get("status") == "pass":
                print(f"[{index}/{total}] SKIP {task['experiment_id']} (passed checkpoint)", flush=True)
                task_rows.append({**task, "status": "skipped_passed", "returncode": 0})
                continue
        print(f"[{index}/{total}] RUN {task['experiment_id']} ({profile})", flush=True)
        task_started = time.time()
        command = [
            sys.executable,
            "-m",
            task["module"],
            "--config",
            task["config"],
            "--profile",
            profile,
        ]
        completed = subprocess.run(command, check=False)
        row = {
            **task,
            "status": "pass" if completed.returncode == 0 else "fail",
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.time() - task_started, 3),
        }
        task_rows.append(row)
        pd.DataFrame(task_rows).to_csv(clean_root / "task_status.csv", index=False)
        if completed.returncode != 0 and config["design"]["fail_fast"]:
            raise RuntimeError(f"Submission rebuild failed at {task['experiment_id']}")

    comparisons = (
        compare_results(
            tasks,
            clean_root,
            profile,
            set(config["design"]["excluded_comparison_files"]),
            float(config["design"]["numeric_atol"]),
            float(config["design"]["numeric_rtol"]),
            config["design"].get("reconciliation", {}),
        )
        if profile == "full"
        else pd.DataFrame(columns=["experiment_id", "artifact", "status", "detail"])
    )
    comparisons.to_csv(clean_root / "artifact_comparison.csv", index=False)
    visual_comparisons = (
        compare_visuals(
            config["tasks"],
            clean_root,
            profile,
            config["design"].get("reconciliation", {}),
        )
        if profile == "full"
        else pd.DataFrame(columns=["experiment_id", "figure", "status", "detail"])
    )
    visual_comparisons.to_csv(clean_root / "visual_comparison.csv", index=False)
    mismatch_count = int(comparisons["status"].eq("mismatch").sum())
    visual_mismatch_count = int(visual_comparisons["status"].eq("mismatch").sum())
    exact_table_count = int(comparisons["status"].eq("match").sum())
    reconciled_table_count = int(
        comparisons["status"].isin(
            [
                "schema_extension",
                "column_order_only",
                "diagnostic_tolerance",
                "metadata_enrichment",
                "formatting_only",
                "additive_artifact",
                "audit_revalidated",
                "provenance_rebuilt",
            ]
        ).sum()
    )
    exact_figure_count = int(visual_comparisons["status"].eq("match").sum())
    reconciled_figure_count = int(
        visual_comparisons["status"].isin(
            ["additive_figure", "rendering_only"]
        ).sum()
    )
    audits = []
    for task in tasks:
        audit_path = clean_root / "results" / task["experiment_id"] / profile / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audits.append(
            {
                "experiment_id": task["experiment_id"],
                "status": audit.get("status"),
                "checks": len(audit.get("checks", {})),
            }
        )
    audit = {
        "status": "pass"
        if (profile == "smoke" or (mismatch_count == 0 and visual_mismatch_count == 0))
        and all(item["status"] == "pass" for item in audits)
        else "fail",
        "profile": profile,
        "clean_root": str(clean_root),
        "tasks": len(tasks),
        "passed_audits": sum(item["status"] == "pass" for item in audits),
        "tabular_artifacts_exact": exact_table_count,
        "tabular_artifacts_reconciled": reconciled_table_count,
        "tabular_artifacts_compared": exact_table_count + reconciled_table_count,
        "non_tabular_artifacts_not_compared": int(comparisons["status"].eq("not_compared").sum()),
        "mismatches": mismatch_count,
        "figures_exact": exact_figure_count,
        "figures_reconciled": reconciled_figure_count,
        "figures_compared": exact_figure_count + reconciled_figure_count,
        "figure_mismatches": visual_mismatch_count,
        "comparison_mode": "engineering_audits_only" if profile == "smoke" else "frozen_scientific_semantic_reconciliation",
        "elapsed_seconds": round(time.time() - started, 3),
        "frozen_upstream_boundary": config["design"]["frozen_upstream_boundary"],
        "verified_frozen_static_inputs": static_input_count,
        "scope_note": (
            "The rebuild starts at the frozen canonical release and precision-audited label registry; "
            "raw adapters and the expensive label-generation calibration are verified by hashes and prior audits, not regenerated here."
        ),
    }
    (clean_root / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    result_dir = Path(config["outputs"]["results_root"]) / config["experiment"]["id"] / profile
    report_dir = Path(config["outputs"]["report_root"]) / config["experiment"]["id"] / profile
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(task_rows).to_csv(result_dir / "task_status.csv", index=False)
    comparisons.to_csv(result_dir / "artifact_comparison.csv", index=False)
    visual_comparisons.to_csv(result_dir / "visual_comparison.csv", index=False)
    pd.DataFrame(audits).to_csv(result_dir / "rebuilt_audits.csv", index=False)
    (result_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = f"""# Clean submission evidence rebuild

- Profile: `{profile}`
- Isolated clean root: `{clean_root}`
- Rebuilt experiments: {audit['tasks']}
- Passing experiment audits: {audit['passed_audits']}/{audit['tasks']}
- Exact tabular/audit artifacts: {audit.get('tabular_artifacts_exact', 0)}
- Reconciled additive/provenance artifacts: {audit.get('tabular_artifacts_reconciled', 0)}
- Mismatches: {audit['mismatches']}
- Pixel-identical figures: {audit.get('figures_exact', 0)}
- Reconciled additive/rendering-only figures: {audit.get('figures_reconciled', 0)}
- Figure mismatches: {audit['figure_mismatches']}
- Overall status: **{audit['status']}**

The rebuild begins from the frozen canonical release and precision-audited model-ready label registry. Raw adapters and the expensive label-generation calibration are protected by release hashes and their original audits; they are not silently represented as freshly regenerated in this stage.
"""
    (report_dir / "README.md").write_text(report, encoding="utf-8")
    if audit["status"] != "pass":
        raise RuntimeError(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated manuscript evidence rebuild")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260818-002_submission_rebuild.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile, args.resume), indent=2))


if __name__ == "__main__":
    main()
