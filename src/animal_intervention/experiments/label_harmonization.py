from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.label_contract import (
    LABEL_CONTRACT_VERSION,
    build_model_ready_labels,
    validate_model_ready_labels,
)
from animal_intervention.evaluation import aggregate_label_precision
from animal_intervention.transmission.mappers import (
    CoalescedDurationContactMapper,
    compile_primary_exposure,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reconstruction_error(
    labels: pd.DataFrame, block_estimates: pd.DataFrame
) -> float:
    keys = ["anchor_id", "candidate_id"]
    if "network_id" in labels.columns and "network_id" in block_estimates.columns:
        keys.insert(0, "network_id")
    reconstructed = (
        block_estimates.groupby(
            keys, observed=True, as_index=False
        )["unconditional_value"]
        .mean()
        .rename(columns={"unconditional_value": "reconstructed_value"})
    )
    compared = labels.merge(
        reconstructed,
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    if compared[["robust_intervention_value", "reconstructed_value"]].isna().any().any():
        raise ValueError("robust labels do not reconcile with block estimates")
    return float(
        (
            compared["robust_intervention_value"]
            - compared["reconstructed_value"]
        )
        .abs()
        .max()
    )


def _update_source_artifacts(
    *,
    source_config: dict[str, Any],
    source_config_path: Path,
    profile: str,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    experiment_id = str(source_config["experiment"]["id"])
    results_dir = (
        root / source_config["outputs"]["results_root"] / experiment_id / profile
    )
    report_dir = (
        root / source_config["outputs"]["report_root"] / experiment_id / profile
    )
    labels = pd.read_csv(
        results_dir / "robust_anchor_labels.csv", dtype={"candidate_id": str}
    )
    block_estimates = pd.read_csv(
        results_dir / "block_estimates.csv", dtype={"candidate_id": str}
    )
    dataset = CanonicalDataset.read(root / source_config["data"]["canonical_path"])
    mapper_name = str(source_config["data"].get("mapper", "primary"))
    stream = (
        CoalescedDurationContactMapper().compile(dataset)
        if mapper_name == "coalesced_duration"
        else compile_primary_exposure(dataset)
    )
    anchor_metadata_path = results_dir / "anchor_metadata.csv"
    anchor_metadata = (
        pd.read_csv(anchor_metadata_path)
        if anchor_metadata_path.exists()
        else None
    )
    model_ready = build_model_ready_labels(
        labels=labels,
        block_estimates=block_estimates,
        stream=stream,
        config=source_config,
        profile=profile,
        anchor_metadata=anchor_metadata,
    )
    contract_audit = validate_model_ready_labels(model_ready)
    contract_audit["label_reconstruction_max_abs_error"] = _reconstruction_error(
        labels, block_estimates
    )
    contract_audit["source_config"] = source_config_path.relative_to(root).as_posix()
    contract_audit["source_config_sha256"] = _sha256(source_config_path)
    contract_audit["label_contract_version"] = LABEL_CONTRACT_VERSION
    if contract_audit["label_reconstruction_max_abs_error"] > 1e-12:
        contract_audit["status"] = "needs_revision"
        contract_audit["checks"]["label_reconstruction"] = False
    else:
        contract_audit["checks"]["label_reconstruction"] = True

    top_k = int(source_config["stability"]["top_k"])
    stability, separation, precision = aggregate_label_precision(
        block_estimates, top_k
    )
    for directory in (results_dir, report_dir):
        model_ready.to_csv(directory / "model_ready_labels.csv", index=False)
        stability.to_csv(
            directory / "aggregate_label_random_stability.csv", index=False
        )
        separation.to_csv(
            directory / "aggregate_label_candidate_separation.csv", index=False
        )
        (directory / "aggregate_label_precision_metrics.json").write_text(
            json.dumps(precision, indent=2), encoding="utf-8"
        )
        (directory / "label_contract_audit.json").write_text(
            json.dumps(contract_audit, indent=2), encoding="utf-8"
        )

    audit_path = report_dir / "audit_summary.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["label_contract_status"] = contract_audit["status"]
        audit["label_contract_version"] = LABEL_CONTRACT_VERSION
        audit.setdefault("median_metrics", {}).update(precision)
        audit_text = json.dumps(audit, indent=2)
        audit_path.write_text(audit_text, encoding="utf-8")
        (results_dir / "audit_summary.json").write_text(audit_text, encoding="utf-8")

    summary = {
        "dataset_id": stream.dataset_id,
        "experiment_id": experiment_id,
        "profile": profile,
        "label_rows": int(len(model_ready)),
        "anchors": int(
            model_ready[["network_id", "anchor_time"]].drop_duplicates().shape[0]
        ),
        "networks": int(model_ready["network_id"].nunique()),
        "candidate_union": int(model_ready["candidate_id"].nunique()),
        "lookback_seconds": float(model_ready["lookback_seconds"].iloc[0]),
        "horizon_seconds": float(model_ready["horizon_seconds"].iloc[0]),
        "primary_mapper": str(model_ready["primary_mapper"].iloc[0]),
        "beta_unit": str(model_ready["beta_unit"].iloc[0]),
        "random_block_count": int(model_ready["random_block_count"].iloc[0]),
        "introduction_sampling": ",".join(
            sorted(model_ready["introduction_sampling"].unique())
        ),
        "minimum_non_index_sampling_fraction": float(
            model_ready["non_index_sampling_fraction"].min()
        ),
        "median_single_block_rank_correlation": precision[
            "aggregate_label_single_block_spearman"
        ],
        "averaged_block_rank_reliability": precision[
            "aggregate_label_spearman_brown_reliability"
        ],
        "minimum_anchor_rank_reliability": precision[
            "minimum_anchor_spearman_brown_reliability"
        ],
        "aggregate_label_candidate_separation_icc": precision[
            "aggregate_label_candidate_separation_icc"
        ],
        "single_block_candidate_separation_icc": precision[
            "aggregate_label_single_block_candidate_separation_icc"
        ],
        "averaged_block_candidate_separation_icc": precision[
            "aggregate_label_mean_candidate_separation_icc"
        ],
        "contract_status": contract_audit["status"],
    }
    marker = "## Harmonized model-facing label contract"
    readme_path = report_dir / "README.md"
    if readme_path.exists():
        existing = readme_path.read_text(encoding="utf-8").split(marker)[0].rstrip()
        contract_note = f"""

{marker}

- Contract version: `{LABEL_CONTRACT_VERSION}`; status: `{contract_audit['status']}`.
- Durable row key: `(dataset_id, network_id, anchor_time, candidate_id)`.
- Scenario-averaged single-block rank correlation: {precision['aggregate_label_single_block_spearman']:.3f}.
- Reliability estimate for the delivered {precision['aggregate_label_block_count']}-block mean ranking: {precision['aggregate_label_spearman_brown_reliability']:.3f}; minimum anchor estimate: {precision['minimum_anchor_spearman_brown_reliability']:.3f}.
- Scenario-averaged single-block candidate-separation ICC: {precision['aggregate_label_single_block_candidate_separation_icc']:.3f}.
- Delivered block-mean candidate-separation ICC: {precision['aggregate_label_mean_candidate_separation_icc']:.3f}.
- Index-introduction sampling: `{summary['introduction_sampling']}`; minimum sampling fraction: {summary['minimum_non_index_sampling_fraction']:.3f}.

These are Monte Carlo precision diagnostics, not field-accuracy estimates.
`model_ready_labels.csv` contains the harmonized training interface and explicit
mapper, beta-unit, time-window, population, and sampling provenance.
"""
        readme_path.write_text(existing + contract_note, encoding="utf-8")

    manifest_path = results_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["label_contract_version"] = LABEL_CONTRACT_VERSION
        manifest["label_contract_status"] = contract_audit["status"]
        manifest["label_contract_harmonized_at_utc"] = datetime.now(UTC).isoformat(
            timespec="seconds"
        )
        manifest["outputs"] = sorted(path.name for path in results_dir.iterdir())
        manifest_text = json.dumps(manifest, indent=2)
        manifest_path.write_text(manifest_text, encoding="utf-8")
        (report_dir / "run_manifest.json").write_text(
            manifest_text, encoding="utf-8"
        )
    return model_ready, summary, contract_audit


def run(config_path: Path) -> tuple[Path, Path]:
    root = _repository_root()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile = str(config["profile"])
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile
    report_dir = root / config["outputs"]["report_root"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    label_tables = []
    summaries = []
    source_audits: dict[str, Any] = {}
    sources = config["sources"]
    for source in tqdm(sources, desc="Harmonizing label interfaces"):
        source_path = root / source["config_path"]
        source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        labels, summary, audit = _update_source_artifacts(
            source_config=source_config,
            source_config_path=source_path,
            profile=str(source.get("profile", "full")),
            root=root,
        )
        label_tables.append(labels)
        summaries.append(summary)
        source_audits[str(summary["dataset_id"])] = audit

    combined = pd.concat(label_tables, ignore_index=True)
    combined_audit = validate_model_ready_labels(combined)
    combined_audit["source_audits"] = source_audits
    summary_frame = pd.DataFrame(summaries).sort_values("dataset_id", ignore_index=True)
    if not summary_frame["contract_status"].eq("passed").all():
        combined_audit["status"] = "needs_revision"

    for directory in (results_dir, report_dir):
        combined.to_csv(directory / "model_ready_labels.csv", index=False)
        summary_frame.to_csv(directory / "dataset_summary.csv", index=False)
        (directory / "validation_report.json").write_text(
            json.dumps(combined_audit, indent=2), encoding="utf-8"
        )
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    rows = "\n".join(
        f"| {row.dataset_id} | {int(row.label_rows)} | {int(row.anchors)} | "
        f"{row.primary_mapper} | {row.introduction_sampling} | "
        f"{row.averaged_block_rank_reliability:.3f} | "
        f"{row.single_block_candidate_separation_icc:.3f} | "
        f"{row.averaged_block_candidate_separation_icc:.3f} |"
        for row in summary_frame.itertuples(index=False)
    )
    readme = f"""# Cross-dataset intervention-label interface v{LABEL_CONTRACT_VERSION}

This bundle harmonizes scientific meaning and provenance, not dataset-specific
observation physics. Mappers, beta units, window lengths, population sizes, and
index-case sampling fractions remain explicit columns rather than being forced
to match.

| dataset | labels | anchors | mapper | index sampling | averaged-block rank reliability | single-block label ICC | averaged-label ICC |
|---|---:|---:|---|---|---:|---:|---:|
{rows}

`model_ready_labels.csv` uses one row per dataset-network-anchor-candidate. The primary
target is normalized avoided attack rate. The within-anchor priority percentile
is retained as a scale-robust secondary target. Future contacts are label-only
information and must never enter deployment features.

Random blocks are independent keyed Monte Carlo world sets. Averaging blocks
reduces simulation noise; it does not validate field causality or pathogen
calibration. Exact top-k membership remains diagnostic only.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "experiment_id": experiment_id,
        "status": "completed",
        "validation_status": combined_audit["status"],
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "completed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path),
        "source_count": len(sources),
        "label_rows": len(combined),
        "simulation_rerun": False,
    }
    manifest_text = json.dumps(manifest, indent=2)
    (results_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
    (report_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonize model-facing intervention labels across datasets"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260815-001_label_interface_harmonization.yaml"),
    )
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
