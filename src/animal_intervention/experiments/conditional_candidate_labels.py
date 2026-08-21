from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.evaluation import stable_hash_order
from animal_intervention.simulation import (
    DetectionProfile,
    PairedTemporalSIREngine,
    SIRParameters,
    detection_time_from_seed,
    observe_detected_cases,
    pre_detection_event_signature,
    pre_detection_scores,
    states_at,
)

from .outbreak_response_pilot import (
    DATASET_LABELS,
    _git_value,
    _isolation_action,
    _keyed_seed,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _selected_parameters,
    _sha256,
)


CONTEXT_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_id",
    "parameter_id",
    "detection_profile",
    "evidence_profile",
    "random_block",
    "initial_infected",
    "world_seed",
]


def _percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1 or values.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    return values.rank(method="average", pct=True)


def _spearman(values_a: pd.Series, values_b: pd.Series) -> float:
    ranks_a = values_a.rank(method="average").to_numpy(float)
    ranks_b = values_b.rank(method="average").to_numpy(float)
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return np.nan
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _candidate_task(
    *,
    dataset_id: str,
    system_family: str,
    window: dict[str, Any],
    parameter: Any,
    detection_profile: DetectionProfile,
    evidence_profile: str,
    secondary_case_sensitivity: float,
    stable_scores: pd.DataFrame,
    seed_nodes: list[str],
    half_life_fraction: float,
    experiment_seed: int,
) -> pd.DataFrame:
    anchor = window["anchor"]
    stream = window["future"]
    eligible = list(map(str, window["eligible"]))
    population_size = len(stream.nodes())
    parameters = SIRParameters(
        beta=float(parameter.beta),
        recovery_rate=float(parameter.recovery_rate_per_second),
    )
    mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
    detection_time = detection_time_from_seed(
        anchor.anchor_time,
        anchor.horizon_end,
        mean_period,
        detection_profile,
    )
    if detection_time is None:
        return pd.DataFrame()
    engine = PairedTemporalSIREngine()
    rows: list[dict[str, Any]] = []
    stable = stable_scores.copy()
    stable["candidate_id"] = stable["candidate_id"].astype(str)
    for initial in seed_nodes:
        world_seed = _keyed_seed(
            experiment_seed,
            dataset_id,
            anchor.anchor_id,
            parameter.parameter_id,
            0,
            initial,
        )
        natural = engine.simulate(
            stream,
            parameters,
            initial_infected=(initial,),
            start_time=anchor.anchor_time,
            end_time=anchor.horizon_end,
            world_seed=world_seed,
        )
        decision_states = states_at(natural, detection_time)
        detected = observe_detected_cases(
            decision_states,
            trigger_node=str(initial),
            secondary_case_sensitivity=secondary_case_sensitivity,
            world_seed=world_seed,
        )
        scores = pre_detection_scores(
            stream,
            detected_nodes=detected,
            start_time=anchor.anchor_time,
            detection_time=detection_time,
            half_life=mean_period * half_life_fraction,
        )
        scores["candidate_id"] = scores["candidate_id"].astype(str)
        scores = scores.loc[scores["candidate_id"].isin(eligible)].merge(
            stable,
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        scores = scores.loc[~scores["candidate_id"].isin(detected)].copy()
        scores["stable_percentile"] = _percentile(scores["stable_score"])
        scores["activity_percentile"] = _percentile(scores["current_activity"])
        scores["tracing_percentile"] = _percentile(scores["contact_to_detected"])
        standard = engine.simulate(
            stream,
            parameters,
            initial_infected=(initial,),
            start_time=anchor.anchor_time,
            end_time=anchor.horizon_end,
            world_seed=world_seed,
            action=_isolation_action(
                "standard_care", detected, detection_time, anchor.horizon_end
            ),
        )
        natural_signature = pre_detection_event_signature(natural, detection_time)
        if pre_detection_event_signature(standard, detection_time) != natural_signature:
            raise AssertionError("standard care diverged before detection")
        infectious_count = sum(state == "I" for state in decision_states.values())
        for candidate in scores.itertuples(index=False):
            augmented = engine.simulate(
                stream,
                parameters,
                initial_infected=(initial,),
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
                action=_isolation_action(
                    "standard_plus_candidate",
                    tuple(sorted(set(detected) | {str(candidate.candidate_id)})),
                    detection_time,
                    anchor.horizon_end,
                ),
            )
            if pre_detection_event_signature(augmented, detection_time) != natural_signature:
                raise AssertionError("candidate intervention diverged before detection")
            avoided = standard.final_size - augmented.final_size
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "network_id": str(window["network_id"]),
                    "system_family": system_family,
                    "anchor_id": anchor.anchor_id,
                    "anchor_time": anchor.anchor_time,
                    "horizon_end": anchor.horizon_end,
                    "parameter_id": parameter.parameter_id,
                    "detection_profile": detection_profile.name,
                    "detection_delay_fraction": detection_profile.delay_fraction_of_mean_infectious_period,
                    "detection_time": detection_time,
                    "evidence_profile": evidence_profile,
                    "secondary_case_sensitivity": secondary_case_sensitivity,
                    "random_block": 0,
                    "initial_infected": str(initial),
                    "world_seed": world_seed,
                    "candidate_id": str(candidate.candidate_id),
                    "population_size": population_size,
                    "detected_case_count": len(detected),
                    "detected_nodes": "|".join(detected),
                    "infectious_case_count_diagnostic": infectious_count,
                    "candidate_infectious_diagnostic": decision_states[str(candidate.candidate_id)] == "I",
                    "stable_score": float(candidate.stable_score),
                    "current_activity": float(candidate.current_activity),
                    "contact_to_detected": float(candidate.contact_to_detected),
                    "stable_percentile": float(candidate.stable_percentile),
                    "activity_percentile": float(candidate.activity_percentile),
                    "tracing_percentile": float(candidate.tracing_percentile),
                    "natural_final_size": natural.final_size,
                    "standard_final_size": standard.final_size,
                    "candidate_final_size": augmented.final_size,
                    "avoided_infections": avoided,
                    "conditional_attack_rate_value": avoided / population_size,
                }
            )
    return pd.DataFrame(rows)


def _context_summary(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby(CONTEXT_KEYS, observed=True, sort=True)
        .agg(
            dataset_id=("dataset_id", "first"),
            candidates=("candidate_id", "size"),
            positive_candidate_fraction=("conditional_attack_rate_value", lambda x: float(x.gt(0).mean())),
            negative_candidate_fraction=("conditional_attack_rate_value", lambda x: float(x.lt(0).mean())),
            value_standard_deviation=("conditional_attack_rate_value", "std"),
            maximum_value=("conditional_attack_rate_value", "max"),
            minimum_value=("conditional_attack_rate_value", "min"),
        )
        .reset_index(drop=True)
    )


def _dataset_summary(labels: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, frame in labels.groupby("dataset_id", observed=True):
        context_frame = contexts.loc[contexts["dataset_id"].eq(dataset_id)]
        correlations = []
        for _, group in frame.groupby(CONTEXT_KEYS, observed=True):
            if group["conditional_attack_rate_value"].nunique() > 1:
                correlations.append(
                    _spearman(
                        group["tracing_percentile"],
                        group["conditional_attack_rate_value"],
                    )
                )
        rows.append(
            {
                "dataset_id": dataset_id,
                "labels": len(frame),
                "contexts": len(context_frame),
                "variable_context_fraction": float(context_frame["value_standard_deviation"].fillna(0).gt(0).mean()),
                "positive_label_fraction": float(frame["conditional_attack_rate_value"].gt(0).mean()),
                "negative_label_fraction": float(frame["conditional_attack_rate_value"].lt(0).mean()),
                "mean_context_tracing_spearman": float(np.nanmean(correlations)) if correlations else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_labels(labels: pd.DataFrame, contexts: pd.DataFrame, report_dir: Path) -> None:
    dataset_order = sorted(labels["dataset_id"].unique())
    display = [DATASET_LABELS.get(item, item) for item in dataset_order]
    values = [100 * labels.loc[labels["dataset_id"].eq(item), "conditional_attack_rate_value"].to_numpy(float) for item in dataset_order]
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.violinplot(values, showmedians=True, showextrema=False)
    axis.set_xticks(range(1, len(display) + 1), display, rotation=20, ha="right")
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_ylabel("Conditional attack-rate value (percentage points)")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Outbreak-time candidate intervention-value distributions", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.925, "Each candidate is compared with the same-context standard-care replay; symmetric-log scale", ha="center", color="#555555")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.20)
    fig.savefig(report_dir / "candidate_value_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot = contexts.copy()
    plot["variable"] = plot["value_standard_deviation"].fillna(0).gt(0)
    summary = plot.groupby("dataset_id", observed=True).agg(variable_fraction=("variable", "mean"), mean_positive_fraction=("positive_candidate_fraction", "mean")).reindex(dataset_order)
    x = np.arange(len(summary))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(x - width / 2, summary["variable_fraction"], width, label="Contexts with candidate separation", color="#4C78A8")
    axis.bar(x + width / 2, summary["mean_positive_fraction"], width, label="Mean positive-candidate fraction", color="#F58518")
    axis.set_xticks(x, display, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Fraction")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Conditional-label learnability diagnostics", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.925, "Separation is assessed within paired decision contexts", ha="center", color="#555555")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.20)
    fig.savefig(report_dir / "label_learnability_gate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile = config["profiles"][profile_name]
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stable_path = Path(config["data"]["stable_prediction_path"])
    stable_predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    stable_predictions["anchor_time"] = pd.to_datetime(stable_predictions["anchor_time"], format="mixed")
    fingerprint = hashlib.sha256(config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    tasks = []
    support_rows = []
    detections = [DetectionProfile(**item) for item in config["decision"]["detection_profiles"]]
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        for window in windows:
            window.setdefault("network_id", default_network_id)
        available = set(stable_predictions.loc[stable_predictions["dataset_id"].eq(dataset_id), ["network_id", "anchor_time"]].itertuples(index=False, name=None))
        windows = [window for window in windows if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time)) in available]
        windows = windows[: int(profile["max_anchors_per_dataset"])]
        parameters = _selected_parameters(Path(specification["source_results"]) / "parameter_selection.csv", None)
        for window in windows:
            compatible = []
            for parameter in parameters.itertuples(index=False):
                period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
                supported = all(detection_time_from_seed(window["anchor"].anchor_time, window["anchor"].horizon_end, period, item) is not None for item in detections)
                support_rows.append({"dataset_id": dataset_id, "network_id": str(window["network_id"]), "anchor_id": window["anchor"].anchor_id, "parameter_id": parameter.parameter_id, "supported": supported})
                if supported:
                    compatible.append(parameter)
            compatible.sort(key=lambda item: float(item.mean_attack_rate))
            if not compatible:
                continue
            parameter = compatible[len(compatible) // 2]
            for detection in detections:
                for evidence in config["decision"]["evidence_profiles"]:
                    tasks.append((dataset_id, specification, window, parameter, detection, evidence))
    frames = []
    for dataset_id, specification, window, parameter, detection, evidence in tqdm(tasks, desc="Conditional-label tasks", unit="task"):
        anchor = window["anchor"]
        identity = f"{fingerprint}|{dataset_id}|{window['network_id']}|{anchor.anchor_id}|{parameter.parameter_id}|{detection.name}|{evidence['name']}"
        checkpoint = checkpoint_dir / f"{dataset_id}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.csv.gz"
        if bool(config["execution"].get("resume", True)) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"candidate_id": str, "initial_infected": str})
            if not frame.empty:
                frames.append(frame)
                continue
        stable = _matching_stable_scores(stable_predictions, dataset_id, str(window["network_id"]), anchor.anchor_time, window["eligible"])
        seeds = stable_hash_order(list(map(str, window["eligible"])), int(config["evaluation"]["seed"]), dataset_id, anchor.anchor_id, "candidate_label_seeds")[: int(profile["seeds_per_anchor"])]
        frame = _candidate_task(dataset_id=dataset_id, system_family=str(specification["system_family"]), window=window, parameter=parameter, detection_profile=detection, evidence_profile=str(evidence["name"]), secondary_case_sensitivity=float(evidence["secondary_case_sensitivity"]), stable_scores=stable, seed_nodes=seeds, half_life_fraction=float(config["decision"]["tracing_half_life_fraction_of_mean_infectious_period"]), experiment_seed=int(config["evaluation"]["seed"]))
        frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    contexts = _context_summary(labels)
    datasets = _dataset_summary(labels, contexts)
    labels.to_csv(results_dir / "conditional_candidate_labels.csv.gz", index=False, compression="gzip")
    contexts.to_csv(results_dir / "context_label_summary.csv", index=False)
    datasets.to_csv(results_dir / "dataset_label_summary.csv", index=False)
    pd.DataFrame(support_rows).to_csv(results_dir / "parameter_detection_support.csv", index=False)
    gate = config["evaluation"]["gate"]
    minimum_datasets = int(
        profile.get(
            "minimum_datasets_with_variable_contexts",
            gate["minimum_datasets_with_variable_contexts"],
        )
    )
    datasets_passing = int(datasets["variable_context_fraction"].gt(0).sum())
    overall_variable = float(contexts["value_standard_deviation"].fillna(0).gt(0).mean())
    audit = {
        "status": "pass",
        "checks": {
            "candidate_keys_unique": not labels.duplicated(CONTEXT_KEYS + ["candidate_id"]).any(),
            "detected_candidates_excluded": bool(labels.apply(lambda row: str(row.candidate_id) not in set(str(row.detected_nodes).split("|")), axis=1).all()),
            "standard_outcome_shared_within_context": bool(labels.groupby(CONTEXT_KEYS, observed=True)["standard_final_size"].nunique().eq(1).all()),
            "finite_labels_and_features": bool(np.isfinite(labels[["conditional_attack_rate_value", "stable_score", "current_activity", "contact_to_detected"]].to_numpy(float)).all()),
            "paired_arithmetic": bool(np.allclose(labels["conditional_attack_rate_value"], (labels["standard_final_size"] - labels["candidate_final_size"]) / labels["population_size"])),
        },
        "scientific_gate": {
            "status": "pass" if datasets_passing >= minimum_datasets and overall_variable >= float(gate["minimum_variable_context_fraction"]) else "fail",
            "datasets_with_variable_contexts": datasets_passing,
            "minimum_datasets": minimum_datasets,
            "variable_context_fraction": overall_variable,
            "minimum_variable_context_fraction": float(gate["minimum_variable_context_fraction"]),
        },
        "labels": len(labels),
        "contexts": len(contexts),
        "datasets": labels["dataset_id"].nunique(),
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"candidate-label audit failed: {audit}")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _plot_labels(labels, contexts, report_dir)
    manifest = {"experiment_id": experiment_id, "profile": profile_name, "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"), "elapsed_seconds": round(time.perf_counter() - started, 3), "git_commit": _git_value(["rev-parse", "HEAD"]), "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])), "python": platform.python_version(), "config_path": str(config_path), "config_sha256": _sha256(config_path), "audit_status": audit["status"], "scientific_gate": audit["scientific_gate"]["status"]}
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (report_dir / "README.md").write_text(f"# Conditional candidate-value learnability gate\n\nProfile: {profile_name}; labels: {len(labels):,}; decision contexts: {len(contexts)}; datasets: {labels['dataset_id'].nunique()}. Artifact audit: **{audit['status']}**. Scientific gate: **{audit['scientific_gate']['status']}**.\n\nDiagnostic infection-state columns are retained only for audit and cannot be used as predictor inputs.\n", encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conditional candidate intervention labels")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260816-010_conditional_candidate_labels.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
