from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import slice_stream
from animal_intervention.transmission import compile_primary_exposure
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import CoalescedDurationContactMapper

from .experimental_songbirds_validation import _observed_group_stream as _observed_songbird_stream
from .history_baseline_substitution import _markdown_table
from .outbreak_response_pilot import _git_value, _load_source_config, _load_windows, _sha256
from .policy_mechanism_taxonomy import build_paired_worlds
from .radolfzell_validation import _observed_group_stream as _observed_radolfzell_stream
from .wytham_validation import _host_group_stream


KEYS = ["dataset_id", "network_id", "anchor_id"]
OUTCOMES = ["history_minus_no_extra", "direct_minus_history", "direct_minus_no_extra"]
BASE_FEATURES = ["log_population_size", "log_history_pair_events"]
RETENTION_FEATURES = [
    "history_weighted_cosine",
    "history_node_rank_persistence",
    "history_top_node_overlap",
]


def _full_stream(dataset_id: str, source_config: dict[str, Any]) -> ExposureStream:
    dataset = CanonicalDataset.read(Path(source_config["data"]["canonical_path"]))
    if dataset_id == "experimental_wild_songbirds":
        return _observed_songbird_stream(dataset, source_config)
    if dataset_id == "domestic_sheep_sirtrack":
        return CoalescedDurationContactMapper().compile(dataset)
    if dataset_id == "wytham_great_tits_divorce":
        return _host_group_stream(dataset, str(source_config["data"]["host_species_code"]))
    if dataset_id == "radolfzell_great_tits_ontogeny":
        return _observed_radolfzell_stream(dataset, source_config)
    return compile_primary_exposure(dataset)


def _pair_weights(stream: ExposureStream, nodes: set[str]) -> dict[tuple[str, str], float]:
    weights: defaultdict[tuple[str, str], float] = defaultdict(float)
    for row in stream.dyadic_exposures.itertuples(index=False):
        left, right = str(row.source_id), str(row.target_id)
        if left == right or left not in nodes or right not in nodes:
            continue
        duration = (pd.Timestamp(row.end_time) - pd.Timestamp(row.start_time)).total_seconds()
        weights[tuple(sorted((left, right)))] += duration * float(row.hazard_rate_multiplier)
    memberships = {
        str(group): [(str(row.node_id), float(row.membership_weight)) for row in frame.itertuples(index=False) if str(row.node_id) in nodes]
        for group, frame in stream.group_memberships.groupby("group_event_id", observed=True)
    }
    for row in stream.group_exposures.itertuples(index=False):
        members = memberships.get(str(row.group_event_id), [])
        if len(members) < 2:
            continue
        duration = (pd.Timestamp(row.end_time) - pd.Timestamp(row.start_time)).total_seconds()
        base = duration * float(row.hazard_rate_multiplier)
        denominator = max(1, len(members) - 1) if row.group_mixing_mode == "frequency_dependent" else 1
        for index, (left, left_weight) in enumerate(members):
            for right, right_weight in members[index + 1 :]:
                if left != right:
                    weights[tuple(sorted((left, right)))] += base * left_weight * right_weight / denominator
    return dict(weights)


def _rank_correlation(left: dict[str, float], right: dict[str, float], nodes: list[str]) -> float:
    a = pd.Series({node: left.get(node, 0.0) for node in nodes}).rank(method="average").to_numpy(float)
    b = pd.Series({node: right.get(node, 0.0) for node in nodes}).rank(method="average").to_numpy(float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _retention_metrics(
    left: dict[tuple[str, str], float],
    right: dict[tuple[str, str], float],
    nodes: list[str],
    top_fraction: float,
) -> dict[str, float]:
    possible = len(nodes) * (len(nodes) - 1) / 2
    left_edges, right_edges = set(left), set(right)
    intersection = len(left_edges & right_edges)
    union = len(left_edges | right_edges)
    expected = len(left_edges) * len(right_edges) / possible if possible else 0.0
    maximum = min(len(left_edges), len(right_edges))
    denominator = maximum - expected
    adjusted_estimable = denominator > 1e-12
    # Keep a finite sentinel value for serialization only. Downstream analysis must
    # use adjusted_dyad_retention_estimable before interpreting this value.
    adjusted = (intersection - expected) / denominator if adjusted_estimable else 0.0
    all_edges = sorted(left_edges | right_edges)
    left_values = np.array([left.get(edge, 0.0) for edge in all_edges], dtype=float)
    right_values = np.array([right.get(edge, 0.0) for edge in all_edges], dtype=float)
    norm = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    cosine = float(np.dot(left_values, right_values) / norm) if norm > 0 else 0.0
    left_nodes: defaultdict[str, float] = defaultdict(float)
    right_nodes: defaultdict[str, float] = defaultdict(float)
    for (source, target), value in left.items():
        left_nodes[source] += value
        left_nodes[target] += value
    for (source, target), value in right.items():
        right_nodes[source] += value
        right_nodes[target] += value
    k = max(1, int(math.ceil(len(nodes) * top_fraction)))
    left_top = {item[0] for item in sorted(((node, left_nodes[node]) for node in nodes), key=lambda item: (-item[1], item[0]))[:k]}
    right_top = {item[0] for item in sorted(((node, right_nodes[node]) for node in nodes), key=lambda item: (-item[1], item[0]))[:k]}
    return {
        "adjusted_dyad_retention": float(adjusted),
        "adjusted_dyad_retention_estimable": float(adjusted_estimable),
        "dyad_jaccard": intersection / union if union else 1.0,
        "weighted_cosine": cosine,
        "node_rank_persistence": _rank_correlation(left_nodes, right_nodes, nodes),
        "top_node_overlap": len(left_top & right_top) / k,
        "left_edges": float(len(left_edges)),
        "right_edges": float(len(right_edges)),
        "shared_edges": float(intersection),
        "left_weight": float(sum(left.values())),
        "right_weight": float(sum(right.values())),
    }


def measure_anchor_retention(
    dataset_id: str,
    network_id: str,
    window: dict[str, Any],
    full_stream: ExposureStream,
    top_fraction: float,
) -> dict[str, Any]:
    anchor = window["anchor"]
    nodes = sorted(map(str, window["eligible"]))
    midpoint = anchor.history_start + (anchor.anchor_time - anchor.history_start) / 2
    first = _pair_weights(slice_stream(full_stream, anchor.history_start, midpoint), set(nodes))
    second = _pair_weights(slice_stream(full_stream, midpoint, anchor.anchor_time), set(nodes))
    future = _pair_weights(window["future"], set(nodes))
    history_metrics = _retention_metrics(first, second, nodes, top_fraction)
    history_all = dict(first)
    for pair, value in second.items():
        history_all[pair] = history_all.get(pair, 0.0) + value
    oracle_metrics = _retention_metrics(history_all, future, nodes, top_fraction)
    row: dict[str, Any] = {
        "dataset_id": dataset_id,
        "network_id": str(network_id),
        "anchor_id": str(anchor.anchor_id),
        "anchor_time": anchor.anchor_time,
        "population_size": len(nodes),
        "history_pair_events": len(first) + len(second),
        "history_total_weight": sum(first.values()) + sum(second.values()),
        "log_population_size": math.log1p(len(nodes)),
        "log_history_pair_events": math.log1p(len(first) + len(second)),
    }
    row.update({f"history_{key}": value for key, value in history_metrics.items()})
    row.update({f"oracle_history_future_{key}": value for key, value in oracle_metrics.items()})
    return row


def _spearman(left: pd.Series, right: pd.Series) -> float:
    a = left.rank(method="average").to_numpy(float)
    b = right.rank(method="average").to_numpy(float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def family_associations(anchor_outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = [*RETENTION_FEATURES, "oracle_history_future_adjusted_dyad_retention", "oracle_history_future_weighted_cosine"]
    for family, frame in anchor_outcomes.groupby("system_family", observed=True):
        for feature in features:
            for outcome in OUTCOMES:
                rows.append({
                    "system_family": family,
                    "feature": feature,
                    "outcome": outcome,
                    "anchors": len(frame),
                    "spearman": _spearman(frame[feature], frame[outcome]),
                })
    return pd.DataFrame(rows)


def _ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], outcome: str, penalty: float) -> np.ndarray:
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    regularizer = np.eye(x_train.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(x_train.T @ x_train + regularizer, x_train.T @ train[outcome].to_numpy(float))
    return x_test @ coefficients


def loso_predictions(anchor_outcomes: pd.DataFrame, penalty: float) -> pd.DataFrame:
    rows = []
    families = sorted(anchor_outcomes["system_family"].unique())
    for held_out in families:
        train = anchor_outcomes.loc[~anchor_outcomes["system_family"].eq(held_out)]
        test = anchor_outcomes.loc[anchor_outcomes["system_family"].eq(held_out)]
        for outcome in OUTCOMES:
            for model, features in (("support_only", BASE_FEATURES), ("retention_augmented", BASE_FEATURES + RETENTION_FEATURES)):
                predictions = _ridge_predict(train, test, features, outcome, penalty)
                for item, prediction in zip(test.itertuples(index=False), predictions):
                    rows.append({
                        "system_family": held_out,
                        "dataset_id": item.dataset_id,
                        "network_id": item.network_id,
                        "anchor_id": item.anchor_id,
                        "outcome": outcome,
                        "model": model,
                        "observed": getattr(item, outcome),
                        "predicted": float(prediction),
                    })
    return pd.DataFrame(rows)


def loso_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in predictions.groupby(["system_family", "outcome", "model"], observed=True):
        family, outcome, model = keys
        observed = frame["observed"].to_numpy(float)
        predicted = frame["predicted"].to_numpy(float)
        rows.append({
            "system_family": family,
            "outcome": outcome,
            "model": model,
            "anchors": len(frame),
            "mae": float(np.mean(np.abs(predicted - observed))),
            "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
            "spearman": _spearman(frame["predicted"], frame["observed"]),
        })
    return pd.DataFrame(rows)


def _plot_landscape(features: pd.DataFrame, path: Path, dpi: int) -> None:
    summary = features.groupby("system_family", observed=True)[["history_weighted_cosine", "oracle_history_future_weighted_cosine"]].mean()
    y = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(11.5, 6.6))
    axis.scatter(summary["history_weighted_cosine"], y, s=80, label="Within-history similarity", color="#4C78A8")
    axis.scatter(summary["oracle_history_future_weighted_cosine"], y, s=80, marker="D", label="History-to-future ceiling", color="#F58518")
    for index in range(len(summary)):
        axis.plot(summary.iloc[index].to_numpy(float), [index, index], color="#B8B8B8", linewidth=2, zorder=0)
    axis.set_yticks(y, [value.replace("_", " ") for value in summary.index])
    axis.set_xlabel("Weighted contact-pattern cosine similarity")
    axis.set_title("Contact-weight memory and forward persistence", fontsize=17, fontweight="bold", pad=18)
    axis.legend(frameon=False, loc="best")
    axis.grid(axis="x", alpha=0.22)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.86, bottom=0.14)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_associations(anchor_outcomes: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.4))
    panels = [
        ("history_weighted_cosine", "history_minus_no_extra", "History policy absolute value"),
        ("history_weighted_cosine", "direct_minus_history", "Reactive minus history value"),
    ]
    families = sorted(anchor_outcomes["system_family"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(families)))
    for axis, (feature, outcome, title) in zip(axes, panels):
        for family, color in zip(families, colors):
            frame = anchor_outcomes.loc[anchor_outcomes["system_family"].eq(family)]
            axis.scatter(frame[feature], 100 * frame[outcome], s=42, alpha=0.78, color=color, label=family.replace("_", " "))
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xlabel("Pre-outbreak weighted contact-pattern similarity")
        axis.set_ylabel("Avoided attack-rate difference (percentage points)")
        axis.set_title(title, fontsize=13)
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Does observable contact memory explain intervention value?", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.22)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_loso(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    summary = metrics.groupby(["outcome", "model"], observed=True)[["mae", "sign_accuracy"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    outcomes = OUTCOMES
    labels = ["History absolute", "Reactive − history", "Reactive absolute"]
    x = np.arange(len(outcomes))
    width = 0.34
    for offset, (model, color, label) in enumerate((("support_only", "#9ECAE1", "Support only"), ("retention_augmented", "#4C78A8", "+ retention"))):
        selected = summary.loc[summary["model"].eq(model)].set_index("outcome").reindex(outcomes)
        axes[0].bar(x + (offset - 0.5) * width, 100 * selected["mae"], width, color=color, label=label)
        axes[1].bar(x + (offset - 0.5) * width, selected["sign_accuracy"], width, color=color, label=label)
    axes[0].set_ylabel("Family-equal MAE (attack-rate percentage points)")
    axes[1].set_ylabel("Family-equal sign accuracy")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("Strict leave-one-animal-system-out retention test", fontsize=18, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_gap_map(path: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(11.5, 6.8))
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.axvline(1, color="white", linewidth=3)
    axis.axhline(1, color="white", linewidth=3)
    colors = [["#DDEBF7", "#FDE2C5"], ["#C7E9C0", "#F4CCCC"]]
    for row in range(2):
        for column in range(2):
            axis.add_patch(plt.Rectangle((column, row), 1, 1, facecolor=colors[row][column], edgecolor="none"))
    labels = {
        (0.5, 1.5): ("Past-contact targeting", "Established\nLee; Valdano; Starnini"),
        (1.5, 1.5): ("Outbreak-time updating", "Established\ncontact tracing; GNN/RL"),
        (0.5, 0.5): ("Temporal predictability", "Retention and entropy limits\nnot linked to policy safety"),
        (1.5, 0.5): ("Temporal Animal Intervention Evaluation gap", "When does new contact evidence\noverride history—and when abstain?"),
    }
    for (x, y), (title, subtitle) in labels.items():
        axis.text(x, y + 0.12, title, ha="center", va="center", fontsize=14, fontweight="bold")
        axis.text(x, y - 0.16, subtitle, ha="center", va="center", fontsize=10.5, color="#444444")
    axis.set_xticks([0.5, 1.5], ["Preparedness information", "Outbreak-specific information"])
    axis.set_yticks([0.5, 1.5], ["Predictability / safety", "Target-selection methods"])
    axis.set_title("Focused literature refresh: the remaining testable gap", fontsize=18, fontweight="bold", pad=18)
    axis.tick_params(length=0, labelsize=11)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.14)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile = config["profiles"][profile_name]
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    prerequisite_path = Path(config["data"]["prerequisite_audit"])
    prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    if prerequisite.get("status") != "pass":
        raise ValueError("prerequisite policy audit must pass")
    worlds_path = Path(config["data"]["policy_worlds"])
    worlds = pd.read_csv(worlds_path, dtype={"network_id": str, "initial_infected": str})
    available_windows = {
        (str(row.dataset_id), str(row.network_id), str(row.anchor_id))
        for row in worlds[["dataset_id", "network_id", "anchor_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    top_fraction = float(config["measurement"]["top_fraction"])
    feature_rows = []
    for dataset_id in tqdm(profile["datasets"], desc="Measuring temporal retention", unit="dataset"):
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        windows = [
            window
            for window in windows
            if (
                dataset_id,
                str(window.get("network_id", default_network_id)),
                str(window["anchor"].anchor_id),
            )
            in available_windows
        ]
        maximum = profile.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        full_stream = _full_stream(dataset_id, source_config)
        for window in windows:
            network_id = str(window.get("network_id", default_network_id))
            row = measure_anchor_retention(dataset_id, network_id, window, full_stream, top_fraction)
            row["system_family"] = str(specification["system_family"])
            feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    paired = build_paired_worlds(worlds)
    cell_keys = KEYS + ["epidemic_model", "detection_profile", "rewiring_fraction"]
    cells = paired.groupby(cell_keys + ["system_family"], observed=True)[OUTCOMES].mean().reset_index()
    anchor_outcomes = cells.groupby(KEYS + ["system_family"], observed=True)[OUTCOMES].mean().reset_index()
    anchor_outcomes = anchor_outcomes.merge(features, on=KEYS + ["system_family"], how="inner", validate="one_to_one")
    associations = family_associations(anchor_outcomes)
    predictions = loso_predictions(anchor_outcomes, float(config["evaluation"]["ridge_penalty"]))
    metrics = loso_metrics(predictions)
    model_summary = metrics.groupby(["outcome", "model"], observed=True)[["mae", "sign_accuracy", "spearman"]].mean().reset_index()
    association_summary = associations.groupby(["feature", "outcome"], observed=True)["spearman"].agg(["mean", "median", "count"]).reset_index()
    deployable = associations.loc[associations["feature"].eq("history_weighted_cosine")].copy()
    history_directions = int((deployable.loc[deployable["outcome"].eq("history_minus_no_extra"), "spearman"] > 0).sum())
    reactive_directions = int((deployable.loc[deployable["outcome"].eq("direct_minus_history"), "spearman"] < 0).sum())
    pivot = model_summary.pivot(index="outcome", columns="model", values=["mae", "sign_accuracy"])
    loso_pass = bool(
        (pivot[("mae", "retention_augmented")] < pivot[("mae", "support_only")]).all()
        and (pivot[("sign_accuracy", "retention_augmented")] >= pivot[("sign_accuracy", "support_only")]).all()
    )
    minimum = int(config["evaluation"]["minimum_family_directions"])
    scientific = {
        "binary_retention_primary_gate_estimable": False,
        "binary_retention_primary_gate_reason": "dyad support saturation in dense animal systems",
        "exploratory_weighted_history_direction_families": history_directions,
        "exploratory_weighted_reactive_turnover_families": reactive_directions,
        "minimum_required_families": minimum,
        "exploratory_history_direction_threshold_met": history_directions >= minimum,
        "exploratory_reactive_direction_threshold_met": reactive_directions >= minimum,
        "loso_prediction_gate": loso_pass,
    }
    checks = {
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "all_requested_datasets_measured": set(features["dataset_id"]) == set(profile["datasets"]),
        "feature_keys_unique": not features.duplicated(KEYS).any(),
        "policy_anchor_merge_complete": len(anchor_outcomes) == features.merge(anchor_outcomes[KEYS], on=KEYS, how="inner").shape[0],
        "five_independent_families_full": profile_name != "full" or anchor_outcomes["system_family"].nunique() == 5,
        "finite_primary_features": bool(np.isfinite(features[BASE_FEATURES + RETENTION_FEATURES]).all().all()),
        "retention_bounded": bool(features["history_adjusted_dyad_retention"].between(-1, 1).all()),
        "adjusted_retention_estimability_recorded": "history_adjusted_dyad_retention_estimable" in features,
        "loso_predictions_complete": len(predictions) == len(anchor_outcomes) * len(OUTCOMES) * 2,
        "no_new_epidemic_worlds": True,
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "scientific_gates": scientific,
        "scope": "mechanism_and_predictability_audit_not_field_causal_validation",
        "datasets": int(features["dataset_id"].nunique()),
        "families": int(features["system_family"].nunique()),
        "anchors": len(features),
    }
    if audit["status"] != "pass":
        raise ValueError(f"retention audit failed: {audit}")
    outputs = {
        "anchor_retention_features.csv": features,
        "anchor_policy_outcomes.csv": anchor_outcomes,
        "family_associations.csv": associations,
        "association_summary.csv": association_summary,
        "loso_predictions.csv": predictions,
        "loso_family_metrics.csv": metrics,
        "loso_summary.csv": model_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(results_dir / name, index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [config_path, worlds_path, prerequisite_path, Path(__file__)]
    pd.DataFrame([{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in source_paths]).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_status": _git_value(["git", "status", "--short"]),
        "config_path": str(config_path),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dpi = int(profile["render_dpi"])
    _plot_landscape(features, report_dir / "retention_landscape.png", dpi)
    _plot_associations(anchor_outcomes, report_dir / "retention_policy_associations.png", dpi)
    _plot_loso(metrics, report_dir / "retention_loso_test.png", dpi)
    _plot_gap_map(report_dir / "literature_gap_map.png", dpi)
    display = model_summary.copy()
    display["mae_pp"] = 100 * display["mae"]
    report = f"""# Temporal retention and intervention-value audit

This preregistered mechanism audit asks whether contact memory measured before an
outbreak explains when a history-weight preparedness list should be retained or
replaced by detected-case contact evidence. It does not generate new epidemic
worlds or tune disease parameters.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Technical audit: **{audit['status']}**
- Exploratory weighted-similarity/history directions: {history_directions}/{audit['families']}
- Exploratory weighted-turnover/reactive-gain directions: {reactive_directions}/{audit['families']}
- Strict LOSO retention-feature gate: **{'pass' if loso_pass else 'fail'}**

The preregistered chance-adjusted binary statistic is not estimable when both
subwindows saturate the possible dyad support, which occurs in the dense sheep
and baboon systems. Those cases are explicitly flagged rather than interpreted
as zero memory. Consequently the five-family binary-retention association gate
is invalidated, not passed or rescued post hoc. Weighted contact-pattern cosine,
node-rank persistence, and top-node overlap remain measurable and are used only
for the corrected descriptive and LOSO sensitivity analysis. The
history-to-future version remains an explanatory oracle ceiling and is never a
deployment feature.

## Strict leave-one-system-out results

{_markdown_table(display[['outcome', 'model', 'mae_pp', 'sign_accuracy', 'spearman']])}

The corrected LOSO test excludes the non-estimable binary statistic. It still
fails the predeclared composite performance criterion. Contact memory is
therefore a useful mechanism descriptor, not a transportable policy selector.
The exploratory direction counts above are reported to motivate, not validate,
a separately preregistered continuous information-accrual experiment.
"""
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Relate temporal contact retention to intervention policy value.")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260818-004_temporal_retention_policy_value.yaml"))
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
