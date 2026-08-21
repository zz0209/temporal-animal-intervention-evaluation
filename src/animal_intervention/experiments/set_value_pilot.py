from __future__ import annotations

import argparse
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

from animal_intervention.evaluation import stable_hash_order
from animal_intervention.simulation import (
    DetectionProfile,
    PairedTemporalSIREngine,
    SIRParameters,
    detection_time_from_seed,
    observe_detected_cases,
    pre_detection_event_signature,
    pre_detection_scores,
    select_additional_targets,
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
    "dataset_id", "network_id", "anchor_id", "parameter_id",
    "detection_profile", "evidence_profile", "budget_fraction", "initial_infected",
    "random_block", "world_seed",
]
FEATURE_COLUMNS = [
    "detection_delay_fraction", "secondary_case_sensitivity", "population_size",
    "detected_case_count", "budget", "budget_fraction_realized",
    "set_stable_mean", "set_stable_max", "set_stable_std",
    "set_tracing_mean", "set_tracing_max", "set_tracing_std",
    "set_activity_mean", "set_activity_max", "set_positive_tracing_fraction",
    "set_stable_tracing_product_mean",
]


def _percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1 or values.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    return values.rank(method="average", pct=True)


def _perturb_set(
    base: tuple[str, ...], candidates: list[str], fraction: float, seed: int, label: str
) -> tuple[str, ...]:
    if not base:
        return ()
    pool = [node for node in candidates if node not in base]
    replacements = min(
        len(base), len(pool), max(1, int(round(len(base) * fraction)))
    )
    if replacements == 0:
        return tuple(sorted(base))
    removed = set(stable_hash_order(list(base), seed, label, "remove")[:replacements])
    kept = [node for node in base if node not in removed]
    added = stable_hash_order(pool, seed, label, "add")[:replacements]
    return tuple(sorted(kept + added))


def _sample_sets(
    score_table: pd.DataFrame, detected: tuple[str, ...], budget: int,
    world_seed: int, random_sets: int, swap_fractions: list[float],
) -> list[tuple[str, tuple[str, ...]]]:
    candidates = sorted(set(score_table["candidate_id"].astype(str)) - set(detected))
    generated: list[tuple[str, tuple[str, ...]]] = []
    bases = {}
    for method in ("stable_watchlist", "contact_to_detected", "stable_plus_tracing"):
        selected = select_additional_targets(
            score_table, method=method, budget=budget,
            detected_nodes=detected, world_seed=world_seed,
        )
        bases[method] = selected
        generated.append((f"{method}_base", selected))
    for replicate in range(random_sets):
        selected = select_additional_targets(
            score_table, method="random", budget=budget, detected_nodes=detected,
            world_seed=_keyed_seed(world_seed, "random_set", replicate),
        )
        generated.append((f"random_{replicate:02d}", selected))
    for method, base in bases.items():
        for fraction in swap_fractions:
            generated.append((
                f"{method}_swap_{int(100 * fraction):02d}",
                _perturb_set(base, candidates, fraction, world_seed, f"{method}:{fraction}"),
            ))
    by_signature: dict[tuple[str, ...], list[str]] = {}
    for source, selected in generated:
        by_signature.setdefault(tuple(sorted(selected)), []).append(source)
    return [("|".join(sources), signature) for signature, sources in by_signature.items()]


def _set_features(score_table: pd.DataFrame, selected: tuple[str, ...]) -> dict[str, float]:
    frame = score_table.set_index("candidate_id").loc[list(selected)]
    stable = frame["stable_percentile"].to_numpy(float)
    tracing = frame["tracing_percentile"].to_numpy(float)
    activity = frame["activity_percentile"].to_numpy(float)
    return {
        "set_stable_mean": float(stable.mean()),
        "set_stable_max": float(stable.max()),
        "set_stable_std": float(stable.std()),
        "set_tracing_mean": float(tracing.mean()),
        "set_tracing_max": float(tracing.max()),
        "set_tracing_std": float(tracing.std()),
        "set_activity_mean": float(activity.mean()),
        "set_activity_max": float(activity.max()),
        "set_positive_tracing_fraction": float((frame["contact_to_detected"].to_numpy(float) > 0).mean()),
        "set_stable_tracing_product_mean": float((stable * tracing).mean()),
    }


def _run_contexts(
    *, dataset_id: str, system_family: str, window: dict[str, Any], parameter: Any,
    detection_profile: DetectionProfile, evidence_profile: str,
    secondary_case_sensitivity: float, stable_scores: pd.DataFrame,
    seed_nodes: list[str], config: dict[str, Any], budget_fraction: float | None = None,
    random_blocks: int | None = None,
) -> pd.DataFrame:
    anchor = window["anchor"]
    stream = window["future"]
    eligible = list(map(str, window["eligible"]))
    population_size = len(stream.nodes())
    parameters = SIRParameters(beta=float(parameter.beta), recovery_rate=float(parameter.recovery_rate_per_second))
    mean_period = pd.Timedelta(days=float(parameter.mean_infectious_period_days))
    detection_time = detection_time_from_seed(anchor.anchor_time, anchor.horizon_end, mean_period, detection_profile)
    if detection_time is None:
        return pd.DataFrame()
    engine = PairedTemporalSIREngine()
    rows: list[dict[str, Any]] = []
    requested_budget_fraction = float(
        config["decision"]["budget_fraction"] if budget_fraction is None else budget_fraction
    )
    block_count = int(1 if random_blocks is None else random_blocks)
    stable = stable_scores.copy()
    stable["candidate_id"] = stable["candidate_id"].astype(str)
    world_pairs = [
        (initial, block) for initial in seed_nodes for block in range(block_count)
    ]
    for initial, block in world_pairs:
        world_seed = _keyed_seed(config["evaluation"]["seed"], dataset_id, anchor.anchor_id, parameter.parameter_id, block, initial)
        natural = engine.simulate(stream, parameters, initial_infected=(initial,), start_time=anchor.anchor_time, end_time=anchor.horizon_end, world_seed=world_seed)
        decision_states = states_at(natural, detection_time)
        detected = observe_detected_cases(decision_states, trigger_node=str(initial), secondary_case_sensitivity=secondary_case_sensitivity, world_seed=world_seed)
        scores = pre_detection_scores(stream, detected_nodes=detected, start_time=anchor.anchor_time, detection_time=detection_time, half_life=mean_period * float(config["decision"]["tracing_half_life_fraction_of_mean_infectious_period"]))
        scores["candidate_id"] = scores["candidate_id"].astype(str)
        scores = scores.loc[scores["candidate_id"].isin(eligible)].merge(stable, on="candidate_id", how="left", validate="one_to_one")
        remaining = scores.loc[~scores["candidate_id"].isin(detected)].copy()
        remaining["stable_percentile"] = _percentile(remaining["stable_score"])
        remaining["activity_percentile"] = _percentile(remaining["current_activity"])
        remaining["tracing_percentile"] = _percentile(remaining["contact_to_detected"])
        scores = scores.merge(remaining[["candidate_id", "stable_percentile", "activity_percentile", "tracing_percentile"]], on="candidate_id", how="left")
        budget = min(len(remaining), max(int(config["decision"]["minimum_budget"]), int(math.ceil(len(remaining) * requested_budget_fraction)))) if len(remaining) else 0
        if budget == 0:
            continue
        sets = _sample_sets(scores, detected, budget, world_seed, int(config["decision"]["random_sets"]), list(map(float, config["decision"]["swap_fractions"])))
        standard = engine.simulate(stream, parameters, initial_infected=(initial,), start_time=anchor.anchor_time, end_time=anchor.horizon_end, world_seed=world_seed, action=_isolation_action("standard_care", detected, detection_time, anchor.horizon_end))
        signature = pre_detection_event_signature(natural, detection_time)
        if pre_detection_event_signature(standard, detection_time) != signature:
            raise AssertionError("standard care diverged before detection")
        for source_methods, selected in sets:
            augmented = engine.simulate(stream, parameters, initial_infected=(initial,), start_time=anchor.anchor_time, end_time=anchor.horizon_end, world_seed=world_seed, action=_isolation_action("standard_plus_set", tuple(sorted(set(detected) | set(selected))), detection_time, anchor.horizon_end))
            if pre_detection_event_signature(augmented, detection_time) != signature:
                raise AssertionError("set intervention diverged before detection")
            avoided = standard.final_size - augmented.final_size
            row = {
                "dataset_id": dataset_id, "network_id": str(window["network_id"]), "system_family": system_family,
                "anchor_id": anchor.anchor_id, "anchor_time": anchor.anchor_time, "parameter_id": parameter.parameter_id,
                "detection_profile": detection_profile.name, "detection_delay_fraction": detection_profile.delay_fraction_of_mean_infectious_period,
                "evidence_profile": evidence_profile, "secondary_case_sensitivity": secondary_case_sensitivity,
                "budget_fraction": requested_budget_fraction, "initial_infected": str(initial),
                "random_block": block, "world_seed": world_seed, "detected_nodes": "|".join(detected),
                "detected_case_count": len(detected), "population_size": population_size, "budget": budget,
                "budget_fraction_realized": budget / max(1, len(remaining)), "source_methods": source_methods,
                "selected_nodes": "|".join(selected), "set_signature": hashlib.sha256("|".join(selected).encode()).hexdigest()[:16],
                "natural_final_size": natural.final_size, "standard_final_size": standard.final_size,
                "set_final_size": augmented.final_size, "avoided_infections": avoided,
                "set_attack_rate_value": avoided / population_size,
            }
            row.update(_set_features(remaining, selected))
            rows.append(row)
    return pd.DataFrame(rows)


def _context_summary(values: pd.DataFrame) -> pd.DataFrame:
    summary = values.groupby(CONTEXT_KEYS, observed=True, sort=True).agg(
        system_family=("system_family", "first"), budget=("budget", "first"),
        sets=("set_signature", "size"),
        distinct_values=("set_attack_rate_value", "nunique"), value_std=("set_attack_rate_value", "std"),
        positive_set_fraction=("set_attack_rate_value", lambda x: float(x.gt(0).mean())),
        negative_set_fraction=("set_attack_rate_value", lambda x: float(x.lt(0).mean())),
        best_value=("set_attack_rate_value", "max"), worst_value=("set_attack_rate_value", "min"),
    ).reset_index()
    summary["value_spread"] = summary["best_value"] - summary["worst_value"]
    return summary


def _dataset_summary(contexts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, group in contexts.groupby("dataset_id", observed=True, sort=True):
        multinode = group.loc[group["budget"].gt(1)]
        rows.append({
            "dataset_id": dataset_id,
            "contexts": len(group),
            "median_budget": float(group["budget"].median()),
            "multinode_contexts": len(multinode),
            "variable_multinode_context_fraction": float(multinode["distinct_values"].gt(1).mean()) if len(multinode) else np.nan,
            "mean_multinode_spread": float(multinode["value_spread"].mean()) if len(multinode) else np.nan,
        })
    return pd.DataFrame(rows)


def _ridge_loso(values: pd.DataFrame, penalty: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    for family in sorted(values["system_family"].unique()):
        train = values.loc[values["system_family"].ne(family)]
        test = values.loc[values["system_family"].eq(family)].copy()
        x_train = train[FEATURE_COLUMNS].to_numpy(float)
        x_test = test[FEATURE_COLUMNS].to_numpy(float)
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale == 0] = 1
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        x_train = np.column_stack([np.ones(len(x_train)), x_train])
        x_test = np.column_stack([np.ones(len(x_test)), x_test])
        regularizer = np.eye(x_train.shape[1]) * penalty
        regularizer[0, 0] = 0
        coefficients = np.linalg.solve(x_train.T @ x_train + regularizer, x_train.T @ train["set_attack_rate_value"].to_numpy(float))
        test["predicted_set_value"] = x_test @ coefficients
        predictions.append(test)
    predicted = pd.concat(predictions, ignore_index=True)
    decisions = []
    for key, group in predicted.groupby(CONTEXT_KEYS, observed=True):
        chosen = group.loc[group["predicted_set_value"].idxmax()]
        oracle = group.loc[group["set_attack_rate_value"].idxmax()]
        fusion = group.loc[group["source_methods"].str.contains("stable_plus_tracing_base", regex=False)]
        if fusion.empty:
            raise ValueError("missing stable-plus-tracing baseline set")
        baseline = fusion.iloc[0]
        decisions.append({**dict(zip(CONTEXT_KEYS, key)), "system_family": chosen.system_family, "ridge_value": chosen.set_attack_rate_value, "fusion_value": baseline.set_attack_rate_value, "sampled_oracle_value": oracle.set_attack_rate_value, "ridge_minus_fusion": chosen.set_attack_rate_value - baseline.set_attack_rate_value, "oracle_regret": oracle.set_attack_rate_value - chosen.set_attack_rate_value})
    return predicted, pd.DataFrame(decisions)


def _plot(values: pd.DataFrame, contexts: pd.DataFrame, decisions: pd.DataFrame | None, report_dir: Path) -> None:
    summary = _dataset_summary(contexts)
    labels = [f"{DATASET_LABELS.get(item, item)} (budget {int(budget)})" for item, budget in zip(summary["dataset_id"], summary["median_budget"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].barh(labels, summary["variable_multinode_context_fraction"].fillna(0), color="#4C78A8")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Fraction of multi-node contexts with unequal set values")
    axes[0].set_title("Set-value separation", fontweight="bold")
    for bar, contexts_count in zip(bars, summary["multinode_contexts"]):
        if contexts_count == 0:
            axes[0].text(0.02, bar.get_y() + bar.get_height() / 2, "singleton only", va="center", color="#666666")
    axes[1].barh(labels, 100 * summary["mean_multinode_spread"].fillna(0), color="#F58518")
    axes[1].set_xlabel("Mean best-minus-worst set value (percentage points)")
    axes[1].set_title("Decision-relevant spread", fontweight="bold")
    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
    fig.suptitle("Fixed-budget sampled-set label diagnostics", fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.925, "Only contexts with intervention budget >= 2 count as set-learning evidence", ha="center", color="#555555")
    fig.subplots_adjust(left=0.27, right=0.98, top=0.82, bottom=0.14, wspace=0.48)
    fig.savefig(report_dir / "set_value_gate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if decisions is not None:
        frame = decisions.groupby("system_family", observed=True).agg(ridge_minus_fusion=("ridge_minus_fusion", "mean"), oracle_regret=("oracle_regret", "mean")).reset_index()
        fig, axis = plt.subplots(figsize=(10, 6))
        x = np.arange(len(frame))
        axis.bar(x - 0.18, 100 * frame["ridge_minus_fusion"], 0.36, label="Ridge minus fusion", color="#4C78A8")
        axis.bar(x + 0.18, 100 * frame["oracle_regret"], 0.36, label="Sampled-oracle regret", color="#F58518")
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticks(x, [item.replace("_", " ") for item in frame["system_family"]], rotation=20, ha="right")
        axis.set_ylabel("Attack-rate difference (percentage points)")
        axis.set_title("Leave-one-family-out sampled-set reranking", fontsize=17, fontweight="bold")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.18)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.23)
        fig.savefig(report_dir / "loso_ridge_diagnostic.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(config_path: Path) -> tuple[Path, Path]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / "pilot"
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / "pilot"
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir): directory.mkdir(parents=True, exist_ok=True)
    stable_path = Path(config["data"]["stable_prediction_path"])
    predictions = pd.read_csv(stable_path, dtype={"candidate_id": str, "network_id": str})
    predictions["anchor_time"] = pd.to_datetime(predictions["anchor_time"], format="mixed")
    fingerprint = hashlib.sha256(config_path.read_bytes() + stable_path.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    detections = [DetectionProfile(**item) for item in config["decision"]["detection_profiles"]]
    tasks = []
    for dataset_id in config["profile"]["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network = str(specification.get("network_id", "all"))
        for window in windows: window.setdefault("network_id", default_network)
        available = set(predictions.loc[predictions["dataset_id"].eq(dataset_id), ["network_id", "anchor_time"]].itertuples(index=False, name=None))
        windows = [window for window in windows if (str(window["network_id"]), pd.Timestamp(window["anchor"].anchor_time)) in available][:int(config["profile"]["max_anchors_per_dataset"])]
        parameters = _selected_parameters(Path(specification["source_results"]) / "parameter_selection.csv", None)
        for window in windows:
            compatible = [parameter for parameter in parameters.itertuples(index=False) if all(detection_time_from_seed(window["anchor"].anchor_time, window["anchor"].horizon_end, pd.Timedelta(days=float(parameter.mean_infectious_period_days)), item) is not None for item in detections)]
            compatible.sort(key=lambda item: float(item.mean_attack_rate))
            if not compatible: continue
            parameter = compatible[len(compatible) // 2]
            for detection in detections:
                for evidence in config["decision"]["evidence_profiles"]: tasks.append((dataset_id, specification, window, parameter, detection, evidence))
    frames = []
    for dataset_id, specification, window, parameter, detection, evidence in tqdm(tasks, desc="Set-value tasks", unit="task"):
        anchor = window["anchor"]
        identity = f"{fingerprint}|{dataset_id}|{window['network_id']}|{anchor.anchor_id}|{parameter.parameter_id}|{detection.name}|{evidence['name']}"
        checkpoint = checkpoint_dir / f"{dataset_id}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}.csv.gz"
        if bool(config["execution"]["resume"]) and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str})
            if not frame.empty: frames.append(frame); continue
        stable = _matching_stable_scores(predictions, dataset_id, str(window["network_id"]), anchor.anchor_time, window["eligible"])
        seeds = stable_hash_order(list(map(str, window["eligible"])), int(config["evaluation"]["seed"]), dataset_id, anchor.anchor_id, "set_value_seeds")[:int(config["profile"]["seeds_per_anchor"])]
        frame = _run_contexts(dataset_id=dataset_id, system_family=str(specification["system_family"]), window=window, parameter=parameter, detection_profile=detection, evidence_profile=str(evidence["name"]), secondary_case_sensitivity=float(evidence["secondary_case_sensitivity"]), stable_scores=stable, seed_nodes=seeds, config=config)
        frame.to_csv(checkpoint, index=False, compression="gzip"); frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    contexts = _context_summary(values)
    family_rows = []
    for family, group in contexts.groupby("system_family", observed=True, sort=True):
        multinode = group.loc[group["budget"].gt(1)]
        family_rows.append({
            "system_family": family,
            "contexts": len(group),
            "multinode_contexts": len(multinode),
            "variable_multinode_context_fraction": float(multinode["distinct_values"].gt(1).mean()) if len(multinode) else 0.0,
            "mean_multinode_spread": float(multinode["value_spread"].mean()) if len(multinode) else 0.0,
        })
    family_summary = pd.DataFrame(family_rows)
    multinode_contexts = contexts.loc[contexts["budget"].gt(1)]
    variable_families = int(family_summary["variable_multinode_context_fraction"].gt(0).sum())
    overall_variable = float(multinode_contexts["distinct_values"].gt(1).mean()) if len(multinode_contexts) else 0.0
    scientific_pass = variable_families >= int(config["evaluation"]["minimum_families_with_variable_contexts"]) and overall_variable >= float(config["evaluation"]["minimum_variable_context_fraction"])
    predicted = decisions = None
    if scientific_pass:
        predicted, decisions = _ridge_loso(values, float(config["evaluation"]["ridge_penalty"]))
        predicted.to_csv(results_dir / "loso_set_predictions.csv.gz", index=False, compression="gzip")
        decisions.to_csv(results_dir / "loso_context_decisions.csv", index=False)
    else:
        for stale_path in (results_dir / "loso_set_predictions.csv.gz", results_dir / "loso_context_decisions.csv", report_dir / "loso_ridge_diagnostic.png"):
            stale_path.unlink(missing_ok=True)
    values.to_csv(results_dir / "sampled_set_values.csv.gz", index=False, compression="gzip")
    contexts.to_csv(results_dir / "context_set_summary.csv", index=False)
    family_summary.to_csv(results_dir / "family_set_summary.csv", index=False)
    dataset_summary = _dataset_summary(contexts)
    dataset_summary.to_csv(results_dir / "dataset_set_summary.csv", index=False)
    disjoint = values.apply(lambda row: not bool(set(str(row.selected_nodes).split("|")) & set(str(row.detected_nodes).split("|"))), axis=1)
    set_sizes = values["selected_nodes"].map(lambda value: len(str(value).split("|")))
    audit = {"status": "pass", "checks": {"set_keys_unique": not values.duplicated(CONTEXT_KEYS + ["set_signature"]).any(), "detected_nodes_excluded": bool(disjoint.all()), "fixed_budget": bool(set_sizes.eq(values["budget"]).all()), "standard_shared_within_context": bool(values.groupby(CONTEXT_KEYS, observed=True)["standard_final_size"].nunique().eq(1).all()), "paired_arithmetic": bool(np.allclose(values["set_attack_rate_value"], (values["standard_final_size"] - values["set_final_size"]) / values["population_size"])), "finite_features_and_outcomes": bool(np.isfinite(values[FEATURE_COLUMNS + ["set_attack_rate_value"]].to_numpy(float)).all())}, "scientific_gate": {"status": "pass" if scientific_pass else "fail", "families_with_variable_multinode_contexts": variable_families, "minimum_families": int(config["evaluation"]["minimum_families_with_variable_contexts"]), "variable_multinode_context_fraction": overall_variable, "minimum_variable_context_fraction": float(config["evaluation"]["minimum_variable_context_fraction"]), "multinode_contexts": len(multinode_contexts)}, "sets": len(values), "contexts": len(contexts), "families": values["system_family"].nunique()}
    if not all(audit["checks"].values()): audit["status"] = "fail"; raise ValueError(f"set-value audit failed: {audit}")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _plot(values, contexts, decisions, report_dir)
    manifest = {"experiment_id": experiment_id, "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"), "elapsed_seconds": round(time.perf_counter() - started, 3), "git_commit": _git_value(["rev-parse", "HEAD"]), "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])), "python": platform.python_version(), "config_path": str(config_path), "config_sha256": _sha256(config_path), "artifact_audit": audit["status"], "scientific_gate": audit["scientific_gate"]["status"]}
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset_lines = "\n".join(
        f"- {DATASET_LABELS.get(row.dataset_id, row.dataset_id)}: median budget {int(row.median_budget)}; "
        + (f"variable multi-node contexts {row.variable_multinode_context_fraction:.1%}; mean spread {100 * row.mean_multinode_spread:.2f} percentage points." if row.multinode_contexts else "singleton-only under the 5% budget.")
        for row in dataset_summary.itertuples(index=False)
    )
    (report_dir / "README.md").write_text(
        f"# Fixed-budget sampled-set value pilot\n\nSets: {len(values):,}; contexts: {len(contexts)}; independent families: {values['system_family'].nunique()}. Artifact audit: **{audit['status']}**. Scientific gate: **{audit['scientific_gate']['status']}**.\n\n"
        f"The gate counts only contexts with budget >= 2. {len(multinode_contexts)} contexts met that definition; {overall_variable:.1%} had unequal sampled-set values, spanning {variable_families} independent families versus the required {int(config['evaluation']['minimum_families_with_variable_contexts'])}.\n\n{dataset_lines}\n\n"
        "No learned set planner is retained when the pre-specified multi-node set-label gate fails.\n",
        encoding="utf-8",
    )
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-budget set-value pilot")
    parser.add_argument("--config", type=Path, default=Path("configs/EXP-20260816-011_set_value_pilot.yaml"))
    args = parser.parse_args()
    results, reports = run(args.config)
    print(f"Results: {results}"); print(f"Reports: {reports}")


if __name__ == "__main__": main()
