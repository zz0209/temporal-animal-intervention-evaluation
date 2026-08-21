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
from animal_intervention.surveillance import greedy_history_coverage
from animal_intervention.transmission import ExposureStream

from .history_baseline_substitution import _markdown_table
from .intervention_delivery_sensitivity import (
    SYSTEM_FAMILY_LABELS,
    _parameter_pool,
    _select_parameter_regimes,
)
from .outbreak_response_pilot import (
    _git_value,
    _keyed_seed,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _sha256,
)
from .response_targeting_increment import (
    WORLD_KEYS,
    _compute_decomposition,
    _random_response_targets,
)
from .role_aware_sentinel_response import (
    _detection_metrics,
    _replay_response,
    _top_history,
)
from .sequential_preparedness_update import _budget, _parameters


def _with_group_mode(stream: ExposureStream, mode: str) -> ExposureStream:
    groups = stream.group_exposures.copy()
    if groups.empty:
        raise ValueError("group-mixing sensitivity requires group exposures")
    groups["group_mixing_mode"] = mode
    result = ExposureStream(
        dataset_id=stream.dataset_id,
        population_nodes=stream.population_nodes,
        dyadic_exposures=stream.dyadic_exposures,
        group_exposures=groups,
        group_memberships=stream.group_memberships,
        metadata={**stream.metadata, "group_mixing_mode": mode},
    )
    result.validate()
    return result


def _member_time_mean_competitors(stream: ExposureStream) -> float:
    groups = stream.group_exposures[["group_event_id", "start_time", "end_time"]].copy()
    sizes = (
        stream.group_memberships.groupby("group_event_id", observed=True)
        .size()
        .rename("group_size")
    )
    groups = groups.join(sizes, on="group_event_id")
    groups = groups.loc[groups["group_size"].ge(2)].copy()
    durations = (
        pd.to_datetime(groups["end_time"]) - pd.to_datetime(groups["start_time"])
    ).dt.total_seconds().to_numpy(float)
    sizes_array = groups["group_size"].to_numpy(float)
    weights = durations * sizes_array
    if not len(weights) or float(weights.sum()) <= 0:
        raise ValueError("group stream contains no positive multi-animal member-time")
    return float(np.average(sizes_array - 1.0, weights=weights))


def _alternate_window(window: dict[str, Any]) -> tuple[dict[str, Any], float]:
    denominator = _member_time_mean_competitors(window["future"])
    altered = dict(window)
    altered["history"] = _with_group_mode(window["history"], "undiluted_clique")
    altered["future"] = _with_group_mode(window["future"], "undiluted_clique")
    return altered, denominator


def _alternate_task_rows(task: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    decision = config["decision"]
    evaluation = config["evaluation"]
    window, denominator = _alternate_window(task["window"])
    parameter = task["parameter"]._replace(
        beta=float(task["parameter"].beta) / denominator
    )
    rows = []
    for block in range(int(task["random_blocks"])):
        for initial in task["seeds"]:
            anchor = window["anchor"]
            start_time = pd.Timestamp(anchor.anchor_time)
            end_time = pd.Timestamp(anchor.horizon_end)
            mean_period = pd.Timedelta(
                days=float(parameter.mean_infectious_period_days)
            )
            engine, parameters = _parameters(parameter, task["model"], mean_period)
            world_seed = _keyed_seed(
                int(evaluation["seed"]),
                task["dataset_id"],
                anchor.anchor_id,
                parameter.parameter_id,
                block,
                str(initial),
            )
            natural = engine.simulate(
                window["future"],
                parameters,
                initial_infected=(str(initial),),
                start_time=start_time,
                end_time=end_time,
                world_seed=world_seed,
            )
            eligible = set(map(str, window["eligible"]))
            population_size = len(window["future"].nodes())
            sentinel_budget = _budget(
                len(eligible),
                int(decision["minimum_budget"]),
                float(decision["sentinel_budget_fraction"]),
            )
            sentinel_seed = _keyed_seed(
                int(evaluation["seed"]),
                task["dataset_id"],
                anchor.anchor_id,
                "sentinel_set",
                block,
            )
            sentinels = set(
                greedy_history_coverage(
                    window["history"], eligible, sentinel_budget, seed=sentinel_seed
                )
            )
            detection = _detection_metrics(
                natural,
                sentinels,
                population_size,
                float(decision["early_detection_threshold_fraction"]),
            )
            detected = set(detection["detected_nodes"])
            response_capacity = _budget(
                len(eligible),
                int(decision["minimum_budget"]),
                float(decision["response_budget_fraction"]),
            )
            response_budget = min(response_capacity, len(eligible - detected))
            history_additional = _top_history(
                task["stable_scores"],
                eligible,
                response_budget,
                sentinel_seed,
                excluded=detected,
            )
            action_delay = mean_period * float(
                decision["action_delay_fraction_of_mean_infectious_period"]
            )
            common = {
                "dataset_id": task["dataset_id"],
                "network_id": task["network_id"],
                "system_family": task["system_family"],
                "analysis_cluster_id": task["analysis_cluster_id"],
                "anchor_id": anchor.anchor_id,
                "anchor_time": start_time,
                "horizon_end": end_time,
                "parameter_id": parameter.parameter_id,
                "beta": float(parameter.beta),
                "mean_infectious_period_days": float(
                    parameter.mean_infectious_period_days
                ),
                "epidemic_model": str(task["model"]["name"]),
                "random_block": block,
                "initial_infected": str(initial),
                "world_seed": world_seed,
                "population_size": population_size,
                "sentinel_method": "history_coverage",
                "sentinel_budget": sentinel_budget,
                "response_budget": response_budget,
                "response_capacity": response_capacity,
                "sentinel_nodes": "|".join(sorted(sentinels)),
                "detected": bool(detection["detected"]),
                "detection_time": detection["detection_time"],
                "detection_burden": int(detection["detection_burden"]),
                "detection_burden_rate": float(detection["detection_burden_rate"]),
                "early_detection": bool(detection["early_detection"]),
                "natural_final_size": natural.final_size,
            }
            policy_rows = []
            for method, additional in [
                ("case_only", set()),
                ("history_weight", history_additional),
            ]:
                targets = detected | additional
                result, action_start = _replay_response(
                    engine=engine,
                    parameters=parameters,
                    future=window["future"],
                    natural=natural,
                    initial=str(initial),
                    world_seed=world_seed,
                    start_time=start_time,
                    end_time=end_time,
                    detection_time=detection["detection_time"],
                    action_delay=action_delay,
                    targets=targets,
                    residual=float(decision["residual_contact_multiplier"]),
                )
                policy_rows.append(
                    {
                        **common,
                        "policy": f"history_coverage__{method}",
                        "response_method": method,
                        "response_nodes": "|".join(sorted(targets)),
                        "action_start": action_start,
                        "final_size": result.final_size,
                        "final_attack_rate": result.final_size / population_size,
                    }
                )
            random_outcomes = []
            random_nodes = []
            for target_replicate in range(int(task["random_target_replicates"])):
                response_seed = _keyed_seed(
                    int(evaluation["seed"]),
                    task["dataset_id"],
                    task["network_id"],
                    anchor.anchor_id,
                    parameter.parameter_id,
                    task["model"]["name"],
                    block,
                    str(initial),
                    "random_response",
                    target_replicate,
                )
                additional = _random_response_targets(
                    eligible, response_budget, response_seed, excluded=detected
                )
                targets = detected | additional
                result, action_start = _replay_response(
                    engine=engine,
                    parameters=parameters,
                    future=window["future"],
                    natural=natural,
                    initial=str(initial),
                    world_seed=world_seed,
                    start_time=start_time,
                    end_time=end_time,
                    detection_time=detection["detection_time"],
                    action_delay=action_delay,
                    targets=targets,
                    residual=float(decision["residual_contact_multiplier"]),
                )
                random_outcomes.append(result.final_size)
                random_nodes.append("|".join(sorted(targets)))
            policy_rows.append(
                {
                    **common,
                    "policy": "history_coverage__random",
                    "response_method": "random",
                    "response_nodes": " || ".join(random_nodes),
                    "action_start": action_start,
                    "final_size": float(np.mean(random_outcomes)),
                    "final_attack_rate": float(np.mean(random_outcomes))
                    / population_size,
                    "random_target_replicates": len(random_outcomes),
                }
            )
            combined = pd.DataFrame(policy_rows)
            combined["mapping"] = "hazard_normalized_undiluted_clique"
            combined["primary_beta"] = float(task["parameter"].beta)
            combined["mapping_beta"] = float(parameter.beta)
            combined["member_time_mean_competitors"] = denominator
            rows.append(combined)
    return pd.concat(
        [frame.dropna(axis=1, how="all") for frame in rows], ignore_index=True
    )


def _primary_worlds(config: dict[str, Any], profile: dict[str, Any]) -> pd.DataFrame:
    source = pd.read_csv(
        config["data"]["source_policy_worlds"],
        dtype={"initial_infected": str, "network_id": str},
    )
    source = source.loc[
        source["dataset_id"].isin(profile["datasets"])
        & source["sentinel_method"].eq("history_coverage")
        & source["response_method"].isin(["case_only", "history_weight"])
    ].copy()
    random = pd.read_csv(
        config["data"]["source_random_replicates"],
        dtype={"initial_infected": str, "network_id": str},
    )
    random = random.loc[
        random["dataset_id"].isin(profile["datasets"])
        & random["target_replicate"].lt(int(profile["random_target_replicates"]))
    ].copy()
    random = (
        random.groupby(WORLD_KEYS, observed=True, as_index=False)
        .agg(
            {
                **{
                    column: "first"
                    for column in random.columns
                    if column
                    not in set(
                        WORLD_KEYS
                        + ["target_replicate", "final_size", "final_attack_rate"]
                    )
                },
                "final_size": "mean",
                "final_attack_rate": "mean",
            }
        )
    )
    result = pd.concat([source, random], ignore_index=True)
    result["mapping"] = "frequency_dependent"
    result["primary_beta"] = result["beta"]
    result["mapping_beta"] = result["beta"]
    result["member_time_mean_competitors"] = 1.0
    return result


def _mapping_metrics(worlds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mapping, frame in worlds.groupby("mapping", observed=True):
        contrasts = _compute_decomposition(frame)
        response = contrasts.copy()
        natural = frame.drop_duplicates(WORLD_KEYS).copy()
        natural["natural_attack_rate"] = (
            natural["natural_final_size"] / natural["population_size"]
        )
        natural = natural[WORLD_KEYS + ["natural_attack_rate"]]
        response = response.merge(natural, on=WORLD_KEYS, validate="many_to_one")
        response["mapping"] = mapping
        rows.append(response)
    return pd.concat(rows, ignore_index=True)


def _plot_semantics(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    keys = WORLD_KEYS + ["contrast"]
    primary = metrics.loc[metrics["mapping"].eq("frequency_dependent")]
    alternate = metrics.loc[
        metrics["mapping"].eq("hazard_normalized_undiluted_clique")
    ]
    paired = primary.merge(alternate, on=keys, suffixes=("_primary", "_alternate"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    panels = [
        ("natural_attack_rate", "capacity_increment", "Baseline attack rate"),
        ("value", "capacity_increment", "Extra response-capacity value"),
        ("value", "targeting_increment", "History targeting over random"),
    ]
    colors = {"temporal_sir": "#4C78A8", "temporal_seir_erlang": "#F58518"}
    for axis, (column, contrast, title) in zip(axes, panels):
        selected = paired.loc[paired["contrast"].eq(contrast)]
        for model, frame in selected.groupby("epidemic_model", observed=True):
            axis.scatter(
                100 * frame[f"{column}_primary"],
                100 * frame[f"{column}_alternate"],
                s=28,
                alpha=0.65,
                color=colors[str(model)],
                label=str(model).replace("temporal_", "").replace("_erlang", ""),
            )
        values = np.concatenate(
            [
                100 * selected[f"{column}_primary"].to_numpy(float),
                100 * selected[f"{column}_alternate"].to_numpy(float),
            ]
        )
        low, high = float(values.min()), float(values.max())
        margin = max(0.5, 0.05 * (high - low))
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#666666")
        axis.set_xlim(low - margin, high + margin)
        axis.set_ylim(low - margin, high + margin)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_xlabel("Frequency-dependent primary (%)")
        axis.set_ylabel("Hazard-normalized clique sensitivity (%)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Group-event conclusions under two transmission semantics", fontsize=20, weight="bold")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.13, wspace=0.28)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = dict(config["profiles"][profile_name])
    evaluation = config["evaluation"]
    stable = pd.read_csv(
        config["data"]["stable_prediction_path"],
        dtype={"candidate_id": str, "network_id": str},
    )
    stable["anchor_time"] = pd.to_datetime(stable["anchor_time"], format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / config["experiment"]["id"] / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / config["experiment"]["id"] / profile_name
    checkpoint_dir = results_dir / "checkpoints"
    for directory in (results_dir, report_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(dataset_id, source_config)
        default_network_id = str(specification.get("network_id", "all"))
        for window in windows:
            window.setdefault("network_id", default_network_id)
        available_keys = set(
            stable.loc[
                stable["dataset_id"].eq(dataset_id),
                ["network_id", "anchor_time"],
            ].itertuples(index=False, name=None)
        )
        windows = [
            window
            for window in windows
            if (
                str(window["network_id"]),
                pd.Timestamp(window["anchor"].anchor_time),
            )
            in available_keys
        ]
        maximum = profile.get("max_anchors_per_dataset")
        if maximum is not None:
            windows = windows[: int(maximum)]
        parameters = _parameter_pool(
            Path(specification["source_results"]) / "parameter_selection.csv",
            str(evaluation["parameter_pool"]),
        )
        for window in windows:
            network_id = str(window.get("network_id", specification.get("network_id", "all")))
            compatible = []
            for parameter in parameters.itertuples(index=False):
                delay = pd.Timedelta(days=float(parameter.mean_infectious_period_days)) * float(
                    config["decision"]["action_delay_fraction_of_mean_infectious_period"]
                )
                if pd.Timestamp(window["anchor"].anchor_time) + delay < pd.Timestamp(window["anchor"].horizon_end):
                    compatible.append(parameter)
            selected = _select_parameter_regimes(
                compatible, str(evaluation["parameter_selection_mode"])
            )
            if len(selected) != 1:
                continue
            _, parameter = selected[0]
            score = _matching_stable_scores(
                stable,
                dataset_id,
                network_id,
                window["anchor"].anchor_time,
                window["eligible"],
            )
            seeds = stable_hash_order(
                list(map(str, window["eligible"])),
                int(evaluation["seed"]),
                dataset_id,
                window["anchor"].anchor_id,
                "role_aware_seeds",
            )[: int(profile["seeds_per_anchor"])]
            cluster = f"{dataset_id}::{network_id}::{window['anchor'].anchor_id}"
            for model in config["decision"]["epidemic_models"]:
                tasks.append(
                    {
                        "dataset_id": dataset_id,
                        "network_id": network_id,
                        "system_family": specification["system_family"],
                        "analysis_cluster_id": cluster,
                        "window": window,
                        "parameter": parameter,
                        "model": dict(model),
                        "stable_scores": score,
                        "seeds": seeds,
                        "random_blocks": int(profile["random_blocks"]),
                        "random_target_replicates": int(profile["random_target_replicates"]),
                    }
                )

    fingerprint = hashlib.sha256(config_path.read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    frames = []
    for task in tqdm(tasks, desc="Group-mixing semantic worlds", unit="task"):
        identity = "|".join(
            [
                fingerprint,
                task["dataset_id"],
                task["network_id"],
                task["window"]["anchor"].anchor_id,
                task["model"]["name"],
            ]
        )
        checkpoint = checkpoint_dir / f"alternate_{hashlib.sha256(identity.encode()).hexdigest()[:18]}.csv.gz"
        if not checkpoint.exists() and config["execution"].get("resume", True):
            matching_checkpoints = []
            for candidate in checkpoint_dir.glob("alternate_*.csv.gz"):
                preview = pd.read_csv(
                    candidate,
                    nrows=1,
                    dtype={"network_id": str, "initial_infected": str},
                )
                if preview.empty:
                    continue
                row = preview.iloc[0]
                if (
                    str(row.get("dataset_id")) == task["dataset_id"]
                    and str(row.get("network_id")) == task["network_id"]
                    and str(row.get("anchor_id"))
                    == task["window"]["anchor"].anchor_id
                    and str(row.get("epidemic_model")) == task["model"]["name"]
                ):
                    matching_checkpoints.append(candidate)
            if matching_checkpoints:
                checkpoint = max(
                    matching_checkpoints,
                    key=lambda candidate: candidate.stat().st_mtime,
                )
        if checkpoint.exists() and config["execution"].get("resume", True):
            frame = pd.read_csv(checkpoint, dtype={"initial_infected": str, "network_id": str})
        else:
            frame = _alternate_task_rows(task, config)
            frame.to_csv(checkpoint, index=False, compression="gzip")
        frames.append(frame)
    alternate = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames],
        ignore_index=True,
    )
    primary = _primary_worlds(config, profile)
    alternate_keys = set(map(tuple, alternate[WORLD_KEYS].drop_duplicates().itertuples(index=False, name=None)))
    primary = primary.loc[
        primary[WORLD_KEYS].apply(tuple, axis=1).isin(alternate_keys)
    ].copy()
    worlds = pd.concat([primary, alternate], ignore_index=True)
    metrics = _mapping_metrics(worlds)
    paired = metrics.loc[metrics["mapping"].eq("frequency_dependent")].merge(
        metrics.loc[metrics["mapping"].eq("hazard_normalized_undiluted_clique")],
        on=WORLD_KEYS + ["contrast"],
        suffixes=("_primary", "_alternate"),
        validate="one_to_one",
    )
    summaries = (
        metrics.groupby(
            ["mapping", "epidemic_model", "system_family", "contrast"],
            observed=True,
        )
        .agg(
            mean_natural_attack_rate=("natural_attack_rate", "mean"),
            mean_response_value=("value", "mean"),
            contexts=("analysis_cluster_id", "nunique"),
        )
        .reset_index()
    )
    audit_checks = {
        "three_group_datasets": set(worlds["dataset_id"]) == set(profile["datasets"]),
        "two_mapping_arms": worlds["mapping"].nunique() == 2,
        "world_contrast_keys_paired": len(paired) == len(metrics) // 2,
        "finite_scale": bool(np.isfinite(alternate["member_time_mean_competitors"]).all()),
        "scale_exceeds_one": bool(alternate["member_time_mean_competitors"].gt(1).all()),
        "alternate_beta_reduced": bool(alternate["mapping_beta"].lt(alternate["primary_beta"]).all()),
        "three_policy_arms_each_mapping": bool(
            worlds.groupby(WORLD_KEYS + ["mapping"], observed=True)["response_method"].nunique().eq(3).all()
        ),
    }
    audit = {
        "status": "pass" if all(audit_checks.values()) else "fail",
        "checks": audit_checks,
        "datasets": int(worlds["dataset_id"].nunique()),
        "families": int(worlds["system_family"].nunique()),
        "paired_worlds": int(paired[WORLD_KEYS].drop_duplicates().shape[0]),
        "scope": "hazard_normalized_group_mixing_semantic_sensitivity",
    }
    if audit["status"] != "pass":
        raise ValueError(f"group-mixing semantic audit failed: {audit}")
    worlds.to_csv(results_dir / "mapping_policy_worlds.csv.gz", index=False, compression="gzip")
    metrics.to_csv(results_dir / "mapping_world_metrics.csv.gz", index=False, compression="gzip")
    paired.to_csv(results_dir / "paired_mapping_metrics.csv.gz", index=False, compression="gzip")
    summaries.to_csv(results_dir / "mapping_family_summary.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [
        config_path,
        Path(__file__),
        Path(config["data"]["source_policy_worlds"]),
        Path(config["data"]["source_random_replicates"]),
        Path(config["data"]["stable_prediction_path"]),
    ]
    source_paths.extend(
        Path(config["data"]["datasets"][dataset_id]["source_config"])
        for dataset_id in profile["datasets"]
    )
    pd.DataFrame(
        [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        ]
    ).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "profile": profile_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot_semantics(metrics, report_dir / "group_mixing_semantic_sensitivity.png", int(profile["render_dpi"]))
    report = "# Group-mixing semantic sensitivity\n\n"
    report += "The alternative arm replaces frequency-dependent group mixing with an undiluted clique, then analytically reduces beta by the member-time-weighted mean number of co-members. This matches first-order exposure opportunity without fitting beta to the observed intervention result.\n\n"
    report += _markdown_table(summaries)
    report += "\n\nThis is a measurement-semantics sensitivity analysis across two independent bird-system families, not additional independent biological validation.\n"
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group-event transmission-semantics sensitivity.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
