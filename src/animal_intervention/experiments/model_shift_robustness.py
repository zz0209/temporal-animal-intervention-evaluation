from __future__ import annotations

import argparse
from datetime import UTC, datetime
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

from animal_intervention.transmission.contract import ExposureStream

from .historical_set_planning import _even_windows
from .intervention_delivery_sensitivity import (
    SYSTEM_FAMILY_LABELS,
    _hierarchical_summary,
    _parameter_pool,
    _select_parameter_regimes,
)
from .outbreak_response_pilot import (
    _git_value,
    _keyed_seed,
    _load_source_config,
    _load_windows,
    _sha256,
)
from .role_aware_sentinel_response import _replay_response
from .sequential_preparedness_update import _parameters


MODELS = ("temporal_sir", "temporal_seir_erlang")
MODEL_LABELS = {"temporal_sir": "SIR", "temporal_seir_erlang": "SEIR/Erlang"}
COLORS = {"temporal_sir": "#8C2D3E", "temporal_seir_erlang": "#2F6B4F"}
CONTEXT_KEYS = [
    "dataset_id",
    "network_id",
    "system_family",
    "anchor_id",
    "anchor_time",
    "initial_infected",
]


def _parse_set(signature: Any) -> set[str]:
    if pd.isna(signature) or str(signature) == "":
        return set()
    return set(str(signature).split("|"))


def _robust_selections(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    score_means = (
        scores.groupby(CONTEXT_KEYS + ["epidemic_model", "set_signature", "set_size"], observed=True, dropna=False)["value"]
        .mean()
        .reset_index()
    )
    selection_wide = selections.pivot(
        index=CONTEXT_KEYS + ["budget"],
        columns="epidemic_model",
        values=["history_exact", "stable"],
    )
    rows: list[dict[str, Any]] = []
    for key, frame in score_means.groupby(CONTEXT_KEYS, observed=True, dropna=False, sort=True):
        model_names = set(frame["epidemic_model"])
        if set(MODELS) - model_names:
            continue
        selection_key = (*key, int(frame["set_size"].max()))
        if selection_key not in selection_wide.index:
            continue
        selected = selection_wide.loc[selection_key]
        budget = int(selection_key[-1])
        target = frame.loc[frame["set_size"].eq(budget)]
        matrix = target.pivot(index="set_signature", columns="epidemic_model", values="value").dropna(subset=list(MODELS))
        if matrix.empty:
            continue
        normalized_regret = pd.DataFrame(index=matrix.index)
        for model in MODELS:
            values = matrix[model].astype(float)
            span = float(values.max() - values.min())
            normalized_regret[model] = 0.0 if span <= 1e-12 else (float(values.max()) - values) / span
        normalized_regret["worst_regret"] = normalized_regret[list(MODELS)].max(axis=1)
        normalized_regret["mean_regret"] = normalized_regret[list(MODELS)].mean(axis=1)
        sir_plan = str(selected[("history_exact", MODELS[0])])
        seir_plan = str(selected[("history_exact", MODELS[1])])
        normalized_regret["model_vote"] = normalized_regret.index.to_series().map(
            lambda signature: int(signature == sir_plan) + int(signature == seir_plan)
        )
        robust_plan = str(
            normalized_regret.reset_index()
            .sort_values(
                ["worst_regret", "mean_regret", "model_vote", "set_signature"],
                ascending=[True, True, False, True],
            )
            .iloc[0]["set_signature"]
        )
        stable_values = {
            str(selected[("stable", model)]) for model in MODELS
        }
        if len(stable_values) != 1:
            raise ValueError(f"Stable set differs by epidemic model for context {key}")
        rows.append(
            {
                **dict(zip(CONTEXT_KEYS, key)),
                "budget": budget,
                "sir_plan": sir_plan,
                "seir_plan": seir_plan,
                "robust_plan": robust_plan,
                "stable": stable_values.pop(),
                "model_specific_agreement": sir_plan == seir_plan,
                "robust_worst_normalized_history_regret": float(normalized_regret.loc[robust_plan, "worst_regret"]),
                "robust_mean_normalized_history_regret": float(normalized_regret.loc[robust_plan, "mean_regret"]),
            }
        )
    return pd.DataFrame(rows)


def _model_specification(base_config: dict[str, Any], model_name: str) -> dict[str, Any]:
    matches = [item for item in base_config["decision"]["epidemic_models"] if item["name"] == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one specification for {model_name}")
    return dict(matches[0])


def _context_tasks(
    base_config: dict[str, Any],
    robust: pd.DataFrame,
    maximum_contexts: int | None,
) -> list[dict[str, Any]]:
    contexts = robust.sort_values(CONTEXT_KEYS).copy()
    if maximum_contexts is not None:
        contexts = contexts.head(int(maximum_contexts))
    tasks: list[dict[str, Any]] = []
    for dataset_id, dataset_contexts in contexts.groupby("dataset_id", observed=True, sort=True):
        specification = base_config["data"]["datasets"][str(dataset_id)]
        source_config = _load_source_config(Path(specification["source_config"]))
        windows = _load_windows(str(dataset_id), source_config)
        default_network = str(specification.get("network_id", "all"))
        for window in windows:
            window.setdefault("network_id", default_network)
        lookup = {
            (str(window["network_id"]), str(window["anchor"].anchor_id), pd.Timestamp(window["anchor"].anchor_time)): window
            for window in windows
        }
        parameters = _parameter_pool(
            Path(specification["source_results"]) / "parameter_selection.csv",
            str(base_config["evaluation"]["parameter_pool"]),
        )
        selected_parameter = _select_parameter_regimes(
            list(parameters.itertuples(index=False)),
            str(base_config["evaluation"]["parameter_selection_mode"]),
        )
        if len(selected_parameter) != 1:
            raise ValueError(f"Expected one parameter profile for {dataset_id}")
        parameter = selected_parameter[0][1]
        for context in dataset_contexts.to_dict("records"):
            lookup_key = (
                str(context["network_id"]),
                str(context["anchor_id"]),
                pd.Timestamp(context["anchor_time"]),
            )
            if lookup_key not in lookup:
                raise KeyError(f"Missing temporal window {lookup_key}")
            for evaluator_model in MODELS:
                tasks.append(
                    {
                        **context,
                        "window": lookup[lookup_key],
                        "parameter": parameter,
                        "model": _model_specification(base_config, evaluator_model),
                    }
                )
    return tasks


def _evaluate_task(task: dict[str, Any], config: dict[str, Any], future_blocks: int) -> pd.DataFrame:
    anchor = task["window"]["anchor"]
    start = pd.Timestamp(anchor.anchor_time)
    end = pd.Timestamp(anchor.horizon_end)
    model_name = str(task["model"]["name"])
    mean_period = pd.Timedelta(days=float(task["parameter"].mean_infectious_period_days))
    engine, parameters = _parameters(task["parameter"], task["model"], mean_period)
    future: ExposureStream = task["window"]["future"]
    population = len(future.nodes())
    initial = str(task["initial_infected"])
    methods = {
        "sir_plan": _parse_set(task["sir_plan"]),
        "seir_plan": _parse_set(task["seir_plan"]),
        "robust_plan": _parse_set(task["robust_plan"]),
        "stable": _parse_set(task["stable"]),
    }
    rows: list[dict[str, Any]] = []
    for block in range(int(future_blocks)):
        world_seed = _keyed_seed(
            int(config["evaluation"]["seed"]),
            task["dataset_id"],
            task["network_id"],
            task["anchor_id"],
            model_name,
            initial,
            "model_shift_future",
            block,
        )
        natural = engine.simulate(
            future,
            parameters,
            initial_infected=(initial,),
            start_time=start,
            end_time=end,
            world_seed=world_seed,
        )
        case_only, _ = _replay_response(
            engine=engine,
            parameters=parameters,
            future=future,
            natural=natural,
            initial=initial,
            world_seed=world_seed,
            start_time=start,
            end_time=end,
            detection_time=start,
            action_delay=pd.Timedelta(0),
            targets={initial},
            residual=float(config["decision"]["residual_contact_multiplier"]),
        )
        for method, additional in methods.items():
            result, _ = _replay_response(
                engine=engine,
                parameters=parameters,
                future=future,
                natural=natural,
                initial=initial,
                world_seed=world_seed,
                start_time=start,
                end_time=end,
                detection_time=start,
                action_delay=pd.Timedelta(0),
                targets={initial, *additional},
                residual=float(config["decision"]["residual_contact_multiplier"]),
            )
            rows.append(
                {
                    "dataset_id": task["dataset_id"],
                    "network_id": task["network_id"],
                    "system_family": task["system_family"],
                    "analysis_cluster_id": f"{task['dataset_id']}::{task['network_id']}::{task['anchor_id']}",
                    "anchor_id": task["anchor_id"],
                    "anchor_time": start,
                    "initial_infected": initial,
                    "evaluator_model": model_name,
                    "future_block": block,
                    "world_seed": world_seed,
                    "population_size": population,
                    "budget": int(task["budget"]),
                    "method": method,
                    "selected_nodes": "|".join(sorted(additional)),
                    "value": (case_only.final_size - result.final_size) / population,
                }
            )
    return pd.DataFrame(rows)


def _contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset_id", "network_id", "system_family", "analysis_cluster_id",
        "anchor_id", "anchor_time", "initial_infected", "evaluator_model",
        "future_block", "world_seed", "population_size", "budget",
    ]
    wide = worlds.pivot(index=keys, columns="method", values="value").reset_index()
    matched = np.where(
        wide["evaluator_model"].eq(MODELS[0]),
        wide["sir_plan"],
        wide["seir_plan"],
    )
    crossed = np.where(
        wide["evaluator_model"].eq(MODELS[0]),
        wide["seir_plan"],
        wide["sir_plan"],
    )
    candidate_oracle = wide[["sir_plan", "seir_plan"]].max(axis=1)
    definitions = {
        "matched_minus_cross_model": matched - crossed,
        "robust_minus_matched": wide["robust_plan"] - matched,
        "robust_regret_to_candidate_oracle": candidate_oracle - wide["robust_plan"],
        "matched_regret_to_candidate_oracle": candidate_oracle - matched,
        "robust_minus_stable": wide["robust_plan"] - wide["stable"],
    }
    frames = []
    for name, values in definitions.items():
        frame = wide[keys].copy()
        frame["contrast"] = name
        frame["value"] = np.asarray(values, dtype=float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _plot(summary: pd.DataFrame, family: pd.DataFrame, robust: pd.DataFrame, path: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(9.2, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], hspace=0.62, wspace=0.58)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    family_agreement = (
        robust.groupby("system_family", observed=True)["model_specific_agreement"].mean().sort_values()
    )
    y = np.arange(len(family_agreement))
    axes[0].barh(y, 100 * family_agreement.to_numpy(float), color="#6B5A7E", height=0.62)
    axes[0].set_yticks(y, [SYSTEM_FAMILY_LABELS.get(name, name) for name in family_agreement.index])
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Identical past-selected sets (%)")
    axes[0].set_title("A  Selection agreement", loc="left", weight="bold")
    axes[0].grid(axis="x", color="#E2E2E2", lw=0.6)

    selected_contrasts = ["matched_minus_cross_model", "robust_regret_to_candidate_oracle"]
    labels = ["Matched model minus\ncross-model plan", "Robust-plan regret to\nbetter candidate"]
    for y_value, (contrast, label) in enumerate(zip(selected_contrasts, labels)):
        for offset, model in [(-0.12, MODELS[0]), (0.12, MODELS[1])]:
            row = summary.loc[summary["contrast"].eq(contrast) & summary["evaluator_model"].eq(model)].iloc[0]
            mean, low, high = 100 * row[["family_equal_mean", "ci_low", "ci_high"]].to_numpy(float)
            axes[1].errorbar(
                mean, y_value + offset, xerr=[[mean - low], [high - mean]],
                fmt="s" if model == MODELS[0] else "D", color=COLORS[model],
                ms=5, capsize=2.5, lw=1.1,
            )
    axes[1].axvline(0, color="#767676", ls=(0, (3, 2)), lw=0.8)
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Avoided attack-rate difference (points)")
    axes[1].set_title("B  Future model-shift cost", loc="left", weight="bold")
    axes[1].grid(axis="x", color="#E2E2E2", lw=0.6)

    selected = family.loc[family["contrast"].eq("matched_minus_cross_model")].copy()
    family_order = sorted(selected["system_family"].unique())
    positions = np.arange(len(family_order))
    for offset, model in [(-0.13, MODELS[0]), (0.13, MODELS[1])]:
        values = (
            selected.loc[selected["evaluator_model"].eq(model)]
            .set_index("system_family")
            .reindex(family_order)["mean_value"]
            .to_numpy(float)
        )
        axes[2].scatter(100 * values, positions + offset, marker="s" if model == MODELS[0] else "D", color=COLORS[model], s=28, label=MODEL_LABELS[model])
    axes[2].axvline(0, color="#767676", ls=(0, (3, 2)), lw=0.8)
    axes[2].set_yticks(positions, [SYSTEM_FAMILY_LABELS.get(name, name) for name in family_order])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Matched-minus-cross-model value (points)")
    axes[2].set_title("C  Family effects", loc="left", weight="bold")
    axes[2].grid(axis="x", color="#E2E2E2", lw=0.6)
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle("Past-selected intervention sets under epidemic-model shift", weight="bold", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.90)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][profile_name]
    base_config_path = Path(config["data"]["planning_config"])
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    planning_dir = Path(config["data"]["planning_results"])
    scores = pd.read_csv(planning_dir / "history_set_values.csv.gz", dtype={"initial_infected": str, "set_signature": str})
    selections = pd.read_csv(planning_dir / "planner_selections.csv.gz", dtype={"initial_infected": str})
    for frame in (scores, selections):
        frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], format="mixed")
    robust = _robust_selections(scores, selections)
    tasks = _context_tasks(base_config, robust, profile.get("maximum_contexts"))
    result_frames = []
    for task in tqdm(tasks, desc="Cross-model future replay", unit="task"):
        result_frames.append(_evaluate_task(task, config, int(profile["future_blocks"])))
    worlds = pd.concat(result_frames, ignore_index=True)
    contrasts = _contrasts(worlds)
    summary, family = _hierarchical_summary(
        contrasts,
        value_column="value",
        group_columns=["evaluator_model", "contrast"],
        bootstrap_replicates=int(profile.get("bootstrap_replicates", config["evaluation"]["bootstrap_replicates"])),
        seed=int(config["evaluation"]["seed"]),
    )
    expected_methods = {"sir_plan", "seir_plan", "robust_plan", "stable"}
    checks = {
        "both_evaluator_models": set(worlds["evaluator_model"]) == set(MODELS),
        "all_policy_arms": worlds.groupby(CONTEXT_KEYS + ["evaluator_model", "future_block"], observed=True)["method"].nunique().eq(len(expected_methods)).all(),
        "paired_worlds": worlds.groupby(CONTEXT_KEYS + ["evaluator_model", "future_block"], observed=True)["world_seed"].nunique().eq(1).all(),
        "equal_budget": worlds.groupby(CONTEXT_KEYS + ["evaluator_model", "future_block"], observed=True)["selected_nodes"].apply(lambda values: len({len(_parse_set(value)) for value in values})).eq(1).all(),
        "finite_values": np.isfinite(worlds["value"]).all() and np.isfinite(contrasts["value"]).all(),
        "bounded_values": worlds["value"].between(-1, 1).all(),
        "past_only_selection_source": True,
        "all_full_contexts": profile_name != "full" or len(robust) == worlds[CONTEXT_KEYS].drop_duplicates().shape[0],
        "five_families_full": profile_name != "full" or worlds["system_family"].nunique() == 5,
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {name: bool(value) for name, value in checks.items()},
        "contexts": int(worlds[CONTEXT_KEYS].drop_duplicates().shape[0]),
        "families": int(worlds["system_family"].nunique()),
        "future_policy_evaluations": int(len(worlds)),
        "model_specific_selection_agreement": float(robust["model_specific_agreement"].mean()),
    }
    if audit["status"] != "pass":
        raise ValueError(audit)
    results_dir = Path(config["outputs"]["results_root"]) / config["experiment"]["id"] / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / config["experiment"]["id"] / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    robust.to_csv(results_dir / "past_model_selections.csv", index=False)
    worlds.to_csv(results_dir / "future_policy_worlds.csv.gz", index=False, compression="gzip")
    contrasts.to_csv(results_dir / "model_shift_contrasts.csv.gz", index=False, compression="gzip")
    summary.to_csv(results_dir / "contrast_summary.csv", index=False)
    family.to_csv(results_dir / "family_contrasts.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = {**config, "runtime": {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
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
        "source_sha256": _sha256(Path(__file__)),
        "inputs": {
            str(base_config_path): _sha256(base_config_path),
            str(planning_dir / "history_set_values.csv.gz"): _sha256(planning_dir / "history_set_values.csv.gz"),
            str(planning_dir / "planner_selections.csv.gz"): _sha256(planning_dir / "planner_selections.csv.gz"),
        },
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot(summary, family, robust, report_dir / "model_shift_robustness.png", int(profile["render_dpi"]))
    (report_dir / "STAGE_REPORT.md").write_text(
        "# Epidemic-model shift stress test\n\n"
        f"Model-specific past planners selected the same set in {100 * audit['model_specific_selection_agreement']:.1f}% of contexts. "
        "Future replay evaluates each frozen set under both epidemic models with paired contacts and random draws. "
        "The robust arm minimizes the worst normalized regret across the two historical replay objectives.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test past-selected intervention sets under epidemic-model shift.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
