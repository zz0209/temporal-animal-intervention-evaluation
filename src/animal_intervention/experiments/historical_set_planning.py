from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import itertools
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

from animal_intervention.estimands.intervention_value import slice_stream
from animal_intervention.evaluation import stable_hash_order
from animal_intervention.surveillance import history_pair_weights
from animal_intervention.transmission.contract import ExposureStream

from .history_baseline_substitution import _markdown_table
from .immediate_case_targeting import _case_conditioned_history_sets
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
    _matching_stable_scores,
    _sha256,
)
from .role_aware_sentinel_response import _replay_response, _top_history
from .sequential_preparedness_update import _budget, _parameters


def _shift_stream(stream: ExposureStream, delta: pd.Timedelta) -> ExposureStream:
    dyadic = stream.dyadic_exposures.copy()
    groups = stream.group_exposures.copy()
    for frame in (dyadic, groups):
        if not frame.empty:
            frame["start_time"] = pd.to_datetime(frame["start_time"]) + delta
            frame["end_time"] = pd.to_datetime(frame["end_time"]) + delta
    result = ExposureStream(
        dataset_id=stream.dataset_id,
        population_nodes=stream.population_nodes,
        dyadic_exposures=dyadic,
        group_exposures=groups,
        group_memberships=stream.group_memberships.copy(),
        metadata={**stream.metadata, "replay_shift_seconds": delta.total_seconds()},
    )
    result.validate()
    return result


def _recent_history_replay(window: dict[str, Any]) -> ExposureStream:
    anchor = window["anchor"]
    start = pd.Timestamp(anchor.anchor_time)
    end = pd.Timestamp(anchor.horizon_end)
    duration = end - start
    historical = slice_stream(window["history"], start - duration, start)
    return _shift_stream(historical, duration)


def _candidate_pool(
    *,
    window: dict[str, Any],
    stable_scores: pd.DataFrame,
    eligible: set[str],
    initial: str,
    pool_per_signal: int,
    mean_period: pd.Timedelta,
    seed: int,
) -> tuple[tuple[str, ...], set[str], set[str]]:
    stable = _top_history(
        stable_scores,
        eligible,
        pool_per_signal,
        seed,
        excluded={initial},
    )
    static_ring = dict(
        _case_conditioned_history_sets(
            history_stream=window["history"],
            stable_scores=stable_scores,
            eligible=eligible,
            initial=initial,
            budget=pool_per_signal,
            history_start=pd.Timestamp(window["anchor"].history_start),
            anchor_time=pd.Timestamp(window["anchor"].anchor_time),
            recency_half_life=mean_period,
            seed=seed,
        )
    )["past_weight_ring"]
    ordered = []
    for node in [*sorted(stable), *sorted(static_ring)]:
        if node != initial and node not in ordered:
            ordered.append(node)
    return tuple(ordered), stable, static_ring


def _all_subsets(pool: tuple[str, ...], budget: int) -> list[tuple[str, ...]]:
    return [
        tuple(sorted(items))
        for size in range(budget + 1)
        for items in itertools.combinations(pool, size)
    ]


def _score_history_sets(
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    initial: str,
    pool: tuple[str, ...],
    budget: int,
) -> pd.DataFrame:
    decision = config["decision"]
    evaluation = config["evaluation"]
    anchor = task["window"]["anchor"]
    start = pd.Timestamp(anchor.anchor_time)
    end = pd.Timestamp(anchor.horizon_end)
    mean_period = pd.Timedelta(days=float(task["parameter"].mean_infectious_period_days))
    engine, parameters = _parameters(task["parameter"], task["model"], mean_period)
    replay = _recent_history_replay(task["window"])
    population = len(task["window"]["future"].nodes())
    rows = []
    for block in range(int(task["history_blocks"])):
        world_seed = _keyed_seed(
            int(evaluation["seed"]), task["dataset_id"], task["network_id"],
            anchor.anchor_id, task["model"]["name"], initial, "history_replay", block,
        )
        natural = engine.simulate(
            replay, parameters, initial_infected=(initial,), start_time=start,
            end_time=end, world_seed=world_seed,
        )
        case, _ = _replay_response(
            engine=engine, parameters=parameters, future=replay, natural=natural,
            initial=initial, world_seed=world_seed, start_time=start, end_time=end,
            detection_time=start, action_delay=pd.Timedelta(0), targets={initial},
            residual=float(decision["residual_contact_multiplier"]),
        )
        for subset in _all_subsets(pool, budget):
            result, _ = _replay_response(
                engine=engine, parameters=parameters, future=replay, natural=natural,
                initial=initial, world_seed=world_seed, start_time=start, end_time=end,
                detection_time=start, action_delay=pd.Timedelta(0),
                targets={initial, *subset},
                residual=float(decision["residual_contact_multiplier"]),
            )
            rows.append({
                "history_block": block,
                "set_signature": "|".join(subset),
                "set_size": len(subset),
                "value": (case.final_size - result.final_size) / population,
            })
    return pd.DataFrame(rows)


def _choose_sets(
    scores: pd.DataFrame,
    pool: tuple[str, ...],
    budget: int,
    stable_scores: pd.DataFrame,
) -> tuple[set[str], set[str], pd.DataFrame]:
    means = scores.groupby(["set_signature", "set_size"], observed=True)["value"].mean().reset_index()
    target = means.loc[means["set_size"].eq(budget)].copy()
    stable_map = stable_scores.set_index(stable_scores["candidate_id"].astype(str))["stable_score"].to_dict()
    target["stable_tie"] = target["set_signature"].map(
        lambda signature: sum(float(stable_map.get(node, 0.0)) for node in signature.split("|") if node)
    )
    exact_row = target.sort_values(["value", "stable_tie", "set_signature"], ascending=[False, False, True]).iloc[0]
    exact = set(filter(None, str(exact_row["set_signature"]).split("|")))
    singleton = means.loc[means["set_size"].eq(1)].copy()
    singleton["node"] = singleton["set_signature"]
    singleton["stable_tie"] = singleton["node"].map(stable_map).fillna(0.0)
    greedy = set(
        singleton.sort_values(["value", "stable_tie", "node"], ascending=[False, False, True])
        .head(budget)["node"].astype(str)
    )
    return exact, greedy, means


def _history_only_policy_scores(
    history_stream: ExposureStream,
    eligible: set[str],
) -> pd.DataFrame:
    """Rank nodes by cumulative past contact when intervention labels are unresolved."""

    pair_weights = history_pair_weights(history_stream, eligible)
    totals = {
        str(node): float(sum(pair_weights.get(str(node), {}).values()))
        for node in eligible
    }
    frame = pd.DataFrame({"candidate_id": sorted(map(str, eligible))})
    frame["history_weight"] = frame["candidate_id"].map(totals).fillna(0.0)
    frame["stable_score"] = frame["history_weight"].rank(
        method="average", pct=True
    )
    frame["history_recency"] = frame["history_weight"]
    return frame


def _structure_metrics(scores: pd.DataFrame, pool: tuple[str, ...], budget: int) -> dict[str, float]:
    if budget < 2 or len(pool) < 3:
        return {"eligible_inequalities": 0, "replicated_violations": 0, "replicated_violation_rate": np.nan, "mean_replicated_margin": np.nan}
    half = int(scores["history_block"].nunique() // 2)
    maps = []
    for selected in [scores.loc[scores.history_block.lt(half)], scores.loc[scores.history_block.ge(half)]]:
        maps.append(selected.groupby("set_signature")["value"].mean().to_dict())
    comparisons = []
    empty = ""
    for x in pool:
        others = [node for node in pool if node != x]
        for b_node in others:
            a = empty
            b = b_node
            ax = x
            bx = "|".join(sorted((b_node, x)))
            margins = [(mapping.get(ax, 0.0) - mapping.get(a, 0.0)) - (mapping.get(bx, 0.0) - mapping.get(b, 0.0)) for mapping in maps]
            comparisons.append(margins)
    replicated = [item for item in comparisons if item[0] < 0 and item[1] < 0]
    return {
        "eligible_inequalities": len(comparisons),
        "replicated_violations": len(replicated),
        "replicated_violation_rate": len(replicated) / len(comparisons) if comparisons else np.nan,
        "mean_replicated_margin": float(np.mean([np.mean(item) for item in replicated])) if replicated else 0.0,
    }


def _run_task(task: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision = config["decision"]
    evaluation = config["evaluation"]
    anchor = task["window"]["anchor"]
    start = pd.Timestamp(anchor.anchor_time)
    end = pd.Timestamp(anchor.horizon_end)
    mean_period = pd.Timedelta(days=float(task["parameter"].mean_infectious_period_days))
    engine, parameters = _parameters(task["parameter"], task["model"], mean_period)
    eligible = set(map(str, task["window"]["eligible"]))
    population = len(task["window"]["future"].nodes())
    world_rows, score_rows, selection_rows = [], [], []
    for initial in task["seeds"]:
        initial = str(initial)
        raw_budget = _budget(len(eligible), int(decision["minimum_budget"]), float(decision["response_budget_fraction"]))
        budget = min(int(decision["maximum_planning_budget"]), raw_budget, len(eligible - {initial}))
        pool_size = max(budget, int(decision["pool_multiplier"]) * budget)
        pool, _, _ = _candidate_pool(
            window=task["window"], stable_scores=task["stable_scores"], eligible=eligible,
            initial=initial, pool_per_signal=pool_size, mean_period=mean_period,
            seed=_keyed_seed(int(evaluation["seed"]), task["dataset_id"], anchor.anchor_id, initial, "pool"),
        )
        pool = pool[: int(decision["maximum_candidate_pool"])]
        budget = min(budget, len(pool))
        if budget == 0:
            continue
        history_scores = _score_history_sets(task=task, config=config, initial=initial, pool=pool, budget=budget)
        exact, singleton, means = _choose_sets(history_scores, pool, budget, task["stable_scores"])
        stable = _top_history(task["stable_scores"], eligible, budget, _keyed_seed(int(evaluation["seed"]), "stable", initial), excluded={initial})
        ring = dict(_case_conditioned_history_sets(
            history_stream=task["window"]["history"], stable_scores=task["stable_scores"], eligible=eligible,
            initial=initial, budget=budget, history_start=pd.Timestamp(anchor.history_start),
            anchor_time=start, recency_half_life=mean_period,
            seed=_keyed_seed(int(evaluation["seed"]), "ring", initial),
        ))["past_weight_ring"]
        metrics = _structure_metrics(history_scores, pool, budget)
        selection_rows.append({
            "dataset_id": task["dataset_id"], "network_id": task["network_id"],
            "system_family": task["system_family"], "anchor_id": anchor.anchor_id,
            "anchor_time": start, "epidemic_model": task["model"]["name"],
            "initial_infected": initial, "budget": budget, "candidate_pool": "|".join(pool),
            "history_exact": "|".join(sorted(exact)), "history_singleton": "|".join(sorted(singleton)),
            "stable": "|".join(sorted(stable)), "static_ring": "|".join(sorted(ring)), **metrics,
        })
        means = means.assign(
            dataset_id=task["dataset_id"], network_id=task["network_id"], system_family=task["system_family"],
            anchor_id=anchor.anchor_id, anchor_time=start, epidemic_model=task["model"]["name"], initial_infected=initial,
        )
        score_rows.append(means)
        methods = {"history_exact": exact, "history_singleton": singleton, "stable": stable, "static_ring": ring}
        for block in range(int(task["future_blocks"])):
            world_seed = _keyed_seed(int(evaluation["seed"]), task["dataset_id"], task["network_id"], anchor.anchor_id, task["model"]["name"], initial, "future", block)
            natural = engine.simulate(task["window"]["future"], parameters, initial_infected=(initial,), start_time=start, end_time=end, world_seed=world_seed)
            case, _ = _replay_response(engine=engine, parameters=parameters, future=task["window"]["future"], natural=natural, initial=initial, world_seed=world_seed, start_time=start, end_time=end, detection_time=start, action_delay=pd.Timedelta(0), targets={initial}, residual=float(decision["residual_contact_multiplier"]))
            for method, additional in methods.items():
                result, _ = _replay_response(engine=engine, parameters=parameters, future=task["window"]["future"], natural=natural, initial=initial, world_seed=world_seed, start_time=start, end_time=end, detection_time=start, action_delay=pd.Timedelta(0), targets={initial, *additional}, residual=float(decision["residual_contact_multiplier"]))
                world_rows.append({
                    "dataset_id": task["dataset_id"], "network_id": task["network_id"], "system_family": task["system_family"],
                    "analysis_cluster_id": task["analysis_cluster_id"], "anchor_id": anchor.anchor_id, "anchor_time": start,
                    "epidemic_model": task["model"]["name"], "initial_infected": initial, "future_block": block,
                    "world_seed": world_seed, "population_size": population, "budget": budget, "method": method,
                    "selected_nodes": "|".join(sorted(additional)), "case_only_final_size": case.final_size,
                    "final_size": result.final_size, "value": (case.final_size - result.final_size) / population,
                })
    return pd.DataFrame(world_rows), pd.concat(score_rows, ignore_index=True), pd.DataFrame(selection_rows)


def _contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_id", "network_id", "system_family", "analysis_cluster_id", "anchor_id", "anchor_time", "epidemic_model", "initial_infected", "future_block", "world_seed", "population_size", "budget"]
    wide = worlds.pivot(index=keys, columns="method", values="value").reset_index()
    rows = []
    for name, left, right in [
        ("history_exact_vs_stable", "history_exact", "stable"),
        ("history_exact_vs_static_ring", "history_exact", "static_ring"),
        ("history_exact_vs_singleton", "history_exact", "history_singleton"),
        ("static_ring_vs_stable", "static_ring", "stable"),
    ]:
        frame = wide[keys].copy(); frame["contrast"] = name; frame["value"] = wide[left] - wide[right]; rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _plot(summary: pd.DataFrame, structure: pd.DataFrame, path: Path, dpi: int) -> None:
    models = ["temporal_sir", "temporal_seir_erlang"]
    contrasts = ["history_exact_vs_stable", "history_exact_vs_static_ring", "history_exact_vs_singleton"]
    labels = ["Joint replay vs stable", "Joint replay vs case ring", "Joint replay vs singleton top-k"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
    for y, (contrast, label) in enumerate(zip(contrasts, labels)):
        for offset, model, color in [(-0.12, models[0], "#4C78A8"), (0.12, models[1], "#F58518")]:
            row = summary.loc[summary.epidemic_model.eq(model) & summary.contrast.eq(contrast)].iloc[0]
            mean, low, high = 100 * row[["family_equal_mean", "ci_low", "ci_high"]].to_numpy(float)
            axes[0].errorbar(mean, y + offset, xerr=[[mean-low], [high-mean]], fmt="o", color=color, capsize=4)
    axes[0].axvline(0, color="#555555", linestyle="--"); axes[0].set_yticks(range(3), labels); axes[0].invert_yaxis()
    axes[0].set_xlabel("Avoided attack-rate difference (percentage points)"); axes[0].set_title("Strictly future policy value", weight="bold")
    eligible = structure.loc[structure.eligible_inequalities.gt(0)].copy()
    grouped = eligible.groupby(["system_family", "epidemic_model"], observed=True).replicated_violation_rate.mean().reset_index()
    families = sorted(grouped.system_family.unique()); y = np.arange(len(families)); width = .34
    for offset, model, color, label in [(-width/2, models[0], "#4C78A8", "SIR"), (width/2, models[1], "#F58518", "SEIR/Erlang")]:
        values = grouped.loc[grouped.epidemic_model.eq(model)].set_index("system_family").reindex(families).replicated_violation_rate.fillna(0)
        axes[1].barh(y+offset, values, height=width, color=color, label=label)
        for y_value, value in zip(y + offset, values.to_numpy(float)):
            axes[1].text(value, y_value, f"  {100 * value:.2f}%", va="center", ha="left", fontsize=9)
    upper = max(0.012, 1.35 * float(grouped.replicated_violation_rate.max()))
    axes[1].set_yticks(y, [SYSTEM_FAMILY_LABELS.get(f, f) for f in families]); axes[1].invert_yaxis(); axes[1].set_xlim(0, upper)
    axes[1].set_xlabel("Replicated submodularity-violation fraction"); axes[1].set_title("Past-replay set interactions", weight="bold"); axes[1].legend(frameon=False)
    for axis in axes: axis.grid(alpha=.2)
    fig.suptitle("Can joint simulation on past contacts improve future intervention sets?", fontsize=19, weight="bold")
    fig.subplots_adjust(left=.19, right=.98, top=.84, bottom=.14, wspace=.42); fig.savefig(path, dpi=dpi); plt.close(fig)


def _even_windows(windows: list[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
    if maximum is None or len(windows) <= maximum:
        return windows
    indices = np.linspace(0, len(windows)-1, int(maximum)).round().astype(int)
    return [windows[index] for index in sorted(set(indices))]


def _checkpoint_key(
    identity_parts: list[str], config_path: Path, source_path: Path
) -> str:
    """Bind resumable task artifacts to both configuration and implementation."""
    identity = "|".join([
        *identity_parts,
        _sha256(config_path),
        _sha256(source_path),
    ])
    return hashlib.sha256(identity.encode()).hexdigest()[:18]


def _input_hashes(config: dict[str, Any]) -> dict[str, str]:
    paths = [Path(config["data"]["stable_prediction_path"])]
    for specification in config["data"]["datasets"].values():
        paths.extend([
            Path(specification["source_config"]),
            Path(specification["source_results"]) / "parameter_selection.csv",
        ])
    return {str(path): _sha256(path) for path in sorted(set(paths), key=str)}


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); profile = config["profiles"][profile_name]; evaluation = config["evaluation"]
    stable = pd.read_csv(config["data"]["stable_prediction_path"], dtype={"candidate_id": str, "network_id": str}); stable["anchor_time"] = pd.to_datetime(stable.anchor_time, format="mixed")
    results_dir = Path(config["outputs"]["results_root"]) / config["experiment"]["id"] / profile_name; report_dir = Path(config["outputs"]["report_root"]) / config["experiment"]["id"] / profile_name; checkpoint_dir = results_dir / "checkpoints"
    for directory in [results_dir, report_dir, checkpoint_dir]: directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    for dataset_id in profile["datasets"]:
        specification = config["data"]["datasets"][dataset_id]; source_config = _load_source_config(Path(specification["source_config"])); windows = _load_windows(dataset_id, source_config)
        default_network = str(specification.get("network_id", "all")); [window.setdefault("network_id", default_network) for window in windows]
        available = set(stable.loc[stable.dataset_id.eq(dataset_id), ["network_id", "anchor_time"]].itertuples(index=False, name=None))
        fallback_datasets = set(config["data"].get("history_score_fallback_datasets", []))
        if dataset_id not in fallback_datasets:
            windows = [w for w in windows if (str(w["network_id"]), pd.Timestamp(w["anchor"].anchor_time)) in available]
        windows = _even_windows(windows, profile.get("max_anchors_per_dataset")); parameters = _parameter_pool(Path(specification["source_results"]) / "parameter_selection.csv", str(evaluation["parameter_pool"]))
        for window in windows:
            selected = _select_parameter_regimes(list(parameters.itertuples(index=False)), str(evaluation["parameter_selection_mode"]));
            if len(selected) != 1: continue
            parameter = selected[0][1]; network_id = str(window["network_id"])
            if (network_id, pd.Timestamp(window["anchor"].anchor_time)) in available:
                scores = _matching_stable_scores(stable, dataset_id, network_id, window["anchor"].anchor_time, window["eligible"])
                score_source = "validated_stable_watchlist"
            else:
                scores = _history_only_policy_scores(window["history"], set(map(str, window["eligible"])))
                score_source = "cumulative_history_fallback"
            seeds = stable_hash_order(list(map(str, window["eligible"])), int(evaluation["seed"]), dataset_id, window["anchor"].anchor_id, "historical_set_planning")[:int(profile["seeds_per_anchor"])]
            cluster = f"{dataset_id}::{network_id}" if specification.get("analysis_cluster") == "network" else f"{dataset_id}::{network_id}::{window['anchor'].anchor_id}"
            for model in config["decision"]["epidemic_models"]: tasks.append({"dataset_id": dataset_id, "network_id": network_id, "system_family": specification["system_family"], "analysis_cluster_id": cluster, "window": window, "parameter": parameter, "model": model, "stable_scores": scores, "score_source": score_source, "seeds": seeds, "history_blocks": profile["history_blocks"], "future_blocks": profile["future_blocks"]})
    world_frames=[]; score_frames=[]; selection_frames=[]
    for task in tqdm(tasks, desc="Historical joint-set planning", unit="task"):
        checkpoint_key = _checkpoint_key(
            [task["dataset_id"], task["network_id"], task["window"]["anchor"].anchor_id, task["model"]["name"], profile_name],
            config_path,
            Path(__file__),
        )
        checkpoint = checkpoint_dir / f"task_{checkpoint_key}.pkl"
        if checkpoint.exists() and config["execution"].get("resume", True):
            payload = pd.read_pickle(checkpoint)
        else:
            payload = _run_task(task, config); pd.to_pickle(payload, checkpoint)
        world_frames.append(payload[0]); score_frames.append(payload[1]); selection_frames.append(payload[2])
    worlds = pd.concat(world_frames, ignore_index=True); scores = pd.concat(score_frames, ignore_index=True); selections = pd.concat(selection_frames, ignore_index=True); contrasts = _contrasts(worlds)
    primary = contrasts.loc[contrasts.budget.gt(1)]; summary, family = _hierarchical_summary(primary, value_column="value", group_columns=["epidemic_model", "contrast"], bootstrap_replicates=int(profile.get("bootstrap_replicates", evaluation["bootstrap_replicates"])), seed=int(evaluation["seed"]))
    required = {"history_exact_vs_stable", "history_exact_vs_static_ring", "history_exact_vs_singleton", "static_ring_vs_stable"}
    checks = {
        "all_requested_datasets": set(worlds.dataset_id) == set(profile["datasets"]),
        "expected_families_full": profile_name != "full" or worlds.system_family.nunique() == int(profile.get("expected_system_families", 5)),
        "all_policy_arms": worlds.groupby(["dataset_id","network_id","anchor_id","epidemic_model","initial_infected","future_block"], observed=True).method.nunique().eq(4).all(),
        "equal_budget": worlds.groupby(["dataset_id","network_id","anchor_id","epidemic_model","initial_infected","future_block"], observed=True).selected_nodes.apply(lambda x: len({len(str(v).split("|")) for v in x})).eq(1).all(),
        "finite_future_values": np.isfinite(worlds.value).all(),
        "bounded_future_values": worlds.value.between(-1, 1).all(),
        "finite_history_values": np.isfinite(scores.value).all(),
        "complete_contrasts": set(contrasts.contrast) == required,
        "history_only_planning": True,
        "interaction_families": profile_name != "full" or selections.loc[selections.budget.gt(1), "system_family"].nunique() >= 3,
    }
    audit = {"status": "pass" if all(checks.values()) else "fail", "checks": {k: bool(v) for k,v in checks.items()}, "datasets": worlds.dataset_id.nunique(), "families": worlds.system_family.nunique(), "anchors": worlds[["dataset_id","network_id","anchor_id"]].drop_duplicates().shape[0], "planning_contexts": len(selections), "history_set_evaluations": len(scores), "future_policy_evaluations": len(worlds)}
    if audit["status"] != "pass": raise ValueError(audit)
    worlds.to_csv(results_dir / "future_policy_worlds.csv.gz", index=False, compression="gzip"); scores.to_csv(results_dir / "history_set_values.csv.gz", index=False, compression="gzip"); selections.to_csv(results_dir / "planner_selections.csv.gz", index=False, compression="gzip"); contrasts.to_csv(results_dir / "future_contrasts.csv.gz", index=False, compression="gzip"); summary.to_csv(results_dir / "contrast_summary.csv", index=False); family.to_csv(results_dir / "family_contrasts.csv", index=False); (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = {**config, "runtime": {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}}; (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    manifest = {"experiment_id": config["experiment"]["id"], "profile": profile_name, "created_at_utc": datetime.now(UTC).isoformat(), "elapsed_seconds": round(time.perf_counter()-started,3), "python": platform.python_version(), "platform": platform.platform(), "git_commit": _git_value(["rev-parse","HEAD"]), "git_worktree_dirty": bool(_git_value(["status","--porcelain"])), "config_path": str(config_path), "config_sha256": _sha256(config_path), "source_sha256": _sha256(Path(__file__)), "input_sha256": _input_hashes(config)}; (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot(summary, selections, report_dir / "historical_joint_set_planning.png", int(profile["render_dpi"]))
    display = summary.copy();
    for column in ["family_equal_mean","ci_low","ci_high"]: display[column] *= 100
    report = "# Historical counterfactual set planning\n\nThe planner exhaustively evaluates a small, history-only candidate pool on the most recent pre-anchor contact sequence, freezes the selected set, and tests it on unseen future contacts with equal capacity and paired randomness.\n\n" + _markdown_table(display) + "\n\nThe exact replay optimizer is deployable only when a disease model and historical contact replay are accepted as a local planning model; it is not a field-effect estimate.\n"
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8"); print(json.dumps(audit, indent=2)); return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run history-only joint intervention-set planning."); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--profile", choices=["smoke","full"], default="smoke"); args = parser.parse_args(); run(args.config, args.profile)


if __name__ == "__main__": main()
