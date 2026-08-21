from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from animal_intervention.estimands.intervention_value import _candidate_action
from animal_intervention.evaluation import stable_hash_order
from animal_intervention.simulation import PairedTemporalSIREngine, SIRParameters

from .oxford_predefense import _keyed_seed


WORLD_COLUMNS = [
    "task_id",
    "anchor_id",
    "parameter_id",
    "block_id",
    "candidate_id",
    "introduction_stratum",
    "introduction_position",
    "introduction_replicate",
    "initial_infected",
    "world_seed",
    "population_size",
    "baseline_final_size",
    "intervention_final_size",
    "avoided_infections",
]


def _task_id(
    anchor_id: str,
    parameter_id: str,
    block_id: int,
    *,
    seed: int,
    non_index_cases: int,
    self_replicates: int,
    candidates: list[str],
) -> str:
    readable = anchor_id.replace("::", "_").replace(":", "_")
    digest = hashlib.sha256(
        "|".join(
            [
                anchor_id,
                parameter_id,
                str(block_id),
                str(seed),
                str(non_index_cases),
                str(self_replicates),
                *candidates,
            ]
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"{readable}__{parameter_id}__block_{block_id:03d}__{digest}"


def _simulate_stability_task(task: dict[str, Any]) -> pd.DataFrame:
    window = task["window"]
    anchor = window["anchor"]
    eligible = list(window["eligible"])
    candidates = list(task["candidates"])
    non_index_cases = min(int(task["non_index_cases"]), len(eligible) - 1)
    block_id = int(task["block_id"])
    seed = int(task["seed"])
    parameter = task["parameter"]
    parameter_id = str(parameter["parameter_id"])
    sir = SIRParameters(
        beta=float(parameter["beta"]),
        recovery_rate=float(parameter["recovery_rate_per_day"]) / 86400.0,
    )
    introduction_order = stable_hash_order(
        eligible, seed, anchor.anchor_id, "stability_indices", block_id
    )
    initials_by_candidate = {
        candidate: [node for node in introduction_order if node != candidate][
            :non_index_cases
        ]
        for candidate in candidates
    }
    required_initials = sorted(
        {node for selected in initials_by_candidate.values() for node in selected}
    )
    engine = PairedTemporalSIREngine()
    baseline_by_initial = {}
    for initial in required_initials:
        world_seed = _keyed_seed(
            seed, anchor.anchor_id, "non_index", initial, block_id
        )
        baseline_by_initial[initial] = engine.simulate(
            window["future"],
            sir,
            initial_infected=[initial],
            start_time=anchor.anchor_time,
            end_time=anchor.horizon_end,
            world_seed=world_seed,
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        action = _candidate_action(candidate, anchor, task["action_config"])
        for position, initial in enumerate(initials_by_candidate[candidate]):
            world_seed = _keyed_seed(
                seed, anchor.anchor_id, "non_index", initial, block_id
            )
            baseline = baseline_by_initial[initial]
            intervention = engine.simulate(
                window["future"],
                sir,
                initial_infected=[initial],
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
                action=action,
            )
            rows.append(
                {
                    "task_id": task["task_id"],
                    "anchor_id": anchor.anchor_id,
                    "parameter_id": parameter_id,
                    "block_id": block_id,
                    "candidate_id": candidate,
                    "introduction_stratum": "non_index",
                    "introduction_position": position,
                    "introduction_replicate": block_id,
                    "initial_infected": initial,
                    "world_seed": world_seed,
                    "population_size": window["population_size"],
                    "baseline_final_size": baseline.final_size,
                    "intervention_final_size": intervention.final_size,
                    "avoided_infections": baseline.final_size
                    - intervention.final_size,
                }
            )
        for replicate in range(int(task["self_replicates"])):
            world_seed = _keyed_seed(
                seed,
                anchor.anchor_id,
                "self_index",
                candidate,
                block_id,
                replicate,
            )
            baseline = engine.simulate(
                window["future"],
                sir,
                initial_infected=[candidate],
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
            )
            intervention = engine.simulate(
                window["future"],
                sir,
                initial_infected=[candidate],
                start_time=anchor.anchor_time,
                end_time=anchor.horizon_end,
                world_seed=world_seed,
                action=action,
            )
            rows.append(
                {
                    "task_id": task["task_id"],
                    "anchor_id": anchor.anchor_id,
                    "parameter_id": parameter_id,
                    "block_id": block_id,
                    "candidate_id": candidate,
                    "introduction_stratum": "self_index",
                    "introduction_position": -1,
                    "introduction_replicate": replicate,
                    "initial_infected": candidate,
                    "world_seed": world_seed,
                    "population_size": window["population_size"],
                    "baseline_final_size": baseline.final_size,
                    "intervention_final_size": intervention.final_size,
                    "avoided_infections": baseline.final_size
                    - intervention.final_size,
                }
            )
    return pd.DataFrame(rows, columns=WORLD_COLUMNS)


def summarize_stability_worlds(worlds: pd.DataFrame) -> pd.DataFrame:
    required = set(WORLD_COLUMNS).difference({"task_id", "introduction_position"})
    missing = required.difference(worlds.columns)
    if missing:
        raise ValueError(f"stability worlds are missing columns: {sorted(missing)}")
    frame = worlds.copy()
    frame["avoided_attack_rate"] = (
        frame["avoided_infections"] / frame["population_size"]
    )
    rows: list[dict[str, Any]] = []
    keys = ["anchor_id", "parameter_id", "block_id", "candidate_id"]
    for key, group in frame.groupby(keys, observed=True, sort=False):
        anchor_id, parameter_id, block_id, candidate_id = key
        self_values = group.loc[
            group["introduction_stratum"].eq("self_index"), "avoided_attack_rate"
        ]
        non_values = group.loc[
            group["introduction_stratum"].eq("non_index"), "avoided_attack_rate"
        ]
        if self_values.empty or non_values.empty:
            raise ValueError("every candidate block requires self and non-index worlds")
        eligible_population = int(group["population_size"].iloc[0])
        known_value = float(self_values.mean())
        non_index_value = float(non_values.mean())
        rows.append(
            {
                "anchor_id": anchor_id,
                "parameter_id": parameter_id,
                "block_id": block_id,
                "candidate_id": candidate_id,
                "eligible_population": eligible_population,
                "outcome_population": eligible_population,
                "self_index_worlds": len(self_values),
                "non_index_worlds": len(non_values),
                "known_index_value": known_value,
                "non_index_value": non_index_value,
                "unconditional_value": known_value / eligible_population
                + non_index_value
                * (eligible_population - 1)
                / eligible_population,
            }
        )
    estimates = pd.DataFrame(rows)
    estimates["rank"] = estimates.groupby(
        ["anchor_id", "parameter_id", "block_id"], observed=True
    )["unconditional_value"].rank(method="average", ascending=False)
    return estimates


def run_checkpointed_stability(
    prepared: list[dict[str, Any]],
    parameters: pd.DataFrame,
    action_config: dict[str, Any],
    *,
    random_blocks: int,
    non_index_cases: int,
    self_replicates: int,
    candidate_limit: int | None,
    seed: int,
    checkpoint_dir: Path,
    max_workers: int,
    resume: bool = True,
    progress_label: str = "Checkpointed stability simulations",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if random_blocks < 2:
        raise ValueError("random_blocks must be at least two")
    if non_index_cases < 1 or self_replicates < 1:
        raise ValueError("world counts must be positive")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for window in prepared:
        candidates = list(window["eligible"])
        if candidate_limit is not None:
            candidates = sorted(
                candidates,
                key=lambda node: (
                    -int(window["history_support"].get(node, 0)),
                    node,
                ),
            )[:candidate_limit]
        for parameter in parameters.to_dict(orient="records"):
            for block_id in range(random_blocks):
                identifier = _task_id(
                    window["anchor"].anchor_id,
                    str(parameter["parameter_id"]),
                    block_id,
                    seed=seed,
                    non_index_cases=non_index_cases,
                    self_replicates=self_replicates,
                    candidates=candidates,
                )
                tasks.append(
                    {
                        "task_id": identifier,
                        "window": window,
                        "parameter": parameter,
                        "block_id": block_id,
                        "candidates": candidates,
                        "non_index_cases": non_index_cases,
                        "self_replicates": self_replicates,
                        "action_config": action_config,
                        "seed": seed,
                        "expected_rows": len(candidates)
                        * (min(non_index_cases, len(window["eligible"]) - 1) + self_replicates),
                    }
                )

    completed: list[pd.DataFrame] = []
    pending: list[dict[str, Any]] = []
    total_rows = sum(int(task["expected_rows"]) for task in tasks)
    progress = tqdm(total=total_rows, desc=progress_label)
    for task in tasks:
        checkpoint = checkpoint_dir / f"{task['task_id']}.csv.gz"
        if resume and checkpoint.exists():
            frame = pd.read_csv(checkpoint, dtype={"candidate_id": str, "initial_infected": str})
            if len(frame) != int(task["expected_rows"]) or set(frame["task_id"]) != {
                task["task_id"]
            }:
                raise ValueError(f"Invalid stability checkpoint: {checkpoint}")
            completed.append(frame)
            progress.update(len(frame))
        else:
            pending.append(task)
    try:
        if max_workers == 1:
            for task in pending:
                frame = _simulate_stability_task(task)
                frame.to_csv(
                    checkpoint_dir / f"{task['task_id']}.csv.gz",
                    index=False,
                    compression="gzip",
                )
                completed.append(frame)
                progress.update(len(frame))
        elif pending:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_simulate_stability_task, task): task
                    for task in pending
                }
                for future in as_completed(futures):
                    task = futures[future]
                    frame = future.result()
                    frame.to_csv(
                        checkpoint_dir / f"{task['task_id']}.csv.gz",
                        index=False,
                        compression="gzip",
                    )
                    completed.append(frame)
                    progress.update(len(frame))
    finally:
        progress.close()
    if len(completed) != len(tasks):
        raise RuntimeError("not all stability tasks completed")
    worlds = pd.concat(completed, ignore_index=True)
    worlds = worlds.sort_values(
        [
            "anchor_id",
            "parameter_id",
            "block_id",
            "candidate_id",
            "introduction_stratum",
            "introduction_position",
            "introduction_replicate",
        ],
        kind="stable",
        ignore_index=True,
    )
    worlds["avoided_attack_rate"] = (
        worlds["avoided_infections"] / worlds["population_size"]
    )
    return worlds, summarize_stability_worlds(worlds)
