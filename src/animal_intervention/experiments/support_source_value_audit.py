from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from animal_intervention.evaluation.baseline_ranking import CONTEXT_COLUMNS, animal_system_family


KEY_COLUMNS = [*CONTEXT_COLUMNS, "candidate_id"]
SUPPORT_COLUMNS = [
    "event_rate_per_day",
    "contact_opportunity_rate",
    "eligible_partner_fraction",
    "observed_partner_count",
    "recency_score",
    "recent_activity_fraction",
    "active_span_fraction",
    "first_seen_fraction",
    "last_seen_gap_fraction",
    "observed_span_fraction",
    "roster_coverage_fraction",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _source_values(path: Path, dataset_id: str) -> pd.DataFrame:
    columns = [
        "anchor_id",
        "candidate_id",
        "initial_infected",
        "introduction_stratum",
        "parameter_id",
        "block_id",
        "world_seed",
        "population_size",
        "baseline_final_size",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda name: name in columns, chunksize=250_000):
        same = chunk["candidate_id"].astype(str).eq(chunk["initial_infected"].astype(str))
        self_index = chunk["introduction_stratum"].astype(str).eq("self_index")
        selected = chunk.loc[same & self_index].copy()
        if selected.empty:
            continue
        selected["source_attack_rate"] = (
            selected["baseline_final_size"] / selected["population_size"]
        )
        parts.append(selected)
    if not parts:
        raise ValueError(f"No self-index source worlds found for {dataset_id}")
    worlds = pd.concat(parts, ignore_index=True)
    deduplication_key = [
        "anchor_id",
        "candidate_id",
        "parameter_id",
        "block_id",
        "world_seed",
    ]
    duplicate_worlds = int(worlds.duplicated(deduplication_key).sum())
    if duplicate_worlds:
        raise ValueError(f"Duplicate source worlds for {dataset_id}: {duplicate_worlds}")
    summary = (
        worlds.groupby(["anchor_id", "candidate_id"], observed=True, sort=True)
        .agg(
            source_attack_rate=("source_attack_rate", "mean"),
            source_attack_rate_sd=("source_attack_rate", "std"),
            source_worlds=("world_seed", "nunique"),
        )
        .reset_index()
    )
    summary["dataset_id"] = dataset_id
    summary["candidate_id"] = summary["candidate_id"].astype(str)
    return summary


def _merge_intervals_fraction(
    starts: pd.Series,
    ends: pd.Series,
    history_start: pd.Timestamp,
    anchor_time: pd.Timestamp,
) -> float:
    intervals = sorted(
        (
            max(pd.Timestamp(start), history_start),
            min(pd.Timestamp(end), anchor_time),
        )
        for start, end in zip(starts, ends, strict=True)
        if pd.Timestamp(end) > history_start and pd.Timestamp(start) < anchor_time
    )
    if not intervals:
        return 0.0
    total = pd.Timedelta(0)
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    duration = anchor_time - history_start
    return float(total / duration) if duration > pd.Timedelta(0) else np.nan


def _history_timing_support(dataset_id: str, labels: pd.DataFrame, root: Path) -> pd.DataFrame:
    processed = root / dataset_id / "processed"
    dyadic = pd.read_parquet(
        processed / "dyadic_events.parquet",
        columns=["source_id", "target_id", "start_time", "end_time"],
    )
    if len(dyadic):
        source = dyadic[["source_id", "start_time", "end_time"]].rename(
            columns={"source_id": "candidate_id"}
        )
        target = dyadic[["target_id", "start_time", "end_time"]].rename(
            columns={"target_id": "candidate_id"}
        )
        observations = pd.concat([source, target], ignore_index=True)
    else:
        events = pd.read_parquet(
            processed / "group_events.parquet",
            columns=["group_event_id", "start_time", "end_time"],
        )
        memberships = pd.read_parquet(
            processed / "group_memberships.parquet",
            columns=["group_event_id", "node_id"],
        )
        observations = memberships.merge(
            events,
            on="group_event_id",
            how="inner",
            validate="many_to_one",
        ).rename(columns={"node_id": "candidate_id"})
        observations = observations[["candidate_id", "start_time", "end_time"]]
    observations["candidate_id"] = observations["candidate_id"].astype(str)
    observations["start_time"] = pd.to_datetime(observations["start_time"], utc=True)
    observations["end_time"] = pd.to_datetime(observations["end_time"], utc=True)
    roster = pd.read_parquet(
        processed / "observation_windows.parquet",
        columns=["node_id", "window_start", "window_end"],
    ).rename(columns={"node_id": "candidate_id"})
    roster["candidate_id"] = roster["candidate_id"].astype(str)
    roster["window_start"] = pd.to_datetime(roster["window_start"], utc=True)
    roster["window_end"] = pd.to_datetime(roster["window_end"], utc=True)

    rows: list[dict[str, Any]] = []
    contexts = labels[
        [*CONTEXT_COLUMNS, "history_start", "candidate_id"]
    ].drop_duplicates()
    contexts["history_start"] = pd.to_datetime(contexts["history_start"], format="mixed", utc=True)
    for context, candidates in contexts.groupby(CONTEXT_COLUMNS, observed=True, sort=True):
        history_start = candidates["history_start"].iloc[0]
        anchor_time = pd.Timestamp(context[2])
        duration = anchor_time - history_start
        window_observations = observations.loc[
            observations["start_time"].lt(anchor_time)
            & observations["end_time"].gt(history_start)
        ].copy()
        window_observations["effective_start"] = window_observations["start_time"].clip(
            lower=history_start
        )
        window_observations["effective_end"] = window_observations["end_time"].clip(
            upper=anchor_time
        )
        timing = window_observations.groupby("candidate_id", observed=True).agg(
            first_seen=("effective_start", "min"),
            last_seen=("effective_end", "max"),
        )
        for candidate in candidates["candidate_id"].astype(str):
            if candidate not in timing.index:
                first_fraction = np.nan
                last_gap = np.nan
                observed_span = np.nan
            else:
                first_seen = timing.loc[candidate, "first_seen"]
                last_seen = timing.loc[candidate, "last_seen"]
                first_fraction = float((first_seen - history_start) / duration)
                last_gap = float((anchor_time - last_seen) / duration)
                observed_span = float((last_seen - first_seen) / duration)
            candidate_roster = roster.loc[roster["candidate_id"].eq(candidate)]
            coverage = _merge_intervals_fraction(
                candidate_roster["window_start"],
                candidate_roster["window_end"],
                history_start,
                anchor_time,
            )
            rows.append(
                {
                    **dict(zip(CONTEXT_COLUMNS, context, strict=True)),
                    "candidate_id": candidate,
                    "first_seen_fraction": first_fraction,
                    "last_seen_gap_fraction": last_gap,
                    "observed_span_fraction": observed_span,
                    "roster_coverage_fraction": coverage,
                }
            )
    return pd.DataFrame(rows)


def _context_correlations(frame: pd.DataFrame, minimum_candidates: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["source_attack_rate", *SUPPORT_COLUMNS]
    for context, group in frame.groupby(CONTEXT_COLUMNS, observed=True, sort=True):
        row: dict[str, Any] = dict(zip(CONTEXT_COLUMNS, context, strict=True))
        row["system_family"] = animal_system_family(str(context[0]))
        row["candidate_count"] = len(group)
        for metric in metrics:
            pair = group[[metric, "robust_intervention_value"]].dropna()
            usable = (
                len(pair) >= minimum_candidates
                and pair[metric].nunique() > 1
                and pair["robust_intervention_value"].nunique() > 1
            )
            row[f"spearman_{metric}"] = (
                float(pair.corr(method="spearman").iloc[0, 1]) if usable else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _family_summary(contexts: pd.DataFrame) -> pd.DataFrame:
    correlation_columns = [column for column in contexts if column.startswith("spearman_")]
    rows: list[dict[str, Any]] = []
    for family, group in contexts.groupby("system_family", observed=True, sort=True):
        row: dict[str, Any] = {
            "system_family": family,
            "contexts": len(group),
        }
        for column in correlation_columns:
            values = group[column].dropna()
            row[f"mean_{column}"] = float(values.mean()) if len(values) else np.nan
            row[f"median_{column}"] = float(values.median()) if len(values) else np.nan
            row[f"estimable_{column}"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _blocked_interval(
    contexts: pd.DataFrame,
    column: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    eligible = contexts.dropna(subset=[column]).copy()
    family_contexts = {
        family: group[column].to_numpy(float)
        for family, group in eligible.groupby("system_family", observed=True)
    }
    point = float(np.mean([values.mean() for values in family_contexts.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    families = sorted(family_contexts)
    for index in range(replicates):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        family_means = []
        for family in sampled_families:
            values = family_contexts[str(family)]
            sampled = rng.choice(values, size=len(values), replace=True)
            family_means.append(float(sampled.mean()))
        draws[index] = float(np.mean(family_means))
    return {
        "family_equal_mean": point,
        "blocked_ci_low": float(np.quantile(draws, 0.025)),
        "blocked_ci_high": float(np.quantile(draws, 0.975)),
        "families": len(families),
        "contexts": len(eligible),
    }


def _plot_source_vs_intervention(frame: pd.DataFrame, path: Path, dpi: int) -> None:
    families = sorted(frame["system_family"].unique())
    palette = dict(zip(families, plt.cm.tab10(np.linspace(0, 1, len(families))), strict=True))
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for family in families:
        selected = frame.loc[frame["system_family"].eq(family)]
        axis.scatter(
            selected["source_attack_rate"],
            selected["robust_intervention_value"],
            s=15,
            alpha=0.45,
            color=palette[family],
            label=family.replace("_", " "),
        )
    axis.set_title("Index-case source impact and prospective isolation value are distinct")
    axis.set_xlabel("Mean attack rate when the animal is the index case")
    axis.set_ylabel("Mean attack rate avoided by prospective singleton isolation")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _plot_support_correlations(family_summary: pd.DataFrame, path: Path, dpi: int) -> None:
    metrics = ["source_attack_rate", *SUPPORT_COLUMNS]
    columns = [f"mean_spearman_{metric}" for metric in metrics]
    matrix = family_summary.set_index("system_family")[columns].rename(
        columns=lambda value: value.removeprefix("mean_spearman_").replace("_", " ")
    )
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(
        np.arange(len(matrix.index)),
        [value.replace("_", " ") for value in matrix.index],
    )
    axis.set_title("Within-window rank association with intervention value")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            axis.text(column, row, label, ha="center", va="center", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Mean within-context Spearman correlation")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run(config_path: Path, profile: str) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile_config = config["profiles"][profile]
    experiment_id = config["experiment"]["id"]
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    labels_path = Path(config["inputs"]["labels"])
    features_path = Path(config["inputs"]["history_features"])
    labels = pd.read_csv(labels_path, dtype={"candidate_id": str, "network_id": str})
    features = pd.read_csv(features_path, dtype={"candidate_id": str, "network_id": str})
    labels["anchor_time"] = pd.to_datetime(labels["anchor_time"], format="mixed", utc=True)
    features["anchor_time"] = pd.to_datetime(
        features["anchor_time"], format="mixed", utc=True
    )
    selected_datasets = set(profile_config["datasets"])
    labels = labels.loc[labels["dataset_id"].isin(selected_datasets)].copy()
    features = features.loc[features["dataset_id"].isin(selected_datasets)].copy()
    max_contexts = profile_config.get("max_contexts_per_dataset")
    if max_contexts is not None:
        keep = (
            labels[CONTEXT_COLUMNS]
            .drop_duplicates()
            .sort_values(CONTEXT_COLUMNS)
            .groupby("dataset_id", observed=True)
            .head(int(max_contexts))
        )
        labels = labels.merge(keep, on=CONTEXT_COLUMNS, how="inner", validate="many_to_one")
    base_support_columns = [column for column in SUPPORT_COLUMNS if column in features.columns]
    merged = labels.merge(
        features[[*KEY_COLUMNS, *base_support_columns]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    timing_parts = [
        _history_timing_support(
            dataset_id,
            labels.loc[labels["dataset_id"].eq(dataset_id)].copy(),
            Path(config["inputs"]["canonical_root"]),
        )
        for dataset_id in sorted(selected_datasets)
    ]
    timing = pd.concat(timing_parts, ignore_index=True)
    merged = merged.merge(timing, on=KEY_COLUMNS, how="left", validate="one_to_one")

    source_parts = []
    for dataset_id in sorted(selected_datasets):
        source = _source_values(Path(config["inputs"]["source_worlds"][dataset_id]), dataset_id)
        anchor_map = (
            labels.loc[labels["dataset_id"].eq(dataset_id), ["anchor_id", "anchor_time", "network_id"]]
            .drop_duplicates()
        )
        source = source.merge(anchor_map, on="anchor_id", how="inner", validate="many_to_one")
        source_parts.append(source)
    sources = pd.concat(source_parts, ignore_index=True)
    sources["anchor_time"] = pd.to_datetime(sources["anchor_time"], utc=True)
    merged = merged.merge(
        sources[[*KEY_COLUMNS, "source_attack_rate", "source_attack_rate_sd", "source_worlds"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    merged["system_family"] = merged["dataset_id"].map(animal_system_family)
    contexts = _context_correlations(
        merged,
        int(config["design"]["minimum_candidates_for_context_correlation"]),
    )
    family = _family_summary(contexts)
    replicates = int(profile_config["bootstrap_replicates"])
    source_interval = _blocked_interval(
        contexts,
        "spearman_source_attack_rate",
        replicates,
        int(config["design"]["seed"]),
    )
    overall_rows = []
    for offset, metric in enumerate(["source_attack_rate", *SUPPORT_COLUMNS]):
        column = f"spearman_{metric}"
        if contexts[column].notna().sum() == 0:
            continue
        interval = _blocked_interval(
            contexts,
            column,
            replicates,
            int(config["design"]["seed"]) + offset,
        )
        overall_rows.append({"metric": metric, **interval})
    overall = pd.DataFrame(overall_rows)

    duplicate_labels = int(merged.duplicated(KEY_COLUMNS).sum())
    missing_support = int(merged[SUPPORT_COLUMNS].isna().any(axis=1).sum())
    missing_source = int(merged["source_attack_rate"].isna().sum())
    checks = {
        "label_keys_unique": duplicate_labels == 0,
        "history_support_complete": missing_support == 0,
        "source_value_complete": missing_source == 0,
        "source_value_bounded": bool(merged["source_attack_rate"].between(0, 1).all()),
        "intervention_value_finite": bool(np.isfinite(merged["robust_intervention_value"]).all()),
        "history_only_contract": bool(config["design"]["history_only_support"]),
        "all_requested_datasets_present": set(merged["dataset_id"]) == selected_datasets,
        "five_independent_families_full": profile != "full" or merged["system_family"].nunique() == 5,
        "source_and_intervention_not_aliased": not np.allclose(
            merged["source_attack_rate"], merged["robust_intervention_value"]
        ),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "datasets": int(merged["dataset_id"].nunique()),
        "independent_families": int(merged["system_family"].nunique()),
        "labels": len(merged),
        "contexts": len(contexts),
        "missing_support_rows": missing_support,
        "missing_source_rows": missing_source,
        "source_value_rank_association": source_interval,
        "interpretation": (
            "Source impact and intervention value answer different counterfactual questions. "
            "Support associations are diagnostics for observation-history dependence, not causal effects."
        ),
    }
    if audit["status"] != "pass":
        raise RuntimeError(json.dumps(audit, indent=2))

    serializable = merged.copy()
    serializable["anchor_time"] = serializable["anchor_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    context_output = contexts.copy()
    context_output["anchor_time"] = pd.to_datetime(context_output["anchor_time"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_csv(serializable, results_dir / "candidate_support_and_values.csv")
    _write_csv(context_output, results_dir / "context_correlations.csv")
    _write_csv(family, results_dir / "family_support_summary.csv")
    _write_csv(overall, results_dir / "overall_support_associations.csv")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "config_sha256": _sha256(config_path),
        "input_sha256": {
            "labels": _sha256(labels_path),
            "history_features": _sha256(features_path),
            **{
                dataset_id: _sha256(Path(path))
                for dataset_id, path in config["inputs"]["source_worlds"].items()
                if dataset_id in selected_datasets
            },
        },
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    dpi = int(profile_config["render_dpi"])
    _plot_source_vs_intervention(merged, report_dir / "source_vs_intervention.png", dpi)
    _plot_support_correlations(family, report_dir / "support_correlations.png", dpi)
    readme = f"""# Observation-support and source-value audit

This frozen-artifact diagnostic compares two different model-based quantities: attack rate when an animal is the simulated index case, and attack rate avoided when that animal is prospectively isolated. It also measures how intervention value is associated with history-only observation support within each prediction window.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['independent_families']}
- Candidate labels: {audit['labels']}
- Prediction contexts: {audit['contexts']}
- Audit: **{audit['status']}**
- Family-equal source/value rank association: {source_interval['family_equal_mean']:.3f} [{source_interval['blocked_ci_low']:.3f}, {source_interval['blocked_ci_high']:.3f}]

Associations with activity, recency, and coverage are expected because those variables define the available contact history. They do not by themselves demonstrate observation-start bias. The experiment uses no post-anchor feature and does not change any epidemic parameter. Source impact is not interchangeable with prospective intervention value.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return results_dir, report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit observation support and source-versus-intervention value")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260818-001_support_source_value_audit.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    results, reports = run(args.config, args.profile)
    print(f"Results: {results}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
