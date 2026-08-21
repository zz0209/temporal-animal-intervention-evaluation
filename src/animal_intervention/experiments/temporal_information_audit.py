from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import yaml

from animal_intervention.centrality import build_history_features, build_temporal_audit_tables
from animal_intervention.centrality.reference_baselines import TEMPORAL_AUDIT_METHODS
from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.evaluation.baseline_ranking import (
    CONTEXT_COLUMNS,
    animal_system_family,
    evaluate_baseline_scores,
    fit_baseline_scores,
    fit_feature_ablation_scores,
)
from animal_intervention.transmission import compile_named_exposure


STATIC_METHODS = (
    "static_degree",
    "static_strength",
    "static_pagerank",
    "static_eigenvector",
    "static_k_core",
)
KEYS = ["dataset_id", "network_id", "anchor_time", "candidate_id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in ("history_start", "anchor_time", "horizon_end"):
        if column in output:
            parsed = pd.to_datetime(output[column], format="mixed")
            if parsed.isna().any():
                raise ValueError(f"{column} contains invalid timestamps")
            output[column] = parsed.map(lambda value: value.isoformat())
    output.to_csv(path, index=False)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _select_labels(
    labels: pd.DataFrame,
    datasets: list[str],
    max_contexts: int | None,
) -> pd.DataFrame:
    selected = labels.loc[labels["dataset_id"].isin(datasets)].copy()
    if max_contexts is None:
        return selected
    outputs = []
    for _, frame in selected.groupby("dataset_id", sort=True, observed=True):
        contexts = frame[CONTEXT_COLUMNS].drop_duplicates().head(max_contexts)
        outputs.append(frame.merge(contexts, on=CONTEXT_COLUMNS, how="inner"))
    return pd.concat(outputs, ignore_index=True)


def _selected_betas(path: Path) -> tuple[float, ...]:
    selection = pd.read_csv(path)
    selected = selection.loc[selection["selected"].astype(bool), "beta"]
    betas = tuple(float(value) for value in selected)
    if not betas or any(beta <= 0 for beta in betas):
        raise ValueError(f"{path} has no valid selected beta values")
    return betas


def _dataset_fingerprint(
    dataset_id: str,
    dataset_labels: pd.DataFrame,
    *,
    mapper_name: str,
    betas: tuple[float, ...],
    attenuation: float,
    shuffle_replicates: int,
    random_seed: int,
    time_rules: tuple[str, ...],
) -> str:
    label_columns = [
        "label_id",
        "network_id",
        "candidate_id",
        "history_start",
        "anchor_time",
        "primary_mapper",
    ]
    normalized = dataset_labels[label_columns].astype(str).sort_values(label_columns)
    payload = {
        "dataset_id": dataset_id,
        "mapper_name": mapper_name,
        "betas": betas,
        "attenuation": attenuation,
        "shuffle_replicates": shuffle_replicates,
        "random_seed": random_seed,
        "time_rules": time_rules,
        "label_hash": hashlib.sha256(
            pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
        ).hexdigest(),
        "history_source_hash": _sha256(
            Path("src/animal_intervention/centrality/history_features.py")
        ),
        "reference_source_hash": _sha256(
            Path("src/animal_intervention/centrality/reference_baselines.py")
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _evaluate_shuffles(
    labels: pd.DataFrame,
    shuffled: pd.DataFrame,
    *,
    top_fraction: float,
) -> pd.DataFrame:
    target_columns = [
        *KEYS,
        "robust_intervention_value",
        "robust_priority_percentile",
    ]
    outputs: list[pd.DataFrame] = []
    for replicate, frame in shuffled.groupby("shuffle_replicate", sort=True):
        wide = frame.pivot(
            index=KEYS, columns="method", values="score"
        ).reset_index()
        wide.columns.name = None
        wide = wide.rename(
            columns={method: f"score_{method}" for method in TEMPORAL_AUDIT_METHODS}
        )
        evaluation_frame = labels[target_columns].merge(
            wide, on=KEYS, validate="one_to_one"
        )
        context_metrics, _ = evaluate_baseline_scores(
            evaluation_frame, top_fraction=top_fraction
        )
        context_metrics["shuffle_replicate"] = int(replicate)
        outputs.append(context_metrics)
    return pd.concat(outputs, ignore_index=True)


def _evaluate_sensitivity(
    labels: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    top_fraction: float,
) -> pd.DataFrame:
    target_columns = [
        *KEYS,
        "robust_intervention_value",
        "robust_priority_percentile",
    ]
    outputs: list[pd.DataFrame] = []
    for (time_rule, direction), frame in sensitivity.groupby(
        ["event_time_rule", "direction"], sort=True, observed=True
    ):
        score_name = f"score_exposure_communicability_{direction}"
        evaluation_frame = labels[target_columns].merge(
            frame[KEYS + ["score"]].rename(columns={"score": score_name}),
            on=KEYS,
            validate="one_to_one",
        )
        context_metrics, _ = evaluate_baseline_scores(
            evaluation_frame, top_fraction=top_fraction
        )
        context_metrics["event_time_rule"] = str(time_rule)
        context_metrics["direction"] = str(direction)
        outputs.append(context_metrics)
    return pd.concat(outputs, ignore_index=True)


def _paired_shuffle_differences(
    ordered_metrics: pd.DataFrame,
    shuffled_metrics: pd.DataFrame,
) -> pd.DataFrame:
    metrics = ["spearman", "value_capture_above_random", "oracle_regret"]
    ordered = ordered_metrics.loc[
        ordered_metrics["method"].isin(TEMPORAL_AUDIT_METHODS),
        [*CONTEXT_COLUMNS, "system_family", "method", *metrics],
    ]
    shuffled = shuffled_metrics[
        [
            *CONTEXT_COLUMNS,
            "system_family",
            "method",
            "shuffle_replicate",
            *metrics,
        ]
    ]
    merged = ordered.merge(
        shuffled,
        on=[*CONTEXT_COLUMNS, "system_family", "method"],
        suffixes=("_ordered", "_shuffled"),
        validate="one_to_many",
    )
    merged["spearman_gain"] = (
        merged["spearman_ordered"] - merged["spearman_shuffled"]
    )
    merged["value_capture_gain"] = (
        merged["value_capture_above_random_ordered"]
        - merged["value_capture_above_random_shuffled"]
    )
    merged["regret_reduction"] = (
        merged["oracle_regret_shuffled"] - merged["oracle_regret_ordered"]
    )
    return merged


def _family_equal_point(
    frame: pd.DataFrame, column: str
) -> tuple[float, float]:
    context_means = frame.groupby(
        [*CONTEXT_COLUMNS, "system_family"], observed=True
    )[column].mean().reset_index()
    unit_means = context_means.groupby(
        ["system_family", "dataset_id", "network_id"], observed=True
    )[column].mean().reset_index()
    family_means = unit_means.groupby("system_family", observed=True)[column].mean()
    return float(family_means.mean()), float(family_means.gt(0).mean())


def _blocked_shuffle_summary(
    differences: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    outcomes = ["spearman_gain", "value_capture_gain", "regret_reduction"]
    for method, method_frame in differences.groupby("method", observed=True):
        families = sorted(method_frame["system_family"].unique())
        for outcome in outcomes:
            point, positive_fraction = _family_equal_point(method_frame, outcome)
            hierarchy: dict[str, list[list[np.ndarray]]] = {}
            for family in families:
                family_frame = method_frame.loc[
                    method_frame["system_family"].eq(family)
                ]
                units: list[list[np.ndarray]] = []
                for _, unit_frame in family_frame.groupby(
                    ["dataset_id", "network_id"], sort=False, observed=True
                ):
                    contexts = [
                        context_frame[outcome].to_numpy(dtype=float)
                        for _, context_frame in unit_frame.groupby(
                            CONTEXT_COLUMNS, sort=False, observed=True
                        )
                    ]
                    units.append(contexts)
                hierarchy[family] = units
            draws: list[float] = []
            for _ in range(replicates):
                sampled_families = rng.choice(
                    families, size=len(families), replace=True
                )
                family_values: list[float] = []
                for family in sampled_families:
                    units = hierarchy[str(family)]
                    sampled_units = rng.integers(0, len(units), size=len(units))
                    unit_values: list[float] = []
                    for unit_index in sampled_units:
                        contexts = units[int(unit_index)]
                        sampled_contexts = rng.integers(
                            0, len(contexts), size=len(contexts)
                        )
                        context_values: list[float] = []
                        for context_index in sampled_contexts:
                            values = contexts[int(context_index)]
                            context_values.append(
                                float(values[int(rng.integers(0, len(values)))])
                            )
                        unit_values.append(float(np.mean(context_values)))
                    family_values.append(float(np.mean(unit_values)))
                draws.append(float(np.mean(family_values)))
            samples = np.asarray(draws)
            rows.append(
                {
                    "method": method,
                    "outcome": outcome,
                    "family_equal_mean": point,
                    "blocked_ci_low": float(np.quantile(samples, 0.025)),
                    "blocked_ci_high": float(np.quantile(samples, 0.975)),
                    "positive_family_fraction": positive_fraction,
                    "families": len(families),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(rows)


def _context_randomization_summary(differences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [*CONTEXT_COLUMNS, "system_family", "method"]
    for keys, frame in differences.groupby(group_columns, observed=True):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "shuffle_replicates": len(frame),
                "ordered_spearman": float(frame["spearman_ordered"].iloc[0]),
                "mean_shuffled_spearman": float(frame["spearman_shuffled"].mean()),
                "mean_spearman_gain": float(frame["spearman_gain"].mean()),
                "ordered_better_fraction": float(frame["spearman_gain"].gt(0).mean()),
                "one_sided_randomization_p": float(
                    (1 + frame["spearman_shuffled"].ge(frame["spearman_ordered"].iloc[0]).sum())
                    / (len(frame) + 1)
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_increment_summary(
    context_metrics: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    methods = {
        "static": "ridge_static_summary_loso",
        "temporal": "ridge_temporal_summary_loso",
    }
    selected = context_metrics.loc[
        context_metrics["method"].isin(methods.values()),
        [
            *CONTEXT_COLUMNS,
            "system_family",
            "method",
            "spearman",
            "value_capture_above_random",
            "oracle_regret",
        ],
    ]
    wide = selected.pivot(
        index=[*CONTEXT_COLUMNS, "system_family"],
        columns="method",
        values=["spearman", "value_capture_above_random", "oracle_regret"],
    ).reset_index()
    wide.columns = [
        column if isinstance(column, str) else "__".join(value for value in column if value)
        for column in wide.columns
    ]
    differences = pd.DataFrame(
        {
            **{column: wide[column] for column in [*CONTEXT_COLUMNS, "system_family"]},
            "method": "temporal_summary_increment",
            "shuffle_replicate": 0,
            "spearman_gain": (
                wide[f"spearman__{methods['temporal']}"]
                - wide[f"spearman__{methods['static']}"]
            ),
            "value_capture_gain": (
                wide[f"value_capture_above_random__{methods['temporal']}"]
                - wide[f"value_capture_above_random__{methods['static']}"]
            ),
            "regret_reduction": (
                wide[f"oracle_regret__{methods['static']}"]
                - wide[f"oracle_regret__{methods['temporal']}"]
            ),
        }
    )
    return _blocked_shuffle_summary(
        differences,
        replicates=bootstrap_replicates,
        seed=seed,
    )


def _score_resolution(context_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = context_metrics.loc[
        context_metrics["method"].isin(TEMPORAL_AUDIT_METHODS)
    ].copy()
    selected["nonconstant"] = selected["score_has_variation"].astype(float)
    return (
        selected.groupby(["system_family", "method"], observed=True)
        .agg(
            contexts=("anchor_time", "size"),
            nonconstant_fraction=("nonconstant", "mean"),
            median_unique_scores=("score_unique_count", "median"),
            median_candidates=("candidate_count", "median"),
        )
        .reset_index()
    )


def _plot_order_null(summary: pd.DataFrame, path: Path) -> None:
    display = summary.loc[
        summary["outcome"].isin(("spearman_gain", "value_capture_gain"))
    ].copy()
    display["label"] = display["method"].str.replace("_", " ")
    method_order = list(TEMPORAL_AUDIT_METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0), constrained_layout=True)
    for axis, outcome, title in zip(
        axes,
        ("spearman_gain", "value_capture_gain"),
        ("Rank-correlation difference", "Decision-value difference"),
    ):
        frame = (
            display.loc[display["outcome"].eq(outcome)]
            .set_index("method")
            .reindex(method_order)
            .reset_index()
        )
        positions = np.arange(len(frame))
        errors = np.vstack(
            [
                frame["family_equal_mean"] - frame["blocked_ci_low"],
                frame["blocked_ci_high"] - frame["family_equal_mean"],
            ]
        )
        axis.errorbar(
            frame["family_equal_mean"],
            positions,
            xerr=errors,
            fmt="o",
            color="#4C78A8",
            ecolor="#9ECAE1",
            capsize=5,
            markersize=7,
        )
        axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
        axis.set_yticks(positions, frame["label"])
        axis.invert_yaxis()
        axis.set_xlabel("Observed order minus one shuffled order")
        axis.set_title(title, fontweight="bold")
        sns.despine(ax=axis)
    fig.suptitle(
        "Paired temporal-order randomization audit",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_information_ladder(family_metrics: pd.DataFrame, path: Path) -> None:
    methods = [
        "activity",
        "static_strength",
        "ridge_static_summary_loso",
        "ridge_temporal_summary_loso",
        "exposure_communicability_broadcast",
        "exposure_communicability_receive",
    ]
    selected = family_metrics.loc[family_metrics["method"].isin(methods)]
    pivot = selected.pivot(
        index="system_family", columns="method", values="mean_spearman"
    ).reindex(columns=methods)
    fig, axis = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean within-window rank correlation"},
        ax=axis,
    )
    axis.set_title(
        "History-information baselines by animal-system family",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    axis.set_xlabel("History-only method")
    axis.set_ylabel("Animal-system family")
    axis.tick_params(axis="x", rotation=28)
    axis.tick_params(axis="y", rotation=0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_resolution(resolution: pd.DataFrame, path: Path) -> None:
    pivot = resolution.pivot(
        index="system_family", columns="method", values="nonconstant_fraction"
    ).reindex(columns=TEMPORAL_AUDIT_METHODS)
    fig, axis = plt.subplots(figsize=(14.5, 6.8), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Fraction of windows with a non-constant ranking"},
        ax=axis,
    )
    axis.set_title(
        "Temporal-score resolution by animal-system family",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    axis.set_xlabel("Temporal method")
    axis.set_ylabel("Animal-system family")
    axis.tick_params(axis="x", rotation=28)
    axis.tick_params(axis="y", rotation=0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_time_sensitivity(metrics: pd.DataFrame, path: Path) -> None:
    family = (
        metrics.groupby(
            ["system_family", "direction", "event_time_rule"], observed=True
        )["spearman"]
        .mean()
        .reset_index()
    )
    rules = ["start", "midpoint", "end"]
    palette = {"start": "#9ECAE1", "midpoint": "#4C78A8", "end": "#F28E2B"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), constrained_layout=True)
    for axis, direction in zip(axes, ("broadcast", "receive")):
        frame = family.loc[family["direction"].eq(direction)].copy()
        families = sorted(frame["system_family"].unique())
        positions = {family_name: index for index, family_name in enumerate(families)}
        offsets = {"start": -0.18, "midpoint": 0.0, "end": 0.18}
        for rule in rules:
            rule_frame = frame.loc[frame["event_time_rule"].eq(rule)]
            axis.scatter(
                rule_frame["spearman"],
                [positions[value] + offsets[rule] for value in rule_frame["system_family"]],
                label=rule,
                color=palette[rule],
                s=48,
            )
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.set_yticks(range(len(families)), families)
        axis.set_xlabel("Mean within-window rank correlation")
        axis.set_title(direction.title(), fontweight="bold")
        axis.legend(title="Interval timestamp", loc="lower right")
        sns.despine(ax=axis)
    fig.suptitle(
        "Exposure communicability sensitivity to interval timestamp",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile_config = config["profiles"][profile]
    dataset_ids = list(profile_config.get("datasets", config["data"]["datasets"]))
    labels = pd.read_csv(
        config["data"]["label_path"], dtype={"candidate_id": str}
    )
    labels["history_start"] = pd.to_datetime(labels["history_start"], format="mixed")
    labels["anchor_time"] = pd.to_datetime(labels["anchor_time"], format="mixed")
    labels = _select_labels(
        labels, dataset_ids, profile_config.get("max_contexts_per_dataset")
    )

    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile
    checkpoint_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    attenuation = float(config["temporal"]["attenuation"])
    shuffle_replicates = int(profile_config["shuffle_replicates"])
    random_seed = int(config["evaluation"]["random_seed"])
    time_rules = tuple(config["temporal"]["event_time_sensitivity"])
    history_tables: list[pd.DataFrame] = []
    ordered_tables: list[pd.DataFrame] = []
    shuffled_tables: list[pd.DataFrame] = []
    sensitivity_tables: list[pd.DataFrame] = []
    stream_audits: list[dict[str, object]] = []
    for dataset_id in tqdm(dataset_ids, desc="Temporal information datasets"):
        dataset_labels = labels.loc[labels["dataset_id"].eq(dataset_id)].copy()
        mapper_names = set(dataset_labels["primary_mapper"].astype(str))
        if len(mapper_names) != 1:
            raise ValueError(f"{dataset_id} has inconsistent primary mapper provenance")
        mapper_name = mapper_names.pop()
        beta_path = Path(config["data"]["parameter_selection_paths"][dataset_id])
        betas = _selected_betas(beta_path)
        if set(dataset_labels["parameter_contexts"].astype(int)) != {len(betas)}:
            raise ValueError(
                f"{dataset_id} selected beta count does not match label provenance"
            )
        fingerprint = _dataset_fingerprint(
            dataset_id,
            dataset_labels,
            mapper_name=mapper_name,
            betas=betas,
            attenuation=attenuation,
            shuffle_replicates=shuffle_replicates,
            random_seed=random_seed,
            time_rules=time_rules,
        )
        paths = {
            name: checkpoint_dir / f"{dataset_id}_{name}.parquet"
            for name in ("history", "ordered", "shuffled", "sensitivity")
        }
        audit_path = checkpoint_dir / f"{dataset_id}_audit.json"
        if all(path.exists() for path in paths.values()) and audit_path.exists():
            cached = json.loads(audit_path.read_text(encoding="utf-8"))
            if cached.get("task_fingerprint") == fingerprint:
                history_tables.append(pd.read_parquet(paths["history"]))
                ordered_tables.append(pd.read_parquet(paths["ordered"]))
                shuffled_tables.append(pd.read_parquet(paths["shuffled"]))
                sensitivity_tables.append(pd.read_parquet(paths["sensitivity"]))
                stream_audits.append(cached["stream_audit"])
                continue
        dataset = CanonicalDataset.read(
            Path(config["data"]["canonical_root"]) / dataset_id / "processed"
        )
        stream = compile_named_exposure(dataset, mapper_name)
        history = build_history_features(
            dataset, dataset_labels, exposure_stream=stream
        )
        context_count = dataset_labels[
            ["network_id", "history_start", "anchor_time"]
        ].drop_duplicates().shape[0]
        with tqdm(
            total=context_count,
            desc=f"{dataset_id} contexts",
            leave=False,
        ) as context_bar:
            ordered, shuffled, sensitivity = build_temporal_audit_tables(
                dataset,
                stream,
                dataset_labels,
                attenuation=attenuation,
                betas=betas,
                shuffle_replicates=shuffle_replicates,
                random_seed=random_seed,
                sensitivity_time_rules=time_rules,
                context_progress=lambda _completed, _total: context_bar.update(1),
            )
        stream_audit = {
            "dataset_id": dataset_id,
            "recorded_mapper": mapper_name,
            "compiled_mapper": stream.metadata.get("mapper"),
            "selected_betas": betas,
            "parameter_selection_path": str(beta_path),
            "parameter_selection_sha256": _sha256(beta_path),
            "dyadic_exposures": len(stream.dyadic_exposures),
            "group_exposures": len(stream.group_exposures),
            "group_memberships": len(stream.group_memberships),
        }
        for name, frame in (
            ("history", history),
            ("ordered", ordered),
            ("shuffled", shuffled),
            ("sensitivity", sensitivity),
        ):
            _write_parquet_atomic(frame, paths[name])
        audit_path.write_text(
            json.dumps(
                {"task_fingerprint": fingerprint, "stream_audit": stream_audit},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        history_tables.append(history)
        ordered_tables.append(ordered)
        shuffled_tables.append(shuffled)
        sensitivity_tables.append(sensitivity)
        stream_audits.append(stream_audit)

    history = pd.concat(history_tables, ignore_index=True)
    ordered = pd.concat(ordered_tables, ignore_index=True)
    shuffled = pd.concat(shuffled_tables, ignore_index=True)
    sensitivity = pd.concat(sensitivity_tables, ignore_index=True)
    feature_labels = labels.merge(history, on=KEYS, validate="one_to_one")
    scored = fit_baseline_scores(
        feature_labels,
        ridge_alpha=float(config["evaluation"]["ridge_alpha"]),
        seed=random_seed,
    )
    feature_ablation = fit_feature_ablation_scores(
        feature_labels,
        ridge_alpha=float(config["evaluation"]["ridge_alpha"]),
    )
    scored = scored.merge(
        feature_ablation[
            KEYS
            + [
                "score_ridge_static_summary_loso",
                "score_ridge_temporal_summary_loso",
            ]
        ],
        on=KEYS,
        validate="one_to_one",
    )
    scored = scored.merge(ordered, on=KEYS, validate="one_to_one")
    for method in (*STATIC_METHODS, *TEMPORAL_AUDIT_METHODS):
        scored[f"score_{method}"] = scored[method]
    top_fraction = float(config["evaluation"]["top_fraction"])
    context_metrics, family_metrics = evaluate_baseline_scores(
        scored, top_fraction=top_fraction
    )
    shuffled_metrics = _evaluate_shuffles(
        labels, shuffled, top_fraction=top_fraction
    )
    sensitivity_metrics = _evaluate_sensitivity(
        labels, sensitivity, top_fraction=top_fraction
    )
    differences = _paired_shuffle_differences(context_metrics, shuffled_metrics)
    summary = _blocked_shuffle_summary(
        differences,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=random_seed,
    )
    context_randomization = _context_randomization_summary(differences)
    feature_summary = _feature_increment_summary(
        context_metrics,
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=random_seed + 1,
    )
    resolution = _score_resolution(context_metrics)

    expected_shuffled_rows = (
        len(labels) * shuffle_replicates * len(TEMPORAL_AUDIT_METHODS)
    )
    expected_sensitivity_rows = len(labels) * len(time_rules) * 2
    audit_checks = {
        "label_history_ordered_rows_match": len(labels)
        == len(history)
        == len(ordered)
        == len(scored),
        "core_keys_unique": not history.duplicated(KEYS).any()
        and not ordered.duplicated(KEYS).any(),
        "shuffled_rows_complete": len(shuffled) == expected_shuffled_rows,
        "sensitivity_rows_complete": len(sensitivity) == expected_sensitivity_rows,
        "all_scores_finite": np.isfinite(
            ordered[[*STATIC_METHODS, *TEMPORAL_AUDIT_METHODS]].to_numpy(dtype=float)
        ).all()
        and np.isfinite(shuffled["score"].to_numpy(dtype=float)).all()
        and np.isfinite(sensitivity["score"].to_numpy(dtype=float)).all(),
        "all_recorded_mappers_recompiled": all(
            row["recorded_mapper"] == row["compiled_mapper"]
            for row in stream_audits
        ),
        "all_shuffle_replicates_evaluated": shuffled_metrics[
            "shuffle_replicate"
        ].nunique()
        == shuffle_replicates,
        "every_context_evaluated": context_metrics[CONTEXT_COLUMNS]
        .drop_duplicates()
        .shape[0]
        == labels[CONTEXT_COLUMNS].drop_duplicates().shape[0],
        "shuffle_uncertainty_retained": differences.groupby(
            [*CONTEXT_COLUMNS, "method"], observed=True
        )["shuffle_replicate"].nunique().eq(shuffle_replicates).all(),
    }
    audit = {
        "status": "pass" if all(audit_checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in audit_checks.items()},
        "labels": len(labels),
        "contexts": labels[CONTEXT_COLUMNS].drop_duplicates().shape[0],
        "datasets": labels["dataset_id"].nunique(),
        "system_families": scored["dataset_id"].map(animal_system_family).nunique(),
        "shuffle_replicates": shuffle_replicates,
        "temporal_methods": len(TEMPORAL_AUDIT_METHODS),
    }
    if audit["status"] != "pass":
        raise ValueError(f"temporal information audit failed: {audit['checks']}")

    primary_methods = list(config["interpretation"]["primary_order_methods"])
    primary_rows = summary.loc[
        summary["method"].isin(primary_methods)
        & summary["outcome"].eq("spearman_gain")
    ].copy()
    interpretation = {
        "status": (
            "order_increment_detected"
            if (
                primary_rows["blocked_ci_low"].gt(0)
                & primary_rows["positive_family_fraction"].ge(
                    float(config["interpretation"]["minimum_positive_family_fraction"])
                )
            ).any()
            else "no_robust_order_increment_detected"
        ),
        "scope": (
            "This decision concerns the predeclared duration-aware broadcast/receive "
            "centralities, not whether temporal networks affect epidemic mechanics."
        ),
        "primary_order_methods": primary_methods,
    }

    _write_csv(history, results_dir / "history_features.csv")
    _write_csv(ordered, results_dir / "ordered_scores.csv")
    _write_csv(scored, results_dir / "baseline_predictions.csv")
    _write_csv(context_metrics, results_dir / "ordered_context_metrics.csv")
    _write_csv(family_metrics, results_dir / "ordered_family_metrics.csv")
    _write_csv(shuffled_metrics, results_dir / "shuffle_context_metrics.csv")
    _write_csv(differences, results_dir / "paired_shuffle_differences.csv")
    _write_csv(summary, results_dir / "paired_shuffle_summary.csv")
    _write_csv(
        context_randomization, results_dir / "context_randomization_summary.csv"
    )
    _write_csv(feature_summary, results_dir / "temporal_summary_increment.csv")
    _write_csv(resolution, results_dir / "score_resolution.csv")
    _write_csv(sensitivity_metrics, results_dir / "event_time_sensitivity.csv")
    (results_dir / "stream_audit.json").write_text(
        json.dumps(stream_audits, indent=2, default=str), encoding="utf-8"
    )
    (results_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (results_dir / "interpretation.json").write_text(
        json.dumps(interpretation, indent=2), encoding="utf-8"
    )
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({**config, "profile": profile}, sort_keys=False),
        encoding="utf-8",
    )

    _plot_order_null(summary, report_dir / "paired_temporal_order_audit.png")
    _plot_information_ladder(
        family_metrics, report_dir / "history_information_ladder.png"
    )
    _plot_resolution(resolution, report_dir / "temporal_score_resolution.png")
    _plot_time_sensitivity(
        sensitivity_metrics, report_dir / "event_time_sensitivity.png"
    )

    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "python": platform.python_version(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "label_sha256": _sha256(Path(config["data"]["label_path"])),
        "source_sha256": {
            "reference_baselines": _sha256(
                Path("src/animal_intervention/centrality/reference_baselines.py")
            ),
            "history_features": _sha256(
                Path("src/animal_intervention/centrality/history_features.py")
            ),
            "baseline_ranking": _sha256(
                Path("src/animal_intervention/evaluation/baseline_ranking.py")
            ),
            "experiment_runner": _sha256(Path(__file__)),
        },
        "audit_status": audit["status"],
        "interpretation_status": interpretation["status"],
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    readme = f"""# Corrected temporal-information audit

This experiment repairs the EXP-20260816-004 comparison without rerunning
epidemic labels. It keeps each shuffled ordering as a separate replicate,
uses paired within-window comparisons, and carries shuffle uncertainty into
the blocked interval. It also adds incoming direction, duration-aware event
gains derived from the label-defining selected beta ensemble, score-resolution
diagnostics, and start/midpoint/end interval-timestamp sensitivity.

- Group events remain intact simultaneous hyper-events.
- No within-batch multi-step paths are allowed.
- Static summaries and within-window temporal summaries are compared with
  matched LOSO ridge models.
- The result does not test whether temporal ordering changes epidemic mechanics;
  that is already represented by the event-driven simulator.

Interpretation status: **{interpretation['status']}**.

## Figures

- `paired_temporal_order_audit.png`: observed-minus-single-shuffle differences.
- `history_information_ladder.png`: matched history-information baselines.
- `temporal_score_resolution.png`: where a method can or cannot rank animals.
- `event_time_sensitivity.png`: sensitivity to interval start/midpoint/end timing.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"audit": audit, "interpretation": interpretation}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the corrected temporal-information audit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260816-005_temporal_information_audit.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
