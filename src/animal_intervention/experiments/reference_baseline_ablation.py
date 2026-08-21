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

from animal_intervention.centrality import build_history_features, build_reference_centralities
from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.evaluation.baseline_ranking import (
    CONTEXT_COLUMNS,
    evaluate_baseline_scores,
    fit_baseline_scores,
)
from animal_intervention.transmission import compile_named_exposure


REFERENCE_METHODS = [
    "static_degree",
    "static_strength",
    "static_pagerank",
    "static_eigenvector",
    "static_k_core",
    "shuffled_temporal_reach",
    "ordered_temporal_reach",
    "shuffled_dynamic_communicability",
    "ordered_dynamic_communicability",
]


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


def _dataset_task_fingerprint(
    dataset_id: str,
    dataset_labels: pd.DataFrame,
    *,
    mapper_name: str,
    attenuation: float,
    shuffle_replicates: int,
    random_seed: int,
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
        "attenuation": attenuation,
        "shuffle_replicates": shuffle_replicates,
        "random_seed": random_seed,
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


def _order_differences(context_metrics: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "temporal_reach": (
            "ordered_temporal_reach",
            "shuffled_temporal_reach",
        ),
        "dynamic_communicability": (
            "ordered_dynamic_communicability",
            "shuffled_dynamic_communicability",
        ),
    }
    rows: list[pd.DataFrame] = []
    metrics = ["spearman", "value_capture_above_random", "oracle_regret"]
    keys = [*CONTEXT_COLUMNS, "system_family"]
    for temporal_metric, (ordered_name, shuffled_name) in pairs.items():
        ordered = context_metrics.loc[
            context_metrics["method"].eq(ordered_name), keys + metrics
        ]
        shuffled = context_metrics.loc[
            context_metrics["method"].eq(shuffled_name), keys + metrics
        ]
        merged = ordered.merge(
            shuffled,
            on=keys,
            suffixes=("_ordered", "_shuffled"),
            validate="one_to_one",
        )
        merged["temporal_metric"] = temporal_metric
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
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def _blocked_bootstrap(
    differences: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result_rows: list[dict[str, object]] = []
    gain_columns = ["spearman_gain", "value_capture_gain", "regret_reduction"]
    for temporal_metric, frame in differences.groupby(
        "temporal_metric", observed=True
    ):
        family_means = frame.groupby("system_family", observed=True)[
            gain_columns
        ].mean()
        families = list(family_means.index)
        draws = {column: [] for column in gain_columns}
        for _ in range(replicates):
            sampled_families = rng.choice(families, size=len(families), replace=True)
            values = {column: [] for column in gain_columns}
            for family in sampled_families:
                family_frame = frame.loc[frame["system_family"].eq(family)]
                unit_keys = family_frame[["dataset_id", "network_id"]].drop_duplicates()
                sampled_units = rng.integers(0, len(unit_keys), size=len(unit_keys))
                unit_values = {column: [] for column in gain_columns}
                for unit_index in sampled_units:
                    unit = unit_keys.iloc[int(unit_index)]
                    selected = family_frame.loc[
                        family_frame["dataset_id"].eq(unit["dataset_id"])
                        & family_frame["network_id"].eq(unit["network_id"])
                    ]
                    for column in gain_columns:
                        unit_values[column].append(float(selected[column].mean()))
                for column in gain_columns:
                    values[column].append(float(np.mean(unit_values[column])))
            for column in gain_columns:
                draws[column].append(float(np.mean(values[column])))
        for column in gain_columns:
            samples = np.asarray(draws[column], dtype=float)
            result_rows.append(
                {
                    "temporal_metric": temporal_metric,
                    "outcome": column,
                    "family_equal_mean": float(family_means[column].mean()),
                    "blocked_ci_low": float(np.quantile(samples, 0.025)),
                    "blocked_ci_high": float(np.quantile(samples, 0.975)),
                    "positive_family_fraction": float(
                        family_means[column].gt(0).mean()
                    ),
                    "families": len(families),
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(result_rows)


def _plot_family_heatmap(family_metrics: pd.DataFrame, path: Path) -> None:
    selected = family_metrics.loc[family_metrics["method"].isin(REFERENCE_METHODS)]
    pivot = selected.pivot(
        index="system_family", columns="method", values="mean_spearman"
    ).reindex(columns=REFERENCE_METHODS)
    fig, axis = plt.subplots(figsize=(17, 7.2), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Mean within-anchor rank correlation"},
        ax=axis,
    )
    axis.set_title(
        "Reference centrality ranking by animal-system family",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    axis.set_xlabel("History-only method")
    axis.set_ylabel("Animal-system family")
    axis.tick_params(axis="x", rotation=32)
    axis.tick_params(axis="y", rotation=0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_order_increment(summary: pd.DataFrame, path: Path) -> None:
    display = summary.loc[
        summary["outcome"].isin(("spearman_gain", "value_capture_gain"))
    ].copy()
    display["label"] = display["temporal_metric"].str.replace("_", " ")
    outcomes = ["spearman_gain", "value_capture_gain"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    for axis, outcome in zip(axes, outcomes):
        frame = display.loc[display["outcome"].eq(outcome)].reset_index(drop=True)
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
            markersize=8,
        )
        axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
        axis.set_yticks(positions, frame["label"])
        axis.set_xlabel(
            "Ordered minus shuffled rank correlation"
            if outcome == "spearman_gain"
            else "Ordered minus shuffled value capture"
        )
        axis.set_title(
            "Ranking gain" if outcome == "spearman_gain" else "Decision-value gain",
            fontweight="bold",
        )
        sns.despine(ax=axis)
    fig.suptitle(
        "Increment from observed temporal order",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_order_scatter(context_metrics: pd.DataFrame, path: Path) -> None:
    differences = _order_differences(context_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    for axis, (metric, frame) in zip(
        axes, differences.groupby("temporal_metric", sort=True, observed=True)
    ):
        sns.scatterplot(
            data=frame,
            x="spearman_shuffled",
            y="spearman_ordered",
            hue="system_family",
            s=45,
            alpha=0.75,
            ax=axis,
        )
        lower = min(axis.get_xlim()[0], axis.get_ylim()[0])
        upper = max(axis.get_xlim()[1], axis.get_ylim()[1])
        axis.plot([lower, upper], [lower, upper], "--", color="#444444", linewidth=1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(metric.replace("_", " ").title(), fontweight="bold")
        axis.set_xlabel("Time-shuffled history rank correlation")
        axis.set_ylabel("Observed-order history rank correlation")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("Animal-system family")
    fig.suptitle(
        "Observed versus time-shuffled history by prediction window",
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

    history_tables: list[pd.DataFrame] = []
    reference_tables: list[pd.DataFrame] = []
    stream_audits: list[dict[str, object]] = []
    for dataset_id in tqdm(dataset_ids, desc="Reference baseline datasets"):
        dataset = CanonicalDataset.read(
            Path(config["data"]["canonical_root"]) / dataset_id / "processed"
        )
        dataset_labels = labels.loc[labels["dataset_id"].eq(dataset_id)].copy()
        mapper_names = set(dataset_labels["primary_mapper"].astype(str))
        if len(mapper_names) != 1:
            raise ValueError(f"{dataset_id} has inconsistent primary mapper provenance")
        mapper_name = mapper_names.pop()
        task_fingerprint = _dataset_task_fingerprint(
            dataset_id,
            dataset_labels,
            mapper_name=mapper_name,
            attenuation=float(config["temporal"]["attenuation"]),
            shuffle_replicates=int(profile_config["shuffle_replicates"]),
            random_seed=int(config["evaluation"]["random_seed"]),
        )
        history_checkpoint = checkpoint_dir / f"{dataset_id}_history.parquet"
        reference_checkpoint = checkpoint_dir / f"{dataset_id}_reference.parquet"
        audit_checkpoint = checkpoint_dir / f"{dataset_id}_audit.json"
        if history_checkpoint.exists() and reference_checkpoint.exists() and audit_checkpoint.exists():
            cached_audit = json.loads(audit_checkpoint.read_text(encoding="utf-8"))
            if cached_audit.get("task_fingerprint") == task_fingerprint:
                history_tables.append(pd.read_parquet(history_checkpoint))
                reference_tables.append(pd.read_parquet(reference_checkpoint))
                stream_audits.append(cached_audit["stream_audit"])
                continue
        stream = compile_named_exposure(dataset, mapper_name)
        history_table = build_history_features(
            dataset, dataset_labels, exposure_stream=stream
        )
        reference_table = build_reference_centralities(
            dataset,
            stream,
            dataset_labels,
            attenuation=float(config["temporal"]["attenuation"]),
            shuffle_replicates=int(profile_config["shuffle_replicates"]),
            random_seed=int(config["evaluation"]["random_seed"]),
        )
        stream_audit = {
            "dataset_id": dataset_id,
            "recorded_mapper": mapper_name,
            "compiled_mapper": stream.metadata.get("mapper"),
            "dyadic_exposures": len(stream.dyadic_exposures),
            "group_exposures": len(stream.group_exposures),
            "group_memberships": len(stream.group_memberships),
            "mapper_metadata": stream.metadata,
        }
        _write_parquet_atomic(history_table, history_checkpoint)
        _write_parquet_atomic(reference_table, reference_checkpoint)
        audit_checkpoint.write_text(
            json.dumps(
                {
                    "task_fingerprint": task_fingerprint,
                    "stream_audit": stream_audit,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        history_tables.append(history_table)
        reference_tables.append(reference_table)
        stream_audits.append(stream_audit)

    keys = ["dataset_id", "network_id", "anchor_time", "candidate_id"]
    history = pd.concat(history_tables, ignore_index=True)
    reference = pd.concat(reference_tables, ignore_index=True)
    feature_labels = labels.merge(history, on=keys, validate="one_to_one")
    scored = fit_baseline_scores(
        feature_labels,
        ridge_alpha=float(config["evaluation"]["ridge_alpha"]),
        seed=int(config["evaluation"]["random_seed"]),
    )
    scored = scored.merge(reference, on=keys, validate="one_to_one")
    for method in REFERENCE_METHODS:
        scored[f"score_{method}"] = scored[method]
    context_metrics, family_metrics = evaluate_baseline_scores(
        scored, top_fraction=float(config["evaluation"]["top_fraction"])
    )
    differences = _order_differences(context_metrics)
    increment_summary = _blocked_bootstrap(
        differences,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["random_seed"]),
    )

    primary = increment_summary.loc[
        increment_summary["temporal_metric"].eq(
            config["gate"]["primary_temporal_metric"]
        )
    ].set_index("outcome")
    family_differences = differences.loc[
        differences["temporal_metric"].eq(config["gate"]["primary_temporal_metric"])
    ].groupby("system_family", observed=True)[
        ["spearman_gain", "value_capture_gain"]
    ].mean()
    gate_checks = {
        "minimum_mean_spearman_gain": float(
            primary.loc["spearman_gain", "family_equal_mean"]
        )
        >= float(config["gate"]["minimum_mean_spearman_gain"]),
        "minimum_mean_value_capture_gain": float(
            primary.loc["value_capture_gain", "family_equal_mean"]
        )
        >= float(config["gate"]["minimum_mean_value_capture_gain"]),
        "minimum_positive_family_fraction": float(
            family_differences["spearman_gain"].gt(0).mean()
        )
        >= float(config["gate"]["minimum_positive_family_fraction"]),
    }
    gate = {
        "status": "pass" if all(gate_checks.values()) else "not_passed",
        "primary_temporal_metric": config["gate"]["primary_temporal_metric"],
        "checks": gate_checks,
        "interpretation": (
            "This exploratory gate decides whether temporal order justifies encoder work; "
            "it is not a field-effect claim."
        ),
    }

    audit_checks = {
        "label_history_reference_rows_match": len(labels)
        == len(history)
        == len(reference)
        == len(scored),
        "history_keys_unique": not history.duplicated(keys).any(),
        "reference_keys_unique": not reference.duplicated(keys).any(),
        "all_reference_scores_finite": np.isfinite(
            reference[REFERENCE_METHODS].to_numpy(dtype=float)
        ).all(),
        "all_recorded_mappers_recompiled": all(
            row["recorded_mapper"] == row["compiled_mapper"] for row in stream_audits
        ),
        "every_context_evaluated": context_metrics[CONTEXT_COLUMNS]
        .drop_duplicates()
        .shape[0]
        == labels[CONTEXT_COLUMNS].drop_duplicates().shape[0],
        "all_order_pairs_complete": len(differences)
        == 2 * labels[CONTEXT_COLUMNS].drop_duplicates().shape[0],
    }
    audit = {
        "status": "pass" if all(audit_checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in audit_checks.items()},
        "labels": len(labels),
        "contexts": labels[CONTEXT_COLUMNS].drop_duplicates().shape[0],
        "datasets": labels["dataset_id"].nunique(),
        "system_families": scored["system_family"].nunique(),
        "shuffle_replicates": int(profile_config["shuffle_replicates"]),
    }
    if audit["status"] != "pass":
        raise ValueError(f"reference baseline audit failed: {audit['checks']}")

    _write_csv(history, results_dir / "history_features.csv")
    _write_csv(reference, results_dir / "reference_centralities.csv")
    _write_csv(scored, results_dir / "baseline_predictions.csv")
    _write_csv(context_metrics, results_dir / "context_metrics.csv")
    _write_csv(family_metrics, results_dir / "family_metrics.csv")
    _write_csv(differences, results_dir / "temporal_order_differences.csv")
    _write_csv(increment_summary, results_dir / "temporal_order_summary.csv")
    (results_dir / "stream_audit.json").write_text(
        json.dumps(stream_audits, indent=2, default=str), encoding="utf-8"
    )
    (results_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (results_dir / "gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({**config, "profile": profile}, sort_keys=False),
        encoding="utf-8",
    )

    _plot_family_heatmap(family_metrics, report_dir / "centrality_by_family.png")
    _plot_order_increment(
        increment_summary, report_dir / "temporal_order_increment.png"
    )
    _plot_order_scatter(
        context_metrics, report_dir / "ordered_vs_shuffled.png"
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
        "gate_status": gate["status"],
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    readme = f"""# Reference centrality and temporal-order ablation

This experiment compares deployment-safe history summaries, reference static
centralities, and two order-aware temporal centralities against the same
simulation-derived intervention labels.

- Primary event stream is recompiled from each label's recorded mapper.
- Static graph weights use integrated pre-beta exposure opportunity.
- Group events remain simultaneous hyper-events and use frequency-dependent
  pair weights in the aggregate view.
- Temporal methods allow at most one transition per simultaneous event batch;
  they do not invent within-timestamp cascades.
- Shuffling permutes intact simultaneous batches and preserves their contents,
  weights, concurrency structure, and count.
- Outer evaluation leaves one independent animal-system family out; linked
  Wytham deposits remain in one fold.

Gate status: **{gate['status']}**. This is an exploratory engineering gate for
whether temporal-order encoder work is justified, not field validation.

## Figures

- `centrality_by_family.png`: reference-method rank correlation by held-out family.
- `temporal_order_increment.png`: family-equal ordered-minus-shuffled gains with
  network/anchor-blocked bootstrap intervals.
- `ordered_vs_shuffled.png`: context-level ordered versus shuffled comparison.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"audit": audit, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reference centrality and temporal-order ablations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260816-004_reference_baseline_ablation.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
