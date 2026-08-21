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

from animal_intervention.centrality import build_history_features
from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.evaluation.baseline_ranking import (
    CONTEXT_COLUMNS,
    evaluate_baseline_scores,
    fit_baseline_scores,
)
from animal_intervention.transmission import compile_named_exposure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in ("history_start", "anchor_time", "horizon_end"):
        if column in output:
            parsed = pd.to_datetime(output[column], format="mixed")
            if parsed.isna().any():
                raise ValueError(f"{column} contains missing or invalid timestamps")
            output[column] = parsed.map(lambda value: value.isoformat())
    output.to_csv(path, index=False)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_worktree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _select_profile_labels(
    labels: pd.DataFrame,
    datasets: list[str],
    max_contexts: int | None,
) -> pd.DataFrame:
    selected = labels.loc[labels["dataset_id"].isin(datasets)].copy()
    if max_contexts is None:
        return selected
    keep: list[pd.DataFrame] = []
    for _, frame in selected.groupby("dataset_id", sort=True, observed=True):
        contexts = frame[CONTEXT_COLUMNS].drop_duplicates().head(max_contexts)
        keep.append(frame.merge(contexts, on=CONTEXT_COLUMNS, how="inner"))
    return pd.concat(keep, ignore_index=True)


def _plot_heatmap(
    family_metrics: pd.DataFrame,
    value_column: str,
    title: str,
    color_label: str,
    path: Path,
) -> None:
    pivot = family_metrics.pivot(
        index="system_family", columns="method", values=value_column
    )
    preferred = [
        "random",
        "activity",
        "partner_diversity",
        "contact_opportunity",
        "recent_activity",
        "composite",
        "ridge_loso",
    ]
    pivot = pivot.reindex(columns=[value for value in preferred if value in pivot])
    height = max(4.8, 0.7 * len(pivot) + 2.4)
    width = max(11.0, 1.35 * len(pivot.columns) + 3.5)
    fig, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.6,
        cbar_kws={"label": color_label},
        ax=axis,
    )
    axis.set_title(title, fontsize=17, fontweight="bold", pad=18)
    axis.set_xlabel("History-only baseline")
    axis.set_ylabel("Held-out animal-system family")
    axis.tick_params(axis="x", rotation=28)
    axis.tick_params(axis="y", rotation=0)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_overall(family_metrics: pd.DataFrame, path: Path) -> None:
    method_order = [
        "random",
        "activity",
        "partner_diversity",
        "contact_opportunity",
        "recent_activity",
        "composite",
        "ridge_loso",
    ]
    summary = (
        family_metrics.groupby("method", observed=True)["mean_spearman"]
        .agg(["mean", "min", "max"])
        .reindex(method_order)
        .dropna()
    )
    positions = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(11.5, 6.6), constrained_layout=True)
    errors = np.vstack(
        [summary["mean"] - summary["min"], summary["max"] - summary["mean"]]
    )
    axis.errorbar(
        positions,
        summary["mean"],
        yerr=errors,
        fmt="o",
        markersize=8,
        capsize=5,
        color="#4C78A8",
        ecolor="#9ECAE1",
    )
    axis.axhline(0, color="#444444", linewidth=1, linestyle="--")
    axis.set_xticks(positions, summary.index, rotation=28, ha="right")
    axis.set_ylabel("Mean within-anchor rank correlation")
    axis.set_xlabel("History-only baseline")
    axis.set_title(
        "Do simple history features predict intervention priority?",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0,
        1.02,
        "Points average held-out families; whiskers show the weakest and strongest family",
        transform=axis.transAxes,
        color="#555555",
        fontsize=10.5,
    )
    sns.despine(ax=axis)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _audit(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    scored: pd.DataFrame,
    context_metrics: pd.DataFrame,
) -> dict[str, Any]:
    keys = ["dataset_id", "network_id", "anchor_time", "candidate_id"]
    bounded_columns = [
        "eligible_partner_fraction",
        "recency_score",
        "recent_activity_fraction",
        "active_span_fraction",
    ]
    nonnegative_columns = [
        "activity_count",
        "event_rate_per_day",
        "contact_opportunity_rate",
        "weighted_exposure_rate",
        "observed_partner_count",
        "location_count",
        "mean_group_size",
    ]
    random_metrics = context_metrics.loc[context_metrics["method"].eq("random")]
    checks = {
        "feature_rows_match_labels": len(features) == len(labels),
        "feature_keys_unique": not features.duplicated(keys).any(),
        "scored_rows_match_labels": len(scored) == len(labels),
        "no_future_columns_in_features": not any(
            value in features.columns
            for value in ("horizon_end", "robust_intervention_value", "robust_rank")
        ),
        "all_scores_finite": np.isfinite(
            scored.filter(regex=r"^score_").to_numpy(dtype=float)
        ).all(),
        "rank_metrics_finite": np.isfinite(
            context_metrics["spearman"].to_numpy(dtype=float)
        ).all(),
        "every_context_evaluated": context_metrics[CONTEXT_COLUMNS]
        .drop_duplicates()
        .shape[0]
        == labels[CONTEXT_COLUMNS].drop_duplicates().shape[0],
        "bounded_features_within_unit_interval": features[bounded_columns]
        .apply(lambda column: column.between(0.0, 1.0).all())
        .all(),
        "count_and_rate_features_nonnegative": features[nonnegative_columns]
        .ge(0.0)
        .all()
        .all(),
        "random_baseline_is_exact_expectation": random_metrics["spearman"]
        .eq(0.0)
        .all()
        and random_metrics["value_capture_above_random"].fillna(0.0).eq(0.0).all()
        and random_metrics["selection_evaluation"]
        .eq("analytic_random_expectation")
        .all(),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "label_rows": int(len(labels)),
        "feature_rows": int(len(features)),
        "contexts": int(labels[CONTEXT_COLUMNS].drop_duplicates().shape[0]),
        "datasets": int(labels["dataset_id"].nunique()),
        "constant_score_contexts": int((~context_metrics["score_has_variation"]).sum()),
        "interpretation_limits": [
            "simulation-derived labels are model-based targets, not field causal truth",
            "ridge predicts priority percentile and is not calibrated to raw intervention value",
            "formal uncertainty requires network- and anchor-blocked inference",
            "temporal-order value has not yet been tested against shuffled history",
        ],
    }


def run(config_path: Path, profile: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile_config = config["profiles"][profile]
    dataset_ids = list(profile_config.get("datasets", config["data"]["datasets"]))
    max_contexts = profile_config.get("max_contexts_per_dataset")
    label_path = Path(config["data"]["label_path"])
    labels = pd.read_csv(label_path, dtype={"candidate_id": str})
    labels["anchor_time"] = pd.to_datetime(labels["anchor_time"], format="mixed")
    labels = _select_profile_labels(labels, dataset_ids, max_contexts)

    feature_tables: list[pd.DataFrame] = []
    for dataset_id in tqdm(dataset_ids, desc="Building history-only baseline features"):
        dataset = CanonicalDataset.read(
            Path(config["data"]["canonical_root"]) / dataset_id / "processed"
        )
        dataset_labels = labels.loc[labels["dataset_id"].eq(dataset_id)]
        mapper_names = set(dataset_labels["primary_mapper"].astype(str))
        if len(mapper_names) != 1:
            raise ValueError(
                f"{dataset_id} does not have exactly one recorded primary mapper"
            )
        stream = compile_named_exposure(dataset, mapper_names.pop())
        feature_tables.append(
            build_history_features(dataset, labels, exposure_stream=stream)
        )
    features = pd.concat(feature_tables, ignore_index=True)
    feature_labels = labels.merge(
        features,
        on=["dataset_id", "network_id", "anchor_time", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    evaluation = config["evaluation"]
    scored = fit_baseline_scores(
        feature_labels,
        ridge_alpha=float(evaluation["ridge_alpha"]),
        seed=int(evaluation["random_seed"]),
    )
    context_metrics, family_metrics = evaluate_baseline_scores(
        scored, top_fraction=float(evaluation["top_fraction"])
    )
    audit = _audit(labels, features, scored, context_metrics)
    if audit["status"] != "pass":
        raise ValueError(f"baseline audit failed: {audit['checks']}")

    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(features, results_dir / "history_features.csv")
    _write_csv(scored, results_dir / "baseline_predictions.csv")
    _write_csv(context_metrics, results_dir / "context_metrics.csv")
    _write_csv(family_metrics, results_dir / "family_metrics.csv")
    roundtrip = pd.read_csv(
        results_dir / "baseline_predictions.csv", dtype={"candidate_id": str}
    )
    roundtrip_keys = ["dataset_id", "network_id", "anchor_time", "candidate_id"]
    audit["checks"]["csv_roundtrip_keys_unique"] = not roundtrip.duplicated(
        roundtrip_keys
    ).any()
    audit["checks"]["csv_roundtrip_contexts_preserved"] = (
        roundtrip[CONTEXT_COLUMNS].drop_duplicates().shape[0] == audit["contexts"]
    )
    audit["status"] = (
        "pass" if all(audit["checks"].values()) else "fail"
    )
    if audit["status"] != "pass":
        raise ValueError(f"baseline CSV round-trip audit failed: {audit['checks']}")
    (results_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    resolved_config = {**config, "profile": profile}
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    _plot_heatmap(
        family_metrics,
        "mean_spearman",
        "History-only ranking signal in held-out animal systems",
        "Mean rank correlation",
        report_dir / "rank_correlation_by_system.png",
    )
    _plot_heatmap(
        family_metrics,
        "mean_value_capture",
        "Intervention value captured above random selection",
        "Value captured above random",
        report_dir / "value_capture_by_system.png",
    )
    _plot_overall(family_metrics, report_dir / "overall_baseline_signal.png")

    overall = (
        family_metrics.groupby("method", observed=True)[
            ["mean_spearman", "mean_value_capture", "mean_top_set_overlap"]
        ]
        .mean()
        .sort_values("mean_spearman", ascending=False)
    )
    overall.to_csv(results_dir / "overall_metrics.csv")
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "python": platform.python_version(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "label_path": str(label_path),
        "label_sha256": _sha256(label_path),
        "source_sha256": {
            "history_features": _sha256(
                Path("src/animal_intervention/centrality/history_features.py")
            ),
            "baseline_ranking": _sha256(
                Path("src/animal_intervention/evaluation/baseline_ranking.py")
            ),
            "experiment_runner": _sha256(Path(__file__)),
        },
        "random_seed": int(evaluation["random_seed"]),
        "split": evaluation["split"],
        "linked_outer_fold": evaluation["linked_outer_fold"],
        "audit_status": audit["status"],
        "outputs": sorted(path.name for path in results_dir.iterdir()),
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rounded = overall.round(3).reset_index()
    table_lines = [
        "| method | mean rank correlation | mean value capture | mean top-set overlap |",
        "|---|---:|---:|---:|",
    ]
    table_lines.extend(
        f"| {row.method} | {row.mean_spearman:.3f} | "
        f"{row.mean_value_capture:.3f} | {row.mean_top_set_overlap:.3f} |"
        for row in rounded.itertuples(index=False)
    )
    table = "\n".join(table_lines)
    readme = f"""# History-only baseline pilot

This experiment tests whether simple pre-anchor observations can rank animals by
their offline simulation-derived intervention value. It is a predictive baseline,
not field evidence of causal intervention effectiveness.

- Split: leave one independent animal-system family out.
- Wytham divorce and experimental songbirds share one linked outer fold.
- Features use only the pre-anchor portion of interactions; future replay is label-only.
- Eligible-partner fraction and all-observed-partner count are separate features.
- Random-selection metrics are analytic expectations, not one sampled random ranking.
- The supervised ridge baseline predicts within-anchor priority percentile and is
  evaluated as a ranking baseline, not a calibrated raw-benefit predictor.
- Selection metrics are evaluated using the continuous intervention-value label.

## Family-averaged results

{table}

## Figures

- `rank_correlation_by_system.png`: ordering agreement in each held-out family.
- `value_capture_by_system.png`: benefit of the selected top 20% above random selection.
- `overall_baseline_signal.png`: average and cross-family range of rank signal.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"audit": audit, "overall": overall.reset_index().to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run history-only ranking baselines")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260816-002_history_baseline_pilot.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
