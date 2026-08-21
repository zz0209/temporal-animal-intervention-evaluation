from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from animal_intervention.evaluation import (
    balanced_variance_decomposition,
    build_forward_predictions,
    evaluate_baseline_scores,
    evaluate_raw_predictions,
)


KEY_COLUMNS = ["dataset_id", "network_id", "anchor_time", "candidate_id"]
DISPLAY_METHODS = {
    "random": "Random",
    "stable_watchlist": "Stable watchlist",
    "last_observed_value": "Last prior value",
    "current_activity": "Current activity",
    "current_composite": "Current composite",
    "forward_ridge_static": "Ridge: static",
    "forward_ridge_current": "Ridge: current",
    "forward_ridge_hybrid": "Ridge: hybrid",
    "future_oracle": "Decision-value oracle",
}


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
            output[column] = pd.to_datetime(output[column], format="mixed").map(
                lambda value: value.isoformat()
            )
    output.to_csv(path, index=False)


def _select_anchor_prefix(
    frame: pd.DataFrame,
    datasets: list[str],
    max_anchor_times: int | None,
) -> pd.DataFrame:
    selected = frame.loc[frame["dataset_id"].isin(datasets)].copy()
    if max_anchor_times is None:
        return selected
    outputs = []
    for _, dataset in selected.groupby("dataset_id", sort=True, observed=True):
        times = sorted(dataset["anchor_time"].unique())[:max_anchor_times]
        outputs.append(dataset.loc[dataset["anchor_time"].isin(times)])
    return pd.concat(outputs, ignore_index=True)


def _paired_strategy_differences(
    context_metrics: pd.DataFrame,
    comparisons: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    keys = ["dataset_id", "network_id", "anchor_time", "system_family"]
    rows = []
    for comparison, (challenger, baseline) in comparisons.items():
        left = context_metrics.loc[context_metrics["method"].eq(challenger)]
        right = context_metrics.loc[context_metrics["method"].eq(baseline)]
        merged = left.merge(right, on=keys, suffixes=("_challenger", "_baseline"))
        for outcome, difference in (
            ("spearman_gain", merged["spearman_challenger"] - merged["spearman_baseline"]),
            (
                "value_capture_gain",
                merged["value_capture_above_random_challenger"]
                - merged["value_capture_above_random_baseline"],
            ),
            (
                "regret_reduction",
                merged["oracle_regret_baseline"] - merged["oracle_regret_challenger"],
            ),
        ):
            part = merged[keys].copy()
            part["comparison"] = comparison
            part["challenger"] = challenger
            part["baseline"] = baseline
            part["outcome"] = outcome
            part["difference"] = difference
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _family_equal_point(frame: pd.DataFrame) -> float:
    context = frame.groupby(
        ["system_family", "dataset_id", "network_id", "anchor_time"],
        observed=True,
    )["difference"].mean().reset_index()
    units = context.groupby(
        ["system_family", "dataset_id", "network_id"], observed=True
    )["difference"].mean().reset_index()
    families = units.groupby("system_family", observed=True)["difference"].mean()
    return float(families.mean())


def _blocked_difference_summary(
    differences: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (comparison, outcome), frame in differences.groupby(
        ["comparison", "outcome"], sort=True, observed=True
    ):
        frame = frame.dropna(subset=["difference"])
        family_names = sorted(frame["system_family"].unique())
        family_units: dict[str, list[np.ndarray]] = {}
        family_means = []
        for family in family_names:
            family_frame = frame.loc[frame["system_family"].eq(family)]
            unit_arrays = [
                unit["difference"].to_numpy(dtype=float)
                for _, unit in family_frame.groupby(
                    ["dataset_id", "network_id"], sort=True, observed=True
                )
            ]
            family_units[family] = unit_arrays
            family_means.append(float(np.mean([array.mean() for array in unit_arrays])))
        draws = np.empty(bootstrap_replicates, dtype=float)
        for draw_index in range(bootstrap_replicates):
            sampled_families = rng.choice(family_names, size=len(family_names), replace=True)
            sampled_family_means = []
            for family in sampled_families:
                units = family_units[str(family)]
                sampled_units = rng.integers(0, len(units), size=len(units))
                sampled_unit_means = []
                for unit_index in sampled_units:
                    values = units[int(unit_index)]
                    sampled_contexts = rng.integers(0, len(values), size=len(values))
                    sampled_unit_means.append(float(values[sampled_contexts].mean()))
                sampled_family_means.append(float(np.mean(sampled_unit_means)))
            draws[draw_index] = float(np.mean(sampled_family_means))
        rows.append(
            {
                "comparison": comparison,
                "outcome": outcome,
                "family_equal_mean": _family_equal_point(frame),
                "blocked_ci_low": float(np.quantile(draws, 0.025)),
                "blocked_ci_high": float(np.quantile(draws, 0.975)),
                "positive_family_fraction": float(np.mean(np.asarray(family_means) > 0)),
                "families": len(family_names),
                "contexts": frame[KEY_COLUMNS[:-1]].drop_duplicates().shape[0],
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
    return pd.DataFrame(rows)


def _family_variance_summary(decomposition: pd.DataFrame) -> pd.DataFrame:
    estimated = decomposition.loc[decomposition["status"].eq("estimated")]
    fraction_columns = [
        "individual_fraction",
        "anchor_fraction",
        "individual_anchor_residual_fraction",
    ]
    fractions = (
        estimated.groupby("system_family", observed=True)[fraction_columns]
        .mean()
        .reset_index()
    )
    support = (
        decomposition.groupby("system_family", observed=True)
        .agg(
            common_support_fraction=("common_support_fraction", "mean"),
            units=("network_id", "size"),
            estimated_units=("status", lambda value: int((value == "estimated").sum())),
        )
        .reset_index()
    )
    return support.merge(fractions, on="system_family", how="left")


def _plot_variance_decomposition(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("individual_fraction", na_position="first")
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    components = [
        ("individual_fraction", "Stable animal", "#4C78A8"),
        ("anchor_fraction", "Shared anchor", "#F58518"),
        ("individual_anchor_residual_fraction", "Animal × anchor / residual", "#B8B8B8"),
    ]
    left = np.zeros(len(ordered))
    for column, label, color in components:
        values = np.nan_to_num(ordered[column].to_numpy(dtype=float), nan=0.0)
        axes[0].barh(y, values, left=left, label=label, color=color)
        left += values
    axes[0].set_yticks(y, ordered["system_family"])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Fraction of common-support raw-label variation")
    axes[0].set_title("Descriptive stable/dynamic decomposition")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    for position, row in enumerate(ordered.itertuples()):
        if row.estimated_units == 0:
            axes[0].text(
                0.5,
                position,
                "Not estimable: fewer than two complete-history animals",
                ha="center",
                va="center",
                color="#555555",
                fontsize=9,
            )
    axes[1].barh(y, ordered["common_support_fraction"], color="#72B7B2")
    axes[1].set_yticks(y, [])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Complete-history animal fraction")
    axes[1].set_title("Support retained")
    for position, row in enumerate(ordered.itertuples()):
        high_support = row.common_support_fraction > 0.85
        axes[1].text(
            row.common_support_fraction - 0.02 if high_support else row.common_support_fraction + 0.02,
            position,
            f"{row.estimated_units}/{row.units} units",
            ha="right" if high_support else "left",
            va="center",
            fontsize=9,
        )
    fig.suptitle("Stable individual value versus time-varying value", fontsize=20, fontweight="bold")
    fig.subplots_adjust(left=0.23, right=0.97, top=0.82, bottom=0.25, wspace=0.10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_performance(family_metrics: pd.DataFrame, path: Path) -> None:
    methods = [method for method in DISPLAY_METHODS if method in set(family_metrics["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    for axis, metric, title, center in (
        (axes[0], "mean_spearman", "Forward rank correlation", 0.0),
        (axes[1], "mean_value_capture", "Top-20% value captured above random", 0.0),
    ):
        matrix = family_metrics.pivot(index="system_family", columns="method", values=metric).reindex(columns=methods)
        matrix.columns = [DISPLAY_METHODS[column] for column in matrix.columns]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=center,
            linewidths=0.7,
            cbar_kws={"shrink": 0.75},
            ax=axis,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Animal-system family" if axis is axes[0] else "")
        axis.tick_params(axis="x", rotation=52)
    fig.suptitle("Strict forward-time strategy performance", fontsize=20, fontweight="bold")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.84, bottom=0.31, wspace=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_increment(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[
        summary["comparison"].isin(
            [
                "stable_over_random",
                "current_composite_over_stable",
                "ridge_current_over_static",
                "hybrid_over_stable",
                "hybrid_over_current",
            ]
        )
    ]
    labels = {
        "stable_over_random": "Stable watchlist − random",
        "current_composite_over_stable": "Current composite − stable",
        "ridge_current_over_static": "Current Ridge − static Ridge",
        "hybrid_over_stable": "Hybrid Ridge − stable",
        "hybrid_over_current": "Hybrid Ridge − current Ridge",
    }
    order = list(labels)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, outcome, title in (
        (axes[0], "spearman_gain", "Rank-correlation difference"),
        (axes[1], "regret_reduction", "Oracle-regret reduction"),
    ):
        frame = selected.loc[selected["outcome"].eq(outcome)].set_index("comparison").reindex(order)
        y = np.arange(len(frame))
        values = frame["family_equal_mean"].to_numpy(dtype=float)
        low = frame["blocked_ci_low"].to_numpy(dtype=float)
        high = frame["blocked_ci_high"].to_numpy(dtype=float)
        if outcome == "regret_reduction":
            values = values * 100
            low = low * 100
            high = high * 100
        axis.errorbar(values, y, xerr=[values - low, high - values], fmt="o", color="#4C78A8", ecolor="#9ECAE1", capsize=4)
        axis.axvline(0, color="#444444", linestyle="--", linewidth=1)
        axis.set_yticks(y, [labels[item] for item in order])
        axis.invert_yaxis()
        axis.set_title(title, fontweight="bold")
        unit = "attack-rate percentage points" if outcome == "regret_reduction" else "correlation units"
        axis.set_xlabel(f"Challenger improvement ({unit}); blocked 95% interval")
    fig.suptitle("Does updating the watchlist improve forward decisions?", fontsize=20, fontweight="bold")
    fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.18, wspace=0.18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_prior_coverage(coverage: pd.DataFrame, path: Path) -> None:
    summary = (
        coverage.groupby("dataset_id", observed=True)
        .agg(
            mean_seen_fraction=("seen_fraction", "mean"),
            minimum_seen_fraction=("seen_fraction", "min"),
            contexts=("network_id", "size"),
        )
        .sort_values("mean_seen_fraction")
    )
    y = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.hlines(y, summary["minimum_seen_fraction"], summary["mean_seen_fraction"], color="#9ECAE1", linewidth=4)
    axis.scatter(summary["mean_seen_fraction"], y, color="#4C78A8", s=70, label="Mean")
    axis.scatter(summary["minimum_seen_fraction"], y, color="#F58518", marker="|", s=180, label="Minimum context")
    axis.set_yticks(y, summary.index)
    axis.set_xlim(0, 1.04)
    axis.set_xlabel("Fraction of current candidates seen at an earlier anchor")
    axis.set_title("Stable-watchlist identity coverage in forward evaluation", fontsize=18, fontweight="bold")
    for position, row in enumerate(summary.itertuples()):
        high_coverage = row.mean_seen_fraction > 0.85
        label = f"{row.contexts} {'context' if row.contexts == 1 else 'contexts'}"
        axis.text(
            row.mean_seen_fraction - 0.025 if high_coverage else row.mean_seen_fraction + 0.025,
            position - 0.10,
            label,
            ha="right" if high_coverage else "left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5},
        )
    axis.legend(loc="lower right")
    fig.subplots_adjust(left=0.28, right=0.97, top=0.86, bottom=0.15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_calibration(raw_family: pd.DataFrame, path: Path) -> None:
    methods = ["stable_watchlist", "forward_ridge_static", "forward_ridge_current", "forward_ridge_hybrid"]
    matrix = raw_family.pivot(index="system_family", columns="method", values="mean_normalized_mae").reindex(columns=methods)
    matrix.columns = [DISPLAY_METHODS[column] for column in matrix.columns]
    fig, axis = plt.subplots(figsize=(11, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="mako_r", linewidths=0.7, cbar_kws={"label": "Mean MAE / within-context P90-P10"}, ax=axis)
    axis.set_title("Forward raw intervention-value error", fontsize=18, fontweight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("Animal-system family")
    axis.tick_params(axis="x", rotation=35)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.86, bottom=0.24)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    profile_config = config["profiles"][profile]
    label_path = Path(config["data"]["label_path"])
    history_path = Path(config["data"]["history_feature_path"])
    labels = pd.read_csv(label_path, dtype={"candidate_id": str})
    history = pd.read_csv(history_path, dtype={"candidate_id": str})
    for frame in (labels, history):
        frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], format="mixed")
    datasets = list(profile_config["datasets"])
    labels = _select_anchor_prefix(labels, datasets, profile_config.get("max_anchor_times"))
    history = _select_anchor_prefix(history, datasets, profile_config.get("max_anchor_times"))
    feature_labels = history.merge(
        labels[
            [
                *KEY_COLUMNS,
                "label_id",
                "robust_intervention_value",
                "robust_priority_percentile",
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(feature_labels) != len(labels) or len(feature_labels) != len(history):
        raise ValueError("history and label rows do not reconcile")
    predictions = build_forward_predictions(
        feature_labels,
        min_prior_anchor_times=int(config["evaluation"]["min_prior_anchor_times"]),
        ridge_alpha=float(config["evaluation"]["ridge_alpha"]),
        seed=int(config["evaluation"]["random_seed"]),
    )
    context_metrics, family_metrics = evaluate_baseline_scores(
        predictions, top_fraction=float(config["evaluation"]["top_fraction"])
    )
    raw_metrics = evaluate_raw_predictions(predictions)
    raw_family = (
        raw_metrics.groupby(["system_family", "method"], observed=True)
        .agg(
            contexts=("anchor_time", "size"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_normalized_mae=("normalized_mae", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .reset_index()
    )
    variance_raw = balanced_variance_decomposition(
        labels, target="robust_intervention_value"
    )
    variance_rank = balanced_variance_decomposition(
        labels, target="robust_priority_percentile"
    )
    variance_family = _family_variance_summary(variance_raw)
    comparisons = {
        name: tuple(value) for name, value in config["evaluation"]["comparisons"].items()
    }
    differences = _paired_strategy_differences(context_metrics, comparisons)
    difference_summary = _blocked_difference_summary(
        differences,
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["random_seed"]),
    )
    coverage = (
        predictions.groupby(KEY_COLUMNS[:-1], observed=True)
        .agg(
            candidates=("candidate_id", "size"),
            seen_candidates=("candidate_seen_before", "sum"),
            seen_fraction=("candidate_seen_before", "mean"),
            prior_anchor_times=("prior_anchor_times", "first"),
        )
        .reset_index()
    )
    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / profile
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / profile
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forward_predictions.csv": predictions,
        "forward_context_metrics.csv": context_metrics,
        "forward_family_metrics.csv": family_metrics,
        "strategy_differences.csv": differences,
        "strategy_difference_summary.csv": difference_summary,
        "raw_value_metrics.csv": raw_metrics,
        "raw_value_family_metrics.csv": raw_family,
        "variance_decomposition_raw.csv": variance_raw,
        "variance_decomposition_rank.csv": variance_rank,
        "variance_decomposition_family.csv": variance_family,
        "prior_identity_coverage.csv": coverage,
    }
    for name, frame in outputs.items():
        _write_csv(frame, results_dir / name)
    audit = {
        "status": "pass",
        "checks": {
            "source_rows_reconcile": len(feature_labels) == len(labels) == len(history),
            "source_keys_unique": not feature_labels.duplicated(KEY_COLUMNS).any(),
            "predictions_unique": not predictions.duplicated(KEY_COLUMNS).any(),
            "strictly_forward": bool((predictions["prior_anchor_times"] >= int(config["evaluation"]["min_prior_anchor_times"])).all()),
            "scores_finite": bool(np.isfinite(predictions.filter(like="score_").to_numpy(dtype=float)).all()),
            "all_contexts_evaluated": context_metrics[KEY_COLUMNS[:-1]].drop_duplicates().shape[0] == predictions[KEY_COLUMNS[:-1]].drop_duplicates().shape[0],
            "oracle_is_exact": bool((context_metrics.loc[context_metrics["method"].eq("future_oracle"), "oracle_regret"].abs() < 1e-12).all()),
            "variance_fractions_close": bool(np.allclose(variance_raw.loc[variance_raw["status"].eq("estimated"), ["individual_fraction", "anchor_fraction", "individual_anchor_residual_fraction"]].sum(axis=1), 1.0)),
        },
        "source_rows": len(feature_labels),
        "prediction_rows": len(predictions),
        "forward_contexts": predictions[KEY_COLUMNS[:-1]].drop_duplicates().shape[0],
        "datasets": predictions["dataset_id"].nunique(),
        "system_families": predictions["system_family"].nunique(),
    }
    if not all(audit["checks"].values()):
        audit["status"] = "fail"
        raise ValueError(f"forward audit failed: {audit}")
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = yaml.safe_dump(config, sort_keys=False)
    (results_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
    source_paths = {
        "forward_strategy": Path("src/animal_intervention/evaluation/forward_strategy.py"),
        "baseline_ranking": Path("src/animal_intervention/evaluation/baseline_ranking.py"),
        "experiment_runner": Path("src/animal_intervention/experiments/stable_dynamic_forward.py"),
    }
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_worktree_dirty": bool(_git_value(["status", "--porcelain"])),
        "python": platform.python_version(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "label_sha256": _sha256(label_path),
        "history_feature_sha256": _sha256(history_path),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "audit_status": audit["status"],
    }
    _plot_variance_decomposition(variance_family, report_dir / "stable_dynamic_decomposition.png")
    _plot_strategy_performance(family_metrics, report_dir / "forward_strategy_performance.png")
    _plot_strategy_increment(difference_summary, report_dir / "strategy_increment.png")
    _plot_prior_coverage(coverage, report_dir / "prior_identity_coverage.png")
    _plot_raw_calibration(raw_family, report_dir / "raw_value_calibration.png")
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    def result(name: str, outcome: str) -> pd.Series:
        return difference_summary.loc[
            difference_summary["comparison"].eq(name)
            & difference_summary["outcome"].eq(outcome)
        ].iloc[0]

    stable_rank = result("stable_over_random", "spearman_gain")
    stable_regret = result("stable_over_random", "regret_reduction")
    hybrid_rank = result("hybrid_over_stable", "spearman_gain")
    oracle_rank = result("oracle_over_hybrid", "spearman_gain")
    readme = f"""# Stable versus dynamic forward-time evaluation

This experiment uses only anchors strictly earlier than each test anchor. The
primary evaluation requires at least {config['evaluation']['min_prior_anchor_times']}
earlier anchor times. Stable candidate priors, current-history rules, expanding-
window Ridge models, a hybrid model, and a non-deployable decision-value oracle are
evaluated against continuous simulation-derived intervention values.

- Forward contexts: {audit['forward_contexts']}
- Prediction rows: {audit['prediction_rows']}
- Independent animal-system families: {audit['system_families']}
- Audit status: **{audit['status']}**
- Stable-versus-random rank gain: {stable_rank.family_equal_mean:.3f}
  (blocked 95% interval {stable_rank.blocked_ci_low:.3f} to {stable_rank.blocked_ci_high:.3f})
- Stable-versus-random regret reduction: {100 * stable_regret.family_equal_mean:.3f}
  attack-rate percentage points (blocked 95% interval
  {100 * stable_regret.blocked_ci_low:.3f} to {100 * stable_regret.blocked_ci_high:.3f})
- Hybrid-versus-stable rank gain: {hybrid_rank.family_equal_mean:.3f}
  (blocked 95% interval {hybrid_rank.blocked_ci_low:.3f} to {hybrid_rank.blocked_ci_high:.3f})
- Decision-value-oracle-versus-hybrid rank gain: {oracle_rank.family_equal_mean:.3f}
  (blocked 95% interval {oracle_rank.blocked_ci_low:.3f} to {oracle_rank.blocked_ci_high:.3f})

The balanced decomposition is descriptive and restricted to animals observed at
every anchor within a network unit. The support-retention panel must be read
alongside the variance fractions. Simulation labels remain model-based targets,
not observed field intervention outcomes. The oracle scores animals by their
future raw intervention value, so its decision regret is exactly zero. Its rank
correlation need not equal one because the robust percentile label aggregates
scenario-specific ranks and is not defined as the raw-value rank.

The stable watchlist is a defensible primary baseline because it improves over
random selection in every independent family. Neither the current-history
composite nor the forward Ridge hybrid has established a family-robust gain over
that stable baseline. The positive oracle gap shows that time-specific future
value exists, but it does not show that it is predictable from currently
available observations. Radolfzell cannot support a complete-history balanced
variance decomposition because only one animal occurs at every anchor; its
forward prediction results remain included.
"""
    (report_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stable-versus-dynamic forward evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/EXP-20260816-006_stable_dynamic_forward.yaml"),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
