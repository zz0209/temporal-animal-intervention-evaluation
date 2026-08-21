"""Evaluate an exposure-adjusted effective dose for animal disease surveillance."""

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
import yaml


CONTEXT_KEYS = ["dataset_id", "network_id", "anchor_time"]
WORLD_KEYS = [
    "dataset_id",
    "network_id",
    "anchor_time",
    "epidemic_model",
    "random_block",
    "initial_infected",
    "world_seed",
    "sentinel_fraction",
    "recognition_sensitivity",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_nodes(value: Any) -> tuple[str, ...]:
    if pd.isna(value) or str(value).strip() == "":
        return ()
    return tuple(node for node in str(value).split("|") if node)


def _dose_curve(dose: np.ndarray, rate: float) -> np.ndarray:
    return 1.0 - np.exp(-float(rate) * np.asarray(dose, dtype=float))


def _fit_rate(
    dose: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    rates: np.ndarray,
    loss: str,
) -> float:
    """Fit a monotone one-parameter saturation curve on a frozen grid."""
    dose = np.asarray(dose, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(dose) == 0 or weights.sum() <= 0:
        raise ValueError("dose-response fitting requires positive weighted observations")
    weights = weights / weights.sum()
    predictions = 1.0 - np.exp(-np.outer(rates, dose))
    if loss == "log_loss":
        clipped = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        losses = -np.sum(
            weights * (outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)),
            axis=1,
        )
    elif loss == "absolute_error":
        losses = np.sum(weights * np.abs(predictions - outcome), axis=1)
    else:
        raise ValueError(f"unsupported fitting loss: {loss}")
    return float(rates[int(np.argmin(losses))])


def _family_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("system_family", observed=True)["system_family"].transform("size")
    return (1.0 / counts.astype(float)).to_numpy()


def _downstream_case_weights(frame: pd.DataFrame) -> np.ndarray:
    raw = frame["downstream_cases"].astype(float)
    family_totals = raw.groupby(frame["system_family"], observed=True).transform("sum")
    fallback_counts = frame.groupby("system_family", observed=True)["system_family"].transform("size")
    return np.where(family_totals.gt(0), raw / family_totals, 1.0 / fallback_counts)


def _load_analysis_table(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    worlds_path = root / config["data"]["policy_worlds"]
    features_path = root / config["data"]["history_features"]
    worlds = pd.read_csv(worlds_path, dtype={"sentinel_nodes": str, "initial_infected": str})
    features = pd.read_csv(features_path, dtype={"candidate_id": str})
    worlds["anchor_time"] = pd.to_datetime(worlds["anchor_time"], format="mixed")
    features["anchor_time"] = pd.to_datetime(features["anchor_time"], format="mixed")

    response_fraction = float(config["data"]["response_fraction_for_detection_audit"])
    selected = worlds.loc[worlds["response_fraction"].eq(response_fraction)].copy()
    duplicate_counts = selected.groupby(WORLD_KEYS, observed=True).size()
    selected = selected.drop_duplicates(WORLD_KEYS, keep="first").copy()

    exposure_column = str(config["data"]["history_exposure_column"])
    feature_lookup = {
        (dataset_id, str(network_id), anchor_time, str(candidate_id)): float(exposure)
        for dataset_id, network_id, anchor_time, candidate_id, exposure in features[
            CONTEXT_KEYS + ["candidate_id", exposure_column]
        ].itertuples(index=False, name=None)
    }
    totals = (
        features.groupby(CONTEXT_KEYS, observed=True, as_index=False)
        .agg(
            total_history_exposure=(exposure_column, "sum"),
            feature_candidate_count=("candidate_id", "nunique"),
        )
    )
    selected = selected.merge(totals, on=CONTEXT_KEYS, how="left", validate="many_to_one")
    selected["sentinel_history_exposure"] = [
        sum(
            feature_lookup.get(
                (row.dataset_id, str(row.network_id), row.anchor_time, node), np.nan
            )
            for node in _parse_nodes(row.sentinel_nodes)
        )
        for row in selected.itertuples(index=False)
    ]
    selected["nominal_coverage"] = selected["sentinel_budget"] / selected["eligible_size"]
    selected["history_exposure_coverage"] = (
        selected["sentinel_history_exposure"] / selected["total_history_exposure"]
    )
    selected["nominal_effective_dose"] = (
        selected["nominal_coverage"] * selected["recognition_sensitivity"]
    )
    selected["network_effective_dose"] = (
        selected["history_exposure_coverage"] * selected["recognition_sensitivity"]
    )
    selected["detection_success"] = selected["detected"].astype(float)
    selected["downstream_cases"] = (selected["natural_final_size"] - 1).clip(lower=0)
    selected["avoidable_downstream_fraction"] = np.where(
        selected["downstream_cases"].gt(0),
        (selected["natural_final_size"] - selected["detection_burden"])
        / selected["downstream_cases"],
        0.0,
    )
    selected["avoidable_downstream_fraction"] = selected[
        "avoidable_downstream_fraction"
    ].clip(0.0, 1.0)

    audit = {
        "source_world_rows": int(len(worlds)),
        "selected_rows": int(len(selected)),
        "response_arm_duplicate_min": int(duplicate_counts.min()),
        "response_arm_duplicate_max": int(duplicate_counts.max()),
        "missing_context_features": int(selected["total_history_exposure"].isna().sum()),
        "missing_sentinel_exposure": int(selected["sentinel_history_exposure"].isna().sum()),
        "candidate_count_mismatch": int(
            selected.loc[
                selected["feature_candidate_count"].ne(selected["eligible_size"]),
                CONTEXT_KEYS,
            ].drop_duplicates().shape[0]
        ),
    }
    return selected, audit


def _cross_validate(
    table: pd.DataFrame,
    models: list[str],
    rates: np.ndarray,
    family_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_families = sorted(table["system_family"].unique())
    heldout_families = all_families if family_limit is None else all_families[:family_limit]
    prediction_rows: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    for model in models:
        model_table = table.loc[table["epidemic_model"].eq(model)].copy()
        for heldout in heldout_families:
            training = model_table.loc[model_table["system_family"].ne(heldout)].copy()
            testing = model_table.loc[model_table["system_family"].eq(heldout)].copy()
            training_families = sorted(training["system_family"].unique())
            if len(testing) == 0 or len(training_families) != len(all_families) - 1:
                raise ValueError(f"invalid held-out fold for {model}: {heldout}")
            for dose_name in ["nominal_effective_dose", "network_effective_dose"]:
                detection_rate = _fit_rate(
                    training[dose_name].to_numpy(),
                    training["detection_success"].to_numpy(),
                    _family_equal_weights(training),
                    rates,
                    "log_loss",
                )
                burden_rate = _fit_rate(
                    training[dose_name].to_numpy(),
                    training["avoidable_downstream_fraction"].to_numpy(),
                    _downstream_case_weights(training),
                    rates,
                    "absolute_error",
                )
                fold = testing.copy()
                fold["heldout_family"] = heldout
                fold["dose_model"] = dose_name.replace("_effective_dose", "")
                fold["predicted_detection"] = _dose_curve(
                    fold[dose_name].to_numpy(), detection_rate
                )
                fold["predicted_avoidable_fraction"] = _dose_curve(
                    fold[dose_name].to_numpy(), burden_rate
                )
                prediction_rows.append(fold)
                fit_rows.append(
                    {
                        "epidemic_model": model,
                        "heldout_family": heldout,
                        "dose_model": dose_name.replace("_effective_dose", ""),
                        "detection_rate": detection_rate,
                        "burden_rate": burden_rate,
                        "training_families": "|".join(training_families),
                        "training_rows": len(training),
                        "heldout_rows": len(testing),
                    }
                )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(fit_rows)


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["epidemic_model", "heldout_family", "dose_model"]
    for key, frame in predictions.groupby(keys, observed=True, sort=True):
        model, heldout, dose_model = key
        brier = float(
            np.mean((frame["detection_success"] - frame["predicted_detection"]) ** 2)
        )
        weights = frame["downstream_cases"].to_numpy(dtype=float)
        if weights.sum() > 0:
            burden_error = float(
                np.average(
                    np.abs(
                        frame["avoidable_downstream_fraction"]
                        - frame["predicted_avoidable_fraction"]
                    ),
                    weights=weights,
                )
            )
        else:
            burden_error = float(
                np.mean(
                    np.abs(
                        frame["avoidable_downstream_fraction"]
                        - frame["predicted_avoidable_fraction"]
                    )
                )
            )
        rows.extend(
            [
                {
                    "epidemic_model": model,
                    "heldout_family": heldout,
                    "dose_model": dose_model,
                    "endpoint": "detection_brier",
                    "loss": brier,
                },
                {
                    "epidemic_model": model,
                    "heldout_family": heldout,
                    "dose_model": dose_model,
                    "endpoint": "avoidable_burden_mae",
                    "loss": burden_error,
                },
            ]
        )
    return pd.DataFrame(rows)


def _paired_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot(
        index=["epidemic_model", "heldout_family", "endpoint"],
        columns="dose_model",
        values="loss",
    ).reset_index()
    if not {"nominal", "network"}.issubset(wide.columns):
        raise ValueError("both dose models are required for paired comparison")
    wide["improvement"] = wide["nominal"] - wide["network"]
    return wide


def _bootstrap_summary(
    improvements: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (model, endpoint), frame in improvements.groupby(
        ["epidemic_model", "endpoint"], observed=True, sort=True
    ):
        values = frame["improvement"].to_numpy(dtype=float)
        draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                "epidemic_model": model,
                "endpoint": endpoint,
                "families": len(values),
                "family_equal_mean_improvement": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "positive_families": int(np.sum(values > 0)),
                "zero_families": int(np.sum(values == 0)),
                "bootstrap_probability_positive": float(np.mean(draws > 0)),
            }
        )
    return pd.DataFrame(rows)


def _decision(summary: pd.DataFrame, expected_families: int) -> str:
    strong = bool(
        summary["ci_low"].gt(0).all()
        and summary["positive_families"].ge(max(4, expected_families - 1)).all()
    )
    directional = bool(
        summary["family_equal_mean_improvement"].gt(0).all()
        and summary["positive_families"].ge(3).all()
    )
    return "strong" if strong else "directional" if directional else "unsupported"


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _plot_coverage_geometry(table: pd.DataFrame, path: Path, dpi: int) -> None:
    plot = (
        table.groupby(["system_family", "sentinel_fraction"], observed=True, as_index=False)
        .agg(
            nominal_coverage=("nominal_coverage", "mean"),
            history_exposure_coverage=("history_exposure_coverage", "mean"),
            sentinel_budget=("sentinel_budget", "median"),
        )
    )
    families = sorted(plot["system_family"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    fig, axis = plt.subplots(figsize=(10.5, 7.2))
    for family, color in zip(families, colors, strict=True):
        frame = plot.loc[plot["system_family"].eq(family)].sort_values("sentinel_fraction")
        axis.plot(
            frame["nominal_coverage"],
            frame["history_exposure_coverage"],
            marker="o",
            linewidth=2,
            label=family.replace("_", " "),
            color=color,
        )
        annotation_groups = frame.assign(
            nominal_key=frame["nominal_coverage"].round(8),
            exposure_key=frame["history_exposure_coverage"].round(8),
        ).groupby(["nominal_key", "exposure_key"], observed=True, sort=False)
        for (_, _), group in annotation_groups:
            fractions = sorted(round(100 * value) for value in group["sentinel_fraction"])
            row = group.iloc[0]
            if len(fractions) == 1:
                label = f"{fractions[0]}%"
            else:
                animal_word = "animal" if round(row["sentinel_budget"]) == 1 else "animals"
                label = (
                    f"{fractions[0]}–{fractions[-1]}% → "
                    f"{round(row['sentinel_budget'])} {animal_word}"
                )
            axis.annotate(
                label,
                (row["nominal_coverage"], row["history_exposure_coverage"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    limit = max(plot["nominal_coverage"].max(), plot["history_exposure_coverage"].max()) * 1.08
    axis.plot([0, limit], [0, limit], linestyle="--", color="#777777", linewidth=1)
    axis.set(xlabel="Realized fraction of animals monitored", ylabel="Historical exposure share covered")
    axis.set_title("The same nominal monitoring fraction does not mean the same exposure coverage", loc="left", weight="bold")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_improvements(improvements: pd.DataFrame, path: Path, dpi: int) -> None:
    model_labels = {"temporal_sir": "Temporal SIR", "temporal_seir_erlang": "Temporal SEIR"}
    endpoint_labels = {
        "detection_brier": "Detection prediction (Brier)",
        "avoidable_burden_mae": "Avoidable-burden prediction (MAE)",
    }
    combinations = [
        (model, endpoint)
        for model in ["temporal_sir", "temporal_seir_erlang"]
        for endpoint in ["detection_brier", "avoidable_burden_mae"]
        if ((improvements["epidemic_model"].eq(model)) & (improvements["endpoint"].eq(endpoint))).any()
    ]
    fig, axes = plt.subplots(1, len(combinations), figsize=(5.0 * len(combinations), 6.5), sharey=True)
    axes = np.atleast_1d(axes)
    family_order = sorted(improvements["heldout_family"].unique())
    for axis, (model, endpoint) in zip(axes, combinations, strict=True):
        frame = improvements.loc[
            improvements["epidemic_model"].eq(model) & improvements["endpoint"].eq(endpoint)
        ].set_index("heldout_family").reindex(family_order)
        colors = np.where(frame["improvement"].ge(0), "#4C78A8", "#E45756")
        axis.barh(np.arange(len(frame)), frame["improvement"], color=colors)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_title(f"{model_labels.get(model, model)}\n{endpoint_labels[endpoint]}", fontsize=11)
        axis.set_xlabel("Error reduction: nominal minus exposure-adjusted")
        axis.grid(axis="x", alpha=0.2)
        axis.set_yticks(np.arange(len(frame)), [value.replace("_", " ") for value in family_order])
    fig.suptitle("Leave-one-animal-system-out predictive improvement", fontsize=17, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_dose_response(predictions: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex="col", sharey="row")
    endpoint_specs = [
        ("detection_success", "predicted_detection", "Detection probability"),
        ("avoidable_downstream_fraction", "predicted_avoidable_fraction", "Avoidable downstream fraction"),
    ]
    models = ["temporal_sir", "temporal_seir_erlang"]
    for row_index, (outcome, prediction, label) in enumerate(endpoint_specs):
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            frame = predictions.loc[
                predictions["epidemic_model"].eq(model)
                & predictions["dose_model"].eq("network")
            ].copy()
            frame["bin"] = pd.qcut(frame["network_effective_dose"], q=8, duplicates="drop")
            observed = frame.groupby("bin", observed=True).agg(
                dose=("network_effective_dose", "mean"),
                observed=(outcome, "mean"),
                predicted=(prediction, "mean"),
            )
            axis.plot(observed["dose"], observed["observed"], marker="o", label="Observed", color="#4C78A8")
            axis.plot(observed["dose"], observed["predicted"], marker="s", label="LOFO prediction", color="#F58518")
            axis.set_title("Temporal SIR" if model == "temporal_sir" else "Temporal SEIR")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            axis.set_ylim(-0.03, 1.03)
            if row_index == 1:
                axis.set_xlabel("Exposure-adjusted effective surveillance dose")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    fig.suptitle("Out-of-system calibration of effective surveillance dose", fontsize=18, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=2.0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][profile_name]
    experiment_id = str(config["experiment"]["id"])
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile_name
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prerequisite_path = root / config["data"]["prerequisite_audit"]
    prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    table, load_audit = _load_analysis_table(config, root)
    models = list(profile["epidemic_models"])
    table = table.loc[table["epidemic_model"].isin(models)].copy()
    rate_config = config["evaluation"]["curve_parameter_grid"]
    rates = np.geomspace(
        float(rate_config["minimum"]),
        float(rate_config["maximum"]),
        int(rate_config["points"]),
    )
    predictions, fits = _cross_validate(
        table,
        models,
        rates,
        profile["heldout_families"],
    )
    metrics = _fold_metrics(predictions)
    improvements = _paired_improvements(metrics)
    summary = _bootstrap_summary(
        improvements,
        int(profile["bootstrap_replicates"]),
        int(config["evaluation"]["seed"]),
    )
    expected_families = int(improvements["heldout_family"].nunique())
    decision = _decision(summary, expected_families) if expected_families == 5 else "smoke_only"

    monotone = (
        table.groupby(
            WORLD_KEYS[:-2] + ["recognition_sensitivity"], observed=True
        )["history_exposure_coverage"]
        .apply(lambda series: bool(np.all(np.diff(np.sort(series.unique())) >= -1e-12)))
        .all()
    )
    audit = {
        **load_audit,
        "status": "pass",
        "profile": profile_name,
        "datasets": int(table["dataset_id"].nunique()),
        "animal_system_families": int(table["system_family"].nunique()),
        "anchors": int(table[CONTEXT_KEYS].drop_duplicates().shape[0]),
        "epidemic_models": int(table["epidemic_model"].nunique()),
        "analysis_rows": int(len(table)),
        "heldout_folds": expected_families,
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "feature_join_complete": bool(
            load_audit["missing_context_features"] == 0
            and load_audit["missing_sentinel_exposure"] == 0
            and load_audit["candidate_count_mismatch"] == 0
        ),
        "coverage_bounded": bool(
            table["nominal_coverage"].between(0, 1).all()
            and table["history_exposure_coverage"].between(0, 1).all()
        ),
        "avoidable_fraction_bounded": bool(table["avoidable_downstream_fraction"].between(0, 1).all()),
        "sentinel_exposure_nested": bool(monotone),
        "all_predictions_finite": bool(
            np.isfinite(predictions[["predicted_detection", "predicted_avoidable_fraction"]]).all().all()
        ),
        "heldout_family_absent_from_training": bool(
            fits.apply(lambda row: row.heldout_family not in row.training_families.split("|"), axis=1).all()
        ),
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    required_checks = [
        "prerequisite_passed",
        "feature_join_complete",
        "coverage_bounded",
        "avoidable_fraction_bounded",
        "sentinel_exposure_nested",
        "all_predictions_finite",
        "heldout_family_absent_from_training",
    ]
    if not all(audit[key] for key in required_checks):
        audit["status"] = "fail"
        raise ValueError(f"effective-dose audit failed: {audit}")

    table.to_csv(results_dir / "effective_dose_worlds.csv.gz", index=False, compression="gzip")
    predictions.to_csv(results_dir / "loso_predictions.csv.gz", index=False, compression="gzip")
    fits.to_csv(results_dir / "fold_curve_fits.csv", index=False)
    metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    improvements.to_csv(results_dir / "paired_predictive_improvements.csv", index=False)
    summary.to_csv(results_dir / "predictive_improvement_summary.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = {**config, "runtime": {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_hashes": {
            str(config["data"]["policy_worlds"]): _sha256(root / config["data"]["policy_worlds"]),
            str(config["data"]["history_features"]): _sha256(root / config["data"]["history_features"]),
            str(config_path.relative_to(root)): _sha256(config_path),
        },
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dpi = int(profile["render_dpi"])
    _plot_coverage_geometry(table, report_dir / "coverage_geometry.png", dpi)
    _plot_improvements(improvements, report_dir / "loso_predictive_improvement.png", dpi)
    _plot_dose_response(predictions, report_dir / "dose_response_calibration.png", dpi)
    report = [
        "# Effective surveillance dose",
        "",
        "This frozen reanalysis asks whether the product of recognition sensitivity and deployment-visible historical exposure coverage transports early-detection performance across unseen animal systems better than the classical product based on the nominal monitored fraction.",
        "",
        f"- Decision tier: **{decision}**",
        f"- Datasets: {audit['datasets']}",
        f"- Independent animal-system families: {audit['animal_system_families']}",
        f"- Anchors: {audit['anchors']}",
        f"- Analysis rows: {audit['analysis_rows']}",
        f"- Technical audit: **{audit['status']}**",
        "",
        "## Leave-one-system-out comparison",
        "",
        _markdown_table(summary),
        "",
        "Positive improvement means that exposure-adjusted dose reduced prediction error relative to nominal coverage. This is a simulator-conditional calibration result, not a field estimate of diagnostic sensitivity or causal intervention efficacy.",
    ]
    (report_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    audit = run(args.config, args.profile)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
