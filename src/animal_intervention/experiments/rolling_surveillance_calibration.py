"""Evaluate strictly historical within-system surveillance calibration."""

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
import yaml

from .effective_surveillance_dose import (
    _bootstrap_summary,
    _decision,
    _dose_curve,
    _fit_rate,
    _markdown_table,
    _sha256,
)


def _time_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("anchor_time", observed=True)["anchor_time"].transform("size")
    return (1.0 / counts.astype(float)).to_numpy()


def _time_equal_downstream_weights(frame: pd.DataFrame) -> np.ndarray:
    downstream = frame["downstream_cases"].astype(float)
    totals = downstream.groupby(frame["anchor_time"], observed=True).transform("sum")
    counts = frame.groupby("anchor_time", observed=True)["anchor_time"].transform("size")
    return np.where(totals.gt(0), downstream / totals, 1.0 / counts)


def _rolling_predictions(
    worlds: pd.DataFrame,
    models: list[str],
    rates: np.ndarray,
    minimum_prior_times: int,
    family_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_counts = worlds.groupby("system_family", observed=True)["anchor_time"].nunique()
    recurrent = sorted(time_counts.loc[time_counts.ge(minimum_prior_times + 1)].index)
    selected_families = recurrent if family_limit is None else recurrent[:family_limit]
    prediction_rows: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    boundary_rows = [
        {
            "system_family": family,
            "distinct_anchor_times": int(count),
            "self_calibratable": bool(count >= minimum_prior_times + 1),
        }
        for family, count in time_counts.sort_index().items()
    ]
    for model in models:
        for family in selected_families:
            family_worlds = worlds.loc[
                worlds["epidemic_model"].eq(model)
                & worlds["system_family"].eq(family)
            ].copy()
            times = sorted(family_worlds["anchor_time"].unique())
            for evaluation_time in times[minimum_prior_times:]:
                training = family_worlds.loc[family_worlds["anchor_time"].lt(evaluation_time)].copy()
                testing = family_worlds.loc[family_worlds["anchor_time"].eq(evaluation_time)].copy()
                prior_times = int(training["anchor_time"].nunique())
                if prior_times < minimum_prior_times or testing.empty:
                    raise ValueError(f"invalid rolling split for {family} at {evaluation_time}")
                for dose_column, dose_label in [
                    ("nominal_effective_dose", "nominal"),
                    ("network_effective_dose", "network"),
                ]:
                    detection_rate = _fit_rate(
                        training[dose_column].to_numpy(),
                        training["detection_success"].to_numpy(),
                        _time_equal_weights(training),
                        rates,
                        "log_loss",
                    )
                    burden_rate = _fit_rate(
                        training[dose_column].to_numpy(),
                        training["avoidable_downstream_fraction"].to_numpy(),
                        _time_equal_downstream_weights(training),
                        rates,
                        "absolute_error",
                    )
                    fold = testing.copy()
                    fold["calibration_scope"] = "rolling_local"
                    fold["dose_model"] = dose_label
                    fold["prior_anchor_times"] = prior_times
                    fold["latest_training_time"] = training["anchor_time"].max()
                    fold["predicted_detection"] = _dose_curve(
                        fold[dose_column].to_numpy(), detection_rate
                    )
                    fold["predicted_avoidable_fraction"] = _dose_curve(
                        fold[dose_column].to_numpy(), burden_rate
                    )
                    prediction_rows.append(fold)
                    fit_rows.append(
                        {
                            "epidemic_model": model,
                            "system_family": family,
                            "evaluation_time": evaluation_time,
                            "dose_model": dose_label,
                            "prior_anchor_times": prior_times,
                            "latest_training_time": training["anchor_time"].max(),
                            "detection_rate": detection_rate,
                            "burden_rate": burden_rate,
                            "training_rows": len(training),
                            "evaluation_rows": len(testing),
                        }
                    )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(fit_rows), pd.DataFrame(boundary_rows)


def _evaluation_keys() -> list[str]:
    return [
        "dataset_id",
        "network_id",
        "anchor_time",
        "epidemic_model",
        "random_block",
        "initial_infected",
        "world_seed",
        "sentinel_fraction",
        "recognition_sensitivity",
        "dose_model",
    ]


def _align_universal(
    universal: pd.DataFrame, rolling: pd.DataFrame
) -> pd.DataFrame:
    keys = _evaluation_keys()
    evaluation_keys = rolling[keys].drop_duplicates()
    aligned = universal.merge(evaluation_keys, on=keys, how="inner", validate="one_to_one")
    aligned["calibration_scope"] = "universal_loso"
    aligned["prior_anchor_times"] = 0
    aligned["latest_training_time"] = pd.NaT
    return aligned


def _method_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["epidemic_model", "system_family", "calibration_scope", "dose_model"]
    for key, frame in predictions.groupby(keys, observed=True, sort=True):
        model, family, scope, dose = key
        brier = float(
            np.mean((frame["detection_success"] - frame["predicted_detection"]) ** 2)
        )
        weights = frame["downstream_cases"].to_numpy(dtype=float)
        absolute = np.abs(
            frame["avoidable_downstream_fraction"]
            - frame["predicted_avoidable_fraction"]
        )
        burden_mae = float(np.average(absolute, weights=weights)) if weights.sum() else float(absolute.mean())
        method = f"{'rolling' if scope == 'rolling_local' else 'universal'}_{dose}"
        rows.extend(
            [
                {
                    "epidemic_model": model,
                    "system_family": family,
                    "method": method,
                    "endpoint": "detection_brier",
                    "loss": brier,
                },
                {
                    "epidemic_model": model,
                    "system_family": family,
                    "method": method,
                    "endpoint": "avoidable_burden_mae",
                    "loss": burden_mae,
                },
            ]
        )
    return pd.DataFrame(rows)


def _comparison(metrics: pd.DataFrame, comparison: str) -> pd.DataFrame:
    wide = metrics.pivot(
        index=["epidemic_model", "system_family", "endpoint"],
        columns="method",
        values="loss",
    ).reset_index()
    if comparison == "primary":
        wide["improvement"] = wide["universal_nominal"] - wide["rolling_nominal"]
        wide["comparison"] = "universal_nominal_minus_rolling_nominal"
    elif comparison == "secondary":
        wide["improvement"] = wide["rolling_nominal"] - wide["rolling_network"]
        wide["comparison"] = "rolling_nominal_minus_rolling_network"
    else:
        raise ValueError(f"unsupported comparison: {comparison}")
    return wide


def _plot_primary(primary: pd.DataFrame, path: Path, dpi: int) -> None:
    combinations = [
        (model, endpoint)
        for model in ["temporal_sir", "temporal_seir_erlang"]
        for endpoint in ["detection_brier", "avoidable_burden_mae"]
    ]
    families = sorted(primary["system_family"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.3), sharey=True)
    for axis, (model, endpoint) in zip(axes, combinations, strict=True):
        frame = primary.loc[
            primary["epidemic_model"].eq(model) & primary["endpoint"].eq(endpoint)
        ].set_index("system_family").reindex(families)
        colors = np.where(frame["improvement"].ge(0), "#4C78A8", "#E45756")
        axis.barh(range(len(frame)), frame["improvement"], color=colors)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_yticks(range(len(frame)), [name.replace("_", " ") for name in families])
        axis.set_xlabel("Error reduction")
        axis.set_title(
            ("SIR" if model == "temporal_sir" else "SEIR")
            + "\n"
            + ("Detection" if endpoint == "detection_brier" else "Avoidable burden")
        )
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Does rolling local calibration beat universal transfer?", fontsize=18, weight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_curve(predictions: pd.DataFrame, path: Path, dpi: int) -> None:
    rolling = predictions.loc[
        predictions["calibration_scope"].eq("rolling_local")
        & predictions["dose_model"].eq("nominal")
    ].copy()
    grouped = rolling.groupby(
        ["system_family", "epidemic_model", "prior_anchor_times"], observed=True
    ).agg(
        detection_brier=(
            "predicted_detection",
            lambda values: float(
                np.mean(
                    (
                        rolling.loc[values.index, "detection_success"].to_numpy()
                        - values.to_numpy()
                    )
                    ** 2
                )
            ),
        ),
        calibration_rows=("predicted_detection", "size"),
    ).reset_index()
    families = sorted(grouped["system_family"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for axis, family in zip(axes.ravel(), families, strict=False):
        family_frame = grouped.loc[grouped["system_family"].eq(family)]
        for model, frame in family_frame.groupby("epidemic_model", observed=True):
            frame = frame.sort_values("prior_anchor_times")
            axis.plot(
                frame["prior_anchor_times"],
                frame["detection_brier"],
                marker="o",
                label="Temporal SIR" if model == "temporal_sir" else "Temporal SEIR",
            )
        axis.set_title(family.replace("_", " "), fontsize=11)
        axis.set_xlabel("Earlier anchor times used")
        axis.set_ylabel("Forward detection Brier score")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(families):]:
        axis.set_visible(False)
    axes.ravel()[0].legend(frameon=False)
    fig.suptitle(
        "Rolling calibration remains animal-system specific",
        fontsize=17,
        weight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=2.0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profile = config["profiles"][profile_name]
    experiment_id = config["experiment"]["id"]
    results_dir = root / config["outputs"]["results_root"] / experiment_id / profile_name
    report_dir = root / config["outputs"]["report_root"] / experiment_id / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prerequisite_path = root / config["data"]["prerequisite_audit"]
    prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    worlds_path = root / config["data"]["effective_dose_worlds"]
    universal_path = root / config["data"]["universal_loso_predictions"]
    worlds = pd.read_csv(worlds_path, dtype={"initial_infected": str})
    universal = pd.read_csv(universal_path, dtype={"initial_infected": str})
    for frame in [worlds, universal]:
        frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], format="mixed")
    models = list(profile["epidemic_models"])
    rate_config = config["design"]["parameter_grid"]
    rates = np.geomspace(
        float(rate_config["minimum"]),
        float(rate_config["maximum"]),
        int(rate_config["points"]),
    )
    rolling, fits, boundary = _rolling_predictions(
        worlds,
        models,
        rates,
        int(config["design"]["minimum_prior_anchor_times"]),
        profile["recurrent_families"],
    )
    aligned_universal = _align_universal(universal, rolling)
    predictions = pd.concat([rolling, aligned_universal], ignore_index=True, sort=False)
    metrics = _method_metrics(predictions)
    primary = _comparison(metrics, "primary")
    secondary = _comparison(metrics, "secondary")
    primary_summary = _bootstrap_summary(
        primary.rename(columns={"system_family": "heldout_family"}),
        int(profile["bootstrap_replicates"]),
        int(config["evaluation"]["seed"]),
    )
    secondary_summary = _bootstrap_summary(
        secondary.rename(columns={"system_family": "heldout_family"}),
        int(profile["bootstrap_replicates"]),
        int(config["evaluation"]["seed"]) + 1,
    )
    recurrent_count = int(primary["system_family"].nunique())
    decision = _decision(primary_summary, recurrent_count) if recurrent_count == 4 else "smoke_only"

    chronological = bool(
        fits["latest_training_time"].lt(fits["evaluation_time"]).all()
    )
    audit = {
        "status": "pass",
        "profile": profile_name,
        "datasets": int(rolling["dataset_id"].nunique()),
        "recurrent_families": recurrent_count,
        "zero_history_families": int((~boundary["self_calibratable"]).sum()),
        "evaluation_anchor_times": int(rolling[["system_family", "anchor_time"]].drop_duplicates().shape[0]),
        "rolling_prediction_rows": int(len(rolling)),
        "universal_prediction_rows": int(len(aligned_universal)),
        "prerequisite_passed": prerequisite.get("status") == "pass",
        "strictly_chronological": chronological,
        "minimum_prior_times_respected": bool(
            rolling["prior_anchor_times"].ge(int(config["design"]["minimum_prior_anchor_times"])).all()
        ),
        "universal_alignment_complete": bool(len(aligned_universal) == len(rolling)),
        "all_predictions_finite": bool(
            np.isfinite(predictions[["predicted_detection", "predicted_avoidable_fraction"]]).all().all()
        ),
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    checks = [
        "prerequisite_passed",
        "strictly_chronological",
        "minimum_prior_times_respected",
        "universal_alignment_complete",
        "all_predictions_finite",
    ]
    if not all(audit[key] for key in checks):
        audit["status"] = "fail"
        raise ValueError(f"rolling calibration audit failed: {audit}")

    rolling.to_csv(results_dir / "rolling_predictions.csv.gz", index=False, compression="gzip")
    aligned_universal.to_csv(results_dir / "aligned_universal_predictions.csv.gz", index=False, compression="gzip")
    fits.to_csv(results_dir / "rolling_curve_fits.csv", index=False)
    boundary.to_csv(results_dir / "calibration_boundary.csv", index=False)
    metrics.to_csv(results_dir / "method_metrics.csv", index=False)
    primary.to_csv(results_dir / "primary_improvements.csv", index=False)
    secondary.to_csv(results_dir / "secondary_exposure_improvements.csv", index=False)
    primary_summary.to_csv(results_dir / "primary_summary.csv", index=False)
    secondary_summary.to_csv(results_dir / "secondary_summary.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = {**config, "runtime": {"profile": profile_name, "timestamp_utc": datetime.now(UTC).isoformat()}}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id,
        "profile": profile_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_hashes": {
            str(config["data"]["effective_dose_worlds"]): _sha256(worlds_path),
            str(config["data"]["universal_loso_predictions"]): _sha256(universal_path),
            str(config_path.relative_to(root)): _sha256(config_path),
        },
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dpi = int(profile["render_dpi"])
    _plot_primary(primary, report_dir / "local_calibration_improvement.png", dpi)
    _plot_learning_curve(predictions, report_dir / "rolling_learning_curve.png", dpi)
    report = [
        "# Rolling within-system surveillance calibration",
        "",
        "This locked follow-up tests whether strictly earlier simulated contact windows from the same observed animal system can calibrate a forward early-detection curve. It does not use the current or future evaluation window for fitting.",
        "",
        f"- Decision tier: **{decision}**",
        f"- Recurrent animal-system families: {recurrent_count}",
        f"- Evaluation anchor times: {audit['evaluation_anchor_times']}",
        f"- Technical audit: **{audit['status']}**",
        "",
        "## Primary: universal nominal dose minus rolling local nominal dose",
        "",
        _markdown_table(primary_summary),
        "",
        "## Secondary: rolling nominal dose minus rolling exposure-adjusted dose",
        "",
        _markdown_table(secondary_summary),
        "",
        "Positive values indicate lower forward prediction error for the second method named in each comparison. The calibration targets remain simulator-derived and do not replace field validation.",
    ]
    (report_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.profile), indent=2))


if __name__ == "__main__":
    main()
