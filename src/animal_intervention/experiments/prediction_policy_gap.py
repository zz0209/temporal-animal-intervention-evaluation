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

from .outbreak_response_pilot import _git_value, _sha256


def _load_evidence(config: dict[str, Any]) -> pd.DataFrame:
    prediction = pd.read_csv(config["data"]["prediction_summary"])
    prediction = prediction.loc[
        prediction["comparison"].eq("stable_over_random")
        & prediction["outcome"].isin(
            ["spearman_gain", "value_capture_gain", "regret_reduction"]
        )
    ].copy()
    prediction["stage"] = "singleton_forecast"
    prediction["contrast"] = prediction["outcome"]
    prediction["epidemic_model"] = "not_applicable"
    prediction = prediction.rename(
        columns={
            "family_equal_mean": "estimate",
            "blocked_ci_low": "ci_low",
            "blocked_ci_high": "ci_high",
        }
    )
    for path_key, stage in [
        ("endogenous_response_summary", "endogenous_response"),
        ("immediate_response_summary", "immediate_response"),
    ]:
        frame = pd.read_csv(config["data"][path_key])
        frame = frame.loc[
            frame["contrast"].isin(["capacity_increment", "targeting_increment"])
        ].copy()
        frame["stage"] = stage
        frame = frame.rename(
            columns={
                "family_equal_mean": "estimate",
                "ci_low": "ci_low",
                "ci_high": "ci_high",
            }
        )
        prediction = pd.concat(
            [
                prediction,
                frame[
                    [
                        "stage",
                        "contrast",
                        "epidemic_model",
                        "estimate",
                        "ci_low",
                        "ci_high",
                    ]
                ],
            ],
            ignore_index=True,
        )
    return prediction[
        ["stage", "contrast", "epidemic_model", "estimate", "ci_low", "ci_high"]
    ]


def _forest(
    axis: Any,
    frame: pd.DataFrame,
    labels: list[str],
    title: str,
    scale: float,
    xlabel: str,
) -> None:
    y = np.arange(len(frame))
    mean = scale * frame["estimate"].to_numpy(float)
    low = scale * frame["ci_low"].to_numpy(float)
    high = scale * frame["ci_high"].to_numpy(float)
    colors = np.where(low > 0, "#2A9D8F", "#4C78A8")
    axis.errorbar(
        mean,
        y,
        xerr=np.vstack((mean - low, high - mean)),
        fmt="none",
        ecolor="#777777",
        capsize=4,
    )
    axis.scatter(mean, y, s=75, color=colors, zorder=3)
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1.2)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_title(title, fontsize=16, weight="bold")
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=0.25)


def _plot(evidence: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2))
    forecast_order = ["spearman_gain", "value_capture_gain", "regret_reduction"]
    forecast = (
        evidence.loc[evidence["stage"].eq("singleton_forecast")]
        .set_index("contrast")
        .loc[forecast_order]
        .reset_index()
    )
    forecast.loc[forecast["contrast"].eq("regret_reduction"), ["estimate", "ci_low", "ci_high"]] *= 100
    _forest(
        axes[0],
        forecast,
        ["Rank correlation gain", "Top-set value capture gain", "Oracle-regret reduction (points)"],
        "A. Individual-value prediction",
        1.0,
        "Stable history over random",
    )
    order = [
        ("immediate_response", "temporal_sir"),
        ("immediate_response", "temporal_seir_erlang"),
        ("endogenous_response", "temporal_sir"),
        ("endogenous_response", "temporal_seir_erlang"),
    ]
    labels = ["Immediate · SIR", "Immediate · SEIR", "Endogenous · SIR", "Endogenous · SEIR"]
    for axis, contrast, title in [
        (axes[1], "targeting_increment", "B. Which animals?"),
        (axes[2], "capacity_increment", "C. How many animals?"),
    ]:
        selected = evidence.loc[evidence["contrast"].eq(contrast)].set_index(
            ["stage", "epidemic_model"]
        ).loc[order].reset_index()
        _forest(
            axis,
            selected,
            labels,
            title,
            100.0,
            "Avoided attack-rate percentage points",
        )
    fig.suptitle(
        "The prediction-to-policy gap in temporal animal networks",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.5,
        0.04,
        "Green: interval excludes zero. Predicting singleton importance does not guarantee robust equal-capacity set targeting.",
        ha="center",
        fontsize=12,
        color="#555555",
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.84, bottom=0.14, wspace=0.48)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence = _load_evidence(config)
    expected = {
        ("singleton_forecast", "spearman_gain"),
        ("singleton_forecast", "value_capture_gain"),
        ("singleton_forecast", "regret_reduction"),
        ("endogenous_response", "capacity_increment"),
        ("endogenous_response", "targeting_increment"),
        ("immediate_response", "capacity_increment"),
        ("immediate_response", "targeting_increment"),
    }
    observed = set(evidence[["stage", "contrast"]].itertuples(index=False, name=None))
    checks = {
        "all_evidence_cells_present": observed == expected,
        "finite_evidence": bool(
            np.isfinite(evidence[["estimate", "ci_low", "ci_high"]]).all().all()
        ),
        "intervals_ordered": bool(
            (evidence["ci_low"] <= evidence["estimate"]).all()
            and (evidence["estimate"] <= evidence["ci_high"]).all()
        ),
        "forecast_signal_positive": bool(
            evidence.loc[evidence["stage"].eq("singleton_forecast"), "ci_low"].gt(0).all()
        ),
        "targeting_not_robust": bool(
            evidence.loc[evidence["contrast"].eq("targeting_increment"), "ci_low"].le(0).all()
        ),
        "capacity_robust": bool(
            evidence.loc[evidence["contrast"].eq("capacity_increment"), "ci_low"].gt(0).all()
        ),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scope": "cross_stage_prediction_to_policy_evidence_synthesis",
    }
    if audit["status"] != "pass":
        raise ValueError(f"prediction-policy synthesis audit failed: {audit}")
    results_dir = Path(config["outputs"]["results_root"]) / config["experiment"]["id"] / profile_name
    report_dir = Path(config["outputs"]["report_root"]) / config["experiment"]["id"] / profile_name
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(results_dir / "prediction_policy_evidence.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    sources = [config_path, Path(__file__)]
    sources.extend(
        Path(value)
        for key, value in config["data"].items()
        if key.endswith("summary")
    )
    pd.DataFrame(
        [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in sources]
    ).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
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
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot(evidence, report_dir / "prediction_to_policy_gap.png", int(config["profiles"][profile_name]["render_dpi"]))
    report = "# Prediction-to-policy gap\n\n"
    report += "Strictly forward historical scores reliably predict relative singleton intervention value, but neither immediate nor endogenous equal-capacity history targeting has a cross-family interval above zero. In contrast, additional response capacity has a positive interval in both timings and epidemic models. The publishable result is therefore a separation between individual-value predictability and operational set-policy transportability, not a claim that node importance is unknowable.\n"
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prediction-to-policy evidence synthesis.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["full"], default="full")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
