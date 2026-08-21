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

from .history_baseline_substitution import _markdown_table
from .intervention_delivery_sensitivity import (
    SYSTEM_FAMILY_LABELS,
    _hierarchical_summary,
)
from .outbreak_response_pilot import _git_value, _sha256
from .role_aware_sentinel_response import WORLD_KEYS


PATH = [
    "random__case_only",
    "history_weight__case_only",
    "history_coverage__case_only",
    "history_coverage__random",
    "history_coverage__history_weight",
    "full_surveillance__history_weight",
]
COMPONENTS = [
    "sentinel_history",
    "sentinel_coverage_specialization",
    "response_capacity",
    "response_targeting",
    "full_surveillance_opportunity",
]


def decompose_path(wide: pd.DataFrame) -> pd.DataFrame:
    metadata = WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]
    rows = []
    for component, reference, challenger in zip(COMPONENTS, PATH[:-1], PATH[1:]):
        frame = wide[metadata].copy()
        frame["component"] = component
        frame["value"] = (
            wide[reference] - wide[challenger]
        ) / wide["population_size"].astype(float)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _assemble_worlds(role: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    metadata = role.drop_duplicates(WORLD_KEYS)[
        WORLD_KEYS + ["system_family", "analysis_cluster_id", "population_size"]
    ]
    role_wide = role.pivot(index=WORLD_KEYS, columns="policy", values="final_size").reset_index()
    random_arm = matched.loc[matched["response_method"].eq("random"), WORLD_KEYS + ["final_size"]].rename(
        columns={"final_size": "history_coverage__random"}
    )
    wide = metadata.merge(role_wide, on=WORLD_KEYS, validate="one_to_one").merge(
        random_arm, on=WORLD_KEYS, validate="one_to_one"
    )
    missing = [column for column in PATH if column not in wide]
    if missing:
        raise ValueError(f"preparedness path policies are missing: {missing}")
    return wide


def _plot_components(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    labels = {
        "sentinel_history": "History-ranked sentinel placement",
        "sentinel_coverage_specialization": "Coverage-specialized sentinels",
        "response_capacity": "Additional response capacity",
        "response_targeting": "History-ranked response targets",
        "full_surveillance_opportunity": "Residual full-surveillance opportunity",
    }
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.4), sharex=True, sharey=True)
    for axis, model, title in zip(
        axes,
        ["temporal_sir", "temporal_seir_erlang"],
        ["Temporal SIR", "Staged SEIR/Erlang"],
    ):
        frame = summary.loc[summary["epidemic_model"].eq(model)].set_index("component").loc[COMPONENTS]
        y = np.arange(len(COMPONENTS))
        mean = 100 * frame["family_equal_mean"].to_numpy(float)
        low = 100 * frame["ci_low"].to_numpy(float)
        high = 100 * frame["ci_high"].to_numpy(float)
        colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#B279A2"]
        axis.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="none", ecolor="#777777", capsize=4)
        axis.scatter(mean, y, s=85, color=colors, zorder=3)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=16, weight="bold")
        axis.set_yticks(y, [labels[item] for item in COMPONENTS])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Preparedness value decomposition across animal systems", fontsize=21, weight="bold")
    fig.supxlabel("Family-equal reduction in final attack rate (percentage points)", fontsize=14)
    fig.subplots_adjust(left=0.31, right=0.98, top=0.83, bottom=0.14, wspace=0.10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_response_share(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    frame = summary.loc[summary["component"].isin(["response_capacity", "response_targeting"])].copy()
    pivot = frame.pivot(index="epidemic_model", columns="component", values="family_equal_mean")
    total = pivot.sum(axis=1)
    fractions = pivot.div(total, axis=0)
    models = ["temporal_sir", "temporal_seir_erlang"]
    labels = ["Temporal SIR", "Staged SEIR/Erlang"]
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    y = np.arange(len(models))
    capacity = 100 * fractions.loc[models, "response_capacity"].to_numpy(float)
    targeting = 100 * fractions.loc[models, "response_targeting"].to_numpy(float)
    axis.barh(y, capacity, color="#F58518", label="Capacity increment")
    axis.barh(y, targeting, left=capacity, color="#54A24B", label="History-targeting increment")
    for index, (left, right) in enumerate(zip(capacity, targeting)):
        axis.text(left / 2, index, f"{left:.1f}%", ha="center", va="center", color="white", weight="bold")
        axis.text(left + right / 2, index, f"{right:.1f}%", ha="center", va="center", color="white", weight="bold")
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of modeled post-detection response benefit (%)")
    axis.set_title("Capacity and target choice both contribute", fontsize=17, weight="bold")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    axis.grid(axis="x", alpha=0.2)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.84, bottom=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = str(config["experiment"]["id"])
    role_path = Path(config["data"]["role_policy_worlds"])
    matched_path = Path(config["data"]["matched_response_worlds"])
    role = pd.read_csv(role_path, dtype={"initial_infected": str, "network_id": str})
    matched = pd.read_csv(matched_path, dtype={"initial_infected": str, "network_id": str})
    wide = _assemble_worlds(role, matched)
    components = decompose_path(wide)
    summary, family = _hierarchical_summary(
        components,
        value_column="value",
        group_columns=["epidemic_model", "component"],
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["seed"]),
    )

    component_sum = components.groupby(WORLD_KEYS, observed=True)["value"].sum().sort_index()
    total = (
        (wide[PATH[0]] - wide[PATH[-1]]) / wide["population_size"].astype(float)
    )
    total.index = pd.MultiIndex.from_frame(wide[WORLD_KEYS])
    response = summary.loc[summary["component"].isin(["response_capacity", "response_targeting"])].pivot(
        index="epidemic_model", columns="component", values="family_equal_mean"
    )
    response["targeting_share"] = response["response_targeting"] / response.sum(axis=1)
    family_pivot = family.pivot(
        index=["epidemic_model", "system_family"], columns="component", values="mean_value"
    ).reset_index()
    checks = {
        "six_datasets": wide["dataset_id"].nunique() == 6,
        "five_independent_families": wide["system_family"].nunique() == 5,
        "all_81_anchors": wide[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0] == 81,
        "all_path_policies_present": all(column in wide for column in PATH),
        "path_telescopes_per_world": bool(np.allclose(component_sum.to_numpy(float), total.sort_index().to_numpy(float))),
        "finite_components": bool(np.isfinite(components["value"].to_numpy(float)).all()),
        "response_shares_finite": bool(np.isfinite(response["targeting_share"].to_numpy(float)).all()),
    }
    audit = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": {key: bool(value) for key, value in checks.items()},
        "datasets": int(wide["dataset_id"].nunique()),
        "families": int(wide["system_family"].nunique()),
        "anchors": int(wide[["dataset_id", "network_id", "anchor_id"]].drop_duplicates().shape[0]),
        "paired_worlds": len(wide),
        "scope": "fixed_path_model_based_preparedness_value_decomposition",
    }
    if audit["status"] != "pass":
        raise ValueError(f"preparedness decomposition audit failed: {audit}")

    results_dir = Path(config["outputs"]["results_root"]) / experiment_id / "full"
    report_dir = Path(config["outputs"]["report_root"]) / experiment_id / "full"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    components.to_csv(results_dir / "world_components.csv.gz", index=False, compression="gzip")
    summary.to_csv(results_dir / "component_summary.csv", index=False)
    family.to_csv(results_dir / "family_components.csv", index=False)
    family_pivot.to_csv(results_dir / "family_component_matrix.csv", index=False)
    response.reset_index().to_csv(results_dir / "response_benefit_shares.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    resolved = dict(config)
    resolved["runtime"] = {"timestamp_utc": datetime.now(UTC).isoformat()}
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    source_paths = [config_path, role_path, matched_path, Path(__file__)]
    pd.DataFrame(
        [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in source_paths]
    ).to_csv(results_dir / "source_artifact_hashes.csv", index=False)
    manifest = {
        "experiment_id": experiment_id,
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

    dpi = int(config["outputs"]["render_dpi"])
    _plot_components(summary, report_dir / "preparedness_value_decomposition.png", dpi)
    _plot_response_share(summary, report_dir / "response_benefit_shares.png", dpi)
    display = summary.copy()
    for column in ["family_equal_mean", "ci_low", "ci_high"]:
        display[column] = 100 * display[column]
    shares = response.reset_index().copy()
    shares["targeting_share"] = 100 * shares["targeting_share"]
    report = f"""# Preparedness value decomposition

This synthesis follows one fixed operational path from random sentinel monitoring
with case-only response to a non-resource-matched full-surveillance ceiling. Each
component is a paired difference in the same epidemic world, and the five components
exactly telescope to the total path difference in every world.

- Datasets: {audit['datasets']}
- Independent animal-system families: {audit['families']}
- Anchors: {audit['anchors']}
- Paired worlds: {audit['paired_worlds']}
- Technical audit: **{audit['status']}**

All component values are final attack-rate percentage points.

{_markdown_table(display)}

Response-benefit shares:

{_markdown_table(shares)}

The full-surveillance component is a deliberately non-resource-matched opportunity
ceiling. This fixed-path synthesis was formalized after observing its source
experiments and is therefore explanatory rather than a new preregistered confirmatory
test.
"""
    (report_dir / "STAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preparedness value decomposition.")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()
