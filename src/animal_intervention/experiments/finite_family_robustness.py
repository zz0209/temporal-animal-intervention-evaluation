from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from math import comb
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .intervention_delivery_sensitivity import SYSTEM_FAMILY_LABELS
from .outbreak_response_pilot import _sha256


def exact_one_sided_sign_probability(values: pd.Series) -> float:
    """Return the exact probability of at least the observed positive signs."""
    nonzero = values.loc[values.ne(0)].astype(float)
    trials = len(nonzero)
    if trials == 0:
        return 1.0
    positives = int(nonzero.gt(0).sum())
    return sum(comb(trials, k) for k in range(positives, trials + 1)) / (2**trials)


def summarize_family_contrast(
    rows: pd.DataFrame,
    value_column: str = "value",
    unit_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize one contrast at the independent-family level."""
    if unit_columns:
        units = (
            rows.groupby(["system_family", *unit_columns], observed=True)[value_column]
            .mean()
            .reset_index()
        )
        family = (
            units.groupby("system_family", observed=True)[value_column]
            .mean()
            .rename("family_mean")
            .reset_index()
        )
    else:
        family = (
            rows.groupby("system_family", observed=True)[value_column]
            .mean()
            .rename("family_mean")
            .reset_index()
        )
    values = family["family_mean"]
    leave_one_out = []
    for omitted in family["system_family"]:
        retained = family.loc[family["system_family"].ne(omitted), "family_mean"]
        leave_one_out.append(
            {
                "omitted_family": omitted,
                "retained_families": len(retained),
                "mean": retained.mean(),
            }
        )
    summary = pd.DataFrame(
        [
            {
                "families": len(family),
                "family_equal_mean": values.mean(),
                "family_median": values.median(),
                "context_equal_mean": rows[value_column].mean(),
                "population_context_weighted_mean": np.average(
                    rows[value_column], weights=rows["population_size"]
                )
                if "population_size" in rows.columns
                else np.nan,
                "minimum_family_mean": values.min(),
                "maximum_family_mean": values.max(),
                "minimum_leave_one_out_mean": min(row["mean"] for row in leave_one_out),
                "maximum_leave_one_out_mean": max(row["mean"] for row in leave_one_out),
                "positive_families": int(values.gt(0).sum()),
                "zero_families": int(values.eq(0).sum()),
                "exact_one_sided_sign_probability": exact_one_sided_sign_probability(values),
            }
        ]
    )
    return summary, pd.DataFrame(leave_one_out)


def random_draw_audit(
    worlds: pd.DataFrame,
    tie_credit: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the case ring with each saved random allocation in the same world."""
    keys = [
        "dataset_id",
        "network_id",
        "anchor_id",
        "parameter_id",
        "epidemic_model",
        "random_block",
        "initial_infected",
        "world_seed",
    ]
    metadata = worlds.drop_duplicates(keys)[
        keys + ["system_family", "population_size"]
    ]
    wide = worlds.pivot(index=keys, columns="response_method", values="final_size").reset_index()
    saved = worlds.loc[worlds["response_method"].eq("random"), keys + ["random_final_sizes"]]
    merged = wide.merge(metadata, on=keys, validate="one_to_one").merge(
        saved, on=keys, validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        ring_size = float(row.past_recent_ring)
        for replicate, random_size in enumerate(str(row.random_final_sizes).split("|")):
            difference = (float(random_size) - ring_size) / float(row.population_size)
            rows.append(
                {
                    "system_family": row.system_family,
                    "epidemic_model": row.epidemic_model,
                    "random_replicate": replicate,
                    "difference": difference,
                    "strict_win": float(difference > 0),
                    "tie": float(difference == 0),
                    "strict_loss": float(difference < 0),
                    "superiority_credit": 1.0
                    if difference > 0
                    else tie_credit
                    if difference == 0
                    else 0.0,
                }
            )
    detail = pd.DataFrame(rows)
    family = (
        detail.groupby(["epidemic_model", "system_family"], observed=True)
        .agg(
            strict_win_probability=("strict_win", "mean"),
            tie_probability=("tie", "mean"),
            strict_loss_probability=("strict_loss", "mean"),
            probability_of_superiority=("superiority_credit", "mean"),
            mean_advantage=("difference", "mean"),
        )
        .reset_index()
    )
    return detail, family


def _load_contrast(
    path: Path,
    contrast: str,
    model: str | None = None,
) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows.loc[rows["contrast"].eq(contrast)].copy()
    if model is not None:
        rows = rows.loc[rows["epidemic_model"].eq(model)].copy()
    if rows.empty:
        raise ValueError(f"No rows for {contrast} in {path}")
    return rows


def _plot(
    family_effects: pd.DataFrame,
    weighting: pd.DataFrame,
    random_family: pd.DataFrame,
    path: Path,
) -> None:
    palette = {"SIR": "#6A3D9A", "SEIR/Erlang": "#1B7837", "Future": "#8C510A"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    panel = axes[0]
    order = list(dict.fromkeys(family_effects["system_family"]))
    y = np.arange(len(order))
    for label, marker in [("SIR", "s"), ("SEIR/Erlang", "D"), ("Future", "^")]:
        subset = family_effects.loc[family_effects["analysis"].eq(label)].set_index(
            "system_family"
        )
        values = subset.reindex(order)["family_mean"] * 100
        panel.scatter(values, y, marker=marker, s=46, color=palette[label], label=label)
    panel.axvline(0, color="#333333", linewidth=0.8)
    panel.set_yticks(y, [SYSTEM_FAMILY_LABELS.get(item, item) for item in order])
    panel.set_xlabel("Avoided attack rate (points)")
    panel.set_title("Family effects", fontsize=11)
    panel.legend(frameon=False, fontsize=8)

    panel = axes[1]
    metrics = [
        "family_equal_mean",
        "family_median",
        "context_equal_mean",
        "population_context_weighted_mean",
    ]
    labels = ["Equal family", "Median family", "Equal context", "Population × context"]
    x = np.arange(len(metrics))
    width = 0.22
    for index, label in enumerate(["SIR", "SEIR/Erlang", "Future"]):
        values = [float(weighting.loc[weighting["analysis"].eq(label), metric].iloc[0]) * 100 for metric in metrics]
        panel.bar(x + (index - 1) * width, values, width=width, color=palette[label], label=label)
    panel.axhline(0, color="#333333", linewidth=0.8)
    panel.set_xticks(x, labels, rotation=20, ha="right")
    panel.set_ylabel("Estimated gain (points)")
    panel.set_title("Weighting sensitivity", fontsize=11)
    panel.legend(frameon=False, fontsize=8)

    panel = axes[2]
    models = [("temporal_sir", "SIR"), ("temporal_seir_erlang", "SEIR/Erlang")]
    for index, (model, label) in enumerate(models):
        subset = random_family.loc[random_family["epidemic_model"].eq(model)]
        win = subset["strict_win_probability"].mean()
        tie = subset["tie_probability"].mean()
        loss = subset["strict_loss_probability"].mean()
        panel.bar(index, win, color="#1B7837", width=0.62)
        panel.bar(index, tie, bottom=win, color="#BDBDBD", width=0.62)
        panel.bar(index, loss, bottom=win + tie, color="#762A83", width=0.62)
        for segment, center, value, text_color in [
            ("Win", win / 2, win, "white"),
            ("Tie", win + tie / 2, tie, "#222222"),
            ("Loss", win + tie + loss / 2, loss, "white"),
        ]:
            if value >= 0.025:
                if segment == "Loss":
                    label_text = f"Loss {100 * value:.1f}%" if index == 0 else f"{100 * value:.1f}%"
                else:
                    label_text = f"{segment}\n{100 * value:.1f}%" if index == 0 else f"{100 * value:.1f}%"
                panel.text(index, center, label_text, ha="center", va="center", fontsize=8, color=text_color)
    panel.set_xticks([0, 1], ["SIR", "SEIR/Erlang"])
    panel.set_ylim(0, 1)
    panel.set_ylabel("Fraction of single random-list comparisons")
    panel.set_title("Single-draw decision risk", fontsize=11)

    for letter, panel in zip("ABC", axes):
        panel.text(-0.12, 1.04, letter, transform=panel.transAxes, weight="bold")
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        panel.tick_params(labelsize=8)
        panel.xaxis.label.set_size(9)
        panel.yaxis.label.set_size(9)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = Path(config["outputs"]["results_root"]) / config["experiment"]["id"] / "full"
    report_dir = Path(config["outputs"]["report_root"]) / config["experiment"]["id"] / "full"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    analyses: list[tuple[str, pd.DataFrame]] = []
    individual = pd.read_csv(config["data"]["individual_differences"])
    individual = individual.loc[
        individual["comparison"].eq(config["analysis"]["individual_comparison"])
        & individual["outcome"].isin(config["analysis"]["individual_outcomes"])
    ]
    individual_summaries = []
    for outcome, rows in individual.groupby("outcome", observed=True):
        summary, leave = summarize_family_contrast(
            rows, "difference", ["dataset_id", "network_id"]
        )
        summary.insert(0, "analysis", f"individual_{outcome}")
        leave.insert(0, "analysis", f"individual_{outcome}")
        individual_summaries.append(summary)
        leave.to_csv(results_dir / f"individual_{outcome}_leave_one_family_out.csv", index=False)

    case_worlds = pd.read_csv(config["data"]["case_worlds"], dtype={"initial_infected": str})
    population = case_worlds[
        ["dataset_id", "network_id", "anchor_id", "population_size"]
    ].drop_duplicates()
    case = _load_contrast(
        Path(config["data"]["case_contrasts"]), config["analysis"]["case_contrast"]
    )
    if "population_size" not in case.columns:
        case = case.merge(
            population,
            on=["dataset_id", "network_id", "anchor_id"],
            how="left",
            validate="many_to_one",
        )
    for model, label in [("temporal_sir", "SIR"), ("temporal_seir_erlang", "SEIR/Erlang")]:
        rows = case.loc[case["epidemic_model"].eq(model)]
        analyses.append((label, rows))
    forecast = _load_contrast(
        Path(config["data"]["forecast_contrasts"]),
        config["analysis"]["forecast_contrast"],
        config["analysis"]["forecast_model"],
    )
    forecast = forecast.loc[forecast["budget"].gt(1)].merge(
        population,
        on=["dataset_id", "network_id", "anchor_id"],
        how="left",
        validate="many_to_one",
    )
    analyses.append(("Future", forecast))

    summaries = []
    effects = []
    leave_rows = []
    for label, rows in analyses:
        summary, leave = summarize_family_contrast(rows, unit_columns=["analysis_cluster_id"])
        summary.insert(0, "analysis", label)
        leave.insert(0, "analysis", label)
        summaries.append(summary)
        leave_rows.append(leave)
        units = rows.groupby(
            ["system_family", "analysis_cluster_id"], observed=True
        )["value"].mean().reset_index()
        family = units.groupby("system_family", observed=True)["value"].mean().reset_index(name="family_mean")
        if "population_size" in rows.columns:
            avoided = rows.assign(
                avoided_animals=rows["value"] * rows["population_size"]
            ).groupby("system_family", observed=True)["avoided_animals"].mean()
            mean_population = rows.groupby("system_family", observed=True)[
                "population_size"
            ].mean()
            family["mean_avoided_animals"] = family["system_family"].map(avoided)
            family["mean_context_population"] = family["system_family"].map(
                mean_population
            )
        family.insert(0, "analysis", label)
        effects.append(family)

    random_detail, random_family = random_draw_audit(
        case_worlds, float(config["analysis"]["tie_credit"])
    )
    robustness = pd.concat(summaries, ignore_index=True)
    family_effects = pd.concat(effects, ignore_index=True)
    leave = pd.concat(leave_rows, ignore_index=True)
    individual_summary = pd.concat(individual_summaries, ignore_index=True)

    expected_case_families = int(config["analysis"].get("expected_case_families", 5))
    expected_forecast_families = int(
        config["analysis"].get("expected_forecast_families", 4)
    )
    checks = {
        "expected_case_families": case["system_family"].nunique()
        == expected_case_families,
        "expected_forecast_families": forecast["system_family"].nunique()
        == expected_forecast_families,
        "eight_random_draws_per_world": case_worlds.loc[
            case_worlds["response_method"].eq("random"), "random_final_sizes"
        ].str.count(r"\|").add(1).eq(8).all(),
        "finite_robustness_metrics": np.isfinite(
            robustness.select_dtypes(include=[np.number])
        ).all().all(),
        "case_point_estimates_reconcile": bool(
            "expected_case_point_estimates" not in config["analysis"]
            or np.allclose(
                robustness.loc[
                    robustness["analysis"].isin(["SIR", "SEIR/Erlang"]),
                    "family_equal_mean",
                ],
                config["analysis"]["expected_case_point_estimates"],
            )
        ),
        "forecast_point_estimate_reconciles": bool(
            "expected_forecast_point_estimate" not in config["analysis"]
            or np.isclose(
                robustness.loc[
                    robustness["analysis"].eq("Future"), "family_equal_mean"
                ].iloc[0],
                float(config["analysis"]["expected_forecast_point_estimate"]),
                atol=5e-7,
            )
        ),
    }
    audit = {"status": "pass" if all(checks.values()) else "fail", "checks": {key: bool(value) for key, value in checks.items()}}
    if audit["status"] != "pass":
        raise ValueError(audit)

    robustness.to_csv(results_dir / "robustness_summary.csv", index=False)
    family_effects.to_csv(results_dir / "family_effects.csv", index=False)
    leave.to_csv(results_dir / "leave_one_family_out.csv", index=False)
    individual_summary.to_csv(results_dir / "individual_robustness_summary.csv", index=False)
    random_detail.to_csv(results_dir / "single_random_draws.csv.gz", index=False, compression="gzip")
    random_family.to_csv(results_dir / "single_random_draw_family_summary.csv", index=False)
    (results_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (results_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (results_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": config["experiment"]["id"],
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config_sha256": _sha256(config_path),
                "source_sha256": _sha256(Path(__file__)),
                "input_sha256": {key: _sha256(Path(path)) for key, path in config["data"].items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot(family_effects, robustness, random_family, report_dir / "finite_family_robustness.png")
    manuscript_figure = Path(config["outputs"]["manuscript_figure_path"])
    manuscript_figure.parent.mkdir(parents=True, exist_ok=True)
    _plot(family_effects, robustness, random_family, manuscript_figure)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit finite-family robustness of frozen headline results.")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=2))


if __name__ == "__main__":
    main()
