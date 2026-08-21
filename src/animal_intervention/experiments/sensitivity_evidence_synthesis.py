from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_COLUMN = "system_family"
FAMILY_ALIASES = {
    "linked_wytham_songbirds": "linked_wytham_songbird_family",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required sensitivity artifact is missing: {path}")
    return pd.read_csv(path)


def _records(
    frame: pd.DataFrame,
    *,
    domain: str,
    estimand: str,
    scenario_columns: Iterable[str],
    value_column: str = "mean_value",
    source: str,
) -> pd.DataFrame:
    required = {FAMILY_COLUMN, value_column, *scenario_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    selected = frame[[*scenario_columns, FAMILY_COLUMN, value_column]].copy()
    selected.insert(0, "domain", domain)
    selected.insert(1, "estimand", estimand)
    selected["effect"] = selected.pop(value_column).astype(float)
    selected["source_artifact"] = source
    return selected


def summarize_family_effects(family_effects: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        column
        for column in family_effects.columns
        if column not in {FAMILY_COLUMN, "effect", "source_artifact"}
    ]
    rows: list[dict[str, object]] = []
    for key, group in family_effects.groupby(group_columns, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key_values, strict=True))
        values = group["effect"].astype(float)
        record.update(
            {
                "families": int(group[FAMILY_COLUMN].nunique()),
                "family_equal_mean": float(values.mean()),
                "median_family": float(values.median()),
                "family_min": float(values.min()),
                "family_max": float(values.max()),
                "positive_families": int((values > 0).sum()),
                "zero_families": int((values == 0).sum()),
                "negative_families": int((values < 0).sum()),
                "source_artifact": " | ".join(sorted(group["source_artifact"].unique())),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _temporal_score_records(path: Path) -> pd.DataFrame:
    frame = _read(path)
    index = [FAMILY_COLUMN]
    temporal = frame.loc[frame["method"].eq("ridge_temporal_summary_loso")].set_index(index)
    static = frame.loc[frame["method"].eq("ridge_static_summary_loso")].set_index(index)
    joined = temporal.join(static, lsuffix="_temporal", rsuffix="_static", validate="one_to_one")
    metrics = {
        "rank_correlation_gain": ("mean_spearman_temporal", "mean_spearman_static"),
        "top20_value_capture_gain": (
            "mean_value_capture_temporal",
            "mean_value_capture_static",
        ),
        "oracle_regret_reduction": (
            "mean_oracle_regret_static",
            "mean_oracle_regret_temporal",
        ),
    }
    rows = []
    for family, row in joined.iterrows():
        for estimand, (left, right) in metrics.items():
            rows.append(
                {
                    "domain": "temporal_score",
                    "estimand": estimand,
                    FAMILY_COLUMN: family,
                    "effect": float(row[left] - row[right]),
                    "source_artifact": str(path).replace("\\", "/"),
                }
            )
    return pd.DataFrame(rows)


def build(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = config["sources"]
    if not isinstance(sources, dict):
        raise TypeError("sources must be a mapping")
    parts: list[pd.DataFrame] = []

    delivery_absolute_path = Path(str(sources["delivery_absolute_family"]))
    delivery_absolute = _read(delivery_absolute_path)
    delivery_absolute = delivery_absolute.loc[
        delivery_absolute["method"].eq("stable_watchlist")
    ]
    parts.append(
        _records(
            delivery_absolute,
            domain="delivery",
            estimand="history_policy_over_no_action",
            scenario_columns=[
                "detection_profile",
                "action_delay_fraction",
                "residual_contact_multiplier",
            ],
            source=str(delivery_absolute_path).replace("\\", "/"),
        )
    )

    delivery_relative_path = Path(str(sources["delivery_relative_family"]))
    delivery_relative = _read(delivery_relative_path)
    delivery_relative = delivery_relative.loc[
        delivery_relative["method"].eq("stable_plus_tracing")
    ]
    parts.append(
        _records(
            delivery_relative,
            domain="delivery",
            estimand="history_plus_tracing_over_history",
            scenario_columns=[
                "detection_profile",
                "action_delay_fraction",
                "residual_contact_multiplier",
            ],
            source=str(delivery_relative_path).replace("\\", "/"),
        )
    )

    observation_relative_path = Path(str(sources["observation_relative_family"]))
    parts.append(
        _records(
            _read(observation_relative_path),
            domain="observation_error",
            estimand="detected_contacts_over_history",
            scenario_columns=["observation_profile"],
            source=str(observation_relative_path).replace("\\", "/"),
        )
    )

    observation_absolute_path = Path(str(sources["observation_absolute_family"]))
    parts.append(
        _records(
            _read(observation_absolute_path),
            domain="observation_error",
            estimand="policy_over_no_action",
            scenario_columns=["observation_profile", "method"],
            source=str(observation_absolute_path).replace("\\", "/"),
        )
    )

    rewiring_reference_path = Path(str(sources["rewiring_reference_family"]))
    rewiring_reference = _read(rewiring_reference_path)
    rewiring_reference = rewiring_reference.loc[
        rewiring_reference["method"].eq("contact_to_detected")
        & rewiring_reference["rewiring_fraction"].eq(0.0)
    ]
    rewiring_precision_path = Path(str(sources["rewiring_precision_family"]))
    rewiring_precision = _read(rewiring_precision_path)
    rewiring_precision = rewiring_precision.loc[
        rewiring_precision["method"].eq("contact_to_detected")
    ]
    rewiring = pd.concat([rewiring_reference, rewiring_precision], ignore_index=True)
    parts.append(
        _records(
            rewiring,
            domain="rewiring",
            estimand="detected_contacts_over_history",
            scenario_columns=["detection_profile", "rewiring_fraction"],
            source=(
                f"{str(rewiring_reference_path).replace('\\', '/')} | "
                f"{str(rewiring_precision_path).replace('\\', '/')}"
            ),
        )
    )

    for model, key in [
        ("SIR", "sir_regime_family"),
        ("SEIR/Erlang", "seir_regime_family"),
    ]:
        regime_path = Path(str(sources[key]))
        regime = _read(regime_path)
        regime = regime.loc[regime["method"].eq("contact_to_detected")].copy()
        regime["epidemic_model"] = model
        parts.append(
            _records(
                regime,
                domain="epidemic_regime",
                estimand="detected_contacts_over_history",
                scenario_columns=[
                    "epidemic_model",
                    "disease_regime",
                    "detection_profile",
                    "rewiring_fraction",
                ],
                source=str(regime_path).replace("\\", "/"),
            )
        )

    mapping_path = Path(str(sources["group_mapping_family"]))
    mapping = _read(mapping_path).rename(columns={"mean_response_value": "mean_value"})
    parts.append(
        _records(
            mapping,
            domain="group_mapping",
            estimand="response_value",
            scenario_columns=["mapping", "epidemic_model", "contrast"],
            source=str(mapping_path).replace("\\", "/"),
        )
    )

    parts.append(_temporal_score_records(Path(str(sources["temporal_family_metrics"]))))

    local_path = Path(str(sources["local_calibration_family"]))
    local = _read(local_path).rename(columns={"improvement": "mean_value"})
    parts.append(
        _records(
            local,
            domain="local_calibration",
            estimand="error_reduction",
            scenario_columns=["epidemic_model", "endpoint"],
            source=str(local_path).replace("\\", "/"),
        )
    )

    family_effects = pd.concat(parts, ignore_index=True, sort=False)
    family_effects[FAMILY_COLUMN] = family_effects[FAMILY_COLUMN].replace(FAMILY_ALIASES)
    family_effects = family_effects.sort_values(
        ["domain", "estimand", FAMILY_COLUMN], na_position="last"
    ).reset_index(drop=True)
    summary = summarize_family_effects(family_effects)
    summary = summary.sort_values(["domain", "estimand"], na_position="last").reset_index(
        drop=True
    )
    return family_effects, summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentage_points(values: pd.Series) -> pd.Series:
    return values.astype(float) * 100.0


def plot_sensitivity_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot compact quantitative summaries of the major sensitivity domains."""
    green = "#2F6F5E"
    burgundy = "#9E3F53"
    gold = "#9A7B2F"
    charcoal = "#3F4348"
    light = "#D6D8DA"
    marker_map = {"early_detection": "D", "delayed_detection": "s"}
    color_map = {"early_detection": green, "delayed_detection": burgundy}

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.4))

    delivery = summary.loc[summary["domain"].eq("delivery")].copy()
    for axis, estimand, title in [
        (axes[0, 0], "history_policy_over_no_action", "Delivery value"),
        (
            axes[0, 1],
            "history_plus_tracing_over_history",
            "Tracing increment",
        ),
    ]:
        panel = delivery.loc[delivery["estimand"].eq(estimand)]
        for profile in ["early_detection", "delayed_detection"]:
            profile_data = panel.loc[panel["detection_profile"].eq(profile)]
            for delay, linestyle in [(0.0, "-"), (0.1, "--")]:
                line = profile_data.loc[
                    profile_data["action_delay_fraction"].astype(float).eq(delay)
                ].sort_values("residual_contact_multiplier")
                axis.plot(
                    line["residual_contact_multiplier"].astype(float),
                    _percentage_points(line["family_equal_mean"]),
                    color=color_map[profile],
                    marker=marker_map[profile],
                    linestyle=linestyle,
                    linewidth=1.5,
                    markersize=5,
                    label=f"{profile.replace('_', ' ')}, delay {delay:g}",
                )
        axis.axhline(0, color=charcoal, linewidth=0.8)
        axis.set_title(title, fontsize=10, weight="bold")
        axis.set_xlabel("Residual contact multiplier")
        axis.set_ylabel("Effect (attack-rate points)")
        axis.set_xticks([0, 0.25, 0.5])
        axis.grid(axis="y", color=light, linewidth=0.6)

    observation = summary.loc[
        summary["domain"].eq("observation_error")
        & summary["estimand"].eq("detected_contacts_over_history")
    ].sort_values("observation_profile")
    x = np.arange(len(observation))
    means = _percentage_points(observation["family_equal_mean"])
    low = means - _percentage_points(observation["family_min"])
    high = _percentage_points(observation["family_max"]) - means
    axes[0, 2].errorbar(
        x,
        means,
        yerr=np.vstack([low, high]),
        fmt="D",
        color=green,
        ecolor=charcoal,
        capsize=3,
        linewidth=1.2,
    )
    axes[0, 2].axhline(0, color=charcoal, linewidth=0.8)
    axes[0, 2].set_xticks(x, ["Moderate loss", "Reference"])
    axes[0, 2].set_title("Observation loss", fontsize=10, weight="bold")
    axes[0, 2].set_ylabel("Effect (attack-rate points)")
    axes[0, 2].grid(axis="y", color=light, linewidth=0.6)

    rewiring = summary.loc[summary["domain"].eq("rewiring")].copy()
    for profile in ["early_detection", "delayed_detection"]:
        line = rewiring.loc[rewiring["detection_profile"].eq(profile)].sort_values(
            "rewiring_fraction"
        )
        axes[1, 0].plot(
            line["rewiring_fraction"].astype(float),
            _percentage_points(line["family_equal_mean"]),
            color=color_map[profile],
            marker=marker_map[profile],
            linewidth=1.5,
            label=profile.replace("_", " "),
        )
    axes[1, 0].axhline(0, color=charcoal, linewidth=0.8)
    axes[1, 0].set_title("Deterministic rewiring", fontsize=10, weight="bold")
    axes[1, 0].set_xlabel("Fraction of removed hazard reassigned")
    axes[1, 0].set_ylabel("Effect (attack-rate points)")
    axes[1, 0].set_xticks([0, 0.5, 1])
    axes[1, 0].grid(axis="y", color=light, linewidth=0.6)

    regimes = summary.loc[
        summary["domain"].eq("epidemic_regime")
        & summary["detection_profile"].eq("early_detection")
        & summary["rewiring_fraction"].astype(float).eq(0.0)
    ].copy()
    regime_order = ["low", "middle", "high"]
    for model, color, marker in [
        ("SIR", burgundy, "s"),
        ("SEIR/Erlang", green, "D"),
    ]:
        line = regimes.loc[regimes["epidemic_model"].eq(model)].set_index(
            "disease_regime"
        ).loc[regime_order]
        axes[1, 1].plot(
            np.arange(3),
            _percentage_points(line["family_equal_mean"]),
            color=color,
            marker=marker,
            linewidth=1.5,
            label=model,
        )
    axes[1, 1].axhline(0, color=charcoal, linewidth=0.8)
    axes[1, 1].set_xticks(np.arange(3), ["Low", "Middle", "High"])
    axes[1, 1].set_title("Epidemic regime", fontsize=10, weight="bold")
    axes[1, 1].set_ylabel("Effect (attack-rate points)")
    axes[1, 1].grid(axis="y", color=light, linewidth=0.6)

    mapping = summary.loc[
        summary["domain"].eq("group_mapping")
        & summary["contrast"].isin(["capacity_increment", "targeting_increment"])
    ].copy()
    mapping["label"] = (
        mapping["contrast"].map(
            {"capacity_increment": "Cap", "targeting_increment": "Target"}
        )
        + " | "
        + mapping["epidemic_model"].map(
            {"temporal_sir": "SIR", "temporal_seir_erlang": "SEIR"}
        )
        + " | "
        + mapping["mapping"].map(
            {
                "frequency_dependent": "frequency",
                "hazard_normalized_undiluted_clique": "clique",
            }
        )
    )
    mapping = mapping.sort_values(["contrast", "epidemic_model", "mapping"])
    colors = [
        green if value == "frequency_dependent" else gold
        for value in mapping["mapping"]
    ]
    y = np.arange(len(mapping))
    axes[1, 2].barh(y, _percentage_points(mapping["family_equal_mean"]), color=colors)
    axes[1, 2].axvline(0, color=charcoal, linewidth=0.8)
    axes[1, 2].set_yticks(y, mapping["label"])
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("Group-event mapper", fontsize=10, weight="bold")
    axes[1, 2].set_xlabel("Avoided attack rate (points)")
    axes[1, 2].grid(axis="x", color=light, linewidth=0.6)

    for panel_label, axis in zip("ABCDEF", axes.flat, strict=True):
        axis.text(-0.13, 1.08, panel_label, transform=axis.transAxes, weight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.86, wspace=0.48, hspace=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    family_effects, summary = build(config)
    output = config["outputs"]
    if not isinstance(output, dict):
        raise TypeError("outputs must be a mapping")
    results_dir = Path(str(output["results_dir"]))
    manuscript_data = Path(str(output["manuscript_data"]))
    results_dir.mkdir(parents=True, exist_ok=True)
    manuscript_data.parent.mkdir(parents=True, exist_ok=True)
    family_path = results_dir / "sensitivity_family_effects.csv"
    summary_path = results_dir / "sensitivity_summary.csv"
    family_effects.to_csv(family_path, index=False)
    summary.to_csv(summary_path, index=False)
    family_effects.to_csv(manuscript_data, index=False)
    figure_path = Path(str(output["figure"] ))
    plot_sensitivity_summary(summary, figure_path)
    audit = {
        "experiment_id": config["experiment_id"],
        "status": "pass",
        "family_rows": int(len(family_effects)),
        "summary_rows": int(len(summary)),
        "domains": sorted(family_effects["domain"].unique().tolist()),
        "families": sorted(family_effects[FAMILY_COLUMN].dropna().unique().tolist()),
        "source_artifacts": sorted(family_effects["source_artifact"].unique().tolist()),
        "family_csv_sha256": _sha256(family_path),
        "summary_csv_sha256": _sha256(summary_path),
        "manuscript_data_sha256": _sha256(manuscript_data),
        "figure_sha256": _sha256(figure_path),
    }
    (results_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (results_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize family-level sensitivity evidence from frozen experiments."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
