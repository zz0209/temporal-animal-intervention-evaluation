from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "figures"

# Restrained, color-vision-safe manuscript palette. Shapes and line styles carry
# the same distinctions, so the figures remain legible without color.
INK = "#252525"
GREY = "#767676"
LIGHT = "#E2E2E2"
BURGUNDY = "#8C2D3E"
GREEN = "#2F6B4F"
PURPLE = "#6B5A7E"
GOLD = "#9A7B2F"
MODELS = {
    "temporal_sir": ("SIR", BURGUNDY, "s"),
    "temporal_seir_erlang": ("SEIR/Erlang", GREEN, "D"),
}
FAMILY_LABELS = {
    "domestic_sheep_sirtrack": "Sheep",
    "guinea_baboons_sociopatterns": "Baboons",
    "linked_wytham_songbird_family": "Wytham/songbirds",
    "oxford_wildbird_network": "Oxford birds",
    "radolfzell_great_tits_ontogeny": "Radolfzell birds",
    "wild_vampire_bats_proximity": "Vampire bats",
    "free_ranging_sheep_fission_fusion": "Free-ranging sheep",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.4,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / name.replace(".png", ".tiff"), dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, f"({label.lower()})", transform=ax.transAxes, weight="bold", fontsize=9.2, va="bottom")


def _zero(ax: plt.Axes) -> None:
    ax.axvline(0, color=GREY, lw=0.75, ls=(0, (3, 2)), zorder=0)
    ax.grid(axis="x", color=LIGHT, lw=0.55, zorder=0)


def _forest_row(
    ax: plt.Axes,
    y: float,
    mean: float,
    low: float,
    high: float,
    color: str,
    marker: str,
    scale: float = 100.0,
) -> None:
    x = scale * mean
    lo = scale * low
    hi = scale * high
    ax.errorbar(
        x,
        y,
        xerr=[[x - lo], [hi - x]],
        fmt=marker,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.9,
        color=color,
        ecolor=color,
        ms=5.2,
        capsize=2.2,
        lw=1.15,
        zorder=3,
    )


def _model_legend(ax: plt.Axes, loc: str = "best") -> None:
    handles = [
        Line2D([], [], marker=marker, lw=0, color=color, markerfacecolor=color, ms=5, label=label)
        for label, color, marker in MODELS.values()
    ]
    ax.legend(handles=handles, frameon=False, loc=loc, handletextpad=0.4, borderpad=0)


def _draw_contact_raster(ax: plt.Axes) -> None:
    rng = np.random.default_rng(8)
    animals = np.arange(8)
    history_t = np.sort(rng.uniform(0.03, 0.49, 26))
    future_t = np.sort(rng.uniform(0.51, 0.97, 24))
    pairs = []
    for t in np.r_[history_t, future_t]:
        a, b = rng.choice(animals, 2, replace=False)
        pairs.append((t, min(a, b), max(a, b)))
    ax.axvspan(0, 0.5, color="#E9EEE9", zorder=0)
    ax.axvspan(0.5, 1, color="#F2ECEE", zorder=0)
    for t, a, b in pairs:
        color = GREEN if t < 0.5 else BURGUNDY
        ax.plot([t, t], [a, b], color=color, alpha=0.58, lw=0.8)
        ax.plot(t, a, marker="|", color=color, ms=4)
        ax.plot(t, b, marker="|", color=color, ms=4)
    ax.axvline(0.5, color=INK, lw=1)
    ax.text(0.25, 7.75, "Observed history", color=GREEN, ha="center", weight="bold")
    ax.text(0.75, 7.75, "Held-out future replay", color=BURGUNDY, ha="center", weight="bold")
    ax.text(
        0.5,
        1.045,
        "Decision anchor",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=INK,
        fontsize=7.6,
        clip_on=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 9)
    ax.set_yticks(animals, [f"animal {i + 1}" for i in animals])
    ax.set_xticks([0, 0.5, 1], ["lookback start", "anchor", "horizon end"])
    ax.set_xlabel("Replay time", labelpad=8)
    ax.set_ylabel("Illustrative animal")


def figure_1() -> None:
    fig = plt.figure(figsize=(7.45, 6.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], width_ratios=[1.08, 0.92], hspace=0.55, wspace=0.48)

    ax = fig.add_subplot(grid[0, :])
    _panel(ax, "A")
    _draw_contact_raster(ax)

    ax = fig.add_subplot(grid[1, 0])
    _panel(ax, "B")
    labels = ["Source", "Sentinel", "Single target", "Target set"]
    questions = ["If infected, how far does it spread?", "How early does it reveal an outbreak?", "How much burden does one action avert?", "How much burden does a joint action avert?"]
    symbols = ["A(i)", "Tdetect(i)", "V(i)", "V(S)"]
    y = np.arange(4)[::-1]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 3.7)
    for yi, label, question, symbol, color in zip(y, labels, questions, symbols, [GREY, PURPLE, GREEN, BURGUNDY]):
        ax.add_patch(Rectangle((0.02, yi - 0.27), 0.29, 0.54, facecolor="white", edgecolor=color, lw=1.1))
        ax.text(0.165, yi, label, ha="center", va="center", weight="bold", color=color, fontsize=7.1)
        ax.text(0.35, yi + 0.09, question, ha="left", va="center", fontsize=7.2)
        ax.text(0.35, yi - 0.15, symbol, ha="left", va="center", fontsize=8.0, style="italic", color=color)
    ax.text(0.02, 3.48, "Four quantities that are often conflated", weight="bold")
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 1])
    _panel(ax, "C")
    families = [
        "Oxford birds",
        "Baboons",
        "Domestic sheep",
        "Wytham/songbirds",
        "Radolfzell birds",
        "Vampire bats",
        "Free-ranging sheep",
    ]
    yy = np.arange(len(families))[::-1]
    ax.set_xlim(-0.1, 4.65)
    ax.set_ylim(-1.10, 7.0)
    for i, (yi, family) in enumerate(zip(yy, families)):
        train_x = [x for x in range(7) if x != i]
        ax.scatter(np.array(train_x) * 0.38, np.repeat(yi, len(train_x)), marker="|", s=110, color=GREY, lw=1.8)
        ax.scatter(i * 0.38, yi, marker="s", s=31, facecolor="white", edgecolor=BURGUNDY, lw=1.2)
        ax.text(2.92, yi, family, va="center", fontsize=7.2)
    ax.text(0.98, 6.64, "Development systems", ha="center", color=GREY, weight="bold")
    ax.text(3.48, 6.64, "Held-out system", ha="center", color=BURGUNDY, weight="bold")
    ax.text(
        2.05,
        -0.48,
        "Paired replay uses the same future events,\nindex case, and random draws",
        ha="center",
        va="top",
        color=INK,
        fontsize=7.1,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Complete system holdout", loc="left", weight="bold", pad=8)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(False)

    _save(fig, "Fig1.png")


def figure_2() -> None:
    evidence = pd.read_csv(ROOT / "results/EXP-20260819-010/full/prediction_policy_evidence.csv")
    expanded_individual = pd.read_csv(ROOT / "results/EXP-20260820-014/full/individual_robustness_summary.csv").set_index("analysis")
    expanded_case = pd.read_csv(ROOT / "results/EXP-20260820-013/full/contrast_summary.csv")
    expanded_case_family = pd.read_csv(ROOT / "results/EXP-20260820-013/full/family_contrasts.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9), gridspec_kw={"width_ratios": [0.95, 1.12, 1.12]})

    ax = axes[0]
    _panel(ax, "A")
    specs = [
        ("individual_spearman_gain", "Rank correlation", GREEN, "D", 1),
        ("individual_value_capture_gain", "Top-20% value capture", PURPLE, "^", 1),
    ]
    for yi, (key, _, color, marker, scale) in zip(np.arange(2)[::-1], specs):
        row = expanded_individual.loc[key]
        _forest_row(ax, yi, row.family_equal_mean, row.minimum_family_mean, row.maximum_family_mean, color, marker, scale=scale)
    ax.set_yticks(np.arange(2)[::-1], [x[1] for x in specs])
    ax.set_xlabel("Gain over random ranking\n(point is mean; whisker is family range)")
    ax.set_title("Individual prediction", weight="bold")
    _zero(ax)

    row_specs = [("temporal_sir", "SIR"), ("temporal_seir_erlang", "SEIR/Erlang")]
    for ax, contrast, letter, title in [
        (axes[1], "targeting_increment", "B", "Choose targets"),
        (axes[2], "capacity_increment", "C", "Add response capacity"),
    ]:
        _panel(ax, letter)
        for yi, (model, label) in zip(np.arange(2)[::-1], row_specs):
            if contrast == "targeting_increment":
                row = expanded_case[(expanded_case.contrast.eq("recent_ring_vs_random")) & (expanded_case.epidemic_model.eq(model))].iloc[0]
                family_values = expanded_case_family[
                    (expanded_case_family.contrast.eq("recent_ring_vs_random"))
                    & (expanded_case_family.epidemic_model.eq(model))
                ].mean_value
                mean, low, high = row.family_equal_mean, family_values.min(), family_values.max()
            else:
                row = evidence[(evidence.stage.eq("immediate_response")) & (evidence.contrast.eq(contrast)) & (evidence.epidemic_model.eq(model))].iloc[0]
                mean, low, high = row.estimate, row.ci_low, row.ci_high
            _, color, marker = MODELS[model]
            _forest_row(ax, yi, mean, low, high, color, marker)
        ax.set_yticks(np.arange(2)[::-1], [x[1] for x in row_specs])
        ax.tick_params(axis="y", labelsize=7.2, pad=2)
        ax.set_xlabel("Avoided final attack rate (points)")
        ax.set_title(title, weight="bold", fontsize=8.8)
        _zero(ax)
    fig.subplots_adjust(wspace=1.02, left=0.15, right=0.985, bottom=0.22, top=0.88)
    _save(fig, "Fig2.png")


def figure_3() -> None:
    ring = pd.read_csv(ROOT / "results/EXP-20260820-013/full/contrast_summary.csv")
    ring_family = pd.read_csv(ROOT / "results/EXP-20260820-013/full/family_contrasts.csv")
    wait = pd.read_csv(ROOT / "results/EXP-20260818-006/full/value_curve_summary.csv")
    fig = plt.figure(figsize=(7.5, 4.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 1], hspace=0.72, wspace=0.48)

    ax = fig.add_subplot(grid[0, :])
    _panel(ax, "A")
    comparisons = [
        ("recent_ring_vs_random", "Historical case ring vs random"),
        ("recent_ring_vs_stable", "Historical case ring vs preparedness"),
        ("recent_ring_vs_static_ring", "Recency weighting vs cumulative ring"),
    ]
    base_y = np.array([2, 1, 0], dtype=float)
    for offset, model in [(-0.10, "temporal_sir"), (0.10, "temporal_seir_erlang")]:
        _, color, marker = MODELS[model]
        for yi, (contrast, _) in zip(base_y + offset, comparisons):
            row = ring[(ring.contrast.eq(contrast)) & (ring.epidemic_model.eq(model))].iloc[0]
            family_values = ring_family[
                (ring_family.contrast.eq(contrast))
                & (ring_family.epidemic_model.eq(model))
            ].mean_value
            _forest_row(ax, yi, row.family_equal_mean, family_values.min(), family_values.max(), color, marker)
    ax.set_yticks(base_y, [label for _, label in comparisons])
    ax.set_xlabel("Avoided final attack rate (points)")
    ax.set_title("What the detected case adds", weight="bold")
    _zero(ax)
    _model_legend(ax, "lower right")

    lower_axes = []
    for model, letter, cell in [("temporal_sir", "B", grid[1, 0]), ("temporal_seir_erlang", "C", grid[1, 1])]:
        ax = fig.add_subplot(cell)
        lower_axes.append(ax)
        ax.text(-0.02, 1.08, f"({letter.lower()})", transform=ax.transAxes, weight="bold", fontsize=9.2, va="bottom")
        part = wait[wait.epidemic_model.eq(model)]
        styles = [
            ("information_gain", "Information gained", GREEN, "D", "-"),
            ("delay_cost", "Burden accumulated while waiting", BURGUNDY, "s", "-"),
            ("net_wait_value", "Net value of waiting", INK, "^", (0, (3, 2))),
        ]
        for metric, label, color, marker, ls in styles:
            frame = part[part.metric.eq(metric)].sort_values("detection_fraction")
            x = frame.detection_fraction.to_numpy(float)
            y = 100 * frame.family_equal_mean.to_numpy(float)
            if metric == "delay_cost":
                y = -y
            ax.plot(x, y, color=color, marker=marker, ms=4.4, lw=1.2, ls=ls, label=label)
        ax.axhline(0, color=GREY, lw=0.7)
        ax.grid(color=LIGHT, lw=0.5)
        ax.set_xlabel("Decision time (mean infectious periods)")
        ax.set_title(MODELS[model][0], weight="bold")
    fig.text(
        0.035,
        0.335,
        "Contribution (points)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.6,
    )
    handles, labels = lower_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, columnspacing=1.2, handlelength=2.2)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.91)
    _save(fig, "Fig3.png")


def _resource_heatmap(ax: plt.Axes, frame: pd.DataFrame, model: str, title: str, cmap, vmin: float, vmax: float):
    part = frame[frame.epidemic_model.eq(model)].pivot(index="sentinel_fraction", columns="response_fraction", values="family_equal_mean")
    part = 100 * part.sort_index(ascending=False)
    image = ax.imshow(part.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(part.shape[0]):
        for j in range(part.shape[1]):
            value = part.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value > (vmin + vmax) / 2 else INK, fontsize=7.8)
    ax.set_xticks(range(part.shape[1]), [f"{int(100*x)}%" for x in part.columns])
    ax.set_yticks(range(part.shape[0]), [f"{int(100*x)}%" for x in part.index])
    ax.set_xlabel("Response capacity")
    ax.set_ylabel("Monitoring coverage")
    ax.set_title(title, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(0.7)
    return image


def figure_4() -> None:
    capacity = pd.read_csv(ROOT / "results/EXP-20260818-009/full/policy_summary.csv")
    recognition = pd.read_csv(ROOT / "results/EXP-20260818-010/full/detection_lever_contrast_summary.csv")
    cmap = LinearSegmentedColormap.from_list("burden", ["#F5F1E8", "#B99B5B", "#6E293D"])
    fig = plt.figure(figsize=(7.5, 4.05))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.80, 0.80, 1.40], wspace=0.88)
    heat_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    vmin = 100 * capacity.family_equal_mean.min()
    vmax = 100 * capacity.family_equal_mean.max()
    for ax, model, letter in zip(heat_axes, ["temporal_sir", "temporal_seir_erlang"], ["A", "B"]):
        _panel(ax, letter)
        image = _resource_heatmap(ax, capacity, model, MODELS[model][0], cmap, vmin, vmax)
    ax = fig.add_subplot(grid[0, 2])
    _panel(ax, "C")
    specs = [
        ("recognition_gain_at_5pct_monitoring", "Recognition\n50% to 100%"),
        ("monitoring_gain_at_50pct_recognition", "Monitoring\n5% to 20%"),
    ]
    base_y = np.array([1, 0], dtype=float)
    for offset, model in [(-0.10, "temporal_sir"), (0.10, "temporal_seir_erlang")]:
        _, color, marker = MODELS[model]
        for yi, (contrast, _) in zip(base_y + offset, specs):
            row = recognition[(recognition.contrast.eq(contrast)) & (recognition.epidemic_model.eq(model))].iloc[0]
            _forest_row(ax, yi, row.family_equal_mean, row.ci_low, row.ci_high, color, marker)
    ax.set_yticks(base_y, [label for _, label in specs])
    ax.tick_params(axis="y", labelsize=7.2, pad=3)
    ax.set_xlabel("Lower burden at detection (points)")
    ax.set_title("Detection levers", weight="bold")
    _zero(ax)
    handles = [
        Line2D([], [], marker=marker, lw=0, color=color, markerfacecolor=color, ms=5, label=label)
        for label, color, marker in MODELS.values()
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.80, 0.01), ncol=2)
    fig.text(0.28, 0.045, "Cell values are mean final attack rate (%)", ha="center", fontsize=7.6, color=GREY)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.86)
    _save(fig, "Fig4.png")


def figure_5() -> None:
    historical = pd.read_csv(ROOT / "results/EXP-20260820-011/full/contrast_summary.csv")
    historical_family = pd.read_csv(ROOT / "results/EXP-20260820-011/full/family_contrasts.csv")
    forecast = pd.read_csv(ROOT / "results/EXP-20260820-012/full/contrast_summary.csv")
    family = pd.read_csv(ROOT / "results/EXP-20260820-012/full/family_contrasts.csv")
    fig = plt.figure(figsize=(7.5, 4.35))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.04, 1.04, 0.96], wspace=1.08)

    ax = fig.add_subplot(grid[0, 0])
    _panel(ax, "A")
    contrasts = [
        ("history_exact_vs_singleton", "Exact past vs\nsingleton top-k"),
        ("history_exact_vs_stable", "Exact past vs\npreparedness"),
        ("history_exact_vs_static_ring", "Exact past vs\ncumulative ring"),
    ]
    base_y = np.array([2, 1, 0], dtype=float)
    for offset, model in [(-0.10, "temporal_sir"), (0.10, "temporal_seir_erlang")]:
        _, color, marker = MODELS[model]
        for yi, (contrast, _) in zip(base_y + offset, contrasts):
            row = historical[(historical.contrast.eq(contrast)) & (historical.epidemic_model.eq(model))].iloc[0]
            family_values = historical_family[
                (historical_family.contrast.eq(contrast))
                & (historical_family.epidemic_model.eq(model))
            ].mean_value
            _forest_row(ax, yi, row.family_equal_mean, family_values.min(), family_values.max(), color, marker)
    ax.set_yticks(base_y, [x[1] for x in contrasts])
    ax.tick_params(axis="y", labelsize=7.2, pad=2)
    ax.set_xlabel("Avoided final attack rate (points)")
    ax.set_title("Past-only planning", weight="bold", fontsize=8.8)
    _zero(ax)

    ax = fig.add_subplot(grid[0, 1])
    _panel(ax, "B")
    contrasts = [
        ("forecast_oracle_vs_history", "Future vs\nexact past"),
        ("forecast_oracle_vs_stable", "Future vs\npreparedness"),
        ("forecast_oracle_vs_ring", "Future vs\ncase ring"),
    ]
    for offset, model in [(-0.10, "temporal_sir"), (0.10, "temporal_seir_erlang")]:
        _, color, marker = MODELS[model]
        for yi, (contrast, _) in zip(base_y + offset, contrasts):
            row = forecast[(forecast.contrast.eq(contrast)) & (forecast.epidemic_model.eq(model))].iloc[0]
            family_values = family[
                (family.contrast.eq(contrast))
                & (family.epidemic_model.eq(model))
            ].mean_value
            _forest_row(ax, yi, row.family_equal_mean, family_values.min(), family_values.max(), color, marker)
    ax.set_yticks(base_y, [x[1] for x in contrasts])
    ax.tick_params(axis="y", labelsize=7.2, pad=2)
    ax.set_xlabel("Avoided final attack rate (points)")
    ax.set_title("Realized future contacts", weight="bold", fontsize=8.8)
    _zero(ax)

    ax = fig.add_subplot(grid[0, 2])
    _panel(ax, "C")
    part = family[(family.epidemic_model.eq("temporal_seir_erlang")) & (family.contrast.eq("forecast_oracle_vs_history"))].copy()
    part["label"] = part.system_family.map(FAMILY_LABELS).replace({"Wytham/songbirds": "Wytham family", "Oxford birds": "Oxford", "Radolfzell birds": "Radolfzell"})
    part = part.sort_values("mean_value")
    y = np.arange(len(part))
    values = 100 * part.mean_value
    bars = ax.barh(y, values, color=GREEN, height=0.56)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=7.2, color=INK)
    ax.set_yticks(y, part.label)
    ax.axvline(0, color=GREY, lw=0.7)
    ax.grid(axis="x", color=LIGHT, lw=0.5)
    ax.set_xlabel("Future-information gain (points)")
    ax.set_title("SEIR/Erlang family gains", weight="bold", fontsize=8.8)
    ax.set_xlim(0, max(1.75, float(values.max()) * 1.20))
    handles = [
        Line2D([], [], marker=marker, lw=0, color=color, markerfacecolor=color, ms=5, label=label)
        for label, color, marker in MODELS.values()
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.subplots_adjust(bottom=0.18, top=0.88)
    _save(fig, "Fig5.png")


def figure_6() -> None:
    summary = pd.read_csv(ROOT / "results/EXP-20260820-004/full/contrast_summary.csv")
    family = pd.read_csv(ROOT / "results/EXP-20260820-004/full/family_contrasts.csv")
    selections = pd.read_csv(ROOT / "results/EXP-20260820-004/full/past_model_selections.csv")
    fig = plt.figure(figsize=(7.5, 6.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], hspace=0.64, wspace=0.58)

    ax = fig.add_subplot(grid[0, 0])
    _panel(ax, "A")
    agreement = selections.groupby("system_family", observed=True).model_specific_agreement.mean()
    agreement.index = agreement.index.map(FAMILY_LABELS)
    agreement = agreement.rename(index={"Wytham/songbirds": "Wytham family", "Oxford birds": "Oxford", "Radolfzell birds": "Radolfzell"}).sort_values()
    y = np.arange(len(agreement))
    ax.barh(y, 100 * agreement.to_numpy(float), color=PURPLE, height=0.58)
    ax.set_yticks(y, agreement.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Identical past-selected sets (%)")
    ax.set_title("Selection agreement", weight="bold")
    ax.grid(axis="x", color=LIGHT, lw=0.55)

    ax = fig.add_subplot(grid[0, 1])
    _panel(ax, "B")
    contrasts = [
        ("matched_minus_cross_model", "Matched minus\ncross-model"),
        ("robust_regret_to_candidate_oracle", "Robust-plan\nregret"),
    ]
    base_y = np.array([1, 0], dtype=float)
    for offset, model in [(-0.10, "temporal_sir"), (0.10, "temporal_seir_erlang")]:
        _, color, marker = MODELS[model]
        for yi, (contrast, _) in zip(base_y + offset, contrasts):
            row = summary[(summary.contrast.eq(contrast)) & (summary.evaluator_model.eq(model))].iloc[0]
            _forest_row(ax, yi, row.family_equal_mean, row.ci_low, row.ci_high, color, marker)
    ax.set_yticks(base_y, [label for _, label in contrasts])
    ax.tick_params(axis="y", labelsize=7.2, pad=2)
    ax.set_xlabel("Avoided attack-rate difference (points)")
    ax.set_title("Future model-shift cost", weight="bold")
    _zero(ax)

    ax = fig.add_subplot(grid[1, :])
    _panel(ax, "C")
    selected = family.loc[family.contrast.eq("matched_minus_cross_model")].copy()
    family_order = sorted(selected.system_family.unique(), key=lambda value: FAMILY_LABELS.get(value, value))
    y = np.arange(len(family_order))
    for offset, model in [(-0.10, "temporal_sir"), (0.10, "temporal_seir_erlang")]:
        _, color, marker = MODELS[model]
        values = selected.loc[selected.evaluator_model.eq(model)].set_index("system_family").reindex(family_order).mean_value
        ax.scatter(100 * values, y + offset, marker=marker, color=color, s=26, zorder=3)
    labels = [FAMILY_LABELS.get(value, value) for value in family_order]
    labels = [{"Wytham/songbirds": "Wytham family", "Oxford birds": "Oxford", "Radolfzell birds": "Radolfzell"}.get(value, value) for value in labels]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Matched-minus-cross-model value (points)")
    ax.set_title("Family-level model matching effects", weight="bold")
    _zero(ax)
    handles = [
        Line2D([], [], marker=marker, lw=0, color=color, markerfacecolor=color, ms=5, label=label)
        for label, color, marker in MODELS.values()
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.11, top=0.93)
    _save(fig, "Fig6.png")


def main() -> None:
    _style()
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()


if __name__ == "__main__":
    main()
