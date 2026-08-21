from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = ROOT / "reports" / "manuscript" / "figure_1_prospective_contract.png"


def _box(ax, x, y, width, height, title, body, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + height - 0.055, title, fontsize=12, fontweight="bold", color=color)
    ax.text(x + 0.025, y + height - 0.115, body, fontsize=9.3, color="#30343b", va="top", linespacing=1.35)


def _arrow(ax, start, end, color="#5a616a", style="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.7,
        linestyle=style,
        color=color,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.055, 0.945, "Prospective temporal-animal epidemic decision contract", fontsize=22, fontweight="bold")
    fig.text(
        0.055,
        0.905,
        "Deployment uses observed history only; future contacts are reserved for paired offline evaluation",
        fontsize=12.5,
        color="#5a616a",
    )

    _box(ax, 0.05, 0.59, 0.20, 0.22, "1  Observed history", "Canonical temporal events\nDataset-specific semantics\nEligibility and support", "#2b6cb0")
    _box(ax, 0.30, 0.59, 0.20, 0.22, "2  Preparedness", "History-ranked watchlist\nNested sentinel capacity\nNo future leakage", "#2f855a")
    _box(ax, 0.55, 0.59, 0.20, 0.22, "3  Endogenous detection", "Sentinel becomes infected\nRecognition sensitivity\nExplicit action delay", "#b7791f")
    _box(ax, 0.80, 0.59, 0.16, 0.22, "4  Response", "Cases + limited targets\nHistory or reactive rule\nContact reduction", "#c05621")

    _arrow(ax, (0.25, 0.70), (0.30, 0.70))
    _arrow(ax, (0.50, 0.70), (0.55, 0.70))
    _arrow(ax, (0.75, 0.70), (0.80, 0.70))

    ax.plot([0.05, 0.96], [0.52, 0.52], color="#d7dce2", linewidth=1.3)
    ax.text(0.05, 0.535, "OFFLINE PAIRED REPLAY", fontsize=9.5, fontweight="bold", color="#6b7280")

    _box(ax, 0.05, 0.19, 0.25, 0.24, "Shared natural worlds", "Same index infection\nSame future temporal contacts\nSame keyed random primitives", "#5b5fc7")
    _box(ax, 0.375, 0.19, 0.25, 0.24, "Role-specific outcomes", "Source influence\nDetection burden\nAvoided final attack rate", "#805ad5")
    _box(ax, 0.70, 0.19, 0.26, 0.24, "Decision and transfer gates", "Family-equal uncertainty\nAbsolute + relative safety\nWhole-system holdout or abstain", "#9f3a56")

    _arrow(ax, (0.30, 0.31), (0.375, 0.31))
    _arrow(ax, (0.625, 0.31), (0.70, 0.31))
    _arrow(ax, (0.88, 0.59), (0.88, 0.43), color="#9f3a56", style="--")
    _arrow(ax, (0.18, 0.59), (0.18, 0.43), color="#5b5fc7", style="--")

    fig.text(
        0.055,
        0.065,
        "Supported deployment boundary: act early from transparent history, improve monitoring and recognition, "
        "calibrate locally when possible, and abstain from unsupported universal optimization.",
        fontsize=11.5,
        color="#30343b",
    )
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
