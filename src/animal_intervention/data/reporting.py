from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .contract import CanonicalDataset
from .validation import ValidationReport


def _node_activity(dataset: CanonicalDataset) -> pd.Series:
    values: list[pd.Series] = []
    if len(dataset.dyadic_events):
        values.extend([dataset.dyadic_events["source_id"], dataset.dyadic_events["target_id"]])
    if len(dataset.group_memberships):
        values.append(dataset.group_memberships["node_id"])
    if not values:
        return pd.Series(dtype=float)
    return pd.concat(values, ignore_index=True).astype(str).value_counts()


def _event_times(dataset: CanonicalDataset) -> pd.Series:
    values = []
    if len(dataset.dyadic_events):
        values.append(pd.to_datetime(dataset.dyadic_events["start_time"], errors="coerce"))
    if len(dataset.group_events):
        values.append(pd.to_datetime(dataset.group_events["start_time"], errors="coerce"))
    return pd.concat(values, ignore_index=True).dropna() if values else pd.Series(dtype="datetime64[ns]")


def _durations(dataset: CanonicalDataset) -> pd.Series:
    values = []
    if len(dataset.dyadic_events):
        values.append(pd.to_numeric(dataset.dyadic_events["duration_seconds"], errors="coerce"))
    if len(dataset.group_events):
        values.append(pd.to_numeric(dataset.group_events["duration_seconds"], errors="coerce"))
    return pd.concat(values, ignore_index=True).dropna() if values else pd.Series(dtype=float)


def save_quality_figure(dataset: CanonicalDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle(f"{dataset.metadata.dataset_id}: canonical data audit", fontsize=16)

    event_times = _event_times(dataset)
    if len(event_times):
        axes[0, 0].hist(event_times, bins=min(50, max(5, int(np.sqrt(len(event_times))))), color="#2878B5")
        axes[0, 0].set_xlabel("event start time")
        axes[0, 0].tick_params(axis="x", rotation=25)
        axes[0, 0].set_title("Temporal coverage")
    else:
        axes[0, 0].text(0.5, 0.5, "No recoverable timestamps", ha="center", va="center")
        axes[0, 0].set_title("Temporal coverage limitation")

    durations = _durations(dataset)
    positive_durations = durations[durations > 0]
    if len(positive_durations):
        axes[0, 1].hist(np.log10(positive_durations), bins=40, color="#39A96B")
        axes[0, 1].set_xlabel("log10(duration seconds)")
        axes[0, 1].set_title("Event duration distribution")
    else:
        axes[0, 1].text(0.5, 0.5, "No positive durations", ha="center", va="center")

    activity = _node_activity(dataset)
    if len(activity):
        axes[1, 0].hist(np.log10(activity), bins=35, color="#F39C35")
        axes[1, 0].set_xlabel("log10(observed events per node)")
        axes[1, 0].set_title("Node observation/activity coverage")

    if len(dataset.group_memberships):
        group_sizes = dataset.group_memberships.groupby("group_event_id", observed=True).size()
        axes[1, 1].hist(group_sizes, bins=min(40, int(group_sizes.max())), color="#8E5AB5")
        axes[1, 1].set_xlabel("observed members")
        axes[1, 1].set_title("Group-event size distribution")
    elif len(dataset.dyadic_events):
        semantics = dataset.dyadic_events["edge_semantics"].value_counts()
        axes[1, 1].bar(semantics.index.astype(str), semantics.values, color="#8E5AB5")
        axes[1, 1].tick_params(axis="x", rotation=25)
        axes[1, 1].set_title("Dyadic event semantics")
    else:
        axes[1, 1].text(0.5, 0.5, "No interaction records", ha="center", va="center")

    for axis in axes.flat:
        axis.set_ylabel("count")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_quality_report(
    dataset: CanonicalDataset,
    validation: ValidationReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{dataset.metadata.dataset_id}.md"
    figure_path = output_dir / f"{dataset.metadata.dataset_id}.png"
    save_quality_figure(dataset, figure_path)
    summary = dataset.summary()
    severity_counts = pd.Series(
        [issue.severity for issue in validation.issues], dtype="string"
    ).value_counts()
    lines = [
        f"# Data quality audit: `{dataset.metadata.dataset_id}`",
        "",
        "## Dataset and grain",
        "",
        f"- Primary event mode: `{dataset.metadata.primary_event_mode}`",
        f"- Temporal order available: `{dataset.metadata.has_temporal_order}`",
        f"- Individuals: {summary['individuals']:,}",
        f"- Dyadic events: {summary['dyadic_events']:,}",
        f"- Group events: {summary['group_events']:,}",
        f"- Group memberships: {summary['group_memberships']:,}",
        f"- Time range: `{summary['time_start']}` to `{summary['time_end']}`",
        "",
        "## Checks performed",
        "",
        "Required columns; node/event uniqueness; endpoint and membership integrity; self-loops; "
        "time ordering; positive durations; finite measurements; group membership reconciliation.",
        "",
        "## Findings",
        "",
    ]
    if validation.issues:
        lines.extend(
            f"- **{issue.severity.upper()} · {issue.code}** — {issue.message} "
            f"(count={issue.count:,}{f', fraction={issue.fraction:.3%}' if issue.fraction is not None else ''})"
            for issue in validation.issues
        )
    else:
        lines.append("- No schema or domain invariant failures detected.")
    lines.extend(
        [
            "",
            "## Downstream use",
            "",
            (
                "This dataset is eligible for temporal exposure mapping and simulation."
                if dataset.metadata.has_temporal_order and not validation.has_errors
                else "This dataset is not eligible for primary temporal simulation without resolving the issues above."
            ),
            "",
            f"![Canonical data audit]({figure_path.name})",
            "",
            "## Audit metadata",
            "",
            f"- Severity counts: `{json.dumps(severity_counts.to_dict(), default=int)}`",
            f"- Adapter: `{dataset.metadata.adapter_name}` version `{dataset.metadata.adapter_version}`",
            "- Detection-span observation windows are not interpreted as guaranteed continuous device uptime.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, figure_path
