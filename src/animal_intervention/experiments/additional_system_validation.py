from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.estimands.intervention_value import AnchorWindow, node_support, slice_stream
from animal_intervention.transmission.contract import ExposureStream
from animal_intervention.transmission.mappers import DurationContactMapper, GroupMixingMapper

from .g1_sim import _save_figure
from .wytham_validation import run as run_validation


def _restrict_population(stream: ExposureStream, nodes: list[str]) -> ExposureStream:
    retained = set(map(str, nodes))
    dyadic = stream.dyadic_exposures.loc[
        stream.dyadic_exposures["source_id"].astype(str).isin(retained)
        & stream.dyadic_exposures["target_id"].astype(str).isin(retained)
    ].copy()
    memberships = stream.group_memberships.loc[
        stream.group_memberships["node_id"].astype(str).isin(retained)
    ].copy()
    sizes = memberships.groupby("group_event_id", observed=True).size()
    group_ids = set(sizes.loc[sizes.ge(2)].index.astype(str))
    memberships = memberships.loc[
        memberships["group_event_id"].astype(str).isin(group_ids)
    ].copy()
    groups = stream.group_exposures.loc[
        stream.group_exposures["group_event_id"].astype(str).isin(group_ids)
    ].copy()
    result = ExposureStream(
        dataset_id=stream.dataset_id,
        population_nodes=tuple(sorted(retained)),
        dyadic_exposures=dyadic,
        group_exposures=groups,
        group_memberships=memberships,
        metadata=dict(stream.metadata),
    )
    result.validate()
    return result


def _complete_windows(
    stream: ExposureStream,
    bounds: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    windows: dict[str, Any],
    max_anchors: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    lookback = pd.Timedelta(str(windows["lookback"]))
    horizon = pd.Timedelta(str(windows["horizon"]))
    step = pd.Timedelta(str(windows["step"]))
    minimum_support = int(windows["min_history_events_per_node"])
    prepared: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for unit_id, unit_start, unit_end in bounds:
        anchor_time = unit_start + lookback
        local_index = 0
        while anchor_time + horizon <= unit_end and len(prepared) < max_anchors:
            history_start = anchor_time - lookback
            history = slice_stream(stream, history_start, anchor_time)
            history_support = node_support(history)
            eligible = sorted(
                str(node)
                for node, support in history_support.items()
                if int(support) >= minimum_support
            )
            if len(eligible) >= 2:
                future = _restrict_population(
                    slice_stream(stream, anchor_time, anchor_time + horizon), eligible
                )
                future_support = node_support(future)
                if int((future_support > 0).sum()) >= 2:
                    local_index += 1
                    anchor_id = f"{unit_id}::anchor_{local_index:03d}"
                    anchor = AnchorWindow(
                        anchor_id=anchor_id,
                        history_start=history_start,
                        anchor_time=anchor_time,
                        horizon_end=anchor_time + horizon,
                    )
                    prepared.append(
                        {
                            "anchor": anchor,
                            "future": future,
                            "eligible": eligible,
                            "history_support": history_support,
                            "future_support": future_support,
                            "population_size": len(eligible),
                            "network_id": unit_id,
                            "observation_unit_id": unit_id,
                            "history_weekend_spans": [(history_start, anchor_time)],
                            "future_active_count": int((future_support > 0).sum()),
                        }
                    )
                    metadata.append(
                        {
                            "dataset_id": stream.dataset_id,
                            "network_id": unit_id,
                            "observation_unit_id": unit_id,
                            "anchor_id": anchor_id,
                            "history_start": history_start,
                            "anchor_time": anchor_time,
                            "horizon_end": anchor_time + horizon,
                            "eligible_animals": len(eligible),
                            "future_active_animals": int((future_support > 0).sum()),
                        }
                    )
            anchor_time += step
        if len(prepared) >= max_anchors:
            break
    if not prepared:
        raise ValueError("No complete, transmission-capable validation windows were found")
    return prepared, pd.DataFrame(metadata)


def _bat_stream(dataset: CanonicalDataset, config: dict[str, Any]) -> ExposureStream:
    del config
    base = DurationContactMapper().compile(dataset)
    result = ExposureStream(
        dataset_id=base.dataset_id,
        population_nodes=tuple(sorted(dataset.individuals["node_id"].astype(str))),
        dyadic_exposures=base.dyadic_exposures,
        metadata={**base.metadata, "network_id": "lamanai_roost"},
    )
    result.validate()
    return result


def _bat_windows(
    stream: ExposureStream, windows: dict[str, Any], max_anchors: int
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    frame = stream.dyadic_exposures
    bounds = [
        (
            "lamanai_roost",
            pd.Timestamp(frame["start_time"].min()),
            pd.Timestamp(frame["end_time"].max()),
        )
    ]
    return _complete_windows(stream, bounds, windows, max_anchors)


def _sheep_stream(dataset: CanonicalDataset, config: dict[str, Any]) -> ExposureStream:
    mode = str(config["data"].get("group_mixing_mode", "frequency_dependent"))
    base = GroupMixingMapper(mode=mode).compile(dataset)
    groups = base.group_exposures.copy()
    for column in ("start_time", "end_time"):
        values = pd.to_datetime(groups[column], utc=True)
        groups[column] = values.dt.tz_convert(None)
    result = ExposureStream(
        dataset_id=base.dataset_id,
        population_nodes=tuple(sorted(dataset.individuals["node_id"].astype(str))),
        group_exposures=groups,
        group_memberships=base.group_memberships,
        metadata={**base.metadata, "source_time_zone": "UTC"},
    )
    result.validate()
    return result


def _sheep_bounds(stream: ExposureStream) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    intervals = (
        stream.group_exposures[["start_time", "end_time"]]
        .drop_duplicates()
        .sort_values(["start_time", "end_time"])
        .reset_index(drop=True)
    )
    bounds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    start = pd.Timestamp(intervals.iloc[0]["start_time"])
    end = pd.Timestamp(intervals.iloc[0]["end_time"])
    for row in intervals.iloc[1:].itertuples(index=False):
        next_start = pd.Timestamp(row.start_time)
        next_end = pd.Timestamp(row.end_time)
        if next_start - end > pd.Timedelta(seconds=12):
            bounds.append((f"recording_segment_{len(bounds) + 1}", start, end))
            start = next_start
        end = max(end, next_end)
    bounds.append((f"recording_segment_{len(bounds) + 1}", start, end))
    return bounds


def _sheep_windows(
    stream: ExposureStream, windows: dict[str, Any], max_anchors: int
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    return _complete_windows(stream, _sheep_bounds(stream), windows, max_anchors)


def _audit(
    dataset: CanonicalDataset,
    stream: ExposureStream,
    prepared: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    temporal = (
        stream.dyadic_exposures
        if not stream.dyadic_exposures.empty
        else stream.group_exposures
    ).copy()
    temporal["date"] = pd.to_datetime(temporal["start_time"]).dt.floor("D")
    if not stream.dyadic_exposures.empty:
        daily = (
            temporal.groupby("date", observed=True)
            .agg(group_events=("exposure_id", "size"))
            .reset_index()
        )
        active = pd.concat(
            [
                temporal[["date", "source_id"]].rename(columns={"source_id": "node_id"}),
                temporal[["date", "target_id"]].rename(columns={"target_id": "node_id"}),
            ]
        )
        daily = daily.merge(
            active.groupby("date", observed=True)["node_id"]
            .nunique()
            .rename("active_host_individuals")
            .reset_index(),
            on="date",
        )
        daily["season"] = "lamanai_roost"
        sizes = pd.DataFrame({"host_group_size": [2] * len(temporal)})
        event_count = len(stream.dyadic_exposures)
        overlap_keys = ["source_id", "target_id"]
        overlap_free = True
        for _, group in stream.dyadic_exposures.sort_values("start_time").groupby(
            overlap_keys, observed=True
        ):
            if (pd.to_datetime(group["start_time"]).iloc[1:].reset_index(drop=True)
                < pd.to_datetime(group["end_time"]).iloc[:-1].reset_index(drop=True)).any():
                overlap_free = False
                break
    else:
        memberships = stream.group_memberships.merge(
            temporal[["group_event_id", "date"]], on="group_event_id", how="left"
        )
        daily = (
            temporal.groupby("date", observed=True)
            .agg(group_events=("group_event_id", "size"))
            .reset_index()
        )
        daily = daily.merge(
            memberships.groupby("date", observed=True)["node_id"]
            .nunique()
            .rename("active_host_individuals")
            .reset_index(),
            on="date",
        )
        daily["season"] = "free_ranging_flock"
        sizes = (
            stream.group_memberships.groupby("group_event_id", observed=True)
            .size()
            .rename("host_group_size")
            .reset_index(drop=True)
            .to_frame()
        )
        event_count = len(stream.group_exposures)
        overlap_free = True
    valid_intervals = bool(
        (pd.to_datetime(temporal["end_time"]) > pd.to_datetime(temporal["start_time"])).all()
    )
    checks = {
        "canonical_dataset_validated": True,
        "positive_exposure_intervals": valid_intervals,
        "at_least_two_complete_windows": len(prepared) >= 2,
        "all_windows_have_two_candidates": all(len(item["eligible"]) >= 2 for item in prepared),
        "all_windows_have_future_exposure": all(
            item["future_active_count"] >= 2 for item in prepared
        ),
        "dyadic_intervals_do_not_overlap_within_pair": overlap_free,
    }
    audit = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "registered_individuals": int(len(dataset.individuals)),
        "observed_group_stream_individuals": int(len(stream.nodes())),
        "exposure_events": int(event_count),
        "complete_validation_windows": int(len(prepared)),
        "eligible_population_range": [
            int(min(len(item["eligible"]) for item in prepared)),
            int(max(len(item["eligible"]) for item in prepared)),
        ],
    }
    return audit, daily, sizes


def _units(anchor_metadata: pd.DataFrame) -> pd.DataFrame:
    units = anchor_metadata[["dataset_id", "network_id"]].drop_duplicates().copy()
    units["evaluation_unit_id"] = units["dataset_id"] + "::" + units["network_id"]
    units["independent_unit_type"] = "single_animal_system"
    units["split_constraint"] = "keep_all_anchors_in_one_animal_system_family"
    return units.reset_index(drop=True)


def _plot_quality(
    daily: pd.DataFrame,
    group_sizes: pd.DataFrame,
    audit: dict[str, Any],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].plot(daily["date"], daily["active_host_individuals"], color="#0072B2")
    axes[0].set_title("Observed individuals by day")
    axes[0].set_ylabel("Individuals")
    axes[1].plot(daily["date"], daily["group_events"], color="#8E5A2B")
    axes[1].set_title("Exposure events by day")
    axes[1].set_ylabel("Events")
    axes[2].hist(group_sizes["host_group_size"], color="#6A3D9A", bins="auto")
    axes[2].set_title("Exposure group size")
    axes[2].set_xlabel("Individuals")
    axes[2].set_ylabel("Events")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Primary-stream audit, {audit['registered_individuals']} registered animals",
        fontsize=15,
        weight="bold",
    )
    _save_figure(figure, path)


def run(config_path: Path, profile: str) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_id = str(config["data"]["dataset_id"])
    if dataset_id == "wild_vampire_bats_proximity":
        stream_builder, window_builder = _bat_stream, _bat_windows
    elif dataset_id == "free_ranging_sheep_fission_fusion":
        stream_builder, window_builder = _sheep_stream, _sheep_windows
    else:
        raise ValueError(f"Unsupported additional animal system: {dataset_id}")
    return run_validation(
        config_path,
        profile,
        stream_builder=stream_builder,
        window_builder=window_builder,
        audit_builder=_audit,
        evaluation_units_builder=_units,
        data_quality_plotter=_plot_quality,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate additional animal systems")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    arguments = parser.parse_args()
    run(arguments.config, arguments.profile)


if __name__ == "__main__":
    main()
