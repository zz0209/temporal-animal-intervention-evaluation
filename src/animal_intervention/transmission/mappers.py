from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

import numpy as np
import pandas as pd

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.data.validation import require_temporal_capability

from .contract import ExposureStream


def _dyadic_stream(
    dataset: CanonicalDataset,
    selected: pd.DataFrame,
    *,
    mapper_name: str,
    route: str,
    multipliers: pd.Series | float,
    beta_unit: str,
) -> ExposureStream:
    if selected.empty:
        raise ValueError(f"{mapper_name} found no compatible dyadic events")
    duplicate_source = (
        selected["quality_flag"]
        .fillna("")
        .astype(str)
        .str.contains("exact_duplicate_source_record", regex=False)
    )
    excluded_duplicate_count = int(duplicate_source.sum())
    selected = selected.loc[~duplicate_source].copy()
    if isinstance(multipliers, pd.Series):
        multipliers = multipliers.loc[~duplicate_source]
    starts = pd.to_datetime(selected["start_time"], errors="coerce")
    ends = pd.to_datetime(selected["end_time"], errors="coerce")
    usable = starts.notna() & ends.notna() & ends.gt(starts)
    excluded_count = int((~usable).sum())
    selected = selected.loc[usable].copy()
    if isinstance(multipliers, pd.Series):
        multipliers = multipliers.loc[usable]
    if selected.empty:
        raise ValueError(f"{mapper_name} found no positive, time-located exposure intervals")
    if isinstance(multipliers, (int, float)):
        multipliers = pd.Series(float(multipliers), index=selected.index)
    exposures = pd.DataFrame(
        {
            "dataset_id": dataset.metadata.dataset_id,
            "exposure_id": selected["event_id"].map(lambda value: f"exposure:{value}"),
            "source_id": selected["source_id"].astype(str),
            "target_id": selected["target_id"].astype(str),
            "start_time": pd.to_datetime(selected["start_time"]),
            "end_time": pd.to_datetime(selected["end_time"]),
            "hazard_rate_multiplier": pd.to_numeric(multipliers, errors="coerce"),
            "directed": selected["directed"].astype("boolean").fillna(False).astype(bool),
            "transmission_route": route,
            "mapper_name": mapper_name,
            "origin_event_id": selected["event_id"].astype(str),
            "location_id": selected["location_id"],
        }
    )
    stream = ExposureStream(
        dataset_id=dataset.metadata.dataset_id,
        dyadic_exposures=exposures,
        metadata={
            "mapper": mapper_name,
            "beta_unit": beta_unit,
            "excluded_exact_source_duplicates": excluded_duplicate_count,
            "excluded_nonpositive_or_missing_intervals": excluded_count,
        },
    )
    stream.validate()
    return stream


@dataclass(frozen=True, slots=True)
class DurationContactMapper:
    edge_semantics: tuple[str, ...] = ("sensor_proximity",)
    intensity_multiplier: float = 1.0

    def compile(self, dataset: CanonicalDataset) -> ExposureStream:
        require_temporal_capability(dataset)
        events = dataset.dyadic_events
        selected = events.loc[
            events["measurement_type"].eq("duration")
            & events["edge_semantics"].isin(self.edge_semantics)
        ].copy()
        return _dyadic_stream(
            dataset,
            selected,
            mapper_name=type(self).__name__,
            route="direct_contact",
            multipliers=self.intensity_multiplier,
            beta_unit="per_second_of_observed_interval",
        )


def coalesce_overlapping_dyadic_intervals(
    stream: ExposureStream, *, mapper_name: str
) -> ExposureStream:
    """Replace overlapping records for one dyad/location with their time union."""

    if stream.dyadic_exposures.empty:
        raise ValueError("cannot coalesce an empty dyadic exposure stream")
    frame = stream.dyadic_exposures.sort_values(
        [
            "source_id",
            "target_id",
            "directed",
            "transmission_route",
            "location_id",
            "start_time",
            "end_time",
        ]
    )
    group_columns = [
        "source_id",
        "target_id",
        "directed",
        "transmission_route",
        "location_id",
    ]
    rows: list[dict[str, object]] = []
    merged_source_records = 0
    merged_clusters = 0
    removed_overlap_seconds = 0.0
    exposure_index = 0
    for group_key, group in frame.groupby(
        group_columns, observed=True, sort=False, dropna=False
    ):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None
        current_origins: list[str] = []
        summed_seconds = 0.0
        hazard_multiplier: float | None = None

        def emit() -> None:
            nonlocal exposure_index, merged_source_records, merged_clusters
            nonlocal removed_overlap_seconds
            if current_start is None or current_end is None or hazard_multiplier is None:
                return
            union_seconds = (current_end - current_start).total_seconds()
            removed_overlap_seconds += summed_seconds - union_seconds
            if len(current_origins) > 1:
                merged_clusters += 1
                merged_source_records += len(current_origins) - 1
            values = dict(zip(group_columns, group_key))
            rows.append(
                {
                    "dataset_id": stream.dataset_id,
                    "exposure_id": f"coalesced:{exposure_index}",
                    "source_id": values["source_id"],
                    "target_id": values["target_id"],
                    "start_time": current_start,
                    "end_time": current_end,
                    "hazard_rate_multiplier": hazard_multiplier,
                    "directed": values["directed"],
                    "transmission_route": values["transmission_route"],
                    "mapper_name": mapper_name,
                    "origin_event_id": json.dumps(current_origins, separators=(",", ":")),
                    "location_id": values["location_id"],
                }
            )
            exposure_index += 1

        for row in group.itertuples(index=False):
            start = pd.Timestamp(row.start_time)
            end = pd.Timestamp(row.end_time)
            multiplier = float(row.hazard_rate_multiplier)
            if current_start is None or start > current_end:
                emit()
                current_start = start
                current_end = end
                current_origins = [str(row.origin_event_id)]
                summed_seconds = (end - start).total_seconds()
                hazard_multiplier = multiplier
            else:
                if multiplier != hazard_multiplier:
                    raise ValueError(
                        "overlapping intervals with different hazard multipliers cannot be coalesced"
                    )
                current_end = max(current_end, end)
                current_origins.append(str(row.origin_event_id))
                summed_seconds += (end - start).total_seconds()
        emit()

    metadata = dict(stream.metadata)
    metadata.update(
        {
            "mapper": mapper_name,
            "input_exposures_before_coalescing": int(len(frame)),
            "coalesced_exposures": int(len(rows)),
            "merged_source_exposure_count": int(merged_source_records),
            "overlapping_or_touching_clusters": int(merged_clusters),
            "overlap_duration_removed_seconds": float(removed_overlap_seconds),
        }
    )
    result = ExposureStream(
        dataset_id=stream.dataset_id,
        dyadic_exposures=pd.DataFrame(rows),
        metadata=metadata,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class CoalescedDurationContactMapper:
    edge_semantics: tuple[str, ...] = ("sensor_proximity",)
    intensity_multiplier: float = 1.0

    def compile(self, dataset: CanonicalDataset) -> ExposureStream:
        base = DurationContactMapper(
            edge_semantics=self.edge_semantics,
            intensity_multiplier=self.intensity_multiplier,
        ).compile(dataset)
        return coalesce_overlapping_dyadic_intervals(
            base, mapper_name=type(self).__name__
        )


@dataclass(frozen=True, slots=True)
class DetectionIntervalMapper:
    edge_semantics: tuple[str, ...] = ("sensor_proximity",)
    intensity_multiplier: float = 1.0

    def compile(self, dataset: CanonicalDataset) -> ExposureStream:
        require_temporal_capability(dataset)
        events = dataset.dyadic_events
        selected = events.loc[
            events["event_representation"].eq("fixed_bin")
            & events["edge_semantics"].isin(self.edge_semantics)
        ].copy()
        return _dyadic_stream(
            dataset,
            selected,
            mapper_name=type(self).__name__,
            route="direct_contact_proxy",
            multipliers=self.intensity_multiplier,
            beta_unit="per_second_of_detected_proximity",
        )


@dataclass(frozen=True, slots=True)
class AggregatedAssociationMapper:
    scale: float = 1.0
    transform: Literal["linear", "saturating"] = "linear"

    def compile(self, dataset: CanonicalDataset) -> ExposureStream:
        require_temporal_capability(dataset)
        events = dataset.dyadic_events
        selected = events.loc[events["measurement_type"].eq("association_index")].copy()
        weights = pd.to_numeric(selected["measurement_value"], errors="coerce")
        transformed = weights if self.transform == "linear" else 1.0 - np.exp(-weights)
        durations = (
            pd.to_datetime(selected["end_time"]) - pd.to_datetime(selected["start_time"])
        ).dt.total_seconds()
        rate_multiplier = self.scale * transformed / durations
        return _dyadic_stream(
            dataset,
            selected,
            mapper_name=type(self).__name__,
            route="association_proxy",
            multipliers=rate_multiplier,
            beta_unit="per_unit_integrated_association",
        )


@dataclass(frozen=True, slots=True)
class GroupMixingMapper:
    mode: Literal["frequency_dependent", "undiluted_clique"] = "frequency_dependent"
    intensity_multiplier: float = 1.0

    def compile(self, dataset: CanonicalDataset) -> ExposureStream:
        require_temporal_capability(dataset)
        if dataset.group_events.empty or dataset.group_memberships.empty:
            raise ValueError(f"{dataset.metadata.dataset_id} contains no group events")
        groups = dataset.group_events.copy()
        starts = pd.to_datetime(groups["start_time"], errors="coerce")
        ends = pd.to_datetime(groups["end_time"], errors="coerce")
        usable = starts.notna() & ends.notna() & ends.gt(starts)
        excluded_count = int((~usable).sum())
        groups = groups.loc[usable].copy()
        if groups.empty:
            raise ValueError(f"{dataset.metadata.dataset_id} contains no positive group intervals")
        exposures = pd.DataFrame(
            {
                "dataset_id": dataset.metadata.dataset_id,
                "group_event_id": groups["group_event_id"].astype(str),
                "start_time": pd.to_datetime(groups["start_time"]),
                "end_time": pd.to_datetime(groups["end_time"]),
                "hazard_rate_multiplier": float(self.intensity_multiplier),
                "transmission_route": "group_association_proxy",
                "mapper_name": type(self).__name__,
                "group_mixing_mode": self.mode,
                "location_id": groups["location_id"],
            }
        )
        memberships = dataset.group_memberships.loc[
            dataset.group_memberships["group_event_id"].isin(groups["group_event_id"]),
            ["dataset_id", "group_event_id", "node_id", "membership_weight"]
        ].copy()
        stream = ExposureStream(
            dataset_id=dataset.metadata.dataset_id,
            group_exposures=exposures,
            group_memberships=memberships,
            metadata={
                "mapper": type(self).__name__,
                "mode": self.mode,
                "beta_unit": "per_second_of_group_coattendance",
                "excluded_nonpositive_or_missing_intervals": excluded_count,
            },
        )
        stream.validate()
        return stream


def compile_primary_exposure(dataset: CanonicalDataset) -> ExposureStream:
    """Choose the conservative primary mapper from canonical semantics.

    Selection is contract-driven rather than dataset-name-driven. Mixed-modality
    datasets default to the sensor-proximity stream; direct behavior remains an
    explicitly requested sensitivity arm and is never silently pooled.
    """
    if not dataset.group_events.empty:
        return GroupMixingMapper(mode="frequency_dependent").compile(dataset)
    measurement_types = set(dataset.dyadic_events["measurement_type"].dropna().astype(str))
    representations = set(
        dataset.dyadic_events["event_representation"].dropna().astype(str)
    )
    if "association_index" in measurement_types:
        return AggregatedAssociationMapper().compile(dataset)
    if "fixed_bin" in representations:
        return DetectionIntervalMapper().compile(dataset)
    if "duration" in measurement_types:
        return DurationContactMapper().compile(dataset)
    raise ValueError(
        f"{dataset.metadata.dataset_id} has no registered primary transmission mapping"
    )


def compile_named_exposure(
    dataset: CanonicalDataset,
    mapper_name: str,
) -> ExposureStream:
    """Compile the mapper recorded by a model-facing label contract."""
    factories = {
        "DurationContactMapper": DurationContactMapper,
        "CoalescedDurationContactMapper": CoalescedDurationContactMapper,
        "DetectionIntervalMapper": DetectionIntervalMapper,
        "AggregatedAssociationMapper": AggregatedAssociationMapper,
        "GroupMixingMapper": GroupMixingMapper,
    }
    if mapper_name not in factories:
        raise ValueError(f"unsupported recorded primary mapper: {mapper_name}")
    stream = factories[mapper_name]().compile(dataset)
    if stream.metadata.get("mapper") != mapper_name:
        raise ValueError(
            "compiled exposure mapper does not match the recorded primary mapper"
        )
    return stream
