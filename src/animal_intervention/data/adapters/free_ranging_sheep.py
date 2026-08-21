from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pandas as pd

from ..contract import CanonicalDataset, DatasetMetadata
from .base import (
    BaseAdapter,
    apply_observation_bounds,
    make_individuals,
    observation_windows_from_groups,
)


class FreeRangingSheepAdapter(BaseAdapter):
    dataset_id = "free_ranging_sheep_fission_fusion"
    native_resolution_seconds = 6.0
    gap_threshold_seconds = 12.0

    @staticmethod
    def _parse_partition(row: list[str]) -> tuple[tuple[str, ...], ...]:
        groups = [tuple(sorted(str(value) for value in ast.literal_eval(cell))) for cell in row]
        return tuple(sorted(groups))

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        del progress
        raw_dir = Path(raw_dir)
        groups_path = raw_dir / "GroupsPerTime.csv"
        time_path = raw_dir / "ts.csv"
        times = pd.read_csv(time_path)
        timestamps = pd.to_datetime(times["x"], unit="s", utc=True)
        maximum_steps = 5_000 if sample else len(timestamps)

        partitions: list[tuple[tuple[str, ...], ...]] = []
        with groups_path.open(newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.reader(stream)):
                if index >= maximum_steps:
                    break
                partitions.append(self._parse_partition(row))
        timestamps = timestamps.iloc[: len(partitions)].reset_index(drop=True)
        if len(partitions) != len(timestamps):
            raise ValueError("Grouping rows and timestamps do not reconcile")

        runs: list[tuple[int, int, int, tuple[tuple[str, ...], ...]]] = []
        run_start = 0
        segment_id = 0
        for index in range(1, len(partitions)):
            gap_seconds = (timestamps.iloc[index] - timestamps.iloc[index - 1]).total_seconds()
            discontinuity = gap_seconds > self.gap_threshold_seconds
            if partitions[index] != partitions[index - 1] or discontinuity:
                runs.append((run_start, index, segment_id, partitions[index - 1]))
                run_start = index
            if discontinuity:
                segment_id += 1
        if partitions:
            runs.append((run_start, len(partitions), segment_id, partitions[-1]))

        group_rows: list[dict[str, object]] = []
        membership_rows: list[dict[str, object]] = []
        for run_index, (start_index, stop_index, local_segment, partition) in enumerate(runs):
            start_time = timestamps.iloc[start_index]
            if stop_index < len(timestamps):
                observed_gap = (timestamps.iloc[stop_index] - timestamps.iloc[stop_index - 1]).total_seconds()
                end_time = (
                    timestamps.iloc[stop_index]
                    if observed_gap <= self.gap_threshold_seconds
                    else timestamps.iloc[stop_index - 1]
                    + pd.Timedelta(seconds=self.native_resolution_seconds)
                )
            else:
                end_time = timestamps.iloc[-1] + pd.Timedelta(
                    seconds=self.native_resolution_seconds
                )
            duration_seconds = (end_time - start_time).total_seconds()
            for component_index, members in enumerate(partition):
                group_event_id = f"sheep-group:{run_index:06d}:{component_index:02d}"
                group_rows.append(
                    {
                        "dataset_id": self.dataset_id,
                        "group_event_id": group_event_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_seconds": duration_seconds,
                        "location_id": "salt6_paddock",
                        "event_semantics": "author_inferred_spatial_group",
                        "native_time_resolution_seconds": self.native_resolution_seconds,
                        "source_record_id": f"{start_index + 1}:{stop_index}",
                        "quality_flag": "ok",
                        "attributes_json": json.dumps(
                            {
                                "partition_run": run_index,
                                "component_index": component_index,
                                "continuous_segment": local_segment,
                                "near_radius_m": 30,
                                "far_radius_m": 50,
                            },
                            separators=(",", ":"),
                        ),
                    }
                )
                membership_rows.extend(
                    {
                        "dataset_id": self.dataset_id,
                        "group_event_id": group_event_id,
                        "node_id": member,
                        "membership_weight": 1.0,
                        "source_record_id": f"{start_index + 1}:{stop_index}",
                        "attributes_json": "{}",
                    }
                    for member in members
                )
        group_events = pd.DataFrame(group_rows)
        memberships = pd.DataFrame(membership_rows)
        individuals = make_individuals(
            self.dataset_id,
            memberships["node_id"],
            species="Ovis aries",
        )
        windows = observation_windows_from_groups(
            self.dataset_id, group_events, memberships
        )
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Free-ranging sheep fission-fusion group dynamics",
            adapter_name=type(self).__name__,
            source_files=[groups_path.name, time_path.name, "README_source.md"],
            primary_event_mode="group",
            edge_semantics=["author_inferred_spatial_group"],
            measurement_types=["group_membership"],
            has_temporal_order=True,
            has_roster=False,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "Groups use the authors' primary 30 m near and 50 m sticky radii.",
                "Consecutive identical partitions are losslessly run-length encoded as intervals.",
                "Three recording gaps are preserved as gaps and are never converted to contact time.",
                "Group membership indicates spatial grouping rather than observed physical contact.",
                "The paper reports 50 sheep, while the deposited partitions contain 51 identifiers across time and at most 50 within a continuous recording segment; no undocumented identity merge is imposed.",
            ],
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            group_events=group_events,
            group_memberships=memberships,
            observation_windows=windows,
        )
