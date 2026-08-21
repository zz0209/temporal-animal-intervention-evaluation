from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ..contract import CanonicalDataset, DatasetMetadata
from .base import (
    BaseAdapter,
    apply_observation_bounds,
    canonical_pair,
    observation_windows_from_dyadic,
)


class WildVampireBatsAdapter(BaseAdapter):
    dataset_id = "wild_vampire_bats_proximity"
    excluded_sensor_nodes = {"10", "14", "26", "41"}
    analysis_start = pd.Timestamp("2018-04-25 15:00:00")

    @staticmethod
    def _read_member(archive: Path, member: str, **kwargs: object) -> pd.DataFrame:
        with ZipFile(archive) as bundle, bundle.open(f"Figshare/{member}") as stream:
            return pd.read_csv(stream, **kwargs)

    @staticmethod
    def _fuse_overlapping_intervals(events: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (source_id, target_id), group in events.groupby(
            ["source_id", "target_id"], observed=True, sort=True
        ):
            ordered = group.sort_values(["start_time", "end_time", "source_record_id"])
            current: dict[str, object] | None = None
            for row in ordered.itertuples(index=False):
                if current is None or row.start_time > current["end_time"]:
                    if current is not None:
                        rows.append(current)
                    current = {
                        "source_id": source_id,
                        "target_id": target_id,
                        "start_time": row.start_time,
                        "end_time": row.end_time,
                        "maximum_rssi_dbm": int(row.rssi_dbm),
                        "source_record_start": int(row.source_record_id),
                        "source_record_end": int(row.source_record_id),
                        "source_record_count": 1,
                    }
                else:
                    current["end_time"] = max(current["end_time"], row.end_time)
                    current["maximum_rssi_dbm"] = max(
                        int(current["maximum_rssi_dbm"]), int(row.rssi_dbm)
                    )
                    current["source_record_end"] = int(row.source_record_id)
                    current["source_record_count"] = int(current["source_record_count"]) + 1
            if current is not None:
                rows.append(current)
        return pd.DataFrame(rows)

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        del progress
        archive = Path(raw_dir) / "Figshare.zip"
        row_limit = 25_000 if sample else None
        raw = self._read_member(
            archive,
            "Belize_pre-fusion.csv",
            sep=";",
            nrows=row_limit,
        )
        roster = self._read_member(archive, "Belize_tracked_bats02.csv")

        all_sensor_nodes = {
            str(value) for value in pd.concat([raw["SenderID"], raw["EncounteredID"]])
        }
        retained_sensor_nodes = all_sensor_nodes - self.excluded_sensor_nodes
        rssi_threshold = float(raw["RSSI"].quantile(0.85))
        selected = raw.loc[
            raw["SenderID"].astype(str).isin(retained_sensor_nodes)
            & raw["EncounteredID"].astype(str).isin(retained_sensor_nodes)
            & raw["RSSI"].gt(rssi_threshold)
        ].copy()
        selected["source_id"] = selected["SenderID"].astype(str)
        selected["target_id"] = selected["EncounteredID"].astype(str)
        selected["source_id"], selected["target_id"] = canonical_pair(
            selected["source_id"], selected["target_id"]
        )
        selected["start_time"] = pd.to_datetime(
            selected["StartBelizeTime"], errors="coerce"
        )
        selected["end_time"] = pd.to_datetime(
            selected["EndBelizeTime"], errors="coerce"
        )
        selected["source_record_id"] = selected.index.astype(int)
        selected["rssi_dbm"] = pd.to_numeric(selected["RSSI"], errors="coerce")
        selected = selected.loc[
            selected["start_time"].ge(self.analysis_start)
            & selected["end_time"].gt(selected["start_time"])
        ]
        fused = self._fuse_overlapping_intervals(
            selected[
                [
                    "source_id",
                    "target_id",
                    "start_time",
                    "end_time",
                    "source_record_id",
                    "rssi_dbm",
                ]
            ]
        )
        fused["duration_seconds"] = (
            fused["end_time"] - fused["start_time"]
        ).dt.total_seconds()
        events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "event_id": [f"vampire-bat:{index}" for index in fused.index],
                "source_id": fused["source_id"],
                "target_id": fused["target_id"],
                "start_time": fused["start_time"],
                "end_time": fused["end_time"],
                "duration_seconds": fused["duration_seconds"],
                "event_representation": "interval",
                "edge_semantics": "sensor_proximity",
                "measurement_type": "duration",
                "measurement_value": fused["duration_seconds"],
                "measurement_unit": "second",
                "location_id": "Lamanai_Sugar_Mill_roost",
                "directed": False,
                "native_time_resolution_seconds": 1.0,
                "source_record_id": fused.apply(
                    lambda row: (
                        f"{int(row.source_record_start)}:{int(row.source_record_end)}"
                    ),
                    axis=1,
                ),
                "quality_flag": "ok",
                "attributes_json": fused.apply(
                    lambda row: json.dumps(
                        {
                            "maximum_rssi_dbm": int(row.maximum_rssi_dbm),
                            "source_record_count": int(row.source_record_count),
                            "rssi_threshold_dbm": rssi_threshold,
                        },
                        separators=(",", ":"),
                    ),
                    axis=1,
                ),
            }
        )

        roster = roster.loc[
            roster["sensor_node"].astype(str).isin(retained_sensor_nodes)
        ].copy()
        individuals = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "node_id": roster["sensor_node"].astype(str),
                "species": roster["species"].astype(str),
                "sex": roster["sex"].astype(str),
                "age_class": roster["age"].astype(str),
                "group_id": "Lamanai_Sugar_Mill_roost",
                "first_observed_at": pd.NaT,
                "last_observed_at": pd.NaT,
                "attributes_json": roster.apply(
                    lambda row: json.dumps(
                        {
                            "band": row["band"],
                            "weight_g": row["weight"],
                            "treatment": row["treatment"],
                            "treatment_amount": row["treatment_amount"],
                        },
                        default=str,
                        separators=(",", ":"),
                    ),
                    axis=1,
                ),
            }
        ).sort_values("node_id", kind="stable")
        windows = observation_windows_from_dyadic(self.dataset_id, events)
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Wild vampire bat high-resolution proximity encounters",
            adapter_name=type(self).__name__,
            source_files=[archive.name],
            primary_event_mode="dyadic",
            edge_semantics=["sensor_proximity"],
            measurement_types=["duration"],
            has_temporal_order=True,
            has_roster=True,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "The adapter reproduces the associated analysis exclusion list and strict 85th-percentile RSSI filter.",
                "Overlapping records for an unordered dyad are fused according to the published analysis code.",
                "Treatment metadata are retained as context and are not used as infection outcomes.",
                "Detection-span windows do not prove continuous sensor uptime.",
            ],
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            dyadic_events=events,
            observation_windows=windows,
        )
