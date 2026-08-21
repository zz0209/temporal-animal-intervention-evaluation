from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd


INDIVIDUAL_COLUMNS = [
    "dataset_id",
    "node_id",
    "species",
    "sex",
    "age_class",
    "group_id",
    "first_observed_at",
    "last_observed_at",
    "attributes_json",
]

DYADIC_EVENT_COLUMNS = [
    "dataset_id",
    "event_id",
    "source_id",
    "target_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "event_representation",
    "edge_semantics",
    "measurement_type",
    "measurement_value",
    "measurement_unit",
    "location_id",
    "directed",
    "native_time_resolution_seconds",
    "source_record_id",
    "quality_flag",
    "attributes_json",
]

GROUP_EVENT_COLUMNS = [
    "dataset_id",
    "group_event_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "location_id",
    "event_semantics",
    "native_time_resolution_seconds",
    "source_record_id",
    "quality_flag",
    "attributes_json",
]

GROUP_MEMBERSHIP_COLUMNS = [
    "dataset_id",
    "group_event_id",
    "node_id",
    "membership_weight",
    "source_record_id",
    "attributes_json",
]

OBSERVATION_WINDOW_COLUMNS = [
    "dataset_id",
    "node_id",
    "window_start",
    "window_end",
    "observation_status",
    "device_id",
    "coverage_fraction",
    "reason",
    "attributes_json",
]

TABLE_COLUMNS = {
    "individuals": INDIVIDUAL_COLUMNS,
    "dyadic_events": DYADIC_EVENT_COLUMNS,
    "group_events": GROUP_EVENT_COLUMNS,
    "group_memberships": GROUP_MEMBERSHIP_COLUMNS,
    "observation_windows": OBSERVATION_WINDOW_COLUMNS,
}


def empty_table(name: str) -> pd.DataFrame:
    """Return an empty canonical table with stable column order."""
    return pd.DataFrame(columns=TABLE_COLUMNS[name])


def _ordered_table(frame: pd.DataFrame | None, name: str) -> pd.DataFrame:
    frame = empty_table(name) if frame is None else frame.copy()
    for column in TABLE_COLUMNS[name]:
        if column not in frame:
            frame[column] = pd.NA
    return frame[TABLE_COLUMNS[name]]


@dataclass(slots=True)
class DatasetMetadata:
    dataset_id: str
    title: str
    adapter_name: str
    adapter_version: str = "0.1.0"
    source_files: list[str] = field(default_factory=list)
    time_axis: str = "absolute"
    primary_event_mode: str = "dyadic"
    edge_semantics: list[str] = field(default_factory=list)
    measurement_types: list[str] = field(default_factory=list)
    has_temporal_order: bool = True
    has_roster: bool = False
    has_individual_coverage: bool = False
    is_sample: bool = False
    notes: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetMetadata":
        return cls(**value)


@dataclass(slots=True)
class CanonicalDataset:
    metadata: DatasetMetadata
    individuals: pd.DataFrame = field(default_factory=lambda: empty_table("individuals"))
    dyadic_events: pd.DataFrame = field(default_factory=lambda: empty_table("dyadic_events"))
    group_events: pd.DataFrame = field(default_factory=lambda: empty_table("group_events"))
    group_memberships: pd.DataFrame = field(
        default_factory=lambda: empty_table("group_memberships")
    )
    observation_windows: pd.DataFrame = field(
        default_factory=lambda: empty_table("observation_windows")
    )

    def __post_init__(self) -> None:
        for name in TABLE_COLUMNS:
            setattr(self, name, _ordered_table(getattr(self, name), name))

    def tables(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in TABLE_COLUMNS}

    def summary(self) -> dict[str, Any]:
        temporal_starts = []
        temporal_ends = []
        for name in ("dyadic_events", "group_events"):
            frame = getattr(self, name)
            if len(frame):
                temporal_starts.append(pd.to_datetime(frame["start_time"], errors="coerce").min())
                temporal_ends.append(pd.to_datetime(frame["end_time"], errors="coerce").max())
        starts = [value for value in temporal_starts if pd.notna(value)]
        ends = [value for value in temporal_ends if pd.notna(value)]
        return {
            "dataset_id": self.metadata.dataset_id,
            "is_sample": self.metadata.is_sample,
            "individuals": int(len(self.individuals)),
            "dyadic_events": int(len(self.dyadic_events)),
            "group_events": int(len(self.group_events)),
            "group_memberships": int(len(self.group_memberships)),
            "observation_windows": int(len(self.observation_windows)),
            "time_start": min(starts).isoformat() if starts else None,
            "time_end": max(ends).isoformat() if ends else None,
            "edge_semantics": sorted(
                set(self.dyadic_events["edge_semantics"].dropna().astype(str))
                | set(self.group_events["event_semantics"].dropna().astype(str))
            ),
        }

    def write(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables().items():
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        (output_dir / "dataset_metadata.json").write_text(
            json.dumps(self.metadata.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def read(cls, output_dir: str | Path) -> "CanonicalDataset":
        output_dir = Path(output_dir)
        metadata = DatasetMetadata.from_dict(
            json.loads((output_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
        )
        tables = {
            name: pd.read_parquet(output_dir / f"{name}.parquet")
            for name in TABLE_COLUMNS
        }
        return cls(metadata=metadata, **tables)


def json_attributes(**values: Any) -> str:
    clean = {
        key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in values.items()
    }
    return json.dumps(clean, ensure_ascii=False, default=str, separators=(",", ":"))

