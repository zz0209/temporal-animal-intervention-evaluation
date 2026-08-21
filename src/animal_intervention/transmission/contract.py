from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


DYADIC_EXPOSURE_COLUMNS = [
    "dataset_id",
    "exposure_id",
    "source_id",
    "target_id",
    "start_time",
    "end_time",
    "hazard_rate_multiplier",
    "directed",
    "transmission_route",
    "mapper_name",
    "origin_event_id",
    "location_id",
]

GROUP_EXPOSURE_COLUMNS = [
    "dataset_id",
    "group_event_id",
    "start_time",
    "end_time",
    "hazard_rate_multiplier",
    "transmission_route",
    "mapper_name",
    "group_mixing_mode",
    "location_id",
]

GROUP_EXPOSURE_MEMBERSHIP_COLUMNS = [
    "dataset_id",
    "group_event_id",
    "node_id",
    "membership_weight",
]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


@dataclass(slots=True)
class ExposureStream:
    dataset_id: str
    population_nodes: tuple[str, ...] = ()
    dyadic_exposures: pd.DataFrame = field(
        default_factory=lambda: _empty(DYADIC_EXPOSURE_COLUMNS)
    )
    group_exposures: pd.DataFrame = field(
        default_factory=lambda: _empty(GROUP_EXPOSURE_COLUMNS)
    )
    group_memberships: pd.DataFrame = field(
        default_factory=lambda: _empty(GROUP_EXPOSURE_MEMBERSHIP_COLUMNS)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, columns in (
            ("dyadic_exposures", DYADIC_EXPOSURE_COLUMNS),
            ("group_exposures", GROUP_EXPOSURE_COLUMNS),
            ("group_memberships", GROUP_EXPOSURE_MEMBERSHIP_COLUMNS),
        ):
            frame = getattr(self, name).copy()
            for column in columns:
                if column not in frame:
                    frame[column] = pd.NA
            setattr(self, name, frame[columns])

    def validate(self) -> None:
        for name in ("dyadic_exposures", "group_exposures"):
            frame = getattr(self, name)
            if frame.empty:
                continue
            starts = pd.to_datetime(frame["start_time"], errors="coerce")
            ends = pd.to_datetime(frame["end_time"], errors="coerce")
            if starts.isna().any() or ends.isna().any():
                raise ValueError(f"{name} contains missing temporal coordinates")
            if (ends <= starts).any():
                raise ValueError(f"{name} contains non-positive exposure intervals")
            multipliers = pd.to_numeric(frame["hazard_rate_multiplier"], errors="coerce")
            if multipliers.isna().any() or (multipliers < 0).any():
                raise ValueError(f"{name} contains invalid hazard multipliers")
        if not self.group_memberships.empty:
            group_ids = set(self.group_exposures["group_event_id"].astype(str))
            missing = set(self.group_memberships["group_event_id"].astype(str)) - group_ids
            if missing:
                raise ValueError("group memberships reference absent exposure groups")

    def nodes(self) -> set[str]:
        nodes = set(map(str, self.population_nodes))
        nodes.update(self.dyadic_exposures["source_id"].dropna().astype(str))
        nodes.update(self.dyadic_exposures["target_id"].dropna().astype(str))
        nodes.update(self.group_memberships["node_id"].dropna().astype(str))
        return nodes


def combine_streams(*streams: ExposureStream) -> ExposureStream:
    if not streams:
        raise ValueError("at least one exposure stream is required")
    dataset_ids = {stream.dataset_id for stream in streams}
    if len(dataset_ids) != 1:
        raise ValueError("exposure streams from different datasets cannot be combined")
    combined = ExposureStream(
        dataset_id=streams[0].dataset_id,
        population_nodes=tuple(
            sorted({node for stream in streams for node in stream.population_nodes})
        ),
        dyadic_exposures=pd.concat(
            [stream.dyadic_exposures for stream in streams], ignore_index=True
        ),
        group_exposures=pd.concat(
            [stream.group_exposures for stream in streams], ignore_index=True
        ),
        group_memberships=pd.concat(
            [stream.group_memberships for stream in streams], ignore_index=True
        ).drop_duplicates(["group_event_id", "node_id"], keep="first"),
        metadata={"combined_mappers": [stream.metadata for stream in streams]},
    )
    combined.validate()
    return combined
