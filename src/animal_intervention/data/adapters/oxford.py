from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd

from ..contract import CanonicalDataset, DatasetMetadata
from .base import (
    BaseAdapter,
    apply_observation_bounds,
    canonical_pair,
    make_individuals,
    observation_windows_from_dyadic,
)


class OxfordWildbirdAdapter(BaseAdapter):
    dataset_id = "oxford_wildbird_network"
    relative_origin = pd.Timestamp("2000-01-01")

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        del progress
        raw_dir = Path(raw_dir)
        archive = raw_dir / "aves-wildbird-network.zip"
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open("aves-wildbird-network.edges") as stream:
                raw = pd.read_csv(
                    stream,
                    sep=r"\s+",
                    names=["source_id", "target_id", "association_weight", "day"],
                    nrows=250 if sample else None,
                )

        raw["source_id"] = raw["source_id"].astype("string")
        raw["target_id"] = raw["target_id"].astype("string")
        raw["source_id"], raw["target_id"] = canonical_pair(
            raw["source_id"], raw["target_id"]
        )
        start = self.relative_origin + pd.to_timedelta(raw["day"].astype(int) - 1, unit="D")
        events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "event_id": [f"oxford:{index}" for index in raw.index],
                "source_id": raw["source_id"],
                "target_id": raw["target_id"],
                "start_time": start,
                "end_time": start + pd.Timedelta("1d"),
                "duration_seconds": 86_400.0,
                "event_representation": "aggregated_interval",
                "edge_semantics": "daily_association",
                "measurement_type": "association_index",
                "measurement_value": raw["association_weight"].astype(float),
                "measurement_unit": "unitless",
                "location_id": pd.NA,
                "directed": False,
                "native_time_resolution_seconds": 86_400.0,
                "source_record_id": raw.index.astype(str),
                "quality_flag": "ok",
                "attributes_json": [f'{{"source_day":{int(day)}}}' for day in raw["day"]],
            }
        )
        individuals = make_individuals(
            self.dataset_id,
            pd.concat([events["source_id"], events["target_id"]]),
            species="mixed wild birds",
        )
        windows = observation_windows_from_dyadic(self.dataset_id, events)
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Oxford wildbird six-day temporal association network",
            adapter_name=type(self).__name__,
            source_files=[archive.name],
            time_axis="relative_day",
            primary_event_mode="aggregated_dyadic",
            edge_semantics=["daily_association"],
            measurement_types=["association_index"],
            has_temporal_order=True,
            has_roster=False,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "Dates are unavailable; day 1 is anchored to 2000-01-01 for relative-time computation.",
                "Daily association weights are not physical contact durations.",
            ],
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            dyadic_events=events,
            observation_windows=windows,
        )
