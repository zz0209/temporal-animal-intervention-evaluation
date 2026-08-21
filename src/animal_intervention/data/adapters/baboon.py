from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..contract import CanonicalDataset, DatasetMetadata
from .base import (
    BaseAdapter,
    apply_observation_bounds,
    canonical_pair,
    make_individuals,
    normalize_node_id,
    observation_windows_from_dyadic,
)


class GuineaBaboonsAdapter(BaseAdapter):
    dataset_id = "guinea_baboons_sociopatterns"

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        del progress
        raw_dir = Path(raw_dir)
        proximity_path = raw_dir / "baboons_proximity_data.txt.gz"
        observation_path = raw_dir / "baboons_obs_data.txt.gz"
        row_limit = 5_000 if sample else None
        proximity = pd.read_csv(proximity_path, sep="\t", nrows=row_limit)
        observations = pd.read_csv(observation_path, sep="\t", nrows=row_limit)

        proximity["i"] = proximity["i"].map(normalize_node_id)
        proximity["j"] = proximity["j"].map(normalize_node_id)
        proximity["i"], proximity["j"] = canonical_pair(proximity["i"], proximity["j"])
        proximity_start = pd.to_datetime(
            proximity["DateTime"], dayfirst=True, errors="coerce"
        ) + pd.to_timedelta(pd.to_numeric(proximity["t"], errors="coerce") % 60, unit="s")
        proximity_events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "event_id": [f"baboon-proximity:{index}" for index in proximity.index],
                "source_id": proximity["i"],
                "target_id": proximity["j"],
                "start_time": proximity_start,
                "end_time": proximity_start + pd.Timedelta("20s"),
                "duration_seconds": 20.0,
                "event_representation": "fixed_bin",
                "edge_semantics": "sensor_proximity",
                "measurement_type": "binary_presence",
                "measurement_value": 1.0,
                "measurement_unit": "20_second_detection",
                "location_id": pd.NA,
                "directed": False,
                "native_time_resolution_seconds": 20.0,
                "source_record_id": proximity.index.astype(str),
                "quality_flag": "ok",
                "attributes_json": [
                    json.dumps({"unix_time": int(value)}, separators=(",", ":"))
                    for value in proximity["t"]
                ],
            }
        )

        observations["Actor"] = observations["Actor"].map(normalize_node_id)
        observations["Recipient"] = observations["Recipient"].map(normalize_node_id)
        directed = observations.loc[
            (observations["Actor"] != "")
            & (observations["Recipient"] != "")
            & (observations["Actor"] != observations["Recipient"])
        ].copy()
        directed["_exact_source_duplicate"] = directed.duplicated(keep="first")
        observation_start = pd.to_datetime(
            directed["DateTime"], dayfirst=True, errors="coerce"
        )
        observation_duration = pd.to_numeric(directed["Duration"], errors="coerce")
        observation_events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "event_id": [f"baboon-observed:{index}" for index in directed.index],
                "source_id": directed["Actor"],
                "target_id": directed["Recipient"],
                "start_time": observation_start,
                "end_time": observation_start + pd.to_timedelta(observation_duration, unit="s"),
                "duration_seconds": observation_duration,
                "event_representation": "interval",
                "edge_semantics": "direct_behavior",
                "measurement_type": "duration",
                "measurement_value": observation_duration,
                "measurement_unit": "second",
                "location_id": pd.NA,
                "directed": True,
                "native_time_resolution_seconds": 60.0,
                "source_record_id": directed.index.astype(str),
                "quality_flag": directed["_exact_source_duplicate"].map(
                    {True: "exact_duplicate_source_record", False: "ok"}
                ),
                "attributes_json": [
                    json.dumps(
                        {
                            "behavior": behavior,
                            "category": category,
                            "point": point,
                        },
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ":"),
                    )
                    for behavior, category, point in zip(
                        directed["Behavior"], directed["Category"], directed["Point"]
                    )
                ],
            }
        )
        events = pd.concat([proximity_events, observation_events], ignore_index=True)
        nodes = pd.concat([events["source_id"], events["target_id"]], ignore_index=True)
        individuals = make_individuals(
            self.dataset_id,
            nodes,
            species="Papio papio",
        )
        windows = observation_windows_from_dyadic(self.dataset_id, events)
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Guinea baboon sensor proximity and direct behavioral observations",
            adapter_name=type(self).__name__,
            source_files=[proximity_path.name, observation_path.name],
            primary_event_mode="dyadic",
            edge_semantics=["sensor_proximity", "direct_behavior"],
            measurement_types=["binary_presence", "duration"],
            has_temporal_order=True,
            has_roster=False,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "Proximity detections represent 20-second bins.",
                "Behavior observations are a separate directed modality and must not be silently pooled with sensor proximity.",
                "Exact duplicate behavior rows are retained for provenance, flagged, and excluded by transmission mappers.",
                "Detection-span windows do not prove continuous tag uptime.",
            ],
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            dyadic_events=events,
            observation_windows=windows,
        )
