from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ..contract import CanonicalDataset, DatasetMetadata
from .base import (
    BaseAdapter,
    apply_observation_bounds,
    canonical_pair,
    make_individuals,
    normalize_node_id,
    observation_windows_from_dyadic,
)


class DomesticSheepAdapter(BaseAdapter):
    dataset_id = "domestic_sheep_sirtrack"

    def _read_behaviors(self, path: Path, sample: bool, progress: bool) -> pd.DataFrame:
        if sample:
            return pd.read_csv(path, nrows=5_000)
        chunks = pd.read_csv(path, chunksize=100_000)
        return pd.concat(
            list(tqdm(chunks, desc="domestic sheep CSV chunks", disable=not progress, unit="chunk")),
            ignore_index=True,
        )

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        raw_dir = Path(raw_dir)
        behavior_path = raw_dir / "Behaviour_data_Amorris_2023.csv"
        animal_path = raw_dir / "Animal_measurements_data_Amorris_2023.csv"
        raw = self._read_behaviors(behavior_path, sample, progress)
        animal = pd.read_csv(animal_path)

        left = raw["Animal.1"].map(normalize_node_id)
        right = raw["Animal.2"].map(normalize_node_id)
        source, target = canonical_pair(left, right)
        start = pd.to_datetime(
            raw["Start.date"].astype(str) + " " + raw["Start.time"].astype(str),
            dayfirst=True,
            errors="coerce",
        )
        end = pd.to_datetime(
            raw["End.date"].astype(str) + " " + raw["End.time"].astype(str),
            dayfirst=True,
            errors="coerce",
        )
        reported_duration = pd.to_numeric(raw["Encounter.Length"], errors="coerce")
        computed_duration = (end - start).dt.total_seconds()
        duration_matches = (reported_duration - computed_duration).abs().le(1.0)
        exact_source_duplicate = raw.duplicated(keep="first")
        quality_flags = []
        for duration_ok, duplicate in zip(duration_matches, exact_source_duplicate):
            flags = []
            if not duration_ok:
                flags.append("duration_timestamp_mismatch")
            if duplicate:
                flags.append("exact_duplicate_source_record")
            quality_flags.append(";".join(flags) if flags else "ok")
        events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "event_id": [f"sheep-contact:{index}" for index in raw.index],
                "source_id": source,
                "target_id": target,
                "start_time": start,
                "end_time": end,
                "duration_seconds": reported_duration,
                "event_representation": "interval",
                "edge_semantics": "sensor_proximity",
                "measurement_type": "duration",
                "measurement_value": reported_duration,
                "measurement_unit": "second",
                "location_id": raw["Plot"].map(lambda value: f"plot:{normalize_node_id(value)}"),
                "directed": False,
                "native_time_resolution_seconds": 1.0,
                "source_record_id": raw.index.astype(str),
                "quality_flag": quality_flags,
                "attributes_json": [
                    json.dumps(
                        {
                            "treatment_group": treatment_group,
                            "phase": phase,
                            "group": group,
                            "day": int(day),
                            "week": int(week),
                        },
                        default=str,
                        separators=(",", ":"),
                    )
                    for treatment_group, phase, group, day, week in zip(
                        raw["Treatment.Group"],
                        raw["Phase"],
                        raw["Group"],
                        raw["Day"],
                        raw["Week"],
                    )
                ],
            }
        )

        animal["node_id"] = animal["Sheep.ID"].map(normalize_node_id)
        roster = (
            animal.sort_values(["node_id", "Date"])
            .groupby("node_id", observed=True, as_index=False)
            .agg(
                sex=("Sex", "first"),
                group_id=("Treatment.Group", "first"),
                treatment=("Treatment", "first"),
                parasitised=("Parasitised", "first"),
            )
        )
        all_nodes = set(roster["node_id"]) | set(events["source_id"]) | set(events["target_id"])
        individuals = make_individuals(self.dataset_id, all_nodes, species="Ovis aries")
        individuals = individuals.drop(columns=["sex", "group_id", "attributes_json"]).merge(
            roster,
            on="node_id",
            how="left",
            validate="one_to_one",
        )
        individuals["attributes_json"] = [
            json.dumps(
                {"treatment": treatment, "parasitised": parasitised},
                default=str,
                separators=(",", ":"),
            )
            for treatment, parasitised in zip(
                individuals.pop("treatment"), individuals.pop("parasitised")
            )
        ]
        windows = observation_windows_from_dyadic(self.dataset_id, events)
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Domestic sheep Sirtrack proximity-logger contacts",
            adapter_name=type(self).__name__,
            source_files=[behavior_path.name, animal_path.name],
            primary_event_mode="dyadic_interval",
            edge_semantics=["sensor_proximity"],
            measurement_types=["duration"],
            has_temporal_order=True,
            has_roster=True,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "Encounter.Length is retained and cross-checked against exact start/end timestamps.",
                "Exact duplicate source rows are retained for provenance, flagged, and excluded by transmission mappers.",
                "Detected spans are not treated as continuous logger uptime.",
            ],
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            dyadic_events=events,
            observation_windows=windows,
        )
