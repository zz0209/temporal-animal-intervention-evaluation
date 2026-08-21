from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd

from ..contract import CanonicalDataset, DatasetMetadata
from .base import BaseAdapter, canonical_pair, make_individuals, normalize_node_id


class BarnSwallowsAdapter(BaseAdapter):
    dataset_id = "barn_swallows_encounternet"

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        del progress
        raw_dir = Path(raw_dir)
        archive = next(raw_dir.glob("II+Levin+*.zip"))
        with zipfile.ZipFile(archive) as bundle:
            member = next(
                name
                for name in bundle.namelist()
                if name.endswith("/Dyads.csv") and not name.startswith("__MACOSX/")
            )
            with bundle.open(member) as stream:
                raw = pd.read_csv(stream, nrows=250 if sample else None)

        left = raw["This ID 1"].map(normalize_node_id)
        right = raw["This ID 2"].map(normalize_node_id)
        source, target = canonical_pair(left, right)
        duration_1 = pd.to_numeric(raw["Tag1 duration (s)"], errors="coerce")
        duration_2 = pd.to_numeric(raw["Tag2 duration (s)"], errors="coerce")
        duration = pd.concat([duration_1, duration_2], axis=1).mean(axis=1)
        reciprocal = (
            raw["This ID 1"].map(normalize_node_id)
            == raw["Enc ID 2"].map(normalize_node_id)
        ) & (
            raw["ENC ID 1"].map(normalize_node_id)
            == raw["This ID 2"].map(normalize_node_id)
        )
        events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "event_id": [f"barn-dyad:{index}" for index in raw.index],
                "source_id": source,
                "target_id": target,
                "start_time": pd.NaT,
                "end_time": pd.NaT,
                "duration_seconds": duration,
                "event_representation": "unordered_encounter",
                "edge_semantics": "sensor_proximity",
                "measurement_type": "duration",
                "measurement_value": duration,
                "measurement_unit": "second",
                "location_id": pd.NA,
                "directed": False,
                "native_time_resolution_seconds": pd.NA,
                "source_record_id": raw.index.astype(str),
                "quality_flag": reciprocal.map({True: "ok", False: "nonreciprocal_ids"}),
                "attributes_json": [
                    json.dumps(
                        {
                            "tag1_rssi_mean": int(rssi_1),
                            "tag2_rssi_mean": int(rssi_2),
                            "tag1_duration_seconds": int(d1),
                            "tag2_duration_seconds": int(d2),
                        },
                        separators=(",", ":"),
                    )
                    for rssi_1, rssi_2, d1, d2 in zip(
                        raw["Tag1 RSSI mean"],
                        raw["Tag2 RSSI mean"],
                        duration_1,
                        duration_2,
                    )
                ],
            }
        )
        individuals = make_individuals(
            self.dataset_id,
            pd.concat([events["source_id"], events["target_id"]]),
            species="Hirundo rustica erythrogaster",
        )
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Barn swallow Encounternet reciprocal dyadic logs",
            adapter_name=type(self).__name__,
            source_files=[archive.name],
            time_axis="unavailable",
            primary_event_mode="unordered_dyadic",
            edge_semantics=["sensor_proximity"],
            measurement_types=["duration", "rssi"],
            has_temporal_order=False,
            has_roster=False,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "Dyads.csv contains reciprocal RSSI and duration summaries but no timestamps.",
                "Rows are retained without invented times and are ineligible for temporal simulation.",
            ],
        )
        return CanonicalDataset(metadata=metadata, individuals=individuals, dyadic_events=events)
