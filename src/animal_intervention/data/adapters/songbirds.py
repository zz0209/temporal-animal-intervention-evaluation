from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rdata
from tqdm.auto import tqdm

from ..contract import CanonicalDataset, DatasetMetadata
from .base import (
    BaseAdapter,
    apply_observation_bounds,
    make_individuals,
    normalize_node_id,
    observation_windows_from_groups,
)


class ExperimentalSongbirdsAdapter(BaseAdapter):
    dataset_id = "experimental_wild_songbirds"

    @staticmethod
    def _tag_parity(node_id: str) -> str:
        try:
            return "odd" if int(node_id[-1], 16) % 2 else "even"
        except (ValueError, IndexError) as error:
            raise ValueError(f"Cannot derive hexadecimal PIT-tag parity: {node_id}") from error

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        raw_dir = Path(raw_dir)
        source = raw_dir / "dryad_data.RData"
        # rdata warns about POSIX classes in unrelated R objects (notably
        # patch.times). The canonical stream below uses the numeric Unix-second
        # columns in groups.info, so those unsupported constructors are never
        # used to derive timestamps.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message='Missing constructor for R class "POSIX.*"'
            )
            objects = rdata.read_rda(source)
        group_info = objects["groups.info"].reset_index(drop=True)
        gbi = objects["group.by.individual"]
        group_limit = min(200, len(group_info)) if sample else len(group_info)
        group_info = group_info.iloc[:group_limit].copy()
        matrix = np.asarray(gbi.values[:group_limit])
        iterator = tqdm(
            ["decode group matrix"],
            desc="Experimental songbird RData",
            disable=not progress,
            unit="stage",
        )
        for _ in iterator:
            row_indices, column_indices = np.nonzero(matrix > 0)
        starts = pd.to_datetime(
            pd.to_numeric(group_info["start.time"], errors="coerce"),
            unit="s",
            errors="coerce",
        )
        ends = pd.to_datetime(
            pd.to_numeric(group_info["stop.time"], errors="coerce"),
            unit="s",
            errors="coerce",
        )
        group_ids = pd.Series(
            [
                f"songbird:{normalize_node_id(phase).lower()}:{index}"
                for index, phase in enumerate(group_info["experiment.phase"])
            ],
            dtype="string",
        )
        group_events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "group_event_id": group_ids,
                "start_time": starts,
                "end_time": ends,
                "duration_seconds": (ends - starts).dt.total_seconds(),
                "location_id": group_info["unit"].map(
                    lambda value: f"selective_feeder:{normalize_node_id(value)}"
                ),
                "event_semantics": "co_flocking_association",
                "native_time_resolution_seconds": 1.0,
                "source_record_id": group_info.index.astype(str),
                "quality_flag": "ok",
                "attributes_json": [
                    json.dumps(
                        {
                            "allows_odds": bool(allows_odds),
                            "experiment_phase": str(phase),
                        },
                        separators=(",", ":"),
                    )
                    for allows_odds, phase in zip(
                        group_info["allows.odds"], group_info["experiment.phase"]
                    )
                ],
            }
        )
        node_coordinates = np.asarray(gbi.coords["dim_1"].values).astype(str)
        memberships = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "group_event_id": group_ids.iloc[row_indices].to_numpy(),
                "node_id": node_coordinates[column_indices],
                "membership_weight": matrix[row_indices, column_indices].astype(float),
                "source_record_id": [
                    f"gbi:{row}:{column}" for row, column in zip(row_indices, column_indices)
                ],
                "attributes_json": "{}",
            }
        ).drop_duplicates(["group_event_id", "node_id"], keep="first")

        roster = objects["all.individuals"].copy()
        roster["node_id"] = roster["id"].map(normalize_node_id)
        roster = roster.drop_duplicates("node_id", keep="first")
        all_nodes = set(roster["node_id"]) | set(memberships["node_id"])
        individuals = make_individuals(self.dataset_id, all_nodes)
        individuals = individuals.drop(columns=["species"]).merge(
            roster[["node_id", "species"]], on="node_id", how="left", validate="one_to_one"
        )
        individuals["attributes_json"] = individuals["node_id"].map(
            lambda node_id: json.dumps(
                {"tag_parity": self._tag_parity(str(node_id))},
                separators=(",", ":"),
            )
        )
        windows = observation_windows_from_groups(self.dataset_id, group_events, memberships)
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Experimentally manipulated wild-songbird foraging groups",
            adapter_name=type(self).__name__,
            source_files=[source.name],
            primary_event_mode="group_event",
            edge_semantics=["co_flocking_association"],
            measurement_types=["group_membership"],
            has_temporal_order=True,
            has_roster=True,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "group.by.individual and groups.info are retained as event memberships and intervals.",
                "patch.times is not required for the social-foraging contact stream and is not canonicalized here.",
                "Random hexadecimal PIT-tag parity is retained as individual metadata for the selective-feeder manipulation.",
            ],
            adapter_version="0.2.0",
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            group_events=group_events,
            group_memberships=memberships.reset_index(drop=True),
            observation_windows=windows,
        )
