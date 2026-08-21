from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd
from tqdm.auto import tqdm

from ..contract import CanonicalDataset, DatasetMetadata
from .base import (
    BaseAdapter,
    apply_observation_bounds,
    make_individuals,
    normalize_node_id,
    observation_windows_from_groups,
)


class WythamGreatTitsAdapter(BaseAdapter):
    dataset_id = "wytham_great_tits_divorce"
    membership_columns = [
        "ring",
        "flockevent",
        "location",
        "start.time",
        "stop.time",
        "nwkend",
        "day",
        "year",
    ]

    def _read_membership_file(
        self,
        bundle: zipfile.ZipFile,
        member: str,
        season_index: int,
        sample: bool,
        progress: bool,
    ) -> pd.DataFrame:
        if sample:
            frame = pd.read_csv(
                bundle.open(member), nrows=5_000, usecols=self.membership_columns
            )
        else:
            chunks = pd.read_csv(
                bundle.open(member),
                chunksize=200_000,
                usecols=self.membership_columns,
            )
            frame = pd.concat(
                list(
                    tqdm(
                        chunks,
                        desc=f"Wytham membership season {season_index}",
                        disable=not progress,
                        unit="chunk",
                    )
                ),
                ignore_index=True,
            )
        frame["season_index"] = season_index
        frame["source_row"] = frame.index.astype(str)
        return frame

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        raw_dir = Path(raw_dir)
        archive = next(raw_dir.glob("doi_10_5061_dryad_vq83bk453*.zip"))
        membership_frames: list[pd.DataFrame] = []
        roster_frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(archive) as bundle:
            for season_index in (1, 2, 3):
                membership_frames.append(
                    self._read_membership_file(
                        bundle,
                        f"birds_flocking_events_{season_index}.csv",
                        season_index,
                        sample,
                        progress,
                    )
                )
                roster = pd.read_csv(bundle.open(f"individual_records_{season_index}.csv"))
                roster["season_index"] = season_index
                roster_frames.append(roster)

        raw = pd.concat(membership_frames, ignore_index=True)
        raw["node_id"] = raw["ring"].map(normalize_node_id)
        raw["group_event_id"] = [
            f"wytham:{season}:{normalize_node_id(event)}"
            for season, event in zip(raw["season_index"], raw["flockevent"])
        ]
        raw["parsed_start"] = pd.to_datetime(raw["start.time"], errors="coerce")
        raw["parsed_end"] = pd.to_datetime(raw["stop.time"], errors="coerce")
        consistency = raw.groupby("group_event_id", observed=True).agg(
            start_values=("parsed_start", "nunique"),
            end_values=("parsed_end", "nunique"),
            location_values=("location", "nunique"),
        )
        inconsistent = set(
            consistency.index[
                (consistency["start_values"] > 1)
                | (consistency["end_values"] > 1)
                | (consistency["location_values"] > 1)
            ]
        )
        group_source = raw.drop_duplicates("group_event_id", keep="first").copy()
        group_events = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "group_event_id": group_source["group_event_id"],
                "start_time": group_source["parsed_start"],
                "end_time": group_source["parsed_end"],
                "duration_seconds": (
                    group_source["parsed_end"] - group_source["parsed_start"]
                ).dt.total_seconds(),
                "location_id": group_source["location"].map(
                    lambda value: f"feeder:{normalize_node_id(value).lower()}"
                ),
                "event_semantics": "co_flocking_association",
                "native_time_resolution_seconds": 1.0,
                "source_record_id": [
                    f"season{season}:{row}"
                    for season, row in zip(
                        group_source["season_index"], group_source["source_row"]
                    )
                ],
                "quality_flag": group_source["group_event_id"].map(
                    lambda value: "inconsistent_member_times" if value in inconsistent else "ok"
                ),
                "attributes_json": [
                    json.dumps(
                        {"day": int(day), "study_year": int(year), "reported_nwkend": int(nwkend)},
                        separators=(",", ":"),
                    )
                    for day, year, nwkend in zip(
                        group_source["day"], group_source["year"], group_source["nwkend"]
                    )
                ],
            }
        ).reset_index(drop=True)
        memberships = pd.DataFrame(
            {
                "dataset_id": self.dataset_id,
                "group_event_id": raw["group_event_id"],
                "node_id": raw["node_id"],
                "membership_weight": 1.0,
                "source_record_id": [
                    f"season{season}:{row}"
                    for season, row in zip(raw["season_index"], raw["source_row"])
                ],
                "attributes_json": "{}",
            }
        ).drop_duplicates(["group_event_id", "node_id"], keep="first")

        roster = pd.concat(roster_frames, ignore_index=True)
        roster["node_id"] = roster["id"].map(normalize_node_id)
        roster = roster.sort_values("season_index").drop_duplicates("node_id", keep="last")
        all_nodes = set(roster["node_id"]) | set(memberships["node_id"])
        individuals = make_individuals(self.dataset_id, all_nodes)
        roster_fields = roster[["node_id", "species", "sex", "age", "immigrant"]].rename(
            columns={"age": "age_class"}
        )
        individuals = individuals.drop(
            columns=["species", "sex", "age_class", "attributes_json"]
        ).merge(roster_fields, on="node_id", how="left", validate="one_to_one")
        individuals["attributes_json"] = [
            json.dumps({"immigrant": value}, default=str, separators=(",", ":"))
            for value in individuals.pop("immigrant")
        ]
        windows = observation_windows_from_groups(self.dataset_id, group_events, memberships)
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Wytham great-tit RFID-derived flock events",
            adapter_name=type(self).__name__,
            source_files=[archive.name],
            primary_event_mode="group_event",
            edge_semantics=["co_flocking_association"],
            measurement_types=["group_membership"],
            has_temporal_order=True,
            has_roster=True,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "birds_flocking_events tables are used as the loss-preserving event-membership source.",
                "records_events_list tables are lower-level redundant records and are not duplicated into canonical output.",
                "Co-flocking is an association opportunity, not confirmed pairwise physical contact.",
            ],
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            group_events=group_events,
            group_memberships=memberships.reset_index(drop=True),
            observation_windows=windows,
        )
