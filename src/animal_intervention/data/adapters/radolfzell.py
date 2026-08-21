from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

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


def _parse_compact_datetime(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").round().astype("Int64")
    text = numeric.astype("string").str.zfill(12)
    return pd.to_datetime(text, format="%y%m%d%H%M%S", errors="coerce")


class RadolfzellGreatTitsAdapter(BaseAdapter):
    dataset_id = "radolfzell_great_tits_ontogeny"
    gmm_members = (
        *[("summer", f"gmm.summer.w{week}.RData", f"w{week:02d}") for week in range(1, 15)],
        ("autumn", "gmm.autumn.RData", "all"),
        ("winter", "gmm.winter.RData", "all"),
        ("spring", "gmm.spring.RData", "all"),
    )

    def _read_rdata_from_zip(self, bundle: zipfile.ZipFile, member: str) -> dict:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".RData", delete=False) as temporary:
                temporary.write(bundle.read(member))
                temporary_path = Path(temporary.name)
            return rdata.read_rda(temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        raw_dir = Path(raw_dir)
        archive = next(raw_dir.glob("doi_10_5061_dryad_x95x69ps8*.zip"))
        group_frames: list[pd.DataFrame] = []
        membership_frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(archive) as bundle:
            for season, member, sampling_period in tqdm(
                self.gmm_members,
                desc="Radolfzell GMM sampling periods",
                disable=not progress,
                unit="period",
            ):
                objects = self._read_rdata_from_zip(bundle, member)
                gmm = next(iter(objects.values()))
                metadata = gmm["metadata"].reset_index(drop=True)
                gbi = gmm["gbi"]
                if sample:
                    metadata = metadata.iloc[:100].copy()
                    matrix = np.asarray(gbi.values[:100])
                else:
                    matrix = np.asarray(gbi.values)
                starts = _parse_compact_datetime(metadata["Start"])
                ends = _parse_compact_datetime(metadata["End"])
                group_ids = pd.Series(
                    [
                        f"radolfzell:{season}:{sampling_period}:{index}"
                        for index in range(len(metadata))
                    ],
                    dtype="string",
                )
                groups = pd.DataFrame(
                    {
                        "dataset_id": self.dataset_id,
                        "group_event_id": group_ids,
                        "start_time": starts,
                        "end_time": ends,
                        "duration_seconds": (ends - starts).dt.total_seconds(),
                        "location_id": metadata["Location"].map(
                            lambda value: f"feeder:{normalize_node_id(value).lower()}"
                        ),
                        "event_semantics": "co_flocking_association",
                        "native_time_resolution_seconds": 1.0,
                        "source_record_id": [f"{member}:{index}" for index in range(len(metadata))],
                        "quality_flag": "ok",
                        "attributes_json": [
                            json.dumps(
                                {
                                    "season": season,
                                    "sampling_period": sampling_period,
                                },
                                separators=(",", ":"),
                            )
                        ]
                        * len(metadata),
                    }
                )
                row_indices, column_indices = np.nonzero(matrix > 0)
                node_coordinates = np.asarray(gbi.coords["dim_1"].values).astype(str)
                memberships = pd.DataFrame(
                    {
                        "dataset_id": self.dataset_id,
                        "group_event_id": group_ids.iloc[row_indices].to_numpy(),
                        "node_id": node_coordinates[column_indices],
                        "membership_weight": matrix[row_indices, column_indices].astype(float),
                        "source_record_id": [
                            f"{member}:{row}:{column}"
                            for row, column in zip(row_indices, column_indices)
                        ],
                        "attributes_json": "{}",
                    }
                )
                group_frames.append(groups)
                membership_frames.append(memberships)

            roster = pd.read_csv(bundle.open("species_age.txt"), sep="\t")

        group_events = pd.concat(group_frames, ignore_index=True)
        memberships = pd.concat(membership_frames, ignore_index=True).drop_duplicates(
            ["group_event_id", "node_id"], keep="first"
        )
        roster["node_id"] = roster["Pit"].map(normalize_node_id)
        roster = roster.drop_duplicates("node_id", keep="first")
        all_nodes = set(roster["node_id"]) | set(memberships["node_id"])
        individuals = make_individuals(self.dataset_id, all_nodes, species="Parus major")
        roster_fields = roster[["node_id", "Ring", "Species", "Age_in_2020"]].rename(
            columns={"Age_in_2020": "age_class"}
        )
        individuals = individuals.drop(columns=["age_class", "attributes_json"]).merge(
            roster_fields, on="node_id", how="left", validate="one_to_one"
        )
        individuals["species"] = individuals["Species"].fillna(individuals["species"])
        individuals = individuals.drop(columns=["Species"])
        individuals["attributes_json"] = [
            json.dumps({"ring": value}, default=str, separators=(",", ":"))
            for value in individuals.pop("Ring")
        ]
        windows = observation_windows_from_groups(self.dataset_id, group_events, memberships)
        individuals = apply_observation_bounds(individuals, windows)
        metadata = DatasetMetadata(
            dataset_id=self.dataset_id,
            title="Radolfzell great-tit RFID-derived flock events across seasons",
            adapter_name=type(self).__name__,
            adapter_version="0.2.0",
            source_files=[archive.name],
            primary_event_mode="group_event",
            edge_semantics=["co_flocking_association"],
            measurement_types=["group_membership"],
            has_temporal_order=True,
            has_roster=True,
            has_individual_coverage=False,
            is_sample=sample,
            notes=[
                "The fourteen non-overlapping weekly summer GMM objects are concatenated; the overlapping summer aggregate objects are excluded.",
                "Autumn, winter, and spring use their author-provided three-week GMM objects.",
                "GMM group membership represents inferred co-flocking, not confirmed physical contact.",
            ],
        )
        return CanonicalDataset(
            metadata=metadata,
            individuals=individuals,
            group_events=group_events,
            group_memberships=memberships.reset_index(drop=True),
            observation_windows=windows,
        )
