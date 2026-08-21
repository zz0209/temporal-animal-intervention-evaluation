from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..contract import CanonicalDataset, empty_table


class BaseAdapter(ABC):
    dataset_id: str

    @abstractmethod
    def load(
        self,
        raw_dir: str | Path,
        *,
        sample: bool = False,
        progress: bool = True,
    ) -> CanonicalDataset:
        """Read immutable raw files and return canonical observational tables."""


def normalize_node_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def canonical_pair(source: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    source = source.astype("string")
    target = target.astype("string")
    return source.where(source <= target, target), target.where(source <= target, source)


def make_individuals(
    dataset_id: str,
    node_ids: Iterable[object],
    *,
    species: str | None = None,
) -> pd.DataFrame:
    nodes = sorted({normalize_node_id(value) for value in node_ids if normalize_node_id(value)})
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "node_id": nodes,
            "species": species,
            "sex": pd.NA,
            "age_class": pd.NA,
            "group_id": pd.NA,
            "first_observed_at": pd.NaT,
            "last_observed_at": pd.NaT,
            "attributes_json": "{}",
        }
    )


def observation_windows_from_dyadic(dataset_id: str, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or events["start_time"].isna().all():
        return empty_table("observation_windows")
    left = events[["source_id", "start_time", "end_time"]].rename(
        columns={"source_id": "node_id"}
    )
    right = events[["target_id", "start_time", "end_time"]].rename(
        columns={"target_id": "node_id"}
    )
    appearances = pd.concat([left, right], ignore_index=True)
    bounds = appearances.groupby("node_id", observed=True).agg(
        window_start=("start_time", "min"), window_end=("end_time", "max")
    )
    bounds = bounds.reset_index()
    bounds.insert(0, "dataset_id", dataset_id)
    bounds["observation_status"] = "detected_span"
    bounds["device_id"] = pd.NA
    bounds["coverage_fraction"] = pd.NA
    bounds["reason"] = "Bounds of observed interactions; not continuous device uptime"
    bounds["attributes_json"] = "{}"
    return bounds


def observation_windows_from_groups(
    dataset_id: str,
    groups: pd.DataFrame,
    memberships: pd.DataFrame,
) -> pd.DataFrame:
    if groups.empty or memberships.empty:
        return empty_table("observation_windows")
    appearances = memberships[["group_event_id", "node_id"]].merge(
        groups[["group_event_id", "start_time", "end_time"]],
        on="group_event_id",
        how="left",
        validate="many_to_one",
    )
    bounds = appearances.groupby("node_id", observed=True).agg(
        window_start=("start_time", "min"), window_end=("end_time", "max")
    )
    bounds = bounds.reset_index()
    bounds.insert(0, "dataset_id", dataset_id)
    bounds["observation_status"] = "detected_span"
    bounds["device_id"] = pd.NA
    bounds["coverage_fraction"] = pd.NA
    bounds["reason"] = "Bounds of observed group memberships; not continuous device uptime"
    bounds["attributes_json"] = "{}"
    return bounds


def apply_observation_bounds(
    individuals: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    if windows.empty:
        return individuals
    bounds = windows.groupby("node_id", observed=True).agg(
        first_observed_at=("window_start", "min"),
        last_observed_at=("window_end", "max"),
    )
    result = individuals.drop(columns=["first_observed_at", "last_observed_at"]).merge(
        bounds, on="node_id", how="left", validate="one_to_one"
    )
    return result
