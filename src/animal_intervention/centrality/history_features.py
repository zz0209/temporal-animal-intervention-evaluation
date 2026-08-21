from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.transmission import ExposureStream, compile_primary_exposure


KEY_COLUMNS = ["dataset_id", "network_id", "anchor_time", "candidate_id"]


def _overlap_seconds(
    starts: pd.Series,
    ends: pd.Series,
    history_start: pd.Timestamp,
    anchor_time: pd.Timestamp,
) -> np.ndarray:
    clipped_starts = starts.clip(lower=history_start)
    clipped_ends = ends.clip(upper=anchor_time)
    return (clipped_ends - clipped_starts).dt.total_seconds().clip(lower=0).to_numpy()


def _candidate_nodes(
    dataset: CanonicalDataset,
    network_id: str,
) -> set[str] | None:
    if dataset.metadata.dataset_id != "domestic_sheep_sirtrack":
        return None
    individuals = dataset.individuals
    return set(
        individuals.loc[individuals["group_id"].astype(str).eq(network_id), "node_id"]
        .astype(str)
        .tolist()
    )


def _empty_records(candidates: list[str]) -> dict[str, dict[str, object]]:
    return {
        node: {
            "activity_count": 0.0,
            "contact_opportunity_seconds": 0.0,
            "weighted_exposure": 0.0,
            "group_size_sum": 0.0,
            "location_ids": set(),
            "partners": set(),
            "event_times": [],
        }
        for node in candidates
    }


def _add_dyadic_history(
    records: dict[str, dict[str, object]],
    exposures: pd.DataFrame,
    history_start: pd.Timestamp,
    anchor_time: pd.Timestamp,
    allowed_nodes: set[str] | None,
) -> None:
    if exposures.empty:
        return
    starts = pd.to_datetime(exposures["start_time"])
    ends = pd.to_datetime(exposures["end_time"])
    mask = starts.lt(anchor_time) & ends.gt(history_start)
    history = exposures.loc[mask].copy()
    if allowed_nodes is not None:
        history = history.loc[
            history["source_id"].astype(str).isin(allowed_nodes)
            & history["target_id"].astype(str).isin(allowed_nodes)
        ]
    if history.empty:
        return
    history["overlap_seconds"] = _overlap_seconds(
        pd.to_datetime(history["start_time"]),
        pd.to_datetime(history["end_time"]),
        history_start,
        anchor_time,
    )
    for row in history.itertuples(index=False):
        source = str(row.source_id)
        target = str(row.target_id)
        multiplier = float(row.hazard_rate_multiplier)
        event_time = min(pd.Timestamp(row.end_time), anchor_time)
        for node, partner in ((source, target), (target, source)):
            if node not in records:
                continue
            record = records[node]
            record["activity_count"] += 1.0
            record["contact_opportunity_seconds"] += float(row.overlap_seconds)
            record["weighted_exposure"] += float(row.overlap_seconds) * multiplier
            record["group_size_sum"] += 2.0
            record["partners"].add(partner)
            record["event_times"].append(event_time)
            if pd.notna(row.location_id):
                record["location_ids"].add(str(row.location_id))


def _add_group_history(
    records: dict[str, dict[str, object]],
    exposures: pd.DataFrame,
    memberships: pd.DataFrame,
    history_start: pd.Timestamp,
    anchor_time: pd.Timestamp,
    allowed_nodes: set[str] | None,
) -> None:
    if exposures.empty:
        return
    starts = pd.to_datetime(exposures["start_time"])
    ends = pd.to_datetime(exposures["end_time"])
    mask = starts.lt(anchor_time) & ends.gt(history_start)
    history = exposures.loc[mask].copy()
    if history.empty:
        return
    history["overlap_seconds"] = _overlap_seconds(
        pd.to_datetime(history["start_time"]),
        pd.to_datetime(history["end_time"]),
        history_start,
        anchor_time,
    )
    history = history.set_index(history["group_event_id"].astype(str), drop=False)
    event_ids = set(history.index)
    selected_memberships = memberships.loc[
        memberships["group_event_id"].astype(str).isin(event_ids)
    ]
    member_map = {
        str(event_id): frame["node_id"].astype(str).tolist()
        for event_id, frame in selected_memberships.groupby("group_event_id", observed=True)
    }
    for event_id, members in member_map.items():
        if allowed_nodes is not None:
            members = [node for node in members if node in allowed_nodes]
        if len(members) < 2:
            continue
        row = history.loc[event_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        duration = float(row["overlap_seconds"])
        multiplier = float(row["hazard_rate_multiplier"])
        event_time = min(pd.Timestamp(row["end_time"]), anchor_time)
        member_set = set(members)
        for node in members:
            if node not in records:
                continue
            record = records[node]
            record["activity_count"] += 1.0
            record["contact_opportunity_seconds"] += duration * (len(members) - 1)
            record["weighted_exposure"] += duration * multiplier
            record["group_size_sum"] += float(len(members))
            record["partners"].update(member_set - {node})
            record["event_times"].append(event_time)
            if pd.notna(row["location_id"]):
                record["location_ids"].add(str(row["location_id"]))


def _finalize_records(
    labels: pd.DataFrame,
    records: dict[str, dict[str, object]],
    history_start: pd.Timestamp,
    anchor_time: pd.Timestamp,
) -> pd.DataFrame:
    lookback_seconds = max((anchor_time - history_start).total_seconds(), 1.0)
    recent_boundary = anchor_time - pd.Timedelta(seconds=lookback_seconds / 2)
    eligible_nodes = set(labels["candidate_id"].astype(str))
    declared_populations = set(labels["eligible_population"].astype(int))
    if declared_populations != {len(eligible_nodes)}:
        raise ValueError(
            "eligible population does not match candidate rows in history context"
        )
    eligible_denominator = max(len(eligible_nodes) - 1, 1)
    rows: list[dict[str, object]] = []
    for label in labels.itertuples(index=False):
        node = str(label.candidate_id)
        record = records[node]
        event_times = sorted(record["event_times"])
        activity = float(record["activity_count"])
        last_event = event_times[-1] if event_times else history_start
        first_event = event_times[0] if event_times else history_start
        recent_count = sum(value >= recent_boundary for value in event_times)
        rows.append(
            {
                "dataset_id": str(label.dataset_id),
                "network_id": str(label.network_id),
                "anchor_time": anchor_time,
                "candidate_id": node,
                "activity_count": activity,
                "event_rate_per_day": activity * 86400.0 / lookback_seconds,
                "contact_opportunity_rate": float(
                    record["contact_opportunity_seconds"]
                )
                / lookback_seconds,
                "weighted_exposure_rate": float(record["weighted_exposure"])
                / lookback_seconds,
                "eligible_partner_fraction": len(
                    set(record["partners"]) & eligible_nodes
                )
                / eligible_denominator,
                "observed_partner_count": float(len(record["partners"])),
                "location_count": float(len(record["location_ids"])),
                "mean_group_size": (
                    float(record["group_size_sum"]) / activity if activity else 0.0
                ),
                "recency_score": max(
                    0.0,
                    1.0 - (anchor_time - last_event).total_seconds() / lookback_seconds,
                ),
                "recent_activity_fraction": recent_count / activity if activity else 0.0,
                "active_span_fraction": (
                    (last_event - first_event).total_seconds() / lookback_seconds
                    if len(event_times) > 1
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_history_features(
    dataset: CanonicalDataset,
    labels: pd.DataFrame,
    *,
    exposure_stream: ExposureStream | None = None,
) -> pd.DataFrame:
    """Build deployment-safe candidate features using only pre-anchor history."""
    required = set(KEY_COLUMNS + ["history_start", "eligible_population"])
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels are missing required columns: {sorted(missing)}")
    dataset_labels = labels.loc[
        labels["dataset_id"].astype(str).eq(dataset.metadata.dataset_id)
    ].copy()
    if dataset_labels.empty:
        return pd.DataFrame()
    stream = exposure_stream or compile_primary_exposure(dataset)
    if stream.dataset_id != dataset.metadata.dataset_id:
        raise ValueError("exposure stream and canonical dataset IDs do not match")
    dataset_labels["history_start"] = pd.to_datetime(
        dataset_labels["history_start"], format="mixed"
    )
    dataset_labels["anchor_time"] = pd.to_datetime(
        dataset_labels["anchor_time"], format="mixed"
    )
    outputs: list[pd.DataFrame] = []
    context_columns = ["network_id", "history_start", "anchor_time"]
    for (network_id, history_start, anchor_time), frame in dataset_labels.groupby(
        context_columns, sort=True, observed=True
    ):
        candidates = frame["candidate_id"].astype(str).tolist()
        records = _empty_records(candidates)
        allowed_nodes = _candidate_nodes(dataset, str(network_id))
        _add_dyadic_history(
            records,
            stream.dyadic_exposures,
            pd.Timestamp(history_start),
            pd.Timestamp(anchor_time),
            allowed_nodes,
        )
        _add_group_history(
            records,
            stream.group_exposures,
            stream.group_memberships,
            pd.Timestamp(history_start),
            pd.Timestamp(anchor_time),
            allowed_nodes,
        )
        outputs.append(
            _finalize_records(
                frame,
                records,
                pd.Timestamp(history_start),
                pd.Timestamp(anchor_time),
            )
        )
    features = pd.concat(outputs, ignore_index=True)
    if features.duplicated(KEY_COLUMNS).any():
        raise ValueError("history features contain duplicate candidate contexts")
    numeric = features.drop(columns=KEY_COLUMNS)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("history features contain non-finite values")
    return features
