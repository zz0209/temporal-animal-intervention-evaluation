from __future__ import annotations

from itertools import combinations

import networkx as nx
import pandas as pd

from animal_intervention.transmission.contract import ExposureStream


def _add_edge(
    graph: nx.Graph,
    source: str,
    target: str,
    *,
    duration_seconds: float,
    integrated_hazard_multiplier: float,
) -> None:
    if graph.has_edge(source, target):
        graph[source][target]["event_count"] += 1
        graph[source][target]["total_exposure_seconds"] += duration_seconds
        graph[source][target]["integrated_hazard_multiplier"] += integrated_hazard_multiplier
    else:
        graph.add_edge(
            source,
            target,
            event_count=1,
            total_exposure_seconds=float(duration_seconds),
            integrated_hazard_multiplier=float(integrated_hazard_multiplier),
        )


def build_networkx_view(
    stream: ExposureStream,
    *,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> nx.Graph:
    """Aggregate a temporal exposure stream into an auditable NetworkX view."""
    stream.validate()
    starts = pd.concat(
        [
            pd.to_datetime(stream.dyadic_exposures["start_time"], errors="coerce"),
            pd.to_datetime(stream.group_exposures["start_time"], errors="coerce"),
        ]
    ).dropna()
    ends = pd.concat(
        [
            pd.to_datetime(stream.dyadic_exposures["end_time"], errors="coerce"),
            pd.to_datetime(stream.group_exposures["end_time"], errors="coerce"),
        ]
    ).dropna()
    if starts.empty:
        raise ValueError("exposure stream has no time-located events")
    window_start = pd.Timestamp(start_time or starts.min())
    window_end = pd.Timestamp(end_time or ends.max())
    if window_end <= window_start:
        raise ValueError("window end must follow window start")

    graph = nx.Graph(
        dataset_id=stream.dataset_id,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        weight_semantics="integrated_hazard_multiplier_before_beta",
    )
    graph.add_nodes_from(stream.nodes())
    for row in stream.dyadic_exposures.itertuples(index=False):
        event_start = max(pd.Timestamp(row.start_time), window_start)
        event_end = min(pd.Timestamp(row.end_time), window_end)
        if event_end <= event_start:
            continue
        overlap = (event_end - event_start).total_seconds()
        _add_edge(
            graph,
            str(row.source_id),
            str(row.target_id),
            duration_seconds=overlap,
            integrated_hazard_multiplier=overlap * float(row.hazard_rate_multiplier),
        )

    memberships = {
        str(group_id): [str(value) for value in frame["node_id"]]
        for group_id, frame in stream.group_memberships.groupby("group_event_id", observed=True)
    }
    for row in stream.group_exposures.itertuples(index=False):
        event_start = max(pd.Timestamp(row.start_time), window_start)
        event_end = min(pd.Timestamp(row.end_time), window_end)
        if event_end <= event_start:
            continue
        members = memberships.get(str(row.group_event_id), [])
        if len(members) < 2:
            continue
        overlap = (event_end - event_start).total_seconds()
        divisor = len(members) - 1 if row.group_mixing_mode == "frequency_dependent" else 1
        integrated = overlap * float(row.hazard_rate_multiplier) / divisor
        for source, target in combinations(sorted(members), 2):
            _add_edge(
                graph,
                source,
                target,
                duration_seconds=overlap,
                integrated_hazard_multiplier=integrated,
            )
    return graph


def build_snapshot_views(
    stream: ExposureStream,
    *,
    bin_size: str | pd.Timedelta,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> list[nx.Graph]:
    starts = pd.concat(
        [
            pd.to_datetime(stream.dyadic_exposures["start_time"], errors="coerce"),
            pd.to_datetime(stream.group_exposures["start_time"], errors="coerce"),
        ]
    ).dropna()
    ends = pd.concat(
        [
            pd.to_datetime(stream.dyadic_exposures["end_time"], errors="coerce"),
            pd.to_datetime(stream.group_exposures["end_time"], errors="coerce"),
        ]
    ).dropna()
    if starts.empty:
        return []
    window_start = pd.Timestamp(start_time or starts.min())
    window_end = pd.Timestamp(end_time or ends.max())
    step = bin_size if isinstance(bin_size, pd.Timedelta) else pd.Timedelta(bin_size)
    snapshots: list[nx.Graph] = []
    cursor = window_start
    while cursor < window_end:
        next_cursor = min(cursor + step, window_end)
        snapshots.append(
            build_networkx_view(stream, start_time=cursor, end_time=next_cursor)
        )
        cursor = next_cursor
    return snapshots
