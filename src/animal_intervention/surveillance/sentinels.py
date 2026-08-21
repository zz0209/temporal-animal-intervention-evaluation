from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

import pandas as pd

from animal_intervention.evaluation import stable_hash_order
from animal_intervention.transmission.contract import ExposureStream


def history_pair_weights(
    stream: ExposureStream,
    eligible_nodes: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Aggregate symmetric pairwise past-exposure mass without expanding output rows."""

    eligible = set(map(str, eligible_nodes))
    weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in stream.dyadic_exposures.itertuples(index=False):
        source, target = str(row.source_id), str(row.target_id)
        if source == target or source not in eligible or target not in eligible:
            continue
        duration = (pd.Timestamp(row.end_time) - pd.Timestamp(row.start_time)).total_seconds()
        mass = max(0.0, duration * float(row.hazard_rate_multiplier))
        weights[source][target] += mass
        weights[target][source] += mass

    if not stream.group_exposures.empty and not stream.group_memberships.empty:
        memberships = {
            str(group_id): [
                (str(row.node_id), float(row.membership_weight))
                for row in frame.itertuples(index=False)
                if str(row.node_id) in eligible
            ]
            for group_id, frame in stream.group_memberships.groupby(
                "group_event_id", observed=True
            )
        }
        for row in stream.group_exposures.itertuples(index=False):
            members = memberships.get(str(row.group_event_id), [])
            if len(members) < 2:
                continue
            duration = (pd.Timestamp(row.end_time) - pd.Timestamp(row.start_time)).total_seconds()
            divisor = len(members) - 1 if str(row.group_mixing_mode) == "frequency_dependent" else 1
            base = max(0.0, duration * float(row.hazard_rate_multiplier) / divisor)
            for (left, left_weight), (right, right_weight) in combinations(members, 2):
                mass = base * left_weight * right_weight
                weights[left][right] += mass
                weights[right][left] += mass
    return {node: dict(neighbors) for node, neighbors in weights.items()}


def greedy_history_coverage(
    stream: ExposureStream,
    eligible_nodes: Iterable[str],
    budget: int,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Greedily maximize weighted neighbor coverage using history only.

    The set objective is ``sum_v max_{s in S} w(s, v)``. It rewards strong past
    exposure while discounting sentinels that cover the same neighbors.
    """

    eligible = tuple(sorted(set(map(str, eligible_nodes))))
    if budget <= 0 or not eligible:
        return ()
    budget = min(int(budget), len(eligible))
    pair_weights = history_pair_weights(stream, eligible)
    tie_order = stable_hash_order(eligible, seed, "history_coverage_ties")
    tie_rank = {node: index for index, node in enumerate(tie_order)}
    covered = {node: 0.0 for node in eligible}
    selected: list[str] = []
    remaining = set(eligible)
    for _ in range(budget):
        gains = {}
        for candidate in remaining:
            gains[candidate] = sum(
                max(0.0, weight - covered.get(neighbor, 0.0))
                for neighbor, weight in pair_weights.get(candidate, {}).items()
                if neighbor not in selected
            )
        chosen = min(
            remaining,
            key=lambda node: (-gains[node], tie_rank[node], node),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        for neighbor, weight in pair_weights.get(chosen, {}).items():
            covered[neighbor] = max(covered.get(neighbor, 0.0), weight)
    return tuple(selected)
