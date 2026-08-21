from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Callable, Literal

import networkx as nx
import numpy as np
import pandas as pd

from animal_intervention.data.contract import CanonicalDataset
from animal_intervention.transmission import ExposureStream


@dataclass(frozen=True, slots=True)
class Interaction:
    members: tuple[str, ...]
    integrated_weight: float


@dataclass(frozen=True, slots=True)
class TemporalBatch:
    event_time: pd.Timestamp
    interactions: tuple[Interaction, ...]


def _allowed_nodes(dataset: CanonicalDataset, network_id: str) -> set[str] | None:
    if dataset.metadata.dataset_id != "domestic_sheep_sirtrack":
        return None
    return set(
        dataset.individuals.loc[
            dataset.individuals["group_id"].astype(str).eq(network_id), "node_id"
        ].astype(str)
    )


def build_temporal_batches(
    dataset: CanonicalDataset,
    stream: ExposureStream,
    *,
    network_id: str,
    history_start: pd.Timestamp,
    anchor_time: pd.Timestamp,
    event_time_rule: Literal["start", "midpoint", "end"] = "end",
) -> tuple[list[TemporalBatch], set[str]]:
    """Create simultaneous interaction batches from one deployment history window."""
    if event_time_rule not in {"start", "midpoint", "end"}:
        raise ValueError(f"unsupported event_time_rule: {event_time_rule}")

    def event_time(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
        if event_time_rule == "start":
            return start
        if event_time_rule == "midpoint":
            return start + (end - start) / 2
        return end

    allowed = _allowed_nodes(dataset, network_id)
    by_time: dict[pd.Timestamp, list[Interaction]] = {}
    observed_nodes: set[str] = set()

    dyadic = stream.dyadic_exposures
    if not dyadic.empty:
        starts = pd.to_datetime(dyadic["start_time"])
        ends = pd.to_datetime(dyadic["end_time"])
        history = dyadic.loc[starts.lt(anchor_time) & ends.gt(history_start)]
        for row in history.itertuples(index=False):
            source = str(row.source_id)
            target = str(row.target_id)
            if allowed is not None and (source not in allowed or target not in allowed):
                continue
            start = max(pd.Timestamp(row.start_time), history_start)
            end = min(pd.Timestamp(row.end_time), anchor_time)
            duration = (end - start).total_seconds()
            if duration <= 0:
                continue
            members = (source, target)
            interaction = Interaction(
                members=members,
                integrated_weight=duration * float(row.hazard_rate_multiplier),
            )
            by_time.setdefault(event_time(start, end), []).append(interaction)
            observed_nodes.update(members)

    groups = stream.group_exposures
    if not groups.empty:
        starts = pd.to_datetime(groups["start_time"])
        ends = pd.to_datetime(groups["end_time"])
        history = groups.loc[starts.lt(anchor_time) & ends.gt(history_start)].copy()
        history_ids = set(history["group_event_id"].astype(str))
        memberships = stream.group_memberships.loc[
            stream.group_memberships["group_event_id"].astype(str).isin(history_ids)
        ]
        member_map = {
            str(group_id): tuple(dict.fromkeys(frame["node_id"].astype(str)))
            for group_id, frame in memberships.groupby("group_event_id", observed=True)
        }
        for row in history.itertuples(index=False):
            members = member_map.get(str(row.group_event_id), ())
            if allowed is not None:
                members = tuple(node for node in members if node in allowed)
            if len(members) < 2:
                continue
            start = max(pd.Timestamp(row.start_time), history_start)
            end = min(pd.Timestamp(row.end_time), anchor_time)
            duration = (end - start).total_seconds()
            if duration <= 0:
                continue
            divisor = (
                len(members) - 1
                if str(row.group_mixing_mode) == "frequency_dependent"
                else 1
            )
            interaction = Interaction(
                members=tuple(sorted(members)),
                integrated_weight=(
                    duration * float(row.hazard_rate_multiplier) / divisor
                ),
            )
            by_time.setdefault(event_time(start, end), []).append(interaction)
            observed_nodes.update(members)

    batches = [
        TemporalBatch(time, tuple(by_time[time])) for time in sorted(by_time)
    ]
    return batches, observed_nodes


def aggregate_graph(nodes: set[str], batches: list[TemporalBatch]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    for batch in batches:
        for interaction in batch.interactions:
            for source, target in combinations(interaction.members, 2):
                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += interaction.integrated_weight
                    graph[source][target]["event_count"] += 1
                else:
                    graph.add_edge(
                        source,
                        target,
                        weight=float(interaction.integrated_weight),
                        event_count=1,
                    )
    return graph


def static_centralities(graph: nx.Graph) -> dict[str, dict[str, float]]:
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    degree = nx.degree_centrality(graph)
    strength = dict(graph.degree(weight="weight"))
    if graph.number_of_edges():
        node_index = {node: position for position, node in enumerate(nodes)}
        edge_rows = list(graph.edges(data=True))
        sources = np.asarray([node_index[row[0]] for row in edge_rows], dtype=int)
        targets = np.asarray([node_index[row[1]] for row in edge_rows], dtype=int)
        weights = np.asarray([float(row[2]["weight"]) for row in edge_rows])
        strength_array = np.asarray([float(strength[node]) for node in nodes])
        damping = 0.85
        pagerank_array = np.full(len(nodes), 1.0 / len(nodes), dtype=float)
        for _ in range(1000):
            dangling_mass = float(
                pagerank_array[strength_array <= 0].sum()
            )
            updated = np.full(
                len(nodes),
                (1.0 - damping + damping * dangling_mass) / len(nodes),
                dtype=float,
            )
            np.add.at(
                updated,
                targets,
                damping * pagerank_array[sources] * weights / strength_array[sources],
            )
            np.add.at(
                updated,
                sources,
                damping * pagerank_array[targets] * weights / strength_array[targets],
            )
            error = float(np.abs(updated - pagerank_array).sum())
            pagerank_array = updated
            if error < len(nodes) * 1e-10:
                break
        else:
            raise nx.PowerIterationFailedConvergence(1000)
        pagerank = {
            node: float(pagerank_array[position])
            for node, position in node_index.items()
        }
        scaled_weights = weights / float(weights.max())
        eigenvector_array = np.full(len(nodes), 1.0 / np.sqrt(len(nodes)))
        for _ in range(2000):
            updated = eigenvector_array.copy()
            np.add.at(
                updated,
                sources,
                scaled_weights * eigenvector_array[targets],
            )
            np.add.at(
                updated,
                targets,
                scaled_weights * eigenvector_array[sources],
            )
            norm = float(np.linalg.norm(updated))
            updated /= norm
            if float(np.abs(updated - eigenvector_array).sum()) < len(nodes) * 1e-10:
                eigenvector_array = updated
                break
            eigenvector_array = updated
        eigenvector = {
            node: float(eigenvector_array[position])
            for node, position in node_index.items()
        }
        core = nx.core_number(graph)
    else:
        uniform = 1.0 / len(nodes)
        pagerank = {node: uniform for node in nodes}
        eigenvector = {node: uniform for node in nodes}
        core = {node: 0 for node in nodes}
    return {
        node: {
            "static_degree": float(degree[node]),
            "static_strength": float(strength[node]),
            "static_pagerank": float(pagerank[node]),
            "static_eigenvector": float(eigenvector[node]),
            "static_k_core": float(core[node]),
        }
        for node in nodes
    }


def temporal_reachability(
    nodes: list[str], batches: list[TemporalBatch]
) -> dict[str, float]:
    """Return all-source reach fractions under strict inter-batch time order."""
    if not nodes:
        return {}
    index = {node: position for position, node in enumerate(nodes)}
    sources_at_node = [1 << position for position in range(len(nodes))]
    for batch in batches:
        updates: dict[int, int] = {}
        for interaction in batch.interactions:
            member_indices = [index[node] for node in interaction.members]
            combined = 0
            for member_index in member_indices:
                combined |= sources_at_node[member_index]
            for member_index in member_indices:
                updates[member_index] = updates.get(
                    member_index, sources_at_node[member_index]
                ) | combined
        for member_index, value in updates.items():
            sources_at_node[member_index] = value
    reached_counts = np.zeros(len(nodes), dtype=np.int64)
    for value in sources_at_node:
        while value:
            least_bit = value & -value
            reached_counts[least_bit.bit_length() - 1] += 1
            value ^= least_bit
    denominator = max(len(nodes) - 1, 1)
    return {
        node: float(max(reached_counts[position] - 1, 0) / denominator)
        for node, position in index.items()
    }


def temporal_reachability_pair(
    nodes: list[str], batches: list[TemporalBatch]
) -> tuple[dict[str, float], dict[str, float]]:
    """Return outgoing and incoming strict time-respecting reach fractions."""
    if not nodes:
        return {}, {}
    index = {node: position for position, node in enumerate(nodes)}
    sources_at_node = [1 << position for position in range(len(nodes))]
    for batch in batches:
        updates: dict[int, int] = {}
        for interaction in batch.interactions:
            member_indices = [index[node] for node in interaction.members]
            combined = 0
            for member_index in member_indices:
                combined |= sources_at_node[member_index]
            for member_index in member_indices:
                updates[member_index] = updates.get(
                    member_index, sources_at_node[member_index]
                ) | combined
        for member_index, value in updates.items():
            sources_at_node[member_index] = value
    outgoing_counts = np.zeros(len(nodes), dtype=np.int64)
    incoming_counts = np.asarray(
        [value.bit_count() for value in sources_at_node], dtype=np.int64
    )
    for value in sources_at_node:
        while value:
            least_bit = value & -value
            outgoing_counts[least_bit.bit_length() - 1] += 1
            value ^= least_bit
    denominator = max(len(nodes) - 1, 1)
    outgoing = {
        node: float(max(outgoing_counts[position] - 1, 0) / denominator)
        for node, position in index.items()
    }
    incoming = {
        node: float(max(incoming_counts[position] - 1, 0) / denominator)
        for node, position in index.items()
    }
    return outgoing, incoming


def _communicability(
    nodes: list[str],
    batches: list[TemporalBatch],
    *,
    direction: Literal["broadcast", "receive"],
    attenuation: float | None = None,
    betas: tuple[float, ...] | None = None,
) -> dict[str, float]:
    if direction not in {"broadcast", "receive"}:
        raise ValueError(f"unsupported communicability direction: {direction}")
    if (attenuation is None) == (betas is None):
        raise ValueError("provide exactly one of attenuation or betas")
    if attenuation is not None and not 0 < attenuation < 1:
        raise ValueError("attenuation must be between zero and one")
    if betas is not None and (not betas or any(beta <= 0 for beta in betas)):
        raise ValueError("betas must contain only positive values")
    if not nodes:
        return {}
    index = {node: position for position, node in enumerate(nodes)}
    values = np.ones(len(nodes), dtype=float)
    ordered_batches = reversed(batches) if direction == "broadcast" else batches
    for batch in ordered_batches:
        increment = np.zeros(len(nodes), dtype=float)
        for interaction in batch.interactions:
            member_indices = np.array(
                [index[node] for node in interaction.members], dtype=int
            )
            member_values = values[member_indices]
            if betas is None:
                pair_gain = float(attenuation) / max(len(member_indices) - 1, 1)
            else:
                pair_gain = float(
                    np.mean(
                        [
                            -np.expm1(-beta * interaction.integrated_weight)
                            for beta in betas
                        ]
                    )
                )
            increment[member_indices] += pair_gain * (
                member_values.sum() - member_values
            )
        values += increment
        maximum = float(values.max())
        if maximum > 0:
            values /= maximum
    return {node: float(values[position]) for node, position in index.items()}


def dynamic_communicability(
    nodes: list[str],
    batches: list[TemporalBatch],
    *,
    attenuation: float,
) -> dict[str, float]:
    """Compute single-transition-per-batch Grindrod broadcast centrality."""
    return _communicability(
        nodes,
        batches,
        direction="broadcast",
        attenuation=attenuation,
    )


def communicability_pair(
    nodes: list[str],
    batches: list[TemporalBatch],
    *,
    attenuation: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return unweighted Grindrod-style broadcast and receive scores."""
    return (
        _communicability(
            nodes, batches, direction="broadcast", attenuation=attenuation
        ),
        _communicability(
            nodes, batches, direction="receive", attenuation=attenuation
        ),
    )


def exposure_communicability_pair(
    nodes: list[str],
    batches: list[TemporalBatch],
    *,
    betas: tuple[float, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return duration-aware broadcast and receive scores over selected betas."""
    if not betas or any(beta <= 0 for beta in betas):
        raise ValueError("betas must contain only positive values")
    if not nodes:
        return {}, {}
    index = {node: position for position, node in enumerate(nodes)}
    beta_array = np.asarray(betas, dtype=float)

    def calculate(direction: Literal["broadcast", "receive"]) -> dict[str, float]:
        values = np.ones((len(beta_array), len(nodes)), dtype=float)
        ordered_batches = reversed(batches) if direction == "broadcast" else batches
        for batch in ordered_batches:
            increment = np.zeros_like(values)
            for interaction in batch.interactions:
                member_indices = np.asarray(
                    [index[node] for node in interaction.members], dtype=int
                )
                member_values = values[:, member_indices]
                pair_gains = -np.expm1(
                    -beta_array * interaction.integrated_weight
                )[:, None]
                increment[:, member_indices] += pair_gains * (
                    member_values.sum(axis=1, keepdims=True) - member_values
                )
            values += increment
            maxima = values.max(axis=1, keepdims=True)
            values /= np.where(maxima > 0, maxima, 1.0)
        averaged = values.mean(axis=0)
        return {
            node: float(averaged[position]) for node, position in index.items()
        }

    return calculate("broadcast"), calculate("receive")


def shuffled_batches(
    batches: list[TemporalBatch], *, seed: int
) -> list[TemporalBatch]:
    """Permute simultaneous event batches while preserving every batch intact."""
    if len(batches) < 2:
        return list(batches)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(batches))
    times = [batch.event_time for batch in batches]
    return [
        TemporalBatch(times[position], batches[source].interactions)
        for position, source in enumerate(permutation)
    ]


TEMPORAL_AUDIT_METHODS = (
    "temporal_reach_out",
    "temporal_reach_in",
    "dynamic_communicability_broadcast",
    "dynamic_communicability_receive",
    "exposure_communicability_broadcast",
    "exposure_communicability_receive",
)


def temporal_audit_scores(
    nodes: list[str],
    batches: list[TemporalBatch],
    *,
    attenuation: float,
    betas: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    """Compute the ordered temporal methods used by the repaired audit."""
    reach_out, reach_in = temporal_reachability_pair(nodes, batches)
    broadcast, receive = communicability_pair(
        nodes, batches, attenuation=attenuation
    )
    exposure_broadcast, exposure_receive = exposure_communicability_pair(
        nodes, batches, betas=betas
    )
    return {
        "temporal_reach_out": reach_out,
        "temporal_reach_in": reach_in,
        "dynamic_communicability_broadcast": broadcast,
        "dynamic_communicability_receive": receive,
        "exposure_communicability_broadcast": exposure_broadcast,
        "exposure_communicability_receive": exposure_receive,
    }


def build_temporal_audit_tables(
    dataset: CanonicalDataset,
    stream: ExposureStream,
    labels: pd.DataFrame,
    *,
    attenuation: float,
    betas: tuple[float, ...],
    shuffle_replicates: int,
    random_seed: int,
    sensitivity_time_rules: tuple[str, ...] = ("start", "midpoint", "end"),
    context_progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ordered, replicate-level shuffled, and event-time sensitivity scores."""
    if shuffle_replicates < 1:
        raise ValueError("shuffle_replicates must be positive")
    ordered_rows: list[dict[str, object]] = []
    shuffled_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    context_columns = ["network_id", "history_start", "anchor_time"]
    grouped = labels.groupby(context_columns, sort=True, observed=True)
    context_count = grouped.ngroups
    dataset_seed = int.from_bytes(
        hashlib.sha256(dataset.metadata.dataset_id.encode("utf-8")).digest()[:4],
        "big",
    )
    for context_index, (
        (network_id, history_start, anchor_time),
        frame,
    ) in enumerate(grouped):
        history_start = pd.Timestamp(history_start)
        anchor_time = pd.Timestamp(anchor_time)
        candidates = list(dict.fromkeys(frame["candidate_id"].astype(str)))
        batches, observed_nodes = build_temporal_batches(
            dataset,
            stream,
            network_id=str(network_id),
            history_start=history_start,
            anchor_time=anchor_time,
            event_time_rule="end",
        )
        all_nodes = sorted(observed_nodes | set(candidates))
        graph = aggregate_graph(set(all_nodes), batches)
        static = static_centralities(graph)
        ordered = temporal_audit_scores(
            all_nodes, batches, attenuation=attenuation, betas=betas
        )
        for candidate in candidates:
            ordered_rows.append(
                {
                    "dataset_id": dataset.metadata.dataset_id,
                    "network_id": str(network_id),
                    "anchor_time": anchor_time,
                    "candidate_id": candidate,
                    **static[candidate],
                    **{method: scores[candidate] for method, scores in ordered.items()},
                    "history_observed_nodes": len(all_nodes),
                    "history_event_batches": len(batches),
                    "history_interactions": sum(
                        len(batch.interactions) for batch in batches
                    ),
                }
            )
        for replicate in range(shuffle_replicates):
            shuffled = shuffled_batches(
                batches,
                seed=(
                    random_seed
                    + dataset_seed
                    + context_index * 100_003
                    + replicate
                ),
            )
            shuffled_scores = temporal_audit_scores(
                all_nodes, shuffled, attenuation=attenuation, betas=betas
            )
            for method, scores in shuffled_scores.items():
                for candidate in candidates:
                    shuffled_rows.append(
                        {
                            "dataset_id": dataset.metadata.dataset_id,
                            "network_id": str(network_id),
                            "anchor_time": anchor_time,
                            "candidate_id": candidate,
                            "shuffle_replicate": replicate,
                            "method": method,
                            "score": scores[candidate],
                        }
                    )
        for time_rule in sensitivity_time_rules:
            sensitivity_batches, sensitivity_nodes = build_temporal_batches(
                dataset,
                stream,
                network_id=str(network_id),
                history_start=history_start,
                anchor_time=anchor_time,
                event_time_rule=time_rule,
            )
            if sensitivity_nodes != observed_nodes:
                raise ValueError("event-time rule changed the observed node set")
            exposure_broadcast, exposure_receive = exposure_communicability_pair(
                all_nodes, sensitivity_batches, betas=betas
            )
            for direction, scores in (
                ("broadcast", exposure_broadcast),
                ("receive", exposure_receive),
            ):
                for candidate in candidates:
                    sensitivity_rows.append(
                        {
                            "dataset_id": dataset.metadata.dataset_id,
                            "network_id": str(network_id),
                            "anchor_time": anchor_time,
                            "candidate_id": candidate,
                            "event_time_rule": time_rule,
                            "direction": direction,
                            "score": scores[candidate],
                        }
                    )
        if context_progress is not None:
            context_progress(context_index + 1, context_count)
    return (
        pd.DataFrame(ordered_rows),
        pd.DataFrame(shuffled_rows),
        pd.DataFrame(sensitivity_rows),
    )


def build_reference_centralities(
    dataset: CanonicalDataset,
    stream: ExposureStream,
    labels: pd.DataFrame,
    *,
    attenuation: float,
    shuffle_replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    if shuffle_replicates < 1:
        raise ValueError("shuffle_replicates must be positive")
    rows: list[dict[str, object]] = []
    context_columns = ["network_id", "history_start", "anchor_time"]
    grouped = labels.groupby(context_columns, sort=True, observed=True)
    for context_index, (
        (network_id, history_start, anchor_time),
        frame,
    ) in enumerate(grouped):
        history_start = pd.Timestamp(history_start)
        anchor_time = pd.Timestamp(anchor_time)
        candidates = list(dict.fromkeys(frame["candidate_id"].astype(str)))
        batches, observed_nodes = build_temporal_batches(
            dataset,
            stream,
            network_id=str(network_id),
            history_start=history_start,
            anchor_time=anchor_time,
        )
        all_nodes = sorted(observed_nodes | set(candidates))
        graph = aggregate_graph(set(all_nodes), batches)
        static = static_centralities(graph)
        ordered_reach = temporal_reachability(all_nodes, batches)
        ordered_communicability = dynamic_communicability(
            all_nodes, batches, attenuation=attenuation
        )
        shuffled_reach: dict[str, list[float]] = {node: [] for node in all_nodes}
        shuffled_communicability: dict[str, list[float]] = {
            node: [] for node in all_nodes
        }
        for replicate in range(shuffle_replicates):
            shuffled = shuffled_batches(
                batches,
                seed=random_seed + context_index * 100_003 + replicate,
            )
            reach = temporal_reachability(all_nodes, shuffled)
            communicability = dynamic_communicability(
                all_nodes, shuffled, attenuation=attenuation
            )
            reach_percentiles = pd.Series(reach).rank(method="average", pct=True)
            communicability_percentiles = pd.Series(communicability).rank(
                method="average", pct=True
            )
            for node in all_nodes:
                shuffled_reach[node].append(float(reach_percentiles[node]))
                shuffled_communicability[node].append(
                    float(communicability_percentiles[node])
                )
        interaction_count = sum(len(batch.interactions) for batch in batches)
        for candidate in candidates:
            rows.append(
                {
                    "dataset_id": dataset.metadata.dataset_id,
                    "network_id": str(network_id),
                    "anchor_time": anchor_time,
                    "candidate_id": candidate,
                    **static[candidate],
                    "ordered_temporal_reach": ordered_reach[candidate],
                    "ordered_dynamic_communicability": ordered_communicability[
                        candidate
                    ],
                    "shuffled_temporal_reach": float(
                        np.mean(shuffled_reach[candidate])
                    ),
                    "shuffled_dynamic_communicability": float(
                        np.mean(shuffled_communicability[candidate])
                    ),
                    "history_observed_nodes": len(all_nodes),
                    "history_event_batches": len(batches),
                    "history_interactions": interaction_count,
                }
            )
    return pd.DataFrame(rows)
