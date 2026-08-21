from __future__ import annotations

import networkx as nx
import pytest

from animal_intervention.networks.views import build_networkx_view, build_snapshot_views
from animal_intervention.transmission.mappers import DurationContactMapper, GroupMixingMapper


def test_dyadic_network_view_integrates_exposure(dyadic_dataset):
    graph = build_networkx_view(DurationContactMapper().compile(dyadic_dataset))
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph["A"]["B"]["integrated_hazard_multiplier"] == pytest.approx(10.0)


def test_group_projection_is_frequency_diluted(group_dataset):
    graph = build_networkx_view(GroupMixingMapper().compile(group_dataset))
    assert graph.number_of_edges() == 3
    assert graph["A"]["B"]["integrated_hazard_multiplier"] == pytest.approx(5.0)


def test_snapshot_views_keep_empty_time_bins(dyadic_dataset):
    snapshots = build_snapshot_views(
        DurationContactMapper().compile(dyadic_dataset), bin_size="5s"
    )
    assert len(snapshots) == 2
    assert all(graph.graph["window_start"] for graph in snapshots)


def test_network_view_is_invariant_to_node_relabeling(dyadic_dataset):
    stream = DurationContactMapper().compile(dyadic_dataset)
    relabeled = DurationContactMapper().compile(dyadic_dataset)
    mapping = {"A": "node_x", "B": "node_y"}
    relabeled.dyadic_exposures["source_id"] = relabeled.dyadic_exposures[
        "source_id"
    ].map(mapping)
    relabeled.dyadic_exposures["target_id"] = relabeled.dyadic_exposures[
        "target_id"
    ].map(mapping)
    original_graph = build_networkx_view(stream)
    relabeled_graph = build_networkx_view(relabeled)
    assert nx.is_isomorphic(
        original_graph,
        relabeled_graph,
        edge_match=lambda left, right: (
            left["event_count"] == right["event_count"]
            and left["total_exposure_seconds"]
            == pytest.approx(right["total_exposure_seconds"])
            and left["integrated_hazard_multiplier"]
            == pytest.approx(right["integrated_hazard_multiplier"])
        ),
    )
