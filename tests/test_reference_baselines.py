from __future__ import annotations

import pandas as pd

from animal_intervention.centrality import (
    Interaction,
    TemporalBatch,
    aggregate_graph,
    communicability_pair,
    dynamic_communicability,
    exposure_communicability_pair,
    shuffled_batches,
    static_centralities,
    temporal_reachability,
    temporal_reachability_pair,
)


def _ordered_path() -> list[TemporalBatch]:
    return [
        TemporalBatch(
            pd.Timestamp("2020-01-01 01:00:00"),
            (Interaction(("a", "b"), 1.0),),
        ),
        TemporalBatch(
            pd.Timestamp("2020-01-01 02:00:00"),
            (Interaction(("b", "c"), 1.0),),
        ),
    ]


def test_temporal_reachability_respects_the_arrow_of_time() -> None:
    scores = temporal_reachability(["a", "b", "c"], _ordered_path())
    assert scores["a"] == 1.0
    assert scores["c"] == 0.5


def test_dynamic_communicability_identifies_early_broadcaster() -> None:
    scores = dynamic_communicability(
        ["a", "b", "c"], _ordered_path(), attenuation=0.1
    )
    assert scores["a"] > scores["c"]


def test_broadcast_and_receive_follow_opposite_path_directions() -> None:
    broadcast, receive = communicability_pair(
        ["a", "b", "c"], _ordered_path(), attenuation=0.1
    )
    assert broadcast["a"] > broadcast["c"]
    assert receive["c"] > receive["a"]


def test_incoming_and_outgoing_reach_follow_opposite_path_directions() -> None:
    outgoing, incoming = temporal_reachability_pair(
        ["a", "b", "c"], _ordered_path()
    )
    assert outgoing["a"] > outgoing["c"]
    assert incoming["c"] > incoming["a"]


def test_exposure_communicability_uses_integrated_exposure() -> None:
    low_weight = [
        TemporalBatch(
            pd.Timestamp("2020-01-01 01:00:00"),
            (Interaction(("a", "b"), 0.1),),
        ),
        TemporalBatch(
            pd.Timestamp("2020-01-01 02:00:00"),
            (Interaction(("b", "c"), 1.0),),
        ),
    ]
    high_weight = [
        TemporalBatch(
            pd.Timestamp("2020-01-01 01:00:00"),
            (Interaction(("a", "b"), 2.0),),
        ),
        low_weight[1],
    ]
    low_broadcast, _ = exposure_communicability_pair(
        ["a", "b", "c"], low_weight, betas=(0.5,)
    )
    high_broadcast, _ = exposure_communicability_pair(
        ["a", "b", "c"], high_weight, betas=(0.5,)
    )
    assert high_broadcast["a"] > low_broadcast["a"]


def test_exposure_communicability_averages_scenario_scores() -> None:
    combined, _ = exposure_communicability_pair(
        ["a", "b", "c"], _ordered_path(), betas=(0.1, 1.0)
    )
    low, _ = exposure_communicability_pair(
        ["a", "b", "c"], _ordered_path(), betas=(0.1,)
    )
    high, _ = exposure_communicability_pair(
        ["a", "b", "c"], _ordered_path(), betas=(1.0,)
    )
    for node in ("a", "b", "c"):
        assert abs(combined[node] - (low[node] + high[node]) / 2) < 1e-12


def test_same_time_edges_do_not_create_artificial_two_step_walk() -> None:
    batch = TemporalBatch(
        pd.Timestamp("2020-01-01 01:00:00"),
        (Interaction(("a", "b"), 1.0), Interaction(("b", "c"), 1.0)),
    )
    scores = temporal_reachability(["a", "b", "c"], [batch])
    assert scores["a"] == 0.5
    assert scores["c"] == 0.5


def test_group_event_is_one_simultaneous_clique_step() -> None:
    batch = TemporalBatch(
        pd.Timestamp("2020-01-01 01:00:00"),
        (Interaction(("a", "b", "c"), 1.0),),
    )
    scores = temporal_reachability(["a", "b", "c"], [batch])
    assert scores == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_static_reference_scores_match_path_structure() -> None:
    graph = aggregate_graph({"a", "b", "c"}, _ordered_path())
    scores = static_centralities(graph)
    assert scores["b"]["static_degree"] > scores["a"]["static_degree"]
    assert scores["b"]["static_k_core"] == scores["a"]["static_k_core"]
    assert abs(sum(value["static_pagerank"] for value in scores.values()) - 1.0) < 1e-9
    assert scores["b"]["static_eigenvector"] > scores["a"]["static_eigenvector"]


def test_shuffle_preserves_batches_and_is_reproducible() -> None:
    batches = _ordered_path()
    first = shuffled_batches(batches, seed=12)
    second = shuffled_batches(batches, seed=12)
    assert first == second
    assert sorted(repr(batch.interactions) for batch in first) == sorted(
        repr(batch.interactions) for batch in batches
    )
