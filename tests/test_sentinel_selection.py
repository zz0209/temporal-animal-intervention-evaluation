from __future__ import annotations

import pandas as pd

from animal_intervention.surveillance import greedy_history_coverage, history_pair_weights
from animal_intervention.transmission.contract import ExposureStream


def _stream() -> ExposureStream:
    rows = []
    for index, (source, target, weight) in enumerate(
        [("a", "b", 10.0), ("a", "c", 9.0), ("d", "e", 8.0)]
    ):
        rows.append(
            {
                "dataset_id": "toy",
                "exposure_id": f"e{index}",
                "source_id": source,
                "target_id": target,
                "start_time": pd.Timestamp("2020-01-01"),
                "end_time": pd.Timestamp("2020-01-01 00:00:01"),
                "hazard_rate_multiplier": weight,
                "directed": False,
                "transmission_route": "contact",
                "mapper_name": "toy",
                "origin_event_id": f"e{index}",
                "location_id": pd.NA,
            }
        )
    return ExposureStream(dataset_id="toy", population_nodes=("a", "b", "c", "d", "e"), dyadic_exposures=pd.DataFrame(rows))


def test_pair_weights_are_symmetric() -> None:
    weights = history_pair_weights(_stream(), ["a", "b", "c", "d", "e"])
    assert weights["a"]["b"] == weights["b"]["a"] == 10.0


def test_coverage_selection_avoids_redundant_second_hub() -> None:
    selected = greedy_history_coverage(_stream(), ["a", "b", "c", "d", "e"], 2, seed=7)
    assert selected[0] == "a"
    assert selected[1] in {"d", "e"}


def test_coverage_selection_is_deterministic_and_bounded() -> None:
    left = greedy_history_coverage(_stream(), ["a", "b"], 5, seed=3)
    right = greedy_history_coverage(_stream(), ["a", "b"], 5, seed=3)
    assert left == right
    assert set(left) == {"a", "b"}
