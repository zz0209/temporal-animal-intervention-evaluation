from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.group_mixing_semantics import (
    _member_time_mean_competitors,
    _with_group_mode,
)
from animal_intervention.transmission import ExposureStream


def _stream() -> ExposureStream:
    return ExposureStream(
        dataset_id="groups",
        population_nodes=("a", "b", "c", "d"),
        group_exposures=pd.DataFrame(
            [
                {
                    "dataset_id": "groups",
                    "group_event_id": "g1",
                    "start_time": "2020-01-01 00:00:00",
                    "end_time": "2020-01-01 00:01:00",
                    "hazard_rate_multiplier": 1.0,
                    "transmission_route": "group_association_proxy",
                    "mapper_name": "GroupMixingMapper",
                    "group_mixing_mode": "frequency_dependent",
                    "location_id": "x",
                },
                {
                    "dataset_id": "groups",
                    "group_event_id": "g2",
                    "start_time": "2020-01-01 00:00:00",
                    "end_time": "2020-01-01 00:01:00",
                    "hazard_rate_multiplier": 1.0,
                    "transmission_route": "group_association_proxy",
                    "mapper_name": "GroupMixingMapper",
                    "group_mixing_mode": "frequency_dependent",
                    "location_id": "x",
                },
            ]
        ),
        group_memberships=pd.DataFrame(
            [
                {"dataset_id": "groups", "group_event_id": group, "node_id": node, "membership_weight": 1.0}
                for group, nodes in [("g1", ["a", "b"]), ("g2", ["a", "b", "c", "d"])]
                for node in nodes
            ]
        ),
    )


def test_member_time_normalization_weights_animals_and_duration() -> None:
    assert _member_time_mean_competitors(_stream()) == 7 / 3


def test_group_mode_change_does_not_mutate_source() -> None:
    source = _stream()
    changed = _with_group_mode(source, "undiluted_clique")
    assert set(source.group_exposures["group_mixing_mode"]) == {"frequency_dependent"}
    assert set(changed.group_exposures["group_mixing_mode"]) == {"undiluted_clique"}
