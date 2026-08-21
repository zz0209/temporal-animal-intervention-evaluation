from types import SimpleNamespace

import pandas as pd

from animal_intervention.experiments.cost_aware_capacity_allocation import (
    _loso_allocations,
    _recognized_detection_metrics,
)


def _natural() -> SimpleNamespace:
    events = pd.DataFrame(
        {
            "time": pd.to_datetime([
                "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"
            ]),
            "node_id": ["a", "b", "c", "d"],
            "event": ["initial_infection", "infection", "infection", "infection"],
        }
    )
    return SimpleNamespace(event_log=events, final_size=4)


def test_perfect_recognition_finds_first_infected_sentinel() -> None:
    result = _recognized_detection_metrics(_natural(), {"b", "d"}, 4, 1.0, 91)
    assert result["detected"]
    assert result["detection_time"] == pd.Timestamp("2020-01-02")
    assert result["detection_burden"] == 2
    assert result["detected_nodes"] == {"b"}


def test_recognition_is_nested_under_shared_keyed_randomness() -> None:
    low = _recognized_detection_metrics(_natural(), {"b", "c", "d"}, 4, 0.5, 12)
    high = _recognized_detection_metrics(_natural(), {"b", "c", "d"}, 4, 1.0, 12)
    low_time = low["detection_time"] if low["detected"] else pd.Timestamp.max
    assert high["detection_time"] <= low_time
    assert high["detection_burden"] <= low["detection_burden"]


def _family_policy() -> pd.DataFrame:
    rows = []
    for index, family in enumerate(["a", "b", "c", "d", "e"]):
        for sentinel, response, policy, value in [
            (0.10, 0.05, "s10_r05", 0.30 + index / 100),
            (0.05, 0.10, "s05_r10", 0.20 + index / 100),
            (0.20, 0.10, "s20_r10", 0.05 + index / 100),
        ]:
            rows.append(
                {
                    "epidemic_model": "temporal_sir",
                    "recognition_sensitivity": 0.5,
                    "sentinel_fraction": sentinel,
                    "response_fraction": response,
                    "policy": policy,
                    "system_family": family,
                    "mean_value": value,
                }
            )
    return pd.DataFrame(rows)


def test_loso_selector_uses_no_heldout_outcomes_and_respects_cost() -> None:
    result = _loso_allocations(_family_policy(), [1.0], 0.10, 0.05)
    assert set(result["selected_policy"]) == {"s05_r10"}
    assert result["value"].gt(0).all()
    assert result["selected_nominal_cost"].le(result["nominal_budget"]).all()
    assert all(
        row.system_family not in row.training_families.split("|")
        for row in result.itertuples(index=False)
    )
    assert "s20_r10" not in set(result["selected_policy"])
