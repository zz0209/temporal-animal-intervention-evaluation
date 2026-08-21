from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.local_probe_veto import build_forward_evaluations


def test_probe_uses_only_strictly_prior_times_and_vetoes_harm() -> None:
    rows = []
    for anchor_id, time, candidate_value in [
        ("a1", "2020-01-01", 0.4),
        ("a2", "2020-01-02", 0.5),
        ("a3", "2020-01-03", 0.6),
    ]:
        for policy, value in [("s10_r05", 0.3), ("s05_r10", candidate_value)]:
            rows.append(
                {
                    "epidemic_model": "temporal_sir",
                    "recognition_sensitivity": 0.5,
                    "system_family": "family_a",
                    "dataset_id": "dataset_a",
                    "network_id": "network_a",
                    "anchor_id": anchor_id,
                    "anchor_time": pd.Timestamp(time),
                    "policy": policy,
                    "final_attack_rate": value,
                }
            )
    candidates = pd.DataFrame(
        {
            "epidemic_model": ["temporal_sir"],
            "recognition_sensitivity": [0.5],
            "cost_ratio": [1.0],
            "system_family": ["family_a"],
            "zero_shot_policy": ["s05_r10"],
            "training_families": ["family_b|family_c"],
        }
    )
    result = build_forward_evaluations(pd.DataFrame(rows), candidates, [2], "s10_r05")
    probe = result.loc[result["selector"].eq("local_probe_veto")].iloc[0]
    assert probe["anchor_id"] == "a3"
    assert probe["latest_training_time"] < probe["anchor_time"]
    assert probe["selected_policy"] == "s10_r05"
    assert probe["value"] == 0
