from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.safe_transfer_allocation import _select, select_allocations


def test_reference_anchored_maximin_rejects_training_harm() -> None:
    scores = pd.DataFrame(
        {
            "policy": ["reference", "risky", "safe"],
            "mean_improvement": [0.0, 0.2, 0.1],
            "minimum_improvement": [0.0, -0.4, 0.05],
            "maximum_regret": [0.2, 0.4, 0.1],
            "nominal_cost": [0.15, 0.15, 0.15],
        }
    )
    assert _select(scores, "pooled_mean")["policy"] == "risky"
    assert _select(scores, "maximin_reference_anchored")["policy"] == "safe"


def test_complete_family_is_excluded_from_selection() -> None:
    rows = []
    values = {
        "family_a": {"s10_r05": 0.30, "s05_r10": 0.20},
        "family_b": {"s10_r05": 0.30, "s05_r10": 0.20},
        "family_c": {"s10_r05": 0.20, "s05_r10": 0.60},
    }
    metadata = {"s10_r05": (0.10, 0.05), "s05_r10": (0.05, 0.10)}
    for family, policies in values.items():
        for policy, mean_value in policies.items():
            sentinel, response = metadata[policy]
            rows.append(
                {
                    "epidemic_model": "temporal_sir",
                    "recognition_sensitivity": 0.5,
                    "sentinel_fraction": sentinel,
                    "response_fraction": response,
                    "policy": policy,
                    "system_family": family,
                    "mean_value": mean_value,
                }
            )
    allocations, _ = select_allocations(
        pd.DataFrame(rows),
        ["pooled_mean", "maximin_reference_anchored"],
        [1.0],
        0.10,
        0.05,
        "all",
    )
    heldout_c = allocations.loc[
        allocations["system_family"].eq("family_c")
        & allocations["selector"].eq("pooled_mean")
    ].iloc[0]
    assert heldout_c["selected_policy"] == "s05_r10"
    assert "family_c" not in heldout_c["training_families"].split("|")
    assert heldout_c["value"] < 0
