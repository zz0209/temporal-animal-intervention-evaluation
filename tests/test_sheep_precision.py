from __future__ import annotations

import pandas as pd

from animal_intervention.experiments.sheep_precision import precision_curve
from animal_intervention.experiments.stability_parallel import WORLD_COLUMNS


def test_precision_curve_uses_nested_random_block_prefixes() -> None:
    rows = []
    for block_id in range(4):
        for candidate_index, candidate in enumerate(["A", "B", "C"]):
            for stratum, initial in [("non_index", "Z"), ("self_index", candidate)]:
                rows.append(
                    {
                        "task_id": f"t{block_id}",
                        "anchor_id": "anchor_001",
                        "parameter_id": "parameter_001",
                        "block_id": block_id,
                        "candidate_id": candidate,
                        "introduction_stratum": stratum,
                        "introduction_position": 0 if stratum == "non_index" else -1,
                        "introduction_replicate": block_id,
                        "initial_infected": initial,
                        "world_seed": block_id,
                        "population_size": 3,
                        "baseline_final_size": 3,
                        "intervention_final_size": candidate_index + 1,
                        "avoided_infections": 2 - candidate_index,
                    }
                )
    worlds = pd.DataFrame(rows, columns=WORLD_COLUMNS)
    curve, labels = precision_curve(worlds, [2, 4], top_k=1)
    assert set(curve["random_blocks"]) == {2, 4}
    assert set(labels) == {2, 4}
    assert curve.loc[curve["random_blocks"].eq(4), "rank_correlation_to_maximum_level"].eq(1.0).all()
