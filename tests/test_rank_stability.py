from __future__ import annotations

import pandas as pd

from animal_intervention.evaluation import pairwise_rank_stability, stable_hash_order
from animal_intervention.experiments.oxford_predefense import _summaries


def test_stable_hash_order_is_reproducible_and_keyed() -> None:
    values = ["a", "b", "c", "d"]
    assert stable_hash_order(values, 7, "block") == stable_hash_order(
        reversed(values), 7, "block"
    )
    assert stable_hash_order(values, 7, "block") != stable_hash_order(
        values, 8, "block"
    )


def test_pairwise_rank_stability_reports_rank_and_top_k_agreement() -> None:
    frame = pd.DataFrame(
        {
            "context": ["a"] * 4 + ["b"] * 4,
            "node": ["1", "2", "3", "4"] * 2,
            "value": [4, 3, 2, 1, 8, 6, 1, 0],
        }
    )
    result = pairwise_rank_stability(
        frame,
        context_columns=["context"],
        item_column="node",
        value_column="value",
        top_k=2,
    )
    assert len(result) == 1
    assert result.iloc[0]["spearman"] == 1.0
    assert result.iloc[0]["top_k_overlap_fraction"] == 1.0
    assert result.iloc[0]["top_k_jaccard"] == 1.0
    assert result.iloc[0]["mean_top_k_value_retention"] == 1.0
    assert result.iloc[0]["mean_absolute_top_k_regret"] == 0.0


def test_pairwise_rank_stability_reports_value_retention_when_names_change() -> None:
    frame = pd.DataFrame(
        {
            "context": ["a"] * 3 + ["b"] * 3,
            "node": ["1", "2", "3"] * 2,
            "value": [3.0, 2.0, 1.0, 1.0, 3.0, 2.0],
        }
    )
    result = pairwise_rank_stability(
        frame,
        context_columns=["context"],
        item_column="node",
        value_column="value",
        top_k=1,
    )
    assert result.iloc[0]["top_k_overlap_fraction"] == 0.0
    assert result.iloc[0]["mean_top_k_value_retention"] == 0.5
    assert result.iloc[0]["mean_absolute_top_k_regret"] == 1.5


def test_robust_anchor_labels_average_scenarios_but_not_anchors() -> None:
    rows = []
    for anchor, offset in [("anchor_001", 0.0), ("anchor_002", 10.0)]:
        for parameter, parameter_value in [("p1", 1.0), ("p2", 3.0)]:
            for block in [0, 1]:
                for candidate, candidate_value in [("a", 1.0), ("b", 0.0)]:
                    rows.append(
                        {
                            "anchor_id": anchor,
                            "parameter_id": parameter,
                            "block_id": block,
                            "candidate_id": candidate,
                            "unconditional_value": (
                                offset + parameter_value + candidate_value
                            ),
                        }
                    )
    summaries = _summaries(pd.DataFrame(rows), top_k=1, minimum_contexts=1)
    labels = summaries["robust_anchor_labels"].set_index(
        ["anchor_id", "candidate_id"]
    )
    assert labels.loc[("anchor_001", "a"), "robust_intervention_value"] == 3.0
    assert labels.loc[("anchor_002", "a"), "robust_intervention_value"] == 13.0
    assert labels.loc[("anchor_001", "a"), "parameter_contexts"] == 2
