import pandas as pd

from animal_intervention.experiments.sensitivity_evidence_synthesis import summarize_family_effects


def test_summarize_family_effects_preserves_family_grain() -> None:
    frame = pd.DataFrame(
        {
            "domain": ["delivery"] * 3,
            "estimand": ["policy_value"] * 3,
            "scenario": ["reference"] * 3,
            "system_family": ["a", "b", "c"],
            "effect": [0.2, -0.1, 0.0],
            "source_artifact": ["source.csv"] * 3,
        }
    )

    summary = summarize_family_effects(frame).iloc[0]

    assert summary["families"] == 3
    assert summary["family_equal_mean"] == pd.Series([0.2, -0.1, 0.0]).mean()
    assert summary["family_min"] == -0.1
    assert summary["family_max"] == 0.2
    assert summary["positive_families"] == 1
    assert summary["zero_families"] == 1
    assert summary["negative_families"] == 1
