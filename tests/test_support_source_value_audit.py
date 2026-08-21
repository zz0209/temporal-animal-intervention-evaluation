from pathlib import Path

import pandas as pd

from animal_intervention.experiments.support_source_value_audit import (
    _context_correlations,
    _family_summary,
    _merge_intervals_fraction,
)


def test_context_correlations_keep_non_estimable_contexts() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["oxford_wildbird_network"] * 4,
            "network_id": ["all"] * 4,
            "anchor_time": [pd.Timestamp("2000-01-03", tz="UTC")] * 4,
            "robust_intervention_value": [0.1, 0.2, 0.3, 0.4],
            "source_attack_rate": [0.2, 0.3, 0.4, 0.5],
            "event_rate_per_day": [1, 2, 3, 4],
            "contact_opportunity_rate": [1, 2, 3, 4],
            "eligible_partner_fraction": [0.1, 0.2, 0.3, 0.4],
            "observed_partner_count": [1, 2, 3, 4],
            "recency_score": [1, 1, 1, 1],
            "recent_activity_fraction": [0.1, 0.2, 0.3, 0.4],
            "active_span_fraction": [0.1, 0.2, 0.3, 0.4],
            "first_seen_fraction": [0.1, 0.2, 0.3, 0.4],
            "last_seen_gap_fraction": [0.4, 0.3, 0.2, 0.1],
            "observed_span_fraction": [0.5, 0.5, 0.5, 0.5],
            "roster_coverage_fraction": [1, 1, 1, 1],
        }
    )
    result = _context_correlations(frame, minimum_candidates=5)
    assert len(result) == 1
    assert pd.isna(result.loc[0, "spearman_source_attack_rate"])


def test_family_summary_does_not_average_missing_correlations_as_zero() -> None:
    contexts = pd.DataFrame(
        {
            "system_family": ["family_a", "family_a", "family_b"],
            "spearman_source_attack_rate": [0.5, float("nan"), -0.25],
        }
    )
    result = _family_summary(contexts).set_index("system_family")
    assert result.loc["family_a", "mean_spearman_source_attack_rate"] == 0.5
    assert result.loc["family_a", "estimable_spearman_source_attack_rate"] == 1


def test_submission_config_exists() -> None:
    assert Path("configs/EXP-20260818-001_support_source_value_audit.yaml").exists()


def test_merge_intervals_fraction_uses_union_not_sum() -> None:
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-01-11", tz="UTC")
    result = _merge_intervals_fraction(
        pd.Series([start, start + pd.Timedelta(days=4)]),
        pd.Series([start + pd.Timedelta(days=6), end]),
        start,
        end,
    )
    assert result == 1.0
