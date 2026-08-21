import numpy as np
import pandas as pd

from animal_intervention.experiments.intervention_delivery_sensitivity import (
    _detection_timing_contrasts,
    _operational_isolation_action,
    _paired_increments,
    _select_parameter_regimes,
    _random_block_family_summary,
    _rewiring_mechanism_contrasts,
)


def test_parameter_regimes_use_calibrated_attack_rate_order() -> None:
    parameters = pd.DataFrame(
        {
            "parameter_id": ["p3", "p1", "p4", "p2"],
            "mean_attack_rate": [0.3, 0.1, 0.4, 0.2],
        }
    )
    selected = _select_parameter_regimes(
        list(parameters.itertuples(index=False)), "attack_rate_triplet"
    )
    assert [name for name, _ in selected] == ["low", "middle", "high"]
    assert [item.parameter_id for _, item in selected] == ["p1", "p3", "p4"]


def test_parameter_regimes_reject_incomplete_triplet() -> None:
    parameters = pd.DataFrame(
        {"parameter_id": ["p1", "p2"], "mean_attack_rate": [0.1, 0.2]}
    )
    assert not _select_parameter_regimes(
        list(parameters.itertuples(index=False)), "attack_rate_triplet"
    )


def test_operational_action_preserves_delay_and_partial_contact() -> None:
    start = pd.Timestamp("2020-01-01 02:00:00")
    action = _operational_isolation_action(
        "partial", ("b", "a", "a"), start, start + pd.Timedelta(days=1), 0.25
    )
    assert action.start_time == start
    assert action.target_nodes == ("a", "b")
    assert action.contact_multiplier == 0.25
    assert not action.active_at(start - pd.Timedelta(seconds=1))
    assert action.active_at(start)


def test_operational_action_rejects_invalid_residual_contact() -> None:
    start = pd.Timestamp("2020-01-01")
    try:
        _operational_isolation_action(
            "invalid", ("a",), start, start + pd.Timedelta(days=1), 1.1
        )
    except ValueError as error:
        assert "between zero and one" in str(error)
    else:
        raise AssertionError("invalid residual contact was accepted")


def test_paired_increment_uses_matching_delivery_world() -> None:
    rows = []
    for method, value in [("stable_watchlist", 0.1), ("stable_plus_tracing", 0.15)]:
        rows.append(
            {
                "dataset_id": "d",
                "network_id": "n",
                "anchor_id": "a",
                "parameter_id": "p",
                "detection_profile": "early",
                "action_delay_fraction": 0.1,
                "residual_contact_multiplier": 0.25,
                "secondary_case_sensitivity": 0.5,
                "false_positive_rate": 0.02,
                "rewiring_fraction": 0.5,
                "rewiring_mode": "uniform_partner_substitution",
                "random_block": 0,
                "initial_infected": "x",
                "world_seed": 7,
                "system_family": "f",
                "analysis_cluster_id": "c",
                "method": method,
                "attack_rate_reduction": value,
            }
        )
    paired = _paired_increments(pd.DataFrame(rows), "stable_watchlist")
    assert len(paired) == 1
    assert np.isclose(paired.iloc[0]["increment"], 0.05)


def test_detection_timing_contrast_pairs_the_same_natural_world() -> None:
    rows = []
    for profile, increment in [("early_detection", -0.01), ("delayed_detection", 0.02)]:
        rows.append(
            {
                "dataset_id": "d",
                "network_id": "n",
                "anchor_id": "a",
                "parameter_id": "p",
                "random_block": 0,
                "initial_infected": "x",
                "world_seed": 7,
                "detection_profile": profile,
                "action_delay_fraction": 0.1,
                "residual_contact_multiplier": 0.25,
                "secondary_case_sensitivity": 0.5,
                "false_positive_rate": 0.02,
                "rewiring_fraction": 0.5,
                "rewiring_mode": "uniform_partner_substitution",
                "method": "stable_plus_tracing",
                "system_family": "f",
                "analysis_cluster_id": "c",
                "increment": increment,
            }
        )
    contrast = _detection_timing_contrasts(pd.DataFrame(rows))
    assert len(contrast) == 1
    assert np.isclose(contrast.iloc[0]["timing_contrast"], 0.03)


def test_rewiring_mechanism_contrast_uses_stable_minus_direct_hazard() -> None:
    rows = []
    for method, benefit, removed, rewired in [
        ("stable_watchlist", 0.01, 0.20, 0.10),
        ("contact_to_detected", 0.04, 0.08, 0.04),
    ]:
        rows.append(
            {
                "dataset_id": "d",
                "network_id": "n",
                "anchor_id": "a",
                "parameter_id": "p",
                "detection_profile": "early_detection",
                "action_delay_fraction": 0.1,
                "residual_contact_multiplier": 0.25,
                "secondary_case_sensitivity": 0.5,
                "false_positive_rate": 0.0,
                "rewiring_fraction": 0.5,
                "rewiring_mode": "uniform_partner_substitution",
                "random_block": 0,
                "initial_infected": "x",
                "world_seed": 7,
                "system_family": "f",
                "analysis_cluster_id": "c",
                "method": method,
                "attack_rate_reduction": benefit,
                "additional_removed_hazard_fraction": removed,
                "additional_rewired_hazard_fraction": rewired,
            }
        )
    paired, summary = _rewiring_mechanism_contrasts(pd.DataFrame(rows))
    assert len(paired) == 1
    assert np.isclose(paired.iloc[0]["direct_gain_over_stable"], 0.03)
    assert np.isclose(
        paired.iloc[0]["stable_minus_direct_rewired_hazard_fraction"], 0.06
    )
    assert summary.iloc[0]["stable_rewires_more_families"] == 1


def test_random_block_summary_keeps_family_as_top_level_unit() -> None:
    rows = []
    for block in [0, 1]:
        for family, increments in {"f1": [0.1, 0.3], "f2": [-0.1]}.items():
            for cluster_index, increment in enumerate(increments):
                rows.append(
                    {
                        "detection_profile": "early_detection",
                        "disease_regime": "low",
                        "rewiring_fraction": 0.5,
                        "random_block": block,
                        "system_family": family,
                        "analysis_cluster_id": f"{family}-{cluster_index}",
                        "method": "contact_to_detected",
                        "increment": increment,
                    }
                )
    summary = _random_block_family_summary(
        pd.DataFrame(rows), method="contact_to_detected"
    )
    assert len(summary) == 2
    assert summary["disease_regime"].eq("low").all()
    assert np.allclose(summary["family_equal_increment"], 0.05)
    assert summary["families"].eq(2).all()
