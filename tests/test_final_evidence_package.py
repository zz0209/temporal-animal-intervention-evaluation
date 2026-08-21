from pathlib import Path

import pandas as pd

from animal_intervention.experiments.final_evidence_package import (
    load_source_audits,
    reconcile_primary_tables,
)


def test_load_source_audits_preserves_status_and_hash(tmp_path: Path) -> None:
    audit = tmp_path / "EXP-X" / "full" / "audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text('{"status": "pass", "checks": {"a": true}}', encoding="utf-8")
    result = load_source_audits([audit])
    assert result.loc[0, "status"] == "pass"
    assert result.loc[0, "checks"] == 1
    assert len(result.loc[0, "sha256"]) == 64


def test_primary_table_reconciliation_detects_decision_change() -> None:
    keys = {"epidemic_model": "sir", "detection_profile": "early", "rewiring_fraction": 0.0}
    decision = pd.DataFrame([{**keys, "decision": "retain"}])
    taxonomy = pd.DataFrame([{**keys, "decision": "retain"}])
    resilience = pd.DataFrame([keys])
    assert reconcile_primary_tables(decision, taxonomy, resilience) is False
    resilience = pd.concat([resilience] * 8, ignore_index=True)
    assert reconcile_primary_tables(decision, taxonomy, resilience) is True
    taxonomy.loc[0, "decision"] = "abstain"
    assert reconcile_primary_tables(decision, taxonomy, resilience) is False
