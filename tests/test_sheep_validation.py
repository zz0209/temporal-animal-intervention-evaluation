from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from animal_intervention.experiments.sheep_validation import (
    _load_phase_by_date,
    _phase_by_week_start,
)


def test_sheep_phase_mapping_uses_publication_defined_weeks(tmp_path: Path) -> None:
    rows = []
    expected = {
        1: "Pre-parasite",
        2: "Pre-patent",
        3: "Pre-patent",
        4: "Pre-patent",
        5: "Patent-parasite",
        6: "Patent-parasite",
        7: "Patent-parasite",
        8: "Post-parasite",
        9: "Post-parasite",
    }
    start = pd.Timestamp("2019-06-03")
    for week, phase in expected.items():
        rows.append(
            {
                "Date": (start + pd.Timedelta(weeks=week - 1)).strftime("%d/%m/%Y"),
                "Week": week,
                "Phase": phase,
            }
        )
    rows.append({"Date": "25/06/2019", "Week": 4, "Phase": "Pre-patent"})
    path = tmp_path / "measurements.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    mapped = _load_phase_by_date(path)
    frozen = _phase_by_week_start("2019-06-03")

    assert mapped == frozen
    assert mapped[pd.Timestamp("2019-06-17").date()] == "Pre-patent"
    assert mapped[pd.Timestamp("2019-07-01").date()] == "Patent-parasite"
    assert mapped[pd.Timestamp("2019-07-22").date()] == "Post-parasite"
    assert len(mapped) == 63


def test_sheep_phase_mapping_rejects_inconsistent_source_phase(tmp_path: Path) -> None:
    rows = []
    start = pd.Timestamp("2019-06-03")
    for week in range(1, 10):
        rows.append(
            {
                "Date": (start + pd.Timedelta(weeks=week - 1)).strftime("%d/%m/%Y"),
                "Week": week,
                "Phase": "incorrect",
            }
        )
    path = tmp_path / "measurements.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="publication-defined"):
        _load_phase_by_date(path)
