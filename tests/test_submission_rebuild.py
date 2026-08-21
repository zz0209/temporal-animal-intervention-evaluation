from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from animal_intervention.experiments.submission_rebuild import (
    _compare_frame,
    _compare_frame_semantic,
    _compare_image,
    _compare_scientific_subset,
    _rewrite_value,
    _materialize_static_inputs,
)


def test_rewrite_value_redirects_only_selected_experiments() -> None:
    selected = {"EXP-A"}
    clean = Path("reproducibility/run")
    assert _rewrite_value("results/EXP-A/full/a.csv", selected, clean, {}, "smoke") == "results/EXP-A/full/a.csv"
    assert _rewrite_value("results/EXP-A/full/a.csv", selected, clean, {}, "full") == str(
        clean / "results/EXP-A/full/a.csv"
    ).replace("\\", "/")
    assert _rewrite_value("results/EXP-B/full/a.csv", selected, clean, {}, "smoke") == "results/EXP-B/full/a.csv"


def test_compare_frame_accepts_small_numeric_roundoff(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"key": ["a"], "value": [1.0]}).to_csv(left, index=False)
    pd.DataFrame({"key": ["a"], "value": [1.0 + 1e-12]}).to_csv(right, index=False)
    assert _compare_frame(left, right, atol=1e-10, rtol=1e-10) is None


def test_semantic_frame_comparison_marks_additive_columns(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"key": ["a"], "value": [1.0]}).to_csv(left, index=False)
    pd.DataFrame({"value": [1.0], "key": ["a"], "diagnostic": [2.0]}).to_csv(
        right, index=False
    )
    status, detail = _compare_frame_semantic(left, right, 1e-10, 1e-10)
    assert status == "schema_extension"
    assert "diagnostic" in detail


def test_semantic_frame_comparison_reconciles_time_format_and_declared_metadata(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame(
        {"anchor_time": ["2014-01-16 00:00:00"], "sensitivity": [np.nan]}
    ).to_csv(left, index=False)
    pd.DataFrame({"anchor_time": ["2014-01-16"], "sensitivity": [0.5]}).to_csv(
        right, index=False
    )
    status, detail = _compare_frame_semantic(
        left,
        right,
        1e-10,
        1e-10,
        metadata_fill_columns={"sensitivity"},
    )
    assert status == "metadata_enrichment"
    assert "sensitivity" in detail


def test_scientific_subset_uses_only_declared_diagnostic_tolerance() -> None:
    reference = {"effect": 0.1, "rank_diagnostic": -0.279}
    rebuilt = {"effect": 0.1, "rank_diagnostic": -0.282, "new_check": True}
    assert (
        _compare_scientific_subset(
            reference,
            rebuilt,
            atol=1e-10,
            rtol=1e-10,
            diagnostic_tolerances={"scientific_result.rank_diagnostic": 0.005},
        )
        is None
    )
    assert _compare_scientific_subset(
        reference,
        rebuilt,
        atol=1e-10,
        rtol=1e-10,
        diagnostic_tolerances={},
    ) is not None


def test_submission_rebuild_config_exists() -> None:
    assert Path("configs/EXP-20260818-002_submission_rebuild.yaml").exists()


def test_compare_image_detects_pixel_change(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    pixels = np.zeros((120, 120, 3), dtype=np.uint8)
    pixels[:, 60:, :] = 255
    Image.fromarray(pixels).save(left)
    pixels[0, 0, 0] = 255
    Image.fromarray(pixels).save(right)
    difference = _compare_image(left, right)
    assert difference is not None
    assert "pixel mismatch" in difference


def test_materialize_static_inputs_verifies_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("source.csv")
    source.write_text("a\n1\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    clean_root = tmp_path / "clean"
    config = {"design": {"frozen_static_inputs": [{"path": str(source), "sha256": digest}]}}
    assert _materialize_static_inputs(config, clean_root) == 1
    assert (clean_root / source).read_bytes() == source.read_bytes()
