from __future__ import annotations

import json
import zipfile
from pathlib import Path

from animal_intervention import release_archive


def test_release_collects_runtime_model_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    released = {
        path.relative_to(root).as_posix() for path in release_archive.collect_files(root)
    }

    assert "src/animal_intervention/models/__init__.py" in released
    assert "src/animal_intervention/models/set_value.py" in released
    canonical_release = json.loads(
        (root / "data" / "_shared" / "canonical_release.json").read_text(
            encoding="utf-8"
        )
    )
    assert canonical_release["dataset_count"] == 9
    for dataset in canonical_release["datasets"]:
        dataset_id = dataset["dataset_id"]
        for filename in dataset["processed_files_sha256"]:
            assert f"data/{dataset_id}/processed/{filename}" in released
    assert not any(path.endswith(".pt") for path in released)


def test_archive_keeps_source_models_but_excludes_result_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repository"
    output = tmp_path / "dist"
    for name in release_archive.ROOT_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("release test\n", encoding="utf-8")
    for directory in (".github", "configs", "data", "docs", "paper", "reports", "tests"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    report_reference = root / "reports" / "experiment" / "full" / "figure.png"
    report_reference.parent.mkdir(parents=True)
    report_reference.write_bytes(b"frozen figure reference")

    canonical_release = root / "data" / "_shared" / "canonical_release.json"
    canonical_release.parent.mkdir(parents=True)
    canonical_release.write_text(
        '{"release_id":"canonical-test-r01"}\n', encoding="utf-8"
    )

    oxford_processed = root / "data" / "oxford_wildbird_network" / "processed"
    oxford_processed.mkdir(parents=True)
    (oxford_processed / "dataset_metadata.json").write_text(
        '{"dataset_id":"oxford_wildbird_network"}\n', encoding="utf-8"
    )

    source_models = root / "src" / "animal_intervention" / "models"
    source_models.mkdir(parents=True)
    (source_models / "__init__.py").write_text("", encoding="utf-8")
    (source_models / "set_value.py").write_text("VALUE = 1\n", encoding="utf-8")

    result_models = root / "results" / "experiment" / "models"
    result_models.mkdir(parents=True)
    (result_models / "fitted.pt").write_bytes(b"model artifact")

    monkeypatch.setattr(release_archive, "git_head", lambda _: "a" * 40)
    result = release_archive.build_archive(root, output)

    with zipfile.ZipFile(result["archive"]) as archive:
        released = set(archive.namelist())
    assert "src/animal_intervention/models/__init__.py" in released
    assert "src/animal_intervention/models/set_value.py" in released
    assert "data/oxford_wildbird_network/processed/dataset_metadata.json" in released
    assert "reports/experiment/full/figure.png" in released
    assert "results/experiment/models/fitted.pt" not in released
    assert result["file_count"] == len(released) - 1
