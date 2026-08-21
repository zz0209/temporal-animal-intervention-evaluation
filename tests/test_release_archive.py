from __future__ import annotations

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
    for directory in (".github", "configs", "data", "docs", "paper", "tests"):
        (root / directory).mkdir(parents=True, exist_ok=True)

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
    assert "results/experiment/models/fitted.pt" not in released
