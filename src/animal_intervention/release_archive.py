from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


RELEASE_ID = "temporal-animal-intervention-evaluation-reviewer-r1"
REDISTRIBUTABLE_PROCESSED_DATA = {
    "barn_swallows_encounternet",
    "domestic_sheep_sirtrack",
    "experimental_wild_songbirds",
    "free_ranging_sheep_fission_fusion",
    "guinea_baboons_sociopatterns",
    "radolfzell_great_tits_ontogeny",
    "wild_vampire_bats_proximity",
    "wytham_great_tits_divorce",
}
ROOT_FILES = {
    ".gitignore",
    "CITATION.cff",
    "DATA_SOURCES.md",
    "LICENSE",
    "LICENSE-DATA-DOCS",
    "README.md",
    "REPRODUCIBILITY.md",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements.txt",
}
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "dist",
}
EXCLUDED_DATA_DIRECTORY_NAMES = {"interim", "raw"}
EXCLUDED_RESULT_DIRECTORY_NAMES = {"checkpoints", "models", "reproducibility"}
EXCLUDED_SUFFIXES = {".ckpt", ".onnx", ".pth", ".pt", ".pyc", ".tmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def include_path(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts):
        return False
    if parts[0] == "data" and any(
        part in EXCLUDED_DATA_DIRECTORY_NAMES for part in parts[1:]
    ):
        return False
    if parts[0] == "results" and any(
        part in EXCLUDED_RESULT_DIRECTORY_NAMES for part in parts[1:]
    ):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if parts[0] == "data" and "processed" in parts:
        return len(parts) > 1 and parts[1] in REDISTRIBUTABLE_PROCESSED_DATA
    return True


def collect_files(root: Path) -> list[Path]:
    files: set[Path] = {root / name for name in ROOT_FILES}
    for directory in (".github", "configs", "data", "docs", "paper", "results", "src", "tests"):
        base = root / directory
        files.update(
            path for path in base.rglob("*") if path.is_file() and include_path(root, path)
        )
    missing = sorted(path for path in files if not path.exists())
    if missing:
        rendered = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Release inputs are missing:\n{rendered}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_archive(root: Path, output_directory: Path) -> dict[str, object]:
    files = collect_files(root)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / f"{RELEASE_ID}.zip"
    manifest_path = output_directory / f"{RELEASE_ID}-manifest.json"
    checksum_path = output_directory / f"{RELEASE_ID}.zip.sha256"
    file_records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "release_id": RELEASE_ID,
        "base_git_commit": git_head(root),
        "canonical_data_release": "canonical-20260820-r01",
        "raw_third_party_payloads_included": False,
        "processed_data_redistribution": sorted(REDISTRIBUTABLE_PROCESSED_DATA),
        "excluded_processed_dataset": "oxford_wildbird_network",
        "file_count": len(file_records),
        "files": file_records,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 8, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
        manifest_info = zipfile.ZipInfo(
            "RELEASE_MANIFEST.json", date_time=(2026, 8, 20, 0, 0, 0)
        )
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o644 << 16
        archive.writestr(manifest_info, manifest_text.encode("utf-8"))
    archive_hash = sha256(archive_path)
    checksum_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")
    return {
        "release_id": RELEASE_ID,
        "archive": archive_path.as_posix(),
        "archive_sha256": archive_hash,
        "manifest": manifest_path.as_posix(),
        "file_count": len(file_records),
        "archive_bytes": archive_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic reviewer archive.")
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(build_archive(root, args.output), indent=2))


if __name__ == "__main__":
    main()
