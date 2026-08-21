# Reproducibility guide

## Verification layers

The release separates three forms of verification.

1. Contract tests validate schemas, temporal ordering, transmission mapping, paired randomness, and intervention estimands without third-party data.
2. The full test suite validates processed data, frozen experiment artifacts, and publication-facing summaries.
3. The submission rebuild reruns the manuscript evidence chain in a separate output tree and compares tables and figures with frozen outputs.

## Environment

The reference environment is recorded in `requirements-lock.txt`. Install the locked dependencies and the local package from the repository root.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-build-isolation
python -m pip check
```

## Data

The Zenodo archive contains canonical processed tables for redistributable sources. Raw third-party payloads are excluded. Download any missing source archives using `DATA_SOURCES.md`, verify their checksums, place them in `data/<dataset_id>/raw/`, and run the corresponding adapter.

Verify the canonical data boundary with:

```bash
python -m animal_intervention.data.release --verify
```

## Tests

Quick data-independent verification:

```bash
python -m pytest -q tests/test_contract_validation.py tests/test_mappers.py tests/test_network_views.py tests/test_temporal_sir.py tests/test_seir_simulation.py tests/test_paired_simulation.py tests/test_intervention_value.py tests/test_label_contract.py
```

Complete verification with processed data and frozen results:

```bash
python -m pytest -q
```

Tests that reopen raw third-party archives are skipped when those payloads are absent. After downloading and checksum-verifying the sources listed in `DATA_SOURCES.md`, the same command exercises every adapter.

## Manuscript evidence rebuild

```bash
python -m animal_intervention.experiments.submission_rebuild --profile smoke
python -m animal_intervention.experiments.submission_rebuild --profile full
```

The smoke profile checks orchestration and artifact contracts. The full profile rebuilds the manuscript-facing analytical chain and verifies tabular and image outputs against the frozen release.

## Figure generation

```bash
python -m animal_intervention.experiments.publication_figures
python -m animal_intervention.experiments.sensitivity_evidence_synthesis --config configs/EXP-20260820-005_sensitivity_evidence_synthesis.yaml
```

## Provenance

Formal experiment directories contain `resolved_config.yaml`, `run_manifest.json`, source and input hashes, result tables, and audit files. The Zenodo ZIP also contains `RELEASE_MANIFEST.json`, which records the byte count and SHA-256 digest of every included file.
