# Temporal Animal Intervention Evaluation

This repository accompanies the study **Prospective evaluation of individual and joint epidemic interventions across temporal animal networks**. It provides a common event-level interface for heterogeneous animal interaction records, paired temporal epidemic simulations, prospective intervention-value labels, cross-system evaluation, and publication figure generation.

The analysis asks three connected questions.

1. Can contact history identify animals with high marginal intervention value in a later outbreak?
2. Does an accurate individual ranking remain effective when converted into a fixed-budget intervention set?
3. How do a detected case and the timing of action change the value of historical contacts?

The repository evaluates these questions across empirical temporal systems including feeder associations, proximity sensing, and observed group membership. Dataset-specific adapters preserve the meaning and time resolution of each observation process. Policy comparisons use paired epidemic worlds with the same future contacts, index case, and keyed random draws.

## Repository structure

```text
configs/                 Frozen experiment configurations
data/                    Source records, checksums, and local processed tables
docs/                    Dataset-semantics documentation
paper/                   Publication figures and quantitative supplementary table
results/                 Frozen experiment outputs in the Zenodo archive
src/animal_intervention/ Analysis and simulation package
tests/                   Unit, contract, provenance, and experiment audits
```

Raw third-party archives are not redistributed. `DATA_SOURCES.md` lists the permanent source, release or access version, licence, and checksum for every input. The Zenodo reviewer archive includes the complete canonical processed-data release. Oxford derivatives retain the Network Repository Creative Commons Attribution-ShareAlike terms and include source attribution, a modification statement, and processed-file checksums.

## Installation

Python 3.11 or later is required. The archived analysis used Python 3.13.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-build-isolation
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. On macOS or Linux, use `source .venv/bin/activate`.

## Quick test

The data-independent contract suite runs in continuous integration and completes in a few minutes.

```bash
python -m pytest -q \
  tests/test_contract_validation.py \
  tests/test_mappers.py \
  tests/test_network_views.py \
  tests/test_temporal_sir.py \
  tests/test_seir_simulation.py \
  tests/test_paired_simulation.py \
  tests/test_intervention_value.py \
  tests/test_label_contract.py
```

With the processed-data tables from the Zenodo archive present, run the complete verification suite. Raw-adapter tests are reported as skipped until the corresponding third-party payloads have been downloaded and verified.

```bash
python -m animal_intervention.data.release --verify
python -m pytest -q
```

The frozen reviewer release reports `Verified 9 canonical datasets`, followed by 211 passing tests and nine skipped raw-adapter tests. The nine adapters can be exercised after downloading and checksum-verifying the source archives listed in `DATA_SOURCES.md`.

## Full reproduction

The full manuscript-facing evidence rebuild starts from the archived processed data and frozen precision-audited intermediate results. It reruns every analytical stage used in the manuscript and compares generated tables and figures with the archived outputs.

```bash
python -m animal_intervention.experiments.submission_rebuild --profile smoke
python -m animal_intervention.experiments.submission_rebuild --profile full
```

The smoke profile verifies orchestration and artifact contracts from the reviewer archive. The full rebuild is CPU intensive and can take several hours. Every experiment writes a resolved configuration, input and source hashes, checkpoint metadata, result tables, and an audit record. `REPRODUCIBILITY.md` describes the verification layers and expected outputs.

## Generate publication figures

The six main figures and supplementary figures are generated from the frozen family-level result tables.

```bash
python -m animal_intervention.experiments.publication_figures
python -m animal_intervention.experiments.sensitivity_evidence_synthesis \
  --config configs/EXP-20260820-005_sensitivity_evidence_synthesis.yaml
```

Figures are written to `paper/figures/`. In the reviewer archive, the manuscript and electronic supplementary material are also stored under `paper/`.

## Authors

- Zi Zhu, Harvard T.H. Chan School of Public Health, Harvard University. Corresponding author: zizhu@hsph.harvard.edu
- Ruo Yan Hou, Boston University Aram V. Chobanian & Edward Avedisian School of Medicine. sunnyhou@bu.edu

Zi Zhu led the study design and all computational work, including data engineering, formal analysis, methodology, software, simulation, validation, visualisation, and manuscript drafting. Ruo Yan Hou contributed biological-domain expertise, identification and assessment of animal-interaction datasets, validation of biological interpretations, supporting literature research, and manuscript review and editing.

## Licensing

- Original software source and test code are released under the [MIT License](LICENSE).
- Author-owned portions of derived datasets, result tables, repository figures, and documentation are released under [CC BY 4.0](LICENSE-DATA-DOCS).
- Third-party source data retain their original licenses and are not relicensed. `DATA_SOURCES.md` identifies the governing source terms for every dataset. Those terms continue to apply to any derivative that incorporates third-party material.
- Processed Oxford derivatives are distributed under the Network Repository Creative Commons Attribution-ShareAlike policy. Their dataset directory contains the required attribution and transformation notice.

## Citation

Use the metadata in `CITATION.cff`. Dataset-specific citations remain required and are listed in `DATA_SOURCES.md`.
