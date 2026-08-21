# Domestic Sheep — Sirtrack Proximity Loggers

- Dataset DOI: https://doi.org/10.5061/dryad.vhhmgqp15
- Preservation mirror: https://doi.org/10.5281/zenodo.10048889
- Expected key file: `Behaviour_data_Amorris_2023.csv`
- Edge semantics: proximity-logger-recorded contact
- Important check: verify the finest actual temporal grain before classifying this as an event stream
- Download status (2026-08-13): complete; the three deposited files were downloaded from the linked Zenodo preservation record because Dryad's file endpoint currently requires authentication
- Integrity check: local SHA-256 values exactly match the SHA-256 digests published by Dryad
- Reuse terms: the canonical Dryad dataset is released under Dryad's CC0 data policy; scholarly citation of the dataset DOI and associated article is retained. The Zenodo mirror is a download route and does not replace the canonical Dryad citation.

`raw/` is immutable source material. Derived files belong in `interim/` or `processed/`.

For the primary intervention-label experiment, the 60 lambs are treated as 12
independent five-animal social networks because all observed exposures are
within the deposited group assignments. Reciprocal or overlapping logger
intervals for the same dyad are unioned before transmission mapping to avoid
double-counting contact time. The original and coalesced counts remain in the
experiment audit under `EXP-20260815-002`.

The source experiment contains four replicate groups in each of three treatment
classes: fully parasitised, fully non-parasitised (`Control` in the deposited
table), and mixed (`Partially.parasitised` in the deposited table). Each mixed
group contains two experimentally infected and three water-control lambs. Study
phase metadata follows the associated publication and the animal measurement
table: week 1 pre-parasite, weeks 2-4 pre-patent, weeks 5-7 patent-parasite, and
weeks 8-9 post-parasite. The legacy `Phase` field in the behavior table is not
used because its labels do not match this four-phase definition.

The experimental nematode is not treated as the pathogen simulated by Temporal Animal Intervention Evaluation.
Its infection status and phase are observed behavioral context only; Temporal Animal Intervention Evaluation's
temporal SIR counterfactual represents a hypothetical contact-borne pathogen.
