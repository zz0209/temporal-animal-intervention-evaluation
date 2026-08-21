# Radolfzell Great Tits — Ontogeny Study

- Dataset DOI: https://doi.org/10.5061/dryad.x95x69ps8
- Associated study: https://doi.org/10.1093/beheco/arae011
- Expected useful files: seasonal RFID records and published GMM event objects
- Edge semantics: RFID-derived co-foraging/co-flocking association
- Download status (2026-08-13): complete; full Dryad archive, metadata and 37-file manifest are present in `raw/`
- Payload check: archive contains 37/37 expected files; every filename and uncompressed size matches the official manifest
- Reuse terms: Dryad CC0 data policy; cite both the dataset DOI and associated study

## Canonical primary stream

- Adapter version: `0.2.0`.
- The primary summer stream concatenates the 14 non-overlapping weekly files
  `gmm.summer.w1.RData` through `gmm.summer.w14.RData`. The generic
  `gmm.summer.RData` spans only a later six-week interval and
  `gmm.summer.w1_3.RData` overlaps the weekly files, so neither is added to the
  14-week series.
- Autumn, winter and spring each use their complete three-period seasonal GMM
  file. Sampling gaps are unobserved time, not zero-contact epidemic time.
- The rebuilt canonical dataset contains 2,266 registered individuals, 6,551
  inferred group events and 25,963 memberships. Of the group events, 120 have
  zero duration and remain in provenance but are excluded from transmission;
  6,431 have positive duration and 4,458 include at least two tagged great tits.
- There are 306 observed analysis identities. Of these, 199 match the supplied
  roster metadata and 107 are valid identities in the author-provided GMM
  objects without matching age/sex metadata. They remain eligible for network
  and intervention-label analysis but cannot support demographic feature claims.
- Co-membership in an inferred flock is an association proxy, not evidence of
  direct physical contact.

## Primary validation experiment

`EXP-20260815-011` uses two preceding observed 48-hour sampling periods for
history eligibility and the next observed period for offline epidemic replay.
The 15 anchors comprise 12 summer windows and one window in each of autumn,
winter and spring. All seasons belong to one longitudinal population and must
not be treated as four independent datasets.

`raw/` is immutable source material. Derived files belong in `interim/` or `processed/`.
