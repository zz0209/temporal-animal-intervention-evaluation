# Free-ranging sheep — fission-fusion groups

- Source: https://doi.org/10.5061/dryad.59zw3r2d6
- Associated study: https://doi.org/10.1098/rsos.230402
- Study population: 50 free-ranging female Merino sheep; the deposited partitions use 51 observation identifiers across the full study, with at most 50 present in any continuous recording segment
- Temporal coverage: 210,351 ordered observations at nominal six-second resolution over approximately 15.7 days, with three recorded gaps
- Event semantics: author-inferred spatial groups using the primary 30 m near radius and 50 m sticky radius
- License: CC0
- Download status: complete for the primary group partition and timestamp files on 2026-08-20

The adapter losslessly run-length encodes consecutive identical group partitions. Recording gaps remain empty time and never become exposure. Group co-membership is spatial association, not observed physical contact. `raw/` is immutable source material. Derived files belong in `interim/` or `processed/`.

The manuscript reports 50 animals. Deposited identifier 23 occurs only before the second recording gap and identifier 50 occurs only after it; neither is present in the intervening segment. No undocumented identity merge is imposed. Analyses remain within continuous recording segments, so this source-level identifier change cannot create a cross-gap contact or a duplicated simultaneous animal.
