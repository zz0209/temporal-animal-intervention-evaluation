# Experimentally Manipulated Wild Songbirds

- Dataset DOI: https://doi.org/10.5061/dryad.6h4t2
- Associated study: https://doi.org/10.1098/rsbl.2016.0144
- Expected key file: `dryad_data.RData`
- Edge semantics: membership in the same detected social-foraging group
- Download status (2026-08-13): complete; official payload, metadata and file manifest are present in `raw/`
- Payload check: 1,094,512 bytes; MD5 matches Dryad (`cbf0f8fe8d356536d82644e2b19220da`)
- Reuse terms: Dryad CC0 data policy; cite both the dataset DOI and associated study

## Analysis role

- Primary stream: 63,267 positive-duration author-inferred co-flocking events with 237,260 memberships from 375 observed birds across five species.
- Study phases: 40-day pre-manipulation period followed by a 90-day selective-feeder period; phase boundaries are preserved and no analysis window crosses the manipulation boundary.
- Prospective windows: two seven-day history segments followed by one seven-day offline epidemic-replay horizon.
- Population unit: one longitudinal mixed-species Wytham Woods population; the manipulation phase is context, not an independent network.
- Leakage constraint: this dataset and the overlapping `wytham_great_tits_divorce` 2013-14 observation season share part of the feeder observation stream and must remain in the same outer evaluation fold.
- Epidemiological interpretation: co-flocking is an exposure-opportunity proxy. Intervention labels are simulation-derived targets, not observed infections or field causal effects.

`raw/` is immutable source material. Derived files belong in `interim/` or `processed/`.
