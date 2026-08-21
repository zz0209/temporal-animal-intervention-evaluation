# Focused domain and data-semantics audit

Audit date: 2026-08-19

## Purpose and scope

This is a focused, literature-driven audit of the six approved animal datasets and
the transmission mappings used by Temporal Animal Intervention Evaluation. It is not a claim of field validation and
not a systematic review. Primary dataset papers, deposited metadata, measurement-
validation studies and animal-network epidemiology methods were prioritized. The
audit asks a narrow question: what biological process can each observed edge support,
and which paper claims remain defensible without pathogen-specific calibration?

## Evidence-backed interpretation contract

| Dataset | What was observed | Defensible model interpretation | Claims that are not allowed |
|---|---|---|---|
| Oxford wild birds | Repeated feeder co-attendance summarized at day scale | Association-mediated exposure opportunity over a relatively coarse timescale | Exact physical contact, exact infection probability, or a short-timescale airborne/contact pathogen |
| Guinea baboons | Approximately 20-second wearable-sensor proximity plus a separate direct-observation modality | High-resolution face-to-face proximity opportunity within the tagged group; sensor and observed-behavior modalities remain separate | A named behavior, exact distance, complete observation of untagged animals, or pathogen-specific transmission probability |
| Domestic sheep | Binary Sirtrack proximity-logger encounter intervals in twelve separate five-animal groups | Proximity-mediated opportunity for a hypothetical directly transmitted pathogen; parasite treatment is a behavioral context | Transmission of *Teladorsagia circumcincta* along these dyadic edges; that parasite is pasture-mediated and the experiment did not observe its transmission network |
| Wytham great tits | Feeder visits clustered into flock events | Co-flocking/group exposure opportunity at feeders | Simultaneous physical contact between every pair or independent full-strength pairwise contacts |
| Radolfzell great tits | RFID feeder detections clustered into flock events | Co-flocking/group exposure opportunity during sampled 48-hour periods | Complete continuous-life contact or full-strength clique transmission |
| Experimental songbirds | Feeder co-flocking under an imposed social-foraging manipulation | Co-flocking/group exposure and information-pathway opportunity | A pathogen-specific physical-contact network |

## Mapping decision

Temporal Animal Intervention Evaluation keeps the measurement modalities distinct at ingestion and unifies them only
at the exposure-stream interface used by the simulator. Duration encounters retain
duration, fixed-bin detections retain their observation interval, association indices
become integrated association exposure, and flock observations remain group events.

For group events, the primary mapping is frequency-dependent mixing: with one
infectious member, its total event-level transmission pressure is approximately held
constant as observed group size increases. This avoids turning a large inferred flock
into many independent full-strength pairwise contacts. The alternative
`undiluted_clique` mapping is scientifically possible for some highly transmissible
shared-air or shared-environment mechanisms, but it changes total force of infection
with group size. It is therefore a sensitivity model, not a replacement primary.

The comparison must recalibrate the transmission coefficient separately under each
mapping to comparable baseline epidemic severity. Otherwise a mapping with more total
hazard would win mechanically and the experiment would confound semantics with dose.

## Consequences for the paper

The paper may claim model-based preparedness and response value on heterogeneous
animal association/proximity systems. It may not claim measured field efficacy,
identify a named pathogen without additional calibration, or describe every dataset
as physical contact. The common output is deliberately an intervention estimand, not
a universal beta: avoided attack rate under a declared observation-to-exposure
contract.

Two experiments directly followed from this audit and are now complete:

1. Estimate the distribution of equal-capacity random response lists, so that the
   incremental value and percentile of history-ranked targeting are not determined by
   a single random comparator.
2. On the three flock-event datasets, compare frequency-dependent and undiluted-clique
   group mappings after matching baseline epidemic severity, and test whether the main
   preparedness conclusion changes.

`EXP-20260819-007` replaced the single random response list with eight independently
keyed, equal-capacity lists per epidemic world. The resulting history-targeting
increment remained positive on average but did not clear the family-level interval
gate in either epidemic model; SEIR/Erlang was positive in only two of five independent
families. `EXP-20260819-008` then compared the primary frequency-dependent flock
mapping with a hazard-normalized undiluted clique. Capacity and targeting directions
were qualitatively similar, but targeting magnitudes attenuated under the alternative.
Because these three datasets represent only two independent bird-system families,
this is a semantic sensitivity result rather than universal validation.

The audit therefore changes the paper's center of gravity. Historical interaction
structure can predict relative singleton counterfactual value, but a scalar ranking
does not automatically yield a transportable fixed-budget set policy. Response
capacity is more reproducible across systems than fine target ordering. This is a
model-based operational boundary, not evidence that contact history is biologically
irrelevant.

## Sources

Permanent dataset records, licences, acquisition versions, checksums, and associated
publications are listed in `DATA_SOURCES.md`. The article and electronic supplement
provide the methodological references for animal-network construction, proximity
logger calibration, social-versus-transmission network distinctions, and
transmission-hazard scaling.
