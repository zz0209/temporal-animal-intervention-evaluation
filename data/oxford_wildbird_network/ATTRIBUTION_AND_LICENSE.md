# Oxford wild-bird processed-data attribution and licence

## Source

The processed tables in this directory derive from `aves-wildbird-network`, downloaded from the [Network Repository dataset page](https://networkrepository.com/aves-wildbird-network.php) on 13 August 2026. The original ZIP had SHA-256 checksum `a1b8cdf1d73cb75413a1a2893bcf6f59f91f31148453b394057eb6376af3594a`.

## Governing terms

Network Repository states that all repository data are licensed under a [Creative Commons Attribution-ShareAlike License](https://networkrepository.com/policy.php). Its policy page does not state a licence version number. These processed derivatives are redistributed under those same Attribution-ShareAlike terms. The project-level CC BY 4.0 notice does not apply to these files.

## Required attribution

- Rossi RA, Ahmed NK. The Network Data Repository with Interactive Graph Analytics and Visualization. Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015:4292-4293. https://networkrepository.com/pubs/aaai15-nr.pdf
- Network Repository. `aves-wildbird-network` [dataset]. https://networkrepository.com/aves-wildbird-network.php
- Firth JA, Sheldon BC. Experimental manipulation of avian social structure reveals segregation is carried over across contexts. Proceedings of the Royal Society B. 2015;282:20142350. https://doi.org/10.1098/rspb.2014.2350

## Modifications

The source edge list was transformed into the project's canonical event contract. Animal identifiers were retained as strings and undirected pairs were placed in a deterministic order. Each source row became one daily aggregated dyadic event. The source half-weight index was retained as a unitless association measurement. Because calendar dates were unavailable, source days 1 through 6 were represented as relative intervals anchored to 1 January 2000. Empty group-event tables, individual records, observation windows, validation reports, and dataset metadata were added to satisfy the common contract. No claim is made that the daily association weight is a physical contact duration.

The canonical release manifest at `data/_shared/canonical_release.json` records SHA-256 checksums for every redistributed processed file.
