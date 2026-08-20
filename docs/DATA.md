# Data and Model Inputs

## ChEMBL 34

Use the official ChEMBL 34 SQLite release. The archive used for the frozen study had SHA-256:

`7d40409e4f31440674a79e26db628015e82b85b056a7d8234efe9c95b6cbd8e7`

The builder retains exact positive MIC records in `ug.mL-1`, excludes records with a data-validity comment, maps ChEMBL `assay_tax_id` to an accepted NCBI species-level ancestor, aggregates repeated compound-species-tax_id records by median log2 MIC, and requires at least 500 unique compounds per canonical species. Records whose taxonomy entry has no species-level ancestor are excluded and reported in `data_qc.json`. The builder also writes `dataset_flow.csv` and `species_taxonomy_detail.csv` and rejects an eligible table in which one canonical species name maps to multiple accepted tax IDs.

## Organism Mapping and Broad Labels

`bact_mapper.json` and `combined_bioassay_data.tsv` come from IMI-COMBINE revision `3c222f96dfbab8172b4533d6f772f2de3b5350f6`. The former maps organisms to four broad pathogen classes and has SHA-256 `8c36472a45d7ab38e73de37f8e98937fc3b12326b04dcfdc277e3ab1691d0022`. `docs/taxonomy_mapping.csv` is the frozen NCBI Taxonomy EFetch lookup keyed by source tax ID, retrieved on 10 August 2026 and merged-alias audited on 14 August 2026. Historical alias `178876` is resolved to accepted tax ID `5207`; the corrected table has SHA-256 `9dc1eed3c42ce6a1a7e0fe16630f6abb98d805c22aebb12047d3094a477a61a6`. No local NCBI taxdump was used. `combined_bioassay_data.tsv` supplies the broad active-versus-inactive training endpoint.

## MolE

Generate MolE embeddings with the public MolE code and checkpoint, then provide a tab-separated table indexed by compound InChIKey. Model weights and embedding matrices are not distributed here.

## MoLFormer

Download `ibm/MoLFormer-XL-both-10pct` from its official Hugging Face repository. The extraction script uses local files only and writes a compressed NumPy archive containing `compound_inchikey` and `embedding` arrays.

## Maier Screen

The exact prepared inputs are `maier_screening_results.tsv.gz` and `prestwick_library_screened.tsv.gz` from MolE antimicrobial-potential revision `c7a5c4f742d3965f1143cc2046024248d1321d1d`. Their SHA-256 values are `c61d2e5f10efd4381ec5dd6a7c2c5e99d4b059235583140078caf4c7e4a815da` and `ee2e65b35f928d5e95bb206b4cf402a8c6a9c459208d6529d883997ab313be97`. The 40 strain columns are source-supplied binary 0/1 labels with no missing values or duplicate IDs. Broad activity is the maximum over all 40 columns; the E. coli endpoint is the maximum over its two named columns. No label threshold is fitted. Four source compounds without a structure are excluded and listed by the run output.

## Excluded from This Package

- ChEMBL archives and extracted databases;
- raw or prepared activity tables;
- compound-level predictions and run manifests containing local paths;
- pretrained model weights and embeddings;
- figures, result tables, and manuscripts;
- hospital, patient, or other private data.
