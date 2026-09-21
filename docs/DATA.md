# Data and Model Inputs

## ChEMBL 34

Use the official ChEMBL 34 SQLite release. The archive used for the frozen study had SHA-256:

`7d40409e4f31440674a79e26db628015e82b85b056a7d8234efe9c95b6cbd8e7`

The builder retains exact positive MIC records in `ug.mL-1`, excludes records with a data-validity comment, maps ChEMBL `assay_tax_id` to an accepted NCBI species-level ancestor, aggregates repeated compound-species-tax_id records by median log2 MIC, and requires at least 500 unique compounds per canonical species. Records whose taxonomy entry has no species-level ancestor are excluded and reported in `data_qc.json`. The builder also writes `dataset_flow.csv` and `species_taxonomy_detail.csv` and rejects an eligible table in which one canonical species name maps to multiple accepted tax IDs.

The E03 temporal audit reads the activity-linked `docs.year` field from the same
SQLite release. It assigns a compound by its first eligible MIC document year:
training through 2018, validation in 2019-2020, and testing in 2021-2023. Within
each cohort, only measurements from that cohort's year window are aggregated;
missing years and records outside the frozen windows are reported and excluded.
Document year is a publication/database timestamp proxy, not a compound-discovery
date and not evidence of prospective deployment.

The aggregate `mic_pairs_eligible.csv.gz` table is the direct input for E01, E02,
E04, and E05. E03 must start from the SQLite database plus the same organism and
taxonomy mappings so that temporal filtering occurs before aggregation. E05 also
requires the original benchmark `predictions.csv.gz` to reconstruct the exact
test partitions. All prepared matrices, compound-level predictions, checkpoints,
and manifests produced by these commands are local outputs excluded from this
code-only package.

## Organism Mapping and Broad Labels

`bact_mapper.json` and `combined_bioassay_data.tsv` come from IMI-COMBINE revision `3c222f96dfbab8172b4533d6f772f2de3b5350f6`. The former maps organisms to four broad pathogen classes and has SHA-256 `8c36472a45d7ab38e73de37f8e98937fc3b12326b04dcfdc277e3ab1691d0022`. `docs/taxonomy_mapping.csv` is the frozen NCBI Taxonomy EFetch lookup keyed by source tax ID, retrieved on 10 August 2026 and merged-alias audited on 14 August 2026. Historical alias `178876` is resolved to accepted tax ID `5207`; the corrected table has SHA-256 `9dc1eed3c42ce6a1a7e0fe16630f6abb98d805c22aebb12047d3094a477a61a6`. No local NCBI taxdump was used. `combined_bioassay_data.tsv` supplies the broad active-versus-inactive training endpoint.

## MolE

Generate MolE embeddings with the public MolE code and checkpoint, then provide a tab-separated table indexed by compound InChIKey. Model weights and embedding matrices are not distributed here.

## MoLFormer

Download `ibm/MoLFormer-XL-both-10pct` from its official Hugging Face repository. The extraction script uses local files only and writes a compressed NumPy archive containing `compound_inchikey` and `embedding` arrays.

## Maier Screen

The exact prepared inputs are `maier_screening_results.tsv.gz` and `prestwick_library_screened.tsv.gz` from MolE antimicrobial-potential revision `c7a5c4f742d3965f1143cc2046024248d1321d1d`. Their SHA-256 values are `c61d2e5f10efd4381ec5dd6a7c2c5e99d4b059235583140078caf4c7e4a815da` and `ee2e65b35f928d5e95bb206b4cf402a8c6a9c459208d6529d883997ab313be97`. The prepared binary file has 40 strain columns and no duplicate IDs, but its Boolean conversion had encoded 55 missing adjusted-P cells as zero. The source worksheet was therefore re-read before final evaluation. A positive endpoint requires at least one observed adjusted P <= 0.05; a negative requires every relevant strain to be observed and greater than 0.05; other rows are excluded only from that endpoint. Broad activity uses all 40 columns, and the E. coli endpoint uses its two named columns. Four source compounds without a structure are excluded and listed by the run output.

The upstream MolE preparation computed `ExactMolWt` with `Descriptors.ExactMolWt(Chem.MolFromSmiles(pchem_canonical_smile))` before salt removal. It produced `rdkit_no_salt` with `SaltRemover().StripMol(Chem.MolFromSmiles(rdkit_canonical_smile))` without custom definitions or arguments. The external MIC-margin score uses the former for concentration conversion and the latter for model features. The library structures were not matched compound by compound to the salt form or formulation used in the screen.

## External Species Mapping and Model Scope

`docs/maier_taxonomy_mapping.csv` is the verified 40-strain NCBI species lookup retrieved on 8 September 2026. It preserves source names and accepted tax IDs, including historical synonyms. The analysis script joins this lookup to the eligible MIC table and reports benchmark membership and original-refit status. E. coli and Clostridioides difficile are the two matched benchmark species. The original full-data case study analyzed only E. coli; the subsequent joint-model evaluation includes both species. See `docs/JOINT_EXTERNAL.md` for the distinct analysis and required local inputs.

The broad classifier is refitted on AntiMicrobial-KG. The original E. coli regressors are separate full-data single-species refits without a species indicator. The additional joint evaluation reconstructs the original joint models with prediction replay and compares split-matched single-species controls. The original separate external run used the pre-merged taxonomy dataset; the accepted-ID correction concerned Cryptococcus, and the E. coli training subset is unchanged. The joint reconstruction uses the final taxonomy-merged data. Its C. difficile endpoint is the one matched strain's inherited binary label.

The upstream MolE preparation notebook `workflow/01.prepare_training_data.ipynb` sets `PVAL_CUTOFF=0.05` and converts the Maier adjusted-P-value matrix using `screen_df <= PVAL_CUTOFF`. This inherited label is a screening-effect criterion, not a clinical MIC breakpoint. `average_precision_score` computes the quantity reported as AUPRC throughout the manuscript.

## Excluded Materials

- ChEMBL archives and extracted databases;
- raw or prepared activity tables;
- compound-level predictions and run manifests containing local paths;
- pretrained model weights and embeddings;
- figures, result tables, and manuscripts;
- hospital, patient, or other private data.
