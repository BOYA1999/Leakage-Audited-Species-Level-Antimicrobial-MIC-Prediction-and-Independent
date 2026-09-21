# Evidence revision, 12 September 2026

This revision adds no learned-model training. It supplies identifier-linked evidence and two fixed-prediction controls alongside the original code. Historical file names are preserved for provenance. The code is mirrored at https://github.com/BOYA1999/Leakage-Audited-Species-Level-Antimicrobial-MIC-Prediction-and-Independent; the companion data archive remains a separate manuscript supplement and is not mirrored there.

## Replay from the companion data archive

Extract both archives into separate folders. Run from the code folder, with `DATA` replaced by the extracted data folder:

```text
python scripts/replay_cbc_controls.py --data DATA --out control_replay
python scripts/make_cbc_revision_figures.py --data DATA --out cbc_figures
```

The control replay reads only supplied identifier-linked temporal pairs, saved predictions, external binary labels and cohort flags. It reproduces the training-era species-median comparator and 5,000-draw paired joint-model versus molecular-weight contrasts. It does not refit models. Bootstrap intervals condition on the saved five-fit mean scores. The random generator consumes cohorts in sorted species order, then primary, fingerprint-nonidentity and parent-nonidentity order. CSV floating-point parsing uses round-trip precision to preserve exact ties and dilution-boundary comparisons.

The figure script produces main Figures 1-3 and 6 and Supplementary Figures S1, S3 and S4. Main Figure 4 and Supplementary Figure S2 reuse the LOSO and chemical diagnostics from `make_jcim_revision_figures_3_5.py`; main Figure 5 reuses the conformal display from `make_species_mic_figures.py`. Corresponding frozen source tables are in the data archive. Figure numbers in old source filenames reflect historical allocation, not current captions.

## Source record audit

Obtain the exact ChEMBL 34 SQLite database and IMI-COMBINE mapping described in DATA.md. The following read-only database replay verifies median/IQR endpoints and exports activity-level inclusion and exclusion records:

```text
python scripts/audit_cbc_source.py --chembl-db chembl_34.db --pairs DATA/evidence/pair_endpoints.csv.gz --class-mapping bact_mapper.json --out source_replay
```

The source database is not bundled. The full prefilter contains 720,492 MIC records; 383,597 retained records from 6,171 documents and 52,690 assays support 214,068 final pairs. The full record linkage and ordered exclusions are included in the data archive. A unit-conversion audit covers 2,431 records from the 126 pairs with IQR at least 8. Six differences of 0.0005 ug/mL are compatible with decimal rounding; source-publication conditions were not independently verified or harmonized.

## Evidence limits

- Test predictions can be joined by compound identifier, species and seed to `five_split_membership.csv.gz`; scaffold strings are replaced with SHA-256 group identifiers, not reconstructed chemical structures.
- Repeated MIC records are not assumed to be interchangeable technical replicates.
- Historical manifest paths are sanitized; original manifest hashes remain available. Current environment observations are separately dated. Missing historical RF/D-MPNN runner-source hashes remain missing.
- D-MPNN finite-label weighted standardized MAE divides by the finite-label count. Selection uses unweighted standardized validation MAE; evaluation uses inverse-transformed outputs with equal-species averaging.
- Selected epochs are 41, 49, 50, 49 and 46; completed epochs are 46, 50, 50, 50 and 50 for seeds 42, 100, 3544, 2025 and 2026.
- The saved MoLFormer tokenizer reproduces 1,229 compounds exceeding 202 tokens. Token counts and species-level exposure are supplied; no common-untruncated-subset model comparison was run.
- Unit tests and archive checks are technical QA, not independent validation of the biological measurements or a new training reproduction.

Third-party data terms, final journal portal checks and any public release of the companion data archive remain author-controlled decisions. The repository software uses its included MIT licence. No network publication is performed by these scripts.
