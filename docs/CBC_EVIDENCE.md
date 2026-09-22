# Evidence revisions, 12 and 22 September 2026

The 12 September revision added identifier-linked evidence and two fixed-prediction controls without learned-model training. The 22 September revision adds direct estimator-by-representation contrasts, record-quality-stratified descriptive contrasts, and a full current-run Random Forest reproducibility check. Historical file names are preserved for provenance. The code is mirrored at https://github.com/BOYA1999/Leakage-Audited-Species-Level-Antimicrobial-MIC-Prediction-and-Independent; the companion data archive remains a separate manuscript supplement and is not mirrored there.

## Replay from the companion data archive

Extract both archives into separate folders. Run from the code folder, with `DATA` replaced by the extracted data folder:

```text
python scripts/replay_cbc_controls.py --data DATA --out control_replay
python scripts/make_cbc_revision_figures.py --data DATA --out cbc_figures
python scripts/analyze_cbc_interactions.py --benchmark-by-seed DATA/tables/jcim_model_benchmark_by_seed.csv --replicate-groups DATA/tables/mic_error_by_replicate_group.csv --output-dir interaction_audit
```

The control replay reads only supplied identifier-linked temporal pairs, saved predictions, external binary labels and cohort flags. It reproduces the training-era species-median comparator and 5,000-draw paired joint-model versus molecular-weight contrasts. It does not refit models. Bootstrap intervals condition on the saved five-fit mean scores. The random generator consumes cohorts in sorted species order, then primary, fingerprint-nonidentity and parent-nonidentity order. CSV floating-point parsing uses round-trip precision to preserve exact ties and dilution-boundary comparisons.

The figure script produces main Figures 1-3 and 6 and Supplementary Figures S1, S3 and S4. Main Figure 4 and Supplementary Figure S2 reuse the LOSO and chemical diagnostics from `make_jcim_revision_figures_3_5.py`; main Figure 5 reuses the conformal display from `make_species_mic_figures.py`. Corresponding frozen source tables are in the data archive. Figure numbers in old source filenames reflect historical allocation, not current captions.

The interaction script uses the saved shared-split summaries. It reports paired multi-view-minus-Morgan MAE effects for each estimator, direct differences between those effects, and post hoc record-weighted LightGBM effects within three existing endpoint-record strata. The five-split t intervals are descriptive stability intervals, not confidence intervals from independent populations. The record-quality contrasts are associations and do not identify causal effects of measurement heterogeneity.

## Current Random Forest replay

On 22 September 2026, the released runner was executed end to end for Morgan and multi-view RF100 across seeds 42, 100, 3544, 2025 and 2026. The replay matched the historical data hash, package versions, metric contract and split fingerprints. All 319,040 prediction rows and 490 metric rows were reproduced within the numerical tolerances recorded in the companion audit. The maximum absolute prediction difference was `5.33e-15`; the maximum per-row metric difference was `1.03e-6` for one Spearman value; and the maximum mean-performance difference was `4.31e-9` for Spearman. The result is therefore a numerical reproduction, not a bit-identical file reproduction. Runtime fields and compressed-file hashes are not used as equality criteria.

This current replay binds the frozen configuration to the current runner hash. It does not reconstruct the missing source hash of the historical RF execution, and it does not extend the RF model to the temporal, held-out-species or external-screen regimes.

## Source record audit

Obtain the exact ChEMBL 34 SQLite database and IMI-COMBINE mapping described in DATA.md. The following read-only database replay verifies median/IQR endpoints and exports activity-level inclusion and exclusion records:

```text
python scripts/audit_cbc_source.py --chembl-db chembl_34.db --pairs DATA/evidence/pair_endpoints.csv.gz --class-mapping bact_mapper.json --out source_replay
```

The source database is not bundled. The full prefilter contains 720,492 MIC records; 383,597 retained records from 6,171 documents and 52,690 assays support 214,068 final pairs. The full record linkage and ordered exclusions are included in the data archive. A unit-conversion audit covers 2,431 records from the 126 pairs with IQR at least 8. Six differences of 0.0005 ug/mL are compatible with decimal rounding; source-publication conditions were not independently verified or harmonized.

## Evidence limits

- Test predictions can be joined by compound identifier, species and seed to `five_split_membership.csv.gz`; scaffold strings are replaced with SHA-256 group identifiers, not reconstructed chemical structures.
- Repeated MIC records are not assumed to be interchangeable technical replicates.
- Historical manifest paths are sanitized; original manifest hashes remain available. Current environment observations are separately dated. The current RF replay supplies a present runner hash but does not reconstruct the missing historical RF source hash; the historical D-MPNN runner-source hash also remains missing.
- D-MPNN finite-label weighted standardized MAE divides by the finite-label count. Selection uses unweighted standardized validation MAE; evaluation uses inverse-transformed outputs with equal-species averaging.
- Selected epochs are 41, 49, 50, 49 and 46; completed epochs are 46, 50, 50, 50 and 50 for seeds 42, 100, 3544, 2025 and 2026.
- The saved MoLFormer tokenizer reproduces 1,229 compounds exceeding 202 tokens. Token counts and species-level exposure are supplied; no common-untruncated-subset model comparison was run.
- Unit tests and archive checks are technical QA, not independent validation of the biological measurements or a new training reproduction.

Third-party data terms, final journal portal checks and any public release of the companion data archive remain author-controlled decisions. The repository software uses its included MIT licence. No network publication is performed by these scripts.
