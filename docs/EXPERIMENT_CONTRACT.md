# Frozen Experiment Contract

The primary endpoint is median `log2(MIC ug/mL)` for exact ChEMBL 34 MIC measurements. ChEMBL `assay_tax_id` values are mapped through the frozen NCBI table to accepted species-level ancestors, and species require at least 500 unique compounds after compound-species-tax_id aggregation. Historical merged aliases must be resolved before aggregation; the final eligible table must map each canonical species name to exactly one accepted tax ID.

Five seeds are fixed: 42, 100, 3544, 2025, and 2026. Compounds are assigned by Bemis-Murcko scaffold to 70% training, 15% validation/calibration, and 15% test partitions. Exact-compound and scaffold overlap must be zero between partitions.

Compounds with an empty Bemis-Murcko scaffold receive a compound-specific fallback key; this is audited and does not guarantee zero close-analogue similarity for all acyclic compounds.

Fixed primary comparators are species median, Morgan, the full Morgan/MACCS/physicochemical/graph multi-view, frozen MolE, and frozen MoLFormer. Learned representations share estimator settings and species-balanced weighting. Primary internal metrics are macro-species MAE, RMSE, Spearman correlation, one- and two-dilution accuracy, pooled marginal conformal coverage, equal-species coverage summaries, and interval width. Morgan plus physicochemical, MACCS, or graph features, compound-only conditioning models, species-wise conformal calibration, and the MIC replicate-dispersion audit are secondary post hoc diagnostics.

The primary Maier cohort excludes exact standard-InChIKey overlap with the training corpus. Exact overlaps are secondary and never pooled. Maier labels are the source-supplied binary strain labels: the full 40-strain endpoint is their row maximum, and exact species-transfer is the row maximum of the two Escherichia coli columns. Missing or duplicate labels may not be silently imputed or resolved, and no external-label threshold may be fitted.

The E. coli transfer cohort must also be checked against exact InChIKeys and radius-2 Morgan fingerprint identities in the ChEMBL E. coli MIC training set before sensitivity scoring. Standardized-parent equality is an additional post hoc audit based on fragment-parent extraction, uncharging, and canonical isomeric or non-isomeric SMILES without tautomer canonicalization.

Nearest-training Morgan Tanimoto bins are `<0.30`, `0.30-0.50`, `0.50-0.70`, and `>=0.70`. The completed broad-training audit detected 50 exact-key-nonoverlap compounds with Tanimoto 1.0, leaving a 965-compound post hoc sensitivity cohort. The conservative non-isomeric parent audit detected 75 broad-training identities and leaves 940 compounds. The separate E. coli-training audits detected 21 fingerprint identities and 36 non-isomeric parent identities, leaving 994- and 979-compound endpoint-specific post hoc sensitivity cohorts. None replaces the primary cohort. Tanimoto equality is treated as fingerprint identity, not proof of exact structural identity.

No thresholds, species gates, splits, seeds, or primary metrics may be changed after results are inspected to rescue a model claim. Secondary post hoc intervals are descriptive; no multiplicity-adjusted confirmatory inference is claimed.
