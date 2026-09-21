# Frozen Experiment Contract

The primary endpoint is median `log2(MIC ug/mL)` for exact ChEMBL 34 MIC measurements. ChEMBL `assay_tax_id` values are mapped through the frozen NCBI table to accepted species-level ancestors, and species require at least 500 unique compounds after compound-species-tax_id aggregation. Historical merged aliases must be resolved before aggregation; the final eligible table must map each canonical species name to exactly one accepted tax ID.

Five seeds are fixed: 42, 100, 3544, 2025, and 2026. Compounds are assigned by Bemis-Murcko scaffold to 70% training, 15% validation/calibration, and 15% test partitions. Exact-compound and scaffold overlap must be zero between partitions.

Compounds with an empty Bemis-Murcko scaffold receive a compound-specific fallback key; this is audited and does not guarantee zero close-analogue similarity for all acyclic compounds.

Fixed primary comparators are species median, Morgan, the full Morgan/MACCS/physicochemical/graph multi-view, frozen MolE, and frozen MoLFormer. Learned representations share estimator settings and species-balanced weighting. Primary internal metrics are macro-species MAE, RMSE, Spearman correlation, one- and two-dilution accuracy, pooled marginal conformal coverage, equal-species coverage summaries, and interval width. Morgan plus physicochemical, MACCS, or graph features, compound-only conditioning models, species-wise conformal calibration, and the MIC replicate-dispersion audit are secondary post hoc diagnostics.

The primary Maier cohort excludes exact standard-InChIKey overlap with the training corpus. Exact overlaps are secondary and never pooled. Maier labels are the source-supplied binary strain labels: the full 40-strain endpoint is their row maximum, and the originally selected single-species-refit endpoint is the row maximum of the two Escherichia coli columns. Missing or duplicate labels may not be silently imputed or resolved, and no external-label threshold may be fitted.

The E. coli transfer cohort must also be checked against exact InChIKeys and radius-2 Morgan fingerprint identities in the ChEMBL E. coli MIC training set before sensitivity scoring. Standardized-parent equality is an additional post hoc audit based on fragment-parent extraction, uncharging, and canonical isomeric or non-isomeric SMILES without tautomer canonicalization.

Nearest-training Morgan Tanimoto bins are `<0.30`, `0.30-0.50`, `0.50-0.70`, and `>=0.70`. The completed broad-training audit detected 50 exact-key-nonoverlap compounds with Tanimoto 1.0, leaving a 965-compound post hoc sensitivity cohort. The conservative non-isomeric parent audit detected 75 broad-training identities and leaves 940 compounds. The separate E. coli-training audits detected 21 fingerprint identities and 36 non-isomeric parent identities, leaving 994- and 979-compound endpoint-specific post hoc sensitivity cohorts. None replaces the primary cohort. Tanimoto equality is treated as fingerprint identity, not proof of exact structural identity.

No thresholds, species gates, splits, seeds, or primary metrics may be changed after results are inspected to rescue a model claim. Secondary post hoc intervals are descriptive; no multiplicity-adjusted confirmatory inference is claimed.

## Revision-Stage Diagnostics Added on 8 September 2026

The external taxonomic crosswalk, direct full-versus-Morgan-plus-MACCS paired comparison, and equal-compound coverage sensitivity are post hoc additions. They do not alter the original training, primary cohorts, or evaluation labels. The crosswalk identifies an additional C. difficile match. A subsequent revision-stage experiment reconstructs the original joint MIC models and evaluates both matched species using the same binary external labels, with split-matched single-species controls. Its design was fixed before generating new joint scores, but after the external data had been examined in the separate refit analyses.

The joint experiment retains the 1,015-compound primary cohort; whole-MIC-development-corpus fingerprint and standardized-parent exclusions produce distinct 972- and 951-compound sensitivity cohorts. Three representations and five original splits produce 15 joint fits and 30 single-species controls. Prediction replay is required before external scoring. Paired differences use 5,000 compound bootstrap draws and remain descriptive, post hoc and unadjusted. No external quantitative MIC error or interval-coverage claim follows. See `docs/JOINT_EXTERNAL.md` and the portable example configuration.

The graph block consists of handcrafted summary statistics, not a trained GNN. Frozen representations share a fixed downstream model without representation-specific optimization. Calibration residuals are pooled over records while partitions are scaffold-grouped; ordinary row-level exchangeability is not established. Coverage is reported empirically, with separate pair, species, and compound weighting, not as a proven finite-sample guarantee.

## Additional Reliability Audits Added on 10-11 September 2026

E01 replays LightGBM when requested and adds Random Forest and XGBoost with the
same Morgan or multi-view inputs, five seeds, scaffold partitions, species
one-hot encoding, inverse species-frequency weights, and primary metrics.
Runtime and serialized-size outputs are descriptive for the recorded hardware,
software, feature scope, and prediction scope; they are not portable complexity
rankings.

E02 uses Chemprop 2.2.4 as a sparse 48-output D-MPNN. Each output column is one
species, the compound partitions are the original five scaffold splits, and task
weights reproduce inverse training-label-count balancing, normalized to mean one.
Targets are standardized separately for each task using training means and
standard deviations, and missing labels are masked. The training MAE loss uses
the task weights, while checkpoint selection minimizes unweighted MAE pooled
over all finite validation labels on the standardized scale; it is not a
macro-species validation metric. Predictions are transformed back by the
training-fitted checkpoint UnscaleTransform, and final metrics are computed per
species on the original log2 MIC scale before macro averaging. The validation
split selects the checkpoint, so the Chemprop comparison reports point metrics
only; it does not reuse validation residuals for conformal uncertainty.

E03 defines temporal cohorts from each compound's first eligible ChEMBL document
year and filters measurement rows before compound-species aggregation: training
through 2018, validation in 2019-2020, and testing in 2021-2023. Species remain
eligible only when their training-era support is at least 500 compounds. Exact
compound overlap is prohibited. ChEMBL document year is a publication/database
timestamp proxy, not discovery time, and the experiment is retrospective rather
than prospective validation.

E04 holds out one species at a time and compares Morgan compound-only features
with Morgan plus a broad pathogen class encoded from the training species only.
The class model must stop if a held-out class is absent from training. Results are
reported for all held-out pairs, exact compounds seen with another species, and
exact compound nonoverlap. This tests compound transfer within already known
broad classes; it is not a zero-shot species embedding or a joint species-plus-
chemical cold-start experiment. Leave-one-species-out training sets overlap, so
across-species intervals are descriptive stability summaries.

E05 reconstructs each original scaffold test split, measures exact radius-2,
2048-bit Morgan nearest-training similarity, audits empty-Murcko fallback
compounds, and defines apparent MIC cliffs within the same species and nonempty
Murcko scaffold. The primary apparent-cliff definition is Tanimoto at least 0.8
and absolute MIC difference at least 2 log2 units; three threshold sensitivities
are retained. These are outcome-defined post hoc subgroups. Pair edges share
compounds, split replicates overlap, no multiplicity-adjusted confirmatory
inference is claimed, and no causal or universal activity-cliff conclusion follows.
