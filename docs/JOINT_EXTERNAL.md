# Revision-stage joint-model external evaluation

This post hoc analysis evaluates the original joint MIC model configuration for both species matched to the Maier panel: Escherichia coli and Clostridioides difficile. The inherited binary screen is not an external quantitative MIC dataset. The external data had already been examined in the separate refit analyses; the new design was fixed before new joint scores were generated, not before all external analyses.

## Inputs and execution

First regenerate the taxonomy-merged MIC benchmark, the classical-view ablation (including Morgan plus MACCS), and the separate Maier evaluation using the README run order. These runs must retain their manifests and compound-level predictions locally. The joint runner uses those locally generated files; they are not included in this public package.

Edit the relative input/output directories in `docs/joint_external_config.example.json` to point to those runs. The species-run manifest's `data` path and the Maier manifest's `maier_screen` and `maier_library` paths must resolve on the executing machine. The output directory must not already exist.

```powershell
python -m amr_multiview.joint_external_validation --config docs/joint_external_config.example.json
```

This is a full CPU reconstruction with 15 joint and 30 matched single-species fits, not a quick unit test. It saves model checkpoints and individual predictions to the configured local output directory. It does not use a GPU. The released tests use only small synthetic numerical examples.

The internal held-out predictions must replay within the frozen tolerance before external scoring. The joint models retain the five original scaffold splits, 48-species indicator, balanced weights and fixed LightGBM settings. Matched single-species controls use the corresponding original split's training rows, not the original full-data E. coli refits. Scores are log2(0.02 x ExactMolWt) minus predicted log2 MIC, averaged across seeds. The concentration-only control omits the predicted MIC term. In the upstream MolE preparation, `Descriptors.ExactMolWt(Chem.MolFromSmiles(pchem_canonical_smile))` supplied ExactMolWt before salt removal. Model inputs used `SaltRemover().StripMol(Chem.MolFromSmiles(rdkit_canonical_smile))` with no custom definitions or arguments. These library structures were not matched compound by compound to the assay salt form or formulation.

## Overlap and statistics

The original broad-corpus exact-key exclusion and an exact-key exclusion against the complete 63,486-compound MIC development corpus define the primary cohort. Fingerprint and standardized-parent exclusions are computed against that whole MIC corpus, not a species-specific subset. Two species-level endpoints are evaluated separately, with additional E. coli strain-specific point metrics. All paired intervals use 5,000 compound bootstrap draws and are descriptive, post hoc and unadjusted for multiplicity.

## Aggregate outputs for Figure 8 and the supplements

Copy these generated aggregate outputs into the summary-table directory under the names below before running the figure script:

| Generated file | Figure/supplement input |
|---|---|
| metrics.csv | joint_external_metrics.csv |
| paired_differences.csv | joint_external_paired_differences.csv |
| cohort_flow.csv | joint_external_cohort_flow.csv |
| metrics_by_seed.csv | joint_external_metrics_by_seed.csv |
| topk_enrichment.csv | joint_external_topk_enrichment.csv |

Do not add the individual predictions, overlap audit, local manifests or saved models to the code-only release. The manuscript reports the completed aggregate experiment, including negative and uncertain results; reproducing those results requires the public source data and the earlier local runs.
