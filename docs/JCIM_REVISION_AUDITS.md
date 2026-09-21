# Additional Reliability Audits

These revision-stage experiments extend the frozen scaffold benchmark without
changing its endpoint, five seeds, or primary equal-species metrics. Commands use
PowerShell path separators, but the Python modules are platform independent.
Smoke-test options such as `--max-compounds`, `--max-folds`,
`--max-pairs-per-species`, `--max-test-compounds`, and `--max-cliff-groups` do
not constitute a full manuscript run.

## E01: Matched tree-estimator baselines

Input: the taxonomy-merged `mic_pairs_eligible.csv.gz` table.

```powershell
python -m amr_multiview.species_mic_jcim_baselines `
  --data <mic_pairs_eligible.csv.gz> `
  --output-dir <e01_rf_results> `
  --models rf `
  --representations morgan multiview `
  --seeds 42 100 3544 2025 2026 `
  --n-estimators 100 `
  --n-jobs -1

python -m amr_multiview.species_mic_jcim_baselines `
  --data <mic_pairs_eligible.csv.gz> `
  --output-dir <e01_xgboost_results> `
  --models xgboost `
  --representations morgan multiview `
  --seeds 42 100 3544 2025 2026 `
  --n-estimators 300 `
  --xgb-device cuda
```

Use `--models lightgbm` for an exact implementation replay. The module writes
`metrics.csv`, `metrics_summary.csv`, `predictions.csv.gz`, `runtime.csv`,
`split_fingerprints.csv`, `metric_contract.json`, and `manifest.json`. A full
run requires 48 species. The XGBoost device is checked after fitting and an
unrequested CPU fallback is rejected. Timing and serialized byte counts describe
only the recorded environment and code path.

The RF tree count was fixed at 100 before any RF outcome was available because
the predeclared 300-tree run did not complete within the feasible compute
window and emitted no result artifact. All data, partitions, features, seeds,
depth settings, and metrics were otherwise unchanged; parallelism increased
from four workers to all available workers. XGBoost retained 300 trees and the
executed run verified CUDA use.

## E02: Sparse multitask Chemprop D-MPNN

Input: the same eligible table. Execute the following for each frozen seed.

```powershell
python -m amr_multiview.species_mic_chemprop prepare `
  --data <mic_pairs_eligible.csv.gz> `
  --output-dir <seed_dir\prepared> `
  --seed 42

python -m amr_multiview.species_mic_chemprop run `
  --dataset <seed_dir\prepared\chemprop_wide.csv> `
  --manifest <seed_dir\prepared\chemprop_manifest.json> `
  --output-dir <seed_dir\fit> `
  --epochs 50 `
  --batch-size 256 `
  --accelerator gpu `
  --devices 1 `
  --warmup-epochs 3 `
  --patience 5

python -m amr_multiview.species_mic_chemprop evaluate `
  --prepared <seed_dir\prepared\chemprop_wide.csv> `
  --predictions <seed_dir\fit\chemprop_predictions.csv> `
  --manifest <seed_dir\prepared\chemprop_manifest.json> `
  --output-dir <seed_dir\evaluation>
```

Preparation emits the wide sparse-label table, `species_map.csv`, and
`chemprop_manifest.json`. The run emits reproducible command files, logs,
predictions, checkpoints, and `runtime.json`; evaluation emits `metrics.csv`,
`metrics_summary.csv`, `predictions.csv.gz`, and `evaluation_manifest.json`.
All generated artifacts remain outside the public code package. The validation
split selects the best checkpoint, so only point estimates are evaluated and no
Chemprop conformal interval is reported.

Targets are standardized per task using training means and standard deviations;
missing labels are masked. The training MAE loss applies inverse
training-label-count task weights normalized to mean one. Checkpoint selection
minimizes unweighted MAE pooled over all finite validation labels on the
standardized scale, rather than macro-species MAE. Predictions are returned to
the original log2 MIC scale by the checkpoint's training-fitted UnscaleTransform,
then evaluated per species and macro averaged.

A local implementation audit checked code, executed configurations, checkpoints,
and saved outputs for all five seeds. The state dictionaries in `best.pt` matched
the selected checkpoints. Reconstructed validation selection scores differed
from the saved values by at most 6.98e-8, and reconstructed test macro-MAE differed
by at most 2.22e-16. This cross-check does not establish a cryptographic binding
to the exact source executed during training because a training-source snapshot
hash was not recorded. The comparison therefore concerns the fixed configuration
and five scaffold splits used in this study, without claiming optimized
performance for the whole algorithm family.

## E03: Document-year temporal holdout

Inputs: the ChEMBL 34 SQLite database, `bact_mapper.json`, and the frozen taxonomy
mapping.

```powershell
python -m amr_multiview.species_mic_temporal `
  --database <chembl_34.db> `
  --organism-mapping <bact_mapper.json> `
  --taxonomy-mapping docs\taxonomy_mapping.csv `
  --output-dir <e03_results> `
  --min-compounds 500 `
  --n-estimators 300
```

The module writes raw-year and overlap audits, compound cohorts, all and eligible
cohort pairs, species counts, the executed SQL, predictions, metrics, runtime,
and a manifest. Training compounds first appear through 2018, validation
compounds in 2019-2020, and test compounds in 2021-2023; measurement rows are
also restricted to their assigned window before aggregation. Missing years are
excluded and retained in the audit. `docs.year` is a publication/database proxy,
not discovery time or prospective validation.

## E04: Leave-one-species-out boundary

Input: the taxonomy-merged eligible table.

```powershell
python -m amr_multiview.species_mic_cold_start `
  --data <mic_pairs_eligible.csv.gz> `
  --output-dir <e04_results> `
  --expected-species 48 `
  --n-estimators 300
```

Outputs are `predictions.csv.gz`, `metrics.csv`, `metrics_summary.csv`,
`species_level_paired_deltas.csv`, `paired_delta_summary.csv`,
`overlap_checks.csv`, and `manifest.json`. The two models use Morgan compound
features alone or Morgan plus the held-out species' broad pathogen class, with
the class encoder fitted on training species. The audit stops if that class was
not seen during training. Because exact compounds commonly occur with other
species, the all-pairs result is reported beside an exact-compound-nonoverlap
cohort. This is transfer within known classes, not zero-shot species
representation and not simultaneous species-plus-chemical cold start.

## E05: Chemical-space and apparent-cliff diagnostics

Inputs: the eligible table and compound-level predictions from the frozen
Morgan/multi-view scaffold benchmark.

```powershell
python -m amr_multiview.species_mic_chemical_space `
  --data <mic_pairs_eligible.csv.gz> `
  --predictions <species_mic_results\predictions.csv.gz> `
  --output-dir <e05_results> `
  --seeds 42 100 3544 2025 2026 `
  --models morgan multiview
```

Outputs include scaffold and acyclic-fallback summaries, reconstructed split
checks, nearest-training similarities, similarity-error tables, apparent-cliff
edge/scaffold/species tables, cliff-error summaries, and `manifest.json`. Split
reconstruction must match the supplied predictions or the run stops. Apparent
cliffs are same-species, same-nonempty-Murcko pairs defined after outcomes are
known. Their edges are dependent, the five splits overlap, and threshold
sensitivities are exploratory and unadjusted; the files do not establish causal,
universal, or confirmatory activity-cliff behavior.
