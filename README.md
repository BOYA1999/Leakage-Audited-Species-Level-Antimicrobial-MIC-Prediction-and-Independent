# Benchmarking molecular representations across generalization boundaries in species-level antimicrobial MIC prediction

This repository and its submitted code archive contain the code used for the species-level MIC benchmark and the independent phenotypic-screen transfer analyses.

The workflow covers:

- exact ChEMBL 34 MIC extraction, tax_id-based species normalization, and compound-species aggregation;
- five compound-level Bemis-Murcko scaffold splits;
- Morgan, multi-view, MolE, and MoLFormer representations;
- species-balanced LightGBM regression, compound-only conditioning ablation, and pooled or species-wise residual-quantile intervals with empirical coverage evaluation;
- secondary post hoc view ablation for Morgan plus physicochemical descriptors, MACCS keys, graph summaries, and the full multi-view representation;
- exact-key, fingerprint-identity, standardized-parent-identity, and structural-similarity auditing in the Maier 40-strain screen;
- pooled-versus-species-wise coverage audits, a secondary post hoc MIC replicate/IQR audit, paired bootstrap comparisons, and publication-figure generation;
- endpoint-specific E. coli exact-key, fingerprint-identity, and standardized-parent-identity audits;
- prediction-verified reconstruction of the joint MIC models for two matched external species, with split-matched single-species and concentration-only controls;
- matched Random Forest and XGBoost estimator baselines on the frozen five scaffold splits;
- sparse 48-task Chemprop D-MPNN evaluation on the same compound partitions;
- a ChEMBL-document-year temporal audit, leave-one-species-out boundary audit, and chemical-space/apparent-cliff diagnostics.

This supplementary package contains code only. It excludes raw datasets, result tables, manuscripts, figures, model weights, embeddings, checkpoints and local run manifests.

The 12 September 2026 evidence revision adds two fixed-prediction control replays and an activity-source audit; details are in `docs/CBC_EVIDENCE.md`. Identifier-linked predictions and sanitized run records are provided in the separate Supplementary Data archive submitted with the manuscript; they are not mirrored in this code repository. The public repository is https://github.com/BOYA1999/Leakage-Audited-Species-Level-Antimicrobial-MIC-Prediction-and-Independent.

Historical `jcim_*` file names and identifiers are retained for compatibility with
the frozen analysis outputs; they do not designate a target journal.

## Layout

```text
amr_multiview/   Dataset builders, representations, benchmark, and external validation.
scripts/         Statistical summaries and figure generation.
tests/           Small deterministic unit tests with synthetic structures.
docs/            Frozen experiment contract and data provenance.
```

## Environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

MoLFormer extraction requires CUDA. MIC construction, LightGBM training, statistical analysis, and plotting can run on CPU.
The additional model dependencies are pinned to the executed versions in
`requirements.txt`: `chemprop==2.2.4` and `xgboost==3.3.0`. XGBoost defaults to
CPU and verifies the requested device; the Chemprop wrapper defaults to one GPU
but accepts the standard Chemprop accelerator and device options.

## Public Inputs

Download the source data and models from their original providers:

- [ChEMBL downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads)
- [IMI-COMBINE broad-spectrum prediction](https://github.com/IMI-COMBINE/broad_spectrum_prediction)
- [MolE antimicrobial-potential repository](https://github.com/Barabasi-Lab/mole_antimicrobial_potential)
- [MolE pretrained checkpoint](https://zenodo.org/records/10803099)
- [IBM MoLFormer-XL](https://huggingface.co/ibm/MoLFormer-XL-both-10pct)
- [Maier et al. 40-strain screen](https://doi.org/10.1038/nature25979)

See `docs/DATA.md` for required files and the reproducibility boundary.

## Run Order

Build the frozen species-level MIC table:

```powershell
python -m amr_multiview.species_mic `
  --database <chembl_34.db> `
  --organism-mapping <bact_mapper.json> `
  --taxonomy-mapping docs/taxonomy_mapping.csv `
  --output-dir <species_mic_data>
```

This build writes `data_qc.json`, `dataset_flow.csv`, and
`species_taxonomy_detail.csv`. It fails if an eligible canonical species name
maps to more than one accepted NCBI tax ID.

Extract bounded-memory MoLFormer embeddings from a local model checkout:

```powershell
python -m amr_multiview.molformer_embeddings `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --model <molformer_model_directory> `
  --output <molformer_species_embeddings.npz> `
  --batch-size 32 `
  --max-length 202
```

Run the five-seed species MIC benchmark:

```powershell
python -m amr_multiview.species_mic_experiment `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --output-dir <species_mic_results> `
  --representations morgan multiview mole molformer `
  --mole-embeddings <mole_embeddings.tsv> `
  --molformer-embeddings <molformer_species_embeddings.npz>
```

Run the additional reliability audits (E01-E05):

```powershell
# E01: matched tree estimators on the frozen scaffold splits
python -m amr_multiview.species_mic_jcim_baselines `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --output-dir <rf_results> `
  --models rf `
  --representations morgan multiview `
  --seeds 42 100 3544 2025 2026 `
  --n-estimators 100 `
  --n-jobs -1

python -m amr_multiview.species_mic_jcim_baselines `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --output-dir <xgboost_results> `
  --models xgboost `
  --representations morgan multiview `
  --seeds 42 100 3544 2025 2026 `
  --n-estimators 300 `
  --xgb-device cuda

# E02: repeat this block for seeds 42, 100, 3544, 2025, and 2026
python -m amr_multiview.species_mic_chemprop prepare `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --output-dir <chemprop_seed\prepared> `
  --seed 42
python -m amr_multiview.species_mic_chemprop run `
  --dataset <chemprop_seed\prepared\chemprop_wide.csv> `
  --manifest <chemprop_seed\prepared\chemprop_manifest.json> `
  --output-dir <chemprop_seed\fit>
python -m amr_multiview.species_mic_chemprop evaluate `
  --prepared <chemprop_seed\prepared\chemprop_wide.csv> `
  --predictions <chemprop_seed\fit\chemprop_predictions.csv> `
  --manifest <chemprop_seed\prepared\chemprop_manifest.json> `
  --output-dir <chemprop_seed\evaluation>

# E03: ChEMBL document-year holdout
python -m amr_multiview.species_mic_temporal `
  --database <chembl_34.db> `
  --organism-mapping <bact_mapper.json> `
  --taxonomy-mapping docs\taxonomy_mapping.csv `
  --output-dir <temporal_results>

# E04: leave one species out
python -m amr_multiview.species_mic_cold_start `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --output-dir <cold_start_results>

# E05: chemical-space and apparent MIC-cliff audit
python -m amr_multiview.species_mic_chemical_space `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --predictions <species_mic_results\predictions.csv.gz> `
  --output-dir <chemical_space_results> `
  --models morgan multiview
```

Inputs, emitted files, frozen cohorts, and interpretation limits for these
commands are listed in `docs/JCIM_REVISION_AUDITS.md`. Generated predictions,
prepared Chemprop matrices, checkpoints, logs, and run manifests remain local
outputs and are not distributed in this code-only package.

Run the post hoc matched species-conditioning and conformal analysis:

```powershell
python -m amr_multiview.species_mic_experiment `
  --data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --output-dir <conditioning_results> `
  --representations morgan multiview `
  --conditioning both
```

Run the leakage-audited Maier transfer evaluation:

```powershell
python -m amr_multiview.maier_external_validation `
  --training-data <combined_bioassay_data.tsv> `
  --mic-data <species_mic_data\mic_pairs_eligible.csv.gz> `
  --maier-screen <maier_screening_results.tsv.gz> `
  --maier-library <prestwick_library_screened.tsv.gz> `
  --main-mole-embeddings <training_mole_embeddings.tsv> `
  --maier-mole-embeddings <maier_mole_embeddings.tsv> `
  --molformer-embeddings <combined_molformer_embeddings.npz> `
  --output-dir <maier_results>
```

Run the joint-model evaluation after the main benchmark, classical ablation and separate Maier run, following `docs/JOINT_EXTERNAL.md`. That document also maps the five aggregate joint outputs into the summary-table directory required by the updated Figure 8.

Generate summaries and figures:

```powershell
python scripts\analyze_species_mic_results.py `
  --species-dir <species_mic_results> `
  --maier-dir <maier_results> `
  --conditioning-dir <conditioning_results> `
  --species-ablation-dir <species_ablation_results> `
  --maier-ablation-dir <maier_ablation_results> `
  --maier-source-revision c7a5c4f742d3965f1143cc2046024248d1321d1d `
  --output <summary_tables>

python scripts\make_species_mic_figures.py `
  --species-dir <species_mic_results> `
  --conditioning-dir <conditioning_results> `
  --table-dir <summary_tables> `
  --output <figure_directory>
```

Aggregate the additional experiments and render the corresponding audit
figures with explicit local input/output paths:

```powershell
python scripts\analyze_jcim_revision.py `
  --experiments-dir <reliability_audit_results> `
  --original-dir <original_scaffold_benchmark> `
  --tables-dir <summary_tables> `
  --report-dir <revision_report> `
  --conditioning-species-deltas <conditioning_species_deltas.csv>

python scripts\make_jcim_model_benchmark_figure.py `
  --model-summary <summary_tables\jcim_model_benchmark_summary.csv> `
  --paired <summary_tables\jcim_model_paired_differences.csv> `
  --complexity <summary_tables\jcim_complexity_summary.csv> `
  --output-dir <figure_directory>

python scripts\make_jcim_revision_figures_3_5.py `
  --temporal-dir <reliability_audit_results\temporal\full> `
  --cold-start-dir <reliability_audit_results\cold_start\full> `
  --chemical-dir <reliability_audit_results\chemical_space\full> `
  --output-dir <figure_directory>

python scripts\make_jcim_reliability_figure.py --output-dir <figure_directory>
python scripts\make_jcim_toc_graphic.py --output-dir <figure_directory>
```

The Figure 3-5 program reads aggregate experiment outputs plus local
compound-level audit files. Those compound-level inputs are deliberately not
redistributed; regenerate them from the public source data with E03-E05.

The same benchmark script can run the secondary view ablation with
`--representations morgan morgan_physchem morgan_maccs morgan_graph multiview`.
The summary script reports empirical pair-weighted coverage separately from equal-species
coverage, compares pooled with species-wise calibration, generates the species,
conditioning, feature-definition, data-flow, and ablation-uncertainty supplement
tables, records accepted-tax_id-safe replicate/IQR groups, audits the exact Maier
binary-label contract, and performs separate broad-training and E. coli-training
fingerprint and standardized-parent audits.

Revision diagnostics additionally generate `full_vs_maccs_paired.csv`,
`full_vs_maccs_by_seed.csv`, `coverage_weighting_summary.csv`,
`coverage_weighting_by_seed.csv`, and `maier_species_crosswalk.csv`.
The companion Supplementary Data also include
`pretrained_species_comparison.csv`, which gives the 48 species-level
comparisons supporting Supplementary Table S19.
The default external taxonomy lookup is `docs/maier_taxonomy_mapping.csv`.
Use `--maier-taxonomy-mapping` to specify its location explicitly.

## Evidence Boundary

The code supports retrospective molecular ranking and uncertainty analysis. The Maier data are an external fixed-concentration phenotypic screen, not clinical MIC validation. Model scores do not establish prospective antimicrobial activity, experimental safety, pharmacokinetics, mechanism, patentability, or clinical efficacy.

The broad classifiers and original full-data E. coli MIC regressors are independent
refits. The additional joint-model experiment reconstructs the internal 48-species
model and tests both taxonomically matched external species. This closes the model
identity gap for binary ranking, not the absence of external quantitative MIC or
interval-coverage validation. Scaffold-grouped partitions do not establish ordinary
row-level exchangeability for the conformal procedure.

The temporal experiment uses ChEMBL document year as a database timestamp proxy,
not as a compound-discovery date or prospective validation. Leave-one-species-out
uses compound-only features or a broad pathogen class already observed among the
training species; it is not zero-shot species-representation learning. Apparent
MIC cliffs are outcome-defined, post hoc subgroups with dependent pair edges and
must not be read as causal or confirmatory cliff discovery.

The author-owned repository software is released under the MIT License in `LICENSE`. That license does not cover third-party datasets, pretrained weights or source-derived Supplementary Data, which retain their original terms and attribution requirements.
