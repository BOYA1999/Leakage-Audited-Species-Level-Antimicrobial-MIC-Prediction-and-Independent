# Species-Level Antimicrobial MIC Prediction

Code for leakage-audited species-level MIC regression and independent phenotypic-screen representation transfer using public molecular and assay resources.

The workflow covers:

- exact ChEMBL 34 MIC extraction, tax_id-based species normalization, and compound-species aggregation;
- five compound-level Bemis-Murcko scaffold splits;
- Morgan, multi-view, MolE, and MoLFormer representations;
- species-balanced LightGBM regression, compound-only conditioning ablation, and pooled or species-wise finite-sample split-conformal intervals;
- secondary post hoc view ablation for Morgan plus physicochemical descriptors, MACCS keys, graph summaries, and the full multi-view representation;
- exact-key, fingerprint-identity, standardized-parent-identity, and structural-similarity auditing in the Maier 40-strain screen;
- pooled-versus-species-wise coverage audits, a secondary post hoc MIC replicate/IQR audit, paired bootstrap comparisons, and publication-figure generation;
- endpoint-specific E. coli exact-key, fingerprint-identity, and standardized-parent-identity audits.

This is a code-only public package. Raw datasets, result tables, manuscripts, figures, model weights, embeddings, checkpoints, and local run manifests are intentionally excluded.

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
  --table-dir <summary_tables> `
  --output <figure_directory>
```

The same benchmark script can run the secondary view ablation with
`--representations morgan morgan_physchem morgan_maccs morgan_graph multiview`.
The summary script reports pooled marginal coverage separately from equal-species
coverage, compares pooled with species-wise calibration, generates the species,
conditioning, feature-definition, data-flow, and ablation-uncertainty supplement
tables, records accepted-tax_id-safe replicate/IQR groups, audits the exact Maier
binary-label contract, and performs separate broad-training and E. coli-training
fingerprint and standardized-parent audits.

## Evidence Boundary

The code supports retrospective molecular ranking and uncertainty analysis. The Maier data are an external fixed-concentration phenotypic screen, not clinical MIC validation. Model scores do not establish prospective antimicrobial activity, experimental safety, pharmacokinetics, mechanism, patentability, or clinical efficacy.

No project license has been selected in this package. Add an author-approved license before public release.
