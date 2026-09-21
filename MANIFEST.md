# Public Package Manifest

This directory is built from a whitelist.

Included:

- `amr_multiview/__init__.py`
- `amr_multiview/data.py`
- `amr_multiview/features.py`
- `amr_multiview/species_mic.py`
- `amr_multiview/species_mic_experiment.py`
- `amr_multiview/molformer_embeddings.py`
- `amr_multiview/maier_external_validation.py`
- `amr_multiview/joint_external_validation.py`
- `amr_multiview/species_mic_jcim_baselines.py`
- `amr_multiview/species_mic_chemprop.py`
- `amr_multiview/species_mic_temporal.py`
- `amr_multiview/species_mic_cold_start.py`
- `amr_multiview/species_mic_chemical_space.py`
- `scripts/analyze_species_mic_results.py`
- `scripts/make_species_mic_figures.py`
- `scripts/analyze_jcim_revision.py`
- `scripts/make_jcim_model_benchmark_figure.py`
- `scripts/make_jcim_reliability_figure.py`
- `scripts/make_jcim_revision_figures_3_5.py`
- `scripts/make_jcim_toc_graphic.py`
- `tests/test_species_mic.py`
- `tests/test_revision_statistics.py`
- `tests/test_joint_external_validation.py`
- `tests/test_species_mic_jcim_baselines.py`
- `tests/test_species_mic_chemprop.py`
- `tests/test_species_mic_temporal.py`
- `tests/test_species_mic_cold_start.py`
- `tests/test_species_mic_chemical_space.py`
- `docs/DATA.md`
- `docs/EXPERIMENT_CONTRACT.md`
- `docs/taxonomy_mapping.csv`
- `docs/maier_taxonomy_mapping.csv`
- `docs/JOINT_EXTERNAL.md`
- `docs/joint_external_config.example.json`
- `docs/JCIM_REVISION_AUDITS.md`
- `README.md`, `MANIFEST.md`, `LICENSE`, `requirements.txt`, `pyproject.toml`, and `.gitignore`

Excluded:

- manuscripts, rendered figures, result tables, raw data, predictions, and local run manifests;
- clinical or hospital files;
- model weights, embeddings, checkpoints, archives, databases, and caches;
- author names, affiliations, funding statements, credentials, and absolute local paths;
- `.git` history and archives.

The author-owned repository software is distributed under the included MIT `LICENSE`. Third-party datasets, pretrained weights and source-derived Supplementary Data are excluded from that licence.

Evidence-revision additions: `scripts/audit_cbc_source.py`, `scripts/audit_maier_raw_labels.py`, `scripts/replay_cbc_controls.py`, `scripts/make_cbc_revision_figures.py`, `scripts/cbc_academic.mplstyle`, `docs/CBC_EVIDENCE.md`, and `docs/RF_FEASIBILITY_AMENDMENT.md`. `SHA256SUMS.csv` is generated for the submitted archive. The matching public code repository is https://github.com/BOYA1999/Leakage-Audited-Species-Level-Antimicrobial-MIC-Prediction-and-Independent.
