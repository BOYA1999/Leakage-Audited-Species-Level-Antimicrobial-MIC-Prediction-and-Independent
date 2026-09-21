from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from amr_multiview.species_mic import MIC_SQL
from amr_multiview.species_mic_temporal import (
    TEMPORAL_MIC_SQL,
    _species_counts,
    prepare_temporal_frames,
    run_temporal_from_raw,
)


def _temporal_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    species = {
        "Bacterium alpha": (101, "gram_negative"),
        "Bacterium beta": (102, "gram_positive"),
    }
    compounds = [
        ("TA1", "CCO", "Bacterium alpha", 2017, 1.0),
        ("TA2", "CCN", "Bacterium alpha", 2018, 2.0),
        ("TB1", "CCC", "Bacterium beta", 2017, 4.0),
        ("TB2", "CCCl", "Bacterium beta", 2018, 8.0),
        ("VA1", "CCBr", "Bacterium alpha", 2019, 2.0),
        ("VB1", "CCS", "Bacterium beta", 2020, 4.0),
        ("XA1", "C1CCCCC1", "Bacterium alpha", 2021, 8.0),
        ("XB1", "c1ccccc1", "Bacterium beta", 2022, 16.0),
    ]
    rows = []
    for activity_id, (key, smiles, organism, year, mic) in enumerate(compounds, start=1):
        tax_id, _ = species[organism]
        rows.append(
            {
                "compound_inchikey": key,
                "compound_smiles": smiles,
                "mic_ug_ml": mic,
                "raw_organism": organism,
                "source_tax_id": tax_id,
                "strain": "strain-1",
                "assay_chembl_id": f"A{activity_id}",
                "activity_id": activity_id,
                "data_validity_comment": None,
                "doc_id": activity_id,
                "document_year": year,
            }
        )
    rows.extend(
        [
            {
                **rows[0],
                "activity_id": 20,
                "assay_chembl_id": "A20",
                "doc_id": 20,
                "document_year": 2022,
                "mic_ug_ml": 1024.0,
            },
            {
                **rows[4],
                "activity_id": 21,
                "assay_chembl_id": "A21",
                "doc_id": 21,
                "document_year": 2021,
                "mic_ug_ml": 512.0,
            },
            {
                **rows[0],
                "compound_inchikey": "MISSING_YEAR",
                "compound_smiles": "CC(=O)O",
                "activity_id": 22,
                "assay_chembl_id": "A22",
                "doc_id": 22,
                "document_year": np.nan,
            },
        ]
    )
    mapping = {organism: pathogen_class for organism, (_, pathogen_class) in species.items()}
    taxonomy = pd.DataFrame(
        [
            {"tax_id": tax_id, "species_tax_id": tax_id, "species_name": organism}
            for organism, (tax_id, _) in species.items()
        ]
    )
    return pd.DataFrame(rows), mapping, taxonomy


def test_temporal_sql_preserves_mic_filter_and_adds_document_year():
    where_clause = MIC_SQL[MIC_SQL.index("WHERE") :].strip()
    assert where_clause in TEMPORAL_MIC_SQL
    assert "a.doc_id" in TEMPORAL_MIC_SQL
    assert "d.year AS document_year" in TEMPORAL_MIC_SQL
    assert "LEFT JOIN docs d ON a.doc_id = d.doc_id" in TEMPORAL_MIC_SQL


def test_temporal_preparation_splits_before_aggregation_and_excludes_missing_year():
    raw, mapping, taxonomy = _temporal_inputs()
    frames = prepare_temporal_frames(raw, mapping, taxonomy, min_compounds=2)

    assignments = frames["compound_cohorts"].set_index("compound_inchikey")
    assert assignments.loc["TA1", "cohort"] == "train"
    assert assignments.loc["VA1", "cohort"] == "valid"
    assert assignments.loc["XA1", "cohort"] == "test"
    assert assignments["cohort"].isin(["train", "valid", "test"]).all()
    assert len(frames["missing_year_rows"]) == 1
    assert frames["raw_audit"]["missing_document_year_rows_excluded"] == 1
    assert frames["raw_audit"]["measurement_rows_excluded_outside_assigned_window"] == 2

    pairs = frames["pairs"].set_index(["cohort", "compound_inchikey", "organism"])
    assert pairs.loc[("train", "TA1", "Bacterium alpha"), "log2_mic"] == 0.0
    assert pairs.loc[("train", "TA1", "Bacterium alpha"), "measurement_year_max"] == 2017
    assert pairs.loc[("valid", "VA1", "Bacterium alpha"), "log2_mic"] == 1.0
    assert pairs.loc[("valid", "VA1", "Bacterium alpha"), "measurement_year_max"] == 2019

    counts = frames["species_counts"]
    train_counts = counts[counts["cohort"].eq("train")]
    assert (train_counts["compounds"] >= 2).all()
    audit = frames["overlap_audit"]
    assert audit["exact_compound_overlap_zero"]
    assert audit["compound_overlap_train_valid"] == 0
    assert audit["compound_overlap_train_test"] == 0
    assert audit["compound_overlap_valid_test"] == 0
    assert audit["train_first_year_max_le_2018"]
    assert audit["valid_first_year_2019_2020"]
    assert audit["test_first_year_2021_2023"]
    assert audit["train_measurements_max_2018"]
    assert audit["valid_measurements_2019_2020"]
    assert audit["test_measurements_2021_2023"]


def test_training_era_species_threshold_is_strictly_500_compounds():
    rows = []
    for organism, count, tax_id in (("Eligible species", 500, 1), ("Ineligible species", 499, 2)):
        for index in range(count):
            rows.append(
                {
                    "cohort": "train",
                    "organism": organism,
                    "pathogen_class": "class",
                    "tax_id": tax_id,
                    "compound_inchikey": f"{organism}-{index}",
                }
            )
    pairs = pd.DataFrame(rows)
    counts, eligible = _species_counts(pairs, min_compounds=500)
    assert eligible == ["Eligible species"]
    assert set(counts["organism"]) == {"Eligible species"}
    assert counts.loc[counts["cohort"].eq("train"), "compounds"].item() == 500


def test_taxid_ambiguity_is_checked_only_after_training_era_eligibility():
    rows = []
    for index in range(500):
        rows.append(
            {
                "cohort": "train",
                "organism": "Eligible species",
                "pathogen_class": "class",
                "tax_id": 1,
                "compound_inchikey": f"eligible-{index}",
            }
        )
    rows.extend(
        [
            {
                "cohort": "train",
                "organism": "Excluded ambiguous species",
                "pathogen_class": "class",
                "tax_id": tax_id,
                "compound_inchikey": f"excluded-{tax_id}",
            }
            for tax_id in (2, 3)
        ]
    )
    counts, eligible = _species_counts(pd.DataFrame(rows), min_compounds=500)
    assert eligible == ["Eligible species"]
    assert set(counts["organism"]) == {"Eligible species"}

    eligible_ambiguous = pd.DataFrame(rows + [
        {
            "cohort": "test",
            "organism": "Eligible species",
            "pathogen_class": "class",
            "tax_id": 9,
            "compound_inchikey": "eligible-future",
        }
    ])
    with pytest.raises(ValueError, match="Eligible canonical species names"):
        _species_counts(eligible_ambiguous, min_compounds=500)


def test_small_temporal_run_writes_complete_audited_outputs(tmp_path):
    raw, mapping, taxonomy = _temporal_inputs()
    output_dir = tmp_path / "temporal"
    manifest = run_temporal_from_raw(
        raw,
        mapping,
        taxonomy,
        output_dir,
        min_compounds=2,
        n_estimators=3,
        n_jobs=1,
    )

    assert manifest["status"] == "COMPLETE"
    assert manifest["eligible_species"] == 2
    for name in manifest["outputs"]:
        assert (output_dir / name).exists(), name
    saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["cohorts"]["test"].startswith("first eligible MIC year 2021-2023")

    predictions = pd.read_csv(output_dir / "predictions.csv.gz")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    runtime = pd.read_csv(output_dir / "runtime.csv")
    overlap = json.loads((output_dir / "overlap_audit.json").read_text(encoding="utf-8"))
    assert set(predictions["model"]) == {"morgan", "multiview"}
    assert set(metrics["scope"]) == {"species", "pooled", "macro"}
    assert {
        "empirical_conformal_coverage_90",
        "empirical_conformal_width_90",
        "calibration_q_90",
    }.issubset(metrics.columns)
    assert predictions["covered_90"].isin([True, False]).all()
    assert {"database_query", "temporal_preparation", "feature_build", "fit", "predict_valid", "predict_test"}.issubset(
        set(runtime["phase"])
    )
    assert overlap["temporal_contract_pass"]
