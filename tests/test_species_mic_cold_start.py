from argparse import Namespace

import pandas as pd
import pytest

from amr_multiview.species_mic_cold_start import (
    UnknownHeldOutPathogenClassError,
    _train_only_pathogen_features,
    run_cold_start,
)


def _cold_start_pairs() -> pd.DataFrame:
    common = [("COMMON1", "CCO"), ("COMMON2", "c1ccccc1")]
    unique = {
        "Species A": [("A1", "CCN"), ("A2", "CCC")],
        "Species B": [("B1", "CCCl"), ("B2", "CCBr")],
        "Species C": [("C1", "C1CCCCC1"), ("C2", "c1ccncc1")],
        "Species D": [("D1", "CC(=O)O"), ("D2", "CCS")],
    }
    classes = {
        "Species A": "gram_negative",
        "Species B": "gram_negative",
        "Species C": "gram_positive",
        "Species D": "gram_positive",
    }
    rows = []
    for species_index, species in enumerate(unique):
        for compound_index, (key, smiles) in enumerate(common + unique[species]):
            rows.append(
                {
                    "compound_inchikey": key,
                    "canonical_smiles": smiles,
                    "organism": species,
                    "tax_id": 1000 + species_index,
                    "pathogen_class": classes[species],
                    "log2_mic": float(species_index + compound_index),
                }
            )
    return pd.DataFrame(rows)


def test_train_only_pathogen_encoder_stops_on_unseen_class():
    with pytest.raises(UnknownHeldOutPathogenClassError, match="STOP_UNKNOWN"):
        _train_only_pathogen_features(pd.Series(["known"]), pd.Series(["unseen"]))


def test_cold_start_run_records_stop_when_held_out_class_is_unknown(tmp_path):
    pairs = _cold_start_pairs().loc[lambda frame: frame["organism"].isin(["Species A", "Species B", "Species C"])]
    data_path = tmp_path / "pairs.csv"
    output_dir = tmp_path / "stopped"
    pairs.to_csv(data_path, index=False)
    args = Namespace(
        data=str(data_path),
        output_dir=str(output_dir),
        expected_species=3,
        n_estimators=3,
        n_jobs=1,
        seed=7,
        max_folds=None,
        max_pairs_per_species=None,
    )

    with pytest.raises(UnknownHeldOutPathogenClassError, match="STOP_UNKNOWN"):
        run_cold_start(args)

    manifest = pd.read_json(output_dir / "manifest.json", typ="series")
    assert manifest["status"] == "STOP_UNKNOWN_HELD_OUT_PATHOGEN_CLASS"
    overlap = pd.read_csv(output_dir / "overlap_checks.csv")
    stopped = overlap.loc[overlap["held_out_species"].eq("Species C")].iloc[0]
    assert not stopped["pathogen_class_known_from_train"]


def test_small_cold_start_run_writes_paired_species_outputs(tmp_path):
    data_path = tmp_path / "pairs.csv.gz"
    output_dir = tmp_path / "results"
    _cold_start_pairs().to_csv(data_path, index=False, compression="gzip")
    args = Namespace(
        data=str(data_path),
        output_dir=str(output_dir),
        expected_species=4,
        n_estimators=3,
        n_jobs=1,
        seed=7,
        max_folds=2,
        max_pairs_per_species=None,
    )

    manifest = run_cold_start(args)

    assert manifest["status"] == "SMOKE_COMPLETE"
    predictions = pd.read_csv(output_dir / "predictions.csv.gz")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    paired = pd.read_csv(output_dir / "species_level_paired_deltas.csv")
    delta_summary = pd.read_csv(output_dir / "paired_delta_summary.csv")
    overlap = pd.read_csv(output_dir / "overlap_checks.csv")
    assert len(predictions) == 16
    assert len(metrics) == 12
    assert len(paired) == 6
    assert set(metrics["cohort"]) == {
        "all_pairs",
        "compound_seen_in_other_species",
        "exact_compound_nonoverlap",
    }
    assert (overlap["test_compounds_seen_in_other_species"] == 2).all()
    assert (overlap["test_compounds_exact_nonoverlap"] == 2).all()
    assert overlap["pathogen_class_known_from_train"].all()
    assert any(column.startswith("delta_mae_") for column in paired)
    assert not any("_ci95_" in column for column in delta_summary)
    assert {
        "equal_species_mean_delta_descriptive_stability_interval95_low",
        "equal_species_mean_delta_descriptive_stability_interval95_high",
    }.issubset(delta_summary.columns)
    assert "not an independent-sample confidence interval" in manifest["paired_delta_interval"]
