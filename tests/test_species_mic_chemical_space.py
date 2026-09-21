from argparse import Namespace

import pandas as pd
import pytest

from amr_multiview.species_mic_chemical_space import (
    apparent_mic_cliffs,
    build_scaffold_table,
    cliff_error_analysis,
    morgan_fingerprints,
    run_chemical_space,
    species_scaffold_diversity,
)
from amr_multiview.species_mic_experiment import scaffold_partition


def test_scaffold_diversity_and_cliffs_exclude_acyclic_scaffolds():
    pairs = pd.DataFrame(
        [
            {"compound_inchikey": "RING_A", "canonical_smiles": "c1ccccc1", "organism": "Species A", "log2_mic": 0.0},
            {"compound_inchikey": "RING_B", "canonical_smiles": "c1ccccc1", "organism": "Species A", "log2_mic": 3.0},
            {"compound_inchikey": "CHAIN", "canonical_smiles": "CCO", "organism": "Species A", "log2_mic": 4.0},
        ]
    )
    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    scaffolds = build_scaffold_table(compounds)
    diversity, sensitivity = species_scaffold_diversity(pairs, scaffolds)
    all_mode = diversity.loc[
        diversity["assignment_mode"].eq("compound_specific_acyclic_fallback")
    ].iloc[0]
    cyclic = diversity.loc[diversity["assignment_mode"].eq("nonempty_murcko_only")].iloc[0]
    assert all_mode["unique_scaffold_fraction"] == 2 / 3
    assert all_mode["largest_scaffold_share"] == 2 / 3
    assert cyclic["unique_scaffold_fraction"] == 1 / 2
    assert len(sensitivity) == 1

    edges, scaffold_summary, species_summary, counts = apparent_mic_cliffs(
        pairs, scaffolds, morgan_fingerprints(scaffolds)
    )
    assert len(edges) == 1
    assert edges.iloc[0]["primary_sim0.8_delta2"]
    assert edges.iloc[0]["sensitivity_sim0.8_delta3"]
    assert (scaffold_summary["murcko_scaffold"] != "").all()
    assert species_summary["n_apparent_cliff_edges_descriptive_only"].max() == 1
    assert counts == {"eligible_groups_total": 1, "eligible_groups_analyzed": 1}


def test_cliff_threshold_is_inclusive_under_float_roundoff():
    pairs = pd.DataFrame(
        [
            {"compound_inchikey": "RING_A", "canonical_smiles": "c1ccccc1", "organism": "Species A", "log2_mic": 0.0},
            {
                "compound_inchikey": "RING_B",
                "canonical_smiles": "c1ccccc1",
                "organism": "Species A",
                "log2_mic": 1.9999999999999996,
            },
        ]
    )
    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    scaffolds = build_scaffold_table(compounds)

    edges, _, _, _ = apparent_mic_cliffs(pairs, scaffolds, morgan_fingerprints(scaffolds))

    assert len(edges) == 1
    assert edges.iloc[0]["primary_sim0.8_delta2"]


def test_cliff_error_uses_unique_records_and_equal_species_seed_summaries():
    edge_rows = [
        ("Species A", "A1", "A2"),
        ("Species A", "A1", "A4"),
        ("Species B", "B1", "B2"),
        ("Species C", "C1", "C2"),
    ]
    edges = pd.DataFrame(edge_rows, columns=["organism", "compound_a", "compound_b"])
    edges["primary_sim0.8_delta2"] = True
    edges["sensitivity_sim0.7_delta2"] = True
    edges["sensitivity_sim0.9_delta2"] = False
    edges["sensitivity_sim0.8_delta3"] = False
    errors = {
        42: {
            "A1": 1.0, "A2": 3.0, "A3": 1.0, "A4": 2.0,
            "B1": 2.0, "B2": 4.0, "B3": 1.0,
            "C1": 2.0, "C2": 2.0,
        },
        100: {
            "A1": 2.0, "A2": 2.0, "A3": 2.0, "A4": 2.0,
            "B1": 1.0, "B2": 1.0, "B3": 3.0,
            "C1": 2.0, "C2": 2.0,
        },
    }
    organism = {key: f"Species {key[0]}" for key in errors[42]}
    predictions = pd.DataFrame(
        [
            {
                "compound_inchikey": compound,
                "organism": organism[compound],
                "seed": seed,
                "model": "morgan",
                "y_true": 0.0,
                "y_pred": error,
            }
            for seed, seed_errors in errors.items()
            for compound, error in seed_errors.items()
        ]
    )

    result = cliff_error_analysis(predictions, edges)
    by_species = result["by_species"]
    primary = by_species.loc[by_species["configuration"].eq("primary_sim0.8_delta2")]
    assert len(primary) == 4
    assert "Species C" not in set(primary["organism"])
    species_a_42 = primary.loc[
        primary["seed"].eq(42) & primary["organism"].eq("Species A")
    ].iloc[0]
    assert species_a_42["n_cliff_participant_records"] == 3
    assert species_a_42["n_nonparticipant_records"] == 1
    assert species_a_42["cliff_participant_mae"] == pytest.approx(2.0)
    assert species_a_42["nonparticipant_mae"] == pytest.approx(1.0)

    summary = result["summary"]
    primary_summary = summary.loc[
        summary["configuration"].eq("primary_sim0.8_delta2")
        & summary["model"].eq("morgan")
    ]
    seed_42 = primary_summary.loc[
        primary_summary["summary_level"].eq("seed_equal_species_mean")
        & primary_summary["seed"].eq(42)
    ].iloc[0]
    assert seed_42["n_species_equal_weighted"] == 2
    assert seed_42["cliff_participant_mae_equal_species"] == pytest.approx(2.5)
    assert seed_42["nonparticipant_mae_equal_species"] == pytest.approx(1.0)
    assert seed_42["delta_mae_cliff_minus_nonparticipant_equal_species"] == pytest.approx(1.5)
    stability = primary_summary.loc[
        primary_summary["summary_level"].eq("across_seed_descriptive_stability")
    ].iloc[0]
    assert stability["n_seeds"] == 2
    assert stability["cliff_participant_mae_equal_species"] == pytest.approx(2.0)
    assert stability["nonparticipant_mae_equal_species"] == pytest.approx(1.75)
    assert stability["delta_mae_cliff_minus_nonparticipant_equal_species"] == pytest.approx(0.25)
    assert stability["delta_mae_seed_sd"] == pytest.approx(1.7677669529663689)
    assert "not independent" in stability["dependence_note"]


def test_small_chemical_space_run_reconstructs_existing_split(tmp_path):
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "CCO",
        "CCN",
        "CCC",
        "CCCC",
        "CCCl",
        "CCBr",
    ]
    pairs = pd.DataFrame(
        {
            "compound_inchikey": [f"KEY{i:02d}" for i in range(len(smiles))],
            "canonical_smiles": smiles,
            "organism": ["Species A"] * len(smiles),
            "log2_mic": [float(i % 5) for i in range(len(smiles))],
        }
    )
    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    partition, _ = scaffold_partition(compounds, 42)
    test_keys = set(compounds.loc[partition == "test", "compound_inchikey"])
    predictions = pairs.loc[pairs["compound_inchikey"].isin(test_keys), ["compound_inchikey", "organism", "log2_mic"]].copy()
    predictions["seed"] = 42
    predictions["model"] = "morgan"
    predictions["y_true"] = predictions.pop("log2_mic")
    predictions["y_pred"] = predictions["y_true"] + 0.5

    data_path = tmp_path / "pairs.csv.gz"
    prediction_path = tmp_path / "predictions.csv.gz"
    output_dir = tmp_path / "chemical_space"
    pairs.to_csv(data_path, index=False, compression="gzip")
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    args = Namespace(
        data=str(data_path),
        predictions=str(prediction_path),
        output_dir=str(output_dir),
        seeds=[42],
        models=["morgan"],
        max_test_compounds=None,
        max_cliff_groups=None,
    )

    manifest = run_chemical_space(args)

    assert manifest["status"] == "COMPLETE"
    checks = pd.read_csv(output_dir / "split_overlap_checks.csv")
    nearest = pd.read_csv(output_dir / "nearest_train_similarity.csv.gz")
    compound_summary = pd.read_csv(output_dir / "similarity_error_by_compound.csv.gz")
    assert checks.iloc[0]["prediction_split_match"]
    assert checks.iloc[0]["missing_expected_test_pairs"] == 0
    assert checks.iloc[0]["unexpected_prediction_pairs"] == 0
    assert len(nearest) == len(test_keys)
    assert len(compound_summary) == len(test_keys)
    assert (output_dir / "apparent_mic_cliff_scaffold_summary.csv").exists()
    assert (output_dir / "cliff_error_by_species.csv").exists()
    assert (output_dir / "cliff_error_summary.csv").exists()
    assert "cliff_error" in manifest
    assert "edge rows are never inference units" in manifest["cliff_error"]["dependence"]
    assert (output_dir / "manifest.json").exists()


def test_split_gate_rejects_missing_species_row_when_compound_set_matches(tmp_path):
    smiles = [
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "CCO",
        "CCN",
        "CCC",
        "CCCC",
        "CCCl",
        "CCBr",
    ]
    compounds = pd.DataFrame(
        {
            "compound_inchikey": [f"KEY{i:02d}" for i in range(len(smiles))],
            "canonical_smiles": smiles,
        }
    )
    pairs = compounds.merge(pd.DataFrame({"organism": ["Species A", "Species B"]}), how="cross")
    pairs["log2_mic"] = range(len(pairs))
    partition, _ = scaffold_partition(compounds, 42)
    test_keys = set(compounds.loc[partition == "test", "compound_inchikey"])
    predictions = pairs.loc[
        pairs["compound_inchikey"].isin(test_keys),
        ["compound_inchikey", "organism", "log2_mic"],
    ].copy()
    missing_key = sorted(test_keys)[0]
    predictions = predictions.loc[
        ~(
            predictions["compound_inchikey"].eq(missing_key)
            & predictions["organism"].eq("Species B")
        )
    ]
    predictions["seed"] = 42
    predictions["model"] = "morgan"
    predictions["y_true"] = predictions.pop("log2_mic")
    predictions["y_pred"] = predictions["y_true"]

    data_path = tmp_path / "pairs.csv.gz"
    prediction_path = tmp_path / "predictions.csv.gz"
    output_dir = tmp_path / "chemical_space"
    pairs.to_csv(data_path, index=False, compression="gzip")
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    args = Namespace(
        data=str(data_path),
        predictions=str(prediction_path),
        output_dir=str(output_dir),
        seeds=[42],
        models=["morgan"],
        max_test_compounds=None,
        max_cliff_groups=None,
    )

    with pytest.raises(ValueError, match="STOP_SPLIT_RECONSTRUCTION_MISMATCH"):
        run_chemical_space(args)

    checks = pd.read_csv(output_dir / "split_overlap_checks.csv")
    assert checks.iloc[0]["missing_expected_test_compounds"] == 0
    assert checks.iloc[0]["missing_expected_test_pairs"] == 1
    assert not checks.iloc[0]["prediction_split_match"]
