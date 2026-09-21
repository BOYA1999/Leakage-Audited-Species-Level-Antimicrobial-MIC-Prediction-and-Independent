from __future__ import annotations

import argparse
from hashlib import sha256
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr, t as student_t

from .species_mic_experiment import scaffold_partition


RDLogger.DisableLog("rdApp.*")

REQUIRED_PAIR_COLUMNS = {
    "compound_inchikey",
    "canonical_smiles",
    "organism",
    "log2_mic",
}
REQUIRED_PREDICTION_COLUMNS = {
    "compound_inchikey",
    "organism",
    "seed",
    "model",
    "y_true",
    "y_pred",
}
CLIFF_CONFIGS = {
    "primary_sim0.8_delta2": (0.8, 2.0),
    "sensitivity_sim0.7_delta2": (0.7, 2.0),
    "sensitivity_sim0.9_delta2": (0.9, 2.0),
    "sensitivity_sim0.8_delta3": (0.8, 3.0),
}
THRESHOLD_EPSILON = 1e-12


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(pairs: pd.DataFrame, predictions: pd.DataFrame) -> None:
    missing_pairs = sorted(REQUIRED_PAIR_COLUMNS - set(pairs.columns))
    missing_predictions = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing_pairs:
        raise ValueError(f"Pair data are missing: {', '.join(missing_pairs)}")
    if missing_predictions:
        raise ValueError(f"Predictions are missing: {', '.join(missing_predictions)}")
    if pairs[list(REQUIRED_PAIR_COLUMNS)].isna().any().any():
        raise ValueError("Required chemical-space fields contain missing values.")
    if pairs.duplicated(["compound_inchikey", "organism"]).any():
        raise ValueError("Pair data must contain one aggregated row per compound-species pair.")
    if predictions[list(REQUIRED_PREDICTION_COLUMNS)].isna().any().any():
        raise ValueError("Required prediction fields contain missing values.")
    smiles_per_key = pairs.groupby("compound_inchikey")["canonical_smiles"].nunique()
    if (smiles_per_key > 1).any():
        raise ValueError("At least one compound key maps to multiple canonical SMILES.")
    if predictions.duplicated(["model", "seed", "compound_inchikey", "organism"]).any():
        raise ValueError("Predictions contain duplicate model-seed-compound-species rows.")
    reference = pairs[["compound_inchikey", "organism", "log2_mic"]]
    aligned = predictions.merge(
        reference,
        on=["compound_inchikey", "organism"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Predictions contain compound-species pairs absent from the pair data.")
    if not np.allclose(aligned["y_true"], aligned["log2_mic"], rtol=0, atol=1e-6):
        raise ValueError("Prediction y_true values do not match the eligible pair data.")


def build_scaffold_table(compounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for compound_index, row in enumerate(compounds.itertuples(index=False)):
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES for {row.compound_inchikey}.")
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True) if scaffold and scaffold.GetNumAtoms() else ""
        is_acyclic = not scaffold_smiles
        rows.append(
            {
                "compound_index": compound_index,
                "compound_inchikey": row.compound_inchikey,
                "canonical_smiles": row.canonical_smiles,
                "murcko_scaffold": scaffold_smiles,
                "assigned_scaffold": scaffold_smiles or f"ACYCLIC:{row.compound_inchikey}",
                "is_acyclic": is_acyclic,
            }
        )
    return pd.DataFrame(rows)


def _diversity_metrics(frame: pd.DataFrame, scaffold_column: str) -> dict[str, float | int]:
    n_compounds = len(frame)
    if n_compounds == 0:
        return {
            "unique_compounds": 0,
            "unique_scaffolds": 0,
            "unique_scaffold_fraction": np.nan,
            "scaffold_entropy_nats": np.nan,
            "normalized_scaffold_entropy": np.nan,
            "largest_scaffold_share": np.nan,
        }
    counts = frame[scaffold_column].value_counts()
    probabilities = counts.to_numpy(dtype=float) / n_compounds
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalized = entropy / np.log(len(counts)) if len(counts) > 1 else 0.0
    return {
        "unique_compounds": n_compounds,
        "unique_scaffolds": len(counts),
        "unique_scaffold_fraction": len(counts) / n_compounds,
        "scaffold_entropy_nats": entropy,
        "normalized_scaffold_entropy": float(normalized),
        "largest_scaffold_share": float(counts.iloc[0] / n_compounds),
    }


def species_scaffold_diversity(
    pairs: pd.DataFrame,
    scaffold_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated = pairs[["compound_inchikey", "organism"]].merge(
        scaffold_table[["compound_inchikey", "murcko_scaffold", "assigned_scaffold", "is_acyclic"]],
        on="compound_inchikey",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for organism, group in annotated.groupby("organism", sort=True):
        group = group.drop_duplicates("compound_inchikey")
        acyclic_count = int(group["is_acyclic"].sum())
        for mode, subset, scaffold_column in [
            ("compound_specific_acyclic_fallback", group, "assigned_scaffold"),
            ("nonempty_murcko_only", group.loc[~group["is_acyclic"]], "murcko_scaffold"),
        ]:
            row = _diversity_metrics(subset, scaffold_column)
            row.update(
                {
                    "organism": organism,
                    "assignment_mode": mode,
                    "all_species_compounds": len(group),
                    "acyclic_compounds": acyclic_count,
                    "acyclic_fraction": acyclic_count / len(group) if len(group) else np.nan,
                }
            )
            rows.append(row)
    diversity = pd.DataFrame(rows)
    all_mode = diversity.loc[
        diversity["assignment_mode"].eq("compound_specific_acyclic_fallback")
    ].drop(columns="assignment_mode")
    cyclic_mode = diversity.loc[diversity["assignment_mode"].eq("nonempty_murcko_only")].drop(
        columns=["assignment_mode", "all_species_compounds", "acyclic_compounds", "acyclic_fraction"]
    )
    sensitivity = all_mode.merge(cyclic_mode, on="organism", suffixes=("_with_fallback", "_cyclic_only"))
    for metric in [
        "unique_scaffold_fraction",
        "scaffold_entropy_nats",
        "normalized_scaffold_entropy",
        "largest_scaffold_share",
    ]:
        sensitivity[f"delta_{metric}_with_fallback_minus_cyclic_only"] = (
            sensitivity[f"{metric}_with_fallback"] - sensitivity[f"{metric}_cyclic_only"]
        )
    return diversity, sensitivity


def morgan_fingerprints(scaffold_table: pd.DataFrame) -> list:
    fingerprints = []
    for row in scaffold_table.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fingerprints


def nearest_train_similarities(
    pairs: pd.DataFrame,
    compounds: pd.DataFrame,
    scaffold_table: pd.DataFrame,
    fingerprints: list,
    predictions: pd.DataFrame,
    seeds: list[int],
    max_test_compounds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_test_compounds is not None and max_test_compounds < 1:
        raise ValueError("max_test_compounds must be positive.")
    rows = []
    checks = []
    for seed in seeds:
        partition, report = scaffold_partition(compounds, seed)
        train_indices = np.flatnonzero(partition == "train")
        test_indices = np.flatnonzero(partition == "test")
        expected_test = set(compounds.iloc[test_indices]["compound_inchikey"].astype(str))
        expected_test_pairs = set(
            pairs.loc[
                pairs["compound_inchikey"].astype(str).isin(expected_test),
                ["compound_inchikey", "organism"],
            ]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        split_matches = True
        for model in sorted(predictions["model"].astype(str).unique()):
            model_predictions = predictions.loc[
                predictions["seed"].eq(seed) & predictions["model"].astype(str).eq(model)
            ]
            observed_test = set(model_predictions["compound_inchikey"].astype(str))
            observed_test_pairs = set(
                model_predictions[["compound_inchikey", "organism"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
            missing = expected_test - observed_test
            unexpected = observed_test - expected_test
            missing_pairs = expected_test_pairs - observed_test_pairs
            unexpected_pairs = observed_test_pairs - expected_test_pairs
            checks.append(
                {
                    "seed": int(seed),
                    "model": model,
                    "train_compounds": len(train_indices),
                    "expected_test_compounds": len(expected_test),
                    "observed_prediction_compounds": len(observed_test),
                    "missing_expected_test_compounds": len(missing),
                    "unexpected_prediction_compounds": len(unexpected),
                    "expected_test_pairs": len(expected_test_pairs),
                    "observed_prediction_pairs": len(observed_test_pairs),
                    "missing_expected_test_pairs": len(missing_pairs),
                    "unexpected_prediction_pairs": len(unexpected_pairs),
                    "scaffold_overlap_train_test": report["scaffold_overlap_train_test"],
                    "prediction_split_match": not missing
                    and not unexpected
                    and not missing_pairs
                    and not unexpected_pairs,
                }
            )
            if missing or unexpected or missing_pairs or unexpected_pairs:
                split_matches = False
        if not split_matches:
            continue
        if max_test_compounds is not None:
            test_indices = test_indices[:max_test_compounds]
        train_fingerprints = [fingerprints[index] for index in train_indices]
        train_keys = compounds.iloc[train_indices]["compound_inchikey"].astype(str).tolist()
        for test_index in test_indices:
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[test_index], train_fingerprints
            )
            nearest_position = int(np.argmax(similarities))
            scaffold_row = scaffold_table.iloc[test_index]
            rows.append(
                {
                    "seed": int(seed),
                    "compound_inchikey": compounds.iloc[test_index]["compound_inchikey"],
                    "nearest_train_inchikey": train_keys[nearest_position],
                    "nearest_train_tanimoto": float(similarities[nearest_position]),
                    "murcko_scaffold": scaffold_row["murcko_scaffold"],
                    "assigned_scaffold": scaffold_row["assigned_scaffold"],
                    "is_acyclic": bool(scaffold_row["is_acyclic"]),
                }
            )
        for check in checks:
            if check["seed"] == int(seed):
                check["analyzed_test_compounds"] = len(test_indices)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else np.nan


def similarity_error_analysis(
    predictions: pd.DataFrame,
    nearest: pd.DataFrame,
    models: list[str],
) -> dict[str, pd.DataFrame]:
    selected = predictions.loc[predictions["model"].astype(str).isin(models)].merge(
        nearest,
        on=["seed", "compound_inchikey"],
        how="inner",
        validate="many_to_one",
    )
    selected["absolute_error"] = (selected["y_pred"] - selected["y_true"]).abs()
    by_compound = (
        selected.groupby(
            [
                "model",
                "seed",
                "compound_inchikey",
                "nearest_train_inchikey",
                "nearest_train_tanimoto",
                "murcko_scaffold",
                "assigned_scaffold",
                "is_acyclic",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_species=("organism", "nunique"),
            absolute_error_mean_across_species=("absolute_error", "mean"),
            absolute_error_median_across_species=("absolute_error", "median"),
        )
    )
    by_scaffold = (
        by_compound.groupby(
            ["model", "seed", "assigned_scaffold", "is_acyclic"],
            as_index=False,
        )
        .agg(
            n_compounds=("compound_inchikey", "nunique"),
            nearest_train_tanimoto_median=("nearest_train_tanimoto", "median"),
            absolute_error_mean_across_compounds=("absolute_error_mean_across_species", "mean"),
            absolute_error_median_across_compounds=("absolute_error_mean_across_species", "median"),
        )
    )

    summary_rows = []
    for unit, frame, similarity_column, error_column in [
        (
            "equal_compound",
            by_compound,
            "nearest_train_tanimoto",
            "absolute_error_mean_across_species",
        ),
        (
            "equal_scaffold",
            by_scaffold,
            "nearest_train_tanimoto_median",
            "absolute_error_mean_across_compounds",
        ),
    ]:
        for (model, seed), group in frame.groupby(["model", "seed"], sort=True):
            for scope, subset in [
                ("all_with_compound_specific_acyclic_fallback", group),
                ("nonempty_murcko_only", group.loc[~group["is_acyclic"]]),
            ]:
                summary_rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "scope": scope,
                        "inference_unit": unit,
                        "n_units": len(subset),
                        "mean_absolute_error": float(subset[error_column].mean()) if len(subset) else np.nan,
                        "median_absolute_error": float(subset[error_column].median()) if len(subset) else np.nan,
                        "spearman_similarity_absolute_error": _safe_spearman(
                            subset[similarity_column], subset[error_column]
                        ),
                    }
                )

    species_rows = []
    for (model, seed, organism), group in selected.groupby(["model", "seed", "organism"], sort=True):
        for scope, subset in [
            ("all_with_compound_specific_acyclic_fallback", group),
            ("nonempty_murcko_only", group.loc[~group["is_acyclic"]]),
        ]:
            species_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "organism": organism,
                    "scope": scope,
                    "inference_unit": "compound_within_species",
                    "n_compounds": subset["compound_inchikey"].nunique(),
                    "mean_absolute_error": float(subset["absolute_error"].mean()) if len(subset) else np.nan,
                    "spearman_similarity_absolute_error": _safe_spearman(
                        subset["nearest_train_tanimoto"], subset["absolute_error"]
                    ),
                }
            )

    bins = [-1e-12, 0.25, 0.50, 0.75, 0.90, 1.0 + 1e-12]
    labels = ["[0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,0.90]", "(0.90,1.00]"]
    binned = by_compound.copy()
    binned["similarity_bin"] = pd.cut(
        binned["nearest_train_tanimoto"], bins=bins, labels=labels, include_lowest=True
    )
    bin_summary = (
        binned.groupby(["model", "seed", "similarity_bin"], observed=True, as_index=False)
        .agg(
            n_compounds=("compound_inchikey", "nunique"),
            mean_absolute_error=("absolute_error_mean_across_species", "mean"),
            median_absolute_error=("absolute_error_mean_across_species", "median"),
        )
    )
    return {
        "by_compound": by_compound,
        "by_scaffold": by_scaffold,
        "summary": pd.DataFrame(summary_rows),
        "species_summary": pd.DataFrame(species_rows),
        "bins": bin_summary,
    }


def apparent_mic_cliffs(
    pairs: pd.DataFrame,
    scaffold_table: pd.DataFrame,
    fingerprints: list,
    max_groups: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be positive.")
    annotated = pairs[["compound_inchikey", "organism", "log2_mic"]].merge(
        scaffold_table[["compound_index", "compound_inchikey", "murcko_scaffold", "is_acyclic"]],
        on="compound_inchikey",
        how="left",
        validate="many_to_one",
    )
    annotated = annotated.loc[~annotated["is_acyclic"]].copy()
    grouped = [
        (key, group.sort_values("compound_inchikey"))
        for key, group in annotated.groupby(["organism", "murcko_scaffold"], sort=True)
        if group["compound_inchikey"].nunique() >= 2
    ]
    total_groups = len(grouped)
    if max_groups is not None:
        grouped = grouped[:max_groups]

    edge_rows = []
    for (organism, scaffold), group in grouped:
        records = list(group.itertuples(index=False))
        for left_position, left in enumerate(records[:-1]):
            right_records = records[left_position + 1 :]
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[int(left.compound_index)],
                [fingerprints[int(right.compound_index)] for right in right_records],
            )
            for right, similarity in zip(right_records, similarities):
                delta = abs(float(left.log2_mic) - float(right.log2_mic))
                if similarity + THRESHOLD_EPSILON < 0.7 or delta + THRESHOLD_EPSILON < 2.0:
                    continue
                row = {
                    "organism": organism,
                    "murcko_scaffold": scaffold,
                    "compound_a": left.compound_inchikey,
                    "compound_b": right.compound_inchikey,
                    "log2_mic_a": float(left.log2_mic),
                    "log2_mic_b": float(right.log2_mic),
                    "absolute_log2_mic_delta": delta,
                    "morgan_tanimoto": float(similarity),
                }
                for name, (minimum_similarity, minimum_delta) in CLIFF_CONFIGS.items():
                    row[name] = bool(
                        similarity + THRESHOLD_EPSILON >= minimum_similarity
                        and delta + THRESHOLD_EPSILON >= minimum_delta
                    )
                edge_rows.append(row)

    edge_columns = [
        "organism",
        "murcko_scaffold",
        "compound_a",
        "compound_b",
        "log2_mic_a",
        "log2_mic_b",
        "absolute_log2_mic_delta",
        "morgan_tanimoto",
        *CLIFF_CONFIGS,
    ]
    edges = pd.DataFrame(edge_rows, columns=edge_columns)
    scaffold_rows = []
    for (organism, scaffold), group in grouped:
        compounds = set(group["compound_inchikey"].astype(str))
        group_edges = edges.loc[
            edges["organism"].eq(organism) & edges["murcko_scaffold"].eq(scaffold)
        ]
        for config, (minimum_similarity, minimum_delta) in CLIFF_CONFIGS.items():
            selected = group_edges.loc[group_edges[config]]
            cliff_compounds = set(selected["compound_a"].astype(str)) | set(selected["compound_b"].astype(str))
            possible_pairs = len(compounds) * (len(compounds) - 1) // 2
            scaffold_rows.append(
                {
                    "organism": organism,
                    "murcko_scaffold": scaffold,
                    "configuration": config,
                    "minimum_tanimoto": minimum_similarity,
                    "minimum_absolute_log2_mic_delta": minimum_delta,
                    "n_compounds_in_scaffold": len(compounds),
                    "n_possible_within_scaffold_pairs": possible_pairs,
                    "n_apparent_cliff_edges": len(selected),
                    "n_unique_compounds_in_apparent_cliffs": len(cliff_compounds),
                    "edge_fraction": len(selected) / possible_pairs,
                    "compound_fraction_in_apparent_cliffs": len(cliff_compounds) / len(compounds),
                }
            )
    scaffold_columns = [
        "organism",
        "murcko_scaffold",
        "configuration",
        "minimum_tanimoto",
        "minimum_absolute_log2_mic_delta",
        "n_compounds_in_scaffold",
        "n_possible_within_scaffold_pairs",
        "n_apparent_cliff_edges",
        "n_unique_compounds_in_apparent_cliffs",
        "edge_fraction",
        "compound_fraction_in_apparent_cliffs",
    ]
    scaffold_summary = pd.DataFrame(scaffold_rows, columns=scaffold_columns)
    species_rows = []
    if len(scaffold_summary):
        for (organism, configuration), group in scaffold_summary.groupby(
            ["organism", "configuration"], sort=True
        ):
            species_rows.append(
                {
                    "organism": organism,
                    "configuration": configuration,
                    "inference_unit": "nonempty_murcko_scaffold_and_unique_compound",
                    "n_evaluable_scaffolds": len(group),
                    "n_scaffolds_with_apparent_cliff": int((group["n_apparent_cliff_edges"] > 0).sum()),
                    "n_compounds_in_evaluable_scaffolds": int(group["n_compounds_in_scaffold"].sum()),
                    "n_unique_compounds_in_apparent_cliffs": int(
                        group["n_unique_compounds_in_apparent_cliffs"].sum()
                    ),
                    "n_possible_within_scaffold_pairs": int(
                        group["n_possible_within_scaffold_pairs"].sum()
                    ),
                    "n_apparent_cliff_edges_descriptive_only": int(
                        group["n_apparent_cliff_edges"].sum()
                    ),
                    "scaffold_fraction_with_apparent_cliff": float(
                        (group["n_apparent_cliff_edges"] > 0).mean()
                    ),
                    "compound_fraction_in_apparent_cliffs": float(
                        group["n_unique_compounds_in_apparent_cliffs"].sum()
                        / group["n_compounds_in_scaffold"].sum()
                    ),
                }
            )
    species_columns = [
        "organism",
        "configuration",
        "inference_unit",
        "n_evaluable_scaffolds",
        "n_scaffolds_with_apparent_cliff",
        "n_compounds_in_evaluable_scaffolds",
        "n_unique_compounds_in_apparent_cliffs",
        "n_possible_within_scaffold_pairs",
        "n_apparent_cliff_edges_descriptive_only",
        "scaffold_fraction_with_apparent_cliff",
        "compound_fraction_in_apparent_cliffs",
    ]
    return (
        edges,
        scaffold_summary,
        pd.DataFrame(species_rows, columns=species_columns),
        {"eligible_groups_total": total_groups, "eligible_groups_analyzed": len(grouped)},
    )


def _descriptive_stability(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, np.nan, np.nan, np.nan
    standard_deviation = float(array.std(ddof=1))
    half_width = float(student_t.ppf(0.975, len(array) - 1) * standard_deviation / np.sqrt(len(array)))
    return mean, standard_deviation, mean - half_width, mean + half_width


def cliff_error_analysis(predictions: pd.DataFrame, edges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    species_columns = [
        "configuration",
        "minimum_tanimoto",
        "minimum_absolute_log2_mic_delta",
        "model",
        "seed",
        "organism",
        "inference_unit",
        "n_cliff_participant_records",
        "n_nonparticipant_records",
        "cliff_participant_mae",
        "nonparticipant_mae",
        "delta_mae_cliff_minus_nonparticipant",
    ]
    summary_columns = [
        "configuration",
        "minimum_tanimoto",
        "minimum_absolute_log2_mic_delta",
        "model",
        "summary_level",
        "seed",
        "n_seeds",
        "n_species_equal_weighted",
        "n_species_min_across_seeds",
        "n_species_max_across_seeds",
        "n_cliff_participant_records",
        "n_nonparticipant_records",
        "cliff_participant_mae_equal_species",
        "nonparticipant_mae_equal_species",
        "delta_mae_cliff_minus_nonparticipant_equal_species",
        "cliff_participant_mae_seed_sd",
        "nonparticipant_mae_seed_sd",
        "delta_mae_seed_sd",
        "delta_descriptive_t_stability_interval_low",
        "delta_descriptive_t_stability_interval_high",
        "dependence_note",
    ]
    missing_edge_columns = sorted(set(CLIFF_CONFIGS) - set(edges.columns))
    if missing_edge_columns:
        raise ValueError(f"Cliff edges are missing configurations: {', '.join(missing_edge_columns)}")

    scored = predictions[
        ["compound_inchikey", "organism", "seed", "model", "y_true", "y_pred"]
    ].copy()
    scored["absolute_error"] = (scored["y_pred"] - scored["y_true"]).abs()
    species_rows = []
    for configuration, (minimum_tanimoto, minimum_delta) in CLIFF_CONFIGS.items():
        selected_edges = edges.loc[edges[configuration]]
        participants = pd.concat(
            [
                selected_edges[["organism", "compound_a"]].rename(
                    columns={"compound_a": "compound_inchikey"}
                ),
                selected_edges[["organism", "compound_b"]].rename(
                    columns={"compound_b": "compound_inchikey"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates()
        participants["is_cliff_participant"] = True
        annotated = scored.merge(
            participants,
            on=["organism", "compound_inchikey"],
            how="left",
            validate="many_to_one",
        )
        annotated["is_cliff_participant"] = annotated["is_cliff_participant"].eq(True)
        for (model, seed, organism), group in annotated.groupby(
            ["model", "seed", "organism"], sort=True
        ):
            cliff = group.loc[group["is_cliff_participant"]]
            nonparticipant = group.loc[~group["is_cliff_participant"]]
            if cliff.empty or nonparticipant.empty:
                continue
            cliff_mae = float(cliff["absolute_error"].mean())
            nonparticipant_mae = float(nonparticipant["absolute_error"].mean())
            species_rows.append(
                {
                    "configuration": configuration,
                    "minimum_tanimoto": minimum_tanimoto,
                    "minimum_absolute_log2_mic_delta": minimum_delta,
                    "model": model,
                    "seed": int(seed),
                    "organism": organism,
                    "inference_unit": "unique_compound_species_test_record",
                    "n_cliff_participant_records": len(cliff),
                    "n_nonparticipant_records": len(nonparticipant),
                    "cliff_participant_mae": cliff_mae,
                    "nonparticipant_mae": nonparticipant_mae,
                    "delta_mae_cliff_minus_nonparticipant": cliff_mae - nonparticipant_mae,
                }
            )
    by_species = pd.DataFrame(species_rows, columns=species_columns)
    if by_species.empty:
        return {"by_species": by_species, "summary": pd.DataFrame(columns=summary_columns)}

    per_seed = (
        by_species.groupby(
            [
                "configuration",
                "minimum_tanimoto",
                "minimum_absolute_log2_mic_delta",
                "model",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            n_species_equal_weighted=("organism", "nunique"),
            n_cliff_participant_records=("n_cliff_participant_records", "sum"),
            n_nonparticipant_records=("n_nonparticipant_records", "sum"),
            cliff_participant_mae_equal_species=("cliff_participant_mae", "mean"),
            nonparticipant_mae_equal_species=("nonparticipant_mae", "mean"),
            delta_mae_cliff_minus_nonparticipant_equal_species=(
                "delta_mae_cliff_minus_nonparticipant",
                "mean",
            ),
        )
    )
    summary_rows = []
    for row in per_seed.itertuples(index=False):
        summary_rows.append(
            {
                **row._asdict(),
                "summary_level": "seed_equal_species_mean",
                "n_seeds": 1,
                "n_species_min_across_seeds": row.n_species_equal_weighted,
                "n_species_max_across_seeds": row.n_species_equal_weighted,
                "cliff_participant_mae_seed_sd": np.nan,
                "nonparticipant_mae_seed_sd": np.nan,
                "delta_mae_seed_sd": np.nan,
                "delta_descriptive_t_stability_interval_low": np.nan,
                "delta_descriptive_t_stability_interval_high": np.nan,
                "dependence_note": "within-split equal-species descriptive mean; edge rows are not inference units",
            }
        )
    stability_keys = [
        "configuration",
        "minimum_tanimoto",
        "minimum_absolute_log2_mic_delta",
        "model",
    ]
    for keys, group in per_seed.groupby(stability_keys, sort=True):
        cliff_mean, cliff_sd, _, _ = _descriptive_stability(
            group["cliff_participant_mae_equal_species"]
        )
        nonparticipant_mean, nonparticipant_sd, _, _ = _descriptive_stability(
            group["nonparticipant_mae_equal_species"]
        )
        delta_mean, delta_sd, delta_low, delta_high = _descriptive_stability(
            group["delta_mae_cliff_minus_nonparticipant_equal_species"]
        )
        summary_rows.append(
            {
                **dict(zip(stability_keys, keys)),
                "summary_level": "across_seed_descriptive_stability",
                "seed": np.nan,
                "n_seeds": len(group),
                "n_species_equal_weighted": np.nan,
                "n_species_min_across_seeds": int(group["n_species_equal_weighted"].min()),
                "n_species_max_across_seeds": int(group["n_species_equal_weighted"].max()),
                "n_cliff_participant_records": np.nan,
                "n_nonparticipant_records": np.nan,
                "cliff_participant_mae_equal_species": cliff_mean,
                "nonparticipant_mae_equal_species": nonparticipant_mean,
                "delta_mae_cliff_minus_nonparticipant_equal_species": delta_mean,
                "cliff_participant_mae_seed_sd": cliff_sd,
                "nonparticipant_mae_seed_sd": nonparticipant_sd,
                "delta_mae_seed_sd": delta_sd,
                "delta_descriptive_t_stability_interval_low": delta_low,
                "delta_descriptive_t_stability_interval_high": delta_high,
                "dependence_note": "scaffold splits overlap in compounds and are not independent samples; the t interval is descriptive stability only",
            }
        )
    return {
        "by_species": by_species,
        "summary": pd.DataFrame(summary_rows, columns=summary_columns),
    }


def run_chemical_space(args: argparse.Namespace) -> dict:
    data_path = Path(args.data)
    prediction_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(data_path)
    predictions = pd.read_csv(prediction_path)
    _validate_inputs(pairs, predictions)
    predictions["seed"] = predictions["seed"].astype(int)

    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    scaffolds = build_scaffold_table(compounds)
    fingerprints = morgan_fingerprints(scaffolds)
    diversity, fallback_sensitivity = species_scaffold_diversity(pairs, scaffolds)

    available_seeds = sorted(predictions["seed"].unique().astype(int).tolist())
    seeds = available_seeds if args.seeds is None else args.seeds
    missing_seeds = sorted(set(seeds) - set(available_seeds))
    if missing_seeds:
        raise ValueError(f"Predictions do not contain seeds: {missing_seeds}")
    available_models = sorted(predictions["model"].astype(str).unique())
    models = available_models if args.models is None else args.models
    missing_models = sorted(set(models) - set(available_models))
    if missing_models:
        raise ValueError(f"Predictions do not contain models: {missing_models}")
    predictions = predictions.loc[
        predictions["seed"].isin(seeds) & predictions["model"].astype(str).isin(models)
    ].copy()

    nearest, split_checks = nearest_train_similarities(
        pairs,
        compounds,
        scaffolds,
        fingerprints,
        predictions,
        seeds,
        args.max_test_compounds,
    )
    if not split_checks["prediction_split_match"].all():
        split_checks.to_csv(output_dir / "split_overlap_checks.csv", index=False)
        stopped_manifest = {
            "status": "STOP_SPLIT_RECONSTRUCTION_MISMATCH",
            "data": str(data_path.resolve()),
            "data_sha256": _file_sha256(data_path),
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": _file_sha256(prediction_path),
            "seeds": seeds,
            "models": models,
            "reason": "Observed prediction compounds did not reproduce the existing scaffold-test assignment.",
            "outputs": ["split_overlap_checks.csv", "manifest.json"],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(stopped_manifest, indent=2), encoding="utf-8"
        )
        raise ValueError("STOP_SPLIT_RECONSTRUCTION_MISMATCH; see split_overlap_checks.csv.")
    error_tables = similarity_error_analysis(predictions, nearest, models)
    edges, cliff_scaffolds, cliff_species, cliff_counts = apparent_mic_cliffs(
        pairs,
        scaffolds,
        fingerprints,
        args.max_cliff_groups,
    )
    cliff_error_tables = cliff_error_analysis(predictions, edges)

    scaffolds.to_csv(output_dir / "compound_scaffolds.csv.gz", index=False, compression="gzip")
    diversity.to_csv(output_dir / "species_scaffold_summary.csv", index=False)
    fallback_sensitivity.to_csv(output_dir / "acyclic_fallback_sensitivity.csv", index=False)
    split_checks.to_csv(output_dir / "split_overlap_checks.csv", index=False)
    nearest.to_csv(output_dir / "nearest_train_similarity.csv.gz", index=False, compression="gzip")
    error_tables["by_compound"].to_csv(
        output_dir / "similarity_error_by_compound.csv.gz", index=False, compression="gzip"
    )
    error_tables["by_scaffold"].to_csv(output_dir / "similarity_error_by_scaffold.csv", index=False)
    error_tables["summary"].to_csv(output_dir / "similarity_error_summary.csv", index=False)
    error_tables["species_summary"].to_csv(
        output_dir / "similarity_error_species_summary.csv", index=False
    )
    error_tables["bins"].to_csv(output_dir / "similarity_error_bins.csv", index=False)
    edges.to_csv(output_dir / "apparent_mic_cliff_edges.csv.gz", index=False, compression="gzip")
    cliff_scaffolds.to_csv(output_dir / "apparent_mic_cliff_scaffold_summary.csv", index=False)
    cliff_species.to_csv(output_dir / "apparent_mic_cliff_species_summary.csv", index=False)
    cliff_error_tables["by_species"].to_csv(output_dir / "cliff_error_by_species.csv", index=False)
    cliff_error_tables["summary"].to_csv(output_dir / "cliff_error_summary.csv", index=False)

    smoke_only = (
        args.max_test_compounds is not None
        or args.max_cliff_groups is not None
        or set(seeds) != set(available_seeds)
        or set(models) != set(available_models)
    )
    manifest = {
        "status": "SMOKE_COMPLETE" if smoke_only else "COMPLETE",
        "data": str(data_path.resolve()),
        "data_sha256": _file_sha256(data_path),
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": _file_sha256(prediction_path),
        "output_dir": str(output_dir.resolve()),
        "rows": len(pairs),
        "compounds": len(compounds),
        "species": int(pairs["organism"].nunique()),
        "seeds": seeds,
        "models": models,
        "scaffold_split": "existing 70/15/15 compound-level Bemis-Murcko split reconstructed with scaffold_partition",
        "nearest_train_similarity": "exact Morgan radius=2, nBits=2048 Tanimoto",
        "scaffold_fallback": "empty Murcko scaffolds receive compound-specific ACYCLIC keys",
        "acyclic_sensitivity": "all compounds with fallback is reported beside nonempty-Murcko-only summaries",
        "apparent_mic_cliffs": {
            "restriction": "same species and same nonempty Bemis-Murcko scaffold",
            "configurations": CLIFF_CONFIGS,
            "edge_status": "descriptive only; shared compounds make edge rows non-independent",
            "inclusive_threshold_tolerance": THRESHOLD_EPSILON,
            **cliff_counts,
        },
        "cliff_error": {
            "participant_definition": "a test compound-species record is a cliff participant when its compound is an endpoint of at least one true edge for that configuration and species",
            "comparison": "all other unique test compound-species records for the same species are nonparticipants",
            "species_filter": "retain model-seed-species groups only when both participant and nonparticipant records are present",
            "per_seed_summary": "equal-species mean of within-species participant and nonparticipant MAE",
            "across_seed_summary": "mean, sample SD, and two-sided 95% Student-t descriptive stability interval over seed-level equal-species deltas",
            "dependence": "the five scaffold splits overlap in compounds and are not independent samples; edge rows are never inference units",
        },
        "inference_units": {
            "similarity_error": "equal compound, equal scaffold, and within-species compound summaries",
            "apparent_mic_cliffs": "species, nonempty scaffold, and unique compound counts; not raw pair edges",
            "cliff_error": "unique compound-species test records within species, equal-weighted species within seed, and descriptive overlapping-split summaries",
        },
        "max_test_compounds_per_seed": args.max_test_compounds,
        "max_cliff_groups": args.max_cliff_groups,
        "outputs": [
            "compound_scaffolds.csv.gz",
            "species_scaffold_summary.csv",
            "acyclic_fallback_sensitivity.csv",
            "split_overlap_checks.csv",
            "nearest_train_similarity.csv.gz",
            "similarity_error_by_compound.csv.gz",
            "similarity_error_by_scaffold.csv",
            "similarity_error_summary.csv",
            "similarity_error_species_summary.csv",
            "similarity_error_bins.csv",
            "apparent_mic_cliff_edges.csv.gz",
            "apparent_mic_cliff_scaffold_summary.csv",
            "apparent_mic_cliff_species_summary.csv",
            "cliff_error_by_species.csv",
            "cliff_error_summary.csv",
            "manifest.json",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "rdkit": rdBase.rdkitVersion,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit chemical-space and apparent MIC-cliff boundaries.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--max-test-compounds", type=int)
    parser.add_argument("--max-cliff-groups", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    manifest = run_chemical_space(parse_args(argv))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
