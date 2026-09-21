from __future__ import annotations

import argparse
from hashlib import sha256
import json
import platform
import sqlite3
import sys
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
import sklearn
from sklearn.preprocessing import OneHotEncoder

from .features import build_feature_matrix
from .species_mic import CHEMBL34_SQLITE_SHA256, MIC_SQL, _canonical_structures, _looks_like_species
from .species_mic_experiment import _finite_sample_quantile, _metrics, _species_balanced_weights


TEMPORAL_MIC_SQL = MIC_SQL.replace(
    "    a.data_validity_comment\nFROM",
    "    a.data_validity_comment,\n    a.doc_id,\n    d.year AS document_year\nFROM",
).replace(
    "JOIN compound_structures cs ON a.molregno = cs.molregno",
    "JOIN compound_structures cs ON a.molregno = cs.molregno\nLEFT JOIN docs d ON a.doc_id = d.doc_id",
)

COHORTS = ("train", "valid", "test")
MODEL_VIEWS = {"morgan": ["morgan"], "multiview": ["morgan", "maccs", "physchem", "graph"]}
MODEL_RANDOM_STATE = 20260910
REQUIRED_RAW_COLUMNS = {
    "compound_inchikey",
    "compound_smiles",
    "mic_ug_ml",
    "raw_organism",
    "source_tax_id",
    "strain",
    "assay_chembl_id",
    "activity_id",
    "data_validity_comment",
    "doc_id",
    "document_year",
}
POINT_METRICS = ("mae", "rmse", "spearman", "within_one_dilution", "within_two_dilutions")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_temporal_raw(
    raw: pd.DataFrame,
    organism_mapping: dict,
    taxonomy_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    missing_columns = sorted(REQUIRED_RAW_COLUMNS - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing raw temporal columns: {', '.join(missing_columns)}")
    required_taxonomy = {"tax_id", "species_tax_id", "species_name"}
    missing_taxonomy = sorted(required_taxonomy - set(taxonomy_mapping.columns))
    if missing_taxonomy:
        raise ValueError(f"Missing taxonomy mapping columns: {', '.join(missing_taxonomy)}")

    taxonomy = taxonomy_mapping.copy()
    taxonomy["tax_id"] = pd.to_numeric(taxonomy["tax_id"], errors="raise").astype("int64")
    if taxonomy["tax_id"].duplicated().any():
        raise ValueError("Taxonomy mapping contains duplicate tax_id values.")
    taxonomy_by_id = taxonomy.set_index("tax_id")

    frame = raw.copy()
    input_rows = len(frame)
    frame["mic_ug_ml"] = pd.to_numeric(frame["mic_ug_ml"], errors="coerce")
    frame = frame[np.isfinite(frame["mic_ug_ml"]) & frame["mic_ug_ml"].gt(0)].copy()
    numeric_mic_rows = len(frame)
    frame["pathogen_class"] = frame["raw_organism"].map(organism_mapping)
    looks_like_species = frame["raw_organism"].map(_looks_like_species)
    pathogen_unmapped_rows = int(frame["pathogen_class"].isna().sum())
    mapped_non_species_rows = int((frame["pathogen_class"].notna() & ~looks_like_species).sum())
    frame = frame[frame["pathogen_class"].notna() & looks_like_species].copy()
    rows_after_pathogen_mapping = len(frame)

    frame["source_tax_id"] = pd.to_numeric(frame["source_tax_id"], errors="coerce").astype("Int64")
    missing_tax_ids = sorted(set(frame["source_tax_id"].dropna().astype(int)) - set(taxonomy["tax_id"]))
    if missing_tax_ids:
        raise ValueError(f"Taxonomy mapping is missing {len(missing_tax_ids)} source tax_id values.")
    frame["species_tax_id"] = frame["source_tax_id"].map(taxonomy_by_id["species_tax_id"])
    frame["species_name"] = frame["source_tax_id"].map(taxonomy_by_id["species_name"])
    frame["organism"] = frame["species_name"]
    frame["pathogen_class"] = frame["species_name"].map(organism_mapping).fillna(frame["pathogen_class"])
    rows_without_species_ancestor = int(
        (frame["species_tax_id"].isna() | frame["species_name"].isna() | frame["pathogen_class"].isna()).sum()
    )
    frame = frame[
        frame["species_tax_id"].notna()
        & frame["species_name"].notna()
        & frame["pathogen_class"].notna()
    ].copy()
    frame["species_tax_id"] = pd.to_numeric(frame["species_tax_id"], errors="raise").astype("int64")
    frame["tax_id"] = frame["species_tax_id"]

    structure_map = _canonical_structures(frame[["compound_inchikey", "compound_smiles"]])
    frame["canonical_smiles"] = frame["compound_inchikey"].map(structure_map)
    invalid_structure_rows = int(frame["canonical_smiles"].isna().sum())
    frame = frame[frame["canonical_smiles"].notna()].copy()
    frame["document_year"] = pd.to_numeric(frame["document_year"], errors="coerce")
    missing_year = frame[frame["document_year"].isna()].copy()
    known_year = frame[frame["document_year"].notna()].copy()
    known_year["document_year"] = known_year["document_year"].astype("int64")
    known_year["log2_mic_ug_ml"] = np.log2(known_year["mic_ug_ml"])

    audit = {
        "sql_rows": int(input_rows),
        "rows_after_numeric_mic_qc": int(numeric_mic_rows),
        "pathogen_class_unmapped_rows": pathogen_unmapped_rows,
        "pathogen_mapped_non_species_rows": mapped_non_species_rows,
        "rows_after_pathogen_mapping": int(rows_after_pathogen_mapping),
        "rows_without_species_ancestor": rows_without_species_ancestor,
        "invalid_structure_rows": invalid_structure_rows,
        "rows_after_mapping_and_structure_qc": int(len(frame)),
        "missing_document_year_rows_excluded": int(len(missing_year)),
        "known_document_year_rows": int(len(known_year)),
        "known_document_year_min": int(known_year["document_year"].min()) if len(known_year) else None,
        "known_document_year_max": int(known_year["document_year"].max()) if len(known_year) else None,
    }
    return known_year, missing_year, audit


def _assign_temporal_cohorts(known_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    first_year = (
        known_year.groupby("compound_inchikey", as_index=False, observed=True)["document_year"]
        .min()
        .rename(columns={"document_year": "first_eligible_year"})
    )
    first_year["cohort"] = np.select(
        [
            first_year["first_eligible_year"].le(2018),
            first_year["first_eligible_year"].between(2019, 2020),
            first_year["first_eligible_year"].between(2021, 2023),
        ],
        list(COHORTS),
        default="outside_frozen_window",
    )
    frame = known_year.merge(first_year, on="compound_inchikey", how="left", validate="many_to_one")
    in_measurement_window = (
        (frame["cohort"].eq("train") & frame["document_year"].le(2018))
        | (frame["cohort"].eq("valid") & frame["document_year"].between(2019, 2020))
        | (frame["cohort"].eq("test") & frame["document_year"].between(2021, 2023))
    )
    retained = frame[in_measurement_window].copy()
    cohort_audit = {
        "known_year_compounds": int(first_year["compound_inchikey"].nunique()),
        "compounds_outside_frozen_window": int(first_year["cohort"].eq("outside_frozen_window").sum()),
        "measurement_rows_retained": int(len(retained)),
        "measurement_rows_excluded_outside_assigned_window": int(len(frame) - len(retained)),
        "retained_rows_by_cohort": {str(k): int(v) for k, v in retained["cohort"].value_counts().items()},
    }
    return retained, first_year, cohort_audit


def _aggregate_cohort_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "compound_inchikey", "organism", "pathogen_class", "tax_id"]
    grouped = rows.groupby(keys, sort=False, observed=True)
    pairs = grouped.agg(
        canonical_smiles=("canonical_smiles", "first"),
        log2_mic=("log2_mic_ug_ml", "median"),
        mic_ug_ml=("mic_ug_ml", "median"),
        n_measurements=("activity_id", "size"),
        n_assays=("assay_chembl_id", "nunique"),
        n_strains=("strain", "nunique"),
        n_source_taxids=("source_tax_id", "nunique"),
        n_raw_organisms=("raw_organism", "nunique"),
        n_documents=("doc_id", "nunique"),
        measurement_year_min=("document_year", "min"),
        measurement_year_max=("document_year", "max"),
        first_eligible_year=("first_eligible_year", "first"),
    ).reset_index()
    quantiles = grouped["log2_mic_ug_ml"].quantile([0.25, 0.75]).unstack()
    quantiles.columns = ["log2_mic_q25", "log2_mic_q75"]
    pairs = pairs.merge(quantiles.reset_index(), on=keys, how="left", validate="one_to_one")
    pairs["log2_mic_iqr"] = pairs["log2_mic_q75"] - pairs["log2_mic_q25"]
    return pairs


def _species_counts(pairs: pd.DataFrame, min_compounds: int) -> tuple[pd.DataFrame, list[str]]:
    counts = (
        pairs.groupby(["cohort", "organism", "pathogen_class"], as_index=False, observed=True)
        .agg(pairs=("compound_inchikey", "size"), compounds=("compound_inchikey", "nunique"))
    )
    train_counts = counts[counts["cohort"].eq("train")]
    eligible = sorted(train_counts.loc[train_counts["compounds"].ge(min_compounds), "organism"].astype(str))
    if not eligible:
        raise ValueError("No species meet the training-era compound threshold.")

    eligible_pairs = pairs[pairs["organism"].astype(str).isin(eligible)]
    ambiguous_taxids = eligible_pairs.groupby("organism", observed=True)["tax_id"].nunique()
    if (ambiguous_taxids > 1).any():
        raise ValueError(
            f"Eligible canonical species names map to multiple accepted tax_ids: "
            f"{ambiguous_taxids[ambiguous_taxids > 1].to_dict()}"
        )
    ambiguous_classes = eligible_pairs.groupby("organism", observed=True)["pathogen_class"].nunique()
    if (ambiguous_classes > 1).any():
        raise ValueError(
            f"Eligible canonical species names map to multiple pathogen classes: "
            f"{ambiguous_classes[ambiguous_classes > 1].to_dict()}"
        )
    identity = (
        eligible_pairs[["organism", "pathogen_class", "tax_id"]]
        .drop_duplicates("organism")
    )
    grid = pd.MultiIndex.from_product([COHORTS, eligible], names=["cohort", "organism"]).to_frame(index=False)
    eligible_counts = grid.merge(identity, on="organism", how="left", validate="many_to_one").merge(
        counts[["cohort", "organism", "pairs", "compounds"]],
        on=["cohort", "organism"],
        how="left",
        validate="one_to_one",
    )
    eligible_counts[["pairs", "compounds"]] = eligible_counts[["pairs", "compounds"]].fillna(0).astype(int)
    eligible_counts["eligible_from_training_era"] = True
    eligible_counts["min_training_compounds"] = int(min_compounds)
    return eligible_counts, eligible


def _scaffold_and_parent(smiles: str, compound_key: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True) if scaffold and scaffold.GetNumAtoms() else ""
    scaffold_key = scaffold_smiles or f"ACYCLIC:{compound_key}"
    parent = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    return scaffold_key, parent_smiles


def _overlap_audit(pairs: pd.DataFrame, species_counts: pd.DataFrame) -> dict:
    compounds = pairs[["cohort", "compound_inchikey", "canonical_smiles"]].drop_duplicates()
    annotations = {}
    for row in compounds[["compound_inchikey", "canonical_smiles"]].drop_duplicates().itertuples(index=False):
        annotations[str(row.compound_inchikey)] = _scaffold_and_parent(row.canonical_smiles, str(row.compound_inchikey))
    compounds["scaffold_key"] = compounds["compound_inchikey"].map(lambda key: annotations[str(key)][0])
    compounds["standardized_parent"] = compounds["compound_inchikey"].map(lambda key: annotations[str(key)][1])
    sets = {
        cohort: {
            "compound": set(group["compound_inchikey"].astype(str)),
            "scaffold": set(group["scaffold_key"].astype(str)),
            "parent": set(group["standardized_parent"].astype(str)),
        }
        for cohort, group in compounds.groupby("cohort", observed=True)
    }
    for cohort in COHORTS:
        sets.setdefault(cohort, {"compound": set(), "scaffold": set(), "parent": set()})

    intersections = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        for unit in ("compound", "scaffold", "parent"):
            intersections[f"{unit}_overlap_{left}_{right}"] = len(sets[left][unit] & sets[right][unit])

    year_contract = {
        "train_first_year_max_le_2018": bool(
            pairs.loc[pairs["cohort"].eq("train"), "first_eligible_year"].le(2018).all()
        ),
        "valid_first_year_2019_2020": bool(
            pairs.loc[pairs["cohort"].eq("valid"), "first_eligible_year"].between(2019, 2020).all()
        ),
        "test_first_year_2021_2023": bool(
            pairs.loc[pairs["cohort"].eq("test"), "first_eligible_year"].between(2021, 2023).all()
        ),
        "train_measurements_max_2018": bool(
            pairs.loc[pairs["cohort"].eq("train"), "measurement_year_max"].le(2018).all()
        ),
        "valid_measurements_2019_2020": bool(
            pairs.loc[pairs["cohort"].eq("valid"), "measurement_year_min"].ge(2019).all()
            and pairs.loc[pairs["cohort"].eq("valid"), "measurement_year_max"].le(2020).all()
        ),
        "test_measurements_2021_2023": bool(
            pairs.loc[pairs["cohort"].eq("test"), "measurement_year_min"].ge(2021).all()
            and pairs.loc[pairs["cohort"].eq("test"), "measurement_year_max"].le(2023).all()
        ),
    }
    support = {
        cohort: species_counts.loc[species_counts["cohort"].eq(cohort)].set_index("organism")["pairs"].astype(int).to_dict()
        for cohort in COHORTS
    }
    train_species = {species for species, count in support["train"].items() if count > 0}
    valid_species = {species for species, count in support["valid"].items() if count > 0}
    test_species = {species for species, count in support["test"].items() if count > 0}
    exact_overlap_zero = all(value == 0 for key, value in intersections.items() if key.startswith("compound_overlap"))
    audit = {
        **intersections,
        **year_contract,
        "exact_compound_overlap_zero": exact_overlap_zero,
        "training_species": len(train_species),
        "validation_species_with_support": len(valid_species),
        "test_species_with_support": len(test_species),
        "validation_species_without_support": sorted(train_species - valid_species),
        "test_species_without_support": sorted(train_species - test_species),
        "validation_species_absent_from_training": sorted(valid_species - train_species),
        "test_species_absent_from_training": sorted(test_species - train_species),
        "support_by_cohort": support,
    }
    audit["temporal_contract_pass"] = bool(
        exact_overlap_zero
        and all(year_contract.values())
        and not audit["validation_species_absent_from_training"]
        and not audit["test_species_absent_from_training"]
    )
    return audit


def prepare_temporal_frames(
    raw: pd.DataFrame,
    organism_mapping: dict,
    taxonomy_mapping: pd.DataFrame,
    min_compounds: int = 500,
) -> dict:
    known_year, missing_year, raw_audit = _normalize_temporal_raw(raw, organism_mapping, taxonomy_mapping)
    retained_rows, compound_cohorts, cohort_audit = _assign_temporal_cohorts(known_year)
    pairs_all = _aggregate_cohort_pairs(retained_rows)
    species_counts, eligible_species = _species_counts(pairs_all, min_compounds)
    pairs = pairs_all[pairs_all["organism"].astype(str).isin(eligible_species)].copy()
    overlap_audit = _overlap_audit(pairs, species_counts)
    if not overlap_audit["temporal_contract_pass"]:
        raise ValueError("Temporal leakage/support contract failed; see overlap audit.")
    raw_audit.update(cohort_audit)
    raw_audit.update(
        {
            "all_cohort_pairs": int(len(pairs_all)),
            "eligible_cohort_pairs": int(len(pairs)),
            "eligible_species_from_training_era": len(eligible_species),
            "eligible_species_names": eligible_species,
        }
    )
    return {
        "missing_year_rows": missing_year,
        "compound_cohorts": compound_cohorts,
        "pairs_all": pairs_all,
        "pairs": pairs,
        "species_counts": species_counts,
        "raw_audit": raw_audit,
        "overlap_audit": overlap_audit,
    }


def _metric_rows(model_name: str, test: pd.DataFrame, conformal_q: float) -> list[dict]:
    rows = []
    for organism, group in test.groupby("organism", sort=True, observed=True):
        row = _metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        row.update(
            {
                "model": model_name,
                "scope": "species",
                "organism": str(organism),
                "n_pairs": len(group),
                "n_compounds": group["compound_inchikey"].nunique(),
                "empirical_conformal_coverage_90": float(group["covered_90"].mean()),
                "empirical_conformal_width_90": float((group["upper_90"] - group["lower_90"]).mean()),
                "calibration_q_90": conformal_q,
            }
        )
        rows.append(row)

    pooled = _metrics(test["y_true"].to_numpy(), test["y_pred"].to_numpy())
    pooled.update(
        {
            "model": model_name,
            "scope": "pooled",
            "organism": "__pooled__",
            "n_pairs": len(test),
            "n_compounds": test["compound_inchikey"].nunique(),
            "empirical_conformal_coverage_90": float(test["covered_90"].mean()),
            "empirical_conformal_width_90": float((test["upper_90"] - test["lower_90"]).mean()),
            "calibration_q_90": conformal_q,
        }
    )
    macro_source = pd.DataFrame(rows)
    macro = {
        metric: float(macro_source[metric].mean())
        for metric in (*POINT_METRICS, "empirical_conformal_coverage_90", "empirical_conformal_width_90")
    }
    macro.update(
        {
            "model": model_name,
            "scope": "macro",
            "organism": "__macro__",
            "n_pairs": len(test),
            "n_compounds": test["compound_inchikey"].nunique(),
            "calibration_q_90": conformal_q,
        }
    )
    return [*rows, pooled, macro]


def _run_models(
    pairs: pd.DataFrame,
    n_estimators: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_frames = {cohort: pairs[pairs["cohort"].eq(cohort)].reset_index(drop=True) for cohort in COHORTS}
    if any(frame.empty for frame in split_frames.values()):
        empty = [cohort for cohort, frame in split_frames.items() if frame.empty]
        raise ValueError(f"Temporal modeling cohort is empty: {', '.join(empty)}")

    encoder = OneHotEncoder(handle_unknown="error", sparse_output=True, dtype=np.float32)
    species_x = {
        "train": encoder.fit_transform(split_frames["train"][["organism"]]).tocsr(),
        "valid": encoder.transform(split_frames["valid"][["organism"]]).tocsr(),
        "test": encoder.transform(split_frames["test"][["organism"]]).tocsr(),
    }
    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    compound_index = pd.Series(np.arange(len(compounds)), index=compounds["compound_inchikey"])
    row_indices = {
        cohort: split_frames[cohort]["compound_inchikey"].map(compound_index).to_numpy(dtype=int)
        for cohort in COHORTS
    }
    train_weights = _species_balanced_weights(split_frames["train"]["organism"])
    prediction_frames = []
    metric_rows = []
    runtime_rows = []
    for model_name, views in MODEL_VIEWS.items():
        feature_start = perf_counter()
        compound_x = sparse.csr_matrix(build_feature_matrix(compounds["canonical_smiles"], views).x)
        design = {
            cohort: sparse.hstack(
                [compound_x[row_indices[cohort]], species_x[cohort]],
                format="csr",
                dtype=np.float32,
            )
            for cohort in COHORTS
        }
        runtime_rows.append(
            {
                "model": model_name,
                "phase": "feature_build",
                "seconds": perf_counter() - feature_start,
                "rows": len(pairs),
                "feature_columns": design["train"].shape[1],
            }
        )
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=MODEL_RANDOM_STATE,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        fit_start = perf_counter()
        model.fit(
            design["train"],
            split_frames["train"]["log2_mic"].to_numpy(dtype=np.float32),
            sample_weight=train_weights,
        )
        runtime_rows.append(
            {
                "model": model_name,
                "phase": "fit",
                "seconds": perf_counter() - fit_start,
                "rows": len(split_frames["train"]),
                "feature_columns": design["train"].shape[1],
            }
        )
        predictions = {}
        for cohort in ("valid", "test"):
            predict_start = perf_counter()
            predictions[cohort] = model.predict(design[cohort])
            runtime_rows.append(
                {
                    "model": model_name,
                    "phase": f"predict_{cohort}",
                    "seconds": perf_counter() - predict_start,
                    "rows": len(split_frames[cohort]),
                    "feature_columns": design[cohort].shape[1],
                }
            )
        residual = np.abs(split_frames["valid"]["log2_mic"].to_numpy() - predictions["valid"])
        conformal_q = _finite_sample_quantile(residual, coverage=0.90)
        base_columns = [
            "compound_inchikey",
            "canonical_smiles",
            "organism",
            "tax_id",
            "pathogen_class",
            "first_eligible_year",
            "measurement_year_min",
            "measurement_year_max",
        ]
        test = split_frames["test"][base_columns].copy()
        test["y_true"] = split_frames["test"]["log2_mic"].to_numpy()
        test["y_pred"] = predictions["test"]
        test["absolute_error"] = np.abs(test["y_true"] - test["y_pred"])
        test["calibration_q_90"] = conformal_q
        test["lower_90"] = test["y_pred"] - conformal_q
        test["upper_90"] = test["y_pred"] + conformal_q
        test["covered_90"] = test["y_true"].between(test["lower_90"], test["upper_90"], inclusive="both")
        test["model"] = model_name
        prediction_frames.append(test)
        metric_rows.extend(_metric_rows(model_name, test, conformal_q))
        del compound_x, design, model

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(metric_rows), pd.DataFrame(runtime_rows)


def run_temporal_from_raw(
    raw: pd.DataFrame,
    organism_mapping: dict,
    taxonomy_mapping: pd.DataFrame,
    output_dir: str | Path,
    min_compounds: int = 500,
    n_estimators: int = 300,
    n_jobs: int = -1,
    provenance: dict | None = None,
    query_seconds: float | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preparation_start = perf_counter()
    frames = prepare_temporal_frames(raw, organism_mapping, taxonomy_mapping, min_compounds=min_compounds)
    preparation_seconds = perf_counter() - preparation_start

    frames["missing_year_rows"].to_csv(
        output_dir / "raw_missing_year_rows.csv.gz", index=False, compression="gzip"
    )
    frames["compound_cohorts"].to_csv(output_dir / "compound_cohorts.csv.gz", index=False, compression="gzip")
    frames["pairs_all"].to_csv(output_dir / "cohort_pairs_all.csv.gz", index=False, compression="gzip")
    frames["pairs"].to_csv(output_dir / "cohort_pairs_eligible.csv.gz", index=False, compression="gzip")
    frames["species_counts"].to_csv(output_dir / "species_counts.csv", index=False)
    (output_dir / "raw_audit.json").write_text(json.dumps(frames["raw_audit"], indent=2), encoding="utf-8")
    (output_dir / "overlap_audit.json").write_text(
        json.dumps(frames["overlap_audit"], indent=2), encoding="utf-8"
    )
    (output_dir / "query.sql").write_text(TEMPORAL_MIC_SQL.strip() + "\n", encoding="utf-8")

    predictions, metrics, runtime = _run_models(frames["pairs"], n_estimators=n_estimators, n_jobs=n_jobs)
    prep_runtime = pd.DataFrame(
        [
            {
                "model": "__shared__",
                "phase": "database_query",
                "seconds": query_seconds if query_seconds is not None else np.nan,
                "rows": len(raw),
                "feature_columns": np.nan,
            },
            {
                "model": "__shared__",
                "phase": "temporal_preparation",
                "seconds": preparation_seconds,
                "rows": len(frames["pairs"]),
                "feature_columns": np.nan,
            },
        ]
    )
    runtime = pd.concat([prep_runtime, runtime], ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    runtime.to_csv(output_dir / "runtime.csv", index=False)

    provenance = provenance or {"source": "in-memory raw DataFrame"}
    manifest = {
        "status": "COMPLETE",
        "design": "strict retrospective future-compound temporal holdout",
        "source_year_interpretation": "ChEMBL document publication year; not compound discovery time",
        "cohorts": {
            "train": "first eligible MIC year <=2018; measurements <=2018 only",
            "valid": "first eligible MIC year 2019-2020; measurements 2019-2020 only",
            "test": "first eligible MIC year 2021-2023; measurements 2021-2023 only",
        },
        "aggregation": "median log2(MIC ug/mL) after raw rows are restricted to their assigned time window",
        "species_eligibility": f">={min_compounds} unique compounds in aggregated training-era pairs only",
        "models": list(MODEL_VIEWS),
        "species_conditioning": "train-only species one-hot encoding",
        "training_weights": "inverse training species frequency normalized to mean one",
        "conformal": "pooled finite-sample empirical 90% residual quantile calibrated on validation only",
        "random_state": MODEL_RANDOM_STATE,
        "n_estimators": int(n_estimators),
        "n_jobs": int(n_jobs),
        "raw_rows": len(raw),
        "eligible_pairs": len(frames["pairs"]),
        "eligible_compounds": int(frames["pairs"]["compound_inchikey"].nunique()),
        "eligible_species": int(frames["species_counts"]["organism"].nunique()),
        "provenance": provenance,
        "outputs": [
            "raw_audit.json",
            "raw_missing_year_rows.csv.gz",
            "compound_cohorts.csv.gz",
            "cohort_pairs_all.csv.gz",
            "cohort_pairs_eligible.csv.gz",
            "species_counts.csv",
            "overlap_audit.json",
            "metrics.csv",
            "predictions.csv.gz",
            "runtime.csv",
            "query.sql",
            "manifest.json",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_temporal(args: argparse.Namespace) -> dict:
    database = Path(args.database)
    organism_mapping_path = Path(args.organism_mapping)
    taxonomy_mapping_path = Path(args.taxonomy_mapping)
    mapping = json.loads(organism_mapping_path.read_text(encoding="utf-8"))
    taxonomy = pd.read_csv(taxonomy_mapping_path)
    query_start = perf_counter()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        raw = pd.read_sql_query(TEMPORAL_MIC_SQL, connection)
    query_seconds = perf_counter() - query_start
    provenance = {
        "database": str(database.resolve()),
        "database_size_bytes": database.stat().st_size,
        "organism_mapping": str(organism_mapping_path.resolve()),
        "taxonomy_mapping": str(taxonomy_mapping_path.resolve()),
        "organism_mapping_sha256": _file_sha256(organism_mapping_path),
        "taxonomy_mapping_sha256": _file_sha256(taxonomy_mapping_path),
        "chembl_release": 34,
        "chembl_sqlite_archive_sha256": CHEMBL34_SQLITE_SHA256,
    }
    return run_temporal_from_raw(
        raw,
        mapping,
        taxonomy,
        args.output_dir,
        min_compounds=args.min_compounds,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        provenance=provenance,
        query_seconds=query_seconds,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen ChEMBL 34 temporal MIC reliability audit.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--organism-mapping", required=True)
    parser.add_argument("--taxonomy-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-compounds", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    manifest = run_temporal(parse_args(argv))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
