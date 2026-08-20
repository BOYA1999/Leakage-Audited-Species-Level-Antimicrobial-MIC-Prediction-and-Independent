from __future__ import annotations

import argparse
from hashlib import sha256
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CHEMBL34_SQLITE_SHA256 = "7d40409e4f31440674a79e26db628015e82b85b056a7d8234efe9c95b6cbd8e7"

MIC_SQL = """
SELECT
    cs.standard_inchi_key AS compound_inchikey,
    cs.canonical_smiles AS compound_smiles,
    a.standard_value AS mic_ug_ml,
    s.assay_organism AS raw_organism,
    s.assay_tax_id AS source_tax_id,
    s.assay_strain AS strain,
    s.chembl_id AS assay_chembl_id,
    a.activity_id,
    a.data_validity_comment
FROM activities a
JOIN assays s ON a.assay_id = s.assay_id
JOIN compound_structures cs ON a.molregno = cs.molregno
WHERE
    a.standard_type = 'MIC'
    AND a.standard_relation = '='
    AND a.standard_units = 'ug.mL-1'
    AND a.standard_value > 0
    AND s.assay_organism IS NOT NULL
    AND s.assay_tax_id IS NOT NULL
    AND cs.standard_inchi_key IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
    AND a.data_validity_comment IS NULL
"""


def build_species_mic_dataset(
    database: str | Path,
    organism_mapping: str | Path,
    taxonomy_mapping: str | Path,
    output_dir: str | Path,
    min_compounds: int = 500,
) -> dict:
    database = Path(database)
    organism_mapping = Path(organism_mapping)
    taxonomy_mapping = Path(taxonomy_mapping)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(organism_mapping.read_text(encoding="utf-8"))
    taxonomy = pd.read_csv(taxonomy_mapping)
    required_taxonomy = {"tax_id", "species_tax_id", "species_name"}
    missing_taxonomy_columns = sorted(required_taxonomy.difference(taxonomy.columns))
    if missing_taxonomy_columns:
        raise ValueError(f"Missing taxonomy mapping columns: {missing_taxonomy_columns}")
    taxonomy["tax_id"] = pd.to_numeric(taxonomy["tax_id"], errors="raise").astype("int64")
    if taxonomy["tax_id"].duplicated().any():
        raise ValueError("Taxonomy mapping contains duplicate tax_id values.")
    taxonomy_by_id = taxonomy.set_index("tax_id")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        raw = pd.read_sql_query(MIC_SQL, connection)

    input_rows = len(raw)
    raw["mic_ug_ml"] = pd.to_numeric(raw["mic_ug_ml"], errors="coerce")
    raw = raw[np.isfinite(raw["mic_ug_ml"]) & raw["mic_ug_ml"].gt(0)].copy()
    raw["pathogen_class"] = raw["raw_organism"].map(mapping)
    looks_like_species = raw["raw_organism"].map(_looks_like_species)
    pathogen_unmapped_rows = int(raw["pathogen_class"].isna().sum())
    mapped_non_species_rows = int((raw["pathogen_class"].notna() & ~looks_like_species).sum())
    raw = raw[raw["pathogen_class"].notna() & looks_like_species].copy()
    rows_after_pathogen_mapping = len(raw)
    raw["source_tax_id"] = pd.to_numeric(raw["source_tax_id"], errors="coerce").astype("Int64")
    missing_tax_ids = sorted(set(raw["source_tax_id"].dropna().astype(int)) - set(taxonomy["tax_id"]))
    if missing_tax_ids:
        raise ValueError(f"Taxonomy mapping is missing {len(missing_tax_ids)} source tax_id values.")
    raw["species_tax_id"] = raw["source_tax_id"].map(taxonomy_by_id["species_tax_id"])
    raw["species_name"] = raw["source_tax_id"].map(taxonomy_by_id["species_name"])
    raw["organism"] = raw["species_name"]
    raw["pathogen_class"] = raw["species_name"].map(mapping).fillna(raw["pathogen_class"])
    rows_without_species_ancestor = int((raw["species_tax_id"].isna() | raw["species_name"].isna()).sum())
    raw = raw[
        raw["species_tax_id"].notna()
        & raw["species_name"].notna()
        & raw["pathogen_class"].notna()
    ].copy()
    raw["species_tax_id"] = pd.to_numeric(raw["species_tax_id"], errors="raise").astype("int64")
    raw["tax_id"] = raw["species_tax_id"]

    structure_map = _canonical_structures(raw[["compound_inchikey", "compound_smiles"]])
    raw["canonical_smiles"] = raw["compound_inchikey"].map(structure_map)
    invalid_structure_rows = int(raw["canonical_smiles"].isna().sum())
    raw = raw[raw["canonical_smiles"].notna()].copy()
    raw["log2_mic_ug_ml"] = np.log2(raw["mic_ug_ml"])

    keys = ["compound_inchikey", "organism", "pathogen_class", "tax_id"]
    grouped = raw.groupby(keys, sort=False, observed=True)
    pairs = grouped.agg(
        canonical_smiles=("canonical_smiles", "first"),
        log2_mic=("log2_mic_ug_ml", "median"),
        mic_ug_ml=("mic_ug_ml", "median"),
        n_measurements=("activity_id", "size"),
        n_assays=("assay_chembl_id", "nunique"),
        n_strains=("strain", "nunique"),
        n_source_taxids=("source_tax_id", "nunique"),
        n_raw_organisms=("raw_organism", "nunique"),
    ).reset_index()
    quantiles = grouped["log2_mic_ug_ml"].quantile([0.25, 0.75]).unstack()
    quantiles.columns = ["log2_mic_q25", "log2_mic_q75"]
    pairs = pairs.merge(quantiles.reset_index(), on=keys, how="left", validate="one_to_one")
    pairs["log2_mic_iqr"] = pairs["log2_mic_q75"] - pairs["log2_mic_q25"]

    species_counts = (
        pairs.groupby(["organism", "pathogen_class"], as_index=False, observed=True)
        .agg(tax_id=("tax_id", "first"), pairs=("compound_inchikey", "size"), compounds=("compound_inchikey", "nunique"))
        .sort_values(["compounds", "organism"], ascending=[False, True])
    )
    eligible = species_counts.loc[species_counts["compounds"].ge(min_compounds), "organism"]
    eligible_pairs = pairs[pairs["organism"].isin(eligible)].copy()
    eligible_name_taxids = eligible_pairs.groupby("organism", observed=True)["tax_id"].nunique()
    ambiguous_names = eligible_name_taxids[eligible_name_taxids.gt(1)]
    if not ambiguous_names.empty:
        raise ValueError(f"Canonical species names map to multiple accepted tax_ids: {ambiguous_names.to_dict()}")

    source_labels = (
        raw[raw["organism"].isin(eligible)]
        .groupby(["organism", "tax_id", "pathogen_class"], as_index=False, observed=True)
        .agg(source_labels=("raw_organism", lambda values: "; ".join(sorted(set(map(str, values))))))
    )
    taxid_composition = (
        eligible_pairs.groupby(["organism", "tax_id", "pathogen_class"], as_index=False, observed=True)
        .agg(compound_species_pairs=("compound_inchikey", "size"), unique_compounds=("compound_inchikey", "nunique"))
        .merge(source_labels, on=["organism", "tax_id", "pathogen_class"], how="left", validate="one_to_one")
        .sort_values(["pathogen_class", "organism"])
    )

    pairs.to_csv(output_dir / "mic_pairs_all.csv.gz", index=False, compression="gzip")
    eligible_pairs.to_csv(output_dir / "mic_pairs_eligible.csv.gz", index=False, compression="gzip")
    species_counts.to_csv(output_dir / "species_counts.csv", index=False)
    taxid_composition.to_csv(output_dir / "species_taxonomy_detail.csv", index=False)

    dataset_flow = pd.DataFrame(
        [
            {"step": "Exact MIC query after validity filtering", "rows_retained": input_rows, "rows_excluded_at_step": 0, "reason": "Frozen SQL contract"},
            {"step": "Pathogen/class eligibility", "rows_retained": rows_after_pathogen_mapping, "rows_excluded_at_step": input_rows - rows_after_pathogen_mapping, "reason": f"Unmapped class: {pathogen_unmapped_rows}; mapped non-binomial or out-of-scope label: {mapped_non_species_rows}"},
            {"step": "NCBI species ancestor", "rows_retained": rows_after_pathogen_mapping - rows_without_species_ancestor, "rows_excluded_at_step": rows_without_species_ancestor, "reason": "No species-level ancestor in frozen mapping"},
            {"step": "Structure parsing", "rows_retained": len(raw), "rows_excluded_at_step": invalid_structure_rows, "reason": "RDKit-invalid representative structure"},
            {"step": "Median compound-species aggregation", "rows_retained": len(pairs), "rows_excluded_at_step": len(raw) - len(pairs), "reason": "Repeated measurements collapsed; not a biological exclusion"},
            {"step": f">={min_compounds} compounds per species", "rows_retained": len(eligible_pairs), "rows_excluded_at_step": len(pairs) - len(eligible_pairs), "reason": "Frozen species eligibility gate"},
        ]
    )
    dataset_flow.to_csv(output_dir / "dataset_flow.csv", index=False)

    qc = {
        "database": str(database.resolve()),
        "organism_mapping": str(organism_mapping.resolve()),
        "taxonomy_mapping": str(taxonomy_mapping.resolve()),
        "organism_mapping_sha256": _sha256(organism_mapping),
        "taxonomy_mapping_sha256": _sha256(taxonomy_mapping),
        "chembl_release": 34,
        "chembl_sqlite_archive_sha256": CHEMBL34_SQLITE_SHA256,
        "endpoint": "MIC",
        "relation": "=",
        "unit": "ug.mL-1",
        "target": "median log2(MIC ug/mL)",
        "min_compounds_per_species": int(min_compounds),
        "sql_rows": int(input_rows),
        "rows_after_pathogen_mapping": int(rows_after_pathogen_mapping),
        "pathogen_class_unmapped_rows": pathogen_unmapped_rows,
        "pathogen_mapped_non_species_rows": mapped_non_species_rows,
        "rows_without_species_ancestor": rows_without_species_ancestor,
        "invalid_structure_rows": invalid_structure_rows,
        "rows_after_mapping_and_structure_qc": int(len(raw)),
        "taxonomy_mapping_rows": int(len(taxonomy)),
        "taxonomy_source_taxids_missing_species_ancestor": int(taxonomy["species_tax_id"].isna().sum()),
        "taxonomy_normalized_species_names": int(raw["species_name"].nunique()),
        "taxonomy_normalized_species_taxids": int(raw["species_tax_id"].nunique()),
        "taxonomy_source_organism_labels": int(raw["raw_organism"].nunique()),
        "taxonomy_species_labels_changed": int((raw["raw_organism"] != raw["species_name"]).sum()),
        "all_compound_organism_pairs": int(len(pairs)),
        "all_unique_compounds": int(pairs["compound_inchikey"].nunique()),
        "eligible_species": int(len(eligible)),
        "eligible_species_taxids": int(eligible_pairs["tax_id"].nunique()),
        "eligible_pairs": int(len(eligible_pairs)),
        "eligible_unique_compounds": int(eligible_pairs["compound_inchikey"].nunique()),
        "eligible_species_names": eligible.tolist(),
        "eligible_pathogen_class_counts": {
            str(k): int(v) for k, v in species_counts[species_counts["organism"].isin(eligible)]["pathogen_class"].value_counts().items()
        },
        "flagged_data_validity_rows_retained": 0,
    }
    (output_dir / "data_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    (output_dir / "query.sql").write_text(MIC_SQL.strip() + "\n", encoding="utf-8")
    return qc


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_structures(structures: pd.DataFrame) -> dict[str, str | None]:
    representatives = structures.drop_duplicates("compound_inchikey")
    result: dict[str, str | None] = {}
    for key, smiles in representatives[["compound_inchikey", "compound_smiles"]].itertuples(index=False):
        mol = Chem.MolFromSmiles(str(smiles))
        result[str(key)] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
    return result


def _looks_like_species(name: str) -> bool:
    value = str(name).strip()
    lowered = value.lower()
    if " " not in value or " sp." in lowered or " spp." in lowered or "/" in value:
        return False
    first, second, *_ = value.split()
    return first[0].isalpha() and second[0].isalpha()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen ChEMBL 34 species-level MIC dataset.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--organism-mapping", required=True)
    parser.add_argument("--taxonomy-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-compounds", type=int, default=500)
    args = parser.parse_args(argv)
    qc = build_species_mic_dataset(
        args.database,
        args.organism_mapping,
        args.taxonomy_mapping,
        args.output_dir,
        min_compounds=args.min_compounds,
    )
    print(json.dumps(qc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
