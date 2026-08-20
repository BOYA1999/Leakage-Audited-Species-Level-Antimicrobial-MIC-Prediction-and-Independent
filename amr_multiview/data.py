from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

LABEL_ORDER = ["gram-negative", "gram-positive", "acid-fast", "fungi", "inactive"]
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_qc: dict


def load_bioassay_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bioassay data not found: {path}")
    df = pd.read_csv(path, sep="\t")
    required = {"compound_inchikey", "compound_smiles", "best_class"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df.copy()
    df["best_class"] = df["best_class"].astype(str)
    df = df[df["best_class"].isin(LABEL_ORDER)].reset_index(drop=True)
    return df


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def validate_and_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    invalid = 0
    for row in df.itertuples(index=False):
        can = canonicalize_smiles(getattr(row, "compound_smiles"))
        if can is None:
            invalid += 1
            continue
        row_dict = row._asdict()
        row_dict["canonical_smiles"] = can
        rows.append(row_dict)

    clean = pd.DataFrame(rows)
    before_dedup = len(clean)
    clean = clean.drop_duplicates("compound_inchikey", keep="first").reset_index(drop=True)

    qc = {
        "input_rows": int(len(df)),
        "invalid_smiles": int(invalid),
        "valid_rows": int(before_dedup),
        "deduplicated_rows": int(len(clean)),
        "dropped_duplicate_inchikeys": int(before_dedup - len(clean)),
        "label_counts": _counter_to_dict(clean["best_class"]),
        "source_counts": _counter_to_dict(clean.get("compound_source", pd.Series(dtype=str))),
    }
    return clean, qc


def stratified_subsample(
    df: pd.DataFrame, sample_size: int | None, seed: int, label_col: str = "best_class"
) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True)
    fractions = sample_size / len(df)
    groups = []
    for _, group in df.groupby(label_col, group_keys=False):
        n = max(1, int(round(len(group) * fractions)))
        groups.append(group.sample(n, random_state=seed))
    sampled = pd.concat(groups, ignore_index=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(sample_size, random_state=seed)
    return sampled.reset_index(drop=True)


def make_split(
    df: pd.DataFrame,
    strategy: str,
    seed: int,
    test_size: float = 0.2,
    label_col: str = "best_class",
) -> SplitResult:
    if strategy == "random":
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=seed,
            stratify=df[label_col],
        )
    elif strategy == "scaffold":
        train_idx, test_idx = scaffold_split(df, seed=seed, test_size=test_size)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    split_qc = scaffold_leakage_report(df, train_idx, test_idx)
    split_qc.update(
        {
            "strategy": strategy,
            "seed": int(seed),
            "test_size": float(test_size),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_label_counts": _counter_to_dict(df.iloc[train_idx][label_col]),
            "test_label_counts": _counter_to_dict(df.iloc[test_idx][label_col]),
        }
    )
    return SplitResult(np.asarray(train_idx), np.asarray(test_idx), split_qc)


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""


def scaffold_split(df: pd.DataFrame, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "compound_smiles"
    for idx, smiles in enumerate(df[smiles_col]):
        scaffold_to_indices[scaffold_from_smiles(smiles)].append(idx)

    rng = np.random.default_rng(seed)
    groups = list(scaffold_to_indices.values())
    rng.shuffle(groups)

    n_test_target = int(round(len(df) * test_size))
    test_idx: list[int] = []
    train_idx: list[int] = []
    for group in groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    if not train_idx or not test_idx:
        raise ValueError("Scaffold split produced an empty train or test set.")
    return np.asarray(train_idx), np.asarray(test_idx)


def scaffold_leakage_report(df: pd.DataFrame, train_idx: Iterable[int], test_idx: Iterable[int]) -> dict:
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "compound_smiles"
    scaffolds = [scaffold_from_smiles(s) for s in df[smiles_col]]
    train_scaffolds = {scaffolds[i] for i in train_idx}
    test_scaffolds = {scaffolds[i] for i in test_idx}
    overlap = train_scaffolds.intersection(test_scaffolds)
    return {
        "train_unique_scaffolds": int(len(train_scaffolds)),
        "test_unique_scaffolds": int(len(test_scaffolds)),
        "scaffold_overlap_count": int(len(overlap)),
        "scaffold_overlap_fraction_test": float(len(overlap) / max(1, len(test_scaffolds))),
    }


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _counter_to_dict(values: Iterable) -> dict:
    return {str(k): int(v) for k, v in Counter(values).items()}
