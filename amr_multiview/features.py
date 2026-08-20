from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import struct
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")


PHYS_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "NumAromaticRings",
    "NumAliphaticRings",
    "LabuteASA",
]

GRAPH_FEATURES = [
    "n_atoms",
    "n_bonds",
    "n_single_bonds",
    "n_double_bonds",
    "n_triple_bonds",
    "n_aromatic_bonds",
    "n_c",
    "n_n",
    "n_o",
    "n_s",
    "n_halogen",
    "formal_charge",
    "n_rings",
    "n_aromatic_rings",
    "largest_ring_size",
]


@dataclass(frozen=True)
class FeatureMatrix:
    x: np.ndarray
    names: list[str]
    view_ranges: dict[str, tuple[int, int]]


def build_feature_matrix(
    smiles: Iterable[str],
    views: list[str],
    n_bits: int = 1024,
    radius: int = 2,
) -> FeatureMatrix:
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    if any(m is None for m in mols):
        bad = sum(m is None for m in mols)
        raise ValueError(f"Cannot featurize {bad} invalid SMILES.")

    blocks: list[np.ndarray] = []
    names: list[str] = []
    view_ranges: dict[str, tuple[int, int]] = {}

    for view in views:
        start = len(names)
        if view == "morgan":
            block, block_names = _morgan(mols, n_bits=n_bits, radius=radius)
        elif view == "mhfp6":
            block, block_names = _mhfp6(mols)
        elif view == "maccs":
            block, block_names = _maccs(mols)
        elif view == "physchem":
            block, block_names = _physchem(mols)
        elif view == "graph":
            block, block_names = _graph_stats(mols)
        elif view == "rdkit2d":
            block, block_names = _rdkit2d(mols)
        else:
            raise ValueError(f"Unknown feature view: {view}")
        blocks.append(block)
        names.extend([f"{view}:{name}" for name in block_names])
        view_ranges[view] = (start, len(names))

    x = np.hstack(blocks).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return FeatureMatrix(x=x, names=names, view_ranges=view_ranges)


def feature_indices(feature_names: list[str], views: list[str]) -> np.ndarray:
    prefixes = tuple(f"{v}:" for v in views)
    return np.asarray([i for i, name in enumerate(feature_names) if name.startswith(prefixes)])


def _morgan(mols: list[Chem.Mol], n_bits: int, radius: int) -> tuple[np.ndarray, list[str]]:
    arr = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr, [f"bit_{i}" for i in range(n_bits)]


def _maccs(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    n_bits = 167
    arr = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr, [f"key_{i}" for i in range(n_bits)]


def _mhfp6(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    try:
        from mhfp.encoder import MHFPEncoder
    except ImportError as exc:
        raise ImportError("The mhfp package is required for the mhfp6 feature view.") from exc

    n_permutations = 2048
    encoder = MHFPEncoder(n_permutations=n_permutations, seed=42)
    arr = np.zeros((len(mols), n_permutations), dtype=np.float32)
    for i, mol in enumerate(mols):
        arr[i] = _mhfp6_encode_mol_compatible(encoder, mol)
    return arr, [f"perm_{i}" for i in range(n_permutations)]


def _mhfp6_encode_mol_compatible(encoder, mol: Chem.Mol) -> np.ndarray:
    from mhfp.encoder import MHFPEncoder

    tokens = MHFPEncoder.shingling_from_mol(mol, radius=3, rings=True, kekulize=True, min_radius=1)
    hash_values = np.full((encoder.n_permutations, 1), MHFPEncoder.max_hash, dtype=np.uint64)
    perm_a = encoder.permutations_a.astype(np.uint64)
    perm_b = encoder.permutations_b.astype(np.uint64)
    prime = np.uint64(MHFPEncoder.prime)
    max_hash = np.uint64(MHFPEncoder.max_hash)
    for token in tokens:
        token_hash = np.uint64(struct.unpack("<I", sha1(token).digest()[:4])[0])
        hashes = ((perm_a * token_hash + perm_b) % prime) % max_hash
        hash_values = np.minimum(hash_values, hashes)
    return hash_values.reshape((encoder.n_permutations,)).astype(np.float32)


def _physchem(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(PHYS_DESCRIPTORS)
    arr = np.asarray([calc.CalcDescriptors(mol) for mol in mols], dtype=np.float32)
    return arr, PHYS_DESCRIPTORS


def _rdkit2d(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    names = [name for name, _ in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    arr = np.asarray([calc.CalcDescriptors(mol) for mol in mols], dtype=np.float32)
    return arr, names


def _graph_stats(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    rows = []
    for mol in mols:
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        ring_info = mol.GetRingInfo()
        atom_nums = [a.GetAtomicNum() for a in atoms]
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        rows.append(
            [
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                sum(1 for b in bonds if b.GetBondTypeAsDouble() == 1.0 and not b.GetIsAromatic()),
                sum(1 for b in bonds if b.GetBondTypeAsDouble() == 2.0),
                sum(1 for b in bonds if b.GetBondTypeAsDouble() == 3.0),
                sum(1 for b in bonds if b.GetIsAromatic()),
                atom_nums.count(6),
                atom_nums.count(7),
                atom_nums.count(8),
                atom_nums.count(16),
                sum(1 for z in atom_nums if z in {9, 17, 35, 53}),
                sum(a.GetFormalCharge() for a in atoms),
                ring_info.NumRings(),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                max(ring_sizes) if ring_sizes else 0,
            ]
        )
    return np.asarray(rows, dtype=np.float32), GRAPH_FEATURES


def admet_lite(smiles: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    names = [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
        "QED",
        "LipinskiPass",
        "LeadLikePass",
    ]
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append([np.nan] * len(names))
            continue
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        rot = Lipinski.NumRotatableBonds(mol)
        qed = Descriptors.qed(mol)
        lipinski = int(mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)
        lead_like = int(250 <= mw <= 450 and -1 <= logp <= 4 and hba <= 8 and hbd <= 4 and rot <= 10)
        rows.append([mw, logp, tpsa, hba, hbd, rot, qed, lipinski, lead_like])
    return np.asarray(rows, dtype=np.float32), names
