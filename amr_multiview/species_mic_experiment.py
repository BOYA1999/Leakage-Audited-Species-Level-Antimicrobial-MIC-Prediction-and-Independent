from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder

from .features import build_feature_matrix


def run_experiment(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.data)
    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    compound_index = pd.Series(np.arange(len(compounds)), index=compounds["compound_inchikey"])
    pairs["compound_index"] = pairs["compound_inchikey"].map(compound_index).astype(int)

    species_encoder = OneHotEncoder(handle_unknown="error", sparse_output=True, dtype=np.float32)
    species_features = species_encoder.fit_transform(pairs[["organism"]])
    representations = _load_representations(args, compounds)
    conditioning = ["species", "compound-only"] if args.conditioning == "both" else [args.conditioning]

    metrics = []
    predictions = []
    split_reports = []
    for seed in args.seeds:
        partition, split_report = scaffold_partition(compounds, seed)
        split_reports.append(split_report)
        row_partition = partition[pairs["compound_index"].to_numpy()]
        train_idx = np.flatnonzero(row_partition == "train")
        valid_idx = np.flatnonzero(row_partition == "valid")
        test_idx = np.flatnonzero(row_partition == "test")
        y = pairs["log2_mic"].to_numpy(dtype=np.float32)
        weights = _species_balanced_weights(pairs.iloc[train_idx]["organism"])

        train_medians = pairs.iloc[train_idx].assign(target=y[train_idx]).groupby("organism")["target"].median()
        baseline_valid = pairs.iloc[valid_idx]["organism"].map(train_medians).to_numpy()
        baseline_test = pairs.iloc[test_idx]["organism"].map(train_medians).to_numpy()
        _record_result(
            "species_median",
            seed,
            pairs,
            valid_idx,
            test_idx,
            y,
            baseline_valid,
            baseline_test,
            metrics,
            predictions,
        )

        for name, compound_features in representations.items():
            for conditioning_mode in conditioning:
                model_name = name if conditioning_mode == "species" else f"{name}_compound_only"
                pair_features = _pair_features(
                    compound_features,
                    pairs["compound_index"].to_numpy(),
                    species_features if conditioning_mode == "species" else None,
                )
                model = lgb.LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=args.n_estimators,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=args.n_jobs,
                    verbosity=-1,
                )
                model.fit(pair_features[train_idx], y[train_idx], sample_weight=weights)
                valid_pred = model.predict(pair_features[valid_idx])
                test_pred = model.predict(pair_features[test_idx])
                _record_result(
                    model_name,
                    seed,
                    pairs,
                    valid_idx,
                    test_idx,
                    y,
                    valid_pred,
                    test_pred,
                    metrics,
                    predictions,
                )
                del pair_features, model

    metrics_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    summary = (
        metrics_frame[metrics_frame["scope"].eq("macro")]
        .groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            spearman_mean=("spearman", "mean"),
            within_one_dilution_mean=("within_one_dilution", "mean"),
            within_two_dilutions_mean=("within_two_dilutions", "mean"),
            conformal_coverage_mean=("conformal_coverage", "mean"),
            conformal_width_mean=("conformal_width", "mean"),
            species_conformal_coverage_mean=("species_conformal_coverage", "mean"),
            species_conformal_width_mean=("species_conformal_width", "mean"),
        )
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    manifest = {
        "data": str(Path(args.data).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seeds": args.seeds,
        "representations": list(representations),
        "models": ["species_median", *[name if mode == "species" else f"{name}_compound_only" for name in representations for mode in conditioning]],
        "conditioning": args.conditioning,
        "n_estimators": args.n_estimators,
        "split": {"unit": "Bemis-Murcko scaffold", "train": 0.70, "valid": 0.15, "test": 0.15},
        "conformal_alpha": 0.10,
        "rows": int(len(pairs)),
        "compounds": int(len(compounds)),
        "species": int(pairs["organism"].nunique()),
        "split_reports": split_reports,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "lightgbm": lgb.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def scaffold_partition(compounds: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict]:
    scaffold_to_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(compounds.itertuples(index=False)):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True) if scaffold and scaffold.GetNumAtoms() else ""
        key = scaffold_smiles or f"ACYCLIC:{row.compound_inchikey}"
        scaffold_to_indices.setdefault(key, []).append(idx)

    groups = list(scaffold_to_indices.items())
    np.random.default_rng(seed).shuffle(groups)
    target_train = round(0.70 * len(compounds))
    target_valid = round(0.15 * len(compounds))
    partition = np.empty(len(compounds), dtype="U5")
    counts = {"train": 0, "valid": 0, "test": 0}
    scaffold_sets = {"train": set(), "valid": set(), "test": set()}
    for scaffold_id, indices in groups:
        if counts["train"] < target_train:
            split = "train"
        elif counts["valid"] < target_valid:
            split = "valid"
        else:
            split = "test"
        partition[indices] = split
        counts[split] += len(indices)
        scaffold_sets[split].add(scaffold_id)

    report = {
        "seed": int(seed),
        "compound_counts": counts,
        "unique_scaffolds": {k: len(v) for k, v in scaffold_sets.items()},
        "scaffold_overlap_train_valid": len(scaffold_sets["train"] & scaffold_sets["valid"]),
        "scaffold_overlap_train_test": len(scaffold_sets["train"] & scaffold_sets["test"]),
        "scaffold_overlap_valid_test": len(scaffold_sets["valid"] & scaffold_sets["test"]),
    }
    return partition, report


def _load_representations(args: argparse.Namespace, compounds: pd.DataFrame) -> dict[str, np.ndarray | sparse.csr_matrix]:
    result = {}
    view_map = {
        "morgan": ["morgan"],
        "morgan_physchem": ["morgan", "physchem"],
        "morgan_maccs": ["morgan", "maccs"],
        "morgan_graph": ["morgan", "graph"],
        "multiview": ["morgan", "maccs", "physchem", "graph"],
    }
    for name, views in view_map.items():
        if name in args.representations:
            result[name] = sparse.csr_matrix(build_feature_matrix(compounds["canonical_smiles"], views).x)
    if "mole" in args.representations:
        result["mole"] = _load_embedding_table(args.mole_embeddings, compounds)
    if "molformer" in args.representations:
        result["molformer"] = _load_embedding_npz(args.molformer_embeddings, compounds)
    return result


def _load_embedding_table(path: str, compounds: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    missing = compounds.loc[~compounds["compound_inchikey"].isin(frame.index), "compound_inchikey"]
    if len(missing):
        raise ValueError(f"MolE embeddings are missing {len(missing)} compounds")
    return frame.loc[compounds["compound_inchikey"]].to_numpy(dtype=np.float32)


def _load_embedding_npz(path: str, compounds: pd.DataFrame) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    lookup = {key: i for i, key in enumerate(payload["compound_inchikey"].astype(str))}
    missing = [key for key in compounds["compound_inchikey"] if key not in lookup]
    if missing:
        raise ValueError(f"MoLFormer embeddings are missing {len(missing)} compounds")
    order = np.asarray([lookup[key] for key in compounds["compound_inchikey"]])
    return payload["embedding"][order].astype(np.float32, copy=False)


def _pair_features(compound_features, compound_indices: np.ndarray, species_features=None):
    selected = compound_features[compound_indices]
    if species_features is None:
        return selected
    if sparse.issparse(selected):
        return sparse.hstack([selected, species_features], format="csr", dtype=np.float32)
    return np.hstack([selected, species_features.toarray()]).astype(np.float32, copy=False)


def _species_balanced_weights(species: pd.Series) -> np.ndarray:
    counts = species.value_counts()
    weights = species.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights / weights.mean()


def _record_result(model, seed, pairs, valid_idx, test_idx, y, valid_pred, test_pred, metrics, predictions):
    residual = np.abs(y[valid_idx] - valid_pred)
    conformal_q = _finite_sample_quantile(residual)
    calibration = pairs.iloc[valid_idx][["organism"]].copy()
    calibration["residual"] = residual
    species_q = calibration.groupby("organism")["residual"].apply(
        lambda values: _finite_sample_quantile(values.to_numpy())
    )
    test = pairs.iloc[test_idx][["compound_inchikey", "canonical_smiles", "organism", "tax_id", "pathogen_class"]].copy()
    test["y_true"] = y[test_idx]
    test["y_pred"] = test_pred
    test["lower_90"] = test_pred - conformal_q
    test["upper_90"] = test_pred + conformal_q
    test["species_conformal_q"] = test["organism"].map(species_q).to_numpy()
    test["lower_90_species"] = test_pred - test["species_conformal_q"]
    test["upper_90_species"] = test_pred + test["species_conformal_q"]
    test["model"] = model
    test["seed"] = seed
    predictions.append(test)

    species_rows = []
    for organism, group in test.groupby("organism", sort=False):
        row = _metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        row.update(
            {
                "model": model,
                "seed": seed,
                "scope": "species",
                "organism": organism,
                "n": len(group),
                "conformal_coverage": float(
                    ((group["y_true"] >= group["lower_90"]) & (group["y_true"] <= group["upper_90"])).mean()
                ),
                "conformal_width": 2 * conformal_q,
                "species_conformal_coverage": float(
                    ((group["y_true"] >= group["lower_90_species"]) & (group["y_true"] <= group["upper_90_species"])).mean()
                ),
                "species_conformal_width": float((group["upper_90_species"] - group["lower_90_species"]).mean()),
            }
        )
        species_rows.append(row)
    metrics.extend(species_rows)
    macro = pd.DataFrame(species_rows).mean(numeric_only=True).to_dict()
    macro.update({"model": model, "seed": seed, "scope": "macro", "organism": "__macro__", "n": len(test)})
    metrics.append(macro)


def _finite_sample_quantile(residual: np.ndarray, coverage: float = 0.90) -> float:
    residual = np.asarray(residual, dtype=float)
    rank = min(len(residual), int(np.ceil((len(residual) + 1) * coverage)))
    return float(np.partition(residual, rank - 1)[rank - 1])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    rho = spearmanr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else np.nan
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "within_one_dilution": float(np.mean(np.abs(error) <= 1.0)),
        "within_two_dilutions": float(np.mean(np.abs(error) <= 2.0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen species-level MIC benchmark.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["morgan", "morgan_physchem", "morgan_maccs", "morgan_graph", "multiview", "mole", "molformer"],
        required=True,
    )
    parser.add_argument("--mole-embeddings")
    parser.add_argument("--molformer-embeddings")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 3544, 2025, 2026])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--conditioning", choices=["species", "compound-only", "both"], default="species")
    args = parser.parse_args(argv)
    manifest = run_experiment(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
