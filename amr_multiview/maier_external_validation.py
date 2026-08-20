from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import load_bioassay_data, validate_and_deduplicate
from .features import build_feature_matrix

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def run_validation(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main, _ = validate_and_deduplicate(load_bioassay_data(args.training_data))
    screen = pd.read_csv(args.maier_screen, sep="\t")
    library = pd.read_csv(args.maier_library, sep="\t")
    maier = library.merge(screen, on="prestwick_ID", how="inner", validate="one_to_one")
    source_maier_compounds = len(maier)
    excluded_structures = maier[maier["rdkit_no_salt"].isna()][["prestwick_ID", "chemical name"]].copy()
    maier = maier[maier["rdkit_no_salt"].notna()].reset_index(drop=True)
    strain_columns = list(screen.columns[1:])
    maier["any_activity"] = maier[strain_columns].max(axis=1).astype(int)
    main_keys = set(main["compound_inchikey"])
    maier["exact_training_overlap"] = maier["pchem_inchikey"].isin(main_keys)
    maier["nearest_training_tanimoto"] = _nearest_similarity(
        main["canonical_smiles"].tolist(), maier["rdkit_no_salt"].tolist()
    )
    maier["similarity_bin"] = pd.cut(
        maier["nearest_training_tanimoto"],
        bins=[-np.inf, 0.30, 0.50, 0.70, np.inf],
        labels=["<0.30", "0.30-0.50", "0.50-0.70", ">=0.70"],
        right=False,
    )

    representations = _representations(args, main, maier)
    broad_seed_rows = []
    y_broad = main["best_class"].ne("inactive").astype(int).to_numpy()
    for name, (x_train, x_maier) in representations.items():
        seed_predictions = []
        for seed in args.seeds:
            model = _binary_model(seed, args)
            model.fit(x_train, y_broad, sample_weight=_binary_weights(y_broad))
            pred = model.predict_proba(x_maier)[:, 1]
            seed_predictions.append(pred)
            broad_seed_rows.append(
                pd.DataFrame({"prestwick_ID": maier["prestwick_ID"], "model": name, "seed": seed, "score": pred})
            )
        stacked = np.vstack(seed_predictions)
        maier[f"broad_score_{name}"] = stacked.mean(axis=0)
        maier[f"broad_score_sd_{name}"] = stacked.std(axis=0)

    mic = pd.read_csv(args.mic_data)
    ecoli = mic[mic["organism"].eq("Escherichia coli")].copy()
    ecoli_training_keys = set(ecoli["compound_inchikey"])
    maier["ecoli_training_overlap"] = maier["pchem_inchikey"].isin(ecoli_training_keys)
    main_lookup = pd.Series(np.arange(len(main)), index=main["compound_inchikey"])
    if not ecoli["compound_inchikey"].isin(main_lookup.index).all():
        raise ValueError("E. coli MIC compounds are not fully represented in the AntiMicrobial-KG feature table")
    ecoli_indices = ecoli["compound_inchikey"].map(main_lookup).to_numpy(dtype=int)
    y_mic = ecoli["log2_mic"].to_numpy(dtype=np.float32)
    mic_seed_rows = []
    for name, (x_train, x_maier) in representations.items():
        seed_predictions = []
        for seed in args.seeds:
            model = _regression_model(seed, args)
            model.fit(x_train[ecoli_indices], y_mic)
            pred = model.predict(x_maier)
            seed_predictions.append(pred)
            mic_seed_rows.append(
                pd.DataFrame({"prestwick_ID": maier["prestwick_ID"], "model": name, "seed": seed, "pred_log2_mic": pred})
            )
        stacked = np.vstack(seed_predictions)
        maier[f"pred_log2_mic_{name}"] = stacked.mean(axis=0)
        maier[f"mic_margin_{name}"] = np.log2(0.02 * maier["ExactMolWt"].to_numpy()) - stacked.mean(axis=0)

    broad_metrics, strain_metrics, topk, bootstrap = _evaluate_broad(maier, strain_columns, list(representations), args.bootstrap)
    similarity_metrics = _evaluate_similarity_bins(maier, list(representations))
    ecoli_metrics, ecoli_bootstrap = _evaluate_ecoli(maier, strain_columns, list(representations), args.bootstrap)
    bootstrap = pd.concat([bootstrap, ecoli_bootstrap], ignore_index=True)

    prediction_columns = [
        "prestwick_ID",
        "chemical name",
        "pchem_inchikey",
        "rdkit_no_salt",
        "ExactMolWt",
        "exact_training_overlap",
        "ecoli_training_overlap",
        "nearest_training_tanimoto",
        "similarity_bin",
        "any_activity",
        *strain_columns,
    ]
    prediction_columns.extend(
        column
        for name in representations
        for column in [f"broad_score_{name}", f"broad_score_sd_{name}", f"pred_log2_mic_{name}", f"mic_margin_{name}"]
    )
    maier[prediction_columns].to_csv(output_dir / "compound_predictions.csv.gz", index=False, compression="gzip")
    pd.concat(broad_seed_rows, ignore_index=True).to_csv(output_dir / "broad_seed_predictions.csv.gz", index=False, compression="gzip")
    pd.concat(mic_seed_rows, ignore_index=True).to_csv(output_dir / "ecoli_mic_seed_predictions.csv.gz", index=False, compression="gzip")
    broad_metrics.to_csv(output_dir / "broad_metrics.csv", index=False)
    strain_metrics.to_csv(output_dir / "strain_metrics.csv", index=False)
    topk.to_csv(output_dir / "topk_enrichment.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    similarity_metrics.to_csv(output_dir / "similarity_metrics.csv", index=False)
    ecoli_metrics.to_csv(output_dir / "ecoli_mic_metrics.csv", index=False)
    excluded_structures.to_csv(output_dir / "excluded_missing_structures.csv", index=False)

    manifest = {
        "training_data": str(Path(args.training_data).resolve()),
        "mic_data": str(Path(args.mic_data).resolve()),
        "maier_screen": str(Path(args.maier_screen).resolve()),
        "maier_library": str(Path(args.maier_library).resolve()),
        "output_dir": str(output_dir.resolve()),
        "training_compounds": int(len(main)),
        "maier_source_compounds": int(source_maier_compounds),
        "maier_structurally_evaluable_compounds": int(len(maier)),
        "excluded_missing_structures": int(len(excluded_structures)),
        "primary_nonoverlap_compounds": int((~maier["exact_training_overlap"]).sum()),
        "secondary_overlap_compounds": int(maier["exact_training_overlap"].sum()),
        "primary_ecoli_exact_overlap_compounds": int(
            ((~maier["exact_training_overlap"]) & maier["ecoli_training_overlap"]).sum()
        ),
        "maier_strains": len(strain_columns),
        "exact_species_transfer": [column for column in strain_columns if column.startswith("Escherichia coli ")],
        "representations": list(representations),
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "bootstrap": args.bootstrap,
        "environment": {"lightgbm": lgb.__version__, "rdkit": rdBase.rdkitVersion},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _representations(args, main: pd.DataFrame, maier: pd.DataFrame):
    result = {}
    main_smiles = main["canonical_smiles"]
    maier_smiles = maier["rdkit_no_salt"]
    view_map = {
        "morgan": ["morgan"],
        "morgan_physchem": ["morgan", "physchem"],
        "morgan_maccs": ["morgan", "maccs"],
        "morgan_graph": ["morgan", "graph"],
        "multiview": ["morgan", "maccs", "physchem", "graph"],
    }
    for name, views in view_map.items():
        if name not in args.representations:
            continue
        result[name] = (
            sparse.csr_matrix(build_feature_matrix(main_smiles, views).x),
            sparse.csr_matrix(build_feature_matrix(maier_smiles, views).x),
        )
    if "mole" in args.representations:
        if not args.main_mole_embeddings or not args.maier_mole_embeddings:
            raise ValueError("MolE representations require --main-mole-embeddings and --maier-mole-embeddings")
        main_mole = pd.read_csv(args.main_mole_embeddings, sep="\t", index_col=0)
        maier_mole = pd.read_csv(args.maier_mole_embeddings, sep="\t", index_col=0)
        result["mole"] = (
            main_mole.loc[main["compound_inchikey"]].to_numpy(dtype=np.float32),
            maier_mole.loc[maier["prestwick_ID"]].to_numpy(dtype=np.float32),
        )
    if "molformer" in args.representations:
        if not args.molformer_embeddings:
            raise ValueError("MoLFormer representations require --molformer-embeddings")
        payload = np.load(args.molformer_embeddings, allow_pickle=False)
        lookup = {key: i for i, key in enumerate(payload["compound_inchikey"].astype(str))}
        result["molformer"] = (
            payload["embedding"][[lookup[key] for key in main["compound_inchikey"]]].astype(np.float32, copy=False),
            payload["embedding"][[lookup[key] for key in maier["pchem_inchikey"]]].astype(np.float32, copy=False),
        )
    return result


def _binary_model(seed, args):
    return lgb.LGBMClassifier(
        objective="binary",
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


def _regression_model(seed, args):
    return lgb.LGBMRegressor(
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


def _binary_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    weights = np.asarray([1.0 / counts[value] for value in y])
    return weights / weights.mean()


def _nearest_similarity(training_smiles: list[str], query_smiles: list[str]) -> np.ndarray:
    training = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in training_smiles]
    result = np.empty(len(query_smiles), dtype=np.float32)
    for i, smiles in enumerate(query_smiles):
        query = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)
        result[i] = max(DataStructs.BulkTanimotoSimilarity(query, training))
    return result


def _metric(y, score):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {"auroc": roc_auc_score(y, score), "auprc": average_precision_score(y, score)}


def _evaluate_broad(maier, strain_columns, models, n_bootstrap):
    broad_rows, strain_rows, topk_rows, bootstrap_rows = [], [], [], []
    cohorts = {"primary_nonoverlap": ~maier["exact_training_overlap"], "secondary_exact_overlap": maier["exact_training_overlap"]}
    endpoints = {"any_activity": maier["any_activity"], **{column: maier[column] for column in strain_columns}}
    for cohort, mask in cohorts.items():
        for model in models:
            score = maier.loc[mask, f"broad_score_{model}"].to_numpy()
            for endpoint, values in endpoints.items():
                y = values[mask].to_numpy(dtype=int)
                row = {"cohort": cohort, "model": model, "endpoint": endpoint, "n": len(y), "positives": int(y.sum()), "prevalence": y.mean(), **_metric(y, score)}
                (broad_rows if endpoint == "any_activity" else strain_rows).append(row)
            if cohort == "primary_nonoverlap":
                y = maier.loc[mask, "any_activity"].to_numpy(dtype=int)
                prevalence = y.mean()
                for k in [25, 50, 100]:
                    order = np.argsort(-score)[:k]
                    topk_rows.append({"model": model, "k": k, "hits": int(y[order].sum()), "precision": y[order].mean(), "enrichment": y[order].mean() / prevalence})
                bootstrap_rows.extend(_bootstrap_metrics(y, score, model, "broad_any_activity", n_bootstrap))
    return pd.DataFrame(broad_rows), pd.DataFrame(strain_rows), pd.DataFrame(topk_rows), pd.DataFrame(bootstrap_rows)


def _evaluate_similarity_bins(maier, models):
    primary = maier[~maier["exact_training_overlap"]]
    rows = []
    for similarity_bin, group in primary.groupby("similarity_bin", observed=True):
        for model in models:
            y = group["any_activity"].to_numpy(dtype=int)
            score = group[f"broad_score_{model}"].to_numpy()
            rows.append({"similarity_bin": str(similarity_bin), "model": model, "n": len(group), "positives": int(y.sum()), "prevalence": y.mean(), **_metric(y, score)})
    return pd.DataFrame(rows)


def _evaluate_ecoli(maier, strain_columns, models, n_bootstrap):
    primary = maier[~maier["exact_training_overlap"] & ~maier["ecoli_training_overlap"]]
    ecoli_columns = [column for column in strain_columns if column.startswith("Escherichia coli ")]
    rows = []
    bootstrap_rows = []
    for model in models:
        score = primary[f"mic_margin_{model}"].to_numpy()
        for column in ecoli_columns:
            y = primary[column].to_numpy(dtype=int)
            rows.append({"model": model, "strain": column, "n": len(y), "positives": int(y.sum()), "prevalence": y.mean(), **_metric(y, score)})
        pooled = primary[ecoli_columns].max(axis=1).to_numpy(dtype=int)
        rows.append({"model": model, "strain": "E. coli any strain", "n": len(pooled), "positives": int(pooled.sum()), "prevalence": pooled.mean(), **_metric(pooled, score)})
        bootstrap_rows.extend(_bootstrap_metrics(pooled, score, model, "ecoli_mic_margin", n_bootstrap))
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def _bootstrap_metrics(y, score, model, endpoint, n_bootstrap):
    rng = np.random.default_rng(20260729)
    values = {"auroc": [], "auprc": []}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), len(y))
        metric = _metric(y[index], score[index])
        for name in values:
            if np.isfinite(metric[name]):
                values[name].append(metric[name])
    return [
        {"model": model, "endpoint": endpoint, "metric": name, "estimate": _metric(y, score)[name], "ci_low": np.quantile(samples, 0.025), "ci_high": np.quantile(samples, 0.975), "bootstrap": n_bootstrap}
        for name, samples in values.items()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run leakage-controlled Maier 40-strain external validation.")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--mic-data", required=True)
    parser.add_argument("--maier-screen", required=True)
    parser.add_argument("--maier-library", required=True)
    parser.add_argument("--main-mole-embeddings")
    parser.add_argument("--maier-mole-embeddings")
    parser.add_argument("--molformer-embeddings")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["morgan", "morgan_physchem", "morgan_maccs", "morgan_graph", "multiview", "mole", "molformer"],
        default=["morgan", "multiview", "mole", "molformer"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 3544, 2025, 2026])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    manifest = run_validation(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
