from __future__ import annotations

import argparse
from hashlib import sha256
import json
import pickle
import platform
import sys
from pathlib import Path
from tempfile import TemporaryFile
from time import perf_counter

import numpy as np
import pandas as pd
from rdkit import rdBase
import sklearn
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import xgboost
from xgboost import XGBRegressor

from .species_mic_experiment import (
    _load_representations,
    _pair_features,
    _record_result,
    _species_balanced_weights,
    scaffold_partition,
)


DEFAULT_SEEDS = [42, 100, 3544, 2025, 2026]
REQUIRED_COLUMNS = {
    "compound_inchikey",
    "canonical_smiles",
    "organism",
    "tax_id",
    "pathogen_class",
    "log2_mic",
}


def run_experiment(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    pairs = pd.read_csv(data_path)
    _validate_species_mic_input(pairs)

    original_compounds = pairs["compound_inchikey"].nunique()
    if args.max_compounds is not None:
        if args.max_compounds < 3:
            raise ValueError("max_compounds must be at least 3.")
        selected = pairs["compound_inchikey"].drop_duplicates().iloc[: args.max_compounds]
        pairs = pairs[pairs["compound_inchikey"].isin(selected)].copy()

    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    if args.max_compounds is None and pairs["organism"].nunique() != 48:
        raise ValueError("A full E01 run requires exactly 48 species; use max_compounds only for smoke runs.")
    compound_index = pd.Series(np.arange(len(compounds)), index=compounds["compound_inchikey"])
    pairs["compound_index"] = pairs["compound_inchikey"].map(compound_index).astype(int)
    y = pairs["log2_mic"].to_numpy(dtype=np.float32)

    species_encoder = OneHotEncoder(handle_unknown="error", sparse_output=True, dtype=np.float32)
    species_features = species_encoder.fit_transform(pairs[["organism"]])
    representation_start = perf_counter()
    representations = _load_representations(args, compounds)
    representation_seconds = perf_counter() - representation_start

    split_states = {}
    split_rows = []
    compound_keys = compounds["compound_inchikey"].astype(str).to_numpy()
    pair_compound_indices = pairs["compound_index"].to_numpy()
    for seed in args.seeds:
        partition, report = scaffold_partition(compounds, seed)
        row_partition = partition[pair_compound_indices]
        train_idx = np.flatnonzero(row_partition == "train")
        valid_idx = np.flatnonzero(row_partition == "valid")
        test_idx = np.flatnonzero(row_partition == "test")
        if min(len(train_idx), len(valid_idx), len(test_idx)) == 0:
            raise ValueError(f"Seed {seed} produced an empty row partition.")
        split_states[int(seed)] = {
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "weights": _species_balanced_weights(pairs.iloc[train_idx]["organism"]),
            "report": report,
        }
        split_rows.append(
            {
                "seed": int(seed),
                "split_sha256": split_fingerprint(compound_keys, partition),
                "train_compounds": report["compound_counts"]["train"],
                "valid_compounds": report["compound_counts"]["valid"],
                "test_compounds": report["compound_counts"]["test"],
                "scaffold_overlap_train_valid": report["scaffold_overlap_train_valid"],
                "scaffold_overlap_train_test": report["scaffold_overlap_train_test"],
                "scaffold_overlap_valid_test": report["scaffold_overlap_valid_test"],
            }
        )

    metrics = []
    predictions = []
    runtime = []
    for representation_name, compound_features in representations.items():
        pair_features = _pair_features(compound_features, pair_compound_indices, species_features)
        for seed in args.seeds:
            state = split_states[int(seed)]
            for estimator_name in args.models:
                model_name = f"{estimator_name}_{representation_name}"
                model = _build_estimator(estimator_name, args, int(seed))

                started = perf_counter()
                model.fit(
                    pair_features[state["train_idx"]],
                    y[state["train_idx"]],
                    sample_weight=state["weights"],
                )
                fit_seconds = perf_counter() - started
                actual_device = "cpu"
                if estimator_name == "xgboost":
                    actual_device = _verify_xgb_device(model, args.xgb_device)

                started = perf_counter()
                valid_pred = model.predict(pair_features[state["valid_idx"]])
                valid_predict_seconds = perf_counter() - started
                started = perf_counter()
                test_pred = model.predict(pair_features[state["test_idx"]])
                test_predict_seconds = perf_counter() - started

                _record_result(
                    model_name,
                    int(seed),
                    pairs,
                    state["valid_idx"],
                    state["test_idx"],
                    y,
                    valid_pred,
                    test_pred,
                    metrics,
                    predictions,
                )
                runtime.append(
                    {
                        "model": model_name,
                        "estimator": estimator_name,
                        "representation": representation_name,
                        "seed": int(seed),
                        "fit_seconds": fit_seconds,
                        "valid_predict_seconds": valid_predict_seconds,
                        "test_predict_seconds": test_predict_seconds,
                        "predict_seconds": valid_predict_seconds + test_predict_seconds,
                        "serialized_model_bytes": _serialized_model_bytes(model),
                        "compound_feature_columns": int(compound_features.shape[1]),
                        "joint_feature_columns": int(pair_features.shape[1]),
                        "requested_device": args.xgb_device if estimator_name == "xgboost" else "cpu",
                        "actual_device": actual_device,
                    }
                )
        del pair_features

    metrics_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    runtime_frame = pd.DataFrame(runtime)
    summary = _summarize(metrics_frame, runtime_frame)
    split_frame = pd.DataFrame(split_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    runtime_frame.to_csv(output_dir / "runtime.csv", index=False)
    split_frame.to_csv(output_dir / "split_fingerprints.csv", index=False)

    metric_contract = _metric_contract()
    metric_contract_path = output_dir / "metric_contract.json"
    metric_contract_path.write_text(json.dumps(metric_contract, indent=2), encoding="utf-8")
    manifest = {
        "experiment_id": "E01",
        "dataset_kind": "species_mic_regression",
        "data": str(data_path.resolve()),
        "data_sha256": _file_sha256(data_path),
        "output_dir": str(output_dir.resolve()),
        "command": getattr(args, "command", None),
        "models": list(args.models),
        "representations": list(representations),
        "seeds": [int(seed) for seed in args.seeds],
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "xgb_device": args.xgb_device,
        "max_compounds": args.max_compounds,
        "compound_limit_applied": len(compounds) < original_compounds,
        "rows": int(len(pairs)),
        "compounds": int(len(compounds)),
        "species": int(pairs["organism"].nunique()),
        "species_categories": species_encoder.categories_[0].astype(str).tolist(),
        "split": {"unit": "Bemis-Murcko scaffold", "train": 0.70, "valid": 0.15, "test": 0.15},
        "split_fingerprints": split_rows,
        "weighting": "inverse training-species frequency, normalized to mean 1",
        "representation_build_seconds": representation_seconds,
        "metric_contract": str(metric_contract_path.resolve()),
        "metric_contract_sha256": _file_sha256(metric_contract_path),
        "outputs": [
            "metrics.csv",
            "predictions.csv.gz",
            "metrics_summary.csv",
            "runtime.csv",
            "split_fingerprints.csv",
            "metric_contract.json",
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgboost.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def split_fingerprint(compound_keys: np.ndarray, partition: np.ndarray) -> str:
    rows = sorted(zip(map(str, compound_keys), map(str, partition)))
    return sha256("".join(f"{key}\t{split}\n" for key, split in rows).encode("utf-8")).hexdigest()


def _validate_species_mic_input(pairs: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(pairs.columns))
    if missing:
        raise ValueError(f"Not a species-MIC regression table; missing columns: {missing}")
    if "best_class" in pairs.columns:
        raise ValueError("Broad five-class data are prohibited in E01.")
    target = pd.to_numeric(pairs["log2_mic"], errors="coerce")
    if not np.isfinite(target).all():
        raise ValueError("log2_mic must contain only finite numeric values.")
    structures = pairs.groupby("compound_inchikey", observed=True)["canonical_smiles"].nunique(dropna=False)
    if structures.gt(1).any():
        raise ValueError("A compound_inchikey maps to more than one canonical_smiles value.")


def _build_estimator(name: str, args: argparse.Namespace, seed: int):
    if name == "lightgbm":
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
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            criterion="squared_error",
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if name == "xgboost":
        params = {
            "objective": "reg:absoluteerror",
            "n_estimators": args.n_estimators,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "device": args.xgb_device,
            "random_state": seed,
            "n_jobs": args.n_jobs,
            "verbosity": 0,
        }
        if args.xgb_device == "cuda":
            params["fail_on_invalid_gpu_id"] = True
        return XGBRegressor(**params)
    raise ValueError(f"Unsupported E01 estimator: {name}")


def _verify_xgb_device(model: XGBRegressor, requested: str) -> str:
    config = json.loads(model.get_booster().save_config())
    actual = str(config["learner"]["generic_param"]["device"])
    matches = actual == "cpu" if requested == "cpu" else actual.startswith("cuda")
    if not matches:
        raise RuntimeError(f"XGBoost requested {requested!r} but trained on {actual!r}; no fallback is allowed.")
    return actual


def _serialized_model_bytes(model) -> int:
    with TemporaryFile() as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return int(handle.tell())


def _summarize(metrics: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
    performance = (
        metrics[metrics["scope"].eq("macro")]
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
        )
    )
    cost = (
        runtime.groupby("model", as_index=False)
        .agg(
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_total=("fit_seconds", "sum"),
            predict_seconds_mean=("predict_seconds", "mean"),
            serialized_model_bytes_mean=("serialized_model_bytes", "mean"),
            serialized_model_bytes_max=("serialized_model_bytes", "max"),
            compound_feature_columns=("compound_feature_columns", "first"),
            joint_feature_columns=("joint_feature_columns", "first"),
        )
    )
    return performance.merge(cost, on="model", how="inner", validate="one_to_one")


def _metric_contract() -> dict:
    return {
        "experiment_id": "E01",
        "task": "joint species-conditioned regression",
        "target": "median log2(MIC ug/mL)",
        "full_run_species": 48,
        "split": "the frozen five compound-level Bemis-Murcko scaffold partitions",
        "models": ["LGBMRegressor replay", "RandomForestRegressor", "XGBRegressor"],
        "representations": ["Morgan radius 2, 1024 bits", "frozen handcrafted multiview"],
        "conditioning": "one-hot species identity concatenated to molecular features",
        "training_weight": "inverse training-species frequency normalized to mean 1",
        "metrics": ["macro-species MAE", "RMSE", "Spearman", "within one dilution", "within two dilutions"],
        "uncertainty": "pooled and species-wise residual-quantile intervals are empirical coverage audits",
        "cost": "fit/predict wall seconds and in-memory pickle byte length; feature construction is separate",
        "statistical_boundary": "five split values are stability replicates, not independent samples",
        "prohibited_input": "legacy broad five-class labels, predictions, and checkpoints",
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E01 RF/XGBoost species-MIC baselines.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models", nargs="+", choices=["lightgbm", "rf", "xgboost"], default=["rf", "xgboost"]
    )
    parser.add_argument("--representations", nargs="+", choices=["morgan", "multiview"], default=["morgan", "multiview"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-compounds", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    supplied = list(sys.argv[1:] if argv is None else argv)
    args.command = [sys.executable, "-m", "amr_multiview.species_mic_jcim_baselines", *supplied]
    manifest = run_experiment(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
