from __future__ import annotations

import argparse
from hashlib import sha256
import json
import platform
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import rdBase
from scipy import sparse
from scipy.stats import spearmanr, t
import sklearn
from sklearn.preprocessing import OneHotEncoder

from .features import build_feature_matrix
from .species_mic_experiment import _species_balanced_weights


MODELS = ("morgan_compound_only", "morgan_pathogen_class")
COHORTS = ("all_pairs", "compound_seen_in_other_species", "exact_compound_nonoverlap")
METRICS = ("mae", "rmse", "spearman", "within_one_dilution", "within_two_dilutions")
REQUIRED_COLUMNS = {
    "compound_inchikey",
    "canonical_smiles",
    "organism",
    "pathogen_class",
    "log2_mic",
}


class UnknownHeldOutPathogenClassError(RuntimeError):
    """Raised when train-only class encoding cannot represent a held-out species."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_pairs(pairs: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(pairs.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if pairs[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required cold-start fields contain missing values.")
    duplicate_pairs = pairs.duplicated(["compound_inchikey", "organism"]).sum()
    if duplicate_pairs:
        raise ValueError(f"Input contains {duplicate_pairs} duplicate compound-species pairs.")
    smiles_per_key = pairs.groupby("compound_inchikey")["canonical_smiles"].nunique()
    if (smiles_per_key > 1).any():
        raise ValueError("At least one compound key maps to multiple canonical SMILES.")


def _limit_pairs_per_species(pairs: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return pairs.reset_index(drop=True)
    if limit < 1:
        raise ValueError("max_pairs_per_species must be positive.")
    limited = pd.concat(
        [
            group.sample(n=min(limit, len(group)), random_state=seed, replace=False)
            for _, group in pairs.groupby("organism", sort=True)
        ],
        ignore_index=False,
    )
    return limited.sort_index().reset_index(drop=True)


def _train_only_pathogen_features(
    train: pd.Series,
    test: pd.Series,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str]]:
    train_values = train.astype(str)
    test_values = test.astype(str)
    train_classes = set(train_values)
    unknown = sorted(set(test_values) - train_classes)
    if unknown:
        raise UnknownHeldOutPathogenClassError(
            "STOP_UNKNOWN_HELD_OUT_PATHOGEN_CLASS: " + ", ".join(unknown)
        )
    encoder = OneHotEncoder(handle_unknown="error", sparse_output=True, dtype=np.float32)
    train_x = encoder.fit_transform(train_values.to_frame(name="pathogen_class")).tocsr()
    test_x = encoder.transform(test_values.to_frame(name="pathogen_class")).tocsr()
    return train_x, test_x, encoder.categories_[0].astype(str).tolist()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {metric: np.nan for metric in METRICS}
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    rho = np.nan
    if len(y_true) > 1 and np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
        value = spearmanr(y_true, y_pred).statistic
        rho = float(value) if np.isfinite(value) else np.nan
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "spearman": rho,
        "within_one_dilution": float((np.abs(error) <= 1.0).mean()),
        "within_two_dilutions": float((np.abs(error) <= 2.0).mean()),
    }


def _cohort_mask(frame: pd.DataFrame, cohort: str) -> np.ndarray:
    if cohort == "all_pairs":
        return np.ones(len(frame), dtype=bool)
    if cohort == "compound_seen_in_other_species":
        return frame["compound_seen_in_other_species"].to_numpy(dtype=bool)
    if cohort == "exact_compound_nonoverlap":
        return frame["exact_compound_nonoverlap"].to_numpy(dtype=bool)
    raise ValueError(f"Unknown cohort: {cohort}")


def _regressor(args: argparse.Namespace) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )


def _overlap_checks(pairs: pd.DataFrame, held_out_species: list[str]) -> pd.DataFrame:
    rows = []
    duplicate_pairs = int(pairs.duplicated(["compound_inchikey", "organism"]).sum())
    for species in held_out_species:
        test = pairs.loc[pairs["organism"].eq(species)]
        train = pairs.loc[~pairs["organism"].eq(species)]
        train_keys = set(train["compound_inchikey"].astype(str))
        seen = test["compound_inchikey"].astype(str).isin(train_keys)
        train_classes = sorted(train["pathogen_class"].astype(str).unique())
        test_classes = sorted(test["pathogen_class"].astype(str).unique())
        unknown = sorted(set(test_classes) - set(train_classes))
        rows.append(
            {
                "held_out_species": species,
                "train_pairs": len(train),
                "test_pairs": len(test),
                "train_compounds": train["compound_inchikey"].nunique(),
                "test_compounds": test["compound_inchikey"].nunique(),
                "test_compounds_seen_in_other_species": int(seen.sum()),
                "test_compounds_exact_nonoverlap": int((~seen).sum()),
                "cohort_partition_complete": bool(len(test) == int(seen.sum()) + int((~seen).sum())),
                "train_test_species_overlap": int(bool(set(train["organism"]) & {species})),
                "train_pathogen_classes": ";".join(train_classes),
                "held_out_pathogen_classes": ";".join(test_classes),
                "unknown_held_out_pathogen_classes": ";".join(unknown),
                "pathogen_class_known_from_train": not unknown,
                "duplicate_compound_species_pairs_in_input": duplicate_pairs,
            }
        )
    return pd.DataFrame(rows)


def _paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    compound_only = metrics.loc[metrics["model"].eq(MODELS[0])].drop(columns="model")
    plus_class = metrics.loc[metrics["model"].eq(MODELS[1])].drop(columns="model")
    paired = compound_only.merge(
        plus_class,
        on=["held_out_species", "cohort", "n_pairs", "n_compounds"],
        suffixes=("_compound_only", "_pathogen_class"),
        validate="one_to_one",
    )
    for metric in METRICS:
        paired[f"delta_{metric}_pathogen_class_minus_compound_only"] = (
            paired[f"{metric}_pathogen_class"] - paired[f"{metric}_compound_only"]
        )
    return paired


def _summaries(metrics: pd.DataFrame, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    for (model, cohort), frame in metrics.groupby(["model", "cohort"], sort=True):
        row = {"model": model, "cohort": cohort, "species_total": frame["held_out_species"].nunique()}
        for metric in METRICS:
            values = frame[metric].dropna()
            row[f"{metric}_species_evaluable"] = len(values)
            row[f"{metric}_equal_species_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_equal_species_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_equal_species_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        metric_rows.append(row)

    delta_rows = []
    delta_columns = [column for column in paired if column.startswith("delta_")]
    for cohort, frame in paired.groupby("cohort", sort=True):
        for column in delta_columns:
            values = frame[column].dropna()
            mean = float(values.mean()) if len(values) else np.nan
            if len(values) > 1:
                half_width = float(
                    t.ppf(0.975, len(values) - 1)
                    * values.std(ddof=1)
                    / np.sqrt(len(values))
                )
            else:
                half_width = np.nan
            delta_rows.append(
                {
                    "cohort": cohort,
                    "contrast": column.removeprefix("delta_"),
                    "inference_unit": "held_out_species",
                    "species_evaluable": len(values),
                    "equal_species_mean_delta": mean,
                    "equal_species_median_delta": float(values.median()) if len(values) else np.nan,
                    "equal_species_sd_delta": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "equal_species_mean_delta_descriptive_stability_interval95_low": mean - half_width,
                    "equal_species_mean_delta_descriptive_stability_interval95_high": mean + half_width,
                    "species_with_delta_below_zero": int((values < 0).sum()),
                    "species_with_delta_above_zero": int((values > 0).sum()),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def run_cold_start(args: argparse.Namespace) -> dict:
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _limit_pairs_per_species(pd.read_csv(data_path), args.max_pairs_per_species, args.seed)
    _validate_pairs(pairs)
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("max_folds must be positive.")

    species = sorted(pairs["organism"].astype(str).unique())
    if args.expected_species is not None and len(species) != args.expected_species:
        raise ValueError(f"Expected {args.expected_species} species, found {len(species)}.")
    held_out_species = species[: args.max_folds] if args.max_folds is not None else species
    overlap = _overlap_checks(pairs, held_out_species)
    overlap.to_csv(output_dir / "overlap_checks.csv", index=False)

    smoke_only = args.max_folds is not None or args.max_pairs_per_species is not None
    manifest = {
        "status": "SMOKE_COMPLETE" if smoke_only else "COMPLETE",
        "data": str(data_path.resolve()),
        "data_sha256": _file_sha256(data_path),
        "output_dir": str(output_dir.resolve()),
        "design": "leave-one-species-out cold-start regression",
        "inference_unit": "held_out_species",
        "paired_delta_interval": (
            "t-scaled descriptive stability interval across held-out species; not an "
            "independent-sample confidence interval because leave-one-species-out training sets overlap"
        ),
        "cohorts": {
            "all_pairs": "all compound-species pairs for the held-out species",
            "compound_seen_in_other_species": "held-out pairs whose exact compound key occurs in training species",
            "exact_compound_nonoverlap": "held-out pairs whose exact compound key is absent from all training species",
        },
        "models": list(MODELS),
        "pathogen_class_encoding": "OneHotEncoder fitted separately on training species in each fold",
        "rows": len(pairs),
        "compounds": int(pairs["compound_inchikey"].nunique()),
        "species_available": len(species),
        "folds_planned": len(held_out_species),
        "expected_full_folds": args.expected_species,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "max_folds": args.max_folds,
        "max_pairs_per_species": args.max_pairs_per_species,
        "outputs": [
            "predictions.csv.gz",
            "metrics.csv",
            "metrics_summary.csv",
            "species_level_paired_deltas.csv",
            "paired_delta_summary.csv",
            "overlap_checks.csv",
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
    unknown = overlap.loc[~overlap["pathogen_class_known_from_train"]]
    if len(unknown):
        manifest["status"] = "STOP_UNKNOWN_HELD_OUT_PATHOGEN_CLASS"
        manifest["stopped_folds"] = unknown[
            ["held_out_species", "unknown_held_out_pathogen_classes"]
        ].to_dict("records")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise UnknownHeldOutPathogenClassError(
            "STOP_UNKNOWN_HELD_OUT_PATHOGEN_CLASS: at least one held-out class was absent from its training fold."
        )

    compounds = pairs[["compound_inchikey", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    compound_index = pd.Series(np.arange(len(compounds)), index=compounds["compound_inchikey"])
    pair_compound_index = pairs["compound_inchikey"].map(compound_index).to_numpy(dtype=int)
    compound_features = sparse.csr_matrix(
        build_feature_matrix(compounds["canonical_smiles"], ["morgan"]).x
    )
    y = pairs["log2_mic"].to_numpy(dtype=np.float32)

    prediction_frames = []
    metric_rows = []
    for fold, held_out in enumerate(held_out_species, start=1):
        test_idx = np.flatnonzero(pairs["organism"].astype(str).to_numpy() == held_out)
        train_idx = np.flatnonzero(pairs["organism"].astype(str).to_numpy() != held_out)
        train_class_x, test_class_x, encoded_classes = _train_only_pathogen_features(
            pairs.iloc[train_idx]["pathogen_class"], pairs.iloc[test_idx]["pathogen_class"]
        )
        train_compound_x = compound_features[pair_compound_index[train_idx]]
        test_compound_x = compound_features[pair_compound_index[test_idx]]
        feature_sets = {
            MODELS[0]: (train_compound_x, test_compound_x),
            MODELS[1]: (
                sparse.hstack([train_compound_x, train_class_x], format="csr", dtype=np.float32),
                sparse.hstack([test_compound_x, test_class_x], format="csr", dtype=np.float32),
            ),
        }
        weights = _species_balanced_weights(pairs.iloc[train_idx]["organism"])
        train_keys = set(pairs.iloc[train_idx]["compound_inchikey"].astype(str))
        base_columns = [
            column
            for column in [
                "compound_inchikey",
                "canonical_smiles",
                "organism",
                "tax_id",
                "pathogen_class",
            ]
            if column in pairs.columns
        ]
        base = pairs.iloc[test_idx][base_columns].copy()
        base["fold"] = fold
        base["held_out_species"] = held_out
        base["y_true"] = y[test_idx]
        base["all_pairs"] = True
        base["compound_seen_in_other_species"] = base["compound_inchikey"].astype(str).isin(train_keys)
        base["exact_compound_nonoverlap"] = ~base["compound_seen_in_other_species"]
        base["encoded_train_pathogen_classes"] = ";".join(encoded_classes)

        for model_name, (train_x, test_x) in feature_sets.items():
            model = _regressor(args)
            model.fit(train_x, y[train_idx], sample_weight=weights)
            predicted = model.predict(test_x)
            result = base.copy()
            result["model"] = model_name
            result["y_pred"] = predicted
            result["absolute_error"] = np.abs(result["y_pred"] - result["y_true"])
            prediction_frames.append(result)
            for cohort in COHORTS:
                mask = _cohort_mask(result, cohort)
                subset = result.loc[mask]
                row = _metrics(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy())
                row.update(
                    {
                        "held_out_species": held_out,
                        "model": model_name,
                        "cohort": cohort,
                        "n_pairs": len(subset),
                        "n_compounds": subset["compound_inchikey"].nunique(),
                    }
                )
                metric_rows.append(row)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    paired = _paired_deltas(metrics)
    metrics_summary, delta_summary = _summaries(metrics, paired)
    predictions.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    paired.to_csv(output_dir / "species_level_paired_deltas.csv", index=False)
    delta_summary.to_csv(output_dir / "paired_delta_summary.csv", index=False)
    manifest["folds_completed"] = len(held_out_species)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a train-only-encoded species cold-start boundary audit.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-species", type=int, default=48)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-pairs-per-species", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    manifest = run_cold_start(parse_args(argv))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
