from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


SEEDS = [42, 100, 3544, 2025, 2026]
METRICS = ["mae", "rmse", "spearman", "within_one_dilution", "within_two_dilutions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the frozen reliability-audit experiments.")
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--conditioning-species-deltas", type=Path, required=True)
    return parser.parse_args()


def stability(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(values.mean())
    sd = float(values.std()) if len(values) > 1 else np.nan
    half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return {"n": len(values), "mean": mean, "sd": sd, "low": mean - half, "high": mean + half}


def model_results(experiments: Path, original_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = pd.read_csv(original_dir / "metrics.csv")
    original = original[original["scope"].eq("macro") & original["model"].isin(["species_median", "mole", "molformer"])].copy()
    original["pipeline"] = original["model"].map(
        {"species_median": "Species median", "mole": "MolE + LightGBM", "molformer": "MoLFormer + LightGBM"}
    )

    frames = [original]
    for estimator, folder in (
        ("LightGBM", "lightgbm_replay"),
        ("Random Forest", "rf_full"),
        ("XGBoost", "xgb_full"),
    ):
        frame = pd.read_csv(experiments / "baselines" / folder / "metrics.csv")
        frame = frame[frame["scope"].eq("macro")].copy()
        frame["pipeline"] = np.where(
            frame["model"].str.endswith("_multiview"),
            f"Multi-view + {estimator}",
            f"Morgan + {estimator}",
        )
        frames.append(frame)

    chemprop = []
    for seed in SEEDS:
        frame = pd.read_csv(experiments / "chemprop" / f"seed{seed}" / "full" / "metrics_summary.csv")
        chemprop.append(frame[frame["scope"].eq("macro")])
    chemprop = pd.concat(chemprop, ignore_index=True)
    chemprop["pipeline"] = "Chemprop D-MPNN"
    frames.append(chemprop)

    keep = [*METRICS, "pipeline", "seed"]
    by_seed = pd.concat([frame[keep] for frame in frames], ignore_index=True)
    by_seed["seed"] = by_seed["seed"].astype(int)
    by_seed = by_seed.sort_values(["pipeline", "seed"]).reset_index(drop=True)

    summary = by_seed.groupby("pipeline", as_index=False).agg(
        seeds=("seed", "nunique"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        spearman_mean=("spearman", "mean"),
        spearman_sd=("spearman", "std"),
        within_one_mean=("within_one_dilution", "mean"),
        within_one_sd=("within_one_dilution", "std"),
        within_two_mean=("within_two_dilutions", "mean"),
        within_two_sd=("within_two_dilutions", "std"),
    )

    contrasts = [
        ("Multi-view + LightGBM", "Morgan + LightGBM"),
        ("Multi-view + Random Forest", "Morgan + Random Forest"),
        ("Multi-view + XGBoost", "Morgan + XGBoost"),
        ("Chemprop D-MPNN", "Morgan + LightGBM"),
        ("Chemprop D-MPNN", "Multi-view + LightGBM"),
        ("MolE + LightGBM", "Morgan + LightGBM"),
        ("MoLFormer + LightGBM", "Morgan + LightGBM"),
    ]
    paired_rows = []
    indexed = by_seed.set_index(["pipeline", "seed"])
    for candidate, reference in contrasts:
        for metric in METRICS:
            delta = indexed.loc[candidate, metric] - indexed.loc[reference, metric]
            stats = stability(delta)
            paired_rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "orientation": "candidate_minus_reference",
                    **stats,
                    "favorable_seeds": int((delta < 0).sum()) if metric in {"mae", "rmse"} else int((delta > 0).sum()),
                }
            )
    return by_seed, summary, pd.DataFrame(paired_rows)


def complexity_table(model_summary: pd.DataFrame, experiments: Path) -> pd.DataFrame:
    summary_mae = model_summary.set_index("pipeline")["mae_mean"]
    rows = []
    for estimator, folder, estimator_budget in (
        ("LightGBM", "lightgbm_replay", "300 boosting iterations"),
        ("Random Forest", "rf_full", "100 unrestricted-depth trees"),
        ("XGBoost", "xgb_full", "300 boosting iterations"),
    ):
        runtime = pd.read_csv(experiments / "baselines" / folder / "runtime.csv")
        for representation, compound_dim in (("morgan", 1024), ("multiview", 1218)):
            group = runtime[runtime["representation"].eq(representation)]
            actual_devices = group["actual_device"].dropna().astype(str).unique()
            if len(actual_devices) != 1:
                raise ValueError(f"non-unique recorded device for {estimator}/{representation}: {actual_devices}")
            pipeline = f"{representation.replace('morgan', 'Morgan').replace('multiview', 'Multi-view')} + {estimator}"
            rows.append(
                {
                    "pipeline": pipeline,
                    "molecular_dimension": compound_dim,
                    "organism_dimension": 48,
                    "trainable_parameters": np.nan,
                    "fit_seconds_mean": group["fit_seconds"].mean(),
                    "prediction_seconds_mean": group["predict_seconds"].mean(),
                    "prediction_scope": "observed validation and test pairs",
                    "serialized_object_bytes_mean": group["serialized_model_bytes"].mean(),
                    "serialization_measure": "in-memory pickle byte length",
                    "estimator_budget": estimator_budget,
                    "device": actual_devices[0],
                    "mae_mean": summary_mae[pipeline],
                }
            )
    runtimes = [
        json.loads((experiments / "chemprop" / f"seed{seed}" / "full" / "runtime.json").read_text())
        for seed in SEEDS
    ]
    chemprop_logs = [
        (experiments / "chemprop" / f"seed{seed}" / "full" / "train.log").read_text(
            encoding="utf-8", errors="replace"
        )
        for seed in SEEDS
    ]
    if not all("GPU available: True (cuda), used: True" in log for log in chemprop_logs):
        raise ValueError("Chemprop runtime logs do not consistently record CUDA use")
    rows.append(
        {
            "pipeline": "Chemprop D-MPNN",
            "molecular_dimension": np.nan,
            "organism_dimension": 48,
            "trainable_parameters": 332448,
            "fit_seconds_mean": np.mean([row["train_seconds"] for row in runtimes]),
            "prediction_seconds_mean": np.mean([row["predict_seconds"] for row in runtimes]),
            "prediction_scope": "63,486 compounds x 48 tasks",
            "serialized_object_bytes_mean": np.mean([row["deployable_best_pt_bytes"] for row in runtimes]),
            "serialization_measure": "saved best.pt checkpoint bytes",
            "estimator_budget": "50-epoch ceiling with early stopping",
            "device": "cuda (recorded used=True)",
            "mae_mean": summary_mae["Chemprop D-MPNN"],
        }
    )
    for pipeline, dimension in (("MolE + LightGBM", 1000), ("MoLFormer + LightGBM", 768)):
        rows.append(
            {
                "pipeline": pipeline,
                "molecular_dimension": dimension,
                "organism_dimension": 48,
                "trainable_parameters": np.nan,
                "fit_seconds_mean": np.nan,
                "prediction_seconds_mean": np.nan,
                "prediction_scope": "historical extraction and fitting not uniformly instrumented",
                "serialized_object_bytes_mean": np.nan,
                "serialization_measure": "not uniformly recorded",
                "estimator_budget": "fixed embedding plus 300 LightGBM iterations",
                "device": "not uniformly recorded",
                "mae_mean": summary_mae[pipeline],
            }
        )
    return pd.DataFrame(rows)


def temporal_tables(experiments: Path) -> dict[str, pd.DataFrame]:
    folder = experiments / "temporal" / "full"
    metrics = pd.read_csv(folder / "metrics.csv")
    species = metrics[metrics["scope"].eq("species")]
    pivot = species.pivot(index="organism", columns="model", values=METRICS)
    paired = pivot.xs("multiview", level=1, axis=1) - pivot.xs("morgan", level=1, axis=1)
    paired.index.name = "organism"
    paired = paired.reset_index()
    summary_rows = []
    for metric in METRICS:
        stats = stability(paired[metric])
        summary_rows.append(
            {
                "contrast": "temporal_multiview_minus_morgan",
                "metric": metric,
                "inference_unit": "test species",
                **stats,
                "favorable_species": int((paired[metric] < 0).sum()) if metric in {"mae", "rmse"} else int((paired[metric] > 0).sum()),
            }
        )
    support = pd.read_csv(folder / "species_counts.csv")
    support_sensitivity = []
    for threshold in (1, 10, 20, 50, 100):
        for model, group in species[species["n_pairs"].ge(threshold)].groupby("model"):
            support_sensitivity.append(
                {
                    "minimum_test_pairs": threshold,
                    "model": model,
                    "species": group["organism"].nunique(),
                    **{f"{metric}_macro": group[metric].mean() for metric in METRICS},
                }
            )
    overlap = json.loads((folder / "overlap_audit.json").read_text())
    overlap_table = pd.DataFrame(
        [{"check": key, "value": value} for key, value in overlap.items() if not isinstance(value, (dict, list))]
    )
    return {
        "metrics": metrics,
        "paired_by_species": paired,
        "paired_summary": pd.DataFrame(summary_rows),
        "species_support": support,
        "support_sensitivity": pd.DataFrame(support_sensitivity),
        "overlap": overlap_table,
    }


def chemical_tables(
    experiments: Path,
    tables: Path,
    conditioning_species_deltas: Path,
) -> dict[str, pd.DataFrame]:
    folder = experiments / "chemical_space" / "full"
    bins = pd.read_csv(folder / "similarity_error_bins.csv")
    bins = bins[bins["model"].isin(["morgan", "multiview"])]
    pivot = bins.pivot(index=["seed", "similarity_bin"], columns="model", values="mean_absolute_error").reset_index()
    pivot["multiview_minus_morgan_mae"] = pivot["multiview"] - pivot["morgan"]
    bin_rows = []
    for similarity_bin, group in pivot.groupby("similarity_bin", sort=False):
        stats = stability(group["multiview_minus_morgan_mae"])
        count = bins[(bins["model"].eq("morgan")) & bins["similarity_bin"].eq(similarity_bin)]["n_compounds"].mean()
        bin_rows.append(
            {
                "similarity_bin": similarity_bin,
                "test_compounds_mean": count,
                "morgan_mae_mean": group["morgan"].mean(),
                "multiview_mae_mean": group["multiview"].mean(),
                **{f"delta_{key}": value for key, value in stats.items()},
            }
        )
    cliff_species = pd.read_csv(folder / "apparent_mic_cliff_species_summary.csv")
    cliff_burden = cliff_species.groupby("configuration", as_index=False).agg(
        species=("organism", "nunique"),
        evaluable_scaffolds=("n_evaluable_scaffolds", "sum"),
        cliff_scaffolds=("n_scaffolds_with_apparent_cliff", "sum"),
        species_scaffold_memberships=("n_compounds_in_evaluable_scaffolds", "sum"),
        cliff_participant_memberships=("n_unique_compounds_in_apparent_cliffs", "sum"),
        possible_pairs=("n_possible_within_scaffold_pairs", "sum"),
        cliff_edges_descriptive=("n_apparent_cliff_edges_descriptive_only", "sum"),
    )
    cliff_burden["cliff_scaffold_fraction"] = cliff_burden["cliff_scaffolds"] / cliff_burden["evaluable_scaffolds"]
    cliff_burden["cliff_participant_fraction"] = cliff_burden["cliff_participant_memberships"] / cliff_burden["species_scaffold_memberships"]
    cliff_burden["cliff_edge_fraction"] = cliff_burden["cliff_edges_descriptive"] / cliff_burden["possible_pairs"]
    cliff_error = pd.read_csv(folder / "cliff_error_summary.csv")
    cliff_error = cliff_error[
        cliff_error["summary_level"].eq("across_seed_descriptive_stability")
        & cliff_error["model"].isin(["morgan", "multiview"])
    ]
    similarity = pd.read_csv(folder / "similarity_error_summary.csv")
    similarity = similarity[
        similarity["scope"].eq("all_with_compound_specific_acyclic_fallback")
        & similarity["inference_unit"].eq("equal_compound")
        & similarity["model"].isin(["morgan", "multiview"])
    ]
    similarity_summary = similarity.groupby("model", as_index=False).agg(
        seeds=("seed", "nunique"),
        mean_absolute_error=("mean_absolute_error", "mean"),
        spearman_similarity_error=("spearman_similarity_absolute_error", "mean"),
        spearman_similarity_error_sd=("spearman_similarity_absolute_error", "std"),
    )
    compound_error = pd.read_csv(
        folder / "similarity_error_by_compound.csv.gz",
        usecols=["model", "seed", "compound_inchikey", "is_acyclic", "absolute_error_mean_across_species"],
    )
    compound_error = compound_error[compound_error["model"].isin(["morgan", "multiview"])]
    corpus_scaffolds = pd.read_csv(
        folder / "compound_scaffolds.csv.gz",
        usecols=["compound_inchikey", "is_acyclic"],
    ).drop_duplicates("compound_inchikey")
    corpus_compounds = int(corpus_scaffolds["compound_inchikey"].nunique())
    corpus_acyclic_compounds = int(
        corpus_scaffolds.loc[corpus_scaffolds["is_acyclic"], "compound_inchikey"].nunique()
    )
    acyclic_error_rows = []
    for model, group in compound_error.groupby("model", sort=True):
        by_seed = group.groupby(["seed", "is_acyclic"])["absolute_error_mean_across_species"].agg(
            ["mean", "size"]
        )
        means = by_seed["mean"].unstack("is_acyclic").rename(columns={False: "cyclic", True: "acyclic"})
        counts = by_seed["size"].unstack("is_acyclic").rename(columns={False: "cyclic", True: "acyclic"})
        means["overall"] = group.groupby("seed")["absolute_error_mean_across_species"].mean()
        acyclic_delta = stability(means["acyclic"] - means["cyclic"])
        overall_shift = stability(means["overall"] - means["cyclic"])
        acyclic_error_rows.append(
            {
                "model": model,
                "seeds": int(means.index.nunique()),
                "cyclic_compounds_mean": float(counts["cyclic"].mean()),
                "cyclic_test_compounds_min": int(counts["cyclic"].min()),
                "cyclic_test_compounds_max": int(counts["cyclic"].max()),
                "acyclic_compounds_mean": float(counts["acyclic"].mean()),
                "acyclic_test_compounds_min": int(counts["acyclic"].min()),
                "acyclic_test_compounds_max": int(counts["acyclic"].max()),
                "acyclic_test_compounds_union": int(
                    group.loc[group["is_acyclic"], "compound_inchikey"].nunique()
                ),
                "full_corpus_compounds": corpus_compounds,
                "full_corpus_acyclic_compounds": corpus_acyclic_compounds,
                "cyclic_mae_mean": float(means["cyclic"].mean()),
                "cyclic_mae_sd": float(means["cyclic"].std()),
                "acyclic_mae_mean": float(means["acyclic"].mean()),
                "acyclic_mae_sd": float(means["acyclic"].std()),
                "acyclic_minus_cyclic_mae_mean": acyclic_delta["mean"],
                "acyclic_minus_cyclic_mae_sd": acyclic_delta["sd"],
                "acyclic_minus_cyclic_mae_low": acyclic_delta["low"],
                "acyclic_minus_cyclic_mae_high": acyclic_delta["high"],
                "overall_mae_mean": float(means["overall"].mean()),
                "overall_mae_sd": float(means["overall"].std()),
                "overall_minus_cyclic_mae_mean": overall_shift["mean"],
                "overall_minus_cyclic_mae_sd": overall_shift["sd"],
                "overall_minus_cyclic_mae_low": overall_shift["low"],
                "overall_minus_cyclic_mae_high": overall_shift["high"],
                "stability_unit": "five overlapping scaffold splits; descriptive only",
            }
        )
    scaffold_summary = pd.read_csv(folder / "species_scaffold_summary.csv")
    scaffold_summary = scaffold_summary[
        scaffold_summary["assignment_mode"].eq("compound_specific_acyclic_fallback")
    ].copy()
    performance = pd.read_csv(tables / "supplementary_table_s1_species.csv")[[
        "organism", "multiview_mae", "multiview_minus_morgan_mae"
    ]]
    conditioning = pd.read_csv(conditioning_species_deltas)[["organism", "delta_mae"]].rename(
        columns={"delta_mae": "species_aware_minus_compound_only_mae"}
    )
    species_analysis = scaffold_summary.merge(performance, on="organism", validate="one_to_one").merge(
        conditioning, on="organism", validate="one_to_one"
    )
    association_rows = []
    for predictor in (
        "unique_compounds",
        "unique_scaffolds",
        "unique_scaffold_fraction",
        "normalized_scaffold_entropy",
        "largest_scaffold_share",
        "acyclic_fraction",
    ):
        for outcome in (
            "multiview_mae",
            "multiview_minus_morgan_mae",
            "species_aware_minus_compound_only_mae",
        ):
            rho, _ = spearmanr(species_analysis[predictor], species_analysis[outcome], nan_policy="omit")
            association_rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "species": int(species_analysis[[predictor, outcome]].dropna().shape[0]),
                    "spearman_rho_descriptive": float(rho),
                }
            )
    return {
        "similarity_bins": pd.DataFrame(bin_rows),
        "similarity_by_seed": similarity,
        "similarity_summary": similarity_summary,
        "acyclic_error_summary": pd.DataFrame(acyclic_error_rows),
        "cliff_burden": cliff_burden,
        "cliff_error": cliff_error,
        "acyclic_sensitivity": pd.read_csv(folder / "acyclic_fallback_sensitivity.csv"),
        "species_scaffold_summary": scaffold_summary,
        "species_scaffold_associations": pd.DataFrame(association_rows),
    }


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    by_seed, model_summary, model_paired = model_results(args.experiments_dir, args.original_dir)
    outputs = {
        "jcim_model_benchmark_by_seed.csv": by_seed,
        "jcim_model_benchmark_summary.csv": model_summary,
        "jcim_model_paired_differences.csv": model_paired,
        "jcim_complexity_summary.csv": complexity_table(model_summary, args.experiments_dir),
    }
    outputs.update({f"jcim_temporal_{name}.csv": frame for name, frame in temporal_tables(args.experiments_dir).items()})
    cold = args.experiments_dir / "cold_start" / "full"
    outputs["jcim_cold_start_summary.csv"] = pd.read_csv(cold / "metrics_summary.csv")
    outputs["jcim_cold_start_paired.csv"] = pd.read_csv(cold / "paired_delta_summary.csv")
    outputs["jcim_cold_start_overlap.csv"] = pd.read_csv(cold / "overlap_checks.csv")
    outputs.update(
        {
            f"jcim_chemical_{name}.csv": frame
            for name, frame in chemical_tables(
                args.experiments_dir,
                args.tables_dir,
                args.conditioning_species_deltas,
            ).items()
        }
    )
    for name, frame in outputs.items():
        frame.to_csv(args.tables_dir / name, index=False)
        frame.to_csv(args.report_dir / name, index=False)
    manifest = {name: {"rows": len(frame), "columns": list(frame.columns)} for name, frame in outputs.items()}
    (args.report_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()



