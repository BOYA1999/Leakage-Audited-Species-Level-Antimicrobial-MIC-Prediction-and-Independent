from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import lightgbm
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import t
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


RDLogger.DisableLog("rdApp.*")


MODELS = ["species_median", "morgan", "multiview", "mole", "molformer"]
LEARNED_MODELS = MODELS[1:]
METRICS = [
    "mae",
    "rmse",
    "spearman",
    "within_one_dilution",
    "within_two_dilutions",
    "conformal_coverage",
    "conformal_width",
]


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    half_width = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean - half_width, mean + half_width


def paired_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for metric, fn in [("auroc", roc_auc_score), ("auprc", average_precision_score)]:
        estimate = float(fn(y, score_a) - fn(y, score_b))
        differences = []
        while len(differences) < bootstrap:
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size == 2:
                differences.append(float(fn(y[idx], score_a[idx]) - fn(y[idx], score_b[idx])))
        low, high = np.quantile(differences, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "difference": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap": bootstrap,
            }
        )
    return rows


def classification_metrics(frame: pd.DataFrame, cohort: str, endpoint: str, score_prefix: str) -> pd.DataFrame:
    y = frame[endpoint].to_numpy(dtype=int)
    rows = []
    for model in LEARNED_MODELS:
        score = frame[f"{score_prefix}_{model}"].to_numpy(dtype=float)
        rows.append(
            {
                "cohort": cohort,
                "model": model,
                "endpoint": endpoint,
                "n": len(frame),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, score)),
                "auprc": float(average_precision_score(y, score)),
            }
        )
    return pd.DataFrame(rows)


def topk_metrics(frame: pd.DataFrame, cohort: str, endpoint: str, score_prefix: str) -> pd.DataFrame:
    prevalence = float(frame[endpoint].mean())
    rows = []
    for model in LEARNED_MODELS:
        ordered = frame.sort_values(f"{score_prefix}_{model}", ascending=False)
        for k in (25, 50, 100):
            hits = int(ordered.head(k)[endpoint].sum())
            precision = hits / k
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "model": model,
                    "k": k,
                    "hits": hits,
                    "precision": precision,
                    "enrichment": precision / prevalence,
                }
            )
    return pd.DataFrame(rows)


def coverage_audit(species_dir: Path, output: Path) -> None:
    predictions = pd.read_csv(species_dir / "predictions.csv.gz")
    by_seed = []
    by_species = []
    weighting_rows = []
    for (model, seed), frame in predictions.groupby(["model", "seed"], sort=True):
        covered = frame["y_true"].between(frame["lower_90"], frame["upper_90"])
        compound_coverage = covered.groupby(frame["compound_inchikey"]).mean()
        weighting_rows.append({
            "model": model, "seed": int(seed), "test_pairs": len(frame),
            "test_compounds": len(compound_coverage),
            "pair_weighted_coverage": float(covered.mean()),
            "equal_compound_coverage": float(compound_coverage.mean()),
        })
        species = []
        for organism, group in frame.groupby("organism", sort=True):
            species_covered = group["y_true"].between(group["lower_90"], group["upper_90"])
            value = float(species_covered.mean())
            species.append(value)
            by_species.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "organism": organism,
                    "n": len(group),
                    "coverage": value,
                    "interval_width": float((group["upper_90"] - group["lower_90"]).mean()),
                }
            )
        q1, q3 = np.quantile(species, [0.25, 0.75])
        by_seed.append(
            {
                "model": model,
                "seed": int(seed),
                "pooled_marginal_coverage": float(covered.mean()),
                "macro_species_coverage": float(np.mean(species)),
                "species_coverage_median": float(np.median(species)),
                "species_coverage_iqr_low": float(q1),
                "species_coverage_iqr_high": float(q3),
                "species_coverage_min": float(np.min(species)),
                "species_coverage_max": float(np.max(species)),
                "species_ge_nominal": int(np.sum(np.asarray(species) >= 0.90)),
                "species_count": len(species),
                "mean_interval_width": float((frame["upper_90"] - frame["lower_90"]).mean()),
            }
        )
    by_seed_frame = pd.DataFrame(by_seed)
    by_seed_frame.to_csv(output / "species_mic_coverage_by_seed.csv", index=False)
    pd.DataFrame(by_species).to_csv(output / "species_mic_coverage_by_species.csv", index=False)
    summary = (
        by_seed_frame.groupby("model", as_index=False)
        .agg(
            pooled_coverage_mean=("pooled_marginal_coverage", "mean"),
            pooled_coverage_sd=("pooled_marginal_coverage", "std"),
            macro_coverage_mean=("macro_species_coverage", "mean"),
            macro_coverage_sd=("macro_species_coverage", "std"),
            species_coverage_median_mean=("species_coverage_median", "mean"),
            species_coverage_iqr_low_mean=("species_coverage_iqr_low", "mean"),
            species_coverage_iqr_high_mean=("species_coverage_iqr_high", "mean"),
            species_coverage_min_mean=("species_coverage_min", "mean"),
            species_coverage_max_mean=("species_coverage_max", "mean"),
            species_ge_nominal_mean=("species_ge_nominal", "mean"),
            species_count=("species_count", "mean"),
            interval_width_mean=("mean_interval_width", "mean"),
            seeds=("seed", "count"),
        )
    )
    summary.to_csv(output / "species_mic_coverage_summary.csv", index=False)
    weighting = pd.DataFrame(weighting_rows)
    weighting.to_csv(output / "coverage_weighting_by_seed.csv", index=False)
    weighting.groupby("model", as_index=False).agg(
        pair_weighted_mean=("pair_weighted_coverage", "mean"),
        pair_weighted_sd=("pair_weighted_coverage", "std"),
        equal_compound_mean=("equal_compound_coverage", "mean"),
        equal_compound_sd=("equal_compound_coverage", "std"),
        seeds=("seed", "count"),
    ).to_csv(output / "coverage_weighting_summary.csv", index=False)


def full_vs_maccs_audit(species_ablation_dir: Path, output: Path) -> None:
    metrics = pd.read_csv(species_ablation_dir / "metrics.csv")
    macro = metrics.loc[metrics["scope"].eq("macro")]
    rows, seed_rows = [], []
    for metric in METRICS[:5]:
        wide = macro.pivot(index="seed", columns="model", values=metric)
        delta = wide["multiview"] - wide["morgan_maccs"]
        low, high = confidence_interval(delta.to_numpy())
        favorable = delta < 0 if metric in {"mae", "rmse"} else delta > 0
        rows.append({"metric": metric, "mean_difference": delta.mean(), "ci_low": low,
                     "ci_high": high, "favorable_splits": int(favorable.sum()), "seeds": len(delta)})
        seed_rows.extend({"metric": metric, "seed": int(seed), "difference": value} for seed, value in delta.items())
    pd.DataFrame(rows).to_csv(output / "full_vs_maccs_paired.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "full_vs_maccs_by_seed.csv", index=False)


def conditioning_conformal_audit(conditioning_dir: Path, output: Path) -> None:
    metrics = pd.read_csv(conditioning_dir / "metrics.csv")
    macro = metrics.loc[metrics["scope"].eq("macro")]
    model_order = [
        "species_median",
        "morgan_compound_only",
        "morgan",
        "multiview_compound_only",
        "multiview",
    ]
    summary = (
        macro.groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            spearman=("spearman", "mean"),
            within_two_dilutions=("within_two_dilutions", "mean"),
            seeds=("seed", "count"),
        )
        .set_index("model")
        .loc[model_order]
        .reset_index()
    )
    summary.insert(1, "species_information", ~summary["model"].str.endswith("compound_only"))
    summary.to_csv(output / "species_conditioning_ablation.csv", index=False)

    predictions = pd.read_csv(conditioning_dir / "predictions.csv.gz")
    rows = []
    methods = [
        ("pooled", "lower_90", "upper_90"),
        ("species_wise", "lower_90_species", "upper_90_species"),
    ]
    for (model, seed), frame in predictions.groupby(["model", "seed"], sort=True):
        for method, lower, upper in methods:
            covered = frame["y_true"].between(frame[lower], frame[upper])
            working = frame.assign(_covered=covered, _width=frame[upper] - frame[lower])
            species = working.groupby("organism").agg(coverage=("_covered", "mean"), width=("_width", "mean"))
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "method": method,
                    "pooled_coverage": float(covered.mean()),
                    "macro_species_coverage": float(species["coverage"].mean()),
                    "minimum_species_coverage": float(species["coverage"].min()),
                    "median_species_coverage": float(species["coverage"].median()),
                    "species_at_or_above_0_90": int(species["coverage"].ge(0.90).sum()),
                    "pooled_mean_width": float(working["_width"].mean()),
                    "macro_species_width": float(species["width"].mean()),
                }
            )
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "species_conformal_comparison_by_seed.csv", index=False)
    conformal_summary = (
        by_seed.groupby(["model", "method"], as_index=False)
        .agg(
            pooled_coverage=("pooled_coverage", "mean"),
            macro_species_coverage=("macro_species_coverage", "mean"),
            minimum_species_coverage=("minimum_species_coverage", "mean"),
            median_species_coverage=("median_species_coverage", "mean"),
            species_at_or_above_0_90=("species_at_or_above_0_90", "mean"),
            pooled_mean_width=("pooled_mean_width", "mean"),
            macro_species_width=("macro_species_width", "mean"),
            seeds=("seed", "count"),
        )
    )
    conformal_summary.to_csv(output / "species_conformal_comparison.csv", index=False)


def supplementary_species_table(species_dir: Path, output: Path) -> None:
    manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    data_dir = Path(manifest["data"]).parent
    composition = pd.read_csv(data_dir / "species_taxonomy_detail.csv").rename(
        columns={"tax_id": "ncbi_species_tax_id"}
    )
    performance = pd.read_csv(output / "species_mic_per_species.csv")
    morgan = performance.loc[performance["model"].eq("morgan"), ["organism", "mae"]].rename(
        columns={"mae": "morgan_mae"}
    )
    multiview = performance.loc[
        performance["model"].eq("multiview"), ["organism", "mae", "spearman"]
    ].rename(columns={"mae": "multiview_mae", "spearman": "multiview_spearman"})
    coverage = (
        pd.read_csv(output / "species_mic_coverage_by_species.csv")
        .loc[lambda frame: frame["model"].eq("multiview")]
        .groupby("organism", as_index=False)
        .agg(
            mean_test_pairs_per_seed=("n", "mean"),
            multiview_coverage_90=("coverage", "mean"),
            multiview_interval_width=("interval_width", "mean"),
        )
    )
    table = composition.merge(morgan, on="organism", validate="one_to_one")
    table = table.merge(multiview, on="organism", validate="one_to_one")
    table = table.merge(coverage, on="organism", validate="one_to_one")
    table["multiview_minus_morgan_mae"] = table["multiview_mae"] - table["morgan_mae"]
    table = table.sort_values(["pathogen_class", "organism"]).reset_index(drop=True)
    table.to_csv(output / "supplementary_table_s1_species.csv", index=False)

    comparison = performance.loc[performance["model"].isin(LEARNED_MODELS)].pivot(
        index="organism", columns="model", values="mae"
    )[LEARNED_MODELS].add_suffix("_mae")
    comparison = table[["organism", "compound_species_pairs"]].merge(
        comparison, on="organism", validate="one_to_one"
    ).rename(columns={"organism": "species"}).sort_values("species")
    comparison.to_csv(output / "pretrained_species_comparison.csv", index=False)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_species_crosswalk(species_dir: Path, maier_dir: Path, mapping_path: Path, output: Path) -> None:
    manifest = json.loads((maier_dir / "manifest.json").read_text(encoding="utf-8"))
    columns = pd.read_csv(manifest["maier_screen"], sep="\t", nrows=0).columns[1:]
    mapping = pd.read_csv(mapping_path)
    table = pd.DataFrame({"source_strain": columns}).merge(mapping, on="source_strain", how="left", validate="one_to_one")
    if table["accepted_tax_id"].isna().any():
        raise ValueError("Every external strain requires a verified species mapping")
    internal_manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(internal_manifest["data"], usecols=["tax_id", "compound_inchikey"])
    counts = pairs.groupby("tax_id")["compound_inchikey"].nunique()
    table["in_internal_benchmark"] = table["accepted_tax_id"].isin(counts.index)
    table["internal_compounds"] = table["accepted_tax_id"].map(counts).fillna(0).astype(int)
    table["single_species_refit_analyzed"] = table["accepted_tax_id"].eq(562)
    table["analysis_reason"] = np.select(
        [table["single_species_refit_analyzed"], table["in_internal_benchmark"]],
        ["Original selected E. coli case study", "Additional match identified during revision; not analyzed"],
        default="Not in the 48-species benchmark",
    )
    table.to_csv(output / "maier_species_crosswalk.csv", index=False)


def reproducibility_tables(species_dir: Path, maier_dir: Path, output: Path, source_revision: str | None) -> None:
    species_manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    data_dir = Path(species_manifest["data"]).parent
    pd.read_csv(data_dir / "dataset_flow.csv").to_csv(output / "dataset_flow.csv", index=False)
    pd.read_csv(data_dir / "species_taxonomy_detail.csv").to_csv(output / "species_taxonomy_detail.csv", index=False)

    feature_rows = [
        ("Physicochemical", "MolWt", "RDKit descriptor registry: MolWt (average molecular weight)"),
        ("Physicochemical", "MolLogP", "RDKit descriptor registry: MolLogP (Wildman-Crippen logP)"),
        ("Physicochemical", "TPSA", "RDKit descriptor registry: TPSA"),
        ("Physicochemical", "NumHAcceptors", "RDKit descriptor registry: NumHAcceptors"),
        ("Physicochemical", "NumHDonors", "RDKit descriptor registry: NumHDonors"),
        ("Physicochemical", "NumRotatableBonds", "RDKit descriptor registry: NumRotatableBonds"),
        ("Physicochemical", "RingCount", "RDKit descriptor registry: RingCount"),
        ("Physicochemical", "HeavyAtomCount", "RDKit descriptor registry: HeavyAtomCount"),
        ("Physicochemical", "FractionCSP3", "RDKit descriptor registry: FractionCSP3"),
        ("Physicochemical", "NumAromaticRings", "RDKit descriptor registry: NumAromaticRings"),
        ("Physicochemical", "NumAliphaticRings", "RDKit descriptor registry: NumAliphaticRings"),
        ("Physicochemical", "LabuteASA", "RDKit descriptor registry: LabuteASA"),
        ("Graph summary", "n_atoms", "mol.GetNumAtoms()"),
        ("Graph summary", "n_bonds", "mol.GetNumBonds()"),
        ("Graph summary", "n_single_bonds", "Count of non-aromatic bonds with GetBondTypeAsDouble() == 1.0"),
        ("Graph summary", "n_double_bonds", "Count of bonds with GetBondTypeAsDouble() == 2.0"),
        ("Graph summary", "n_triple_bonds", "Count of bonds with GetBondTypeAsDouble() == 3.0"),
        ("Graph summary", "n_aromatic_bonds", "Count of bonds with GetIsAromatic() == True"),
        ("Graph summary", "n_c", "Count of atoms with atomic number 6"),
        ("Graph summary", "n_n", "Count of atoms with atomic number 7"),
        ("Graph summary", "n_o", "Count of atoms with atomic number 8"),
        ("Graph summary", "n_s", "Count of atoms with atomic number 16"),
        ("Graph summary", "n_halogen", "Count of atoms with atomic number in {9, 17, 35, 53}"),
        ("Graph summary", "formal_charge", "Sum of atom.GetFormalCharge() over all atoms"),
        ("Graph summary", "n_rings", "mol.GetRingInfo().NumRings()"),
        ("Graph summary", "n_aromatic_rings", "rdMolDescriptors.CalcNumAromaticRings(mol)"),
        ("Graph summary", "largest_ring_size", "Maximum atom-ring length from mol.GetRingInfo().AtomRings(); 0 if no ring"),
        ("External concentration conversion", "ExactMolWt", "RDKit Descriptors.ExactMolWt(Chem.MolFromSmiles(pchem_canonical_smile)); used as g/mol in the 20 micromolar concentration conversion before salt removal"),
    ]
    pd.DataFrame(feature_rows, columns=["block", "feature", "exact_function_or_definition"]).to_csv(
        output / "handcrafted_feature_definitions.csv", index=False
    )
    settings = [
        ("SMILES parsing", "Chem.MolFromSmiles"),
        ("Morgan", "GetMorganFingerprintAsBitVect(radius=2, nBits=1024, useChirality=False, useBondTypes=True, useFeatures=False, includeRedundantEnvironments=False)"),
        ("MACCS", "MACCSkeys.GenMACCSKeys; 167 bits"),
        ("Parent step 1", "Chem.MolFromSmiles"),
        ("Parent step 2", "rdMolStandardize.FragmentParent"),
        ("Parent step 3", "rdMolStandardize.Uncharger().uncharge"),
        ("Parent step 4", "Chem.MolToSmiles(canonical=True, isomericSmiles=True/False)"),
        ("Parent tautomer handling", "No tautomer canonicalization"),
        ("LightGBM regression", f"objective=L1; n_estimators={species_manifest['n_estimators']}; learning_rate=0.05; num_leaves=63; subsample=0.8; subsample_freq=1; colsample_bytree=0.8; reg_lambda=1.0"),
        ("Species-balanced weights", "Inverse training-species frequency normalized to mean 1"),
        ("MolE embedding", "Public GIN-concat checkpoint; frozen inference; 1000 dimensions"),
        ("MoLFormer embedding", "IBM MoLFormer-XL; BF16; batch size 64; maximum token length 202; frozen inference; 768 dimensions"),
    ]
    pd.DataFrame(settings, columns=["component", "exact_setting"]).to_csv(
        output / "representation_and_parent_settings.csv", index=False
    )
    pd.DataFrame(
        [
            ("Python", platform.python_version()),
            ("RDKit", rdkit.__version__),
            ("LightGBM", lightgbm.__version__),
            ("NumPy", np.__version__),
            ("pandas", pd.__version__),
            ("scikit-learn", sklearn.__version__),
            ("platform", sys.platform),
        ],
        columns=["software", "version"],
    ).to_csv(output / "software_environment.csv", index=False)

    maier_manifest = json.loads((maier_dir / "manifest.json").read_text(encoding="utf-8"))
    screen_path = Path(maier_manifest["maier_screen"])
    library_path = Path(maier_manifest["maier_library"])
    screen = pd.read_csv(screen_path, sep="\t")
    library = pd.read_csv(library_path, sep="\t")
    strain_columns = list(screen.columns[1:])
    values = sorted(pd.unique(screen[strain_columns].to_numpy().ravel()).tolist())
    merged = library.merge(screen, on="prestwick_ID", how="inner", validate="one_to_one")
    ecoli_columns = [column for column in strain_columns if column.startswith("Escherichia coli ")]
    contract = {
        "source_revision": source_revision,
        "screen_file": screen_path.name,
        "screen_sha256": sha256_file(screen_path),
        "library_file": library_path.name,
        "library_sha256": sha256_file(library_path),
        "source_compounds": int(len(screen)),
        "strain_columns": int(len(strain_columns)),
        "source_label_values": values,
        "source_missing_strain_labels": int(screen[strain_columns].isna().sum().sum()),
        "duplicate_screen_ids": int(screen["prestwick_ID"].duplicated().sum()),
        "duplicate_library_ids": int(library["prestwick_ID"].duplicated().sum()),
        "merged_compounds": int(len(merged)),
        "positive_compound_strain_labels": int(screen[strain_columns].to_numpy().sum()),
        "source_broad_positive_compounds": int(screen[strain_columns].max(axis=1).sum()),
        "broad_rule": "positive if max of 40 source-supplied binary strain labels equals 1",
        "ecoli_columns": ecoli_columns,
        "ecoli_rule": "positive if max of the two source-supplied E. coli binary labels equals 1",
        "threshold_fitted": False,
        "upstream_label_definition": "Adjusted P <= 0.05 in the Maier source screening table; binary conversion inherited from the MolE preparation workflow",
        "upstream_label_source": "workflow/01.prepare_training_data.ipynb; PVAL_CUTOFF=0.05; screen_df <= PVAL_CUTOFF",
        "precision_recall_summary": "sklearn.metrics.average_precision_score; average precision (AP); legacy output key: auprc",
        "missing_value_rule": "not applicable: no missing strain-level labels",
        "duplicate_rule": "not applicable: one-to-one IDs in both source tables",
    }
    (output / "maier_activity_label_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def view_ablation_uncertainty(
    species_ablation_dir: Path,
    maier_ablation_dir: Path,
    output: Path,
    bootstrap: int,
    seed: int,
    paired_external: pd.DataFrame,
) -> None:
    models = ["morgan", "morgan_physchem", "morgan_maccs", "morgan_graph", "multiview"]
    full_vs_maccs_audit(species_ablation_dir, output)
    macro = pd.read_csv(species_ablation_dir / "metrics.csv").loc[lambda frame: frame["scope"].eq("macro")]
    reference = macro.loc[macro["model"].eq("morgan")].set_index("seed")["mae"]
    rows = []
    for model in models:
        values = macro.loc[macro["model"].eq(model)].set_index("seed").loc[reference.index, "mae"]
        delta = values.to_numpy() - reference.to_numpy()
        low, high = confidence_interval(delta)
        rows.append(
            {
                "scope": "ChEMBL scaffold holdout",
                "metric": "mae",
                "representation": model,
                "estimate": float(values.mean()),
                "estimate_sd": float(values.std(ddof=1)),
                "difference_vs_morgan": float(delta.mean()),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "interval_type": "paired five-split t stability interval",
                "resamples_or_seeds": len(delta),
            }
        )

    compounds = pd.read_csv(maier_ablation_dir / "compound_predictions.csv.gz")
    primary = compounds.loc[~compounds["exact_training_overlap"]].copy()
    ecoli_columns = [column for column in compounds if column.startswith("Escherichia coli ")]
    primary["ecoli_any_activity"] = primary[ecoli_columns].max(axis=1)
    rng = np.random.default_rng(seed)
    for scope, endpoint, prefix in [
        ("Maier broad primary", "any_activity", "broad_score"),
        ("Maier E. coli primary", "ecoli_any_activity", "mic_margin"),
    ]:
        y = primary[endpoint].to_numpy(dtype=int)
        ref_score = primary[f"{prefix}_morgan"].to_numpy(dtype=float)
        for model in models:
            score = primary[f"{prefix}_{model}"].to_numpy(dtype=float)
            metric_values = {"auroc": roc_auc_score(y, score), "auprc": average_precision_score(y, score)}
            paired = {row["metric"]: row for row in paired_bootstrap(y, score, ref_score, bootstrap, rng)} if model != "morgan" else {}
            for metric, estimate in metric_values.items():
                item = paired.get(metric)
                rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "representation": model,
                        "estimate": float(estimate),
                        "estimate_sd": np.nan,
                        "difference_vs_morgan": 0.0 if item is None else item["difference"],
                        "difference_ci_low": 0.0 if item is None else item["ci_low"],
                        "difference_ci_high": 0.0 if item is None else item["ci_high"],
                        "interval_type": "paired bootstrap percentile interval",
                        "resamples_or_seeds": bootstrap,
                    }
                )
    canonical = paired_external.loc[
        paired_external["cohort"].eq("primary_exact_key_nonoverlap")
        & paired_external["comparison"].eq("multiview_minus_morgan")
    ].set_index(["endpoint", "metric"])
    for row in rows:
        if row["representation"] == "multiview" and row["scope"].startswith("Maier"):
            endpoint = "any_activity" if row["scope"] == "Maier broad primary" else "ecoli_any_activity"
            match = canonical.loc[(endpoint, row["metric"])]
            np.testing.assert_allclose(row["difference_vs_morgan"], match["difference"], rtol=0, atol=1e-12)
            row["difference_ci_low"] = match["ci_low"]
            row["difference_ci_high"] = match["ci_high"]
    pd.DataFrame(rows).to_csv(output / "view_ablation_uncertainty.csv", index=False)


def ecoli_cohort_summary(ecoli: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    labels = {
        "primary_exact_key_nonoverlap": "Primary exact-key non-overlap",
        "posthoc_ecoli_strict_fingerprint_nonidentity": "Fingerprint nonidentity",
        "posthoc_ecoli_strict_parent_nonidentity": "Standardized-parent nonidentity",
    }
    rows = []
    for cohort, label in labels.items():
        subset = ecoli.loc[ecoli["cohort"].eq(cohort)].set_index("model")
        ci = paired.loc[
            paired["cohort"].eq(cohort)
            & paired["endpoint"].eq("ecoli_any_activity")
            & paired["comparison"].eq("multiview_minus_morgan")
        ].set_index("metric")
        rows.append(
            {
                "cohort": label,
                "n": int(subset.loc["morgan", "n"]),
                "positives": int(subset.loc["morgan", "positives"]),
                "morgan_auroc": subset.loc["morgan", "auroc"],
                "multiview_auroc": subset.loc["multiview", "auroc"],
                "delta_auroc": ci.loc["auroc", "difference"],
                "delta_auroc_ci_low": ci.loc["auroc", "ci_low"],
                "delta_auroc_ci_high": ci.loc["auroc", "ci_high"],
                "morgan_auprc": subset.loc["morgan", "auprc"],
                "multiview_auprc": subset.loc["multiview", "auprc"],
                "delta_auprc": ci.loc["auprc", "difference"],
                "delta_auprc_ci_low": ci.loc["auprc", "ci_low"],
                "delta_auprc_ci_high": ci.loc["auprc", "ci_high"],
            }
        )
    pd.DataFrame(rows).to_csv(output / "ecoli_three_cohort_summary.csv", index=False)


def replicate_audit(species_dir: Path, output: Path) -> None:
    manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(
        manifest["data"],
        usecols=["compound_inchikey", "organism", "tax_id", "n_measurements", "log2_mic_iqr"],
    )
    repeated = pairs.loc[pairs["n_measurements"] >= 2, "log2_mic_iqr"].dropna()
    audit = pd.DataFrame(
        [
            {
                "eligible_pairs": len(pairs),
                "repeated_pairs": int((pairs["n_measurements"] >= 2).sum()),
                "repeated_pair_fraction": float((pairs["n_measurements"] >= 2).mean()),
                "replicate_count_median": float(pairs["n_measurements"].median()),
                "replicate_count_p90": float(pairs["n_measurements"].quantile(0.90)),
                "repeated_iqr_median": float(repeated.median()),
                "repeated_iqr_q75": float(repeated.quantile(0.75)),
                "repeated_iqr_max": float(repeated.max()),
            }
        ]
    )
    audit.to_csv(output / "mic_replicate_audit.csv", index=False)

    predictions = pd.read_csv(species_dir / "predictions.csv.gz")
    merged = predictions.merge(
        pairs,
        on=["compound_inchikey", "organism", "tax_id"],
        how="left",
        validate="many_to_one",
    )
    merged["replicate_group"] = np.select(
        [merged["n_measurements"].eq(1), merged["log2_mic_iqr"].eq(0)],
        ["single_measurement", "repeated_zero_iqr"],
        default="repeated_nonzero_iqr",
    )
    rows = []
    correlation_rows = []
    for (model, seed, group_name), frame in merged.groupby(["model", "seed", "replicate_group"], sort=True):
        error = frame["y_pred"] - frame["y_true"]
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "replicate_group": group_name,
                "n": len(frame),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "within_two_dilutions": float((np.abs(error) <= 2.0).mean()),
                "median_measurements": float(frame["n_measurements"].median()),
                "median_iqr": float(frame["log2_mic_iqr"].median()),
            }
        )
    repeated_predictions = merged.loc[merged["n_measurements"] >= 2].copy()
    repeated_predictions["absolute_error"] = (
        repeated_predictions["y_pred"] - repeated_predictions["y_true"]
    ).abs()
    for (model, seed), frame in repeated_predictions.groupby(["model", "seed"], sort=True):
        rho = spearmanr(frame["log2_mic_iqr"], frame["absolute_error"]).statistic
        correlation_rows.append(
            {
                "model": model,
                "seed": int(seed),
                "n_repeated_test_pairs": len(frame),
                "spearman_iqr_absolute_error": float(rho),
            }
        )
    pd.DataFrame(rows).to_csv(output / "mic_error_by_replicate_group.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(output / "mic_iqr_error_correlation.csv", index=False)


def scaffold_audit(species_dir: Path, output: Path) -> None:
    manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(manifest["data"], usecols=["compound_inchikey", "canonical_smiles"])
    compounds = pairs.drop_duplicates("compound_inchikey")
    acyclic = []
    for row in compounds.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        acyclic.append(not scaffold or scaffold.GetNumAtoms() == 0)
    n_acyclic = int(np.sum(acyclic))
    pd.DataFrame(
        [
            {
                "unique_compounds": len(compounds),
                "acyclic_fallback_compounds": n_acyclic,
                "acyclic_fallback_fraction": n_acyclic / len(compounds),
                "fallback_rule": "compound-specific key for empty Bemis-Murcko scaffold",
            }
        ]
    ).to_csv(output / "scaffold_audit.csv", index=False)


def ecoli_overlap_audit(species_dir: Path, maier_dir: Path, output: Path) -> int:
    manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(manifest["data"], usecols=["compound_inchikey", "organism"])
    ecoli_keys = set(pairs.loc[pairs["organism"].eq("Escherichia coli"), "compound_inchikey"])
    compounds = pd.read_csv(maier_dir / "compound_predictions.csv.gz", usecols=["pchem_inchikey", "exact_training_overlap"])
    primary = compounds.loc[~compounds["exact_training_overlap"]]
    overlap = primary.loc[primary["pchem_inchikey"].isin(ecoli_keys)]
    pd.DataFrame(
        [
            {
                "maier_primary_compounds": len(primary),
                "ecoli_mic_training_compounds": len(ecoli_keys),
                "primary_ecoli_exact_inchikey_overlap": len(overlap),
                "cohort_status": "clean" if len(overlap) == 0 else "requires_ecoli_specific_exclusion",
            }
        ]
    ).to_csv(output / "maier_ecoli_overlap_audit.csv", index=False)
    return int(len(overlap))


def ecoli_fingerprint_audit(species_dir: Path, maier_dir: Path, output: Path) -> pd.DataFrame:
    manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(
        manifest["data"],
        usecols=["compound_inchikey", "canonical_smiles", "organism"],
    )
    ecoli = pairs.loc[pairs["organism"].eq("Escherichia coli"), ["compound_inchikey", "canonical_smiles"]].drop_duplicates(
        "compound_inchikey"
    )
    training_keys = []
    training_fingerprints = []
    for row in ecoli.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        if mol is not None:
            training_keys.append(row.compound_inchikey)
            training_fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    compounds = pd.read_csv(
        maier_dir / "compound_predictions.csv.gz",
        usecols=["pchem_inchikey", "rdkit_no_salt", "exact_training_overlap"],
    )
    primary = compounds.loc[~compounds["exact_training_overlap"]].copy()
    rows = []
    for row in primary.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.rdkit_no_salt))
        query = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        similarities = DataStructs.BulkTanimotoSimilarity(query, training_fingerprints)
        index = int(np.argmax(similarities))
        rows.append(
            {
                "pchem_inchikey": row.pchem_inchikey,
                "ecoli_nearest_training_tanimoto": float(similarities[index]),
                "ecoli_nearest_training_inchikey": training_keys[index],
            }
        )
    audit = pd.DataFrame(rows)
    audit["ecoli_fingerprint_identity"] = audit["ecoli_nearest_training_tanimoto"].ge(1.0 - 1e-12)
    audit.to_csv(output / "maier_ecoli_fingerprint_audit.csv", index=False)
    return audit


def _standardized_parent_smiles(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "", ""
    parent = rdMolStandardize.FragmentParent(mol)
    parent = rdMolStandardize.Uncharger().uncharge(parent)
    return (
        Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True),
        Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False),
    )


def standardized_parent_audit(species_dir: Path, maier_dir: Path, output: Path) -> pd.DataFrame:
    maier_manifest = json.loads((maier_dir / "manifest.json").read_text(encoding="utf-8"))
    broad = pd.read_csv(
        maier_manifest["training_data"],
        sep="\t",
        usecols=["compound_inchikey", "compound_smiles"],
    ).drop_duplicates("compound_inchikey")
    species_manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    ecoli = pd.read_csv(
        species_manifest["data"],
        usecols=["compound_inchikey", "canonical_smiles", "organism"],
    )
    ecoli = ecoli.loc[ecoli["organism"].eq("Escherichia coli")].drop_duplicates("compound_inchikey")

    broad_parents = [_standardized_parent_smiles(value) for value in broad["compound_smiles"]]
    ecoli_parents = [_standardized_parent_smiles(value) for value in ecoli["canonical_smiles"]]
    broad_isomeric = {value[0] for value in broad_parents if value[0]}
    broad_nonisomeric = {value[1] for value in broad_parents if value[1]}
    ecoli_isomeric = {value[0] for value in ecoli_parents if value[0]}
    ecoli_nonisomeric = {value[1] for value in ecoli_parents if value[1]}

    compounds = pd.read_csv(
        maier_dir / "compound_predictions.csv.gz",
        usecols=["pchem_inchikey", "rdkit_no_salt", "exact_training_overlap"],
    )
    primary = compounds.loc[~compounds["exact_training_overlap"]].copy()
    query_parents = [_standardized_parent_smiles(value) for value in primary["rdkit_no_salt"]]
    primary["parent_canonical_smiles"] = [value[0] for value in query_parents]
    primary["parent_canonical_nonisomeric_smiles"] = [value[1] for value in query_parents]
    primary["broad_parent_isomeric_identity"] = primary["parent_canonical_smiles"].isin(broad_isomeric)
    primary["broad_parent_nonisomeric_identity"] = primary["parent_canonical_nonisomeric_smiles"].isin(
        broad_nonisomeric
    )
    primary["ecoli_parent_isomeric_identity"] = primary["parent_canonical_smiles"].isin(ecoli_isomeric)
    primary["ecoli_parent_nonisomeric_identity"] = primary["parent_canonical_nonisomeric_smiles"].isin(
        ecoli_nonisomeric
    )
    audit = primary.drop(columns=["rdkit_no_salt", "exact_training_overlap"])
    audit.to_csv(output / "maier_standardized_parent_audit.csv", index=False)
    return audit


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    species_dir = Path(args.species_dir)
    maier_dir = Path(args.maier_dir)
    reproducibility_tables(species_dir, maier_dir, output, args.maier_source_revision)
    external_species_crosswalk(species_dir, maier_dir, Path(args.maier_taxonomy_mapping), output)
    metrics = pd.read_csv(species_dir / "metrics.csv")
    macro = metrics.loc[metrics["scope"].eq("macro")].copy()
    summary_rows = []
    for model in MODELS:
        subset = macro.loc[macro["model"].eq(model)]
        for metric in METRICS:
            values = subset[metric].dropna().to_numpy(dtype=float)
            low, high = confidence_interval(values)
            summary_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "interval_type": "five_split_t_interval",
                    "seeds": len(values),
                }
            )
    species_summary = pd.DataFrame(summary_rows)
    species_summary.to_csv(output / "species_mic_summary.csv", index=False)

    paired_rows = []
    reference = macro.loc[macro["model"].eq("morgan")].set_index("seed")
    for model in ["multiview", "mole", "molformer"]:
        comparison = macro.loc[macro["model"].eq(model)].set_index("seed")
        for metric in METRICS:
            delta = comparison.loc[reference.index, metric].to_numpy() - reference[metric].to_numpy()
            low, high = confidence_interval(delta)
            paired_rows.append(
                {
                    "comparison": f"{model}_minus_morgan",
                    "metric": metric,
                    "mean_difference": float(delta.mean()),
                    "sd_difference": float(delta.std(ddof=1)),
                    "ci_low": low,
                    "ci_high": high,
                    "interval_type": "five_split_t_interval",
                    "wins": int((delta < 0).sum()) if metric in {"mae", "rmse", "conformal_width"} else int((delta > 0).sum()),
                    "seeds": len(delta),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "species_mic_paired_differences.csv", index=False)

    per_species = (
        metrics.loc[metrics["scope"].eq("species")]
        .groupby(["model", "organism"], as_index=False)[METRICS[:5]]
        .mean()
    )
    per_species.to_csv(output / "species_mic_per_species.csv", index=False)
    coverage_audit(species_dir, output)
    supplementary_species_table(species_dir, output)
    replicate_audit(species_dir, output)
    scaffold_audit(species_dir, output)
    if args.conditioning_dir:
        conditioning_conformal_audit(Path(args.conditioning_dir), output)

    pd.read_csv(maier_dir / "bootstrap_ci.csv").to_csv(output / "maier_individual_bootstrap_ci.csv", index=False)
    pd.read_csv(maier_dir / "similarity_metrics.csv").to_csv(output / "maier_similarity_metrics.csv", index=False)
    pd.read_csv(maier_dir / "excluded_missing_structures.csv").to_csv(
        output / "maier_excluded_missing_structures.csv", index=False
    )

    compounds = pd.read_csv(maier_dir / "compound_predictions.csv.gz")
    ecoli_exact_overlap = ecoli_overlap_audit(species_dir, maier_dir, output)
    ecoli_fingerprint = ecoli_fingerprint_audit(species_dir, maier_dir, output)
    parent_audit = standardized_parent_audit(species_dir, maier_dir, output)
    compounds = compounds.merge(ecoli_fingerprint, on="pchem_inchikey", how="left", validate="one_to_one")
    compounds = compounds.merge(parent_audit, on="pchem_inchikey", how="left", validate="one_to_one")
    primary = compounds.loc[~compounds["exact_training_overlap"]].copy()
    if primary["ecoli_fingerprint_identity"].isna().any():
        raise ValueError("E. coli fingerprint audit is incomplete for the primary Maier cohort.")
    primary["ecoli_fingerprint_identity"] = primary["ecoli_fingerprint_identity"].astype(bool)
    parent_identity_columns = [
        "broad_parent_isomeric_identity",
        "broad_parent_nonisomeric_identity",
        "ecoli_parent_isomeric_identity",
        "ecoli_parent_nonisomeric_identity",
    ]
    if primary[parent_identity_columns].isna().any().any():
        raise ValueError("Standardized-parent audit is incomplete for the primary Maier cohort.")
    primary[parent_identity_columns] = primary[parent_identity_columns].astype(bool)
    strict_broad = primary.loc[primary["nearest_training_tanimoto"] < 1.0 - 1e-12].copy()
    strict_parent = primary.loc[~primary["broad_parent_nonisomeric_identity"]].copy()
    strict_ecoli = primary.loc[~primary["ecoli_fingerprint_identity"]].copy()
    strict_ecoli_parent = primary.loc[~primary["ecoli_parent_nonisomeric_identity"]].copy()
    broad = pd.concat(
        [
            classification_metrics(primary, "primary_exact_key_nonoverlap", "any_activity", "broad_score"),
            classification_metrics(strict_broad, "posthoc_strict_fingerprint_nonidentity", "any_activity", "broad_score"),
            classification_metrics(strict_parent, "posthoc_strict_parent_nonidentity", "any_activity", "broad_score"),
        ],
        ignore_index=True,
    )
    broad.to_csv(output / "maier_external_metrics.csv", index=False)

    topk = pd.concat(
        [
            topk_metrics(primary, "primary_exact_key_nonoverlap", "any_activity", "broad_score"),
            topk_metrics(strict_broad, "posthoc_strict_fingerprint_nonidentity", "any_activity", "broad_score"),
            topk_metrics(strict_parent, "posthoc_strict_parent_nonidentity", "any_activity", "broad_score"),
        ],
        ignore_index=True,
    )
    topk.to_csv(output / "maier_topk_enrichment.csv", index=False)

    ecoli_columns = ["Escherichia coli ED1a (NT5078)", "Escherichia coli IAI1 (NT5077)"]
    for frame in (primary, strict_ecoli, strict_ecoli_parent):
        frame["ecoli_any_activity"] = frame[ecoli_columns].max(axis=1)
    ecoli = pd.concat(
        [
            classification_metrics(primary, "primary_exact_key_nonoverlap", "ecoli_any_activity", "mic_margin"),
            classification_metrics(strict_ecoli, "posthoc_ecoli_strict_fingerprint_nonidentity", "ecoli_any_activity", "mic_margin"),
            classification_metrics(strict_ecoli_parent, "posthoc_ecoli_strict_parent_nonidentity", "ecoli_any_activity", "mic_margin"),
        ],
        ignore_index=True,
    )
    ecoli.to_csv(output / "maier_ecoli_mic_transfer.csv", index=False)

    bootstrap_rows = []
    rng = np.random.default_rng(args.seed)
    for cohort, frame in [
        ("primary_exact_key_nonoverlap", primary),
        ("posthoc_strict_fingerprint_nonidentity", strict_broad),
    ]:
        for endpoint, prefix in [("any_activity", "broad_score")]:
            y = frame[endpoint].to_numpy(dtype=int)
            for comparator in ["morgan", "mole", "molformer"]:
                rows = paired_bootstrap(
                    y,
                    frame[f"{prefix}_multiview"].to_numpy(),
                    frame[f"{prefix}_{comparator}"].to_numpy(),
                    args.bootstrap,
                    rng,
                )
                for row in rows:
                    row.update(
                        {
                            "cohort": cohort,
                            "endpoint": endpoint,
                            "comparison": f"multiview_minus_{comparator}",
                        }
                    )
                bootstrap_rows.extend(rows)
    for cohort, frame in [("primary_exact_key_nonoverlap", primary), ("posthoc_ecoli_strict_fingerprint_nonidentity", strict_ecoli)]:
        y = frame["ecoli_any_activity"].to_numpy(dtype=int)
        for comparator in ["morgan", "mole", "molformer"]:
            rows = paired_bootstrap(
                y,
                frame["mic_margin_multiview"].to_numpy(),
                frame[f"mic_margin_{comparator}"].to_numpy(),
                args.bootstrap,
                rng,
            )
            for row in rows:
                row.update(
                    {
                        "cohort": cohort,
                        "endpoint": "ecoli_any_activity",
                        "comparison": f"multiview_minus_{comparator}",
                    }
                )
            bootstrap_rows.extend(rows)
    y = strict_parent["any_activity"].to_numpy(dtype=int)
    for comparator in ["morgan", "mole", "molformer"]:
        rows = paired_bootstrap(
            y,
            strict_parent["broad_score_multiview"].to_numpy(),
            strict_parent[f"broad_score_{comparator}"].to_numpy(),
            args.bootstrap,
            rng,
        )
        for row in rows:
            row.update(
                {
                    "cohort": "posthoc_strict_parent_nonidentity",
                    "endpoint": "any_activity",
                    "comparison": f"multiview_minus_{comparator}",
                }
            )
        bootstrap_rows.extend(rows)
    y = strict_ecoli_parent["ecoli_any_activity"].to_numpy(dtype=int)
    for comparator in ["morgan", "mole", "molformer"]:
        rows = paired_bootstrap(
            y,
            strict_ecoli_parent["mic_margin_multiview"].to_numpy(),
            strict_ecoli_parent[f"mic_margin_{comparator}"].to_numpy(),
            args.bootstrap,
            rng,
        )
        for row in rows:
            row.update(
                {
                    "cohort": "posthoc_ecoli_strict_parent_nonidentity",
                    "endpoint": "ecoli_any_activity",
                    "comparison": f"multiview_minus_{comparator}",
                }
            )
        bootstrap_rows.extend(rows)
    paired_bootstrap_frame = pd.DataFrame(bootstrap_rows)
    paired_bootstrap_frame.to_csv(output / "maier_paired_bootstrap_differences.csv", index=False)
    if args.species_ablation_dir and args.maier_ablation_dir:
        view_ablation_uncertainty(
            Path(args.species_ablation_dir),
            Path(args.maier_ablation_dir),
            output,
            args.bootstrap,
            args.seed,
            paired_bootstrap_frame,
        )
    ecoli_cohort_summary(ecoli, paired_bootstrap_frame, output)

    audit = {
        "species_mic": {
            "rows": int(json.loads((species_dir / "manifest.json").read_text())["rows"]),
            "compounds": int(json.loads((species_dir / "manifest.json").read_text())["compounds"]),
            "species": int(json.loads((species_dir / "manifest.json").read_text())["species"]),
        },
        "maier": {
            "structurally_evaluable": int(len(compounds)),
            "exact_key_overlap": int(compounds["exact_training_overlap"].sum()),
            "primary_exact_key_nonoverlap": int(len(primary)),
            "primary_tanimoto_one": int((primary["nearest_training_tanimoto"] >= 1.0 - 1e-12).sum()),
            "posthoc_strict_broad_fingerprint_nonidentity": int(len(strict_broad)),
            "primary_broad_parent_isomeric_identity": int(primary["broad_parent_isomeric_identity"].sum()),
            "primary_broad_parent_nonisomeric_identity": int(primary["broad_parent_nonisomeric_identity"].sum()),
            "posthoc_strict_broad_parent_nonidentity": int(len(strict_parent)),
            "posthoc_strict_ecoli_fingerprint_nonidentity": int(len(strict_ecoli)),
            "primary_ecoli_fingerprint_identity": int(ecoli_fingerprint["ecoli_fingerprint_identity"].sum()),
            "primary_ecoli_parent_isomeric_identity": int(primary["ecoli_parent_isomeric_identity"].sum()),
            "primary_ecoli_parent_nonisomeric_identity": int(primary["ecoli_parent_nonisomeric_identity"].sum()),
            "posthoc_strict_ecoli_parent_nonidentity": int(len(strict_ecoli_parent)),
            "primary_ecoli_exact_inchikey_overlap": ecoli_exact_overlap,
        },
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-dir", required=True)
    parser.add_argument("--maier-dir", required=True)
    parser.add_argument("--conditioning-dir")
    parser.add_argument("--species-ablation-dir")
    parser.add_argument("--maier-ablation-dir")
    parser.add_argument("--maier-source-revision")
    parser.add_argument("--maier-taxonomy-mapping", default=str(Path(__file__).resolve().parents[1] / "docs" / "maier_taxonomy_mapping.csv"))
    parser.add_argument("--output", default="paper/tables/species_mic_v2")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
