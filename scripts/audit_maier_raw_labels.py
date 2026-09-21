import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


P = 0.05
ECOLI = ["Escherichia coli ED1a (NT5078)", "Escherichia coli IAI1 (NT5077)"]
CDIFF = "Clostridium difficile (NT5083)"
MODELS = ["morgan", "multiview", "mole", "molformer"]
COHORTS = ["primary", "fingerprint_nonidentity", "parent_nonidentity"]


def metrics(y, score):
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def pooled_label(frame, columns):
    observed = frame[columns].notna()
    active = frame[columns].le(P) & observed
    label = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    label[active.any(axis=1)] = 1
    label[observed.all(axis=1) & ~active.any(axis=1)] = 0
    return label


def classification_table(frame, cohorts, endpoint, prefix):
    rows = []
    for cohort, mask in cohorts:
        sub = frame.loc[mask & frame[endpoint].notna()]
        y = sub[endpoint].to_numpy(dtype=int)
        for model in MODELS:
            auroc, ap = metrics(y, sub[f"{prefix}_{model}"].to_numpy())
            rows.append({"cohort": cohort, "model": model, "endpoint": endpoint, "n": len(y),
                         "positives": int(y.sum()), "prevalence": float(y.mean()), "auroc": auroc, "auprc": ap})
    return pd.DataFrame(rows)


def topk_table(frame, cohorts, endpoint, prefix, models=MODELS):
    rows = []
    for cohort, mask in cohorts:
        sub = frame.loc[mask & frame[endpoint].notna()]
        y = sub[endpoint].to_numpy(dtype=int)
        for model in models:
            score = sub[f"{prefix}_{model}"].to_numpy()
            order = np.argsort(-score, kind="stable")
            for k in [25, 50, 100]:
                take = min(k, len(y))
                precision = float(y[order[:take]].mean())
                rows.append({"cohort": cohort, "endpoint": endpoint, "model": model, "k": take,
                             "hits": int(y[order[:take]].sum()), "precision": precision,
                             "enrichment": precision / y.mean()})
    return pd.DataFrame(rows)


def paired(y, a, b, n, rng):
    point = np.array(metrics(y, a)) - np.array(metrics(y, b))
    draws = []
    while len(draws) < n:
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            draws.append(np.array(metrics(y[idx], a[idx])) - np.array(metrics(y[idx], b[idx])))
    limits = np.quantile(np.asarray(draws), [0.025, 0.975], axis=0)
    return [(name, point[i], limits[0, i], limits[1, i]) for i, name in enumerate(["auroc", "auprc"])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-xlsx", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.raw_xlsx, sheet_name="S3a. Adjusted p-values")
    strain_columns = list(raw.columns[4:])
    raw = raw.set_index("prestwick_ID")
    labels = pd.DataFrame(index=raw.index)
    labels["raw_missing_cells"] = raw[strain_columns].isna().sum(axis=1)
    labels["raw_any_activity"] = pooled_label(raw, strain_columns)
    labels["raw_ecoli_any_activity"] = pooled_label(raw, ECOLI)
    labels["raw_cdiff_activity"] = pooled_label(raw, [CDIFF])
    for column in ECOLI + [CDIFF]:
        labels["raw_" + column] = pooled_label(raw, [column])
    labels.to_csv(args.out / "maier_raw_endpoint_labels.csv.gz")

    pred = pd.read_csv(args.data / "identifiers/separate_external_predictions.csv.gz").set_index("prestwick_ID")
    pred = pred.join(labels, how="left", validate="one_to_one")
    primary = ~pred["exact_training_overlap"]
    fp_broad = primary & pred["nearest_training_tanimoto"].lt(1 - 1e-12)
    parent = pd.read_csv(args.tables / "maier_standardized_parent_audit.csv").set_index("pchem_inchikey")
    pred = pred.join(parent[["broad_parent_nonisomeric_identity", "ecoli_parent_nonisomeric_identity"]], on="pchem_inchikey")
    ecoli_fp = pd.read_csv(args.tables / "maier_ecoli_fingerprint_audit.csv").set_index("pchem_inchikey")
    pred = pred.join(ecoli_fp[["ecoli_fingerprint_identity"]], on="pchem_inchikey")
    for column in ["broad_parent_nonisomeric_identity", "ecoli_parent_nonisomeric_identity", "ecoli_fingerprint_identity"]:
        pred[column] = pred[column].fillna(False).astype(bool)
    broad_cohorts = [("primary_exact_key_nonoverlap", primary),
                     ("posthoc_strict_fingerprint_nonidentity", fp_broad),
                     ("posthoc_strict_parent_nonidentity", primary & ~pred["broad_parent_nonisomeric_identity"])]
    ecoli_cohorts = [("primary_exact_key_nonoverlap", primary),
                     ("posthoc_ecoli_strict_fingerprint_nonidentity", primary & ~pred["ecoli_fingerprint_identity"]),
                     ("posthoc_ecoli_strict_parent_nonidentity", primary & ~pred["ecoli_parent_nonisomeric_identity"])]

    broad_metrics = classification_table(pred, broad_cohorts, "raw_any_activity", "broad_score")
    ecoli_metrics = classification_table(pred, ecoli_cohorts, "raw_ecoli_any_activity", "mic_margin")
    broad_metrics["endpoint"] = "any_activity"
    ecoli_metrics["endpoint"] = "ecoli_any_activity"
    broad_metrics.to_csv(args.out / "maier_external_metrics.csv", index=False)
    ecoli_metrics.to_csv(args.out / "maier_ecoli_mic_transfer.csv", index=False)
    broad_topk = topk_table(pred, broad_cohorts, "raw_any_activity", "broad_score")
    broad_topk["endpoint"] = "any_activity"
    broad_topk.to_csv(args.out / "maier_topk_enrichment.csv", index=False)

    similarity_rows = []
    valid_primary = pred.loc[primary & pred["raw_any_activity"].notna()]
    for similarity_bin, group in valid_primary.groupby("similarity_bin", observed=True):
        y = group["raw_any_activity"].to_numpy(dtype=int)
        for model in MODELS:
            auroc, ap = metrics(y, group[f"broad_score_{model}"].to_numpy())
            similarity_rows.append({"similarity_bin": similarity_bin, "model": model, "n": len(y),
                                    "positives": int(y.sum()), "prevalence": float(y.mean()), "auroc": auroc, "auprc": ap})
    pd.DataFrame(similarity_rows).to_csv(args.out / "maier_similarity_metrics.csv", index=False)

    individual = []
    for endpoint, prefix, mask, name in [("raw_any_activity", "broad_score", primary, "broad_any_activity"),
                                         ("raw_ecoli_any_activity", "mic_margin", primary, "ecoli_mic_margin")]:
        sub = pred.loc[mask & pred[endpoint].notna()]
        y = sub[endpoint].to_numpy(dtype=int)
        for model in MODELS:
            score = sub[f"{prefix}_{model}"].to_numpy()
            point = metrics(y, score)
            rng = np.random.default_rng(20260729)
            draws = []
            while len(draws) < 1000:
                idx = rng.integers(0, len(y), len(y))
                if np.unique(y[idx]).size == 2:
                    draws.append(metrics(y[idx], score[idx]))
            limits = np.quantile(np.asarray(draws), [0.025, 0.975], axis=0)
            for i, metric in enumerate(["auroc", "auprc"]):
                individual.append({"model": model, "endpoint": name, "metric": metric, "estimate": point[i],
                                   "ci_low": limits[0, i], "ci_high": limits[1, i], "bootstrap": 1000})
    pd.DataFrame(individual).to_csv(args.out / "maier_individual_bootstrap_ci.csv", index=False)

    rng = np.random.default_rng(20260809)
    paired_rows = []
    paired_order = [(broad_cohorts[0], "raw_any_activity", "broad_score"),
                    (broad_cohorts[1], "raw_any_activity", "broad_score"),
                    (ecoli_cohorts[0], "raw_ecoli_any_activity", "mic_margin"),
                    (ecoli_cohorts[1], "raw_ecoli_any_activity", "mic_margin"),
                    (broad_cohorts[2], "raw_any_activity", "broad_score"),
                    (ecoli_cohorts[2], "raw_ecoli_any_activity", "mic_margin")]
    for (cohort, mask), endpoint, prefix in paired_order:
        sub = pred.loc[mask & pred[endpoint].notna()]
        y = sub[endpoint].to_numpy(dtype=int)
        for comparator in ["morgan", "mole", "molformer"]:
            for metric, difference, low, high in paired(y, sub[f"{prefix}_multiview"].to_numpy(),
                                                        sub[f"{prefix}_{comparator}"].to_numpy(), 5000, rng):
                paired_rows.append({"metric": metric, "difference": difference, "ci_low": low, "ci_high": high,
                                    "bootstrap": 5000, "cohort": cohort,
                                    "endpoint": "any_activity" if endpoint == "raw_any_activity" else "ecoli_any_activity",
                                    "comparison": f"multiview_minus_{comparator}"})
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(args.out / "maier_paired_bootstrap_differences.csv", index=False)

    summaries = []
    names = ["Primary exact-key non-overlap", "Fingerprint nonidentity", "Standardized-parent nonidentity"]
    for name, (cohort, mask) in zip(names, ecoli_cohorts):
        sub = pred.loc[mask & pred["raw_ecoli_any_activity"].notna()]
        row = {"cohort": name, "n": len(sub), "positives": int(sub["raw_ecoli_any_activity"].sum())}
        for model in ["morgan", "multiview"]:
            row[f"{model}_auroc"], row[f"{model}_auprc"] = metrics(sub["raw_ecoli_any_activity"].astype(int), sub[f"mic_margin_{model}"])
        pair = paired_frame[(paired_frame.cohort == cohort) & (paired_frame.endpoint == "ecoli_any_activity") &
                            (paired_frame.comparison == "multiview_minus_morgan")].set_index("metric")
        for metric in ["auroc", "auprc"]:
            row[f"delta_{metric}"] = pair.loc[metric, "difference"]
            row[f"delta_{metric}_ci_low"] = pair.loc[metric, "ci_low"]
            row[f"delta_{metric}_ci_high"] = pair.loc[metric, "ci_high"]
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(args.out / "ecoli_three_cohort_summary.csv", index=False)

    ablation = pd.read_csv(args.data / "identifiers/separate_external_ablation_predictions.csv.gz").set_index("prestwick_ID").join(labels)
    valid = ~ablation["exact_training_overlap"]
    ablation_rows, uncertainty = [], []
    rng = np.random.default_rng(20260809)
    for scope, endpoint, prefix in [("Maier broad primary", "raw_any_activity", "broad_score"),
                                    ("Maier E. coli primary", "raw_ecoli_any_activity", "mic_margin")]:
        sub = ablation.loc[valid & ablation[endpoint].notna()]
        y = sub[endpoint].to_numpy(dtype=int)
        reference = sub[f"{prefix}_morgan"].to_numpy()
        for model in ["morgan", "morgan_physchem", "morgan_maccs", "morgan_graph", "multiview"]:
            score = sub[f"{prefix}_{model}"].to_numpy()
            point = metrics(y, score)
            contrast = {m: (0.0, 0.0, 0.0) for m in ["auroc", "auprc"]}
            if model != "morgan":
                contrast = {m: (d, lo, hi) for m, d, lo, hi in paired(y, score, reference, 5000, rng)}
            for i, metric in enumerate(["auroc", "auprc"]):
                d, lo, hi = contrast[metric]
                uncertainty.append({"scope": scope, "metric": metric, "representation": model, "estimate": point[i],
                                    "estimate_sd": np.nan, "difference_vs_morgan": d, "difference_ci_low": lo,
                                    "difference_ci_high": hi, "interval_type": "paired bootstrap percentile interval",
                                    "resamples_or_seeds": 5000})
    uncertainty = pd.DataFrame(uncertainty)
    for scope, endpoint in [("Maier broad primary", "any_activity"), ("Maier E. coli primary", "ecoli_any_activity")]:
        canonical = paired_frame[(paired_frame.cohort == "primary_exact_key_nonoverlap") &
                                 (paired_frame.endpoint == endpoint) &
                                 (paired_frame.comparison == "multiview_minus_morgan")].set_index("metric")
        for metric in ["auroc", "auprc"]:
            mask = (uncertainty.scope == scope) & (uncertainty.representation == "multiview") & (uncertainty.metric == metric)
            uncertainty.loc[mask, ["difference_vs_morgan", "difference_ci_low", "difference_ci_high"]] = canonical.loc[metric, ["difference", "ci_low", "ci_high"]].to_numpy()
    uncertainty.to_csv(args.out / "view_ablation_external_uncertainty.csv", index=False)
    ecoli_base = ablation.loc[valid]
    for model in ["morgan", "morgan_physchem", "morgan_maccs", "morgan_graph", "multiview"]:
        for endpoint, name in [("raw_" + ECOLI[0], ECOLI[0]), ("raw_" + ECOLI[1], ECOLI[1]), ("raw_ecoli_any_activity", "E. coli any strain")]:
            sub = ecoli_base.loc[ecoli_base[endpoint].notna()]
            y = sub[endpoint].to_numpy(dtype=int)
            auroc, ap = metrics(y, sub[f"mic_margin_{model}"])
            ablation_rows.append({"model": model, "strain": name, "n": len(y), "positives": int(y.sum()),
                                  "prevalence": float(y.mean()), "auroc": auroc, "auprc": ap})
    pd.DataFrame(ablation_rows).to_csv(args.out / "view_ablation_external_ecoli.csv", index=False)

    flags = pd.read_csv(args.data / "identifiers/joint_overlap_flags.csv.gz").set_index("prestwick_ID")
    scores = pd.read_csv(args.data / "identifiers/joint_mean_scores.csv.gz")
    seed_scores = pd.read_csv(args.data / "identifiers/joint_seed_predictions.csv.gz")
    mw = pred["ExactMolWt"].map(lambda value: np.log2(0.02 * value))
    joint_metrics, joint_seed_rows, joint_topk, joint_flow, joint_pairs, mw_pairs, label_export = [], [], [], [], [], [], []
    rng_joint = np.random.default_rng(20260908)
    rng_mw = np.random.default_rng(20260912)
    endpoint_map = {
        "Clostridioides difficile": [("Clostridioides difficile any strain", "raw_cdiff_activity")],
        "Escherichia coli": [("Escherichia coli any strain", "raw_ecoli_any_activity"), *[(c, "raw_" + c) for c in ECOLI]],
    }
    for organism in sorted(endpoint_map):
        wide = scores[scores.organism.eq(organism)].pivot(index="prestwick_ID", columns=["family", "representation"], values="score")
        wide.columns = [f"{a}:{b}" for a, b in wide.columns]
        wide["concentration_only"] = mw
        label_export.append(pd.DataFrame({"prestwick_ID": pred.index, "organism": organism,
                                          "label": pred[endpoint_map[organism][0][1]].astype("Int64"),
                                          "label_status": np.where(pred[endpoint_map[organism][0][1]].notna(), "valid", "ambiguous_missing"),
                                          "molecular_weight_only": mw}))
        for cohort in COHORTS:
            base_ids = flags.index[flags[cohort]]
            for endpoint_name, label_column in endpoint_map[organism]:
                ids = base_ids.intersection(pred.index[pred[label_column].notna()], sort=False)
                y = pred.loc[ids, label_column].to_numpy(dtype=int)
                joint_flow.append({"organism": organism, "endpoint": endpoint_name, "cohort": cohort, "n": len(y),
                                   "positives": int(y.sum()), "prevalence": float(y.mean())})
                for model in wide.columns:
                    auroc, ap = metrics(y, wide.loc[ids, model].to_numpy())
                    joint_metrics.append({"organism": organism, "endpoint": endpoint_name, "cohort": cohort,
                                          "model": model, "n": len(y), "positives": int(y.sum()), "auroc": auroc,
                                          "ap": ap, "status": "estimable"})
                    if endpoint_name.endswith("any strain"):
                        order = np.argsort(-wide.loc[ids, model].to_numpy(), kind="stable")
                        for k in [25, 50, 100]:
                            precision = float(y[order[:k]].mean())
                            joint_topk.append({"organism": organism, "cohort": cohort, "model": model, "k": k,
                                               "hits": int(y[order[:k]].sum()), "precision": precision,
                                               "enrichment": precision / y.mean()})
                if not endpoint_name.endswith("any strain"):
                    continue
                seed_sub = seed_scores[(seed_scores.organism == organism) & seed_scores.prestwick_ID.isin(ids)]
                lookup = pd.Series(y, index=ids)
                for (family, representation, seed), group in seed_sub.groupby(["family", "representation", "seed"]):
                    auroc, ap = metrics(lookup.loc[group.prestwick_ID].to_numpy(), group.score.to_numpy())
                    joint_seed_rows.append({"organism": organism, "cohort": cohort, "family": family,
                                            "representation": representation, "seed": seed, "auroc": auroc, "ap": ap})
                models = ["joint:multiview", "joint:morgan", "joint:morgan_maccs", "single:multiview"]
                matrix = wide.loc[ids, models].to_numpy()
                draws = []
                while len(draws) < 5000:
                    idx = rng_joint.integers(0, len(y), len(y))
                    if np.unique(y[idx]).size == 2:
                        draws.append([metrics(y[idx], matrix[idx, j]) for j in range(len(models))])
                draws = np.asarray(draws)
                for left, right in [(models[0], models[1]), (models[0], models[2]), (models[0], models[3])]:
                    a, b = models.index(left), models.index(right)
                    point = np.array(metrics(y, matrix[:, a])) - np.array(metrics(y, matrix[:, b]))
                    limits = np.quantile(draws[:, a] - draws[:, b], [0.025, 0.975], axis=0)
                    for i, metric in enumerate(["auroc", "ap"]):
                        joint_pairs.append({"organism": organism, "endpoint": endpoint_name, "cohort": cohort,
                                            "comparison": f"{left}_minus_{right}", "metric": metric,
                                            "difference": point[i], "ci_low": limits[0, i], "ci_high": limits[1, i],
                                            "bootstrap": 5000, "interval_type": "paired compound percentile; descriptive post hoc"})
                mw_models = ["morgan", "morgan_maccs", "multiview", "concentration_only"]
                mw_matrix = wide.loc[ids, [f"joint:{m}" if m != "concentration_only" else m for m in mw_models]].to_numpy()
                mw_draws, rejected = [], 0
                while len(mw_draws) < 5000:
                    idx = rng_mw.integers(0, len(y), len(y))
                    if np.unique(y[idx]).size != 2:
                        rejected += 1
                        continue
                    mw_draws.append([metrics(y[idx], mw_matrix[idx, j]) for j in range(4)])
                mw_draws = np.asarray(mw_draws)
                for j, model in enumerate(mw_models[:-1]):
                    point = np.array(metrics(y, mw_matrix[:, j])) - np.array(metrics(y, mw_matrix[:, -1]))
                    limits = np.quantile(mw_draws[:, j] - mw_draws[:, -1], [0.025, 0.975], axis=0)
                    for i, metric in enumerate(["auroc", "ap"]):
                        mw_pairs.append({"organism": organism, "cohort": cohort, "model": model, "metric": metric,
                                         "difference": point[i], "ci_low": limits[0, i], "ci_high": limits[1, i],
                                         "n": len(y), "positives": int(y.sum()), "bootstrap": 5000,
                                         "rejected": rejected, "seed": 20260912})

    for name, rows in [("joint_external_metrics.csv", joint_metrics), ("joint_external_metrics_by_seed.csv", joint_seed_rows),
                       ("joint_external_topk_enrichment.csv", joint_topk), ("joint_external_cohort_flow.csv", joint_flow),
                       ("joint_external_paired_differences.csv", joint_pairs), ("joint_minus_molecular_weight.csv", mw_pairs)]:
        pd.DataFrame(rows).to_csv(args.out / name, index=False)
    pd.concat(label_export).to_csv(args.out / "external_labels_mw.csv.gz", index=False)

    summary = {
        "raw_file": str(args.raw_xlsx), "raw_sha256": hashlib.sha256(args.raw_xlsx.read_bytes()).hexdigest(),
        "rows": len(raw), "strain_columns": len(strain_columns), "missing_cells": int(raw[strain_columns].isna().sum().sum()),
        "compounds_with_missing_cells": int(raw[strain_columns].isna().any(axis=1).sum()),
        "ambiguous_all_raw": {"broad": int(labels.raw_any_activity.isna().sum()),
                              "ecoli": int(labels.raw_ecoli_any_activity.isna().sum()),
                              "cdiff": int(labels.raw_cdiff_activity.isna().sum())},
        "ambiguous_primary_structured": {"broad": int((primary & pred.raw_any_activity.isna()).sum()),
                                         "ecoli": int((primary & pred.raw_ecoli_any_activity.isna()).sum()),
                                         "cdiff": int((primary & pred.raw_cdiff_activity.isna()).sum())},
        "rule": "positive if any relevant observed adjusted P <= 0.05; negative only if all relevant values are observed and > 0.05; otherwise ambiguous and excluded endpoint-wise",
    }
    (args.out / "maier_raw_label_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
