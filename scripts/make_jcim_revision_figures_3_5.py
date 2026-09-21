import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from scipy.stats import t


BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#6F5BA7"
GRAY = "#687386"
LIGHT_GRAY = "#D9DEE7"
GRID = "#E5E8ED"
TEXT = "#172033"
PALE_BLUE = "#E5F2F8"
PALE_ORANGE = "#FBEDE4"
PALE_TEAL = "#E5F3EE"
def set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def clean(ax, grid=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7, zorder=0)


def panel(ax, label, title):
    ax.set_title(title, loc="left", pad=9)
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean, len(values)
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half, len(values)


def save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(png) as image:
        image.convert("RGB").save(png, dpi=(600, 600))


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=GRAY,
        )
    )


def temporal_figure(temporal_dir, output_dir):
    metrics = pd.read_csv(temporal_dir / "metrics.csv")
    counts = pd.read_csv(temporal_dir / "species_counts.csv")
    pairs = pd.read_csv(temporal_dir / "cohort_pairs_eligible.csv.gz", usecols=["cohort", "compound_inchikey", "organism"])
    overlap = json.loads((temporal_dir / "overlap_audit.json").read_text(encoding="utf-8"))

    cohort_order = ["train", "valid", "test"]
    cohort_labels = ["Training\n≤2018", "Validation\n2019–2020", "Test\n2021–2023"]
    pair_counts = counts.groupby("cohort")["pairs"].sum().reindex(cohort_order).astype(int)
    compound_counts = pairs.groupby("cohort")["compound_inchikey"].nunique().reindex(cohort_order).astype(int)
    species_support = (counts.loc[counts["pairs"].gt(0)].groupby("cohort")["organism"].nunique().reindex(cohort_order).astype(int))
    assert pair_counts.tolist() == [179208, 16461, 13999]
    assert compound_counts.tolist() == [51001, 6275, 5359]
    assert species_support.tolist() == [41, 40, 40]
    assert overlap["exact_compound_overlap_zero"] is True

    fig = plt.figure(figsize=(8.4, 6.6))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.15, top=0.90, hspace=0.42, wspace=0.31)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle("Retrospective temporal transfer is harder and support-sensitive", x=0.02, y=0.975, ha="left", fontsize=11.4, fontweight="bold")

    ax_a.axis("off")
    panel(ax_a, "A", "Document-year cohorts separate future compounds")
    xs = [0.02, 0.355, 0.69]
    fills = [PALE_BLUE, PALE_TEAL, PALE_ORANGE]
    edges = [BLUE, TEAL, ORANGE]
    for i, (x, label, fill, edge) in enumerate(zip(xs, cohort_labels, fills, edges)):
        box = FancyBboxPatch(
            (x, 0.30), 0.275, 0.48, transform=ax_a.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=fill, edgecolor=edge, linewidth=1.25,
        )
        ax_a.add_patch(box)
        ax_a.text(x + 0.1375, 0.69, label, transform=ax_a.transAxes, ha="center", va="center", fontsize=9.0, fontweight="bold", color=edge)
        ax_a.text(x + 0.1375, 0.52, f"{pair_counts.iloc[i]:,} pairs", transform=ax_a.transAxes, ha="center", fontsize=8.2, fontweight="bold")
        ax_a.text(x + 0.1375, 0.41, f"{compound_counts.iloc[i]:,} compounds", transform=ax_a.transAxes, ha="center", fontsize=8.0)
        ax_a.text(x + 0.1375, 0.32, f"{species_support.iloc[i]} species", transform=ax_a.transAxes, ha="center", fontsize=8.0)
        if i < 2:
            arrow(ax_a, (x + 0.285, 0.54), (xs[i + 1] - 0.01, 0.54))
    ax_a.text(0.50, 0.17, "Exact compound overlap between cohorts = 0", transform=ax_a.transAxes, ha="center", fontsize=8.3, color=TEAL, fontweight="bold")
    ax_a.text(0.50, 0.07, "205 test scaffolds and 34 test parents also occur in training", transform=ax_a.transAxes, ha="center", fontsize=8.0, color=GRAY)

    test_support = counts.loc[(counts["cohort"].eq("test")) & counts["pairs"].gt(0), ["organism", "pairs"]].sort_values("pairs").reset_index(drop=True)
    assert len(test_support) == 40
    rank = np.arange(1, len(test_support) + 1)
    low = test_support["pairs"].to_numpy() < 10
    ax_b.scatter(rank[~low], test_support.loc[~low, "pairs"], s=26, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax_b.scatter(rank[low], test_support.loc[low, "pairs"], s=34, color=ORANGE, marker="D", edgecolor="white", linewidth=0.5, zorder=4, label="<10 test pairs")
    for threshold in [10, 50, 100]:
        ax_b.axhline(threshold, color=LIGHT_GRAY, linewidth=0.8, linestyle="--", zorder=1)
    threshold_text = "  |  ".join(f"n≥{v}: {(test_support['pairs'] >= v).sum()} species" for v in [10, 20, 50, 100])
    ax_b.text(0.02, 0.96, threshold_text.replace("  |  n≥50", "\nn≥50"), transform=ax_b.transAxes, va="top", fontsize=8.0, color=GRAY)
    multi_species = metrics.loc[(metrics["scope"].eq("species")) & metrics["model"].eq("multiview"), ["organism", "mae"]].merge(test_support, on="organism")
    multi_all = multi_species["mae"].mean()
    multi_100 = multi_species.loc[multi_species["pairs"].ge(100), "mae"].mean()
    ax_b.text(0.98, 0.05, f"Multi-view equal-species MAE\n{multi_all:.3f} (all 40) → {multi_100:.3f} (n≥100; 20 species)", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8.0, color=ORANGE)
    ax_b.set_yscale("log")
    ax_b.set_ylim(1.5, 5000)
    ax_b.set_yticks([2, 10, 50, 100, 500, 1000, 3000], ["2", "10", "50", "100", "500", "1,000", "3,000"])
    ax_b.set_xlabel("Species ranked by 2021–2023 test support")
    ax_b.set_ylabel("Test pairs per species (log scale)")
    ax_b.legend(loc="center left", frameon=False)
    clean(ax_b)
    panel(ax_b, "B", "Test support spans 2–3,032 pairs/species")

    ax_c.axis("off")
    panel(ax_c, "C", "Models remain close on the temporal test")
    mae_ax = ax_c.inset_axes([0.03, 0.09, 0.44, 0.74])
    rho_ax = ax_c.inset_axes([0.57, 0.09, 0.40, 0.74])
    scopes = [("pooled", "Pooled"), ("macro", "Equal-species")]
    for inner, metric, title, xlim in [
        (mae_ax, "mae", "MAE ↓", (2.20, 2.82)),
        (rho_ax, "spearman", "Spearman ρ ↑", (0.19, 0.48)),
    ]:
        for yi, (scope, _) in zip([0.18, 0.82], scopes[::-1]):
            block = metrics.loc[(metrics["scope"].eq(scope)) & metrics["model"].isin(["morgan", "multiview"])].set_index("model")
            a = float(block.loc["morgan", metric])
            b = float(block.loc["multiview", metric])
            inner.plot([a, b], [yi + 0.07, yi - 0.07], color=LIGHT_GRAY, linewidth=1.4, zorder=1)
            inner.scatter(a, yi + 0.07, color=BLUE, s=32, zorder=3)
            inner.scatter(b, yi - 0.07, color=ORANGE, s=32, zorder=3)
            inner.text(a, yi + 0.16, f"{a:.3f}", ha="center", va="bottom", fontsize=8.0, color=BLUE)
            inner.text(b, yi - 0.16, f"{b:.3f}", ha="center", va="top", fontsize=8.0, color=ORANGE)
        inner.set_yticks([0.18, 0.82], ["Equal-species", "Pooled"])
        inner.set_ylim(-0.08, 1.10)
        inner.set_xlim(*xlim)
        inner.set_title(title, loc="center", fontsize=8.8)
        clean(inner, "x")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, label="Morgan"), Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor=ORANGE, label="Multi-view")]
    ax_c.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    species = metrics.loc[metrics["scope"].eq("species")].pivot(index="organism", columns="model")
    specs = [
        ("mae", -1, "MAE"),
        ("rmse", -1, "RMSE"),
        ("spearman", 1, "Spearman ρ"),
        ("within_one_dilution", 1, "Within ±1"),
        ("within_two_dilutions", 1, "Within ±2"),
    ]
    summaries = []
    for metric, direction, label in specs:
        delta = direction * (species[(metric, "multiview")] - species[(metric, "morgan")])
        summaries.append((label, *mean_interval(delta)))
    ypos = np.arange(len(summaries))[::-1]
    for y, (label, mean, low_ci, high_ci, n) in zip(ypos, summaries):
        ax_d.errorbar(mean, y, xerr=[[mean - low_ci], [high_ci - mean]], fmt="o", color=ORANGE, ecolor=ORANGE, markersize=5.5, capsize=3, linewidth=1.3, zorder=3)
        ax_d.text(high_ci + 0.004, y, f"{mean:+.3f} (n={n})", va="center", fontsize=8.0, color=TEXT)
    ax_d.axvline(0, color=GRAY, linewidth=1.0, linestyle="--")
    ax_d.set_yticks(ypos, [x[0] for x in summaries])
    ax_d.set_xlim(-0.10, 0.105)
    ax_d.set_xlabel("Benefit-aligned multi-view difference\n(mean ± 95% descriptive stability interval)")
    clean(ax_d, "x")
    panel(ax_d, "D", "Species-level gains are small and metric-dependent")

    fig.text(0.02, 0.025, "Boundary: ChEMBL document year is a database timestamp proxy, not compound-discovery time; this is retrospective, not prospective, validation.", fontsize=8.0, color=GRAY)
    save(fig, output_dir, "fig3_temporal_transfer")


def cold_start_figure(cold_start_dir, output_dir):
    summary = pd.read_csv(cold_start_dir / "metrics_summary.csv")
    paired = pd.read_csv(cold_start_dir / "paired_delta_summary.csv")
    overlap = pd.read_csv(cold_start_dir / "overlap_checks.csv")
    assert overlap["pathogen_class_known_from_train"].all()
    assert overlap["train_test_species_overlap"].sum() == 0
    assert int(overlap["test_pairs"].sum()) == 214068
    assert int(overlap["test_compounds_exact_nonoverlap"].sum()) == 21485

    fig = plt.figure(figsize=(8.4, 6.6))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.15, top=0.90, hspace=0.42, wspace=0.31)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle("LOSO reveals compound reuse and no consistent broad-class conditioning gain", x=0.02, y=0.975, ha="left", fontsize=11.4, fontweight="bold")

    ax_a.axis("off")
    panel(ax_a, "A", "Each fold holds out one species, not a class")
    boxes = [
        (0.02, 0.53, 0.27, 0.28, PALE_BLUE, BLUE, "TRAIN", "47 species\nmolecular\nstructure"),
        (0.38, 0.53, 0.24, 0.28, PALE_TEAL, TEAL, "OPTIONAL", "broad class\none-hot"),
        (0.71, 0.53, 0.27, 0.28, PALE_ORANGE, ORANGE, "TEST", "1 held-out\nspecies\nall compounds"),
    ]
    for x, y, w, h, face, edge, heading, text in boxes:
        patch = FancyBboxPatch((x, y), w, h, transform=ax_a.transAxes, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=face, edgecolor=edge, linewidth=1.25)
        ax_a.add_patch(patch)
        ax_a.text(x + w / 2, y + h * 0.72, heading, transform=ax_a.transAxes, ha="center", fontsize=8.1, color=edge, fontweight="bold")
        ax_a.text(x + w / 2, y + h * 0.30, text, transform=ax_a.transAxes, ha="center", va="center", fontsize=8.0, fontweight="bold", linespacing=1.12)
    arrow(ax_a, (0.29, 0.67), (0.37, 0.67))
    arrow(ax_a, (0.62, 0.67), (0.70, 0.67))
    ax_a.text(0.50, 0.34, "All four held-out broad classes occur in training", transform=ax_a.transAxes, ha="center", fontsize=8.4, color=TEAL, fontweight="bold")
    ax_a.text(0.50, 0.18, "Tests compound transfer within known classes", transform=ax_a.transAxes, ha="center", fontsize=8.5, fontweight="bold")
    ax_a.text(0.50, 0.08, "No representation for an unseen species → not zero-shot species prediction", transform=ax_a.transAxes, ha="center", fontsize=8.0, color=GRAY)

    cohort_order = ["all_pairs", "compound_seen_in_other_species", "exact_compound_nonoverlap"]
    cohort_labels = ["All held-out\npairs", "Compound seen in\nother species", "Exact compound\nnonoverlap"]
    model_map = {"morgan_compound_only": ("Compound only", BLUE), "morgan_pathogen_class": ("+ Broad class", ORANGE)}
    x = np.arange(3)
    for model, (label, color) in model_map.items():
        block = summary.loc[summary["model"].eq(model)].set_index("cohort").loc[cohort_order]
        vals = block["mae_equal_species_mean"].to_numpy()
        offset = -0.06 if model == "morgan_compound_only" else 0.06
        ax_b.plot(x, vals, color=color, linewidth=1.7, marker="o", markersize=5.5, label=label, zorder=3)
        for xi, value in zip(x, vals):
            ax_b.text(xi + offset, value + (0.035 if model == "morgan_pathogen_class" else -0.055), f"{value:.3f}", ha="center", va="center", fontsize=8.0, color=color)
    ax_b.set_xticks(x, cohort_labels)
    ax_b.set_ylim(1.78, 2.68)
    ax_b.set_ylabel("Equal-species MAE (log2 MIC)")
    ax_b.legend(loc="upper left", frameon=False)
    clean(ax_b, "y")
    panel(ax_b, "B", "Exact-compound nonoverlap is markedly harder")

    effect = paired.loc[paired["contrast"].eq("mae_pathogen_class_minus_compound_only")].set_index("cohort").loc[cohort_order]
    y = np.arange(3)[::-1]
    for yi, (_, row) in zip(y, effect.iterrows()):
        mean = row["equal_species_mean_delta"]
        low_ci = row["equal_species_mean_delta_descriptive_stability_interval95_low"]
        high_ci = row["equal_species_mean_delta_descriptive_stability_interval95_high"]
        ax_c.errorbar(mean, yi, xerr=[[mean - low_ci], [high_ci - mean]], fmt="o", color=ORANGE, ecolor=ORANGE, markersize=5.5, capsize=3, linewidth=1.3, zorder=3)
        ax_c.text(high_ci + 0.004, yi + 0.11, f"{mean:+.3f}", va="center", fontsize=8.0)
        ax_c.text(min(high_ci + 0.004, 0.045), yi - 0.20, f"{int(row['species_with_delta_below_zero'])}/{int(row['species_evaluable'])} species lower", va="center", fontsize=8.0, color=GRAY)
    ax_c.axvline(0, color=GRAY, linewidth=1.0, linestyle="--")
    ax_c.set_yticks(y, cohort_labels)
    ax_c.set_xlim(-0.05, 0.125)
    ax_c.set_ylim(-0.38, 2.38)
    ax_c.set_xlabel("Broad-class − compound-only MAE; negative favors class\n(mean ± 95% descriptive stability interval)")
    clean(ax_c, "x")
    panel(ax_c, "C", "Broad class adds no consistent benefit")

    overlap = overlap.assign(nonoverlap_fraction=overlap["test_compounds_exact_nonoverlap"] / overlap["test_compounds"])
    ax_d.scatter(
        overlap["test_compounds"], 100 * overlap["nonoverlap_fraction"],
        s=22 + 1.2 * np.sqrt(overlap["test_compounds_exact_nonoverlap"]),
        color=BLUE, alpha=0.72, edgecolor="white", linewidth=0.5, zorder=3,
    )
    for name, dx, dy in [("Helicobacter pylori", 8, 0), ("Mycobacterium tuberculosis", 8, -10), ("Staphylococcus aureus", -92, 8)]:
        row = overlap.loc[overlap["held_out_species"].eq(name)].iloc[0]
        ax_d.annotate(name.replace(" ", "\n", 1), (row["test_compounds"], 100 * row["nonoverlap_fraction"]), xytext=(dx, dy), textcoords="offset points", fontsize=8.0, fontstyle="italic", ha="left")
    weighted = overlap["test_compounds_exact_nonoverlap"].sum() / overlap["test_compounds"].sum()
    median = overlap["nonoverlap_fraction"].median()
    zeros = int(overlap["test_compounds_exact_nonoverlap"].eq(0).sum())
    ax_d.text(0.98, 0.64, f"Overall: {weighted:.1%} nonoverlap\nMedian species: {median:.1%}; zero support: {zeros}\nPoint area ∝ nonoverlap count", transform=ax_d.transAxes, ha="right", va="center", fontsize=8.0, color=GRAY)
    ax_d.set_xscale("log")
    ax_d.set_xlim(430, 60000)
    ax_d.set_ylim(-4, 92)
    ax_d.set_xlabel("Held-out pairs per species (log scale)")
    ax_d.set_ylabel("Exact-compound nonoverlap (%)")
    clean(ax_d, "both")
    panel(ax_d, "D", "Exact-compound-nonoverlap support is imbalanced")

    fig.text(0.02, 0.025, "Boundary: folds overlap and all held-out species belong to broad classes already present in training; LOSO does not establish zero-shot species representation.", fontsize=8.0, color=GRAY)
    save(fig, output_dir, "fig4_loso_transfer")


def chemical_figure(chemical_dir, output_dir):
    bins = pd.read_csv(chemical_dir / "similarity_error_bins.csv")
    similarity = pd.read_csv(chemical_dir / "similarity_error_summary.csv")
    compound_error = pd.read_csv(
        chemical_dir / "similarity_error_by_compound.csv.gz",
        usecols=["compound_inchikey", "model", "seed", "is_acyclic", "absolute_error_mean_across_species"],
    )
    compounds = pd.read_csv(chemical_dir / "compound_scaffolds.csv.gz", usecols=["compound_inchikey", "is_acyclic"])
    cliffs = pd.read_csv(chemical_dir / "cliff_error_summary.csv")
    cliff_groups = pd.read_csv(
        chemical_dir / "apparent_mic_cliff_scaffold_summary.csv",
        usecols=["configuration", "n_apparent_cliff_edges"],
    )
    models = ["morgan", "multiview"]
    bins = bins.loc[bins["model"].isin(models)]
    compound_error = compound_error.loc[compound_error["model"].isin(models)]
    assert compounds["compound_inchikey"].nunique() == 63486
    assert compounds.loc[compounds["is_acyclic"], "compound_inchikey"].nunique() == 948
    test_acyclic = compound_error.loc[compound_error["model"].eq("morgan") & compound_error["is_acyclic"]]
    test_acyclic_by_seed = test_acyclic.groupby("seed")["compound_inchikey"].nunique()
    assert (int(test_acyclic_by_seed.min()), int(test_acyclic_by_seed.max())) == (106, 163)
    assert test_acyclic["compound_inchikey"].nunique() == 522

    fig = plt.figure(figsize=(8.4, 6.7))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.16, top=0.90, hspace=0.45, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle("Chemical-space effects are weak, sparse, and definition-sensitive", x=0.02, y=0.975, ha="left", fontsize=11.4, fontweight="bold")

    bin_order = ["[0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,0.90]", "(0.90,1.00]"]
    bin_labels = ["0–0.25", ">0.25–0.50", ">0.50–0.75", ">0.75–0.90", ">0.90–1.00"]
    x = np.arange(len(bin_order))
    mean_counts = bins.groupby("similarity_bin")["n_compounds"].mean().reindex(bin_order)
    for model, color, label in [("morgan", BLUE, "Morgan"), ("multiview", ORANGE, "Multi-view")]:
        means, lows, highs = [], [], []
        for bin_name in bin_order:
            values = bins.loc[(bins["model"].eq(model)) & bins["similarity_bin"].eq(bin_name), "mean_absolute_error"]
            mean, low_ci, high_ci, _ = mean_interval(values)
            means.append(mean)
            lows.append(low_ci)
            highs.append(high_ci)
        means = np.asarray(means)
        ax_a.errorbar(x, means, yerr=[means - np.asarray(lows), np.asarray(highs) - means], color=color, marker="o", markersize=5.0, capsize=3, linewidth=1.6, label=label, zorder=3)
    ax_a.set_xticks(x, [f"{label}\nn≈{count:,.0f}" for label, count in zip(bin_labels, mean_counts)])
    ax_a.set_ylabel("Equal-compound MAE (log2 MIC)")
    ax_a.set_xlabel("Nearest-training-compound Morgan Tanimoto bin")
    ax_a.set_ylim(1.72, 3.03)
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.text(0.02, 0.06, "Strictly monotonic in only 2/5 overlapping splits", transform=ax_a.transAxes, fontsize=8.0, color=GRAY)
    clean(ax_a, "y")
    panel(ax_a, "A", "Similarity bins show only a weak error gradient")

    rho = similarity.loc[
        similarity["model"].isin(models)
        & similarity["scope"].eq("all_with_compound_specific_acyclic_fallback")
        & similarity["inference_unit"].eq("equal_compound")
    ]
    for i, (model, color, label) in enumerate([(models[0], BLUE, "Morgan"), (models[1], ORANGE, "Multi-view")]):
        values = rho.loc[rho["model"].eq(model), "spearman_similarity_absolute_error"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax_b.scatter(i + jitter, values, s=25, facecolor="white", edgecolor=color, linewidth=1.0, zorder=3)
        mean, low_ci, high_ci, _ = mean_interval(values)
        ax_b.errorbar(i, mean, yerr=[[mean - low_ci], [high_ci - mean]], fmt="o", color=color, markersize=6.2, capsize=4, linewidth=1.5, zorder=4)
        ax_b.text(i, high_ci + 0.012, f"{mean:+.3f}", ha="center", fontsize=8.2, color=color, fontweight="bold")
    ax_b.axhline(0, color=GRAY, linewidth=1.0, linestyle="--")
    ax_b.set_xticks([0, 1], ["Morgan", "Multi-view"])
    ax_b.set_ylim(-0.12, 0.075)
    ax_b.set_ylabel("Spearman ρ: similarity vs absolute error")
    ax_b.text(0.02, 0.05, "Points: scaffold splits\nBars: 95% descriptive stability intervals", transform=ax_b.transAxes, fontsize=8.0, color=GRAY)
    clean(ax_b, "y")
    panel(ax_b, "B", "Similarity–error rank correlation remains near zero")

    acyclic = compound_error.groupby(["model", "seed", "is_acyclic"])["absolute_error_mean_across_species"].mean().unstack()
    acyclic["delta"] = acyclic[True] - acyclic[False]
    for i, (model, color, label) in enumerate([(models[0], BLUE, "Morgan"), (models[1], ORANGE, "Multi-view")]):
        values = acyclic.loc[model, "delta"].to_numpy()
        yjit = i + np.linspace(-0.07, 0.07, len(values))
        ax_c.scatter(values, yjit, s=25, facecolor="white", edgecolor=color, linewidth=1.0, zorder=3)
        mean, low_ci, high_ci, _ = mean_interval(values)
        ax_c.errorbar(mean, i, xerr=[[mean - low_ci], [high_ci - mean]], fmt="o", color=color, markersize=6.2, capsize=4, linewidth=1.5, zorder=4)
        ax_c.text(mean, i + 0.13, f"{mean:+.3f}", ha="center", va="center", fontsize=8.2, color=color, fontweight="bold")
    all_scope = similarity.loc[similarity["model"].isin(models) & similarity["inference_unit"].eq("equal_compound")]
    scope_pivot = all_scope.pivot(index=["model", "seed"], columns="scope", values="mean_absolute_error")
    scope_pivot["shift"] = scope_pivot["all_with_compound_specific_acyclic_fallback"] - scope_pivot["nonempty_murcko_only"]
    shifts = scope_pivot.groupby(level=0)["shift"].mean()
    ax_c.axvline(0, color=GRAY, linewidth=1.0, linestyle="--")
    ax_c.set_yticks([0, 1], ["Morgan", "Multi-view"])
    ax_c.set_xlim(-0.05, 0.70)
    ax_c.set_xlabel("Acyclic − cyclic MAE\n(mean ± 95% descriptive stability interval)")
    ax_c.text(0.02, 0.61, "Full corpus: 948/63,486 acyclic (1.49%)", transform=ax_c.transAxes, va="center", fontsize=8.2, color=GRAY)
    ax_c.text(0.02, 0.51, "Test splits: 106–163; 522 unique across splits", transform=ax_c.transAxes, va="center", fontsize=8.0, color=GRAY)
    ax_c.text(0.02, 0.40, f"Overall shift: Morgan {shifts['morgan']:+.4f}; multi-view {shifts['multiview']:+.4f}", transform=ax_c.transAxes, va="center", fontsize=8.0, color=GRAY)
    clean(ax_c, "x")
    panel(ax_c, "C", "Acyclic compounds are rare but harder")

    config_order = ["sensitivity_sim0.7_delta2", "primary_sim0.8_delta2", "sensitivity_sim0.9_delta2", "sensitivity_sim0.8_delta3"]
    config_labels = ["sim≥0.7; |ΔMIC|≥2", "sim≥0.8; |ΔMIC|≥2", "sim≥0.9; |ΔMIC|≥2", "sim≥0.8; |ΔMIC|≥3"]
    across = cliffs.loc[cliffs["model"].isin(models) & cliffs["summary_level"].eq("across_seed_descriptive_stability")].set_index(["configuration", "model"])
    prevalence = 100 * cliff_groups.assign(has_edge=cliff_groups["n_apparent_cliff_edges"].gt(0)).groupby("configuration")["has_edge"].mean()
    ybase = np.arange(len(config_order))[::-1]
    for offset, model, color, label in [(0.09, "morgan", BLUE, "Morgan"), (-0.09, "multiview", ORANGE, "Multi-view")]:
        for y, config in zip(ybase, config_order):
            row = across.loc[(config, model)]
            mean = row["delta_mae_cliff_minus_nonparticipant_equal_species"]
            low_ci = row["delta_descriptive_t_stability_interval_low"]
            high_ci = row["delta_descriptive_t_stability_interval_high"]
            ax_d.errorbar(mean, y + offset, xerr=[[mean - low_ci], [high_ci - mean]], fmt="o", color=color, ecolor=color, markersize=5.2, capsize=3, linewidth=1.2, zorder=3)
    ax_d.axvline(0, color=GRAY, linewidth=1.0, linestyle="--")
    ax_d.set_yticks(ybase, [f"{label}\n{prevalence[config]:.1f}% groups" for label, config in zip(config_labels, config_order)])
    ax_d.set_xlim(-0.17, 0.46)
    ax_d.set_xlabel("Cliff participant − nonparticipant MAE\n(mean ± 95% descriptive stability interval)")
    ax_d.legend(handles=[Line2D([0], [0], marker="o", color=BLUE, label="Morgan"), Line2D([0], [0], marker="o", color=ORANGE, label="Multi-view")], loc="upper right", ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.0)
    clean(ax_d, "x")
    panel(ax_d, "D", "Apparent-cliff error depends on the definition")

    fig.text(0.02, 0.025, "Intervals are descriptive t stability intervals over five overlapping scaffold splits. Cliff participation is outcome-defined and post hoc; edges are not inference units.\nCliff prevalence denominator: 27,451 species × nonempty-Murcko-scaffold groups.", fontsize=8.0, color=GRAY)
    save(fig, output_dir, "fig5_chemical_space_boundaries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--cold-start-dir", type=Path, required=True)
    parser.add_argument("--chemical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    temporal_figure(args.temporal_dir, args.output_dir)
    cold_start_figure(args.cold_start_dir, args.output_dir)
    chemical_figure(args.chemical_dir, args.output_dir)


if __name__ == "__main__":
    main()
