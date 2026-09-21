from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image


MODEL_COLORS = {
    "species_median": "#64748B",
    "morgan": "#0077C8",
    "multiview": "#E43D30",
    "mole": "#00A676",
    "molformer": "#7A4CC2",
}
MODEL_LABELS = {
    "species_median": "Species median",
    "morgan": "Morgan",
    "multiview": "Multi-view",
    "mole": "MolE",
    "molformer": "MoLFormer",
}
VIEW_COLORS = {
    "morgan": "#0077C8",
    "morgan_physchem": "#00A6D6",
    "morgan_maccs": "#F4A300",
    "morgan_graph": "#D81B60",
    "multiview": "#E43D30",
}
VIEW_LABELS = {
    "morgan": "Morgan",
    "morgan_physchem": "+ Physchem",
    "morgan_maccs": "+ MACCS",
    "morgan_graph": "+ Graph",
    "multiview": "Full multi-view",
}
CLASS_COLORS = {
    "gram-negative": "#0077C8",
    "gram-positive": "#E43D30",
    "fungi": "#7A4CC2",
    "acid-fast": "#F4A300",
}
CLASS_LABELS = {
    "gram-negative": "Gram-negative",
    "gram-positive": "Gram-positive",
    "fungi": "Fungi",
    "acid-fast": "Acid-fast",
}
ACCENT = "#00A676"
GRID = "#DDE3EA"
TEXT = "#172033"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.6,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "legend.fontsize": 6.9,
            "text.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.65, zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.09, 1.08, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")


def headline(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, y=0.985, ha="left", fontsize=10.6, fontweight="bold", color=TEXT)


def finish_grid(fig: plt.Figure) -> None:
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.92), pad=0.8, h_pad=2.2, w_pad=1.7)


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
    plt.close(fig)


def source(df: pd.DataFrame, source_dir: Path, name: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(source_dir / name, index=False)


def seed_summary(ax: plt.Axes, macro: pd.DataFrame, metric: str, models: list[str], ylabel: str) -> None:
    x = np.arange(len(models))
    for i, model in enumerate(models):
        values = macro.loc[macro["model"].eq(model), metric].dropna().to_numpy()
        if not len(values):
            continue
        mean = values.mean()
        sd = values.std(ddof=1) if len(values) > 1 else 0
        ax.errorbar(i, mean, yerr=sd, fmt="o", ms=7.2, color=MODEL_COLORS[model],
                    ecolor=MODEL_COLORS[model], capsize=3, lw=1.4, zorder=3)
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=18, facecolor="white",
                   edgecolor=MODEL_COLORS[model], linewidth=0.9, zorder=4)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in models], rotation=24, ha="right")
    ax.set_ylabel(ylabel)
    clean_axis(ax, "y")


def figure1(table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    flow = pd.read_csv(table_dir / "dataset_flow.csv")
    species = pd.read_csv(table_dir / "supplementary_table_s1_species.csv")
    class_summary = species.groupby("pathogen_class", as_index=False).agg(
        species=("organism", "nunique"), pairs=("compound_species_pairs", "sum")
    )
    top = species.nlargest(12, "compound_species_pairs").sort_values("compound_species_pairs")

    fig = plt.figure(figsize=(8.2, 5.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.35])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    headline(fig, "Taxonomy-audited curation yields 48 accepted species and 214,068 MIC pairs")

    ax_a.axis("off")
    labels = [
        ("Exact MIC", "442,271", "validity-filtered"),
        ("Pathogen class", "421,128", "-21,143 rows"),
        ("Species ancestor", "419,169", "-1,959 rows"),
        ("Valid structure", "419,162", "-7 rows"),
        ("Median pairs", "238,679", "aggregation"),
        (">=500/species", "214,068", "48 species"),
    ]
    fills = ["#DDF1FF", "#D9F7F0", "#FFF0C7", "#FCE0EC", "#E9DDF8", "#FFE0D8"]
    edges = ["#0077C8", "#00A676", "#F4A300", "#D81B60", "#7A4CC2", "#E43D30"]
    xs = np.linspace(0.085, 0.915, len(labels))
    for i, ((title, count, note), x, fill, edge) in enumerate(zip(labels, xs, fills, edges)):
        box = FancyBboxPatch((x - 0.068, 0.26), 0.136, 0.52,
                             boxstyle="round,pad=0.012,rounding_size=0.018",
                             transform=ax_a.transAxes, facecolor=fill, edgecolor=edge, linewidth=1.4)
        ax_a.add_patch(box)
        ax_a.text(x, 0.66, title, transform=ax_a.transAxes, ha="center", fontweight="bold", fontsize=7.5)
        ax_a.text(x, 0.49, count, transform=ax_a.transAxes, ha="center", fontweight="bold", fontsize=11, color=edge)
        ax_a.text(x, 0.34, note, transform=ax_a.transAxes, ha="center", fontsize=6.5)
        if i < len(labels) - 1:
            ax_a.annotate("", xy=(xs[i + 1] - 0.078, 0.52), xytext=(x + 0.078, 0.52),
                          xycoords=ax_a.transAxes, textcoords=ax_a.transAxes,
                          arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#536273"})
    ax_a.text(0.5, 0.08, "48 canonical names = 48 accepted NCBI tax IDs; historical alias 178876 resolved to 5207",
              transform=ax_a.transAxes, ha="center", fontsize=7.3, fontweight="bold", color="#7A4CC2")
    ax_a.text(-0.03, 0.80, "A", transform=ax_a.transAxes, fontsize=10.5, fontweight="bold", va="top")

    class_order = class_summary.sort_values("pairs")["pathogen_class"].tolist()
    class_plot = class_summary.set_index("pathogen_class").loc[class_order]
    y = np.arange(len(class_plot))
    ax_b.barh(y, class_plot["pairs"] / 1000, color=[CLASS_COLORS[c] for c in class_order], height=0.68)
    for yi, (cls, row) in enumerate(class_plot.iterrows()):
        ax_b.text(row["pairs"] / 1000 + 1.2, yi, f"{int(row['pairs']):,} pairs | {int(row['species'])} species",
                  va="center", fontsize=6.8, color=TEXT)
    ax_b.set_yticks(y, [CLASS_LABELS[c] for c in class_order])
    ax_b.set_xlabel("Compound-species pairs (thousands)")
    ax_b.set_title("Four retained pathogen classes", loc="left")
    ax_b.set_xlim(0, class_plot["pairs"].max() / 1000 * 1.35)
    clean_axis(ax_b, "x")
    panel_label(ax_b, "B")

    y = np.arange(len(top))
    ax_c.barh(y, top["compound_species_pairs"] / 1000,
              color=[CLASS_COLORS[c] for c in top["pathogen_class"]], height=0.72)
    ax_c.set_yticks(y, top["organism"])
    for tick in ax_c.get_yticklabels():
        tick.set_fontstyle("italic")
    ax_c.set_xlabel("Compound-species pairs (thousands)")
    ax_c.set_title("Largest eligible species cohorts", loc="left")
    clean_axis(ax_c, "x")
    panel_label(ax_c, "C")

    source(flow, source_dir, "fig1_dataset_flow.csv")
    source(class_summary, source_dir, "fig1_pathogen_classes.csv")
    source(top, source_dir, "fig1_largest_species.csv")
    save(fig, output_dir, "fig1_dataset_curation")


def figure2(species_dir: Path, table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    manifest = json.loads((species_dir / "manifest.json").read_text(encoding="utf-8"))
    reports = pd.json_normalize(manifest["split_reports"], sep="_")
    metrics = pd.read_csv(species_dir / "metrics.csv")
    species = pd.read_csv(table_dir / "supplementary_table_s1_species.csv")
    support = metrics.loc[(metrics["scope"].eq("species")) & (metrics["model"].eq("morgan"))]
    support = support.groupby("organism", as_index=False)["n"].agg(["min", "mean", "max"]).reset_index()
    support = support.merge(species[["organism", "pathogen_class"]], on="organism", how="left").sort_values("mean")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.45))
    headline(fig, "Five compound-level scaffold splits have zero assigned-scaffold leakage and stable support")

    seeds = reports["seed"].astype(str).tolist()
    partitions = ["train", "valid", "test"]
    split_colors = {"train": "#0077C8", "valid": "#F4A300", "test": "#E43D30"}
    bottom = np.zeros(len(reports))
    for part in partitions:
        values = reports[f"compound_counts_{part}"].to_numpy()
        axes[0, 0].bar(seeds, values, bottom=bottom, color=split_colors[part], label=part.capitalize())
        bottom += values
    axes[0, 0].set_ylabel("Compounds")
    axes[0, 0].set_title("Compound allocation is stable across seeds", loc="left")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center")
    clean_axis(axes[0, 0], "y")
    panel_label(axes[0, 0], "A")

    x = np.arange(len(reports))
    width = 0.24
    for j, part in enumerate(partitions):
        axes[0, 1].bar(x + (j - 1) * width, reports[f"unique_scaffolds_{part}"], width,
                       color=split_colors[part], label=part.capitalize())
    axes[0, 1].set_xticks(x, seeds)
    axes[0, 1].set_xlabel("Seed")
    axes[0, 1].set_ylabel("Unique assigned scaffolds")
    axes[0, 1].set_title("Scaffold groups remain partition-specific", loc="left")
    clean_axis(axes[0, 1], "y")
    panel_label(axes[0, 1], "B")

    overlap_cols = ["scaffold_overlap_train_valid", "scaffold_overlap_train_test", "scaffold_overlap_valid_test"]
    overlap = reports[overlap_cols].to_numpy()
    axes[1, 0].imshow(np.ones_like(overlap), cmap=plt.matplotlib.colors.ListedColormap(["#C8F3E5"]), aspect="auto")
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            axes[1, 0].text(j, i, str(int(overlap[i, j])), ha="center", va="center",
                            fontsize=10, fontweight="bold", color="#007C63")
    axes[1, 0].set_xticks(range(3), ["Train-valid", "Train-test", "Valid-test"])
    axes[1, 0].set_yticks(range(len(seeds)), seeds)
    axes[1, 0].set_xlabel("Partition comparison")
    axes[1, 0].set_ylabel("Seed")
    axes[1, 0].set_title("All 15 assigned-scaffold overlap checks equal zero", loc="left")
    panel_label(axes[1, 0], "C")

    ranks = np.arange(1, len(support) + 1)
    axes[1, 1].vlines(ranks, support["min"], support["max"], color="#AAB4C0", linewidth=0.8, zorder=1)
    axes[1, 1].scatter(ranks, support["mean"], s=25,
                       color=[CLASS_COLORS[c] for c in support["pathogen_class"]], zorder=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Species ordered by mean test support")
    axes[1, 1].set_ylabel("Test pairs per seed (log scale)")
    axes[1, 1].set_title(f"All 48 species are represented; minimum test support = {int(support['min'].min())}", loc="left")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[c],
                      markeredgecolor="none", label=CLASS_LABELS[c], markersize=5) for c in CLASS_COLORS]
    axes[1, 1].legend(handles=handles, frameon=False, ncol=2, loc="upper left")
    clean_axis(axes[1, 1], "y")
    panel_label(axes[1, 1], "D")

    source(reports, source_dir, "fig2_split_reports.csv")
    source(support, source_dir, "fig2_species_test_support.csv")
    finish_grid(fig)
    save(fig, output_dir, "fig2_scaffold_integrity")


def figure3(species_dir: Path, table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    metrics = pd.read_csv(species_dir / "metrics.csv")
    macro = metrics.loc[metrics["scope"].eq("macro")]
    paired = pd.read_csv(table_dir / "species_mic_paired_differences.csv")
    models = ["species_median", "morgan", "multiview", "mole", "molformer"]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.45))
    headline(fig, "Multi-view gives a stable but small average error reduction over Morgan")
    seed_summary(axes[0, 0], macro, "mae", models, "Macro-species MAE (log2 MIC)")
    axes[0, 0].set_title("Quantitative error", loc="left")
    panel_label(axes[0, 0], "A")

    learned = ["morgan", "multiview", "mole", "molformer"]
    seed_summary(axes[0, 1], macro, "spearman", learned, "Macro-species Spearman rho")
    axes[0, 1].set_title("Rank correlation", loc="left")
    panel_label(axes[0, 1], "B")

    seed_summary(axes[1, 0], macro, "within_two_dilutions", models, "Fraction within +/-2 log2 MIC units")
    axes[1, 0].set_title("Two-dilution accuracy", loc="left")
    panel_label(axes[1, 0], "C")

    metric_map = [
        ("mae", "MAE (log2 MIC)", -1),
        ("rmse", "RMSE (log2 MIC)", -1),
        ("spearman", "Spearman (unitless)", 1),
        ("within_one_dilution", "Within +/-1 (fraction)", 1),
        ("within_two_dilutions", "Within +/-2 (fraction)", 1),
    ]
    rows = []
    subset = paired.loc[paired["comparison"].eq("multiview_minus_morgan")].set_index("metric")
    for metric, label, direction in metric_map:
        row = subset.loc[metric]
        estimate = direction * row["mean_difference"]
        low = direction * row["ci_low"]
        high = direction * row["ci_high"]
        if low > high:
            low, high = high, low
        rows.append({"metric": label, "estimate": estimate, "ci_low": low, "ci_high": high, "wins": row["wins"]})
    effects = pd.DataFrame(rows)
    axes[1, 1].axis("off")
    effect_table = axes[1, 1].table(
        cellText=[[r.metric, f"{r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", f"{int(r.wins)}/5"] for r in effects.itertuples()],
        colLabels=["Metric and unit", "Oriented gain [95% interval]", "Wins"],
        colWidths=[0.40, 0.48, 0.12], cellLoc="left", bbox=[0, 0.18, 1, 0.76],
    )
    effect_table.auto_set_font_size(False)
    effect_table.set_fontsize(6.4)
    for (row, col), cell in effect_table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(weight="bold")
    axes[1, 1].text(0, 0.07, "Positive favors multi-view; magnitudes are not comparable across units.",
                     transform=axes[1, 1].transAxes, fontsize=6.4)
    axes[1, 1].set_title("Paired five-split stability intervals", loc="left")
    panel_label(axes[1, 1], "D")

    source(macro, source_dir, "fig3_macro_seed_metrics.csv")
    source(effects, source_dir, "fig3_multiview_morgan_paired_effects.csv")
    finish_grid(fig)
    save(fig, output_dir, "fig3_internal_performance")


def figure4(conditioning_dir: Path, table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    metrics = pd.read_csv(conditioning_dir / "metrics.csv")
    macro = metrics.loc[metrics["scope"].eq("macro")]
    species = pd.read_csv(table_dir / "supplementary_table_s1_species.csv")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.55))
    headline(fig, "Species identity improves average MIC prediction, but its value varies across organisms")

    pairs = [
        (axes[0, 0], "morgan_compound_only", "morgan", "Morgan", MODEL_COLORS["morgan"]),
        (axes[0, 1], "multiview_compound_only", "multiview", "Multi-view", MODEL_COLORS["multiview"]),
    ]
    macro_summary = []
    for panel, (ax, compound_only, aware, label, color) in zip(["A", "B"], pairs):
        left = macro.loc[macro["model"].eq(compound_only)].set_index("seed")
        right = macro.loc[macro["model"].eq(aware)].set_index("seed")
        seeds = sorted(set(left.index) & set(right.index))
        for seed in seeds:
            ax.plot([0, 1], [left.loc[seed, "mae"], right.loc[seed, "mae"]], color="#B7C1CC", lw=1)
            ax.scatter([0, 1], [left.loc[seed, "mae"], right.loc[seed, "mae"]], s=28,
                       facecolor="white", edgecolor=color, linewidth=1.1, zorder=3)
        means = [left.loc[seeds, "mae"].mean(), right.loc[seeds, "mae"].mean()]
        ax.plot([0, 1], means, color=color, lw=2.4, marker="o", ms=6.5, zorder=4)
        delta_mae = means[1] - means[0]
        delta_spearman = right.loc[seeds, "spearman"].mean() - left.loc[seeds, "spearman"].mean()
        delta_w2 = right.loc[seeds, "within_two_dilutions"].mean() - left.loc[seeds, "within_two_dilutions"].mean()
        ax.text(0.5, 0.97, f"Mean delta MAE {delta_mae:+.3f}\nSpearman {delta_spearman:+.3f} | within +/-2 {delta_w2:+.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color=color, fontweight="bold")
        ax.set_xticks([0, 1], ["Compound only", "+ Species identity"])
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.06)
        ax.set_ylabel("Macro-species MAE")
        ax.set_title(f"{label}: matched seed trajectories", loc="left")
        clean_axis(ax, "y")
        panel_label(ax, panel)
        macro_summary.append({"model": label, "delta_mae": delta_mae, "delta_spearman": delta_spearman,
                              "delta_within_two": delta_w2})

    per = metrics.loc[metrics["scope"].eq("species")]
    aware = per.loc[per["model"].eq("multiview")].groupby("organism")["mae"].mean()
    compound = per.loc[per["model"].eq("multiview_compound_only")].groupby("organism")["mae"].mean()
    delta = (aware - compound).rename("delta_mae").reset_index().merge(
        species[["organism", "pathogen_class"]], on="organism", how="left"
    ).sort_values("delta_mae")
    y = np.arange(len(delta))
    axes[1, 0].barh(y, delta["delta_mae"], color=[CLASS_COLORS[c] for c in delta["pathogen_class"]], height=0.72)
    axes[1, 0].axvline(0, color=TEXT, linewidth=0.9)
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xlabel("Species-aware - compound-only MAE")
    axes[1, 0].set_title(f"Species-level effects are heterogeneous ({int((delta['delta_mae'] < 0).sum())}/48 improve)", loc="left")
    clean_axis(axes[1, 0], "x")
    panel_label(axes[1, 0], "C")

    class_order = ["gram-negative", "gram-positive", "fungi", "acid-fast"]
    values = [delta.loc[delta["pathogen_class"].eq(c), "delta_mae"].to_numpy() for c in class_order]
    box = axes[1, 1].boxplot(values, patch_artist=True, widths=0.58, showfliers=False,
                             medianprops={"color": TEXT, "linewidth": 1.2},
                             whiskerprops={"color": "#7B8794"}, capprops={"color": "#7B8794"})
    for patch, cls in zip(box["boxes"], class_order):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.28)
        patch.set_edgecolor(CLASS_COLORS[cls])
    rng = np.random.default_rng(20260820)
    for i, (cls, vals) in enumerate(zip(class_order, values), start=1):
        axes[1, 1].scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=20,
                           color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.35, alpha=0.9)
    axes[1, 1].axhline(0, color=TEXT, linewidth=0.9)
    axes[1, 1].set_xticks(range(1, 5), [CLASS_LABELS[c] for c in class_order], rotation=22, ha="right")
    axes[1, 1].set_ylabel("Species-aware - compound-only MAE")
    axes[1, 1].set_title("Conditioning effects vary across pathogen classes", loc="left")
    clean_axis(axes[1, 1], "y")
    panel_label(axes[1, 1], "D")

    source(pd.DataFrame(macro_summary), source_dir, "fig4_conditioning_macro_changes.csv")
    source(delta, source_dir, "fig4_conditioning_species_deltas.csv")
    finish_grid(fig)
    save(fig, output_dir, "fig4_species_conditioning")


def figure5(table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    uncertainty = pd.read_csv(table_dir / "view_ablation_uncertainty.csv")
    paired = pd.read_csv(table_dir / "species_mic_paired_differences.csv")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.45))
    headline(fig, "Classical views differ by endpoint under a fixed downstream estimator")

    ax = axes[0, 0]
    ax.axis("off")
    blocks = [
        ("Morgan", "1,024 bits", "#0077C8", 0.27),
        ("MACCS", "167 keys", "#F4A300", 0.20),
        ("Physchem", "12 vars", "#00A6D6", 0.18),
        ("Graph", "15 vars", "#D81B60", 0.18),
    ]
    x = 0.03
    for title, detail, color, width in blocks:
        patch = FancyBboxPatch((x, 0.42), width, 0.30, boxstyle="round,pad=0.012,rounding_size=0.025",
                               transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=1)
        ax.add_patch(patch)
        ax.text(x + width / 2, 0.59, title, transform=ax.transAxes, ha="center", va="center",
                color="white", fontweight="bold", fontsize=8)
        ax.text(x + width / 2, 0.48, detail, transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=6.8)
        x += width + 0.025
    species_patch = FancyBboxPatch((0.27, 0.10), 0.46, 0.16, boxstyle="round,pad=0.01,rounding_size=0.02",
                                   transform=ax.transAxes, facecolor="#E8ECF2", edgecolor="#64748B",
                                   linestyle="--", linewidth=1.1)
    ax.add_patch(species_patch)
    ax.text(0.50, 0.18, "+ 48-level species indicator", transform=ax.transAxes, ha="center",
            fontweight="bold", color="#465567")
    ax.set_title("Full multi-view representation", loc="left")
    panel_label(ax, "A")

    internal = uncertainty.loc[(uncertainty["scope"].eq("ChEMBL scaffold holdout")) &
                               (uncertainty["metric"].eq("mae"))].copy()
    internal = internal.set_index("representation").loc[list(VIEW_LABELS)].reset_index()
    y = np.arange(len(internal))
    axes[0, 1].errorbar(internal["difference_vs_morgan"], y,
                        xerr=np.vstack([internal["difference_vs_morgan"] - internal["difference_ci_low"],
                                        internal["difference_ci_high"] - internal["difference_vs_morgan"]]),
                        fmt="none", ecolor="#536273", capsize=3, lw=1.2)
    axes[0, 1].scatter(internal["difference_vs_morgan"], y, s=48,
                       color=[VIEW_COLORS[r] for r in internal["representation"]], zorder=3)
    axes[0, 1].axvline(0, color=TEXT, linestyle="--", linewidth=1)
    axes[0, 1].set_yticks(y, [VIEW_LABELS[r] for r in internal["representation"]])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("MAE difference vs Morgan (negative favors view)")
    axes[0, 1].set_title("Internal paired five-split ablation", loc="left")
    clean_axis(axes[0, 1], "x")
    panel_label(axes[0, 1], "B")

    external = uncertainty.loc[(uncertainty["scope"].eq("Maier broad primary")) &
                               (uncertainty["metric"].eq("auprc"))].copy()
    external = external.set_index("representation").loc[list(VIEW_LABELS)].reset_index()
    y = np.arange(len(external))
    axes[1, 0].errorbar(external["difference_vs_morgan"], y,
                        xerr=np.vstack([external["difference_vs_morgan"] - external["difference_ci_low"],
                                        external["difference_ci_high"] - external["difference_vs_morgan"]]),
                        fmt="none", ecolor="#536273", capsize=3, lw=1.2)
    axes[1, 0].scatter(external["difference_vs_morgan"], y, s=48,
                       color=[VIEW_COLORS[r] for r in external["representation"]], zorder=3)
    axes[1, 0].axvline(0, color=TEXT, linestyle="--", linewidth=1)
    axes[1, 0].set_yticks(y, [VIEW_LABELS[r] for r in external["representation"]])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Broad-screen AUPRC difference vs Morgan")
    axes[1, 0].set_title("External gain is distributed across compact views", loc="left")
    clean_axis(axes[1, 0], "x")
    panel_label(axes[1, 0], "C")

    comparisons = ["multiview_minus_morgan", "mole_minus_morgan", "molformer_minus_morgan"]
    labels = ["Multi-view", "MolE", "MoLFormer"]
    colors = [MODEL_COLORS["multiview"], MODEL_COLORS["mole"], MODEL_COLORS["molformer"]]
    rows = []
    for comparison, label in zip(comparisons, labels):
        row = paired.loc[(paired["comparison"].eq(comparison)) & (paired["metric"].eq("mae"))].iloc[0]
        rows.append({"representation": label, "mae_reduction": -row["mean_difference"],
                     "ci_low": -row["ci_high"], "ci_high": -row["ci_low"]})
    pretrained = pd.DataFrame(rows)
    y = np.arange(len(pretrained))
    axes[1, 1].errorbar(pretrained["mae_reduction"], y,
                        xerr=np.vstack([pretrained["mae_reduction"] - pretrained["ci_low"],
                                        pretrained["ci_high"] - pretrained["mae_reduction"]]),
                        fmt="none", ecolor="#536273", capsize=3, lw=1.2)
    axes[1, 1].scatter(pretrained["mae_reduction"], y, s=55, color=colors, zorder=3)
    axes[1, 1].axvline(0, color=TEXT, linestyle="--", linewidth=1)
    axes[1, 1].set_yticks(y, pretrained["representation"])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("MAE reduction vs Morgan (positive is better)")
    axes[1, 1].set_title("Frozen embeddings under the fixed MIC model", loc="left")
    clean_axis(axes[1, 1], "x")
    panel_label(axes[1, 1], "D")

    source(internal, source_dir, "fig5_internal_view_ablation.csv")
    source(external, source_dir, "fig5_external_view_ablation.csv")
    source(pretrained, source_dir, "fig5_pretrained_paired_mae.csv")
    finish_grid(fig)
    save(fig, output_dir, "fig5_representation_ablation")


def figure6(species_dir: Path, table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    coverage = pd.read_csv(table_dir / "species_mic_coverage_summary.csv")
    seed_coverage = pd.read_csv(table_dir / "species_mic_coverage_by_seed.csv")
    comparison = pd.read_csv(table_dir / "species_conformal_comparison.csv")
    metrics = pd.read_csv(species_dir / "metrics.csv")
    species = pd.read_csv(table_dir / "supplementary_table_s1_species.csv")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.45))
    headline(fig, "Near-nominal empirical coverage requires wide prediction intervals")

    handles = []
    for _, row in coverage.iterrows():
        model = row["model"]
        seeds = seed_coverage.loc[seed_coverage["model"].eq(model)]
        axes[0, 0].scatter(seeds["mean_interval_width"], seeds["pooled_marginal_coverage"], s=17,
                           facecolors="none", edgecolors=MODEL_COLORS[model], linewidth=0.7, zorder=3)
        axes[0, 0].errorbar(row["interval_width_mean"], row["pooled_coverage_mean"],
                            xerr=seeds["mean_interval_width"].std(), yerr=row["pooled_coverage_sd"],
                            fmt="o", ms=5.5, color=MODEL_COLORS[model], capsize=2, lw=0.8, zorder=4)
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[model],
                              markeredgecolor="white", label=MODEL_LABELS[model], markersize=6))
    axes[0, 0].axhline(0.90, color=TEXT, linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("Mean interval width (log2 MIC)")
    axes[0, 0].set_ylabel("Pair-weighted empirical coverage")
    axes[0, 0].set_ylim(0.85, 0.95)
    axes[0, 0].set_title("Coverage-width trade-off", loc="left")
    axes[0, 0].legend(handles=handles, frameon=False, loc="lower left", ncol=2, fontsize=6.2)
    clean_axis(axes[0, 0], "both")
    panel_label(axes[0, 0], "A")

    mv = comparison.loc[comparison["model"].eq("multiview")].set_index("method")
    metrics_names = ["pooled_coverage", "macro_species_coverage", "minimum_species_coverage"]
    metric_labels = ["Pooled", "Equal-species", "Minimum species"]
    x = np.arange(3)
    width = 0.34
    axes[0, 1].plot(x - width / 2, [mv.loc["pooled", m] for m in metrics_names], "o",
                    color=MODEL_COLORS["multiview"], ms=7, label="Pooled calibration")
    axes[0, 1].plot(x + width / 2, [mv.loc["species_wise", m] for m in metrics_names], "s",
                    color="#F4A300", ms=7, label="Species-wise calibration")
    axes[0, 1].axhline(0.90, color=TEXT, linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(x, metric_labels, rotation=18, ha="right")
    axes[0, 1].set_ylim(0.65, 0.93)
    axes[0, 1].set_ylabel("Coverage")
    axes[0, 1].set_title("Naive species-wise calibration lowers tail coverage", loc="left")
    axes[0, 1].legend(frameon=False, loc="lower left")
    clean_axis(axes[0, 1], "y")
    panel_label(axes[0, 1], "B")

    per = metrics.loc[(metrics["scope"].eq("species")) & (metrics["model"].eq("multiview"))]
    per_species = per.groupby("organism", as_index=False).agg(
        pooled_coverage=("conformal_coverage", "mean"),
        specieswise_coverage=("species_conformal_coverage", "mean"),
        pooled_width=("conformal_width", "mean"),
        specieswise_width=("species_conformal_width", "mean"),
    ).merge(species[["organism", "pathogen_class"]], on="organism", how="left")
    axes[1, 0].plot([0.68, 1.0], [0.68, 1.0], color="#9AA6B2", linestyle="--", linewidth=1)
    axes[1, 0].scatter(per_species["pooled_coverage"], per_species["specieswise_coverage"], s=28,
                       color=[CLASS_COLORS[c] for c in per_species["pathogen_class"]],
                       edgecolor="white", linewidth=0.5)
    axes[1, 0].set_xlim(0.68, 1.0)
    axes[1, 0].set_ylim(0.68, 1.0)
    axes[1, 0].set_xlabel("Pooled-calibration species coverage")
    axes[1, 0].set_ylabel("Species-wise-calibration coverage")
    worse = int((per_species["specieswise_coverage"] < per_species["pooled_coverage"]).sum())
    axes[1, 0].set_title(f"Species-wise coverage decreases for {worse}/48 organisms", loc="left")
    clean_axis(axes[1, 0], "both")
    panel_label(axes[1, 0], "C")

    width_values = [per_species["pooled_width"].to_numpy(), per_species["specieswise_width"].to_numpy()]
    boxes = axes[1, 1].boxplot(width_values, patch_artist=True, widths=0.55, showfliers=False,
                               medianprops={"color": TEXT, "linewidth": 1.2})
    for patch, color in zip(boxes["boxes"], [MODEL_COLORS["multiview"], "#F4A300"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(20260820)
    for i, (vals, color) in enumerate(zip(width_values, [MODEL_COLORS["multiview"], "#F4A300"]), start=1):
        axes[1, 1].scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals, s=15, color=color, alpha=0.65,
                           edgecolor="white", linewidth=0.25)
    axes[1, 1].set_xticks([1, 2], ["Pooled", "Species-wise"])
    axes[1, 1].set_ylabel("Mean interval width (log2 MIC)")
    axes[1, 1].set_title(f"Equal-species mean width: {mv.loc['pooled', 'macro_species_width']:.3f} to {mv.loc['species_wise', 'macro_species_width']:.3f}", loc="left")
    clean_axis(axes[1, 1], "y")
    panel_label(axes[1, 1], "D")

    source(coverage, source_dir, "fig6_coverage_width_summary.csv")
    source(seed_coverage, source_dir, "fig6_coverage_width_seeds.csv")
    source(comparison.loc[comparison["model"].eq("multiview")], source_dir, "fig6_multiview_conformal_methods.csv")
    source(per_species, source_dir, "fig6_multiview_species_coverage.csv")
    finish_grid(fig)
    save(fig, output_dir, "fig6_conformal_uncertainty")


def figure7(table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    metrics = pd.read_csv(table_dir / "maier_external_metrics.csv")
    individual_ci = pd.read_csv(table_dir / "maier_individual_bootstrap_ci.csv")
    paired = pd.read_csv(table_dir / "maier_paired_bootstrap_differences.csv")
    topk = pd.read_csv(table_dir / "maier_topk_enrichment.csv")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.45))
    headline(fig, "Multi-view transfers to broad Maier activity and enriches top-ranked compounds under overlap audits")

    models = ["morgan", "multiview", "mole", "molformer"]
    primary = metrics.loc[metrics["cohort"].eq("primary_exact_key_nonoverlap")].set_index("model")
    x = np.arange(2)
    width = 0.18
    for i, model in enumerate(models):
        estimates = [primary.loc[model, "auroc"], primary.loc[model, "auprc"]]
        lows, highs = [], []
        for metric in ["auroc", "auprc"]:
            row = individual_ci.loc[(individual_ci["model"].eq(model)) &
                                    (individual_ci["endpoint"].eq("broad_any_activity")) &
                                    (individual_ci["metric"].eq(metric))].iloc[0]
            lows.append(row["ci_low"])
            highs.append(row["ci_high"])
        yerr = np.vstack([np.array(estimates) - lows, np.array(highs) - estimates])
        axes[0, 0].bar(x + (i - 1.5) * width, estimates, width, color=MODEL_COLORS[model],
                       label=MODEL_LABELS[model], yerr=yerr, capsize=2.5)
    axes[0, 0].set_xticks(x, ["AUROC", "AUPRC"])
    axes[0, 0].set_ylim(0.45, 0.76)
    axes[0, 0].set_ylabel("Performance")
    axes[0, 0].set_title("Primary exact-key-nonoverlap cohort (n=1,015)", loc="left")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    clean_axis(axes[0, 0], "y")
    panel_label(axes[0, 0], "A")

    cohort_map = {
        "primary_exact_key_nonoverlap": "Exact key",
        "posthoc_strict_fingerprint_nonidentity": "Fingerprint",
        "posthoc_strict_parent_nonidentity": "Parent",
    }
    diff = paired.loc[(paired["comparison"].eq("multiview_minus_morgan")) &
                      (paired["endpoint"].eq("any_activity")) &
                      paired["cohort"].isin(cohort_map)].copy()
    y_base = np.arange(3)
    for metric, marker, color, offset in [("auroc", "o", "#0077C8", -0.10), ("auprc", "s", "#E43D30", 0.10)]:
        subset = diff.loc[diff["metric"].eq(metric)].set_index("cohort").loc[list(cohort_map)]
        y = y_base + offset
        axes[0, 1].errorbar(subset["difference"], y,
                            xerr=np.vstack([subset["difference"] - subset["ci_low"], subset["ci_high"] - subset["difference"]]),
                            fmt=marker, color=color, ecolor="#536273", capsize=3, label=metric.upper(), ms=5.5)
    axes[0, 1].axvline(0, color=TEXT, linestyle="--", linewidth=1)
    axes[0, 1].set_yticks(y_base, list(cohort_map.values()))
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Multi-view - Morgan")
    axes[0, 1].set_title("Paired advantage persists after stricter overlap removal", loc="left")
    axes[0, 1].legend(frameon=False, loc="upper left")
    clean_axis(axes[0, 1], "x")
    panel_label(axes[0, 1], "B")

    selected = topk.loc[topk["cohort"].eq("primary_exact_key_nonoverlap") &
                        topk["model"].isin(["morgan", "multiview"])].copy()
    endpoints = {}
    for model in ["morgan", "multiview"]:
        subset = selected.loc[selected["model"].eq(model)].sort_values("k")
        axes[1, 0].plot(subset["k"], subset["precision"], marker="o", markersize=6,
                        color=MODEL_COLORS[model], linewidth=2)
        endpoints[model] = subset.iloc[-1]["precision"]
        for _, row in subset.iterrows():
            offset = 0.03 if model == "multiview" else -0.05
            axes[1, 0].text(row["k"], row["precision"] + offset, f"{int(row['hits'])}", ha="center",
                            fontsize=6.5, color=MODEL_COLORS[model], fontweight="bold")
    prevalence = primary.loc["multiview", "prevalence"]
    axes[1, 0].axhline(prevalence, color="#64748B", linestyle="--", linewidth=1.2)
    axes[1, 0].text(102, endpoints["multiview"], "Multi-view", color=MODEL_COLORS["multiview"], va="center")
    axes[1, 0].text(102, endpoints["morgan"], "Morgan", color=MODEL_COLORS["morgan"], va="center")
    axes[1, 0].text(102, prevalence, "Cohort prevalence", color="#64748B", va="center")
    axes[1, 0].set_xlim(22, 112)
    axes[1, 0].set_xticks([25, 50, 100])
    axes[1, 0].set_ylim(0.30, 1.05)
    axes[1, 0].set_xlabel("Top k compounds")
    axes[1, 0].set_ylabel("Observed active fraction")
    axes[1, 0].set_title("Top-ranked multi-view compounds are enriched for actives", loc="left")
    clean_axis(axes[1, 0], "y")
    panel_label(axes[1, 0], "C")

    cohort_rows = metrics.loc[(metrics["model"].eq("multiview")) & metrics["cohort"].isin(cohort_map)].copy()
    cohort_rows = cohort_rows.set_index("cohort").loc[list(cohort_map)].reset_index()
    labels = [cohort_map[c] for c in cohort_rows["cohort"]]
    inactive = cohort_rows["n"] - cohort_rows["positives"]
    axes[1, 1].bar(labels, inactive, color="#DCE3EA", label="Inactive")
    axes[1, 1].bar(labels, cohort_rows["positives"], bottom=inactive, color=ACCENT, label="Active")
    for i, row in cohort_rows.iterrows():
        axes[1, 1].text(i, row["n"] + 18, f"n={int(row['n'])}\n{int(row['positives'])} active",
                        ha="center", fontsize=6.7, fontweight="bold")
    axes[1, 1].set_ylim(0, 1150)
    axes[1, 1].set_ylabel("Compounds")
    axes[1, 1].set_title("Cohort sizes under overlap audits", loc="left")
    axes[1, 1].legend(frameon=False, loc="lower right")
    clean_axis(axes[1, 1], "y")
    panel_label(axes[1, 1], "D")

    source(primary.reset_index(), source_dir, "fig7_primary_broad_metrics.csv")
    source(diff, source_dir, "fig7_broad_paired_differences.csv")
    source(selected, source_dir, "fig7_primary_topk.csv")
    source(cohort_rows, source_dir, "fig7_overlap_cohorts.csv")
    finish_grid(fig)
    save(fig, output_dir, "fig7_maier_broad_transfer")


def figure8(table_dir: Path, output_dir: Path, source_dir: Path) -> None:
    similarity = pd.read_csv(table_dir / "maier_similarity_metrics.csv")
    joint = pd.read_csv(table_dir / "joint_external_paired_differences.csv")
    joint = joint.loc[joint["cohort"].eq("primary")].copy()
    bins = ["<0.30", "0.30-0.50", "0.50-0.70", ">=0.70"]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    headline(fig, "External transfer depends on chemical context, species and metric")

    composition = similarity.loc[similarity["model"].eq("morgan")].set_index("similarity_bin").loc[bins].reset_index()
    inactive = composition["n"] - composition["positives"]
    x = np.arange(len(bins))
    axes[0, 0].bar(x, inactive, color="#DCE3EA", label="Inactive")
    axes[0, 0].bar(x, composition["positives"], bottom=inactive, color=ACCENT, label="Active")
    for i, row in composition.iterrows():
        axes[0, 0].text(i, row["n"] + 12, f"n={int(row['n'])}\nprev={row['prevalence']:.2f}", ha="center", fontsize=6.5)
    axes[0, 0].set_xticks(x, bins)
    axes[0, 0].set_xlabel("Nearest-training Morgan Tanimoto")
    axes[0, 0].set_ylabel("Compounds")
    axes[0, 0].set_ylim(0, composition["n"].max() * 1.22)
    axes[0, 0].set_title("Activity prevalence rises with structural proximity", loc="left")
    axes[0, 0].legend(frameon=False, loc="upper right")
    clean_axis(axes[0, 0], "y")
    panel_label(axes[0, 0], "A")

    for model in ["morgan", "multiview", "mole", "molformer"]:
        subset = similarity.loc[similarity["model"].eq(model)].set_index("similarity_bin").loc[bins]
        axes[0, 1].plot(x, subset["auprc"], marker="o", linewidth=1.8, markersize=5,
                        color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    axes[0, 1].plot(x, composition["prevalence"], marker="s", linestyle="--", linewidth=1.4,
                    color="#44515F", label="Active prevalence")
    axes[0, 1].set_xticks(x, bins)
    axes[0, 1].set_ylim(0.15, 0.95)
    axes[0, 1].set_xlabel("Nearest-training Morgan Tanimoto")
    axes[0, 1].set_ylabel("AUPRC / prevalence")
    axes[0, 1].set_title("Similarity-stratified AUPRC is descriptive, not causal", loc="left")
    axes[0, 1].legend(frameon=False, ncol=2, loc="upper left")
    clean_axis(axes[0, 1], "y")
    panel_label(axes[0, 1], "B")

    contrasts = ["joint:multiview_minus_joint:morgan", "joint:multiview_minus_joint:morgan_maccs",
                 "joint:multiview_minus_single:multiview"]
    comparators = ["joint Morgan", "joint Morgan + MACCS", "single-species multi-view"]
    for ax, metric, panel in [(axes[1, 0], "auroc", "C"), (axes[1, 1], "ap", "D")]:
        labels, positions = [], []
        for offset, organism, short, color in [
            (0, "Escherichia coli", "E. coli", "#0077C8"),
            (4, "Clostridioides difficile", "C. difficile", "#E43D30"),
        ]:
            subset = joint.loc[joint["organism"].eq(organism) & joint["metric"].eq(metric)].set_index("comparison").loc[contrasts]
            estimate, low, high = subset["difference"], subset["ci_low"], subset["ci_high"]
            y = np.arange(3) + offset
            ax.errorbar(estimate, y, xerr=np.vstack([estimate - low, high - estimate]),
                        fmt="o", ms=5.5, color=color, ecolor=color, capsize=3, lw=1.2)
            labels.extend([f"{short}\nvs {name}" for name in comparators])
            positions.extend(y)
        ax.axvline(0, color=TEXT, linestyle="--", linewidth=1)
        ax.set_yticks(positions, labels, fontsize=6.2)
        ax.set_ylim(-0.65, 6.65)
        ax.invert_yaxis()
        ax.set_xlabel(f"Joint multi-view - comparator {metric.upper()}")
        title = "C. difficile gains are confined to AUROC" if metric == "auroc" else "E. coli AP is lower than single-species training"
        ax.set_title(title, loc="left")
        clean_axis(ax, "x")
        panel_label(ax, panel)

    source(composition, source_dir, "fig8_similarity_composition.csv")
    source(similarity, source_dir, "fig8_similarity_metrics.csv")
    source(joint, source_dir, "fig8_joint_primary_differences.csv")
    finish_grid(fig)
    save(fig, output_dir, "fig8_novelty_ecoli_boundary")


def write_catalog(output_dir: Path) -> None:
    claims = [
        (1, "fig1_dataset_curation", "Taxonomy-audited curation yields 48 accepted species and 214,068 MIC pairs."),
        (2, "fig2_scaffold_integrity", "Five compound-level scaffold splits have zero assigned-scaffold leakage and stable support."),
        (3, "fig3_internal_performance", "Multi-view gives a stable but small average error reduction over Morgan."),
        (4, "fig4_species_conditioning", "Species identity improves average MIC prediction, but its value varies across organisms."),
        (5, "fig5_representation_ablation", "Classical views differ by endpoint under a fixed downstream estimator."),
        (6, "fig6_conformal_uncertainty", "Near-nominal empirical coverage requires wide prediction intervals."),
        (7, "fig7_maier_broad_transfer", "Multi-view transfers to broad Maier activity and enriches top-ranked compounds under overlap audits."),
        (8, "fig8_novelty_ecoli_boundary", "External transfer depends on chemical context, species and metric; joint multi-view is not uniformly superior."),
    ]
    catalog = []
    for number, stem, claim in claims:
        catalog.append(
            {
                "figure": number,
                "stem": stem,
                "surface": "paper_main_double_column",
                "claim": claim,
                "exports": [f"{stem}.png", f"{stem}.pdf", f"{stem}.svg"],
                "source_data_dir": "source_data",
                "self_review_change": "Split one dashboard-style composite into a question-led figure with brighter stable colors and larger labels.",
            }
        )
    (output_dir / "figure_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-dir", required=True)
    parser.add_argument("--conditioning-dir", required=True)
    parser.add_argument("--table-dir", default="paper/tables/species_mic_v2")
    parser.add_argument("--output", default="paper/figures/main_v3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    species_dir = Path(args.species_dir)
    conditioning_dir = Path(args.conditioning_dir)
    table_dir = Path(args.table_dir)
    output_dir = Path(args.output)
    source_dir = output_dir / "source_data"
    style()
    figure1(table_dir, output_dir, source_dir)
    figure2(species_dir, table_dir, output_dir, source_dir)
    figure3(species_dir, table_dir, output_dir, source_dir)
    figure4(conditioning_dir, table_dir, output_dir, source_dir)
    figure5(table_dir, output_dir, source_dir)
    figure6(species_dir, table_dir, output_dir, source_dir)
    figure7(table_dir, output_dir, source_dir)
    figure8(table_dir, output_dir, source_dir)
    write_catalog(output_dir)
