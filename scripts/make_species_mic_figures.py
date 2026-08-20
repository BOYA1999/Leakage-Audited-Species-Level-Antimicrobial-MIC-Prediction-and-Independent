from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


COLORS = {
    "species_median": "#8A8A8A",
    "morgan": "#2474B5",
    "multiview": "#D04A35",
    "mole": "#2A9D6F",
    "molformer": "#7B5AA6",
}
LABELS = {
    "species_median": "Species median",
    "morgan": "Morgan",
    "multiview": "Multi-view",
    "mole": "MolE",
    "molformer": "MoLFormer",
}
MODELS = list(LABELS)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=400 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")


def workflow(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1", "ChEMBL 34", "Taxonomy-normalized MIC\n48 species; 214,068 pairs"),
        ("2", "Leakage control", "Five scaffold splits\n0 assigned-scaffold overlap"),
        ("3", "Representations", "Morgan, multi-view\nMolE, MoLFormer"),
        ("4", "Quantitative test", "MAE, Spearman\ndilution accuracy\nconformal intervals"),
        ("5", "External screen", "Maier, 40 strains\n1,015 exact-key\nnon-overlap compounds"),
    ]
    x_positions = np.linspace(0.105, 0.895, len(stages))
    box_colors = ["#E8F1F8", "#F3EDE4", "#E7F3EC", "#F9EBE8", "#EFEAF6"]
    for i, ((number, title, detail), x, color) in enumerate(zip(stages, x_positions, box_colors)):
        box = FancyBboxPatch(
            (x - 0.085, 0.25),
            0.17,
            0.54,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=color,
            edgecolor="#4F4F4F",
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(x, 0.67, number, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x, 0.54, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x, 0.37, detail, ha="center", va="center", fontsize=7.1, linespacing=1.18)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.093, 0.50), xytext=(x + 0.093, 0.50),
                        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0})
    ax.text(0.5, 0.08, "Frozen endpoints and splits; no Maier-label threshold tuning", ha="center", fontsize=8)
    save(fig, output_dir, "v2_fig1_workflow")


def species_performance(species_dir: Path, table_dir: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(species_dir / "metrics.csv")
    macro = metrics.loc[metrics["scope"].eq("macro")]
    per_species = pd.read_csv(table_dir / "species_mic_per_species.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    for ax, metric, title, ylabel in [
        (axes[0, 0], "mae", "Quantitative error", "Macro species MAE (log2 MIC)"),
        (axes[0, 1], "spearman", "Rank correlation", "Macro species Spearman rho"),
    ]:
        data = [macro.loc[macro["model"].eq(model), metric].dropna().to_numpy() for model in MODELS]
        means = [np.mean(values) if len(values) else np.nan for values in data]
        sds = [np.std(values, ddof=1) if len(values) > 1 else 0 for values in data]
        x = np.arange(len(MODELS))
        ax.bar(x, means, yerr=sds, color=[COLORS[m] for m in MODELS], width=0.68, capsize=2, linewidth=0)
        for i, values in enumerate(data):
            if len(values):
                ax.scatter(np.full(len(values), i) + np.linspace(-0.10, 0.10, len(values)), values,
                           s=12, facecolors="white", edgecolors="#333333", linewidths=0.5, zorder=3)
            else:
                ax.text(i, 0.03, "NA", ha="center", va="bottom", fontsize=7, color="#555555")
        ax.set_xticks(x, [LABELS[m] for m in MODELS], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    panel_label(axes[0, 0], "A")
    panel_label(axes[0, 1], "B")

    coverage = pd.read_csv(table_dir / "species_mic_coverage_summary.csv").set_index("model")
    x = np.arange(len(MODELS))
    width = 0.34
    axes[1, 0].bar(
        x - width / 2,
        coverage.loc[MODELS, "pooled_coverage_mean"],
        width,
        color=[COLORS[m] for m in MODELS],
        alpha=0.95,
        label="Pooled",
    )
    axes[1, 0].bar(
        x + width / 2,
        coverage.loc[MODELS, "macro_coverage_mean"],
        width,
        color=[COLORS[m] for m in MODELS],
        alpha=0.45,
        label="Species macro",
    )
    axes[1, 0].axhline(0.90, color="#222222", linestyle="--", linewidth=1, label="Nominal 90%")
    axes[1, 0].set_ylim(0.87, 0.93)
    axes[1, 0].set_xticks(x, [LABELS[m] for m in MODELS], rotation=25, ha="right")
    axes[1, 0].set_ylabel("Coverage of nominal 90% intervals")
    axes[1, 0].set_title("Split-conformal intervals", loc="left")
    axes[1, 0].legend(frameon=False, loc="upper right")
    axes[1, 0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    panel_label(axes[1, 0], "C")

    pivot = per_species.pivot(index="organism", columns="model", values="mae")
    delta = (pivot["multiview"] - pivot["morgan"]).sort_values()
    y = np.arange(len(delta))
    axes[1, 1].barh(y, delta, color=np.where(delta < 0, COLORS["multiview"], "#A6A6A6"), height=0.75)
    axes[1, 1].axvline(0, color="#222222", linewidth=0.8)
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel("Multi-view - Morgan MAE", fontsize=9.9)
    axes[1, 1].set_title(f"Species-level differences ({int((delta < 0).sum())}/{len(delta)} favor multi-view)", loc="left", fontsize=9.9)
    axes[1, 1].grid(axis="x", color="#DDDDDD", linewidth=0.6)
    panel_label(axes[1, 1], "D")

    fig.subplots_adjust(wspace=0.30, hspace=0.48)
    save(fig, output_dir, "v2_fig2_species_mic")


def external_validation(table_dir: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(table_dir / "maier_external_metrics.csv")
    individual_ci = pd.read_csv(table_dir / "maier_individual_bootstrap_ci.csv")
    topk = pd.read_csv(table_dir / "maier_topk_enrichment.csv")
    similarity = pd.read_csv(table_dir / "maier_similarity_metrics.csv")
    paired = pd.read_csv(table_dir / "maier_paired_bootstrap_differences.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    primary = metrics.loc[metrics["cohort"].eq("primary_exact_key_nonoverlap")].set_index("model")
    x = np.arange(2)
    width = 0.18
    for i, model in enumerate(MODELS[1:]):
        estimates = [primary.loc[model, "auroc"], primary.loc[model, "auprc"]]
        lows, highs = [], []
        for metric in ("auroc", "auprc"):
            row = individual_ci.loc[(individual_ci["model"].eq(model)) &
                                    (individual_ci["endpoint"].eq("broad_any_activity")) &
                                    (individual_ci["metric"].eq(metric))].iloc[0]
            lows.append(row["ci_low"])
            highs.append(row["ci_high"])
        yerr = np.vstack([np.array(estimates) - lows, np.array(highs) - estimates])
        axes[0, 0].bar(x + (i - 1.5) * width, estimates, width, color=COLORS[model], label=LABELS[model],
                       yerr=yerr, capsize=2)
    axes[0, 0].set_xticks(x, ["AUROC", "AUPRC"])
    axes[0, 0].set_ylim(0.45, 0.76)
    axes[0, 0].set_ylabel("Performance")
    axes[0, 0].set_title("Independent 40-strain screen", loc="left")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0, 0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    panel_label(axes[0, 0], "A")

    selected = topk.loc[
        topk["cohort"].eq("primary_exact_key_nonoverlap") & topk["model"].isin(["morgan", "multiview"])
    ]
    for model in ("morgan", "multiview"):
        subset = selected.loc[selected["model"].eq(model)].sort_values("k")
        axes[0, 1].plot(subset["k"], subset["precision"], marker="o", color=COLORS[model],
                        label=LABELS[model], linewidth=1.6)
    axes[0, 1].axhline(0.383251, color="#777777", linestyle="--", linewidth=1, label="Prevalence")
    axes[0, 1].set_xticks([25, 50, 100])
    axes[0, 1].set_ylim(0.30, 1.02)
    axes[0, 1].set_xlabel("Top k compounds")
    axes[0, 1].set_ylabel("Observed active fraction")
    axes[0, 1].set_title("Screening enrichment", loc="left")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(color="#DDDDDD", linewidth=0.6)
    panel_label(axes[0, 1], "B")

    bins = ["<0.30", "0.30-0.50", "0.50-0.70", ">=0.70"]
    for model in MODELS[1:]:
        subset = similarity.loc[similarity["model"].eq(model)].set_index("similarity_bin").loc[bins]
        axes[1, 0].plot(np.arange(len(bins)), subset["auprc"], marker="o", color=COLORS[model],
                        label=LABELS[model], linewidth=1.4, markersize=4)
    prevalence = similarity.loc[similarity["model"].eq("morgan")].set_index("similarity_bin").loc[bins]
    axes[1, 0].plot(np.arange(len(bins)), prevalence["prevalence"], marker="s", color="#333333",
                    linestyle="--", linewidth=1.2, markersize=4, label="Active prevalence")
    axes[1, 0].set_xticks(np.arange(len(bins)), bins, fontsize=8.8)
    axes[1, 0].set_xlabel("Nearest-training Morgan Tanimoto", fontsize=9.9)
    axes[1, 0].set_ylabel("AUPRC / active prevalence", fontsize=9.9)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_title("Performance and prevalence by proximity", loc="left", fontsize=9.9)
    axes[1, 0].legend(handles=[axes[1, 0].lines[-1]], frameon=False, loc="upper left", fontsize=8.25)
    axes[1, 0].grid(color="#DDDDDD", linewidth=0.6)
    panel_label(axes[1, 0], "C")

    delta = paired.loc[(paired["comparison"].eq("multiview_minus_morgan")) & paired["metric"].eq("auprc")].copy()
    delta["label"] = delta["endpoint"].map({"any_activity": "Broad", "ecoli_any_activity": "E. coli"})
    delta["label"] += delta["cohort"].map({
        "primary_exact_key_nonoverlap": "\nprimary",
        "posthoc_strict_fingerprint_nonidentity": "\nstrict sensitivity",
        "posthoc_ecoli_strict_fingerprint_nonidentity": "\nstrict sensitivity",
    })
    order = [
        "Broad\nprimary",
        "Broad\nstrict sensitivity",
        "E. coli\nprimary",
        "E. coli\nstrict sensitivity",
    ]
    delta = delta.set_index("label").loc[order]
    y = np.arange(len(delta))
    axes[1, 1].errorbar(delta["difference"], y,
                        xerr=np.vstack([delta["difference"] - delta["ci_low"], delta["ci_high"] - delta["difference"]]),
                        fmt="o", color=COLORS["multiview"], ecolor="#333333", capsize=3)
    axes[1, 1].axvline(0, color="#222222", linestyle="--", linewidth=1)
    axes[1, 1].set_yticks(y, order, fontsize=8.25)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("AUPRC difference: multi-view - Morgan", fontsize=9.9)
    axes[1, 1].set_title("Paired bootstrap comparison", loc="left", fontsize=9.9)
    axes[1, 1].grid(axis="x", color="#DDDDDD", linewidth=0.6)
    panel_label(axes[1, 1], "D")

    fig.subplots_adjust(wspace=0.43, hspace=0.42)
    save(fig, output_dir, "v2_fig3_maier_external")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-dir", required=True)
    parser.add_argument("--table-dir", default="paper/tables/species_mic_v2")
    parser.add_argument("--output", default="paper/figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    workflow(output)
    species_performance(Path(args.species_dir), Path(args.table_dir), output)
    external_validation(Path(args.table_dir), output)
