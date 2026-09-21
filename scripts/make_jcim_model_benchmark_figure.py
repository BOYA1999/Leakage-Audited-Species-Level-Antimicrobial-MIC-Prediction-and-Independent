from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


TEXT = "#172033"
GRAY = "#6B7280"
LIGHT = "#D9DEE7"
GRID = "#E7E5E4"
COLORS = {
    "Species median": "#8A8F98",
    "Morgan": "#0072B2",
    "Multi-view": "#D55E00",
    "Chemprop": "#009E73",
    "MolE": "#CCB000",
    "MoLFormer": "#6F5BA7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the model benchmark and cost figure.")
    parser.add_argument("--model-summary", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--complexity", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default="fig2_model_benchmark")
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D8D1C7",
            "axes.labelcolor": TEXT,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.7,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 10.7,
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.65,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "xtick.labelsize": 10.7,
            "ytick.labelsize": 10.7,
            "legend.frameon": False,
            "legend.fontsize": 10.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean(ax: plt.Axes, axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if axis:
        ax.grid(axis=axis, zorder=0)


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(title, loc="left", pad=8)
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top")


def color_for(pipeline: str) -> str:
    for key in ("Species median", "Multi-view", "Morgan", "Chemprop", "MoLFormer", "MolE"):
        if key in pipeline:
            return COLORS[key]
    return GRAY


def short_name(pipeline: str) -> str:
    replacements = {
        "Species median": "Species median",
        "Morgan + LightGBM": "Morgan · LGBM",
        "Multi-view + LightGBM": "Multi-view · LGBM",
        "Morgan + Random Forest": "Morgan · RF100",
        "Multi-view + Random Forest": "Multi-view · RF100",
        "Morgan + XGBoost": "Morgan · XGB",
        "Multi-view + XGBoost": "Multi-view · XGB",
        "Chemprop D-MPNN": "Chemprop D-MPNN",
        "MolE + LightGBM": "MolE · LGBM",
        "MoLFormer + LightGBM": "MoLFormer · LGBM",
    }
    return replacements.get(pipeline, pipeline)


def cost_short_name(pipeline: str) -> str:
    replacements = {
        "Morgan + LightGBM": "M-LGBM",
        "Multi-view + LightGBM": "MV-LGBM",
        "Morgan + Random Forest": "M-RF100",
        "Multi-view + Random Forest": "MV-RF100",
        "Morgan + XGBoost": "M-XGB",
        "Multi-view + XGBoost": "MV-XGB",
        "Chemprop D-MPNN": "D-MPNN",
    }
    return replacements.get(pipeline, pipeline)


def check_inputs(summary: pd.DataFrame, paired: pd.DataFrame, complexity: pd.DataFrame) -> None:
    required_summary = {"pipeline", "seeds", "mae_mean", "mae_sd"}
    required_paired = {"candidate", "reference", "metric", "mean", "low", "high", "n"}
    required_complexity = {
        "pipeline",
        "fit_seconds_mean",
        "serialized_object_bytes_mean",
        "serialization_measure",
        "estimator_budget",
        "device",
        "mae_mean",
    }
    for name, frame, required in (
        ("model summary", summary, required_summary),
        ("paired differences", paired, required_paired),
        ("complexity", complexity, required_complexity),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if summary["pipeline"].duplicated().any():
        raise ValueError("model summary contains duplicate pipelines")
    expected_pipelines = {
        "Species median",
        "Morgan + Random Forest",
        "Morgan + XGBoost",
        "Morgan + LightGBM",
        "Multi-view + Random Forest",
        "Multi-view + XGBoost",
        "Multi-view + LightGBM",
        "Chemprop D-MPNN",
        "MolE + LightGBM",
        "MoLFormer + LightGBM",
    }
    if set(summary["pipeline"]) != expected_pipelines:
        raise ValueError("model summary does not contain the exact frozen pipeline set")
    if summary["seeds"].astype(int).ne(5).any():
        raise ValueError("all performance pipelines must contain five split values")
    if not np.isfinite(summary[["mae_mean", "mae_sd"]].to_numpy(dtype=float)).all():
        raise ValueError("model summary MAE values must be finite")
    if paired.duplicated(["candidate", "reference", "metric"]).any():
        raise ValueError("paired table contains duplicate contrast-metric rows")
    expected_contrasts = {
        ("Multi-view + LightGBM", "Morgan + LightGBM"),
        ("Multi-view + Random Forest", "Morgan + Random Forest"),
        ("Multi-view + XGBoost", "Morgan + XGBoost"),
        ("Chemprop D-MPNN", "Morgan + LightGBM"),
        ("Chemprop D-MPNN", "Multi-view + LightGBM"),
        ("MolE + LightGBM", "Morgan + LightGBM"),
        ("MoLFormer + LightGBM", "Morgan + LightGBM"),
    }
    expected_metrics = {"mae", "rmse", "spearman", "within_one_dilution", "within_two_dilutions"}
    expected_rows = {(candidate, reference, metric) for candidate, reference in expected_contrasts for metric in expected_metrics}
    observed_rows = set(zip(paired["candidate"], paired["reference"], paired["metric"]))
    if observed_rows != expected_rows:
        raise ValueError("paired table does not contain the exact frozen contrast-metric set")
    if paired["n"].astype(int).ne(5).any():
        raise ValueError("all paired rows must contain five split values")
    if not np.isfinite(paired[["mean", "low", "high"]].to_numpy(dtype=float)).all():
        raise ValueError("paired estimates must be finite")
    if not ((paired["low"] <= paired["mean"]) & (paired["mean"] <= paired["high"])).all():
        raise ValueError("paired stability intervals do not contain their point estimates")
    expected_complexity = expected_pipelines.difference({"Species median"})
    if complexity["pipeline"].duplicated().any() or set(complexity["pipeline"]) != expected_complexity:
        raise ValueError("complexity table does not contain the exact non-naive pipeline set")
    if complexity[["serialization_measure", "estimator_budget", "device"]].isna().any().any():
        raise ValueError("complexity provenance labels must be complete")
    merged = complexity[["pipeline", "mae_mean"]].merge(
        summary[["pipeline", "mae_mean"]], on="pipeline", suffixes=("_complexity", "_summary"), validate="one_to_one"
    )
    if not np.allclose(merged["mae_mean_complexity"], merged["mae_mean_summary"], rtol=0, atol=1e-12):
        raise ValueError("complexity and performance tables disagree on MAE")
    cost_pipelines = expected_complexity.difference({"MolE + LightGBM", "MoLFormer + LightGBM"})
    cost = complexity.loc[complexity["pipeline"].isin(cost_pipelines)]
    if not np.isfinite(cost[["fit_seconds_mean", "serialized_object_bytes_mean"]].to_numpy(dtype=float)).all():
        raise ValueError("recorded cost rows must be finite")
    if not cost[["fit_seconds_mean", "serialized_object_bytes_mean"]].gt(0).all().all():
        raise ValueError("recorded cost values must be positive")


def performance_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    order = [
        "Species median",
        "Morgan + Random Forest",
        "Morgan + XGBoost",
        "Morgan + LightGBM",
        "Multi-view + Random Forest",
        "Multi-view + XGBoost",
        "Multi-view + LightGBM",
        "Chemprop D-MPNN",
        "MolE + LightGBM",
        "MoLFormer + LightGBM",
    ]
    block = summary.set_index("pipeline").loc[order].reset_index()
    y = np.arange(len(block))[::-1]
    for yi, row in zip(y, block.itertuples(index=False)):
        ax.errorbar(
            row.mae_mean,
            yi,
            xerr=row.mae_sd,
            fmt="o",
            color=color_for(row.pipeline),
            ecolor=color_for(row.pipeline),
            markersize=5.1,
            linewidth=1.1,
            capsize=2.2,
            zorder=3,
        )
        ax.text(row.mae_mean + row.mae_sd + 0.012, yi, f"{row.mae_mean:.3f}", va="center", fontsize=10.7)
    ax.set_yticks(y, [short_name(value) for value in block["pipeline"]])
    ax.set_ylim(-0.5, len(block) - 0.15)
    low = float((block["mae_mean"] - block["mae_sd"]).min())
    high = float((block["mae_mean"] + block["mae_sd"]).max())
    span = max(high - low, 0.1)
    ax.set_xlim(low - 0.04 * span, high + 0.18 * span)
    ax.set_xlabel("Equal-species MAE of log₂ MIC (µg mL⁻¹)\n(mean ± SD across five scaffold splits) ↓")
    clean(ax, "x")
    panel(ax, "A", "Performance across fixed pipelines")


def paired_panel(ax: plt.Axes, paired: pd.DataFrame) -> None:
    contrasts = [
        ("Multi-view + LightGBM", "Morgan + LightGBM", "MV − Morgan (LGBM)"),
        ("Multi-view + Random Forest", "Morgan + Random Forest", "MV − Morgan (RF100)"),
        ("Multi-view + XGBoost", "Morgan + XGBoost", "MV − Morgan (XGB)"),
        ("Chemprop D-MPNN", "Morgan + LightGBM", "D-MPNN − M-LGBM"),
        ("Chemprop D-MPNN", "Multi-view + LightGBM", "D-MPNN − MV-LGBM"),
        ("MolE + LightGBM", "Morgan + LightGBM", "MolE − M-LGBM"),
        ("MoLFormer + LightGBM", "Morgan + LightGBM", "MoLFormer − M-LGBM"),
    ]
    mae = paired.loc[paired["metric"].eq("mae")].set_index(["candidate", "reference"])
    rows = []
    for candidate, reference, label in contrasts:
        row = mae.loc[(candidate, reference)]
        rows.append((candidate, label, float(row["mean"]), float(row["low"]), float(row["high"]), int(row["n"])))
    y = np.arange(len(rows))[::-1]
    for yi, (candidate, _, mean, low, high, n) in zip(y, rows):
        color = color_for(candidate)
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5.1,
            linewidth=1.2,
            capsize=2.5,
            zorder=3,
        )
        ax.text(high + 0.008, yi, f"{mean:+.3f}", va="center", fontsize=10.7)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1.0)
    ax.set_yticks(y, [row[1] for row in rows])
    ax.set_ylim(-0.5, len(rows) - 0.15)
    low_bound = min(0.0, min(row[3] for row in rows))
    high_bound = max(0.0, max(row[4] for row in rows))
    span = max(high_bound - low_bound, 0.05)
    ax.set_xlim(low_bound - 0.10 * span, high_bound + 0.32 * span)
    ax.set_xlabel(
        "Candidate − reference ΔMAE of log₂ MIC (µg mL⁻¹)\n"
        "95% five-split stability interval; negative favors candidate"
    )
    clean(ax, "x")
    panel(ax, "B", "Paired representation differences")


def cost_panel(ax: plt.Axes, complexity: pd.DataFrame, summary: pd.DataFrame, column: str, title: str, xlabel: str) -> None:
    sd = summary.set_index("pipeline")["mae_sd"]
    block = complexity.loc[pd.to_numeric(complexity[column], errors="coerce").notna()].copy()
    block[column] = pd.to_numeric(block[column])
    block = block.loc[block[column].gt(0)]
    time_offsets = {
        "Morgan + Random Forest": (7, -16),
        "Multi-view + Random Forest": (7, 9),
        "Chemprop D-MPNN": (7, 10),
    }
    size_offsets = {
        "Morgan + Random Forest": (-8, -17),
        "Multi-view + Random Forest": (-8, 9),
        "Chemprop D-MPNN": (7, -28),
    }
    offsets = time_offsets if column == "fit_seconds_mean" else size_offsets
    for row in block.itertuples(index=False):
        x = float(getattr(row, column))
        y = float(row.mae_mean)
        marker = "^" if "cuda" in str(row.device).lower() or "gpu" in str(row.device).lower() else "o"
        color = color_for(row.pipeline)
        ax.errorbar(
            x,
            y,
            yerr=float(sd[row.pipeline]),
            fmt=marker,
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.8,
            linewidth=1.0,
            capsize=2.0,
            zorder=3,
        )
        if row.pipeline in offsets:
            ax.annotate(
                cost_short_name(row.pipeline),
                (x, y),
                xytext=offsets[row.pipeline],
                textcoords="offset points",
                fontsize=10.7,
                color=TEXT,
                ha="right" if column == "serialized_object_bytes_mean" and "Random Forest" in row.pipeline else "left",
            )
    ax.set_xscale("log")
    ax.set_xlim(float(block[column].min()) / 1.6, float(block[column].max()) * 2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Equal-species MAE\nlog₂(MIC [µg mL⁻¹])" if column == "fit_seconds_mean" else "")
    y_low = float((block["mae_mean"] - block["pipeline"].map(sd)).min())
    y_high = float((block["mae_mean"] + block["pipeline"].map(sd)).max())
    y_pad = max(0.03, 0.16 * (y_high - y_low))
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    clean(ax, "both")
    panel(ax, title[0], title[1:])


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.model_summary)
    paired = pd.read_csv(args.paired)
    complexity = pd.read_csv(args.complexity)
    check_inputs(summary, paired, complexity)

    set_style()
    fig = plt.figure(figsize=(8.4, 10.0))
    gs = fig.add_gridspec(3, 2, left=0.23, right=0.97, bottom=0.17, top=0.92, hspace=0.88, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])
    fig.suptitle(
        "Representation benefits depend on the estimator in this MIC benchmark",
        x=0.03,
        y=0.978,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
    )

    performance_panel(ax_a, summary)
    paired_panel(ax_b, paired)
    cost_panel(
        ax_c,
        complexity,
        summary,
        "fit_seconds_mean",
        "CFit time vs MAE",
        "Mean fit wall time per split (s; log scale)",
    )
    cost_panel(
        ax_d,
        complexity,
        summary,
        "serialized_object_bytes_mean",
        "DSerialized bytes vs MAE",
        "Mean serialized bytes (log scale)",
    )
    fig.text(
        0.03,
        0.025,
        "Colors: blue Morgan; orange multi-view; green D-MPNN. Markers: circle CPU; triangle CUDA.\n"
        "Boundaries: fixed pipelines were not compute-matched. RF used all CPU workers;\n"
        "LGBM/XGB used four workers (XGB on CUDA); Chemprop used CUDA.\n"
        "Tree times exclude shared feature construction; Chemprop uses command-level time.\n"
        "Bytes are temporary pickle lengths versus best.pt; MolE/MoLFormer costs were not uniformly recorded.\n"
        "C–D are implementation audits, not framework-neutral rankings.",
        fontsize=10.7,
        color=GRAY,
        va="bottom",
        linespacing=1.25,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source_dir / "fig2_model_benchmark_summary.csv", index=False)
    paired.loc[paired["metric"].eq("mae")].to_csv(source_dir / "fig2_model_paired_mae.csv", index=False)
    complexity.to_csv(source_dir / "fig2_complexity_summary.csv", index=False)

    png = output_dir / f"{args.stem}.png"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(output_dir / f"{args.stem}.pdf", facecolor="white")
    fig.savefig(output_dir / f"{args.stem}.svg", facecolor="white")
    plt.close(fig)
    with Image.open(png) as image:
        image.convert("RGB").save(png, dpi=(600, 600))


if __name__ == "__main__":
    main()
