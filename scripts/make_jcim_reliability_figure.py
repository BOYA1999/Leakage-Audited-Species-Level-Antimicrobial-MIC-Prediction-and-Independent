import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image


TEXT = "#172033"
MUTED = "#607087"
GRID = "#DDE3EA"
BLUE = "#0077C8"
TEAL = "#00A676"
ORANGE = "#F4A300"
RED = "#E43D30"
PURPLE = "#7A4CC2"
PALE_BLUE = "#E8F4FB"
PALE_TEAL = "#E6F6F1"
PALE_ORANGE = "#FFF2D3"
PALE_RED = "#FDE9E6"
PALE_PURPLE = "#EFE8FA"
PALE_GRAY = "#F4F6F8"


def setup(ax, label, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.text(0.075, 1.04, title, transform=ax.transAxes, fontsize=9.1, fontweight="bold", va="top")


def box(ax, x, y, w, h, face, edge, radius=0.025, lw=1.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def check(ax, x, y, color=TEAL, cross=False):
    ax.add_patch(Circle((x, y), 0.022, facecolor=color, edgecolor="none"))
    if cross:
        ax.plot([x - 0.008, x + 0.008], [y - 0.008, y + 0.008], color="white", lw=1.4, solid_capstyle="round")
        ax.plot([x - 0.008, x + 0.008], [y + 0.008, y - 0.008], color="white", lw=1.4, solid_capstyle="round")
    else:
        ax.plot([x - 0.010, x - 0.002, x + 0.012], [y, y - 0.009, y + 0.012], color="white", lw=1.4, solid_capstyle="round")


def panel_a(ax):
    setup(ax, "A", "The question is reliability, not a winning architecture")
    satellites = [
        (0.04, 0.67, "Molecular\nrepresentation", PALE_BLUE, BLUE),
        (0.68, 0.67, "Species\ncontext", PALE_TEAL, TEAL),
        (0.36, 0.08, "Uncertainty\nestimate", PALE_ORANGE, ORANGE),
    ]
    for x, y, text, face, edge in satellites:
        box(ax, x, y, 0.28, 0.20, face, edge)
        ax.text(x + 0.14, y + 0.10, text, ha="center", va="center", fontsize=7.8, fontweight="bold")
    box(ax, 0.285, 0.33, 0.43, 0.26, PALE_PURPLE, PURPLE, radius=0.04, lw=1.5)
    ax.text(0.50, 0.46, "When is species-level\nMIC prediction reliable?", ha="center", va="center", fontsize=9.2, fontweight="bold")
    arrow(ax, (0.28, 0.69), (0.36, 0.57), BLUE)
    arrow(ax, (0.72, 0.69), (0.64, 0.57), TEAL)
    arrow(ax, (0.50, 0.29), (0.50, 0.35), ORANGE)


def panel_b(ax):
    setup(ax, "B", "Curation and leakage controls define the benchmark")
    top = [
        (0.02, "ChEMBL 34\nexact MIC\n442,271 rows", PALE_BLUE, BLUE),
        (0.355, "Taxonomy +\nstructure audit", PALE_TEAL, TEAL),
        (0.690, "Median MIC pairs\n214,068\n48 species", PALE_PURPLE, PURPLE),
    ]
    for x, text, face, edge in top:
        box(ax, x, 0.60, 0.29, 0.25, face, edge)
        ax.text(x + 0.145, 0.725, text, ha="center", va="center", fontsize=7.5, fontweight="bold", linespacing=1.15)
    arrow(ax, (0.315, 0.725), (0.350, 0.725))
    arrow(ax, (0.650, 0.725), (0.685, 0.725))
    controls = [
        (0.02, "COMPOUND", "One compound\none partition", PALE_BLUE, BLUE),
        (0.355, "SCAFFOLD", "Assigned-scaffold\noverlap = 0", PALE_ORANGE, ORANGE),
        (0.690, "TIME", "41 eligible species\n40 with test support", PALE_TEAL, TEAL),
    ]
    for x, heading, text, face, edge in controls:
        box(ax, x, 0.18, 0.29, 0.27, face, edge)
        ax.text(x + 0.145, 0.395, heading, ha="center", va="center", fontsize=7.2, color=edge, fontweight="bold")
        ax.text(x + 0.145, 0.285, text, ha="center", va="center", fontsize=7.4, fontweight="bold", linespacing=1.12)
    ax.text(0.50, 0.10, "Time uses ChEMBL document year, not discovery date", ha="center", va="center", fontsize=7.0, color=MUTED)


def panel_c(ax):
    setup(ax, "C", "Model ladder broadens comparison without implying rank")
    ax.text(0.50, 0.84, "Comparison scope", ha="center", va="center", fontsize=7.4, color=MUTED, fontweight="bold")
    arrow(ax, (0.13, 0.785), (0.87, 0.785), MUTED, lw=1.0)
    items = [
        (0.02, "Species\nmedian", PALE_GRAY, MUTED),
        (0.27, "Fixed features\nRF · XGB\nLightGBM", PALE_BLUE, BLUE),
        (0.52, "Frozen\nembeddings\nMolE\nMoLFormer", PALE_PURPLE, PURPLE),
        (0.77, "48-task\nD-MPNN", PALE_RED, RED),
    ]
    for i, (x, text, face, edge) in enumerate(items):
        box(ax, x, 0.28, 0.205, 0.36, face, edge, radius=0.03)
        ax.text(x + 0.1025, 0.46, text, ha="center", va="center", fontsize=7.1, fontweight="bold", linespacing=1.10)
    ax.text(0.50, 0.15, "Comparison set—not a performance ranking", ha="center", va="center", fontsize=7.5, color=MUTED, fontweight="bold")


def panel_d(ax):
    setup(ax, "D", "Tested boundaries vs unestablished validity")
    box(ax, 0.02, 0.14, 0.55, 0.70, PALE_TEAL, TEAL, radius=0.035)
    box(ax, 0.61, 0.14, 0.37, 0.70, PALE_RED, RED, radius=0.035)
    ax.text(0.295, 0.78, "TESTED", ha="center", va="center", fontsize=8.2, color=TEAL, fontweight="bold")
    ax.text(0.795, 0.78, "NOT ESTABLISHED", ha="center", va="center", fontsize=8.2, color=RED, fontweight="bold")
    tested = [
        (0.67, "Internal scaffold split"),
        (0.54, "Retrospective time split"),
        (0.40, "LOSO compound transfer\nwithin known classes"),
        (0.25, "External phenotypic ranking\n(broad + 2 matched species)"),
    ]
    for y, text in tested:
        check(ax, 0.075, y, TEAL)
        ax.text(0.115, y, text, ha="left", va="center", fontsize=7.2, fontweight="bold", linespacing=1.08)
    not_established = [
        (0.64, "Zero-shot species\nrepresentation"),
        (0.46, "Prospective\nquantitative MIC\nvalidation"),
        (0.26, "Clinical\nvalidation"),
    ]
    for y, text in not_established:
        check(ax, 0.665, y, RED, cross=True)
        ax.text(0.705, y, text, ha="left", va="center", fontsize=7.0, fontweight="bold", linespacing=1.08)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the reliability-audit framework figure.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="fig1_reliability_audit")
    return parser.parse_args()


def main():
    args = parse_args()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "text.color": TEXT,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": None,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4))
    fig.suptitle(
        "A reliability audit maps where antimicrobial MIC models are—and are not—evaluated",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
        color=TEXT,
    )
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.045, top=0.905, hspace=0.31, wspace=0.12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.stem
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("RGB").save(stem.with_suffix(".png"), dpi=(600, 600))


if __name__ == "__main__":
    main()
