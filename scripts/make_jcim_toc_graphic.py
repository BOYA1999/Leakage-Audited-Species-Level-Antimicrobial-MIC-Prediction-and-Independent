import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image


NAVY = "#172033"
BLUE = "#087FC4"
TEAL = "#08A681"
ORANGE = "#F5A300"
RED = "#E83B2E"
PALE_BLUE = "#E8F4FB"
PALE_TEAL = "#E7F6F1"
PALE_ORANGE = "#FFF3D7"
PALE_RED = "#FDEBE8"
LINE = "#607087"


def box(ax, xy, width, height, face, edge, radius=0.035, lw=1.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def molecule(ax):
    center = (0.125, 0.685)
    radius = 0.060
    angles = [30, 90, 150, 210, 270, 330]
    points = [
        (
            center[0] + radius * __import__("math").cos(__import__("math").radians(a)),
            center[1] + radius * __import__("math").sin(__import__("math").radians(a)),
        )
        for a in angles
    ]
    ax.add_patch(Polygon(points, closed=True, fill=False, edgecolor=BLUE, linewidth=2.0))
    ax.plot([points[0][0], 0.220], [points[0][1], 0.725], color=BLUE, lw=2.0)
    ax.plot([0.220, 0.264], [0.725, 0.685], color=BLUE, lw=2.0)
    ax.plot([0.220, 0.258], [0.725, 0.775], color=TEAL, lw=2.0)
    ax.text(0.275, 0.675, "N", color=TEAL, fontsize=8, fontweight="bold", ha="center")
    ax.text(0.271, 0.794, "O", color=RED, fontsize=8, fontweight="bold", ha="center")
    ax.plot([points[3][0], 0.058], [points[3][1], 0.640], color=BLUE, lw=2.0)
    ax.text(0.040, 0.625, "OH", color=RED, fontsize=8, fontweight="bold", ha="center")
    ax.add_patch(Circle((0.220, 0.725), 0.009, facecolor=ORANGE, edgecolor="white", lw=0.5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the graphical abstract.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="toc_graphic")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(3.25, 1.75), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.025, 0.525), 0.270, 0.365, PALE_BLUE, BLUE, radius=0.035)
    molecule(ax)
    ax.text(0.160, 0.455, "MOLECULE", color=LINE, fontsize=8, fontweight="bold", ha="center")
    box(ax, (0.025, 0.145), 0.270, 0.240, PALE_TEAL, TEAL, radius=0.028, lw=1.0)
    ax.text(0.160, 0.350, "SPECIES", color=LINE, fontsize=8, fontweight="bold", ha="center", va="center")
    ax.text(0.160, 0.275, r"$\it{E.\ coli}$", color=NAVY, fontsize=8, ha="center", va="center")
    ax.text(0.160, 0.195, r"$\it{S.\ aureus}$", color=NAVY, fontsize=8, ha="center", va="center")

    ax.add_patch(FancyArrowPatch((0.305, 0.515), (0.350, 0.515), arrowstyle="-|>", mutation_scale=9, color=LINE, lw=1.2))

    box(ax, (0.355, 0.105), 0.285, 0.785, "#F7F9FC", LINE, radius=0.035)
    ax.text(0.498, 0.820, "AUDIT", color=LINE, fontsize=8, fontweight="bold", ha="center")
    audit_rows = [
        (0.665, "Taxonomy", TEAL, PALE_TEAL),
        (0.475, "Leakage", BLUE, PALE_BLUE),
        (0.285, "Temporal", ORANGE, PALE_ORANGE),
    ]
    for y, label, color, fill in audit_rows:
        box(ax, (0.375, y - 0.070), 0.245, 0.140, fill, color, radius=0.022, lw=1.0)
        ax.add_patch(Circle((0.410, y), 0.016, facecolor=color, edgecolor="none"))
        ax.plot([0.402, 0.408, 0.419], [y, y - 0.008, y + 0.011], color="white", lw=1.2, solid_capstyle="round")
        ax.text(0.448, y, label, color=NAVY, fontsize=8, fontweight="bold", ha="left", va="center")

    ax.add_patch(FancyArrowPatch((0.650, 0.515), (0.695, 0.515), arrowstyle="-|>", mutation_scale=9, color=LINE, lw=1.2))

    ax.text(0.845, 0.875, "BOUNDARY", color=LINE, fontsize=8, fontweight="bold", ha="center")
    cards = [
        (0.700, 0.665, "Internal\nscaffold", PALE_BLUE, BLUE),
        (0.700, 0.475, "Future\ncompounds", PALE_ORANGE, ORANGE),
        (0.700, 0.285, "Known-class\ntransfer", PALE_TEAL, TEAL),
        (0.700, 0.095, "External\nendpoint", PALE_RED, RED),
    ]
    for x, y, label, fill, edge in cards:
        box(ax, (x, y), 0.280, 0.145, fill, edge, radius=0.025, lw=1.0)
        ax.text(x + 0.140, y + 0.0725, label, color=NAVY, fontsize=8, fontweight="bold", ha="center", va="center", linespacing=0.92)

    png = args.output_dir / f"{args.stem}.png"
    tif = args.output_dir / f"{args.stem}.tif"
    svg = args.output_dir / f"{args.stem}.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(tif, dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    with Image.open(png) as image:
        image.convert("RGB").save(png, dpi=(300, 300))
    with Image.open(tif) as image:
        image.convert("RGB").save(tif, dpi=(300, 300), compression="tiff_lzw")


if __name__ == "__main__":
    main()
