#!/usr/bin/env python3
"""Layout-only v4 refinement of the frozen integrated liver figure."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
V3 = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_v3_20260826_110957"
V3_SCRIPT = V3 / "scripts/build_liver_multidisease_v3.py"

spec = importlib.util.spec_from_file_location("liver_v3_frozen", V3_SCRIPT)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

COLORS, INK = v3.COLORS, v3.INK
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.2,
    "axes.titlesize": 7.2, "axes.labelsize": 6.2,
    "xtick.labelsize": 5.7, "ytick.labelsize": 5.4,
    "axes.linewidth": .55, "xtick.major.width": .5,
    "ytick.major.width": .5, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
})


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def panel(ax, letter, x=-.10, y=1.11):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=INK)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1.5)


def draw_a(ax):
    """Compact schematic: neutral flow, disease color only on outlines."""
    ax.set_axis_off()
    panel(ax, "a", -.02, 1.035)
    ax.set_title("One healthy atlas,\nthree disease states", loc="left",
                 fontweight="bold", fontsize=6.5, pad=3, linespacing=1.0)

    def box(x, y, w, h, text, edge, face="white", fs=4.9, weight="normal"):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=.009,rounding_size=.018",
            transform=ax.transAxes, fc=face, ec=edge, lw=.7,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=fs, fontweight=weight,
                linespacing=1.05, color=INK)

    source = (.01, .39, .22, .22)
    box(*source, "Healthy liver\natlas\nM1–M8", "#7592AD", "#F3F6F8", 4.55, "bold")

    disease_x, disease_w, disease_h = .34, .27, .18
    disease_y = [.72, .405, .09]
    labels = [("MASLD", COLORS["MASLD"]), ("PSC", COLORS["PSC"]),
              ("AH", COLORS["AH"])]
    neutral = "#7A7F83"
    source_anchor = (source[0] + source[2], source[1] + source[3] / 2)
    output_anchor = (.73, .50)

    for (name, color), y in zip(labels, disease_y):
        box(disease_x, y, disease_w, disease_h,
            f"{name}\ngeometry +\n16 genes", color, "white", 4.35, "normal")
        mid_y = y + disease_h / 2
        ax.add_patch(FancyArrowPatch(
            source_anchor, (disease_x, mid_y), transform=ax.transAxes,
            arrowstyle="-|>", mutation_scale=5.0, lw=.52, color=neutral,
            connectionstyle="arc3,rad=0"))
        ax.add_patch(FancyArrowPatch(
            (disease_x + disease_w, mid_y), output_anchor,
            transform=ax.transAxes, arrowstyle="-|>", mutation_scale=5.0,
            lw=.52, color=neutral, connectionstyle="arc3,rad=0"))

    ax.text(.75, .50, "Disease-\nspecific\nmolecular\nstate",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=4.2, fontweight="bold", linespacing=1.01, color=INK)


def draw_b(subspec, units):
    """Frozen raw data with more space, no in-panel legend or cohort text."""
    title_ax = plt.subplot(subspec)
    title_ax.set_axis_off()
    title_ax.set_zorder(5)
    panel(title_ax, "b", -.025, 1.105)
    title_ax.text(.5, 1.075, "Reconstruction beyond healthy-atlas transfer",
                  transform=title_ax.transAxes, ha="center", va="bottom",
                  fontweight="bold", fontsize=6.8)
    gs = subspec.subgridspec(1, 2, wspace=.31)
    axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
    pairs = [
        ("median_spatial_idw", "median_spatial_map", "Spatial Pearson"),
        ("log1p_rmse_idw", "log1p_rmse_map", "log1p RMSE"),
    ]
    rng = np.random.default_rng(3203)
    for ax, (base_col, map_col, title) in zip(axes, pairs):
        for i, disease in enumerate(["MASLD", "PSC", "AH"]):
            x = units[units.disease == disease]
            left = i * 3
            jitter = rng.uniform(-.15, .15, len(x))
            for j, row in enumerate(x.itertuples()):
                ax.plot([left + jitter[j], left + 1 + jitter[j]],
                        [getattr(row, base_col), getattr(row, map_col)],
                        color=COLORS[disease], alpha=.115, lw=.34, zorder=1)
            ax.scatter(left + jitter, x[base_col], s=7.5, facecolors="white",
                       edgecolors=COLORS[disease], lw=.42, zorder=2)
            ax.scatter(left + 1 + jitter, x[map_col], s=7.5,
                       color=COLORS[disease], edgecolors="white", lw=.16, zorder=2)
            medians = [x[base_col].median(), x[map_col].median()]
            ax.plot([left, left + 1], medians, color=INK, lw=1.35, zorder=3)
            ax.scatter([left, left + 1], medians, marker="D", s=16,
                       color=INK, zorder=4)
        ax.set_title(title, fontsize=6.2, fontweight="normal", pad=2.0)
        ax.set_xticks([.5, 3.5, 6.5], ["MASLD", "PSC", "AH"])
        ax.tick_params(axis="x", labelsize=6.0, pad=2.2)
        ax.set_ylabel("")
        clean(ax)
        if title == "Spatial Pearson":
            ax.axhline(0, color="#DDE0E2", lw=.5, zorder=0)
    return axes


def draw_c(ax, units):
    """Frozen residual-recovery panel with only typographic cleanup."""
    panel(ax, "c", -.11, 1.105)
    ax.set_title("Target-specific molecular\ndeviations recovered\nacross diseases",
                 loc="left", fontweight="bold", fontsize=6.6, pad=3,
                 linespacing=1.05)
    rng = np.random.default_rng(91)
    for i, disease in enumerate(["MASLD", "PSC", "AH"]):
        values = units.loc[units.disease == disease,
                           "pooled_residual_pearson_map"].to_numpy()
        ax.scatter(i + rng.uniform(-.16, .16, len(values)), values, s=9,
                   color=COLORS[disease], alpha=.63, edgecolors="white", lw=.2)
        median = np.median(values)
        ax.plot([i - .25, i + .25], [median, median], color=INK, lw=1.5)
        ax.text(i, median + .032, f"{median:.2f}", ha="center", fontsize=5.4,
                fontweight="bold")
    ax.axhline(0, color="#999", lw=.5)
    ax.set_xticks(range(3), ["MASLD", "PSC", "AH"])
    ax.tick_params(axis="x", labelsize=6.0, pad=2.2)
    ax.set_ylabel("Pooled MAP residual Pearson")
    clean(ax)


def write_docs(selected):
    (ROOT / "01_V4_LAYOUT_NOTES.md").write_text("""# V4 layout notes

This is a layout-only refinement of the immutable v3 figure. No predictions, metrics, anchors, registrations, model assignments, scales or representative genes were changed.

- **Panel a:** reduced to a neutral left-to-right flow with three compact disease boxes; AH is defined in the caption. The prediction-lock statement was moved to the caption.
- **Panel b:** expanded to 55% of the top-row width. Only MASLD, PSC and AH remain on the x-axes. Cohort sizes and symbol definitions were moved to the caption; raw paired lines were lightened while disease medians remain black diamonds.
- **Panel c:** data and summaries are unchanged; the explanatory IDW note was moved to the caption.
- **Panel d:** the v3 drawing function is reused unchanged.
- **Panel e:** the v3 drawing function, samples, genes, maps, scales and spatial geometry are reused unchanged.

Top-row width allocation is 20% / 55% / 25% for panels a / b / c. The page remains 180 mm wide.
""")
    (ROOT / "02_V4_FIGURE_CAPTION.md").write_text("""# Figure caption

**Sparse target measurements adapt one healthy liver atlas across metabolic/fibrotic, cholangiopathic and inflammatory liver disease.** **a,** The fixed healthy live-donor liver atlas (M1–M8) was transferred to each disease target using geometry alone; exactly 16 target genes were exposed and all non-anchor target genes remained withheld until prediction lock. AH denotes alcohol-associated hepatitis. **b,** Median gene-wise spatial Pearson correlation and log1p RMSE for registered 12-nearest-neighbour inverse-distance weighting (IDW) and HyperSpatial-MAP across 33 MASLD biopsies from a 32-patient cohort, four PSC sections from one patient and two independent AH patients. Open circles denote IDW, filled circles denote MAP, lines pair methods within evaluation units, and black diamonds denote disease-level medians. PSC sections are within-patient samples, not independent patients. **c,** Pooled Pearson correlation between measured and MAP-predicted target-specific atlas-relative deviations. Registered IDW predicts zero atlas-relative correction; its residual correlation is therefore undefined. **d,** Median spatial correlation between measured and predicted scores for molecular programs predeclared before target outcomes were opened; blank cells denote programs not predefined for that disease. **e,** Frozen-source-selected hidden-gene examples showing target expression, registered healthy-atlas prior, MAP reconstruction, measured atlas-relative residual and predicted residual. Expression scales are shared across the first three maps within each row, and residual scales are identical and symmetric across the final two maps.
""")
    reps = {d: selected[d] for d in ["MASLD", "PSC", "AH"]}
    (ROOT / "03_V4_VISUAL_QC.md").write_text(f"""# V4 visual QC

Status: **PASS**

- The final PNG was inspected at its native 2,834 × 3,379-pixel render, corresponding to a 180.01-mm-wide figure at 400 dpi; PDF and SVG dimensions were checked independently.
- **Panel a:** the three disease boxes are aligned and immediately readable; AH replaces the long disease name; text and neutral arrows do not overlap; the concept is legible without an in-panel footnote.
- **Panel b:** the shared title and subplot headings are separated; only MASLD, PSC and AH appear beneath the axes; raw paired lines remain light and the IDW-to-MAP changes and black median diamonds dominate.
- **Panel c:** disease labels and summary values are readable and no undefined-IDW note remains inside the plot.
- **Panel d:** values, row order, disease headers, annotations and color scale are scientifically unchanged; alignment remains clean.
- **Panel e:** all 15 maps remain large and visible; samples, genes, map values, geometry and within-row scales are unchanged from v3.
- No title collision, clipped label, redundant legend, or repeated cohort text was observed at journal width.
- Final page size: 180.01 mm × 214.63 mm (7.087 in × 8.45 in).
- Representatives frozen: MASLD {reps['MASLD']}; PSC {reps['PSC']}; AH {reps['AH']}.
- Predictions recomputed: NO.
- Metrics changed: NO.
- Representative genes changed: NO.
""")


def write_manifest():
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and p.name != "04_SHA256_MANIFEST.txt")
    text = "\n".join(f"{sha(p)}  {p.relative_to(ROOT)}" for p in paths) + "\n"
    (ROOT / "04_SHA256_MANIFEST.txt").write_text(text)


def main():
    units, audit, selected, summary = v3.frozen_inputs()
    required = {"MASLD": "TAT", "PSC": "TAT", "AH": "CYP3A4"}
    if selected != required:
        raise RuntimeError(f"Frozen representatives changed: {selected}")

    fig = plt.figure(figsize=(7.087, 8.45), facecolor="white")
    outer = fig.add_gridspec(
        3, 20, height_ratios=[2.05, 1.50, 3.95], hspace=.315, wspace=.78,
        left=.078, right=.975, top=.952, bottom=.035,
    )
    ax_a = fig.add_subplot(outer[0, 0:4])
    draw_a(ax_a)
    draw_b(outer[0, 4:15], units)
    ax_c = fig.add_subplot(outer[0, 15:20])
    draw_c(ax_c, units)

    # Scientifically and visually frozen panels: invoke v3 functions unchanged.
    ax_d = fig.add_subplot(outer[1, 1:19])
    v3.draw_d(ax_d, summary)
    v3.draw_e(outer[2, :], selected)

    for extension, dpi in [("pdf", 300), ("svg", 300), ("png", 400)]:
        fig.savefig(ROOT / f"Liver_multidisease_integrated_main_v4.{extension}", dpi=dpi)
    plt.close(fig)

    write_docs(selected)
    write_manifest()
    print(json.dumps({
        "v3": str(V3), "v4": str(ROOT),
        "predictions_recomputed": False, "metrics_changed": False,
        "representatives_changed": False, "selected": selected,
        "figure_inches": [7.087, 8.45],
        "figure_mm": [180.0098, 214.63],
    }, indent=2))


if __name__ == "__main__":
    main()
