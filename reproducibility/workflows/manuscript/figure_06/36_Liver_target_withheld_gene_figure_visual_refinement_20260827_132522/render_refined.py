#!/usr/bin/env python3
"""Visual-only refinement of the frozen target-withheld liver figures."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[1]
SOURCE = PROJECT / "PI_revision/35_Liver_target_withheld_gene_main_figure_20260827_131229"
SOURCE_SCRIPT = SOURCE / "scripts/build_figures.py"
LOCKED = PROJECT / "PI_revision/34_Liver_target_withheld_gene_interpretation_20260827_094120"

ORIGINAL_SIZE = {"MASLD": 3.5, "PSC": 4.0, "AH": 4.0}
REFINED_SIZE = {"MASLD": 4.9, "PSC": 5.0, "AH": 5.0}
ALPHA = 0.98
METRIC_FONT_OLD = 7.0
METRIC_FONT_NEW = 7.8

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def load_source_module():
    spec = importlib.util.spec_from_file_location("v35_figure", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def scale_limits(z):
    values = [z["truth"], z["prior"], z["pred"],
              z["measured_residual"], z["predicted_residual"]]
    expression_low = min(float(np.nanmin(x)) for x in values[:3])
    expression_high = max(float(np.nanmax(x)) for x in values[:3])
    residual_max = max(float(np.nanmax(np.abs(values[3]))),
                       float(np.nanmax(np.abs(values[4]))), 1e-6)
    return values, expression_low, expression_high, residual_max


def draw_schematic(ax, source, supplementary):
    if supplementary:
        ax.set_axis_off()
        ax.text(0.015, 0.66, "Target-withheld TAT reconstruction",
                fontsize=13, fontweight="bold", va="center")
        ax.text(0.015, 0.20,
                "Healthy liver atlas + 16 measured target genes → target-withheld disease-state expression",
                fontsize=8.4, color="#666B70", va="center")
    else:
        source.draw_schematic(ax)


def render(entries, maps, summary, source, stem, supplementary=False):
    ncol = len(entries)
    width = 10.5 if ncol == 3 else 7.8
    fig = plt.figure(figsize=(width, 12.0), facecolor="white")
    gs = fig.add_gridspec(
        7, ncol + 1,
        height_ratios=[0.72, 1.48, 2.0, 2.0, 2.0, 2.0, 2.0],
        width_ratios=[0.62] + [1.0] * ncol,
        left=0.045, right=0.975, top=0.985, bottom=0.035,
        wspace=0.08, hspace=0.09,
    )

    schematic = fig.add_subplot(gs[0, :])
    draw_schematic(schematic, source, supplementary)
    fig.text(0.022, 0.872, "a" if supplementary else "b",
             fontsize=15, fontweight="bold", va="top")

    for col, (disease, gene, biology) in enumerate(entries, start=1):
        head = fig.add_subplot(gs[1, col])
        head.set_axis_off()
        head.add_patch(plt.Rectangle((0.0, 0.90), 1.0, 0.045, transform=head.transAxes,
                                     color=source.COLORS[disease], lw=0))
        head.text(0.5, 0.78, disease, ha="center", va="center", fontsize=12.2,
                  color=source.COLORS[disease], fontweight="bold")
        head.text(0.5, 0.57, rf"$\it{{{gene}}}$", ha="center", va="center",
                  fontsize=14.0, fontweight="bold")
        head.text(0.5, 0.40, biology, ha="center", va="center", fontsize=8.8)
        record = source.result_row(summary, disease, gene)
        head.text(0.5, 0.15, source.metric_text(record), ha="center", va="center",
                  fontsize=METRIC_FONT_NEW, linespacing=1.27, color="#373B3E",
                  bbox=dict(boxstyle="round,pad=0.30", fc="#F6F7F7", ec="#D6DADC", lw=0.65))

    label_ax = fig.add_subplot(gs[1, 0])
    label_ax.set_axis_off()
    label_ax.text(0.0, 0.55, "Frozen summary\n(IDW → MAP)", fontsize=7.4,
                  color="#666B70", ha="left", va="center")

    for row_index, row_label in enumerate(source.ROWS):
        la = fig.add_subplot(gs[row_index + 2, 0])
        la.set_axis_off()
        la.text(0.98, 0.5, row_label, ha="right", va="center", fontsize=8.4,
                fontweight="bold" if row_index in (0, 2) else "normal",
                color="#303437")
        for col, (disease, gene, _) in enumerate(entries, start=1):
            z = maps[(disease, gene)]
            values, expression_low, expression_high, residual_max = scale_limits(z)
            ax = fig.add_subplot(gs[row_index + 2, col])
            if row_index < 3:
                norm = Normalize(vmin=expression_low, vmax=expression_high)
                cmap = "viridis"
            else:
                norm = TwoSlopeNorm(vmin=-residual_max, vcenter=0, vmax=residual_max)
                cmap = "coolwarm"
            ax.scatter(z["xy"][:, 0], z["xy"][:, 1], c=values[row_index],
                       s=REFINED_SIZE[disease], alpha=ALPHA, cmap=cmap, norm=norm,
                       linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 4:
                ax.text(0.5, -0.07,
                        f"Expression 0–{expression_high:.2f}  ·  Residual ±{residual_max:.2f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=6.7,
                        color="#666B70")

    for ext, dpi in [("pdf", 300), ("svg", 300), ("png", 360)]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


def write_log(entries, maps):
    lines = [
        "# Visual refinement log",
        "",
        "This run changes rendering only. Frozen coordinates, units, genes, predictions, metrics, biological labels and numerical normalization limits were preserved.",
        "",
        "## Rendering parameters",
        "",
        "| Disease / gene | Figure | Original marker size | Refined marker size | Change | Alpha | Expression vmin | Expression vmax | Residual vmin | Residual vmax |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for figure, figure_entries in entries:
        for disease, gene, _ in figure_entries:
            _, expression_low, expression_high, residual_max = scale_limits(maps[(disease, gene)])
            change = 100 * (REFINED_SIZE[disease] / ORIGINAL_SIZE[disease] - 1)
            lines.append(
                f"| {disease} / {gene} | {figure} | {ORIGINAL_SIZE[disease]:.2f} | "
                f"{REFINED_SIZE[disease]:.2f} | +{change:.1f}% | {ALPHA:.2f} | "
                f"{expression_low:.9g} | {expression_high:.9g} | {-residual_max:.9g} | {residual_max:.9g} |"
            )
    lines += [
        "",
        f"- Metric-summary font: {METRIC_FONT_OLD:.1f} pt → {METRIC_FONT_NEW:.1f} pt (+{100 * (METRIC_FONT_NEW / METRIC_FONT_OLD - 1):.1f}%).",
        "- Expression colormap and normalization: unchanged (`viridis`; one shared vmin/vmax across measured target, IDW and MAP within each disease/gene).",
        "- Residual colormap and normalization: unchanged (`coolwarm`; one shared symmetric vmin/vmax across measured and predicted residuals within each disease/gene).",
        "- Color limits changed: **NO**.",
        "- Predictions recomputed: **NO**.",
        "- Metrics recomputed or changed: **NO**.",
        "- Genes or representative units changed: **NO**.",
    ]
    (OUT / "VISUAL_REFINEMENT_LOG.md").write_text("\n".join(lines) + "\n")


def main():
    source = load_source_module()
    summary = pd.read_csv(LOCKED / "TARGET_WITHHELD_GENE_RESULTS.tsv", sep="\t")
    lock = pd.read_csv(LOCKED / "CANDIDATE_GENE_LOCK.tsv", sep="\t")
    source.validate_inputs(summary, lock)

    loader = source.load_frozen_loader()
    maps = {}
    for disease, genes in {"MASLD": ["DCN", "TAT"], "PSC": ["KRT8", "TAT"], "AH": ["NFKBIA"]}.items():
        for gene, values in loader.load_gene_maps(disease, genes).items():
            maps[(disease, gene)] = values

    render(source.MAIN, maps, summary, source, "Liver_target_withheld_genes_main_refined")
    render(source.SUPP, maps, summary, source,
           "Liver_target_withheld_genes_TAT_supplementary_refined", supplementary=True)
    write_log([("main", source.MAIN), ("supplementary", source.SUPP)], maps)


if __name__ == "__main__":
    main()
