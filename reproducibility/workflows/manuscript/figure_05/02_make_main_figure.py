#!/usr/bin/env python3
"""Render the COAD/OV manuscript figure using frozen IDW and MAP outputs."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
ORIGINAL = PROJECT / "PI_revision" / "23_SPATCH_cross_platform_MAP_20260824_111004"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
COLORS = {"IDW": "#8E969E", "MAP": "#2374AB", "anchor": "#D98C4A"}
LABELS = {"COAD": "Colon (COAD)", "OV": "Ovary (OV)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def letter(ax, value):
    ax.text(-.10, 1.08, value, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


results = pd.read_csv(HERE / "02_SPECIMEN_LEVEL_RESULTS_IDW_vs_MAP.tsv", sep="\t").set_index("tumor")
fig = plt.figure(figsize=(15.8, 12.2), constrained_layout=False)
outer = GridSpec(3, 12, figure=fig, height_ratios=[1.0, 1.0, 2.15], hspace=.43, wspace=1.35)

# a — serial-section design
ax = fig.add_subplot(outer[0, 0:4]); ax.axis("off"); letter(ax, "a")
ax.set_title("Same-specimen serial-section cross-platform design", loc="left", fontweight="bold", pad=8)
boxes = [(.02, "Visium HD FFPE\n8-µm reference"), (.37, "Geometry-only\nregistration"),
         (.70, "Xenium 5K target\n16 anchors exposed")]
for x, label in boxes:
    ax.add_patch(plt.Rectangle((x, .44), .25, .26, transform=ax.transAxes,
                               facecolor="#F4F6F8", edgecolor="#555", lw=.8))
    ax.text(x+.125, .57, label, transform=ax.transAxes, ha="center", va="center", fontsize=7.3)
for start, end in ((.275, .36), (.625, .69)):
    ax.annotate("", xy=(end, .57), xytext=(start, .57), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
ax.text(.49, .27, "Registered atlas prior + 16 target genes\n→ hidden Xenium molecular state",
        transform=ax.transAxes, ha="center", color=COLORS["MAP"], fontweight="bold", fontsize=7.7)
ax.text(.5, .08, "Serial sections; no cell-to-cell correspondence assumed",
        transform=ax.transAxes, ha="center", fontsize=7, color="#555")

# b — target budget
ax = fig.add_subplot(outer[0, 4:7]); letter(ax, "b")
ax.set_title("Sparse target measurement", loc="left", fontweight="bold")
tumors = ("COAD", "OV"); x = np.arange(2)
ax.bar(x-.18, [16,16], .36, color=COLORS["anchor"], label="Measured anchors")
hidden = [int(results.loc[t, "hidden_shared_gene_count"]) for t in tumors]
ax.bar(x+.18, hidden, .36, color=COLORS["MAP"], label="Hidden genes")
for pos, value in zip(x+.18, hidden):
    ax.text(pos, value*1.04, str(value), ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(10, 1100); ax.set_ylabel("Genes (log scale)")
ax.set_xticks(x, [LABELS[t] for t in tumors]); ax.legend(frameon=False, fontsize=7, loc="lower right")
clean(ax)

# c — geometry-only registration for primary specimens
sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 7:12], wspace=.08)
for i, tumor in enumerate(tumors):
    ax = fig.add_subplot(sub[0, i])
    if i == 0: letter(ax, "c")
    lock = json.loads((ORIGINAL / "work" / tumor / "source_geometry_lock.json").read_text())
    mapped = np.load(ORIGINAL / "work" / tumor / "seed42_registered_source_xy.npy", mmap_mode="r")
    roi = np.load(ORIGINAL / "work" / tumor / "geometry_only_target_roi_rows.npy")
    target = ad.read_h5ad(ORIGINAL / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_shared_counts.h5ad", backed="r")
    raw = np.asarray(target.obsm["spatial"], np.float32)[roi]; target.file.close()
    transform = lock["target_coordinate_transform"]
    xy = (raw - np.asarray(transform["center"])) / float(transform["scale"])
    take_t = np.linspace(0, len(xy)-1, min(16000, len(xy)), dtype=int)
    take_s = np.linspace(0, len(mapped)-1, min(6500, len(mapped)), dtype=int)
    ax.scatter(xy[take_t,0], xy[take_t,1], s=.14, c="#B7BCC2", rasterized=True)
    ax.scatter(mapped[take_s,0], mapped[take_s,1], s=.18, c=COLORS["MAP"], alpha=.38, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(LABELS[tumor], fontsize=8)
    if i == 0:
        ax.text(-.02, 1.18, "Geometry-only registration", transform=ax.transAxes,
                ha="left", fontweight="bold", fontsize=9)
        ax.text(.02, .02, "Xenium support", transform=ax.transAxes, color="#777", fontsize=6)
        ax.text(.02, .10, "registered Visium HD", transform=ax.transAxes,
                color=COLORS["MAP"], fontsize=6)

# d — IDW vs MAP pair-level endpoints
sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0:8], wspace=.33)
endpoint_specs = [
    ("IDW_median_gene_wise_spatial_pearson", "MAP_median_gene_wise_spatial_pearson",
     "Spatial reconstruction", "Median spatial Pearson"),
    ("IDW_log1p_RMSE", "MAP_log1p_RMSE", "Reconstruction error", "log1p RMSE"),
]
for j, (left, right, title, ylabel) in enumerate(endpoint_specs):
    ax = fig.add_subplot(sub[0, j])
    if j == 0: letter(ax, "d")
    for tumor in tumors:
        values = [results.loc[tumor, left], results.loc[tumor, right]]
        ax.plot([0,1], values, color="#858B90", lw=1.1, marker="o", ms=4.5)
        ax.text(1.04, values[1], LABELS[tumor], va="center", fontsize=6.8)
    ax.set_xlim(-.15, 1.48); ax.set_xticks([0,1], ["IDW", "MAP"]); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold"); clean(ax)
    if j == 0:
        ax.text(-.02, 1.17, "Reconstruction beyond registered atlas transfer",
                transform=ax.transAxes, ha="left", fontweight="bold", fontsize=9)

# e — MAP residual recovery only
ax = fig.add_subplot(outer[1, 8:12]); letter(ax, "e")
ax.set_title("Atlas-relative residual recovery", loc="left", fontweight="bold")
values = [results.loc[t, "MAP_pooled_residual_pearson"] for t in tumors]
bars = ax.bar(np.arange(2), values, .48, color=COLORS["MAP"])
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, value+.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
ax.axhline(0, color="#444", lw=.7); ax.set_ylim(0, .62)
ax.set_xticks(np.arange(2), [LABELS[t] for t in tumors]); ax.set_ylabel("MAP pooled residual Pearson")
ax.text(.50, .98, "Registered IDW predicts zero residual; correlation undefined.",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color="#666")
clean(ax)

# f — unchanged source-predeclared COAD examples
bottom = GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[2, :], hspace=.12, wspace=.06)
tumor = "COAD"
model = np.load(ORIGINAL / "work" / tumor / "source_model.npz", allow_pickle=True)
evaluation = model["evaluation_indices"].astype(int)
pool = model["source_only_representative_gene_pool"].astype(int)[:2]
eval_position = {int(gene): i for i, gene in enumerate(evaluation)}
roi = np.load(ORIGINAL / "work" / tumor / "geometry_only_target_roi_rows.npy")
depth = np.load(ORIGINAL / "work" / tumor / "source_model_predicted_target_depth.npy")
target = ad.read_h5ad(ORIGINAL / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_shared_counts.h5ad", backed="r")
xy = np.asarray(target.obsm["spatial"], np.float32)[roi]
selected = target[roi, pool].X
if hasattr(selected, "to_memory"): selected = selected.to_memory()
truth = np.log1p(selected.toarray().astype(np.float32) *
                 (float(model["source_depth_target"]) / depth[:,None]))
target.file.close()
prior_all = np.load(ORIGINAL / "work" / tumor / "predictions" / "registered_idw_k12_log1p.npy", mmap_mode="r")
map_all = np.load(ORIGINAL / "work" / tumor / "predictions" / "hyperspatial_map_log1p.npy", mmap_mode="r")
take = np.linspace(0, len(xy)-1, min(110000, len(xy)), dtype=int)
titles = ("Xenium truth", "Registered Visium-HD prior", "MAP reconstruction",
          "Measured atlas-relative residual", "Predicted atlas-relative residual")
for row, gene_index in enumerate(pool):
    prior = np.asarray(prior_all[:, eval_position[int(gene_index)]])
    prediction = np.asarray(map_all[:, eval_position[int(gene_index)]])
    fields = (truth[:,row], prior, prediction, truth[:,row]-prior, prediction-prior)
    vmax = float(np.quantile(np.concatenate(fields[:3]), .99))
    rmax = float(np.quantile(np.abs(np.concatenate(fields[3:])), .99))
    for col, field in enumerate(fields):
        ax = fig.add_subplot(bottom[row,col])
        if row == 0 and col == 0: letter(ax, "f")
        if row == 0: ax.set_title(titles[col], fontsize=7.8, fontweight="bold")
        if col < 3:
            ax.scatter(xy[take,0], xy[take,1], c=field[take], s=.22, cmap="viridis",
                       vmin=0, vmax=max(vmax,1e-5), rasterized=True)
        else:
            ax.scatter(xy[take,0], xy[take,1], c=field[take], s=.22, cmap="coolwarm",
                       vmin=-max(rmax,1e-5), vmax=max(rmax,1e-5), rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        if col == 0:
            label = f"COAD · {model['gene_names'][gene_index]}"
            ax.text(-.02, .5, label, transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontstyle="italic", fontsize=8.5)
        if row == 1 and col == 4:
            ax.text(1.0, -.10, "Representative genes fixed by the original source-only rule",
                    transform=ax.transAxes, ha="right", fontsize=6.5, color="#555")

fig.savefig(FIGURES / "SPATCH_cross_platform_IDW_vs_MAP_main.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "SPATCH_cross_platform_IDW_vs_MAP_main.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "SPATCH_cross_platform_IDW_vs_MAP_main.svg", bbox_inches="tight")
plt.close(fig)
