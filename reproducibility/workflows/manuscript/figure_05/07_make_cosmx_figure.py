#!/usr/bin/env python3
"""Make the editable IDW-vs-MAP CosMx manuscript figure."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "07_FIGURES"
FIG.mkdir(exist_ok=True)
TUMORS = ("COAD", "HCC", "OV")
LABELS = {"COAD": "Colon (COAD)", "HCC": "Liver (HCC)", "OV": "Ovary (OV)"}
IDW = "#9AA0A6"
MAP = "#2374AB"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.7, "lines.linewidth": 1.0, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
})


def panel_letter(ax, value: str) -> None:
    ax.text(-0.10, 1.08, value, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


pair = pd.read_csv(ROOT / "03_PAIR_LEVEL_RESULTS.tsv", sep="\t").set_index("tumor").loc[list(TUMORS)]
fig = plt.figure(figsize=(15.8, 13.8))
outer = GridSpec(3, 15, figure=fig, height_ratios=[1.0, 1.0, 2.35], hspace=.42, wspace=1.3)

# a — experimental design
ax = fig.add_subplot(outer[0, :6]); ax.axis("off"); panel_letter(ax, "a")
ax.set_title("Same-specimen serial-section design", loc="left", fontweight="bold", pad=8)
boxes = [(.02, "Visium HD FFPE\n8-µm reference"), (.36, "Geometry-only\nregistration"),
         (.70, "CosMx 6K target\n16 genes exposed")]
for x, label in boxes:
    ax.add_patch(plt.Rectangle((x, .45), .25, .25, transform=ax.transAxes,
                               facecolor="#F4F6F8", edgecolor="#555", lw=.8))
    ax.text(x + .125, .575, label, transform=ax.transAxes, ha="center", va="center")
for start, stop in ((.275, .35), (.615, .69)):
    ax.annotate("", xy=(stop, .575), xytext=(start, .575), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
ax.text(.49, .27, "Registered Visium-HD prior + 16 CosMx genes\n→ hidden CosMx molecular state",
        transform=ax.transAxes, ha="center", color=MAP, fontweight="bold")
ax.text(.5, .07, "Serial sections; no one-to-one cell correspondence assumed",
        transform=ax.transAxes, ha="center", fontsize=7, color="#555")

# b — geometry
sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 6:], wspace=.08)
for index, tumor in enumerate(TUMORS):
    ax = fig.add_subplot(sub[0, index])
    if index == 0:
        panel_letter(ax, "b")
        ax.text(-.02, 1.18, "Geometry-only registration", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    lock = json.loads((ROOT / "work" / tumor / "source_geometry_lock.json").read_text())
    mapped = np.load(ROOT / "work" / tumor / "seed42_registered_source_xy.npy", mmap_mode="r")
    roi = np.load(ROOT / "work" / tumor / "geometry_only_target_roi_rows.npy")
    target = ad.read_h5ad(ROOT / "SEALED_TARGET_8UM" / f"CosMx_{tumor}_8um_shared_counts.h5ad", backed="r")
    raw = np.asarray(target.obsm["spatial"], np.float32)[roi]
    target.file.close()
    transform = lock["target_coordinate_transform"]
    xy = (raw - np.asarray(transform["center"], np.float32)) / float(transform["scale"])
    take_t = np.linspace(0, len(xy) - 1, min(15000, len(xy)), dtype=int)
    take_s = np.linspace(0, len(mapped) - 1, min(6000, len(mapped)), dtype=int)
    ax.scatter(xy[take_t, 0], xy[take_t, 1], s=.14, c="#B7BCC2", rasterized=True)
    ax.scatter(mapped[take_s, 0], mapped[take_s, 1], s=.18, c=MAP, alpha=.35, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(LABELS[tumor], fontsize=8)
    if index == 0:
        ax.text(.02, .02, "CosMx support", transform=ax.transAxes, color="#777", fontsize=6)
        ax.text(.02, .10, "registered Visium HD", transform=ax.transAxes, color=MAP, fontsize=6)

# c, d — reconstruction endpoints
for col, (letter, idw_col, map_col, ylabel, title) in enumerate((
    ("c", "IDW_median_gene_wise_spatial_pearson", "MAP_median_gene_wise_spatial_pearson",
     "Median spatial Pearson", "Spatial reconstruction"),
    ("d", "IDW_log1p_RMSE", "MAP_log1p_RMSE", "log1p RMSE", "Reconstruction error"),
)):
    ax = fig.add_subplot(outer[1, col * 5:(col + 1) * 5]); panel_letter(ax, letter)
    for tumor in TUMORS:
        values = [pair.loc[tumor, idw_col], pair.loc[tumor, map_col]]
        ax.plot([0, 1], values, color="#777", marker="o", markersize=4)
        ax.text(1.05, values[1], tumor, va="center", fontsize=6.5)
    ax.set_xlim(-.15, 1.45); ax.set_xticks([0, 1], ["IDW", "MAP"])
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold"); clean(ax)

# e — residual recovery
ax = fig.add_subplot(outer[1, 10:]); panel_letter(ax, "e")
ax.set_title("Atlas-relative residual recovery", loc="left", fontweight="bold")
x = np.arange(3)
values = pair.MAP_pooled_residual_pearson.to_numpy()
ax.bar(x, values, width=.58, color=MAP)
ax.axhline(0, color="#444", lw=.7)
ax.set_xticks(x, list(TUMORS)); ax.set_ylabel("MAP pooled residual Pearson")
for xx, value in zip(x, values):
    ax.text(xx, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
ax.text(.99, .98, "IDW predicts zero residual; r undefined", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.5, color="#666")
clean(ax)

# f — one source-only representative gene per specimen
bottom = GridSpecFromSubplotSpec(3, 5, subplot_spec=outer[2, :], hspace=.10, wspace=.06)
titles = ("CosMx truth", "Registered Visium-HD prior", "MAP reconstruction",
          "Measured atlas-relative residual", "Predicted atlas-relative residual")
for row, tumor in enumerate(TUMORS):
    model = np.load(ROOT / "work" / tumor / "source_model.npz", allow_pickle=True)
    gene_index = int(model["source_only_representative_gene_pool"].astype(int)[0])
    evaluation = model["evaluation_indices"].astype(int)
    position = {int(gene): i for i, gene in enumerate(evaluation)}[gene_index]
    roi = np.load(ROOT / "work" / tumor / "geometry_only_target_roi_rows.npy")
    depth = np.load(ROOT / "work" / tumor / "source_model_predicted_target_depth.npy")
    target = ad.read_h5ad(ROOT / "SEALED_TARGET_8UM" / f"CosMx_{tumor}_8um_shared_counts.h5ad", backed="r")
    xy = np.asarray(target.obsm["spatial"], np.float32)[roi]
    selected = target[roi, [gene_index]].X
    if hasattr(selected, "to_memory"):
        selected = selected.to_memory()
    selected = selected.toarray() if hasattr(selected, "toarray") else np.asarray(selected)
    truth = np.log1p(selected.ravel().astype(np.float32) *
                     (float(model["source_depth_target"]) / depth))
    target.file.close()
    prior = np.asarray(np.load(ROOT / "work" / tumor / "predictions" /
                               "registered_idw_k12_log1p.npy", mmap_mode="r")[:, position])
    prediction = np.asarray(np.load(ROOT / "work" / tumor / "predictions" /
                                    "hyperspatial_map_log1p.npy", mmap_mode="r")[:, position])
    values = (truth, prior, prediction, truth - prior, prediction - prior)
    vmax = max(float(np.quantile(np.concatenate(values[:3]), .99)), 1e-5)
    rmax = max(float(np.quantile(np.abs(np.concatenate(values[3:])), .99)), 1e-5)
    take = np.linspace(0, len(xy) - 1, min(90000, len(xy)), dtype=int)
    for col, value in enumerate(values):
        ax = fig.add_subplot(bottom[row, col])
        if row == 0 and col == 0:
            panel_letter(ax, "f")
        if row == 0:
            ax.set_title(titles[col], fontsize=8, fontweight="bold")
        if col < 3:
            ax.scatter(xy[take, 0], xy[take, 1], c=value[take], s=.20, cmap="viridis",
                       vmin=0, vmax=vmax, rasterized=True)
        else:
            ax.scatter(xy[take, 0], xy[take, 1], c=value[take], s=.20, cmap="coolwarm",
                       vmin=-rmax, vmax=rmax, rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        if col == 0:
            ax.text(-.02, .5, f"{tumor}  {model['gene_names'][gene_index]}", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontstyle="italic", fontsize=8)
        if row == 2 and col == 4:
            ax.text(1.0, -.10, "One gene per specimen; frozen source-only spatial-score rule",
                    transform=ax.transAxes, ha="right", fontsize=6.5, color="#555")

base = FIG / "SPATCH_VisiumHD_to_CosMx_IDW_vs_MAP"
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
