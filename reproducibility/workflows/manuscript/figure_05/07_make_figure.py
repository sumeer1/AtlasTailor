#!/usr/bin/env python3
"""Create the publication-style SPATCH cross-platform MAP figure draft."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "09_FIGURES"
FIG.mkdir(exist_ok=True)
COLORS = {"registered_idw_k12": "#9AA0A6", "anchor_only_decoder": "#D98C4A", "hyperspatial_map": "#2374AB"}
LABELS = {"COAD": "Colon", "HCC": "Liver", "OV": "Ovary"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def letter(ax, value):
    ax.text(-0.10, 1.08, value, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


pair = pd.read_csv(ROOT / "03_PAIR_LEVEL_RESULTS.tsv", sep="\t")
fig = plt.figure(figsize=(15.8, 12.2), constrained_layout=False)
outer = GridSpec(3, 12, figure=fig, height_ratios=[1.0, 1.0, 2.15], hspace=0.42, wspace=1.4)

# a — design
ax = fig.add_subplot(outer[0, 0:4]); ax.axis("off"); letter(ax, "a")
ax.set_title("Same-specimen serial-section design", loc="left", fontweight="bold", pad=8)
boxes = [(0.02, "Visium HD FFPE\n8-µm reference"), (0.37, "Geometry-only\nregistration"),
         (0.70, "Xenium 5K target\n16 anchors exposed")]
for x, text in boxes:
    ax.add_patch(plt.Rectangle((x, .44), .25, .26, transform=ax.transAxes, facecolor="#F4F6F8", edgecolor="#555", lw=.8))
    ax.text(x+.125, .57, text, transform=ax.transAxes, ha="center", va="center", fontsize=7.3)
for start, end in ((.275,.36),(.625,.69)):
    ax.annotate("", xy=(end, .57), xytext=(start, .57), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
ax.text(.49, .27, "Registered atlas prior + sparse target residuals\n→ hidden Xenium molecular state",
        transform=ax.transAxes, ha="center", color="#2374AB", fontweight="bold", fontsize=7.6)
ax.text(.5, .08, "Serial sections; no cell-to-cell correspondence assumed", transform=ax.transAxes,
        ha="center", fontsize=7, color="#555")

# b — measurement budget
ax = fig.add_subplot(outer[0, 4:7]); letter(ax, "b")
ax.set_title("Sparse target measurement", loc="left", fontweight="bold")
maps = pair.loc[pair.method == "hyperspatial_map"].set_index("tumor").loc[["COAD", "HCC", "OV"]]
x = np.arange(3)
ax.bar(x-.18, [16]*3, .36, color="#D98C4A", label="Measured anchors")
ax.bar(x+.18, maps.hidden_gene_count, .36, color="#2374AB", label="Hidden genes")
ax.set_yscale("log"); ax.set_ylabel("Genes (log scale)"); ax.set_xticks(x, [LABELS[t] for t in maps.index])
ax.legend(frameon=False, fontsize=7, loc="lower right"); clean(ax)

# c — geometry-only registration
sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 7:12], wspace=.08)
for i, tumor in enumerate(("COAD", "HCC", "OV")):
    ax = fig.add_subplot(sub[0, i])
    if i == 0: letter(ax, "c")
    lock = json.loads((ROOT / "work" / tumor / "source_geometry_lock.json").read_text())
    model = np.load(ROOT / "work" / tumor / "source_model.npz", allow_pickle=True)
    mapped = np.load(ROOT / "work" / tumor / "seed42_registered_source_xy.npy", mmap_mode="r")
    roi = np.load(ROOT / "work" / tumor / "geometry_only_target_roi_rows.npy")
    target = ad.read_h5ad(ROOT / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_shared_counts.h5ad", backed="r")
    raw = np.asarray(target.obsm["spatial"], np.float32)[roi]
    target.file.close()
    transform = lock["target_coordinate_transform"]
    xy = (raw - np.asarray(transform["center"])) / float(transform["scale"])
    take_t = np.linspace(0, len(xy)-1, min(12000, len(xy)), dtype=int)
    take_s = np.linspace(0, len(mapped)-1, min(5000, len(mapped)), dtype=int)
    ax.scatter(xy[take_t,0], xy[take_t,1], s=.15, c="#B7BCC2", rasterized=True)
    ax.scatter(mapped[take_s,0], mapped[take_s,1], s=.18, c="#2374AB", alpha=.35, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(LABELS[tumor], fontsize=8)
    if i == 0:
        ax.text(-.02, 1.18, "Geometry-only registration", transform=ax.transAxes,
                ha="left", fontweight="bold", fontsize=9)
    if i == 0:
        ax.text(.02, .02, "Xenium support", transform=ax.transAxes, color="#777", fontsize=6)
        ax.text(.02, .10, "registered Visium", transform=ax.transAxes, color="#2374AB", fontsize=6)

# d — endpoints
sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0:7], wspace=.34)
for j, metric in enumerate(("median_gene_wise_spatial_pearson", "log1p_rmse")):
    ax = fig.add_subplot(sub[0, j])
    if j == 0: letter(ax, "d")
    for i, tumor in enumerate(("COAD", "HCC", "OV")):
        values = [pair.loc[(pair.tumor == tumor) & (pair.method == m), metric].iloc[0]
                  for m in ("registered_idw_k12", "hyperspatial_map")]
        ax.plot([0,1], values, color="#888", lw=.9, marker="o", ms=4)
        ax.text(1.04, values[1], LABELS[tumor], va="center", fontsize=6.5)
    ax.set_xlim(-.15, 1.45); ax.set_xticks([0,1], ["IDW", "MAP"])
    ax.set_ylabel("Median spatial Pearson" if j == 0 else "log1p RMSE")
    ax.set_title("Spatial reconstruction" if j == 0 else "Absolute error", loc="left", fontweight="bold")
    clean(ax)

# e — residual recovery
ax = fig.add_subplot(outer[1, 7:12]); letter(ax, "e")
ax.set_title("Atlas-relative residual recovery", loc="left", fontweight="bold")
x = np.arange(3); width=.25
for offset, method in zip((-.16,.16), ("anchor_only_decoder", "hyperspatial_map")):
    values = [pair.loc[(pair.tumor == t) & (pair.method == method), "pooled_bin_gene_residual_pearson"].iloc[0]
              for t in ("COAD", "HCC", "OV")]
    ax.bar(x+offset, values, width, color=COLORS[method], label=method.replace("registered_idw_k12","IDW").replace("anchor_only_decoder","Anchor only").replace("hyperspatial_map","MAP"))
ax.axhline(0, color="#444", lw=.7); ax.set_xticks(x, [LABELS[t] for t in ("COAD","HCC","OV")])
ax.set_ylabel("Pooled residual Pearson"); ax.legend(frameon=False, ncol=2, fontsize=7)
ax.text(.99, .97, "IDW predicts zero residual (r undefined)", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.5, color="#666")
clean(ax)

# f — two source-only representative genes from COAD
bottom = GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[2, :], hspace=.12, wspace=.06)
tumor = "COAD"; model = np.load(ROOT / "work" / tumor / "source_model.npz", allow_pickle=True)
evaluation = model["evaluation_indices"].astype(int)
pool = model["source_only_representative_gene_pool"].astype(int)[:2]
eval_pos = {int(g): i for i, g in enumerate(evaluation)}
roi = np.load(ROOT / "work" / tumor / "geometry_only_target_roi_rows.npy")
depth = np.load(ROOT / "work" / tumor / "source_model_predicted_target_depth.npy")
target = ad.read_h5ad(ROOT / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_shared_counts.h5ad", backed="r")
xy = np.asarray(target.obsm["spatial"], np.float32)[roi]
selected = target[roi, pool].X
if hasattr(selected, "to_memory"): selected = selected.to_memory()
truth = np.log1p(selected.toarray().astype(np.float32) * (float(model["source_depth_target"]) / depth[:,None]))
target.file.close()
prior_all = np.load(ROOT / "work" / tumor / "predictions" / "registered_idw_k12_log1p.npy", mmap_mode="r")
map_all = np.load(ROOT / "work" / tumor / "predictions" / "hyperspatial_map_log1p.npy", mmap_mode="r")
take = np.linspace(0, len(xy)-1, min(110000, len(xy)), dtype=int)
titles = ("Xenium truth", "Registered Visium prior", "MAP reconstruction", "True residual", "MAP residual")
for row, gene_index in enumerate(pool):
    prior = np.asarray(prior_all[:, eval_pos[int(gene_index)]])
    prediction = np.asarray(map_all[:, eval_pos[int(gene_index)]])
    values = (truth[:,row], prior, prediction, truth[:,row]-prior, prediction-prior)
    vmax = float(np.quantile(np.concatenate(values[:3]), .99))
    rmax = float(np.quantile(np.abs(np.concatenate(values[3:])), .99))
    for col, value in enumerate(values):
        ax = fig.add_subplot(bottom[row,col])
        if row == 0 and col == 0: letter(ax, "f")
        if row == 0: ax.set_title(titles[col], fontsize=8, fontweight="bold")
        if col < 3:
            ax.scatter(xy[take,0], xy[take,1], c=value[take], s=.22, cmap="viridis", vmin=0, vmax=max(vmax,1e-5), rasterized=True)
        else:
            ax.scatter(xy[take,0], xy[take,1], c=value[take], s=.22, cmap="coolwarm", vmin=-max(rmax,1e-5), vmax=max(rmax,1e-5), rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        if col == 0:
            ax.text(-.02, .5, str(model["gene_names"][gene_index]), transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontstyle="italic", fontsize=9)
        if row == 1 and col == 4:
            ax.text(1.0, -.10, "Representative genes predeclared by source-only spatial score",
                    transform=ax.transAxes, ha="right", fontsize=6.5, color="#555")

fig.savefig(FIG / "SPATCH_cross_platform_MAP_main.pdf", bbox_inches="tight")
fig.savefig(FIG / "SPATCH_cross_platform_MAP_main.png", dpi=300, bbox_inches="tight")
plt.close(fig)
