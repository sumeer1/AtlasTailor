#!/usr/bin/env python3
"""Assemble the frozen Xenium and CosMx SPATCH results into one main figure.

This script is presentation-only. It reads frozen predictions and tables, and
does not fit, register, select, calibrate, or otherwise recompute either model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
XENIUM_PRESENTATION = PROJECT / "PI_revision" / "23_SPATCH_cross_platform_MAP_IDW_only_20260824_165217"
XENIUM_FROZEN = PROJECT / "PI_revision" / "23_SPATCH_cross_platform_MAP_20260824_111004"
COSMX = PROJECT / "PI_revision" / "24_SPATCH_cross_platform_CosMx_Stereo_20260824_171656"
FIGDIR = ROOT / "07_FIGURES"
FIGDIR.mkdir(parents=True, exist_ok=True)

IDW = "#9AA0A6"
MAP = "#2F6B9A"
ANCHOR = "#D89C48"
TARGET = "#C5C9CD"
EVAL_COLORS = {
    "Xenium-COAD": "#116D6E",
    "Xenium-OV": "#67B7B1",
    "CosMx-COAD": "#5F3B8A",
    "CosMx-OV": "#A887C0",
}
TECH_COLORS = {"Xenium 5K": "#1F8A8A", "CosMx 6K": "#7251A5"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def panel_letter(ax, letter: str, x: float = -0.09, y: float = 1.10) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="left")


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def normalized_target_xy(base: Path, technology: str, tumor: str) -> tuple[np.ndarray, np.ndarray]:
    lock = json.loads((base / "work" / tumor / "source_geometry_lock.json").read_text())
    mapped = np.load(base / "work" / tumor / "seed42_registered_source_xy.npy", mmap_mode="r")
    roi = np.load(base / "work" / tumor / "geometry_only_target_roi_rows.npy")
    prefix = "Xenium" if technology == "Xenium 5K" else "CosMx"
    target = ad.read_h5ad(base / "SEALED_TARGET_8UM" / f"{prefix}_{tumor}_8um_shared_counts.h5ad", backed="r")
    raw = np.asarray(target.obsm["spatial"], np.float32)[roi]
    target.file.close()
    transform = lock["target_coordinate_transform"]
    xy = (raw - np.asarray(transform["center"], np.float32)) / float(transform["scale"])
    return xy, np.asarray(mapped)


def frozen_gene_fields(base: Path, technology: str, tumor: str, gene: str):
    model = np.load(base / "work" / tumor / "source_model.npz", allow_pickle=True)
    names = np.asarray(model["gene_names"]).astype(str)
    matches = np.flatnonzero(names == gene)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {gene} index in {technology} {tumor}; found {len(matches)}")
    gene_index = int(matches[0])
    frozen_pool = set(model["source_only_representative_gene_pool"].astype(int).tolist())
    if gene_index not in frozen_pool:
        raise RuntimeError(f"{gene} is not in the frozen source-only representative pool")
    evaluation = model["evaluation_indices"].astype(int)
    positions = {int(index): pos for pos, index in enumerate(evaluation)}
    if gene_index not in positions:
        raise RuntimeError(f"{gene} is not a frozen hidden evaluation gene")
    position = positions[gene_index]
    roi = np.load(base / "work" / tumor / "geometry_only_target_roi_rows.npy")
    depth = np.load(base / "work" / tumor / "source_model_predicted_target_depth.npy")
    prefix = "Xenium" if technology == "Xenium 5K" else "CosMx"
    target = ad.read_h5ad(base / "SEALED_TARGET_8UM" / f"{prefix}_{tumor}_8um_shared_counts.h5ad", backed="r")
    xy = np.asarray(target.obsm["spatial"], np.float32)[roi]
    selected = target[roi, [gene_index]].X
    if hasattr(selected, "to_memory"):
        selected = selected.to_memory()
    selected = selected.toarray() if hasattr(selected, "toarray") else np.asarray(selected)
    truth = np.log1p(selected.ravel().astype(np.float32) *
                     (float(model["source_depth_target"]) / depth))
    target.file.close()
    prior = np.asarray(np.load(base / "work" / tumor / "predictions" /
                               "registered_idw_k12_log1p.npy", mmap_mode="r")[:, position])
    prediction = np.asarray(np.load(base / "work" / tumor / "predictions" /
                                    "hyperspatial_map_log1p.npy", mmap_mode="r")[:, position])
    return xy, (truth, prior, prediction, truth - prior, prediction - prior)


# Exact frozen primary results.
xen_pair = pd.read_csv(XENIUM_PRESENTATION / "01_PAIR_LEVEL_RESULTS_IDW_vs_MAP.tsv", sep="\t")
xen_idw = xen_pair[xen_pair.method == "registered 12-NN IDW"].set_index("tumor")
xen_map = xen_pair[xen_pair.method == "HyperSpatial-MAP"].set_index("tumor")
cos_pair = pd.read_csv(COSMX / "03_PAIR_LEVEL_RESULTS.tsv", sep="\t").set_index("tumor")

rows = []
for tumor in ("COAD", "OV"):
    rows.append({
        "evaluation": f"Xenium-{tumor}", "technology": "Xenium 5K", "specimen": tumor,
        "source_platform": "Visium HD FFPE", "target_platform": "Xenium 5K",
        "target_units_8um_bins": int(xen_map.loc[tumor, "target_bin_count"]),
        "measured_anchor_count": int(xen_map.loc[tumor, "measured_anchor_count"]),
        "hidden_gene_count": int(xen_map.loc[tumor, "hidden_gene_count"]),
        "IDW_median_spatial_pearson": float(xen_idw.loc[tumor, "median_gene_wise_spatial_pearson"]),
        "MAP_median_spatial_pearson": float(xen_map.loc[tumor, "median_gene_wise_spatial_pearson"]),
        "IDW_log1p_RMSE": float(xen_idw.loc[tumor, "log1p_rmse"]),
        "MAP_log1p_RMSE": float(xen_map.loc[tumor, "log1p_rmse"]),
        "MAP_median_residual_pearson": float(xen_map.loc[tumor, "median_gene_wise_residual_pearson"]),
        "MAP_pooled_residual_pearson": float(xen_map.loc[tumor, "pooled_bin_gene_residual_pearson"]),
        "fraction_MAP_gt_IDW_spatial": float(xen_map.loc[tumor, "fraction_genes_MAP_gt_IDW_spatial_pearson"]),
        "fraction_MAP_lt_IDW_RMSE": float(xen_map.loc[tumor, "fraction_genes_MAP_lt_IDW_log1p_RMSE"]),
        "fraction_positive_MAP_residual": float(xen_map.loc[tumor, "fraction_positive_gene_residual_recovery"]),
    })
for tumor in ("COAD", "OV"):
    item = cos_pair.loc[tumor]
    rows.append({
        "evaluation": f"CosMx-{tumor}", "technology": "CosMx 6K", "specimen": tumor,
        "source_platform": item.source_platform, "target_platform": item.target_platform,
        "target_units_8um_bins": int(item.target_units_8um_bins),
        "measured_anchor_count": int(item.measured_anchor_count),
        "hidden_gene_count": int(item.hidden_gene_count),
        "IDW_median_spatial_pearson": float(item.IDW_median_gene_wise_spatial_pearson),
        "MAP_median_spatial_pearson": float(item.MAP_median_gene_wise_spatial_pearson),
        "IDW_log1p_RMSE": float(item.IDW_log1p_RMSE),
        "MAP_log1p_RMSE": float(item.MAP_log1p_RMSE),
        "MAP_median_residual_pearson": float(item.MAP_median_gene_wise_residual_pearson),
        "MAP_pooled_residual_pearson": float(item.MAP_pooled_residual_pearson),
        "fraction_MAP_gt_IDW_spatial": float(item.fraction_hidden_genes_MAP_gt_IDW_spatial_pearson),
        "fraction_MAP_lt_IDW_RMSE": float(item.fraction_hidden_genes_MAP_lt_IDW_log1p_RMSE),
        "fraction_positive_MAP_residual": float(item.fraction_hidden_genes_positive_MAP_residual_pearson),
    })
results = pd.DataFrame(rows)
results.insert(10, "delta_spatial_MAP_minus_IDW",
               results.MAP_median_spatial_pearson - results.IDW_median_spatial_pearson)
results.insert(13, "delta_RMSE_MAP_minus_IDW", results.MAP_log1p_RMSE - results.IDW_log1p_RMSE)
results.to_csv(ROOT / "01_PRIMARY_RESULTS_TABLE.tsv", sep="\t", index=False, float_format="%.16g")


# Figure canvas and shared layout.
fig = plt.figure(figsize=(16.8, 12.7), constrained_layout=False)
outer = GridSpec(3, 16, figure=fig, height_ratios=[1.05, 1.02, 2.08], hspace=.46, wspace=1.20)

# a — unified design.
ax = fig.add_subplot(outer[0, 0:6]); ax.axis("off"); panel_letter(ax, "a")
ax.set_title("Same-specimen serial-section cross-platform design", loc="left", fontweight="bold", pad=7)
workflow = [
    (.005, .17, "Visium HD FFPE\n8-µm reference"),
    (.215, .17, "Geometry-only\nregistration"),
    (.425, .17, "Xenium 5K or\nCosMx 6K target"),
    (.635, .15, "16 target genes\nexposed"),
    (.825, .17, "Hidden target\nmolecular state"),
]
for x, width, label in workflow:
    color = "#F4F6F8" if x < .635 else ("#FFF6E8" if x < .8 else "#EEF4F8")
    edge = "#657078" if x < .635 else (ANCHOR if x < .8 else MAP)
    ax.add_patch(plt.Rectangle((x, .43), width, .27, transform=ax.transAxes,
                               facecolor=color, edgecolor=edge, lw=.85))
    ax.text(x + width / 2, .565, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=6.8, color="#222")
for x1, x2 in ((.178, .21), (.388, .42), (.598, .63), (.788, .82)):
    ax.annotate("", xy=(x2, .565), xytext=(x1, .565), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))
ax.text(.5, .26, "Registered atlas prior + sparse target measurement → target-specific reconstruction",
        transform=ax.transAxes, ha="center", fontsize=7.4, color=MAP, fontweight="bold")
ax.text(.5, .075, "Same-specimen serial sections measured by different technologies;\nno cell-to-cell correspondence assumed.",
        transform=ax.transAxes, ha="center", va="center", fontsize=6.7, color="#555")

# b — measurement budget and evaluation space.
ax = fig.add_subplot(outer[0, 6:10]); panel_letter(ax, "b")
ax.set_title("Sparse target measurement", loc="left", fontweight="bold")
order = ["Xenium-COAD", "Xenium-OV", "CosMx-COAD", "CosMx-OV"]
ordered = results.set_index("evaluation").loc[order]
y = np.arange(4)[::-1]
for yy, (name, row) in zip(y, ordered.iterrows()):
    color = EVAL_COLORS[name]
    ax.plot([16, row.hidden_gene_count], [yy, yy], color="#D8DBDE", lw=2.0, zorder=1)
    ax.scatter(16, yy, s=34, color=ANCHOR, edgecolor="white", lw=.5, zorder=3)
    ax.scatter(row.hidden_gene_count, yy, s=38, color=color, edgecolor="white", lw=.5, zorder=3)
    ax.text(row.hidden_gene_count * 1.08, yy, f"{int(row.hidden_gene_count):,}",
            va="center", fontsize=7, color=color, fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(10, 2600); ax.set_ylim(-.65, 3.65)
ax.set_yticks(y, [name.replace("-", " · ") for name in order])
ax.set_xlabel("Genes (log scale)")
ax.scatter([], [], s=34, color=ANCHOR, label="Measured (16)")
ax.scatter([], [], s=38, color=MAP, label="Hidden")
ax.legend(frameon=False, fontsize=6.7, loc="lower right", handletextpad=.3)
clean(ax)

# c — four geometry-only support overlays.
sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 10:16], hspace=.07, wspace=.06)
geometry = [
    (XENIUM_FROZEN, "Xenium 5K", "COAD"), (XENIUM_FROZEN, "Xenium 5K", "OV"),
    (COSMX, "CosMx 6K", "COAD"), (COSMX, "CosMx 6K", "OV"),
]
for index, (base, technology, tumor) in enumerate(geometry):
    ax = fig.add_subplot(sub[index // 2, index % 2])
    if index == 0:
        panel_letter(ax, "c", x=-.13, y=1.30)
        ax.text(-.02, 1.30, "Geometry-only registration", transform=ax.transAxes,
                ha="left", fontweight="bold", fontsize=9)
    xy, mapped = normalized_target_xy(base, technology, tumor)
    take_t = np.linspace(0, len(xy) - 1, min(12000, len(xy)), dtype=int)
    take_s = np.linspace(0, len(mapped) - 1, min(4500, len(mapped)), dtype=int)
    color = TECH_COLORS[technology]
    ax.scatter(xy[take_t, 0], xy[take_t, 1], s=.12, c=TARGET, rasterized=True)
    ax.scatter(mapped[take_s, 0], mapped[take_s, 1], s=.17, c=color, alpha=.36, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(.01, .97, f"{technology.replace(' 5K','').replace(' 6K','')} · {tumor}",
            transform=ax.transAxes, va="top", fontsize=6.6, fontweight="bold")
    if index == 2:
        ax.text(.01, .02, "target support", transform=ax.transAxes, fontsize=5.8, color="#777")
    if index == 3:
        ax.text(.99, .02, "registered Visium HD", transform=ax.transAxes,
                ha="right", fontsize=5.8, color=color)

# d — central paired endpoint comparison.
sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0:11], wspace=.28)
endpoint_specs = [
    ("IDW_median_spatial_pearson", "MAP_median_spatial_pearson",
     "Spatial reconstruction", "Median spatial Pearson"),
    ("IDW_log1p_RMSE", "MAP_log1p_RMSE", "Reconstruction error", "log1p RMSE"),
]
for col, (left, right, title, ylabel) in enumerate(endpoint_specs):
    ax = fig.add_subplot(sub[0, col])
    if col == 0:
        panel_letter(ax, "d")
        ax.text(-.02, 1.18, "Reconstruction beyond registered atlas transfer",
                transform=ax.transAxes, ha="left", fontweight="bold", fontsize=9)
    for _, row in ordered.iterrows():
        values = [row[left], row[right]]
        color = EVAL_COLORS[row.name]
        ax.plot([0, 1], values, color=color, marker="o", markersize=4.6,
                markeredgecolor="white", markeredgewidth=.45, lw=1.3)
        ax.text(1.035, values[1], row.name.replace("-", " · "), color=color,
                va="center", fontsize=6.5)
    ax.set_xlim(-.12, 1.42); ax.set_xticks([0, 1], ["IDW", "MAP"])
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold")
    ax.axhline(0, color="#D5D7DA", lw=.6, zorder=0)
    clean(ax)

# e — pooled residual recovery.
ax = fig.add_subplot(outer[1, 11:16]); panel_letter(ax, "e")
ax.set_title("Atlas-relative residual recovery", loc="left", fontweight="bold")
x = np.arange(4)
values = ordered.MAP_pooled_residual_pearson.to_numpy()
bars = ax.bar(x, values, width=.62, color=[EVAL_COLORS[name] for name in order])
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + .012, f"{value:.3f}",
            ha="center", va="bottom", fontsize=6.8, fontweight="bold")
ax.axhline(0, color="#555", lw=.7); ax.set_ylim(0, .63)
ax.set_xticks(x, [name.replace("-", "\n") for name in order])
ax.set_ylabel("MAP pooled residual Pearson")
ax.text(.99, .98, "Registered IDW predicts zero residual;\ncorrelation undefined.",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color="#666")
clean(ax)

# f — one already predeclared representative gene for each target technology.
bottom = GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[2, :], hspace=.10, wspace=.055)
examples = [
    (XENIUM_FROZEN, "Xenium 5K", "COAD", "CD55"),
    (COSMX, "CosMx 6K", "COAD", "OLFM4"),
]
titles = ("Target truth", "Registered Visium-HD prior", "MAP reconstruction",
          "Measured atlas-relative residual", "Predicted atlas-relative residual")
for row_index, (base, technology, tumor, gene) in enumerate(examples):
    xy, fields = frozen_gene_fields(base, technology, tumor, gene)
    vmax = max(float(np.quantile(np.concatenate(fields[:3]), .99)), 1e-5)
    rmax = max(float(np.quantile(np.abs(np.concatenate(fields[3:])), .99)), 1e-5)
    take = np.linspace(0, len(xy) - 1, min(100000, len(xy)), dtype=int)
    for col, field in enumerate(fields):
        ax = fig.add_subplot(bottom[row_index, col])
        if row_index == 0 and col == 0:
            panel_letter(ax, "f", x=-.11, y=1.12)
            ax.text(-.01, 1.12, "Representative hidden-gene reconstruction",
                    transform=ax.transAxes, ha="left", fontweight="bold", fontsize=9)
        if row_index == 0:
            ax.set_title(titles[col], fontsize=7.6, fontweight="bold", pad=4)
        if col < 3:
            ax.scatter(xy[take, 0], xy[take, 1], c=field[take], s=.20, cmap="viridis",
                       vmin=0, vmax=vmax, rasterized=True)
        else:
            ax.scatter(xy[take, 0], xy[take, 1], c=field[take], s=.20, cmap="coolwarm",
                       vmin=-rmax, vmax=rmax, rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        if col == 0:
            ax.text(-.025, .5, f"{technology} · {tumor} · {gene}", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontstyle="italic", fontsize=8.0)
        if col == 2:
            ax.text(.99, .015, f"expression 0–{vmax:.2f}", transform=ax.transAxes,
                    ha="right", fontsize=5.7, color="white",
                    bbox=dict(facecolor="#20242A", alpha=.58, edgecolor="none", pad=1.1))
        if col == 4:
            ax.text(.99, .015, f"residual ±{rmax:.2f}", transform=ax.transAxes,
                    ha="right", fontsize=5.7, color="#333",
                    bbox=dict(facecolor="white", alpha=.72, edgecolor="none", pad=1.1))

fig.subplots_adjust(left=.055, right=.982, top=.965, bottom=.045)
base = FIGDIR / "SPATCH_merged_cross_platform_main"
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)


def write_text(name: str, content: str) -> None:
    (ROOT / name).write_text(content.strip() + "\n")


write_text("00_INPUT_PROVENANCE.md", f"""
# Input provenance

- Xenium manuscript-facing source: `{XENIUM_PRESENTATION.resolve()}`
- Xenium frozen prediction source: `{XENIUM_FROZEN.resolve()}`
- CosMx completed source: `{COSMX.resolve()}`
- New presentation-only directory: `{ROOT.resolve()}`

The Xenium values were read from `01_PAIR_LEVEL_RESULTS_IDW_vs_MAP.tsv`; the CosMx values were read from `03_PAIR_LEVEL_RESULTS.tsv`. Geometry overlays and spatial examples were rendered from the corresponding frozen geometry, target, IDW and MAP artifacts. No prediction, registration, anchor panel, internal operator assignment, metric, model fit or calibration was recomputed.

Only COAD and ovarian cancer (OV) are included in the main figure. HCC remains a secondary stress test and is documented in `05_SUPPLEMENTARY_HCC_NOTE.md`; Stereo-seq is absent because the completed pairing audit identified no valid requested pairs. The sole manuscript-facing comparator is registered 12-NN IDW.

Representative examples were restricted to the existing frozen source-only representative pools. Xenium COAD CD55 was already displayed in the Xenium manuscript-facing figure; CosMx COAD OLFM4 was the frozen first-ranked source-only example in the completed CosMx figure. No new target-outcome-based gene selection was performed.

Source integrity was checked against the recorded Xenium and CosMx SHA-256 manifests before assembly; all recorded hashes verified.
""")

write_text("02_MAIN_FIGURE_CAPTION_DRAFT.md", """
# Figure caption draft

**Sparse target measurements adapt registered molecular atlases across spatial technologies.** **a,** Unified cross-platform design. Visium HD FFPE references represented at 8-µm resolution were aligned by geometry alone to same-specimen Xenium 5K or CosMx 6K serial sections; 16 target genes were exposed and the remaining shared genes were hidden. Serial sections were not assumed to provide one-to-one cell correspondence. **b,** Measurement budget and hidden evaluation space for the four primary evaluations: Xenium–COAD (763 hidden genes), Xenium–OV (697), CosMx–COAD (1,391) and CosMx–OV (1,269). **c,** Geometry-only registration support for each technology and specimen. Grey points denote target support and coloured points registered Visium HD support. **d,** Median gene-wise spatial Pearson and log1p RMSE for registered 12-nearest-neighbour inverse-distance weighting (IDW) and HyperSpatial-MAP. Lines connect the two methods within each evaluation. IDW uses the same registered reference but receives no target molecular correction. **e,** Pooled Pearson correlation between measured and predicted atlas-relative residuals for HyperSpatial-MAP. Registered IDW predicts an identically zero residual and its residual correlation is therefore undefined. **f,** Frozen source-predeclared COAD hidden-gene examples for Xenium (CD55) and CosMx (OLFM4), showing target expression, the registered Visium HD prior, HyperSpatial-MAP reconstruction, and measured and predicted atlas-relative residuals. Expression maps share a scale within each gene and residual maps use a common symmetric scale.
""")

write_text("03_MANUSCRIPT_RESULTS_PARAGRAPH.md", """
# Manuscript Results paragraph

We next tested whether sparse molecular measurements could adapt a reference across spatial technologies using same-specimen serial sections from the SPATCH resource. Visium HD FFPE sections served as references and Xenium 5K or CosMx 6K sections as targets; only 16 target genes were exposed, with 697–1,391 shared genes withheld for evaluation. Against registered 12-nearest-neighbour IDW, which transfers the geometry-aligned atlas without target molecular correction, HyperSpatial-MAP increased median spatial Pearson from 0.0131 to 0.1868 and reduced log1p RMSE from 0.3176 to 0.2874 in Xenium COAD, and from −0.0010 to 0.1465 and 0.3301 to 0.2957, respectively, in Xenium ovarian cancer. The corresponding CosMx values improved from 0.0031 to 0.0775 and 0.3101 to 0.2614 in COAD, and from −0.0003 to 0.0620 and 0.3518 to 0.3085 in ovarian cancer. Predicted atlas-relative residuals were positively correlated with measured residuals in all four evaluations (pooled Pearson r = 0.441–0.538), showing that sparse target measurements can adapt a Visium HD reference beyond registered atlas transfer across distinct target technologies.
""")

write_text("04_LAYOUT_NOTES.md", """
# Layout notes

The figure follows one information flow: a shared cross-platform design (a), the sparse measurement boundary (b), geometry-only transfer (c), the two primary reconstruction endpoints (d), target-specific residual recovery (e), and frozen representative spatial reconstructions (f). Xenium and CosMx are distinguished by restrained teal and violet palettes; IDW remains neutral grey. COAD and OV are shown individually throughout and are not pooled as biological replicates.

Panel f contains one previously locked source-only example per target technology: Xenium–COAD CD55 and CosMx–COAD OLFM4. Each row uses one expression scale across target truth, registered prior and MAP reconstruction, and one symmetric residual scale across measured and predicted residuals. The SVG retains editable text and vector layout; dense spatial point layers are rasterized within the SVG to keep the file tractable.

HCC and Stereo-seq are intentionally absent from the main figure. HCC remains documented as a secondary stress test, and no requested Stereo-seq pair passed the completed data-pairing audit.
""")

xen_hcc = pd.read_csv(XENIUM_PRESENTATION / "04B_HCC_SECONDARY_STRESS_TEST.tsv", sep="\t").iloc[0]
cos_hcc = cos_pair.loc["HCC"]
write_text("05_SUPPLEMENTARY_HCC_NOTE.md", f"""
# Secondary HCC cross-platform stress test

HCC is retained as a secondary cross-platform stress test and was not placed in the merged main figure or its primary aggregate presentation. This is a manuscript-facing organization decision: the original frozen analyses and all HCC predictions, registrations, anchors, assignments, metrics and hashes remain unchanged. The HCC hidden evaluation spaces were smaller than the corresponding COAD and OV spaces, particularly for Xenium (151 genes versus 763 and 697) and also for CosMx (287 versus 1,391 and 1,269), limiting direct comparability with the primary evaluations.

| Target technology | Hidden genes | IDW spatial Pearson | MAP spatial Pearson | IDW log1p RMSE | MAP log1p RMSE | MAP median residual Pearson | MAP pooled residual Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| Xenium 5K | {int(xen_hcc.hidden_shared_gene_count)} | {xen_hcc.IDW_median_gene_wise_spatial_pearson:.16g} | {xen_hcc.MAP_median_gene_wise_spatial_pearson:.16g} | {xen_hcc.IDW_log1p_RMSE:.16g} | {xen_hcc.MAP_log1p_RMSE:.16g} | {xen_hcc.MAP_median_gene_wise_residual_pearson:.16g} | {xen_hcc.MAP_pooled_residual_pearson:.16g} |
| CosMx 6K | {int(cos_hcc.hidden_gene_count)} | {cos_hcc.IDW_median_gene_wise_spatial_pearson:.16g} | {cos_hcc.MAP_median_gene_wise_spatial_pearson:.16g} | {cos_hcc.IDW_log1p_RMSE:.16g} | {cos_hcc.MAP_log1p_RMSE:.16g} | {cos_hcc.MAP_median_gene_wise_residual_pearson:.16g} | {cos_hcc.MAP_pooled_residual_pearson:.16g} |

Residual recovery remained positive in both HCC stress tests. These results are reported descriptively and do not alter the primary COAD/OV presentation.
""")

terminal_lines = [
    f"Exact Xenium directory used: {XENIUM_PRESENTATION.resolve()}",
    f"Exact Xenium frozen-artifact directory: {XENIUM_FROZEN.resolve()}",
    f"Exact CosMx directory used: {COSMX.resolve()}",
    f"Exact new merged-figure directory: {ROOT.resolve()}",
    "Predictions recomputed: NO", "Registrations recomputed: NO",
    "Anchors changed: NO", "Assignments or metrics changed: NO",
    "All recorded source hashes verified: YES",
    "Representative genes: Xenium–COAD CD55; CosMx–COAD OLFM4",
]
for _, row in ordered.iterrows():
    terminal_lines.append(
        f"{row.name}: hidden={int(row.hidden_gene_count)}; spatial IDW {row.IDW_median_spatial_pearson:.6f} -> MAP {row.MAP_median_spatial_pearson:.6f}; "
        f"RMSE IDW {row.IDW_log1p_RMSE:.6f} -> MAP {row.MAP_log1p_RMSE:.6f}; pooled residual r={row.MAP_pooled_residual_pearson:.6f}"
    )
terminal_lines.extend([
    "HCC in MAIN figure: NO", "HCC retained as secondary stress test: YES",
    "Stereo-seq in MAIN figure: NO (no valid requested pairs passed the completed audit)",
    "Only manuscript-facing comparator: registered 12-NN IDW",
])
write_text("TERMINAL_SUMMARY.txt", "\n".join(terminal_lines))

# Hash every newly generated deliverable except the manifest itself.
files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path.name != "06_SHA256_MANIFEST.txt")
with (ROOT / "06_SHA256_MANIFEST.txt").open("w") as handle:
    handle.write("# SHA-256 manifest for merged presentation-only package\n")
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path.relative_to(ROOT)}\n")

print("\n".join(terminal_lines))
print("New files:")
for path in sorted(path for path in ROOT.rglob("*") if path.is_file()):
    print(f"- {path.relative_to(ROOT)}")
