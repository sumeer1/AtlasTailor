#!/usr/bin/env python3
"""Render the HyperSpatial supplementary figure suite from locked results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
import numpy as np

from generate_hyperspatial_nature_figures import (
    ANALYSIS,
    COLORS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    OUT,
    ROOT,
    clean,
    panel,
    projection,
    read_tsv,
    scatter_field,
    style,
)


SUPP = OUT / "supplementary_figures"
SUPP.mkdir(parents=True, exist_ok=True)
MARKERS = ("noto", "tbxta", "shha")
PANEL_COLUMNS = [294, 430, 374]


def save_supp(fig, stem):
    fig.savefig(SUPP / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(SUPP / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def marker_bundle():
    data = np.load(ANALYSIS / "six_marker_maps.npz")
    genes = data["genes"].astype(str)
    indices = [int(np.where(genes == gene)[0][0]) for gene in MARKERS]
    return data, genes, indices


def spatial_r(observed, predicted):
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 3 or np.std(predicted[valid]) == 0:
        return np.nan
    return float(np.corrcoef(observed[valid], predicted[valid])[0, 1])


def supplementary_1_additional_marker_atlas():
    data = np.load(ANALYSIS / "six_marker_maps.npz")
    metrics = read_tsv(ANALYSIS / "six_marker_metric_table.tsv")
    all_genes = data["genes"].astype(str)
    indices = np.asarray(
        [index for index, gene in enumerate(all_genes) if gene not in MARKERS]
    )
    genes = all_genes[indices]
    observed = data["observed_log"][:, indices]
    predicted = data["predicted_log"][:, indices]
    metrics = [metrics[index] for index in indices]
    uv = projection(data["coordinates"])
    fig = plt.figure(figsize=(7.2, 4.65))
    gs = gridspec.GridSpec(
        3,
        len(genes),
        figure=fig,
        height_ratios=[1, 1, 0.48],
        hspace=0.10,
        wspace=0.025,
    )
    for col, gene in enumerate(genes):
        vmax = np.quantile(observed[:, col], 0.995)
        for row, (label, values) in enumerate((("Observed", observed), ("HyperSpatial", predicted))):
            ax = fig.add_subplot(gs[row, col])
            if row == 0 and col == 0:
                panel(ax, "a", -0.16, 1.16)
            if row == 0:
                ax.set_title(r"$\it{" + gene + "}$")
            scatter_field(ax, uv, values[:, col], vmax, seed=70 + col)
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    label,
                    rotation=90,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    weight="bold",
                    color=COLORS["hyper"] if row else COLORS["ink"],
                )
            if row:
                ax.text(
                    0.03,
                    0.05,
                    f"r = {float(metrics[col]['spatial_pearson']):.2f}\nDice = {float(metrics[col]['topk_dice']):.2f}",
                    transform=ax.transAxes,
                    fontsize=5.5,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.80, "pad": 1},
                )
        cax = fig.axes[-2].inset_axes([0.20, -0.02, 0.60, 0.027])
        plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap="magma"), cax=cax, orientation="horizontal")
        cax.set_xticks([0, vmax], ["0", "max"])
        cax.tick_params(labelsize=4.8, length=1, pad=1)

    ax = fig.add_subplot(gs[2, :])
    panel(ax, "b", -0.02, 1.14)
    x = np.arange(len(genes))
    specs = [
        ("Spatial Pearson", "spatial_pearson", COLORS["geometry"]),
        ("Top-k Dice", "topk_dice", COLORS["hyper"]),
        ("AUPRC / prevalence", "auprc_enrichment", COLORS["gold"]),
    ]
    for row, (label, key, color) in enumerate(specs):
        values = np.asarray([float(record[key]) for record in metrics])
        sizes = values / max(values.max(), 1) if "AUPRC" in label else values
        ax.plot(x, np.full(len(x), row), color=COLORS["light"], lw=1.5)
        ax.scatter(x, np.full(len(x), row), s=18 + 48 * np.clip(sizes, 0, 1), color=color)
        for col, value in enumerate(values):
            ax.text(col, row + 0.22, f"{value:.2f}", ha="center", fontsize=5.4)
    ax.set_yticks(range(3), [x[0] for x in specs])
    ax.set_xticks(x, genes, fontstyle="italic")
    ax.set_ylim(-0.45, 2.55)
    ax.invert_yaxis()
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    save_supp(fig, "Supplementary_Figure_1_additional_marker_atlas")


def supplementary_2_component_maps():
    data, genes, selected = marker_bundle()
    observed = data["observed_log"][:, selected]
    uv = projection(data["coordinates"])
    run = ROOT / "E1_to_E2_seed42/predictions"
    rows = []
    for method, color in zip(METHOD_ORDER[:-1], METHOD_COLORS[:-1]):
        values = np.load(run / f"{method}.npy", mmap_mode="r")[:, PANEL_COLUMNS]
        rows.append((METHOD_LABELS[method], values, color))
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = gridspec.GridSpec(len(rows), 3, figure=fig, hspace=0.05, wspace=0.025)
    vmaxes = [np.quantile(observed[:, col], 0.995) for col in range(3)]
    for row, (label, values, color) in enumerate(rows):
        for col, gene in enumerate(MARKERS):
            ax = fig.add_subplot(gs[row, col])
            if row == 0 and col == 0:
                panel(ax, "a", -0.13, 1.16)
            if row == 0:
                ax.set_title(r"$\it{" + gene + "}$")
            scatter_field(ax, uv, values[:, col], vmaxes[col], seed=90 + col)
            if col == 0:
                ax.text(-0.06, 0.5, label, rotation=90, transform=ax.transAxes, ha="right", va="center", color=color, weight="bold" if row in (0, 4) else "normal")
            ax.text(
                0.04,
                0.05,
                f"r = {spatial_r(observed[:, col], values[:, col]):.2f}",
                transform=ax.transAxes,
                fontsize=5.8,
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1},
            )
    save_supp(fig, "Supplementary_Figure_2_component_spatial_ablation")


def supplementary_3_endpoint_landscape():
    external_rows = read_tsv(Path("results_v10/wemerfish_3stage/E2_locked_confirmation/final_comparison.tsv"))
    external = {row["method"]: row for row in external_rows}
    hyper = {
        row["method"]: row
        for row in read_tsv(ROOT / "E1_to_E2_seed42/metric_table.tsv")
    }["hyperspatial_local_attention_rank64"]
    rows = [
        ("HyperSpatial", hyper, COLORS["hyper"]),
        ("GP / Kriging", external["anchor_gp_kriging"], COLORS["gold"]),
        ("Spateo-adapted", external["spateo_adapted_geometry_transport"], COLORS["purple"]),
        ("Coordinate–time field", external["v9_coordinate_backbone"], COLORS["muted"]),
        ("Temporal IDW", external["rigid_temporal_idw"], COLORS["transport"]),
    ]
    specs = [
        ("Spatial\nPearson", "svg_spatial_pearson_median", True),
        ("AUPRC /\nprevalence", "auprc_enrichment_median", True),
        ("Top-k\nDice", "topk_dice_median", True),
        ("Centroid\nerror", "weighted_centroid_error_median", False),
        ("Spatial\nEMD", "spatial_emd_median", False),
        ("log1p\nRMSE", "log1p_rmse", False),
        ("Moran\nerror", "moran_error_median", False),
    ]
    matrix = np.asarray([[float(row[key]) for _, key, _ in specs] for _, row, _ in rows])
    score = np.zeros_like(matrix)
    for col, (_, _, higher) in enumerate(specs):
        lo, hi = matrix[:, col].min(), matrix[:, col].max()
        normalized = (matrix[:, col] - lo) / max(hi - lo, 1e-12)
        score[:, col] = normalized if higher else 1 - normalized
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1], hspace=0.52)
    ax = fig.add_subplot(gs[0])
    panel(ax, "a", -0.11, 1.12)
    image = ax.imshow(score, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)), [x[0] for x in rows])
    ax.set_xticks(range(len(specs)), [x[0] for x in specs])
    for row in range(len(rows)):
        for col in range(len(specs)):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if score[row, col] < 0.35 else COLORS["ink"])
    cax = ax.inset_axes([1.02, 0.08, 0.018, 0.84])
    plt.colorbar(image, cax=cax, label="within-endpoint score")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    ax = fig.add_subplot(gs[1])
    panel(ax, "b", -0.11, 1.14)
    x = np.arange(len(specs))
    for row, (label, _, color) in enumerate(rows):
        ax.plot(x, score[row], "-o", lw=1.15, ms=3.2, color=color, label=label)
    ax.set_xticks(x, [x[0].replace("\n", " ") for x in specs], rotation=22, ha="right")
    ax.set_ylabel("normalized performance")
    ax.set_ylim(-0.04, 1.05)
    clean(ax)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.50), fontsize=5.7)
    save_supp(fig, "Supplementary_Figure_3_corrected_endpoint_landscape")


def supplementary_4_official_spateo():
    alignment = np.load(ROOT / "official_spateo/predictions/official_spateo_alignment.npz", allow_pickle=True)
    reconstruction = np.load(ROOT / "official_spateo/predictions/official_spateo_gp_reconstruction.npz", allow_pickle=True)
    results = json.loads((ROOT / "official_spateo/results.json").read_text())
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1.12], hspace=0.36, wspace=0.24)
    target_centered = alignment["target_coordinates"] - alignment["target_coordinates"].mean(0)
    target_basis = np.linalg.svd(target_centered, full_matrices=False)[2][:2].T
    target_uv = target_centered @ target_basis
    raw_source = (alignment["source_coordinates"] - alignment["target_coordinates"].mean(0)) @ target_basis
    aligned_centered = alignment["aligned_coordinates"] - alignment["target_coordinates"].mean(0)
    aligned_uv = aligned_centered @ target_basis
    for col, (title, source_uv) in enumerate((("Before Morpho", raw_source), ("After official Morpho", aligned_uv))):
        ax = fig.add_subplot(gs[0, col])
        if col == 0:
            panel(ax, "a", -0.12, 1.16)
        ax.scatter(target_uv[:, 0], target_uv[:, 1], s=2, color=COLORS["e2e1"], alpha=0.40, linewidths=0, label="target")
        ax.scatter(source_uv[:, 0], source_uv[:, 1], s=2, color=COLORS["e1e2"], alpha=0.40, linewidths=0, label="source")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        if col == 0:
            ax.legend(frameon=False, markerscale=2.5, fontsize=5.8, loc="lower left")
    ax = fig.add_subplot(gs[0, 2:])
    panel(ax, "b", -0.06, 1.16)
    ax.axis("off")
    ax.text(0.02, 0.80, "Official Spateo", fontsize=9, weight="bold", color=COLORS["purple"])
    ax.text(0.02, 0.58, "Alignment", color=COLORS["muted"])
    ax.text(0.58, 0.58, f"{results['official_alignment']['runtime_seconds']:.2f} s")
    ax.text(0.02, 0.42, "Post-alignment Chamfer", color=COLORS["muted"])
    ax.text(0.58, 0.42, f"{results['official_alignment']['geometry_only_chamfer_after_alignment']:.4f}")
    ax.text(0.02, 0.20, "Expression is used by official alignment;\nthis is not HyperSpatial's held-out-expression task.", fontsize=6.2)

    observed = reconstruction["observed"]
    predicted = reconstruction["predicted"]
    genes = reconstruction["genes"].astype(str)
    correlations = np.asarray([spatial_r(observed[:, j], predicted[:, j]) for j in range(observed.shape[1])])
    selected = np.argsort(np.nan_to_num(correlations, nan=-2))[-3:][::-1]
    uv = projection(reconstruction["coordinates"])
    for col, index in enumerate(selected):
        outer = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, col], wspace=0.02)
        for subcol, (label, values) in enumerate((("Observed", observed[:, index]), ("SVGP", predicted[:, index]))):
            ax = fig.add_subplot(outer[0, subcol])
            if col == 0 and subcol == 0:
                panel(ax, "c", -0.20, 1.22)
            vmax = np.quantile(observed[:, index], 0.99)
            scatter_field(ax, uv, values, vmax, seed=130 + col)
            ax.set_title((r"$\it{" + genes[index] + "}$\n" if subcol == 0 else "") + label, fontsize=6.3)
        fig.axes[-1].text(0.50, -0.09, f"r = {correlations[index]:.2f}", transform=fig.axes[-1].transAxes, ha="center", fontsize=5.5)
    ax = fig.add_subplot(gs[1, 3])
    panel(ax, "d", -0.08, 1.16)
    ax.scatter(correlations, np.arange(len(correlations)), s=9, color=COLORS["purple"], alpha=0.7)
    ax.axvline(np.nanmedian(correlations), color=COLORS["ink"], lw=1.0)
    ax.set_xlabel("within-embryo spatial Pearson")
    ax.set_ylabel("32 genes")
    ax.set_yticks([])
    ax.set_title("Official SVGP reconstruction")
    clean(ax)
    ax.text(0.04, 0.04, f"median = {np.nanmedian(correlations):.2f}\nRMSE = {results['official_reconstruction']['log1p_rmse']:.2f}", transform=ax.transAxes, fontsize=6)
    save_supp(fig, "Supplementary_Figure_4_official_Spateo_audit")


def supplementary_5_missing_slabs():
    data, _, selected = marker_bundle()
    observed = data["observed_log"][:, selected]
    uv = projection(data["coordinates"])
    full = np.load(ROOT / "E1_to_E2_seed42/predictions/hyperspatial_local_attention_rank64.npy", mmap_mode="r")[:, PANEL_COLUMNS]
    rows = [("Complete sources", full)]
    for fraction in (10, 20, 30):
        rows.append((f"{fraction}% z slab removed", np.load(ANALYSIS / f"predictions/missing_z_{fraction}pct.npy", mmap_mode="r")[:, PANEL_COLUMNS]))
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.04, wspace=0.025)
    vmaxes = [np.quantile(observed[:, col], 0.995) for col in range(3)]
    for row, (label, values) in enumerate(rows):
        for col, gene in enumerate(MARKERS):
            ax = fig.add_subplot(gs[row, col])
            if row == 0 and col == 0:
                panel(ax, "a", -0.13, 1.16)
            if row == 0:
                ax.set_title(r"$\it{" + gene + "}$")
            scatter_field(ax, uv, values[:, col], vmaxes[col], seed=170 + col)
            if col == 0:
                ax.text(-0.06, 0.5, label, rotation=90, transform=ax.transAxes, ha="right", va="center", color=COLORS["muted"] if row else COLORS["ink"], weight="bold" if row == 0 else "normal")
            change = np.mean(np.abs(values[:, col] - full[:, col]))
            if row:
                ax.text(0.04, 0.05, f"mean |Δ| = {change:.2f}", transform=ax.transAxes, fontsize=5.4, bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1})
    save_supp(fig, "Supplementary_Figure_5_missing_slab_marker_fields")


def supplementary_6_efficiency_and_stability():
    runtime = read_tsv(ANALYSIS / "runtime_memory_table.tsv")
    records = read_tsv(ROOT / "per_run_metrics.tsv")
    full = [row for row in records if row["method"] == "hyperspatial_local_attention_rank64"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    for index, ax in enumerate(axes.flat):
        panel(ax, chr(ord("a") + index), -0.12, 1.16)
    colors = [COLORS["e1e2"] if row["direction"] == "E1_to_E2" else COLORS["e2e1"] for row in runtime]
    times = np.asarray([float(row["prediction_seconds"]) for row in runtime])
    throughput = np.asarray([float(row["query_throughput_spots_per_second"]) for row in runtime])
    axes[0, 0].scatter(times, throughput, s=28, c=colors, edgecolors="white", lw=0.4)
    axes[0, 0].set_xlabel("prediction time (s)")
    axes[0, 0].set_ylabel("queries s$^{-1}$")
    axes[0, 0].set_title("Amortized query performance")
    clean(axes[0, 0])

    gpu_mb = np.asarray([float(row["peak_gpu_memory_bytes"]) / 2**20 for row in runtime])
    checkpoint_mb = np.asarray([float(row["checkpoint_bytes"]) / 2**20 for row in runtime])
    axes[0, 1].scatter(gpu_mb, checkpoint_mb, s=30, c=colors, edgecolors="white", lw=0.4)
    axes[0, 1].set_xlabel("peak GPU memory (MiB)")
    axes[0, 1].set_ylabel("serialized model (MiB)")
    axes[0, 1].set_title("Compact deployment footprint")
    clean(axes[0, 1])

    specs = [
        ("svg_spatial_pearson_median", "Spatial Pearson"),
        ("topk_dice_median", "Top-k Dice"),
        ("log1p_rmse", "log1p RMSE"),
    ]
    ax = axes[1, 0]
    for col, (key, label) in enumerate(specs):
        all_values = np.asarray([float(row[key]) for row in full])
        reference = all_values.mean()
        for seed in ("42", "43", "44"):
            pair = [row for row in full if row["seed"] == seed]
            pair.sort(key=lambda row: row["direction"])
            values = [float(row[key]) / reference for row in pair]
            x = np.asarray([col - 0.10, col + 0.10])
            ax.plot(x, values, color=COLORS["light"], lw=1)
            ax.scatter(x, values, s=18, color=[COLORS["e1e2"], COLORS["e2e1"]], zorder=2)
    axes[1, 0].set_xticks(range(3), [x[1] for x in specs])
    axes[1, 0].axhline(1, color=COLORS["muted"], lw=0.7, ls=(0, (2, 2)))
    axes[1, 0].set_ylabel("value / six-run mean")
    axes[1, 0].set_title("Bidirectional three-seed stability")
    clean(axes[1, 0])

    ax = axes[1, 1]
    directions = ("E1_to_E2", "E2_to_E1")
    for index, direction in enumerate(directions):
        subset = [row for row in full if row["direction"] == direction]
        y = np.asarray([float(row["svg_spatial_pearson_median"]) for row in subset])
        x = np.asarray([float(row["log1p_rmse"]) for row in subset])
        ax.scatter(x, y, s=30, color=COLORS["e1e2"] if index == 0 else COLORS["e2e1"], label=direction.replace("_to_", " → "))
        for row, xx, yy in zip(subset, x, y):
            ax.text(xx, yy, row["seed"], fontsize=5.2, ha="center", va="bottom")
    ax.set_xlabel("log1p RMSE")
    ax.set_ylabel("spatial Pearson")
    ax.set_title("Accuracy trade-off")
    clean(ax)
    ax.legend(frameon=False, fontsize=5.8)
    save_supp(fig, "Supplementary_Figure_6_efficiency_and_seed_stability")


def captions_and_manifest():
    captions = """Supplementary Figure 1 | Six-marker molecular atlas.
Observed and HyperSpatial-reconstructed expression for the six predefined markers, with marker-wise spatial Pearson, prevalence-matched Dice and AUPRC enrichment.

Supplementary Figure 2 | Spatial component ablation.
Matched maps for noto, tbxta and shha after local transport, diffeomorphic registration, local attention and the complete rank-64 residual operator. All rows use the observed-data colour limit for each gene.

Supplementary Figure 3 | Corrected endpoint landscape.
Exact held-out-embryo metrics for HyperSpatial and matched baselines. Heat-map colours are normalized within each endpoint solely for visualization; printed values are the untransformed metrics.

Supplementary Figure 4 | Official Spateo audit.
Official Morpho alignment and official SVGP within-embryo reconstruction. These use Spateo's supported settings and are explicitly separate from the held-out-embryo molecular-prediction task.

Supplementary Figure 5 | Marker-level missing-slab robustness.
Predicted noto, tbxta and shha fields after removing contiguous source z slabs. Colour limits are fixed across removal levels.

Supplementary Figure 6 | Efficiency and seed stability.
Prediction time, throughput, GPU footprint, serialized size and bidirectional seed-level endpoint stability for the frozen operator.
"""
    (SUPP / "supplementary_figure_captions.txt").write_text(captions)
    stems = [
        "Supplementary_Figure_1_six_marker_atlas",
        "Supplementary_Figure_2_component_spatial_ablation",
        "Supplementary_Figure_3_corrected_endpoint_landscape",
        "Supplementary_Figure_4_official_Spateo_audit",
        "Supplementary_Figure_5_missing_slab_marker_fields",
        "Supplementary_Figure_6_efficiency_and_seed_stability",
    ]
    manifest = {
        "figures": stems,
        "formats": ["vector PDF", "350 dpi PNG"],
        "shared_map_scales": True,
        "main_figure_2_markers": list(MARKERS),
        "task_honesty": "Official Spateo panels remain labelled as distinct supported tasks.",
    }
    (SUPP / "supplementary_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    style()
    supplementary_1_additional_marker_atlas()
    supplementary_2_component_maps()
    supplementary_3_endpoint_landscape()
    supplementary_4_official_spateo()
    supplementary_5_missing_slabs()
    supplementary_6_efficiency_and_stability()
    captions_and_manifest()


if __name__ == "__main__":
    main()
