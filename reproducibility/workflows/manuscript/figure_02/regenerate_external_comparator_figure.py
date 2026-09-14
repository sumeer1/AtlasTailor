#!/usr/bin/env python3
"""Figure-only integration of locked TransImpSpa metrics into the frozen gimVI figure.

No baseline result is modified. The script reads frozen gimVI source tables and
locked TransImpSpa predictions, recalculates only TransImpSpa evaluation metrics
with the original figure's metric implementation, and writes to a new directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from run_wemerfish_temporal_holdout import (  # noqa: E402
    canonicalize, evaluate, measured_counts, normalize_counts, pearson_columns,
)

OUT = REPO / "PI_revision/02C_gimvi_transimpspa_figure_revision_20260817_154438"
GIMVI = REPO / "results_v10/hyperspatial_map_gimvi_baseline_20260805_095807"
TRANSIMP = REPO / "PI_revision/02B_spatial_reconstruction_comparator_20260817_132013"
FROZEN = REPO / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726"
DATA = REPO / "data_v10/wemerfish_zebrafish_3stage"
SEED = 42
TRANSIMP_EVAL_SEED = SEED + 7  # appended after the seven original method identities

METHODS = (
    "registered_nearest_neighbor", "registered_idw_k12", "anchor_only_calibrated",
    "gimvi_adapted", "atlas_plus_global_anchor_residual",
    "atlas_plus_local_anchor_residual", "hyperspatial_map_calibrated_fusion",
)
PLOT_METHODS = (
    "registered_nearest_neighbor", "gimvi_adapted", "atlas_plus_global_anchor_residual",
    "atlas_plus_local_anchor_residual", "hyperspatial_map_calibrated_fusion", "transimpspa",
)
LEGEND_METHODS = (
    "registered_nearest_neighbor", "registered_idw_k12", "gimvi_adapted",
    "atlas_plus_global_anchor_residual", "atlas_plus_local_anchor_residual",
    "hyperspatial_map_calibrated_fusion", "transimpspa",
)
LABELS = {
    "registered_nearest_neighbor": "Nearest neighbour", "registered_idw_k12": "Registered IDW",
    "anchor_only_calibrated": "Anchor-only", "gimvi_adapted": "gimVI (adapted)",
    "atlas_plus_global_anchor_residual": "Global residual",
    "atlas_plus_local_anchor_residual": "Local residual",
    "hyperspatial_map_calibrated_fusion": "HyperSpatial-MAP", "transimpspa": "TransImpSpa",
}
COLORS = {
    "registered_nearest_neighbor": "#777777", "registered_idw_k12": "#E69F00",
    "anchor_only_calibrated": "#0072B2", "gimvi_adapted": "#56B4E9",
    "atlas_plus_global_anchor_residual": "#8E6C8A",
    "atlas_plus_local_anchor_residual": "#009E73",
    "hyperspatial_map_calibrated_fusion": "#C83E4D", "transimpspa": "#6A3D9A",
}
RUNS = [
    {"key": "50p_E1_to_E2", "stage": "50%", "source": "E1", "target": "E2",
     "path": FROZEN / "04_predictions/50p_E1_to_E2",
     "source_file": DATA / "weMERFISH_measured_A_50p_E1.h5ad",
     "target_file": DATA / "weMERFISH_measured_A_50p_E2.h5ad"},
    {"key": "50p_E2_to_E1", "stage": "50%", "source": "E2", "target": "E1",
     "path": FROZEN / "04_predictions/50p_E2_to_E1",
     "source_file": DATA / "weMERFISH_measured_A_50p_E2.h5ad",
     "target_file": DATA / "weMERFISH_measured_A_50p_E1.h5ad"},
    {"key": "75p_E1_to_E2", "stage": "75%", "source": "E1", "target": "E2",
     "path": REPO / "results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2",
     "source_file": DATA / "weMERFISH_measured_B_75p_E1.h5ad",
     "target_file": DATA / "weMERFISH_measured_B_75p_E2.h5ad"},
    {"key": "75p_E2_to_E1", "stage": "75%", "source": "E2", "target": "E1",
     "path": REPO / "results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1",
     "source_file": DATA / "weMERFISH_measured_B_75p_E2.h5ad",
     "target_file": DATA / "weMERFISH_measured_B_75p_E1.h5ad"},
    {"key": "6s_E1_to_E2", "stage": "6-somite", "source": "E1", "target": "E2",
     "path": FROZEN / "04_predictions/6s_E1_to_E2",
     "source_file": DATA / "weMERFISH_measured_C_6s_E1_rescaled_z.h5ad",
     "target_file": DATA / "weMERFISH_measured_C_6s_E2_rescaled_z.h5ad"},
    {"key": "6s_E2_to_E1", "stage": "6-somite", "source": "E2", "target": "E1",
     "path": FROZEN / "04_predictions/6s_E2_to_E1",
     "source_file": DATA / "weMERFISH_measured_C_6s_E2_rescaled_z.h5ad",
     "target_file": DATA / "weMERFISH_measured_C_6s_E1_rescaled_z.h5ad"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pooled(a, b):
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom else np.nan


def transimpspa_metrics():
    rows = []
    for r in RUNS:
        fm = json.loads((r["path"] / "prediction_manifest.json").read_text())
        em = json.loads((TRANSIMP / "protocol" / f"{r['key']}_PREDICTION_MANIFEST.json").read_text())
        ai = np.asarray(em["anchor_indices"], int)
        na = np.asarray(em["non_anchor_indices"], int)
        source, _, genes = measured_counts(r["source_file"])
        target, xyz, target_genes = measured_counts(r["target_file"])
        assert np.array_equal(genes, target_genes)
        depth = float(np.median(np.maximum(source.sum(1), 1)))
        truth = np.log1p(normalize_counts(target, depth)).astype(np.float32)[:, na]
        coords = canonicalize(xyz)[0]
        thresholds = np.quantile(normalize_counts(source, depth), 0.8, axis=0)[na]
        svg_global = np.asarray([g for g in fm["evaluation_svg_indices"] if g in set(na)], int)
        remap = {g: i for i, g in enumerate(na)}
        svg = np.asarray([remap[g] for g in svg_global], int)
        base_path = Path(fm["methods"][str(SEED)]["predictions"]["registered_idw_k12"]["path"])
        base = np.asarray(np.load(base_path, mmap_mode="r"), np.float32)[:, na]
        pred_path = Path(em["prediction"])
        assert sha256(pred_path) == em["prediction_sha256"]
        pred = np.asarray(np.load(pred_path, mmap_mode="r"), np.float32)[:, na]
        result = evaluate("transimpspa", truth, pred, coords, svg, thresholds, TRANSIMP_EVAL_SEED)
        result.pop("per_gene_pearson")
        ci = result.pop("svg_spatial_pearson_bootstrap_ci95")
        residual = pearson_columns(truth - base, pred - base)
        rows.append({
            "target_key": r["key"], "stage": r["stage"], "source": r["source"], "target": r["target"],
            "method": "transimpspa", "rho_res_pooled": pooled(truth - base, pred - base),
            "residual_spatial_pearson_median": float(np.nanmedian(residual)), **result,
            "pearson_ci_low": ci[0], "pearson_ci_high": ci[1], "evaluated_non_anchor_genes": len(na),
        })
    return pd.DataFrame(rows)


def summarize(metrics):
    return metrics.groupby("method", as_index=False).agg(
        targets=("target_key", "nunique"), rho_res_pooled=("rho_res_pooled", "mean"),
        residual_spatial_pearson_median=("residual_spatial_pearson_median", "mean"),
        spatial_pearson=("svg_spatial_pearson_median", "mean"),
        auprc_enrichment=("auprc_enrichment_median", "mean"), dice=("topk_dice_median", "mean"),
        centroid_error=("weighted_centroid_error_median", "mean"), spatial_emd=("spatial_emd_median", "mean"),
        log1p_rmse=("log1p_rmse", "mean"), moran_error=("moran_error_median", "mean"),
    )


def transimpspa_runtime():
    rows = []
    for r in RUNS:
        x = json.loads((TRANSIMP / "protocol" / f"{r['key']}_PREDICTION_MANIFEST.json").read_text())
        rows.append({"target_key": r["key"], "runtime_seconds": x["total_runtime_seconds"],
                     "peak_gpu_memory_mb": x["peak_gpu_memory_mb"],
                     "peak_cpu_rss_mb": x["peak_cpu_rss_mb"], "method": "transimpspa"})
    return pd.DataFrame(rows)


def plot(metrics, summary, runtime):
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=.075, right=.975, top=.92, bottom=.18, wspace=.70, hspace=.72)
    for i, ax in enumerate(axs.flat):
        ax.text(-.18, 1.12, "abcdef"[i], transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")
    order = [r["key"] for r in RUNS]
    for method in PLOT_METHODS:
        d = metrics[metrics.method == method].set_index("target_key").reindex(order)
        axs[0, 0].scatter(range(6), d.rho_res_pooled, s=24, color=COLORS[method], label=LABELS[method])
    axs[0, 0].axhline(0, color="#999", ls=":", lw=.7)
    axs[0, 0].set_ylabel("Pooled residual Pearson ↑")
    axs[0, 0].set_xticks(range(6), [r["stage"] + " " + r["target"] for r in RUNS], rotation=35, ha="right")

    selected = ["gimvi_adapted", "transimpspa", "hyperspatial_map_calibrated_fusion"]
    sm = summary.set_index("method")
    endpoints = [("spatial_pearson", 1), ("auprc_enrichment", 1), ("dice", 1),
                 ("centroid_error", 0), ("spatial_emd", 0), ("log1p_rmse", 0), ("moran_error", 0)]
    matrix = []
    for method in selected:
        row = []
        for endpoint, higher in endpoints:
            base = sm.loc["registered_idw_k12", endpoint]
            row.append(sm.loc[method, endpoint] - base if higher else base - sm.loc[method, endpoint])
        matrix.append(row)
    im = axs[0, 1].imshow(matrix, aspect="auto", cmap="RdBu_r")
    axs[0, 1].set_yticks(range(3), [LABELS[m] for m in selected])
    axs[0, 1].set_xticks(range(7), ["Pearson", "AUPRC", "Dice", "Centroid", "EMD", "RMSE", "Moran"],
                        rotation=40, ha="right")
    axs[0, 1].set_title("Favourable effect vs registered IDW")
    fig.colorbar(im, ax=axs[0, 1], fraction=.04)

    for method in selected:
        axs[0, 2].scatter(sm.loc[method, "rho_res_pooled"], sm.loc[method, "dice"], s=55, color=COLORS[method])
        axs[0, 2].annotate(LABELS[method], (sm.loc[method, "rho_res_pooled"], sm.loc[method, "dice"]),
                           xytext=(3, 3), textcoords="offset points", fontsize=6)
    axs[0, 2].margins(x=.18, y=.20)
    axs[0, 2].set_xlabel("Residual Pearson ↑"); axs[0, 2].set_ylabel("Dice ↑")
    axs[0, 2].set_title("Specimen adaptation and localization")

    for j, endpoint in enumerate(["weighted_centroid_error_median", "spatial_emd_median", "log1p_rmse"]):
        base = metrics[metrics.method == "registered_idw_k12"].set_index("target_key")[endpoint]
        for i, method in enumerate(selected):
            current = metrics[metrics.method == method].set_index("target_key")[endpoint].reindex(base.index)
            values = 100 * (base - current) / base
            axs[1, 0].scatter(np.full(len(values), j) + (i - 1) * .11, values, s=18,
                              color=COLORS[method], alpha=.8)
    axs[1, 0].axhline(0, color="#999", ls=":", lw=.7)
    axs[1, 0].set_ylabel("Improvement vs registered IDW (%) ↑")
    axs[1, 0].set_xticks(range(3), ["Centroid", "EMD", "RMSE"])
    axs[1, 0].set_title("Localization and reconstruction")

    for method in ("gimvi_adapted", "transimpspa"):
        d = runtime[runtime.method == method]
        axs[1, 1].scatter(d.runtime_seconds, d.peak_gpu_memory_mb / 1024, s=35, color=COLORS[method],
                          label=LABELS[method])
    axs[1, 1].set_xlabel("Runtime incl. source calibration (s)")
    axs[1, 1].set_ylabel("Peak GPU memory (GB)")
    axs[1, 1].set_title("Comparator resource use")

    r = RUNS[2]
    fm = json.loads((r["path"] / "prediction_manifest.json").read_text())
    source, _, genes = measured_counts(r["source_file"])
    target, xyz, _ = measured_counts(r["target_file"])
    xyz = canonicalize(xyz)[0]
    depth = float(np.median(np.maximum(source.sum(1), 1)))
    truth = np.log1p(normalize_counts(target, depth))
    gi = int(np.flatnonzero(genes == "wnt11f2")[0])
    gimvi_manifest = json.loads((GIMVI / "01_protocol" / f"{r['key']}_PREDICTION_MANIFEST.json").read_text())
    trans_manifest = json.loads((TRANSIMP / "protocol" / f"{r['key']}_PREDICTION_MANIFEST.json").read_text())
    paths = {
        "gimvi_adapted": Path(gimvi_manifest["prediction"]),
        "hyperspatial_map_calibrated_fusion": Path(fm["methods"][str(SEED)]["predictions"]["hyperspatial_map_calibrated_fusion"]["path"]),
        "transimpspa": Path(trans_manifest["prediction"]),
    }
    values = [truth[:, gi]] + [np.load(paths[m], mmap_mode="r")[:, gi]
                               for m in ("hyperspatial_map_calibrated_fusion", "gimvi_adapted", "transimpspa")]
    vmax = np.quantile(np.concatenate(values), .99)
    for i, (label, values_i) in enumerate(zip(["Measured", "MAP", "gimVI", "TransImpSpa"], values)):
        inset = axs[1, 2].inset_axes([i * .25, .04, .235, .88])
        inset.scatter(xyz[:, 0], xyz[:, 1], c=values_i, s=.25, cmap="viridis", vmin=0, vmax=vmax,
                      rasterized=True)
        inset.axis("off"); inset.set_aspect("equal"); inset.set_title(label, fontsize=6.5, pad=2)
    axs[1, 2].axis("off")
    axs[1, 2].set_title("Source-predeclared wnt11f2 · shared scale", fontsize=7)

    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[m], label=LABELS[m]) for m in LEGEND_METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=4, frameon=False,
               fontsize=7, columnspacing=1.5, handletextpad=.4)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / "figures" / f"GIMVI_TransImpSpa_external_baseline_comparison.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=.15, facecolor="white")
    plt.close(fig)


def main():
    original_metrics = pd.read_csv(GIMVI / "03_metrics/PER_TARGET_METRICS.tsv", sep="\t")
    original_runtime = pd.read_csv(GIMVI / "03_metrics/RUNTIME_MEMORY.tsv", sep="\t")
    trans_metrics = transimpspa_metrics()
    metrics = pd.concat([original_metrics, trans_metrics], ignore_index=True)
    summary = summarize(metrics)
    runtime = pd.concat([original_runtime, transimpspa_runtime()], ignore_index=True)
    trans_metrics.to_csv(OUT / "TRANSIMPSPA_FIGURE_METRICS.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "FIGURE_METHOD_SUMMARY.tsv", sep="\t", index=False)
    runtime.to_csv(OUT / "FIGURE_RUNTIME_MEMORY.tsv", sep="\t", index=False)
    plot(metrics, summary, runtime)

    inputs = [GIMVI / "03_metrics/PER_TARGET_METRICS.tsv", GIMVI / "03_metrics/METHOD_SUMMARY.tsv",
              GIMVI / "03_metrics/RUNTIME_MEMORY.tsv",
              GIMVI / "04_figures/GIMVI_external_baseline_comparison.pdf",
              GIMVI / "04_figures/GIMVI_external_baseline_comparison.svg",
              GIMVI / "04_figures/GIMVI_external_baseline_comparison.png",
              TRANSIMP / "protocol/PREDICTIONS_LOCKED.json"]
    for r in RUNS:
        inputs.append(TRANSIMP / "protocol" / f"{r['key']}_PREDICTION_MANIFEST.json")
    pd.DataFrame([{"path": str(p), "sha256": sha256(p)} for p in inputs]).to_csv(
        OUT / "INPUT_HASHES.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
