#!/usr/bin/env python3
"""Locked TransImpSpa benchmark against frozen HyperSpatial-MAP predictions.

The program has separate prepare, predict, and evaluate phases.  The prediction
phase reads full molecular data from the source embryo, but reads only geometry,
gene names, IDs, and the 16 frozen anchor columns from the target H5AD.  Hidden
target columns can be opened only by the evaluate phase after all six prediction
files have been serialized and hashed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from run_hyperspatial_map_stage1_20260803 import (  # noqa: E402
    depth_predictor,
    octants,
    predicted_depth,
    target_anchor_counts,
)
from run_wemerfish_temporal_holdout import (  # noqa: E402
    canonicalize,
    coordinates_only,
    evaluate,
    measured_counts,
    normalize_counts,
    pearson_columns,
)

OUT = REPO / "PI_revision/02B_spatial_reconstruction_comparator_20260817_132013"
FROZEN = REPO / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726"
DATA = REPO / "data_v10/wemerfish_zebrafish_3stage"
TRANSIMP_COMMIT = "e0761359a297f4f8adbfef968ccb5393ce72409d"
SEED = 10
MAP_SEED = 42
N_EPOCHS = 2000
N_NEIGHS = 6
LATENT_DIM = 256
LR = 0.01
WEIGHT_DECAY = 0.01
WT_SPA = 1.0

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

METHODS = ["registered_idw_k12", "anchor_only_calibrated",
           "hyperspatial_map_calibrated_fusion", "transimpspa"]
LABELS = {"registered_idw_k12": "Registered IDW", "anchor_only_calibrated": "Anchor-only",
          "hyperspatial_map_calibrated_fusion": "HyperSpatial-MAP", "transimpspa": "TransImpSpa"}
COLORS = {"registered_idw_k12": "#8A8A8A", "anchor_only_calibrated": "#E69F00",
          "hyperspatial_map_calibrated_fusion": "#2A6FBB", "transimpspa": "#6A3D9A"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj, exclusive: bool = True):
    mode = "x" if exclusive else "w"
    with path.open(mode) as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def write_csv(path: Path, frame: pd.DataFrame):
    frame.to_csv(path, index=False)


def official_model(repo: Path):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != TRANSIMP_COMMIT:
        raise RuntimeError(f"TransImp repository must be at {TRANSIMP_COMMIT}; found {commit}")
    model_path = repo / "transpa/model.py"
    if sha256(model_path) != "99514354e41de3d947e5b7049cea01f61c92a33d373d829c3ad8d85cb170d4b8":
        raise RuntimeError("Official TransImp model.py hash differs from locked paper-version source")
    spec = importlib.util.spec_from_file_location("locked_transpa_model", model_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def knn_adjacency(xyz: np.ndarray) -> sparse.coo_matrix:
    """Published Squidpy generic-coordinate default: directed binary 6-NN graph."""
    tree = NearestNeighbors(n_neighbors=N_NEIGHS, metric="euclidean").fit(xyz)
    _, indices = tree.kneighbors()
    rows = np.repeat(np.arange(len(xyz)), N_NEIGHS)
    return sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, indices.ravel())),
                             shape=(len(xyz), len(xyz)))


def torch_adjacency(adj: sparse.coo_matrix, device: torch.device):
    indices = torch.as_tensor(np.vstack([adj.row, adj.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(adj.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, adj.shape, device=device).coalesce()


def fit_predict(module, reference_train: np.ndarray, target_train: np.ndarray,
                reference_test: np.ndarray, target_xyz: np.ndarray, device: torch.device):
    x = torch.as_tensor(reference_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(target_train, dtype=torch.float32, device=device)
    adj = knn_adjacency(target_xyz)
    spa = module.SpaAutoCorr(y, torch_adjacency(adj, device), method="moranI")
    model = module.TransImp(
        dim_tgt_outputs=y.shape[0], dim_ref_inputs=x.shape[0], dim_hid=LATENT_DIM,
        clip_max=10, spa_inst=spa, mapping_mode="lowrank", device=device, seed=SEED,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    started = time.perf_counter()
    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, imp_loss, spa_loss = model.loss(
            x, y, wt_spa=WT_SPA, wt_l1norm=None, wt_l2norm=None
        )
        loss.backward()
        optimizer.step()
        if epoch in (0, 99, 499, 999, 1499, 1999):
            history.append({"epoch": epoch + 1, "loss": float(loss.detach()),
                            "translation_loss": float(imp_loss), "spatial_loss": float(spa_loss)})
    with torch.no_grad():
        pred = model.predict(torch.as_tensor(reference_test, dtype=torch.float32, device=device))
    torch.cuda.synchronize()
    return pred.astype(np.float32), model.state_dict(), history, time.perf_counter() - started


def affine(pred: np.ndarray, truth: np.ndarray):
    px, py = pred.mean(0), truth.mean(0)
    x = pred - px
    slope = np.clip((x * (truth - py)).sum(0) / ((x * x).sum(0) + 1e-3), 0, 3).astype(np.float32)
    intercept = (py - slope * px).astype(np.float32)
    return slope, intercept


def apply_affine(pred: np.ndarray, calibration):
    slope, intercept = calibration
    return np.maximum(pred * slope + intercept, 0).astype(np.float32)


def frozen_paths(manifest: dict):
    p = manifest["methods"][str(MAP_SEED)]["predictions"]
    def find(names):
        for name in names:
            if name in p:
                return Path(p[name]["path"])
        raise KeyError(names)
    return {
        "registered_idw_k12": find(["registered_idw_k12"]),
        "anchor_only_calibrated": find(["anchor_only_calibrated", "anchor_only_low_rank"]),
        "hyperspatial_map_calibrated_fusion": find(["hyperspatial_map_calibrated_fusion", "hyperspatial_map"]),
    }


def prepare(transimp_repo: Path):
    module = official_model(transimp_repo)
    del module
    protocol_rows, anchors, hashes = [], [], []
    for r in RUNS:
        manifest_path = r["path"] / "prediction_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        ai = np.asarray(manifest["anchor_indices"], int)
        ag = np.asarray(manifest["anchor_genes"], str)
        if len(ai) != 16:
            raise AssertionError("Expected exactly 16 frozen anchors")
        xyz, genes, ids, coord_key = coordinates_only(r["target_file"])
        if not np.array_equal(genes[ai], ag) or len(genes) != 495:
            raise AssertionError("Frozen gene/anchor mismatch")
        fp = frozen_paths(manifest)
        protocol_rows.append({"target_key": r["key"], "stage": r["stage"], "source": r["source"],
                              "target": r["target"], "source_cells": measured_counts(r["source_file"])[0].shape[0],
                              "target_cells": len(ids), "genes": len(genes), "anchors": len(ai),
                              "hidden_genes": len(genes) - len(ai), "coordinate_key": coord_key})
        anchors.extend({"target_key": r["key"], "rank": j + 1, "gene": g, "gene_index": int(i)}
                       for j, (g, i) in enumerate(zip(ag, ai)))
        for kind, path in [("source_h5ad", r["source_file"]), ("target_h5ad", r["target_file"]),
                           ("prediction_manifest", manifest_path), *fp.items()]:
            hashes.append({"target_key": r["key"], "kind": kind, "path": str(path), "sha256": sha256(path)})
    model_path = transimp_repo / "transpa/model.py"
    hashes.append({"target_key": "external", "kind": "TransImp_model.py", "path": str(model_path),
                   "sha256": sha256(model_path)})
    write_csv(OUT / "protocol/RUN_MANIFEST.csv", pd.DataFrame(protocol_rows))
    write_csv(OUT / "protocol/FROZEN_ANCHORS.csv", pd.DataFrame(anchors))
    write_csv(OUT / "protocol/INPUT_HASHES_BEFORE.csv", pd.DataFrame(hashes))
    config = {
        "status": "FROZEN_BEFORE_TARGET_PREDICTION", "external_method": "TransImpSpa",
        "paper_doi": "10.1016/j.patter.2024.101021", "official_repository": "https://github.com/qiaochen/tranSpa",
        "official_commit": TRANSIMP_COMMIT, "seed": SEED, "n_epochs": N_EPOCHS,
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "latent_dimension": LATENT_DIM,
        "mapping_mode": "lowrank", "signature_mode": "cell", "spatial_regularizer": "Moran's I MSE",
        "spatial_weight": WT_SPA, "graph": "directed binary 6-nearest-neighbor graph in frozen 3D target coordinates",
        "target_training_columns": "the exact 16 frozen target anchors only",
        "target_hidden_columns": "479 non-anchor genes; unavailable until all prediction hashes are locked",
        "normalization": "source median-depth normalization and log1p; target anchor depth inferred from anchors by frozen source-only depth model",
        "source_only_output_calibration": "gene-wise nonnegative affine calibration fitted on source octant 0; octant 7 validation; reference excludes both",
        "hyperparameter_selection": "published settings; no target-truth tuning",
    }
    write_json(OUT / "protocol/FROZEN_CONFIG.json", config)


def predict_one(r: dict, module, device: torch.device):
    manifest = json.loads((r["path"] / "prediction_manifest.json").read_text())
    ai = np.asarray(manifest["anchor_indices"], int)
    na = np.setdiff1d(np.arange(495), ai)
    source_counts, source_xyz_raw, genes = measured_counts(r["source_file"])
    target_xyz_raw, target_genes, target_ids, coord_key = coordinates_only(r["target_file"])
    if not np.array_equal(genes, target_genes):
        raise AssertionError("Source/target gene axes differ")
    target_anchor_raw = target_anchor_counts(r["target_file"], ai)
    source_xyz = canonicalize(source_xyz_raw)[0]
    target_xyz = canonicalize(target_xyz_raw)[0]
    depth = float(np.median(np.maximum(source_counts.sum(1), 1)))
    source_log = np.log1p(normalize_counts(source_counts, depth)).astype(np.float32)
    target_depth = predicted_depth(target_anchor_raw, depth_predictor(source_counts, ai))
    target_anchor_log = np.log1p(target_anchor_raw * (depth / target_depth)[:, None]).astype(np.float32)

    labels = octants(source_xyz)
    pseudo_mask = (labels == 0) | (labels == 7)
    reference_mask = ~pseudo_mask
    pseudo_xyz = source_xyz[pseudo_mask]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pseudo_pred, _, pseudo_history, pseudo_seconds = fit_predict(
        module, source_log[reference_mask][:, ai], source_log[pseudo_mask][:, ai],
        source_log[reference_mask][:, na], pseudo_xyz, device,
    )
    pseudo_labels = labels[pseudo_mask]
    calibration = affine(pseudo_pred[pseudo_labels == 0], source_log[pseudo_mask][pseudo_labels == 0][:, na])
    pseudo_val = apply_affine(pseudo_pred[pseudo_labels == 7], calibration)
    validation_corr = pearson_columns(source_log[pseudo_mask][pseudo_labels == 7][:, na], pseudo_val)
    validation_rmse = float(np.sqrt(np.mean((source_log[pseudo_mask][pseudo_labels == 7][:, na] - pseudo_val) ** 2)))
    del pseudo_pred
    gc.collect()
    torch.cuda.empty_cache()

    target_pred, state, final_history, final_seconds = fit_predict(
        module, source_log[:, ai], target_anchor_log, source_log[:, na], target_xyz, device,
    )
    calibrated = apply_affine(target_pred, calibration)
    full = np.empty((len(target_xyz), len(genes)), np.float32)
    full[:, na] = calibrated
    full[:, ai] = target_anchor_log
    pred_dir = OUT / "predictions" / r["key"]
    pred_dir.mkdir(exist_ok=False)
    prediction_path = pred_dir / "transimpspa_prediction_log1p_locked.npy"
    np.save(prediction_path, full)
    model_path = pred_dir / "transimpspa_model_state.pt"
    torch.save(state, model_path)
    np.savez(pred_dir / "source_only_affine_calibration.npz", slope=calibration[0], intercept=calibration[1], non_anchor_indices=na)
    write_csv(pred_dir / "training_history.csv", pd.DataFrame(
        [{"phase": "source_pseudo", **x} for x in pseudo_history] + [{"phase": "target", **x} for x in final_history]
    ))
    elapsed = time.perf_counter() - started
    peak_gpu = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
    return {
        "target_key": r["key"], "prediction": str(prediction_path), "prediction_sha256": sha256(prediction_path),
        "model_state": str(model_path), "model_state_sha256": sha256(model_path), "target_cells": len(target_ids),
        "coordinate_key": coord_key, "anchor_indices": ai.tolist(), "anchor_genes": genes[ai].tolist(),
        "non_anchor_indices": na.tolist(), "target_hidden_genes_loaded": False,
        "source_validation_median_spatial_pearson": float(np.nanmedian(validation_corr)),
        "source_validation_log1p_rmse": validation_rmse, "pseudo_runtime_seconds": pseudo_seconds,
        "target_runtime_seconds": final_seconds, "total_runtime_seconds": elapsed,
        "peak_gpu_memory_mb": peak_gpu, "peak_cpu_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def predict(transimp_repo: Path, device_name: str):
    config = json.loads((OUT / "protocol/FROZEN_CONFIG.json").read_text())
    if config["status"] != "FROZEN_BEFORE_TARGET_PREDICTION":
        raise RuntimeError("Protocol is not frozen")
    module = official_model(transimp_repo)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifests = []
    for r in RUNS:
        path = OUT / "protocol" / f"{r['key']}_PREDICTION_MANIFEST.json"
        if path.exists():
            item = json.loads(path.read_text())
            if sha256(Path(item["prediction"])) != item["prediction_sha256"]:
                raise RuntimeError("Locked prediction hash mismatch")
        else:
            item = predict_one(r, module, device)
            write_json(path, item)
        manifests.append(item)
    lock = {"status": "ALL_SIX_TRANSIMPSPA_PREDICTIONS_LOCKED_BEFORE_TARGET_TRUTH_ACCESS",
            "external_commit": TRANSIMP_COMMIT,
            "predictions": [{"target_key": x["target_key"], "path": x["prediction"],
                             "sha256": x["prediction_sha256"]} for x in manifests]}
    write_json(OUT / "protocol/PREDICTIONS_LOCKED.json", lock)
    write_json(OUT / "protocol/UNBLINDING_SENTINEL.json",
               {"status": "TARGET_NON_ANCHOR_TRUTH_MAY_NOW_BE_OPENED_FOR_EVALUATION_ONLY",
                "prediction_lock_sha256": sha256(OUT / "protocol/PREDICTIONS_LOCKED.json")})


def pooled_pearson(x, y):
    a, b = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else np.nan


def evaluate_all():
    lock_path = OUT / "protocol/PREDICTIONS_LOCKED.json"
    sentinel = OUT / "protocol/UNBLINDING_SENTINEL.json"
    if not lock_path.exists() or not sentinel.exists():
        raise RuntimeError("Predictions are not locked; target truth must remain inaccessible")
    lock = json.loads(lock_path.read_text())
    for x in lock["predictions"]:
        if sha256(Path(x["path"])) != x["sha256"]:
            raise RuntimeError("Prediction changed after lock")
    rows, gene_rows = [], []
    for r in RUNS:
        manifest = json.loads((r["path"] / "prediction_manifest.json").read_text())
        external = json.loads((OUT / "protocol" / f"{r['key']}_PREDICTION_MANIFEST.json").read_text())
        ai = np.asarray(external["anchor_indices"], int)
        na = np.asarray(external["non_anchor_indices"], int)
        source_counts, _, genes = measured_counts(r["source_file"])
        target_counts, target_xyz_raw, target_genes = measured_counts(r["target_file"])
        if not np.array_equal(genes, target_genes):
            raise AssertionError("Target gene axis changed at unblinding")
        depth = float(np.median(np.maximum(source_counts.sum(1), 1)))
        truth = np.log1p(normalize_counts(target_counts, depth)).astype(np.float32)[:, na]
        xyz = canonicalize(target_xyz_raw)[0]
        thresholds = np.quantile(normalize_counts(source_counts, depth), 0.8, axis=0)[na]
        svg_global = np.asarray([g for g in manifest["evaluation_svg_indices"] if g in set(na)], int)
        remap = {g: i for i, g in enumerate(na)}
        svg = np.asarray([remap[g] for g in svg_global], int)
        paths = frozen_paths(manifest)
        paths["transimpspa"] = Path(external["prediction"])
        base = np.asarray(np.load(paths["registered_idw_k12"], mmap_mode="r"), np.float32)[:, na]
        method_preds = {}
        for method in METHODS:
            pred = np.asarray(np.load(paths[method], mmap_mode="r"), np.float32)[:, na]
            method_preds[method] = pred
            result = evaluate(method, truth, pred, xyz, svg, thresholds, 20260817 + METHODS.index(method))
            per_gene_spatial = np.asarray(result.pop("per_gene_pearson"), float)
            ci = result.pop("svg_spatial_pearson_bootstrap_ci95")
            residual = pearson_columns(truth - base, pred - base)
            pooled = np.nan if method == "registered_idw_k12" else pooled_pearson(truth - base, pred - base)
            rows.append({"target_key": r["key"], "stage": r["stage"], "source": r["source"], "target": r["target"],
                         "method": method, "label": LABELS[method], "evaluated_non_anchor_genes": len(na),
                         "source_defined_spatial_gene_count": len(svg), "pooled_cell_gene_residual_pearson": pooled,
                         "residual_spatial_pearson_median_all_genes": float(np.nanmedian(residual)) if method != "registered_idw_k12" else np.nan,
                         **result, "spatial_pearson_ci_low": ci[0], "spatial_pearson_ci_high": ci[1]})
            if method in ("hyperspatial_map_calibrated_fusion", "transimpspa"):
                for j, gene_index in enumerate(na):
                    gene_rows.append({"target_key": r["key"], "stage": r["stage"], "source": r["source"],
                                      "target": r["target"], "method": method, "gene": genes[gene_index],
                                      "gene_index": int(gene_index), "spatial_pearson": per_gene_spatial[j],
                                      "residual_spatial_pearson": residual[j]})
    metrics = pd.DataFrame(rows)
    genes_long = pd.DataFrame(gene_rows)
    write_csv(OUT / "metrics/RESULTS_BY_TARGET_LONG.csv", metrics)
    pivot = genes_long.pivot_table(index=["target_key", "stage", "source", "target", "gene", "gene_index"],
                                   columns="method", values=["spatial_pearson", "residual_spatial_pearson"]).reset_index()
    pivot.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in pivot.columns]
    pivot["delta_spatial_pearson_MAP_minus_comparator"] = (
        pivot["spatial_pearson_hyperspatial_map_calibrated_fusion"] - pivot["spatial_pearson_transimpspa"])
    pivot["delta_residual_pearson_MAP_minus_comparator"] = (
        pivot["residual_spatial_pearson_hyperspatial_map_calibrated_fusion"] - pivot["residual_spatial_pearson_transimpspa"])
    write_csv(OUT / "GENE_LEVEL_RESULTS.csv", pivot)
    write_comparison_table(metrics)
    make_figures(metrics, pivot)
    summary_tables(metrics, pivot)


def make_figures(metrics: pd.DataFrame, genes: pd.DataFrame):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    order = [r["key"] for r in RUNS]
    short = ["50%\nE1→E2", "50%\nE2→E1", "75%\nE1→E2", "75%\nE2→E1", "6-somite\nE1→E2", "6-somite\nE2→E1"]
    endpoints = [("svg_spatial_pearson_median", "Spatial Pearson ↑"),
                 ("residual_spatial_pearson_median_all_genes", "Residual Pearson ↑"),
                 ("auprc_enrichment_median", "AUPRC enrichment ↑"), ("topk_dice_median", "Dice ↑"),
                 ("weighted_centroid_error_median", "Centroid error ↓"), ("spatial_emd_median", "Spatial EMD ↓"),
                 ("log1p_rmse", "log1p RMSE ↓")]
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.8))
    for ax, (column, label) in zip(axes.flat, endpoints):
        # Anchor-only remains in the frozen tables but is intentionally omitted
        # from the diagnostic figures at the PI's request.
        for method in ("registered_idw_k12", "hyperspatial_map_calibrated_fusion", "transimpspa"):
            d = metrics[metrics.method == method].set_index("target_key").reindex(order)
            ax.plot(range(6), d[column], marker="o", ms=3.6, lw=1.1, color=COLORS[method], label=LABELS[method])
        ax.set_title(label, fontsize=8.5)
        ax.set_xticks(range(6), short, fontsize=6.5)
    axes.flat[-1].axis("off")
    plotted_methods = ("registered_idw_k12", "hyperspatial_map_calibrated_fusion", "transimpspa")
    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", lw=1.2, label=LABELS[m]) for m in plotted_methods]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Matched 16-gene spatial reconstruction benchmark", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.07, 0.99, 0.94])
    save_figure(fig, "R1_target_wise_absolute_performance")

    high = {"svg_spatial_pearson_median", "residual_spatial_pearson_median_all_genes", "auprc_enrichment_median", "topk_dice_median"}
    mat = []
    for column, _ in endpoints:
        p = metrics.pivot(index="target_key", columns="method", values=column).reindex(order)
        d = p["hyperspatial_map_calibrated_fusion"] - p["transimpspa"]
        if column not in high:
            d = -d
        mat.append(d.to_numpy())
    mat = np.asarray(mat)
    vmax = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(6), short)
    ax.set_yticks(range(7), [x[1].replace(" ↑", "").replace(" ↓", "") for x in endpoints])
    ax.set_title("HyperSpatial-MAP − TransImpSpa (positive = MAP better)", fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(mat[i,j]) > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Favourable MAP effect")
    fig.tight_layout()
    save_figure(fig, "R2_MAP_minus_comparator_effects")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharex=True)
    for ax, col, title in [(axes[0], "delta_residual_pearson_MAP_minus_comparator", "Residual Pearson"),
                           (axes[1], "delta_spatial_pearson_MAP_minus_comparator", "Spatial Pearson")]:
        vals = [genes.loc[genes.target_key == key, col].dropna().to_numpy() for key in order]
        parts = ax.violinplot(vals, positions=np.arange(6), showextrema=False, widths=0.82)
        for body in parts["bodies"]:
            body.set_facecolor("#2A6FBB"); body.set_edgecolor("none"); body.set_alpha(0.55)
        med = [np.nanmedian(v) for v in vals]
        ax.scatter(range(6), med, s=14, color="#111111", zorder=3)
        ax.axhline(0, color="#777777", lw=0.8, ls="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(6), short, fontsize=6.5)
        ax.set_ylabel("Gene-level MAP − TransImpSpa")
    fig.suptitle("Gene-level matched-comparator effects", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "R3_gene_level_comparison")


def save_figure(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / "figures" / f"{stem}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summary_tables(metrics: pd.DataFrame, genes: pd.DataFrame):
    rows = []
    for endpoint, higher in [("svg_spatial_pearson_median", True),
                             ("residual_spatial_pearson_median_all_genes", True),
                             ("pooled_cell_gene_residual_pearson", True),
                             ("auprc_enrichment_median", True), ("topk_dice_median", True),
                             ("weighted_centroid_error_median", False), ("spatial_emd_median", False),
                             ("log1p_rmse", False)]:
        p = metrics.pivot(index="target_key", columns="method", values=endpoint)
        for comparator in ("transimpspa", "anchor_only_calibrated", "registered_idw_k12"):
            d = p["hyperspatial_map_calibrated_fusion"] - p[comparator]
            if not higher:
                d = -d
            rows.append({"endpoint": endpoint, "comparison": f"MAP_minus_{comparator}",
                         "mean_favourable_effect": d.mean(), "median_favourable_effect": d.median(),
                         "directions_MAP_better": int((d > 0).sum()), "directions_evaluable": int(d.notna().sum())})
    write_csv(OUT / "metrics/SUMMARY_EFFECTS.csv", pd.DataFrame(rows))
    gsum = []
    for stage, d in list(genes.groupby("stage")) + [("all", genes)]:
        for col in ("delta_residual_pearson_MAP_minus_comparator", "delta_spatial_pearson_MAP_minus_comparator"):
            x = d[col].dropna()
            gsum.append({"stage": stage, "effect": col, "n_gene_direction_records": len(x),
                         "median": x.median(), "q25": x.quantile(.25), "q75": x.quantile(.75),
                         "fraction_MAP_higher": float((x > 0).mean()),
                         "fraction_comparator_higher": float((x < 0).mean())})
    write_csv(OUT / "metrics/GENE_LEVEL_SUMMARY.csv", pd.DataFrame(gsum))


def write_comparison_table(metrics: pd.DataFrame):
    endpoint_specs = [
        ("svg_spatial_pearson_median", "spatial Pearson", True),
        ("residual_spatial_pearson_median_all_genes", "target-specific residual spatial Pearson", True),
        ("auprc_enrichment_median", "AUPRC enrichment", True),
        ("topk_dice_median", "prevalence-matched Dice", True),
        ("weighted_centroid_error_median", "normalized centroid error", False),
        ("spatial_emd_median", "spatial EMD", False),
        ("log1p_rmse", "log1p RMSE", False),
        ("pooled_cell_gene_residual_pearson", "pooled cell×gene residual Pearson", True),
    ]
    rows = []
    for r in RUNS:
        d = metrics[metrics.target_key == r["key"]].set_index("method")
        for column, label, higher in endpoint_specs:
            vals = {m: d.loc[m, column] for m in METHODS}
            sign = 1 if higher else -1
            rows.append({
                "target_key": r["key"], "stage": r["stage"], "source": r["source"], "target": r["target"],
                "endpoint": label, "higher_is_better": higher,
                "registered_IDW": vals["registered_idw_k12"],
                "anchor_only": vals["anchor_only_calibrated"],
                "HyperSpatial_MAP": vals["hyperspatial_map_calibrated_fusion"],
                "TransImpSpa": vals["transimpspa"],
                "MAP_minus_TransImpSpa_favourable": sign * (vals["hyperspatial_map_calibrated_fusion"] - vals["transimpspa"]),
                "anchor_minus_TransImpSpa_favourable": sign * (vals["anchor_only_calibrated"] - vals["transimpspa"]),
                "TransImpSpa_minus_IDW_favourable": sign * (vals["transimpspa"] - vals["registered_idw_k12"]),
                "aggregation": "median over source-defined spatial-gene subset" if column not in ("log1p_rmse", "residual_spatial_pearson_median_all_genes", "pooled_cell_gene_residual_pearson") else ("all 479 non-anchor genes" if column != "pooled_cell_gene_residual_pearson" else "Pearson after pooling all target-cell×479-gene residual entries"),
            })
    write_csv(OUT / "RESULTS_BY_TARGET.csv", pd.DataFrame(rows))


def verify_frozen_inputs():
    before = pd.read_csv(OUT / "protocol/INPUT_HASHES_BEFORE.csv")
    after = before.copy()
    after["sha256"] = [sha256(Path(path)) for path in after["path"]]
    write_csv(OUT / "protocol/INPUT_HASHES_AFTER.csv", after)
    joined = before.merge(after, on=["target_key", "kind", "path"], suffixes=("_before", "_after"))
    joined["unchanged"] = joined.sha256_before == joined.sha256_after
    write_csv(OUT / "protocol/HASH_VERIFICATION.csv", joined)
    status = bool(joined.unchanged.all())
    write_json(OUT / "protocol/FINAL_SAFETY_CHECK.json",
               {"status": "PASS" if status else "FAIL", "files_checked": len(joined),
                "all_frozen_inputs_unchanged": status,
                "prediction_lock_sha256": sha256(OUT / "protocol/PREDICTIONS_LOCKED.json")},
               exclusive=False)
    env = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (OUT / "environment.txt").write_text(
        f"python={platform.python_version()}\nexecutable={sys.executable}\n"
        f"platform={platform.platform()}\ntorch={torch.__version__}\n"
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}\n\n{env}"
    )
    if not status:
        raise RuntimeError("One or more frozen inputs changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "predict", "evaluate", "summarize", "verify"])
    parser.add_argument("--transimp-repo", type=Path, default=Path("/tmp/tranSpa_audit"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.transimp_repo)
    elif args.phase == "predict":
        predict(args.transimp_repo, args.device)
    elif args.phase == "evaluate":
        evaluate_all()
    elif args.phase == "summarize":
        long_path = OUT / "metrics/RESULTS_BY_TARGET_LONG.csv"
        if long_path.exists():
            metrics = pd.read_csv(long_path)
        else:
            metrics = pd.read_csv(OUT / "RESULTS_BY_TARGET.csv")
            write_csv(long_path, metrics)
        write_comparison_table(metrics)
    else:
        verify_frozen_inputs()


if __name__ == "__main__":
    main()
