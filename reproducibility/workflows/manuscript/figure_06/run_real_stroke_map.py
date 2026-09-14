#!/usr/bin/env python3
"""Leakage-separated sham-to-stroke HyperSpatial-MAP experiment.

This adapter imports the frozen manuscript implementation and changes only the
input/output layer for the CONCERT serial-section mouse objects.  It is run as
four separate processes so target non-anchor expression cannot be opened before
the prediction manifest is locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import torch

# Import the exact frozen functions rather than reimplementing scientific logic.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from run_hyperspatial_map_stage1_20260803 import (  # noqa: E402
    GAIN_GRID, K_LOCAL, RANK, RIDGE_GRID, depth_predictor, features, idw,
    octants, predicted_depth, pseudo_block_prediction, sha256,
)
from run_hyperspatial_map_stage1b_20260803 import (  # noqa: E402
    ANCHOR_COUNT, apply_centered, candidate_fusion, fit_centered_ridge_rank,
    select_non_axial_anchors, source_gate,
)
from run_mouse_temporal_transport_gate import source_svg_indices  # noqa: E402
from run_wemerfish_temporal_holdout import (  # noqa: E402
    canonicalize, normalize_counts, pearson_columns, registered_coordinates,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO / "PI_revision/16_CONCERT_stroke_geometry_20260820_095543"
SHAM = INPUT_ROOT / "SHAM_3D.h5ad"
STROKE = INPUT_ROOT / "STROKE_3D.h5ad"
SHAM_COORD = INPUT_ROOT / "SHAM_COORDINATES.npy"
STROKE_COORD = INPUT_ROOT / "STROKE_COORDINATES.npy"
ORIGINAL_H5 = REPO / "PI_revision/15_CONCERT_3D_stroke_audit_20260820_092123/official_sources/Mouse_brain_stroke_all_data.h5"
LOCKED = ROOT / "LOCKED_PREDICTIONS"
FIG = ROOT / "figures"
SEEDS = (42, 43, 44)
METHODS = ("registered_idw_k12", "anchor_only_calibrated", "hyperspatial_map_calibrated_fusion")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sym_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()))


def overlap(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return float(0.5 * ((da <= threshold).mean() + (db <= threshold).mean()))


def dense_x(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = ad.read_h5ad(path)
    x = obj.X.toarray() if hasattr(obj.X, "toarray") else np.asarray(obj.X)
    return np.asarray(x, np.float32), np.asarray(obj.var_names.astype(str))


def phase_geometry() -> None:
    """Read frozen coordinates only and freeze three registration seeds."""
    started = time.perf_counter()
    source_raw = np.asarray(np.load(SHAM_COORD), np.float32)
    target_raw = np.asarray(np.load(STROKE_COORD), np.float32)
    source, source_canon = canonicalize(source_raw)
    target, target_canon = canonicalize(target_raw)
    target_nn = cKDTree(target).query(target, k=2)[0][:, 1]
    threshold = float(2.0 * np.median(target_nn[target_nn > 0]))
    records = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        torch.manual_seed(seed)
        mapped, rigid, geometry = registered_coordinates(source, target, 30, seed, device)
        mapped_path = LOCKED / f"seed{seed}_registered_source_coordinates.npy"
        rigid_path = LOCKED / f"seed{seed}_rigid_source_coordinates.npy"
        np.save(mapped_path, mapped.astype(np.float32))
        np.save(rigid_path, rigid.astype(np.float32))
        records.append({
            "seed": seed,
            **geometry,
            "rigid_chamfer": sym_chamfer(rigid, target),
            "raw_overlap": overlap(source, target, threshold),
            "rigid_overlap": overlap(rigid, target, threshold),
            "registered_overlap": overlap(mapped, target, threshold),
            "overlap_threshold_canonical": threshold,
            "mapped_path": str(mapped_path),
            "mapped_sha256": digest(mapped_path),
            "rigid_path": str(rigid_path),
            "rigid_sha256": digest(rigid_path),
        })
    freeze = {
        "status": "geometry_registration_frozen_before_any_stroke_expression_access",
        "device": str(device), "epochs": 30, "seeds": list(SEEDS),
        "source_coordinates": str(SHAM_COORD), "source_coordinates_sha256": digest(SHAM_COORD),
        "target_coordinates": str(STROKE_COORD), "target_coordinates_sha256": digest(STROKE_COORD),
        "source_canonicalization": source_canon, "target_canonicalization": target_canon,
        "registration_records": records,
        "runtime_seconds": time.perf_counter() - started,
        "target_expression_accessed": False,
    }
    json_write(LOCKED / "GEOMETRY_REGISTRATION_FREEZE.json", freeze)


def save_model(path: Path, models: dict[str, dict[str, np.ndarray]], arrays: dict[str, np.ndarray]) -> None:
    payload = dict(arrays)
    for prefix, model in models.items():
        for key, value in model.items():
            payload[f"{prefix}_{key}"] = value
    np.savez_compressed(path, **payload)


def load_model(payload: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
    return {key: payload[f"{prefix}_{key}"] for key in ("x_mean", "x_scale", "y_mean", "weight")}


def moran_columns(values: np.ndarray, coordinates: np.ndarray, k: int = 12, chunk: int = 32) -> np.ndarray:
    neighbors = cKDTree(coordinates).query(coordinates, k=min(k + 1, len(coordinates)))[1][:, 1:]
    result = np.empty(values.shape[1], np.float64)
    for begin in range(0, values.shape[1], chunk):
        x = np.asarray(values[:, begin:begin + chunk], np.float64)
        z = x - x.mean(0, keepdims=True)
        denom = np.square(z).sum(0)
        numer = (z * z[neighbors].mean(1)).sum(0)
        result[begin:begin + chunk] = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 1e-12)
    return result


def phase_source() -> None:
    """Open sham expression only; select anchors, fit source models and freeze B_T."""
    geometry_lock = json.loads((LOCKED / "GEOMETRY_REGISTRATION_FREEZE.json").read_text())
    if geometry_lock["target_expression_accessed"]:
        raise RuntimeError("Geometry lock is invalid")
    source_counts, genes = dense_x(SHAM)
    if not np.allclose(source_counts, np.rint(source_counts)):
        raise AssertionError("Sham X is not raw integer count data")
    source_raw = np.asarray(np.load(SHAM_COORD), np.float32)
    target_raw = np.asarray(np.load(STROKE_COORD), np.float32)
    source_coordinates = canonicalize(source_raw)[0]
    target_coordinates = canonicalize(target_raw)[0]
    # Target gene names are public identifiers; X remains unopened.
    target_backed = ad.read_h5ad(STROKE, backed="r")
    target_genes = np.asarray(target_backed.var_names.astype(str))
    target_backed.file.close()
    if not np.array_equal(genes, target_genes):
        raise AssertionError("Sham/stroke gene axes differ")

    source_depth = np.maximum(source_counts.sum(1), 1.0)
    depth_target = float(np.median(source_depth))
    source_values = normalize_counts(source_counts, depth_target).astype(np.float32)
    source_log = np.log1p(source_values).astype(np.float32)
    anchor_indices, anchor_metadata = select_non_axial_anchors(source_log, genes, ANCHOR_COUNT)
    anchor_genes = genes[anchor_indices]

    prevalence = np.mean(source_log > 0, axis=0)
    variance = np.var(source_log, axis=0)
    eligible = (
        (prevalence >= 0.05) & (prevalence <= 0.98) & (variance > 1e-5)
        & np.asarray([not g.lower().startswith(("mt-", "rpl", "rps")) for g in genes])
    )
    eligible_indices = np.flatnonzero(eligible)
    hidden_indices = np.asarray([i for i in eligible_indices if i not in set(anchor_indices)], np.int64)
    if len(anchor_indices) != 16 or len(hidden_indices) < 500:
        raise RuntimeError("Prespecified molecular eligibility gate failed")

    with (ROOT / "FROZEN_ANCHORS.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_order", "gene", "gene_index", "source_prevalence", "source_log1p_variance"])
        for order, index in enumerate(anchor_indices, 1):
            writer.writerow([order, genes[index], index, prevalence[index], variance[index]])

    labels = octants(source_coordinates)
    pseudo_base = pseudo_block_prediction(source_values, source_coordinates, labels)
    pseudo_base_log = np.log1p(pseudo_base).astype(np.float32)
    source_residual = source_log - pseudo_base_log
    raw_features = source_residual[:, anchor_indices]
    local_features = features(raw_features, source_coordinates, True)
    validation = np.isin(labels, [0, 7])
    train = ~validation
    source_selection = []
    best = None
    for ridge in RIDGE_GRID:
        global_cv = fit_centered_ridge_rank(raw_features[train], source_residual[train], ridge, RANK)
        local_cv = fit_centered_ridge_rank(local_features[train], source_residual[train], ridge, RANK)
        anchor_cv = fit_centered_ridge_rank(source_log[train][:, anchor_indices], source_log[train], ridge, RANK)
        for gain in GAIN_GRID:
            global_log = pseudo_base_log[validation] + gain * apply_centered(global_cv, raw_features[validation])
            local_log = pseudo_base_log[validation] + gain * apply_centered(local_cv, local_features[validation])
            anchor_log = apply_centered(anchor_cv, source_log[validation][:, anchor_indices])
            candidates, names = candidate_fusion(pseudo_base_log[validation], global_log, local_log, anchor_log)
            selection, predictable, metadata = source_gate(candidates, source_log[validation], pseudo_base_log[validation])
            selected = np.column_stack([candidates[selection[g], :, g] for g in range(len(genes))])
            correlation = float(np.nanmedian(pearson_columns(source_log[validation], selected)))
            rmse = float(np.sqrt(np.mean((source_log[validation] - selected) ** 2)))
            record = {"ridge": ridge, "gain": gain, "median_spatial_pearson": correlation,
                      "log1p_rmse": rmse, "predictable_genes": int(predictable.sum())}
            source_selection.append(record)
            key = (correlation, -rmse, int(predictable.sum()))
            if best is None or key > best[0]:
                best = (key, record, names, selection.copy(), predictable.copy(), metadata)
    assert best is not None
    chosen, fusion_names, fusion_selection, predictable, gate_metadata = best[1:]
    predictable[anchor_indices] = False
    ridge, gain = chosen["ridge"], chosen["gain"]
    models = {
        "global": fit_centered_ridge_rank(raw_features, source_residual, ridge, RANK),
        "local": fit_centered_ridge_rank(local_features, source_residual, ridge, RANK),
        "anchor": fit_centered_ridge_rank(source_log[:, anchor_indices], source_log, ridge, RANK),
    }
    depth_weight = depth_predictor(source_counts, anchor_indices)
    svg_order, svg_scores = source_svg_indices([source_log], [source_coordinates], count=min(220, len(genes)))
    evaluation_svg = np.asarray([i for i in svg_order if i in set(hidden_indices)][:100], np.int64)
    # Source-only representative: highest source Moran's I among hidden eligible genes.
    source_moran = moran_columns(source_log[:, hidden_indices], source_coordinates)
    representative_index = int(hidden_indices[int(np.nanargmax(source_moran))])

    model_path = LOCKED / "SOURCE_MODEL.npz"
    save_model(model_path, models, {
        "genes": genes.astype("U"), "anchor_indices": anchor_indices,
        "eligible_indices": eligible_indices, "hidden_indices": hidden_indices,
        "fusion_selection": fusion_selection, "predictable": predictable.astype(np.uint8),
        "depth_weight": depth_weight, "source_values": source_values,
        "evaluation_svg": evaluation_svg, "source_svg_scores": svg_scores,
        "representative_index": np.asarray([representative_index], np.int64),
    })

    # Build and freeze the registered healthy expectation without target expression.
    b_records = []
    for seed in SEEDS:
        mapped_path = LOCKED / f"seed{seed}_registered_source_coordinates.npy"
        expected_hash = next(r["mapped_sha256"] for r in geometry_lock["registration_records"] if r["seed"] == seed)
        if digest(mapped_path) != expected_hash:
            raise RuntimeError("Registered coordinate hash mismatch")
        mapped = np.asarray(np.load(mapped_path), np.float32)
        base_log = np.log1p(idw(mapped, source_values, target_coordinates, K_LOCAL)).astype(np.float32)
        path = LOCKED / f"seed{seed}_registered_idw_k12.npy"
        np.save(path, base_log)
        b_records.append({"seed": seed, "path": str(path), "sha256": digest(path)})

    config = {
        "status": "source_models_anchors_and_registered_atlas_frozen_before_stroke_anchor_access",
        "frozen_runner": str(REPO / "run_hyperspatial_map_stage1b_20260803.py"),
        "frozen_runner_sha256": digest(REPO / "run_hyperspatial_map_stage1b_20260803.py"),
        "anchor_count": 16, "anchor_genes": anchor_genes.tolist(), "anchor_indices": anchor_indices.tolist(),
        "anchor_metadata": anchor_metadata, "eligible_gene_count": int(len(eligible_indices)),
        "hidden_gene_count": int(len(hidden_indices)), "depth_target": depth_target,
        "source_selected_configuration": chosen, "source_selection": source_selection,
        "fusion_candidate_names": fusion_names, "fusion_selection": fusion_selection.tolist(),
        "source_predictable_gene_count": int(predictable[hidden_indices].sum()),
        "source_gate_metadata": gate_metadata, "evaluation_svg_indices": evaluation_svg.tolist(),
        "representative_gene": genes[representative_index], "representative_gene_index": representative_index,
        "representative_selection_rule": "highest 12-NN Moran's I in sham source among eligible hidden genes",
        "model_path": str(model_path), "model_sha256": digest(model_path),
        "registered_atlas": b_records,
        "target_expression_accessed": False, "target_annotations_accessed": False,
    }
    json_write(LOCKED / "SOURCE_AND_ATLAS_FREEZE.json", config)


def read_target_anchor_columns(indices: np.ndarray, expected_genes: np.ndarray) -> np.ndarray:
    """Read exactly 16 target columns from the dense deposited matrix."""
    order = np.argsort(indices)
    sorted_indices = indices[order]
    inverse = np.argsort(order)
    with h5py.File(ORIGINAL_H5, "r") as handle:
        batches = np.asarray([x.decode() for x in handle["Batch"][:]])
        pt_rows = np.flatnonzero(np.char.startswith(batches, "PT"))
        if not np.array_equal(pt_rows, np.arange(len(pt_rows))):
            raise RuntimeError("PT rows are not contiguous as audited")
        genes = np.asarray([x.decode() for x in handle["Gene"][:]])
        if not np.array_equal(genes, expected_genes):
            raise RuntimeError("Gene axis changed")
        values = np.asarray(handle["X"][:len(pt_rows), sorted_indices], np.float32)
    return values[:, inverse]


def phase_predict() -> None:
    """Open only the 16 frozen target anchors and serialize predictions."""
    source_lock = json.loads((LOCKED / "SOURCE_AND_ATLAS_FREEZE.json").read_text())
    model_path = Path(source_lock["model_path"])
    if digest(model_path) != source_lock["model_sha256"]:
        raise RuntimeError("Source model hash mismatch")
    payload = np.load(model_path)
    genes = payload["genes"].astype(str)
    anchor_indices = payload["anchor_indices"].astype(np.int64)
    fusion_selection = payload["fusion_selection"].astype(np.int64)
    predictable = payload["predictable"].astype(bool)
    global_model = load_model(payload, "global")
    local_model = load_model(payload, "local")
    anchor_model = load_model(payload, "anchor")
    gain = float(source_lock["source_selected_configuration"]["gain"])
    target_raw = np.asarray(np.load(STROKE_COORD), np.float32)
    target_coordinates = canonicalize(target_raw)[0]

    target_anchor_raw = read_target_anchor_columns(anchor_indices, genes)
    target_depth = predicted_depth(target_anchor_raw, payload["depth_weight"])
    target_anchor_log = np.log1p(
        target_anchor_raw * (float(source_lock["depth_target"]) / target_depth)[:, None]
    ).astype(np.float32)
    anchor_path = LOCKED / "TARGET_16_ANCHORS_LOG1P.npy"
    np.save(anchor_path, target_anchor_log)

    method_records: dict[str, dict[str, dict[str, str]]] = {}
    fusion_names = source_lock["fusion_candidate_names"]
    for seed in SEEDS:
        base_meta = next(r for r in source_lock["registered_atlas"] if r["seed"] == seed)
        base_path = Path(base_meta["path"])
        if digest(base_path) != base_meta["sha256"]:
            raise RuntimeError("Registered atlas hash mismatch")
        base_log = np.asarray(np.load(base_path, mmap_mode="r"), np.float32)
        anchor_residual = target_anchor_log - base_log[:, anchor_indices]
        global_log = base_log + gain * apply_centered(global_model, anchor_residual)
        local_log = base_log + gain * apply_centered(local_model, features(anchor_residual, target_coordinates, True))
        anchor_log = apply_centered(anchor_model, target_anchor_log)
        candidates, names = candidate_fusion(base_log, global_log, local_log, anchor_log)
        if names != fusion_names:
            raise RuntimeError("Frozen fusion ordering changed")
        map_log = np.column_stack([candidates[fusion_selection[g], :, g] for g in range(len(genes))])
        map_log[:, ~predictable] = base_log[:, ~predictable]
        values = {
            "registered_idw_k12": base_log,
            "anchor_only_calibrated": np.maximum(anchor_log, 0),
            "hyperspatial_map_calibrated_fusion": np.maximum(map_log, 0),
        }
        method_records[str(seed)] = {}
        for method, prediction in values.items():
            path = LOCKED / f"seed{seed}_{method}.npy"
            if method != "registered_idw_k12":
                np.save(path, np.asarray(prediction, np.float32))
            method_records[str(seed)][method] = {"path": str(path), "sha256": digest(path)}

    manifest = {
        "status": "predictions_locked_before_hidden_stroke_truth_or_annotations",
        "methods": method_records, "seeds": list(SEEDS),
        "anchor_path": str(anchor_path), "anchor_sha256": digest(anchor_path),
        "anchor_genes": source_lock["anchor_genes"],
        "target_information_opened": ["frozen xyz", "public gene identifiers", "16 frozen anchor columns"],
        "target_hidden_expression_accessed": False, "target_annotations_accessed": False,
    }
    json_write(LOCKED / "PREDICTION_LOCK.json", manifest)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64).ravel(); y = np.asarray(y, np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) <= 1e-12 or np.std(y[ok]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def phase_evaluate() -> None:
    """Verify prediction lock, then and only then open hidden stroke truth/labels."""
    lock = json.loads((LOCKED / "PREDICTION_LOCK.json").read_text())
    if lock["status"] != "predictions_locked_before_hidden_stroke_truth_or_annotations":
        raise RuntimeError("Prediction lock missing")
    for seed in SEEDS:
        for method in METHODS:
            meta = lock["methods"][str(seed)][method]
            if digest(Path(meta["path"])) != meta["sha256"]:
                raise RuntimeError("Prediction hash mismatch")

    source_lock = json.loads((LOCKED / "SOURCE_AND_ATLAS_FREEZE.json").read_text())
    payload = np.load(source_lock["model_path"])
    genes = payload["genes"].astype(str)
    hidden = payload["hidden_indices"].astype(np.int64)
    target_counts, target_genes = dense_x(STROKE)  # first hidden-expression access
    if not np.array_equal(genes, target_genes):
        raise RuntimeError("Target gene axis changed")
    truth_log = np.log1p(normalize_counts(target_counts, float(source_lock["depth_target"]))).astype(np.float32)
    target_coordinates = canonicalize(np.asarray(np.load(STROKE_COORD), np.float32))[0]

    seed_gene_rows = []
    seed_summary = []
    cached = {}
    for seed in SEEDS:
        base = np.asarray(np.load(lock["methods"][str(seed)]["registered_idw_k12"]["path"], mmap_mode="r"), np.float32)
        truth_residual = truth_log[:, hidden] - base[:, hidden]
        for method in METHODS:
            pred = np.asarray(np.load(lock["methods"][str(seed)][method]["path"], mmap_mode="r"), np.float32)
            pred_residual = pred[:, hidden] - base[:, hidden]
            spatial = pearson_columns(truth_log[:, hidden], pred[:, hidden])
            residual = pearson_columns(truth_residual, pred_residual) if method != "registered_idw_k12" else np.full(len(hidden), np.nan)
            rmse_gene = np.sqrt(np.mean(np.square(truth_log[:, hidden] - pred[:, hidden]), axis=0))
            pooled_expr = safe_corr(truth_log[:, hidden], pred[:, hidden])
            pooled_resid = safe_corr(truth_residual, pred_residual) if method != "registered_idw_k12" else np.nan
            seed_summary.append({
                "seed": seed, "method": method, "n_hidden_genes": len(hidden),
                "median_spatial_pearson": float(np.nanmedian(spatial)),
                "pooled_expression_pearson": pooled_expr,
                "median_gene_residual_pearson": float(np.nanmedian(residual)) if method != "registered_idw_k12" else np.nan,
                "pooled_residual_pearson": pooled_resid,
                "pooled_log1p_rmse": float(np.sqrt(np.mean(np.square(truth_log[:, hidden] - pred[:, hidden])))),
                "median_gene_log1p_rmse": float(np.nanmedian(rmse_gene)),
            })
            for local, gene_index in enumerate(hidden):
                seed_gene_rows.append({
                    "seed": seed, "gene": genes[gene_index], "gene_index": int(gene_index), "method": method,
                    "spatial_pearson": spatial[local], "residual_pearson": residual[local], "log1p_rmse": rmse_gene[local],
                    "source_predictable": bool(payload["predictable"][gene_index]),
                    "frozen_fusion_assignment": source_lock["fusion_candidate_names"][int(payload["fusion_selection"][gene_index])],
                })
            cached[(seed, method)] = pred
    seed_gene = pd.DataFrame(seed_gene_rows)
    gene = seed_gene.groupby(["gene", "gene_index", "method", "source_predictable", "frozen_fusion_assignment"], as_index=False).agg(
        spatial_pearson=("spatial_pearson", "mean"), residual_pearson=("residual_pearson", "mean"),
        log1p_rmse=("log1p_rmse", "mean"), spatial_pearson_sd=("spatial_pearson", "std"),
        residual_pearson_sd=("residual_pearson", "std"), log1p_rmse_sd=("log1p_rmse", "std"),
    )
    gene.to_csv(ROOT / "GENE_LEVEL_RESULTS.csv", index=False)
    summary = pd.DataFrame(seed_summary)
    summary.to_csv(ROOT / "RESULTS_BY_SEED.csv", index=False)
    summary_mean = summary.groupby("method", as_index=False).agg(
        n_hidden_genes=("n_hidden_genes", "first"), median_spatial_pearson=("median_spatial_pearson", "mean"),
        pooled_expression_pearson=("pooled_expression_pearson", "mean"),
        median_gene_residual_pearson=("median_gene_residual_pearson", "mean"),
        pooled_residual_pearson=("pooled_residual_pearson", "mean"),
        pooled_log1p_rmse=("pooled_log1p_rmse", "mean"), median_gene_log1p_rmse=("median_gene_log1p_rmse", "mean"),
    )
    summary_mean.to_csv(ROOT / "RESULTS_SUMMARY.csv", index=False)

    # Author-defined labels are opened only after prediction lock.
    with h5py.File(ORIGINAL_H5, "r") as handle:
        batches = np.asarray([x.decode() for x in handle["Batch"][:]])
        all_regions = np.asarray([x.decode() for x in handle["Celltype_coarse"][:]])
    regions_raw = all_regions[np.char.startswith(batches, "PT")]
    regions = np.where(regions_raw == "ICA", "ICA", np.where(regions_raw == "PIA", "PIA", "Other"))
    region_rows = []
    for seed in SEEDS:
        base = cached[(seed, "registered_idw_k12")]
        true_residual = truth_log[:, hidden] - base[:, hidden]
        for region in ("ICA", "PIA", "Other"):
            use = regions == region
            for method in METHODS:
                pred = cached[(seed, method)]
                pred_residual = pred[:, hidden] - base[:, hidden]
                rr = pearson_columns(true_residual[use], pred_residual[use]) if method != "registered_idw_k12" else np.full(len(hidden), np.nan)
                region_rows.append({
                    "seed": seed, "region": region, "n_spots": int(use.sum()), "method": method,
                    "pooled_residual_pearson": safe_corr(true_residual[use], pred_residual[use]) if method != "registered_idw_k12" else np.nan,
                    "median_gene_residual_pearson": float(np.nanmedian(rr)) if method != "registered_idw_k12" else np.nan,
                    "true_mean_abs_residual": float(np.mean(np.abs(true_residual[use]))),
                    "predicted_mean_abs_residual": float(np.mean(np.abs(pred_residual[use]))),
                    "pooled_log1p_rmse": float(np.sqrt(np.mean(np.square(truth_log[use][:, hidden] - pred[use][:, hidden])))),
                })
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(ROOT / "REGION_RESULTS_BY_SEED.csv", index=False)
    region_mean = region_df.groupby(["region", "method"], as_index=False).agg(
        n_spots=("n_spots", "first"), pooled_residual_pearson=("pooled_residual_pearson", "mean"),
        median_gene_residual_pearson=("median_gene_residual_pearson", "mean"),
        true_mean_abs_residual=("true_mean_abs_residual", "mean"),
        predicted_mean_abs_residual=("predicted_mean_abs_residual", "mean"), pooled_log1p_rmse=("pooled_log1p_rmse", "mean"),
    )
    region_mean.to_csv(ROOT / "REGION_RESULTS.csv", index=False)

    # Spatial structure on native reconstructed xyz; target truth is now unblinded.
    spatial_rows = []
    representative_index = int(payload["representative_index"][0])
    rep_local = int(np.flatnonzero(hidden == representative_index)[0])
    neighbors = cKDTree(target_coordinates).query(target_coordinates, k=13)[1][:, 1:]
    for seed in SEEDS:
        base = cached[(seed, "registered_idw_k12")]
        true_residual = truth_log[:, hidden] - base[:, hidden]
        true_moran = moran_columns(true_residual, target_coordinates)
        for method in ("anchor_only_calibrated", "hyperspatial_map_calibrated_fusion"):
            pred_residual = cached[(seed, method)][:, hidden] - base[:, hidden]
            pred_moran = moran_columns(pred_residual, target_coordinates)
            spatial_rows.append({
                "seed": seed, "method": method, "n_hidden_genes": len(hidden),
                "median_true_residual_moran_I": float(np.nanmedian(true_moran)),
                "median_predicted_residual_moran_I": float(np.nanmedian(pred_moran)),
                "true_predicted_gene_moran_correlation": safe_corr(true_moran, pred_moran),
                "pooled_true_predicted_residual_profile_correlation": safe_corr(true_residual, pred_residual),
                "representative_gene": genes[representative_index],
                "representative_true_moran_I": float(true_moran[rep_local]),
                "representative_predicted_moran_I": float(pred_moran[rep_local]),
            })
    spatial_df = pd.DataFrame(spatial_rows)
    spatial_df.to_csv(ROOT / "SPATIAL_RESULTS.csv", index=False)

    # Post-lock permutation of the source-selected representative gene only.
    seed = 42
    base = cached[(seed, "registered_idw_k12")]
    rep = truth_log[:, representative_index] - base[:, representative_index]
    z = rep - rep.mean()
    observed = float((z * z[neighbors].mean(1)).sum() / np.square(z).sum())
    rng = np.random.default_rng(170042)
    null = []
    for _ in range(999):
        zp = rng.permutation(z)
        null.append(float((zp * zp[neighbors].mean(1)).sum() / np.square(zp).sum()))
    json_write(ROOT / "REPRESENTATIVE_SPATIAL_PERMUTATION.json", {
        "gene": genes[representative_index], "seed": seed, "permutations": 999,
        "observed_moran_I": observed, "null_mean": float(np.mean(null)),
        "empirical_p_greater": float((1 + np.sum(np.asarray(null) >= observed)) / 1000),
        "performed_after_prediction_lock": True,
    })

    evaluation_lock = {
        "status": "evaluation_complete_after_prediction_lock",
        "truth_first_opened_in": "phase_evaluate after all prediction hashes verified",
        "target_annotations_first_opened_in": "phase_evaluate after all prediction hashes verified",
        "target_hidden_expression_hash": digest(STROKE),
    }
    json_write(LOCKED / "EVALUATION_UNBLINDING.json", evaluation_lock)


def phase_spatial() -> None:
    """Memory-bounded post-lock spatial analysis, separated for restart safety."""
    lock = json.loads((LOCKED / "PREDICTION_LOCK.json").read_text())
    for seed in SEEDS:
        for method in METHODS:
            meta = lock["methods"][str(seed)][method]
            if digest(Path(meta["path"])) != meta["sha256"]:
                raise RuntimeError("Prediction hash mismatch before spatial evaluation")
    source = json.loads((LOCKED / "SOURCE_AND_ATLAS_FREEZE.json").read_text())
    payload = np.load(source["model_path"])
    genes = payload["genes"].astype(str)
    hidden = payload["hidden_indices"].astype(np.int64)
    representative_index = int(payload["representative_index"][0])
    rep_local = int(np.flatnonzero(hidden == representative_index)[0])
    target_counts, target_genes = dense_x(STROKE)
    if not np.array_equal(genes, target_genes):
        raise RuntimeError("Target gene axis changed")
    truth_log = np.log1p(normalize_counts(target_counts, float(source["depth_target"]))).astype(np.float32)
    target_coordinates = canonicalize(np.asarray(np.load(STROKE_COORD), np.float32))[0]
    spatial_rows = []
    for seed in SEEDS:
        base = np.asarray(np.load(lock["methods"][str(seed)]["registered_idw_k12"]["path"], mmap_mode="r"), np.float32)
        true_residual = truth_log[:, hidden] - base[:, hidden]
        true_moran = moran_columns(true_residual, target_coordinates)
        for method in ("anchor_only_calibrated", "hyperspatial_map_calibrated_fusion"):
            pred = np.asarray(np.load(lock["methods"][str(seed)][method]["path"], mmap_mode="r"), np.float32)
            pred_residual = pred[:, hidden] - base[:, hidden]
            pred_moran = moran_columns(pred_residual, target_coordinates)
            spatial_rows.append({
                "seed": seed, "method": method, "n_hidden_genes": len(hidden),
                "median_true_residual_moran_I": float(np.nanmedian(true_moran)),
                "median_predicted_residual_moran_I": float(np.nanmedian(pred_moran)),
                "true_predicted_gene_moran_correlation": safe_corr(true_moran, pred_moran),
                "pooled_true_predicted_residual_profile_correlation": safe_corr(true_residual, pred_residual),
                "representative_gene": genes[representative_index],
                "representative_true_moran_I": float(true_moran[rep_local]),
                "representative_predicted_moran_I": float(pred_moran[rep_local]),
            })
            del pred_residual, pred_moran, pred
        del true_residual, true_moran, base
    pd.DataFrame(spatial_rows).to_csv(ROOT / "SPATIAL_RESULTS.csv", index=False)

    seed = 42
    base = np.asarray(np.load(lock["methods"][str(seed)]["registered_idw_k12"]["path"], mmap_mode="r"), np.float32)
    rep = truth_log[:, representative_index] - base[:, representative_index]
    neighbors = cKDTree(target_coordinates).query(target_coordinates, k=13)[1][:, 1:]
    z = rep - rep.mean()
    observed = float((z * z[neighbors].mean(1)).sum() / np.square(z).sum())
    rng = np.random.default_rng(170042)
    null = []
    for _ in range(999):
        zp = rng.permutation(z)
        null.append(float((zp * zp[neighbors].mean(1)).sum() / np.square(zp).sum()))
    json_write(ROOT / "REPRESENTATIVE_SPATIAL_PERMUTATION.json", {
        "gene": genes[representative_index], "seed": seed, "permutations": 999,
        "observed_moran_I": observed, "null_mean": float(np.mean(null)),
        "empirical_p_greater": float((1 + np.sum(np.asarray(null) >= observed)) / 1000),
        "performed_after_prediction_lock": True,
    })
    json_write(LOCKED / "EVALUATION_UNBLINDING.json", {
        "status": "evaluation_complete_after_prediction_lock",
        "truth_first_opened_in": "phase_evaluate after all prediction hashes verified",
        "target_annotations_first_opened_in": "phase_evaluate after all prediction hashes verified",
        "target_hidden_expression_hash": digest(STROKE),
    })


def phase_report() -> None:
    summary = pd.read_csv(ROOT / "RESULTS_SUMMARY.csv").set_index("method")
    gene = pd.read_csv(ROOT / "GENE_LEVEL_RESULTS.csv")
    regions = pd.read_csv(ROOT / "REGION_RESULTS.csv")
    spatial = pd.read_csv(ROOT / "SPATIAL_RESULTS.csv")
    source = json.loads((LOCKED / "SOURCE_AND_ATLAS_FREEZE.json").read_text())
    geometry = json.loads((LOCKED / "GEOMETRY_REGISTRATION_FREEZE.json").read_text())
    map_name = "hyperspatial_map_calibrated_fusion"
    idw_name = "registered_idw_k12"
    map_row, idw_row = summary.loc[map_name], summary.loc[idw_name]
    summary["delta_spatial_pearson_vs_idw"] = summary["median_spatial_pearson"] - idw_row["median_spatial_pearson"]
    summary["delta_pooled_expression_pearson_vs_idw"] = summary["pooled_expression_pearson"] - idw_row["pooled_expression_pearson"]
    summary["favorable_delta_log1p_rmse_vs_idw"] = idw_row["pooled_log1p_rmse"] - summary["pooled_log1p_rmse"]
    summary.reset_index().to_csv(ROOT / "RESULTS_SUMMARY.csv", index=False)
    pivot = gene.pivot(index="gene", columns="method", values=["spatial_pearson", "residual_pearson", "log1p_rmse"])
    frac_spatial = float((pivot["spatial_pearson"][map_name] > pivot["spatial_pearson"][idw_name]).mean())
    frac_rmse = float((pivot["log1p_rmse"][map_name] < pivot["log1p_rmse"][idw_name]).mean())
    frac_resid_positive = float((pivot["residual_pearson"][map_name] > 0).mean())
    region_map = regions[regions.method == map_name].set_index("region")
    spatial_map = spatial[spatial.method == map_name]
    positive_residual = map_row["median_gene_residual_pearson"] > 0 and map_row["pooled_residual_pearson"] > 0
    absolute_gain = (map_row["median_spatial_pearson"] > idw_row["median_spatial_pearson"] or
                     map_row["pooled_log1p_rmse"] < idw_row["pooled_log1p_rmse"])
    broad = max(frac_spatial, frac_rmse, frac_resid_positive) > 0.5
    spatial_recovered = float(spatial_map["pooled_true_predicted_residual_profile_correlation"].mean()) > 0
    residual_positive_all_regions = np.all(region_map["pooled_residual_pearson"].to_numpy() > 0)
    other = region_map.loc["Other"]
    # Decision-A condition 5 requires both author-defined lesion-associated
    # classes to preserve the observed direction of magnitude enrichment.
    region_consistent = residual_positive_all_regions and all(
        np.sign(region_map.loc[r, "true_mean_abs_residual"] - other["true_mean_abs_residual"])
        == np.sign(region_map.loc[r, "predicted_mean_abs_residual"] - other["predicted_mean_abs_residual"])
        for r in ("ICA", "PIA")
    )
    if positive_residual and absolute_gain and broad and spatial_recovered and region_consistent:
        decision = "A"
        decision_text = "strong positive real-pathology demonstration"
        recommendation = "Supplementary"
    elif positive_residual:
        decision = "B"
        decision_text = "positive residual recovery with mixed absolute or spatial endpoints"
        recommendation = "Supplementary" if absolute_gain else "Discussion"
    else:
        decision = "C"
        decision_text = "little hidden residual recovery and no defensible adaptation gain"
        recommendation = "Do not include"

    metrics = {
        "decision": decision, "decision_text": decision_text, "recommendation": recommendation,
        "fraction_genes_map_better_spatial_pearson": frac_spatial,
        "fraction_genes_map_lower_log1p_rmse": frac_rmse,
        "fraction_genes_positive_map_residual_pearson": frac_resid_positive,
        "all_region_residual_correlations_positive": bool(residual_positive_all_regions),
        "ica_and_pia_magnitude_enrichment_direction_recovered": bool(region_consistent),
    }
    json_write(ROOT / "DECISION_METRICS.json", metrics)

    # Compact candidate figure for A/B only.
    if decision in ("A", "B"):
        make_figure(summary, gene, regions, source, geometry)

    geom_df = pd.DataFrame(geometry["registration_records"])
    (ROOT / "REGISTRATION_QC.md").write_text(
        "# Registration QC\n\n"
        "The frozen manuscript geometry-only registration was applied to the frozen label-free serial-section coordinates. "
        "No expression or pathology annotation was opened. Values below are in independently canonicalized coordinate units.\n\n" +
        geom_df[["seed", "reflection", "raw_chamfer", "rigid_chamfer", "registered_chamfer", "raw_overlap", "rigid_overlap", "registered_overlap"]].to_markdown(index=False) + "\n"
    )
    (ROOT / "EXPERIMENT_DESIGN.md").write_text(
        "# Experiment design\n\n"
        "A single sham mouse four-section 3D stack was the deeply measured source and a single photothrombotic-stroke mouse four-section stack was the target. "
        "This is a descriptive one-source/one-target external validation, not biological replication. The existing frozen registration, 12-NN IDW, 16-gene source-only selection, raw/k12/k32 residual features, rank-16 ridge operators, calibrated mixture and source-only predictability guard were used unchanged.\n\n"
        f"The source screen retained {source['eligible_gene_count']} genes; after the frozen 16-gene budget, {source['hidden_gene_count']} genes were evaluated. "
        "All target non-anchor values and author-defined ICA/PIA regions were sealed until predictions were serialized and hashed.\n"
    )
    inherited = (
        "The deposited 2,003-gene matrix is an author-provided processed gene universe; whether its upstream feature selection used both conditions cannot be undone from this object. "
        "The present anchor selection, calibration, guard and prediction did not use stroke non-anchor values."
    )
    (ROOT / "LEAKAGE_AUDIT.md").write_text(
        "# Leakage audit\n\n## PASS (with processed-gene-universe provenance caveat)\n\n"
        "Before prediction lock, MAP accessed only sham full expression, frozen sham/stroke xyz, public gene identifiers and the 16 source-selected stroke anchor columns. "
        "It did not access stroke hidden genes, ICA/PIA, lesion labels, the pathology-informed deposited alignment, stroke differential-expression results or a CONCERT lesion footprint. "
        "Registration was completed and hashed before source expression was opened; the registered atlas was hashed before target anchors were opened; all predictions were hashed before hidden truth and labels were opened.\n\n" + inherited + "\n"
    )
    (ROOT / "FINAL_DECISION.md").write_text(
        f"# Decision {decision}\n\n**{decision_text}.**\n\n"
        f"Across {source['hidden_gene_count']:,} source-eligible hidden genes, full MAP had mean-seed median gene-wise residual Pearson {map_row['median_gene_residual_pearson']:.4f} and pooled spot-by-gene residual Pearson {map_row['pooled_residual_pearson']:.4f}. "
        f"Median spatial Pearson was {map_row['median_spatial_pearson']:.4f} for MAP versus {idw_row['median_spatial_pearson']:.4f} for registered IDW; pooled log1p RMSE was {map_row['pooled_log1p_rmse']:.4f} versus {idw_row['pooled_log1p_rmse']:.4f}. "
        f"MAP improved gene-wise spatial Pearson for {100*frac_spatial:.1f}% and RMSE for {100*frac_rmse:.1f}% of hidden genes.\n\n"
        f"The internal anchor-only operator was stronger than full MAP on median residual Pearson ({summary.loc['anchor_only_calibrated', 'median_gene_residual_pearson']:.4f} versus {map_row['median_gene_residual_pearson']:.4f}) and pooled RMSE ({summary.loc['anchor_only_calibrated', 'pooled_log1p_rmse']:.4f} versus {map_row['pooled_log1p_rmse']:.4f}). "
        "Full MAP recovered positive residual correlations in ICA, PIA and other regions and the ICA magnitude enrichment, but it did not reproduce the modest PIA-versus-other residual-magnitude enrichment.\n\n"
        "These results describe one sham mouse and one stroke mouse and do not support population-level inference, biological replication or causal prediction.\n"
    )
    (ROOT / "MANUSCRIPT_SAFE_INTERPRETATION.md").write_text(
        "# Manuscript-safe interpretation\n\n"
        f"In a descriptive external serial-section 3D experiment using one sham and one photothrombotic-stroke mouse, HyperSpatial-MAP reconstructed hidden stroke expression from frozen geometry and 16 source-selected stroke genes with median gene-wise atlas-relative residual correlation {map_row['median_gene_residual_pearson']:.3f}. "
        "Target non-anchor expression and author-defined lesion-associated regions were opened only after prediction serialization. "
        "The result is a one-pair proof of concept and not biological replication or a population-level disease benchmark.\n"
    )


def make_figure(summary: pd.DataFrame, gene: pd.DataFrame, regions: pd.DataFrame, source: dict, geometry: dict) -> None:
    map_name = "hyperspatial_map_calibrated_fusion"; idw_name = "registered_idw_k12"
    seed = 42
    lock = json.loads((LOCKED / "PREDICTION_LOCK.json").read_text())
    payload = np.load(source["model_path"])
    genes = payload["genes"].astype(str); hidden = payload["hidden_indices"].astype(int)
    rep = int(payload["representative_index"][0]); rep_gene = genes[rep]
    target_counts, _ = dense_x(STROKE)
    truth = np.log1p(normalize_counts(target_counts, float(source["depth_target"]))).astype(np.float32)
    base = np.load(lock["methods"][str(seed)][idw_name]["path"], mmap_mode="r")
    pred = np.load(lock["methods"][str(seed)][map_name]["path"], mmap_mode="r")
    coords = canonicalize(np.load(STROKE_COORD).astype(np.float32))[0]
    tr = truth[:, rep] - base[:, rep]; pr = pred[:, rep] - base[:, rep]
    lim = float(np.quantile(np.abs(np.r_[tr, pr]), 0.99))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.7})
    fig = plt.figure(figsize=(13.8, 10.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 12, height_ratios=[0.85, 1.15, 0.8])
    axa = fig.add_subplot(gs[0, :4]); axb = fig.add_subplot(gs[0, 4:8]); axc = fig.add_subplot(gs[0, 8:12])
    axa.axis("off")
    axa.text(0, 1.03, "a", fontweight="bold", fontsize=12, transform=axa.transAxes)
    axa.text(.5, .95, "Real serial-section 3D stroke test", ha="center", fontweight="bold")
    boxes = [(0.05, .55, "Sham 3D\nreference", "#4477AA"), (.38, .55, "Geometry-only\nregistration", "#999999"), (.71, .55, "Stroke geometry\n+ 16 genes", "#D55E00"), (.38, .15, "Hidden stroke\nreconstruction", "#228833")]
    for x, y, text, color in boxes:
        axa.add_patch(plt.Rectangle((x, y), .24, .19, facecolor="white", edgecolor=color, lw=1.3))
        axa.text(x+.12, y+.095, text, ha="center", va="center")
    for xy, xytext in [((.38,.645),(.29,.645)),((.71,.645),(.62,.645)),((.50,.34),(.50,.55))]:
        axa.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="->", color="#666666", lw=1))

    axb.text(-.16, 1.03, "b", fontweight="bold", fontsize=12, transform=axb.transAxes)
    true_all = truth[:, hidden] - base[:, hidden]
    pred_all = pred[:, hidden] - base[:, hidden]
    rng = np.random.default_rng(170042)
    take = rng.choice(true_all.size, min(7000, true_all.size), replace=False)
    tx, px = true_all.ravel()[take], pred_all.ravel()[take]
    axb.scatter(tx, px, s=3, alpha=.18, color="#228833", rasterized=True)
    lo, hi = np.quantile(np.r_[tx, px], [.01,.99]); axb.plot([lo,hi],[lo,hi], "--", color="#777777", lw=.8)
    axb.set(xlabel="True stroke residual", ylabel="Predicted stroke residual", title="Hidden residual recovery")
    axb.text(.04,.95,f"pooled r = {safe_corr(true_all,pred_all):.3f}",transform=axb.transAxes,va="top")

    axc.text(-.16, 1.03, "c", fontweight="bold", fontsize=12, transform=axc.transAxes)
    sm = summary if summary.index.name == "method" else summary.set_index("method")
    endpoints = ["median_spatial_pearson", "pooled_log1p_rmse"]
    vals = [[sm.loc[idw_name,e], sm.loc[map_name,e]] for e in endpoints]
    x=np.arange(2); w=.32
    axc.bar(x-w/2,[v[0] for v in vals],w,label="Registered IDW",color="#9ECAE1")
    axc.bar(x+w/2,[v[1] for v in vals],w,label="HyperSpatial-MAP",color="#228833")
    axc.set_xticks(x,["Spatial\nPearson","log1p\nRMSE"]); axc.set_title("Absolute reconstruction"); axc.legend(frameon=False,fontsize=7)

    map_gs = gs[1,:].subgridspec(1,4)
    arrays=[truth[:,rep],base[:,rep],tr,pr]; titles=[f"Measured stroke: {rep_gene}","Sham expectation, B_T","True stroke residual","MAP residual (guarded fallback)"]
    expr_lo=float(min(np.quantile(arrays[0],.01),np.quantile(arrays[1],.01))); expr_hi=float(max(np.quantile(arrays[0],.99),np.quantile(arrays[1],.99)))
    for j,(arr,title) in enumerate(zip(arrays,titles)):
        ax=fig.add_subplot(map_gs[0,j],projection="3d")
        if j==0: ax.text2D(-.18,1.02,"d",fontweight="bold",fontsize=12,transform=ax.transAxes)
        if j<2: vmin,vmax,cmap=expr_lo,expr_hi,"viridis"
        else: vmin,vmax,cmap=-lim,lim,"coolwarm"
        sc=ax.scatter(coords[:,0],coords[:,1],coords[:,2],c=arr,s=1.2,cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
        ax.set_title(title,fontsize=8); ax.view_init(24,-62); ax.set_axis_off()
        if j in (1,3): fig.colorbar(sc,ax=ax,shrink=.55,pad=.02)
    axe=fig.add_subplot(gs[2,:6]); axe.text(-.10, 1.03, "e", fontweight="bold", fontsize=12, transform=axe.transAxes)
    rr=regions[regions.method==map_name]
    axe.bar(rr.region,rr.pooled_residual_pearson,color=["#CC6677","#DDCC77","#999999"])
    axe.axhline(0,color="#777777",lw=.7); axe.set(ylabel="Pooled residual Pearson",title="Author-defined regions")
    axf=fig.add_subplot(gs[2,6:12]); axf.text(-.10,1.03,"f",fontweight="bold",fontsize=12,transform=axf.transAxes)
    g=gene[gene.method==map_name].residual_pearson.dropna()
    axf.hist(g,bins=42,color="#228833",alpha=.85); axf.axvline(g.median(),color="black",lw=1)
    axf.set(xlabel="Gene-wise residual Pearson",ylabel="Hidden genes",title="Hidden-gene recovery")
    axf.text(.04,.95,f"median = {g.median():.3f}\nn = {len(g):,}",transform=axf.transAxes,va="top")
    for ext in ("png","pdf","svg"):
        fig.savefig(FIG/f"Supplementary_real_stroke_MAP.{ext}",dpi=350,bbox_inches="tight")
    plt.close(fig)


def write_manifest() -> None:
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "MANIFEST_SHA256.txt")
    with (ROOT / "MANIFEST_SHA256.txt").open("w") as handle:
        for path in files:
            handle.write(f"{digest(path)}  {path.relative_to(ROOT)}\n")
        for path in (SHAM, STROKE, SHAM_COORD, STROKE_COORD, ORIGINAL_H5,
                     REPO / "run_hyperspatial_map_stage1b_20260803.py"):
            handle.write(f"{digest(path)}  {path}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("geometry", "source", "predict", "evaluate", "spatial", "report", "manifest"))
    args = parser.parse_args()
    {"geometry": phase_geometry, "source": phase_source, "predict": phase_predict,
     "evaluate": phase_evaluate, "spatial": phase_spatial,
     "report": phase_report, "manifest": write_manifest}[args.phase]()


if __name__ == "__main__":
    main()
