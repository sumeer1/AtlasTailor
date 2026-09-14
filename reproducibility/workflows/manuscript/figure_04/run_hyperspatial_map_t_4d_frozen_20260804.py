#!/usr/bin/env python3
"""Protocol-locked HyperSpatial-MAP-T cross-fitted 50%-to-75% experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_map_t")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

from hyperspatial_map_t_core_20260804 import (
    LateTruthVault, apply_centered, apply_depth, assert_family_split,
    assert_no_identifier_features, canonicalize, depth_model, domain_metrics,
    family_blocks, fit_centered_ridge_rank, future_residuals, idw, json_dump_new,
    lineage_transport, moran, multiscale, normalize_counts, pearson_columns,
    pooled_pearson, read_counts, read_gene_names, read_geometry, read_ids, sha256,
    smooth, spearman_columns, temporal_guard, tracking_overlap,
)
from run_hyperspatial_map_stage1b_20260803 import (
    candidate_fusion, select_non_axial_anchors, source_gate,
)
from run_hyperspatial_map_stage1_20260803 import GAIN_GRID, RIDGE_GRID
from run_mouse_temporal_transport_gate import best_rigid, source_svg_indices


ROOT = Path("results_v10/hyperspatial_map_t_4d_frozen_20260804_112617")
INPUTS = Path("results_v10/hyperspatial_map_t_4d_frozen_20260804_105810/00_data_audit/inputs")
GENES = 495
ANCHORS = 16
RANK = 16
SEED = 42
N_BOOT = 2000
N_PERM = 100
FOLDS = {
    "A": {"source": "E1", "target": "E2", "dir": "03_fold_A"},
    "B": {"source": "E2", "target": "E1", "dir": "04_fold_B"},
}
EXPECTED = {
    "matched_A_50p_E1_sphere_logFC.h5ad": (95826927, "0955a9cdaebb05af9cbd43ce19ec4708d5f2b00f1126e7e68fbad798c0427eee"),
    "matched_A_50p_E2_sphere_logFC.h5ad": (98043322, "e8fa2361037fd01af40dde3aa210d3b35d3785f31ab7ee688469046921f5e467"),
    "matched_B_75p_E1_sphere_logFC.h5ad": (139027006, "1ec253632c60774436f7082cf6bc9fd6b1dfb2b43915faadf79c1541d32d1b7b"),
    "matched_B_75p_E2_sphere_logFC.h5ad": (143213048, "6339c1b4574b3c6418a7d188632932407847e5084b83e294253ba765bf387b11"),
    "Tracking_Matrix.csv": (168261126, "c3336821760775e9eee40c15852213bceb735d630b932ae212887767ff6988bc"),
}
METHODS = (
    "B1_lineage_registered_idw", "B2_lineage_early_map",
    "B3_mean_developmental_correction", "B4_anchor_only_temporal",
    "B5_trajectory_free_map_t", "B6_trajectory_aware_map_t",
)
COLORS = {"B1_lineage_registered_idw": "#E69F00", "B2_lineage_early_map": "#7F7F7F",
          "B3_mean_developmental_correction": "#56B4E9", "B4_anchor_only_temporal": "#CC79A7",
          "B5_trajectory_free_map_t": "#009E73", "B6_trajectory_aware_map_t": "#D62728"}


def write_tsv_new(path, rows):
    rows = list(rows)
    if not rows:
        path.open("x").write("")
        return
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def ensure_layout():
    if not ROOT.exists():
        raise RuntimeError("Active result root must be created before execution")
    for name in ("00_data_audit", "01_protocol_lock", "02_source_validation", "03_fold_A",
                 "04_fold_B", "05_baselines", "06_tracking_overlap", "07_overlap_excluded_retraining",
                 "08_bootstrap", "09_permutations", "10_figures_main", "11_figures_supplement",
                 "12_source_data", "13_models", "14_logs"):
        (ROOT / name).mkdir(exist_ok=True)


def audit_inputs():
    rows = []
    for name, (size, digest) in EXPECTED.items():
        path = INPUTS / name
        actual = sha256(path)
        if path.stat().st_size != size or actual != digest:
            raise RuntimeError(f"Integrity failure: {name}")
        rows.append({"file": name, "path": str(path.resolve()), "bytes": size, "sha256": actual})
    write_tsv_new(ROOT / "00_data_audit/FILE_HASHES.tsv", rows)
    gene_axes = [read_gene_names(INPUTS / f"matched_{stage}_{embryo}_sphere_logFC.h5ad")
                 for embryo in ("E1", "E2") for stage in ("A_50p", "B_75p")]
    if not all(np.array_equal(gene_axes[0], axis) for axis in gene_axes[1:]):
        raise RuntimeError("Gene axes differ")
    audit = {
        "status": "eligible", "measured_genes": int(len(gene_axes[0])),
        "tracking_timepoints": list(range(231)), "tracking_interval_seconds": 90,
        "lineage": {"E1": {"ancestors": 3907, "descendants": 6807},
                    "E2": {"ancestors": 3998, "descendants": 7028}},
        "target_late_access_policy": "sealed until predictions, checksums and sentinel",
        "provenance_qualification": "E1 and E2 partly reuse one published tracking map; molecular embryos are independent, tracking systems are not",
        "prior_schema_audit_limitation": "A prior eligibility-only AnnData schema inspection materialized target-late obs tables containing logFC-named columns; no values were printed, modeled or used for decisions. This run uses selective HDF5 access.",
    }
    json_dump_new(ROOT / "00_data_audit/DATA_MANIFEST.json", audit)
    (ROOT / "00_data_audit/DATASET_ELIGIBILITY_REPORT.md").open("x").write(
        "# Dataset eligibility\n\n**ELIGIBLE.** All five downloaded Dryad files match published byte sizes and frozen SHA256 values. "
        "The 495-gene axes agree; E1 links 3,907 TM0 ancestors to 6,807 TM230 descendants and E2 links "
        "3,998 to 7,028, with no many-parent events and complete late-to-TM0 ancestry. The experiment is "
        "tracking-map-conditioned cross-embryo molecular forecasting, not validation in two independently live-imaged systems.\n")
    return gene_axes[0]


def read_tracking(relevant_ancestors):
    names = ["node id", "timepoint", "x", "y", "z", "number of children", "number of parents",
             "parent id", "ancestor id"]
    chunks = []
    for chunk in pd.read_csv(INPUTS / "Tracking_Matrix.csv", usecols=names, chunksize=250000):
        keep = chunk["ancestor id"].isin(relevant_ancestors)
        if keep.any(): chunks.append(chunk.loc[keep].copy())
    frame = pd.concat(chunks, ignore_index=True)
    for name in ("node id", "timepoint", "number of children", "number of parents", "parent id", "ancestor id"):
        frame[name] = frame[name].astype(np.int64)
    return frame


def build_tracking_data(frame, embryo, early_ids, late_ids):
    relevant = set(map(int, early_ids["Amat_id"]))
    ef = frame[frame["ancestor id"].isin(relevant)]
    node_sets = {int(a): set(map(int, g["node id"].values)) for a, g in ef.groupby("ancestor id", sort=False)}
    node = ef.set_index("node id")
    ancestor_for_late = np.asarray([int(node.at[int(n), "ancestor id"]) for n in late_ids["Amat_id"]], np.int64)
    start_lookup = {int(a): i for i, a in enumerate(early_ids["Amat_id"])}
    if any(int(a) not in start_lookup for a in ancestor_for_late):
        raise RuntimeError(f"{embryo}: late ancestor absent from matched early cells")
    starts = np.asarray([[node.at[int(a), "x"], node.at[int(a), "y"], node.at[int(a), "z"]]
                         for a in ancestor_for_late], np.float32)
    endpoints = np.asarray([[node.at[int(n), "x"], node.at[int(n), "y"], node.at[int(n), "z"]]
                            for n in late_ids["Amat_id"]], np.float32)
    displacement = endpoints - starts
    path_length = np.zeros(len(late_ids["Amat_id"]), np.float32)
    max_speed = np.zeros_like(path_length)
    curvature = np.zeros_like(path_length)
    lineage_depth = np.zeros_like(path_length)
    for i, endpoint in enumerate(late_ids["Amat_id"]):
        points = []
        current = int(endpoint)
        while current >= 0:
            row = node.loc[current]
            points.append((float(row["x"]), float(row["y"]), float(row["z"])))
            if int(row["timepoint"]) == 0: break
            current = int(row["parent id"])
            if len(points) > 232: raise RuntimeError("Tracking parent cycle")
        p = np.asarray(points[::-1], np.float32)
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        path_length[i] = step.sum(); max_speed[i] = step.max(initial=0) / 90.; lineage_depth[i] = len(p) - 1
        if len(p) > 2:
            v = np.diff(p, axis=0)
            cos = np.sum(v[:-1] * v[1:], 1) / np.maximum(np.linalg.norm(v[:-1], axis=1) * np.linalg.norm(v[1:], axis=1), 1e-8)
            curvature[i] = np.mean(np.arccos(np.clip(cos, -1, 1)))
    family_counts = pd.Series(ancestor_for_late).value_counts().to_dict()
    descendants = np.asarray([family_counts[int(a)] for a in ancestor_for_late], np.float32)
    division = (descendants > 1).astype(np.float32)
    # Source-independent trajectory-neighbourhood descriptors.
    k = min(13, len(endpoints))
    ns = cKDTree(starts).query(starts, k=k)[1][:, 1:]
    ne = cKDTree(endpoints).query(endpoints, k=k)[1][:, 1:]
    density0 = 1 / np.maximum(cKDTree(starts).query(starts, k=k)[0][:, -1], 1e-5) ** 3
    density1 = 1 / np.maximum(cKDTree(endpoints).query(endpoints, k=k)[0][:, -1], 1e-5) ** 3
    overlap = np.asarray([len(set(ancestor_for_late[ns[i]]) & set(ancestor_for_late[ne[i]])) /
                          max(len(set(ancestor_for_late[ns[i]]) | set(ancestor_for_late[ne[i]])), 1)
                          for i in range(len(endpoints))], np.float32)
    coherence = np.zeros(len(endpoints), np.float32)
    for i in range(len(endpoints)):
        local = displacement[ne[i]].mean(0)
        coherence[i] = np.dot(displacement[i], local) / max(np.linalg.norm(displacement[i]) * np.linalg.norm(local), 1e-8)
    direct = np.linalg.norm(displacement, axis=1)
    trajectory = np.c_[starts, endpoints, displacement, direct, path_length,
                       path_length / np.maximum(lineage_depth * 90., 1), max_speed,
                       path_length / np.maximum(direct, 1e-5), curvature, division,
                       descendants, lineage_depth, density0, density1, density1 - density0,
                       overlap, 1 - overlap, coherence].astype(np.float32)
    names = ["start_x", "start_y", "start_z", "endpoint_x", "endpoint_y", "endpoint_z",
             "displacement_x", "displacement_y", "displacement_z", "displacement_magnitude",
             "path_length", "mean_velocity", "maximum_velocity", "tortuosity", "curvature",
             "division_indicator", "number_descendants", "tree_depth", "density_tm0",
             "density_tm230", "density_change", "neighbourhood_overlap", "neighbourhood_turnover",
             "local_displacement_coherence"]
    assert_no_identifier_features(names)
    return {"ancestor": ancestor_for_late, "starts": starts, "endpoints": endpoints,
            "features": trajectory, "feature_names": names, "node_sets": node_sets,
            "division": division, "path_length": path_length, "displacement": direct,
            "turnover": 1-overlap}


def registered_source(source_xyz, target_xyz):
    # Frozen geometry method: best rigid/reflection followed by the existing
    # diffeomorphic field. CPU is used because no CUDA device is visible.
    from hyperspatial_molecular_hologram import RegistrationConfig, fit_geometry_field
    rigid, reflection, candidates = best_rigid(source_xyz, target_xyz)
    device = torch.device("cpu")
    field, history = fit_geometry_field(rigid, target_xyz, device,
                                        RegistrationConfig(epochs=30, batch_size=768, seed=SEED))
    with torch.no_grad():
        mapped = field(torch.as_tensor(rigid, dtype=torch.float32)).numpy().astype(np.float32)
    raw = .5 * (cKDTree(target_xyz).query(source_xyz)[0].mean() + cKDTree(source_xyz).query(target_xyz)[0].mean())
    final = .5 * (cKDTree(target_xyz).query(mapped)[0].mean() + cKDTree(mapped).query(target_xyz)[0].mean())
    return mapped, {"reflection": reflection, "raw_chamfer": float(raw), "registered_chamfer": float(final),
                    "final_loss": float(history["loss"][-1]), "device": "cpu"}


def early_map_fit_predict(source_counts, source_xyz, target_anchor_counts, target_xyz, genes, anchor_indices):
    depth_target = float(np.median(np.maximum(source_counts.sum(1), 1)))
    source_values = normalize_counts(source_counts, depth_target).astype(np.float32)
    source_log = np.log1p(source_values).astype(np.float32)
    dmodel = depth_model(source_counts, anchor_indices)
    target_depth = apply_depth(target_anchor_counts, dmodel)
    target_anchor_log = np.log1p(target_anchor_counts * (depth_target / target_depth)[:, None]).astype(np.float32)
    blocks = family_blocks(source_xyz)
    # OOF registered-IDW analogue for source pseudo-family validation.
    pseudo_base = np.empty_like(source_values)
    for block in np.unique(blocks):
        va = blocks == block; tr = ~va
        pseudo_base[va] = idw(source_xyz[tr], source_values[tr], source_xyz[va], 12)
    pseudo_log = np.log1p(pseudo_base)
    residual = source_log - pseudo_log
    raw = residual[:, anchor_indices]
    local = multiscale(raw, source_xyz)
    best = None
    for ridge in RIDGE_GRID:
        oof_components = []
        for block in np.unique(blocks):
            va = blocks == block; tr = ~va
            gm = fit_centered_ridge_rank(raw[tr], residual[tr], ridge, RANK)
            lm = fit_centered_ridge_rank(local[tr], residual[tr], ridge, RANK)
            am = fit_centered_ridge_rank(source_log[tr][:, anchor_indices], source_log[tr], ridge, RANK)
            oof_components.append((va, apply_centered(gm, raw[va]), apply_centered(lm, local[va]),
                                   apply_centered(am, source_log[va][:, anchor_indices])))
        for gain in GAIN_GRID:
            predictions = np.empty((11, len(source_log), len(genes)), np.float32)
            names = None
            for va, global_residual, local_residual, a in oof_components:
                g = pseudo_log[va] + gain * global_residual
                l = pseudo_log[va] + gain * local_residual
                cand, names = candidate_fusion(pseudo_log[va], g, l, a)
                predictions[:, va] = cand
            selection, guard, meta = source_gate(predictions, source_log, pseudo_log)
            selected = np.column_stack([predictions[selection[j], :, j] for j in range(len(genes))])
            key = (float(np.nanmedian(pearson_columns(source_log, selected))),
                   -float(np.sqrt(np.mean((source_log-selected)**2))), int(guard.sum()))
            if best is None or key > best[0]:
                best = (key, float(ridge), float(gain), names, selection.copy(), guard.copy(), meta, selected.copy())
    _, ridge, gain, names, selection, guard, meta, source_early_oof = best
    guard[anchor_indices] = False
    source_early_oof[:, ~guard] = pseudo_log[:, ~guard]
    gm = fit_centered_ridge_rank(raw, residual, ridge, RANK)
    lm = fit_centered_ridge_rank(local, residual, ridge, RANK)
    am = fit_centered_ridge_rank(source_log[:, anchor_indices], source_log, ridge, RANK)
    mapped, geometry = registered_source(source_xyz, target_xyz)
    base_target = np.log1p(idw(mapped, source_values, target_xyz, 12)).astype(np.float32)
    target_residual = target_anchor_log - base_target[:, anchor_indices]
    g = base_target + gain * apply_centered(gm, target_residual)
    l = base_target + gain * apply_centered(lm, multiscale(target_residual, target_xyz))
    a = apply_centered(am, target_anchor_log)
    candidates, check = candidate_fusion(base_target, g, l, a)
    if check != names: raise AssertionError("Early MAP candidates changed")
    target_map = np.column_stack([candidates[selection[j], :, j] for j in range(len(genes))])
    target_map[:, ~guard] = base_target[:, ~guard]
    return {
        "source_log": source_log, "source_values": source_values, "depth_target": depth_target,
        "source_oof_base": pseudo_log, "source_oof_map": np.maximum(source_early_oof, 0),
        "source_anchor_residual": raw, "target_base": base_target,
        "target_map": np.maximum(target_map, 0), "target_anchor_log": target_anchor_log,
        "target_anchor_residual": target_residual, "guard": guard, "fusion_selection": selection,
        "config": {"ridge": ridge, "gain": gain, "candidate_names": names,
                   "predictable_genes": int(guard.sum()), "geometry": geometry,
                   "depth_target": depth_target},
    }


def low_rank_context(source_matrix, target_matrix, rank=16):
    mean = source_matrix.mean(0)
    _, _, vt = np.linalg.svd(source_matrix - mean, full_matrices=False)
    basis = vt[:rank].T.astype(np.float32)
    return ((source_matrix - mean) @ basis).astype(np.float32), ((target_matrix - mean) @ basis).astype(np.float32), basis


def feature_matrices(early, source_tracking, target_tracking, source_index, target_index, source_depth):
    src_anchor = early["source_anchor_residual"][source_index]
    tgt_anchor = early["target_anchor_residual"][target_index]
    src_multi = multiscale(src_anchor, canonicalize(source_tracking["endpoints"])[0])
    tgt_multi = multiscale(tgt_anchor, canonicalize(target_tracking["endpoints"])[0])
    src_context, tgt_context, context_basis = low_rank_context(
        early["source_oof_map"][source_index], early["target_map"][target_index])
    src_residual_latent, tgt_residual_latent, residual_basis = low_rank_context(
        (early["source_oof_map"] - early["source_oof_base"])[source_index],
        (early["target_map"] - early["target_base"])[target_index])
    src_depth = np.log1p(source_depth[source_index, None]).astype(np.float32)
    # Target depth is predicted from permitted anchors; use its normalized early estimate.
    tgt_depth = np.log1p(np.maximum(np.expm1(early["target_map"]).sum(1), 1))[target_index, None].astype(np.float32)
    molecular_names = ([f"anchor_residual_raw_{i}" for i in range(ANCHORS)]
                       + [f"anchor_residual_k12_{i}" for i in range(ANCHORS)]
                       + [f"anchor_residual_k32_{i}" for i in range(ANCHORS)]
                       + [f"registered_context_pc{i}" for i in range(RANK)]
                       + ["source_trained_depth"]
                       + [f"early_map_residual_pc{i}" for i in range(RANK)])
    source_molecular = np.c_[src_multi, src_context, src_depth, src_residual_latent].astype(np.float32)
    target_molecular = np.c_[tgt_multi, tgt_context, tgt_depth, tgt_residual_latent].astype(np.float32)
    assert_no_identifier_features(molecular_names)
    return source_molecular, target_molecular, molecular_names, context_basis, residual_basis


def crossfit_temporal(features, truth_residual, blocks, ridge_grid=RIDGE_GRID):
    oof_by_ridge = {}
    for ridge in ridge_grid:
        oof = np.empty_like(truth_residual)
        for block in np.unique(blocks):
            va = blocks == block; tr = ~va
            model = fit_centered_ridge_rank(features[tr], truth_residual[tr], ridge, RANK)
            oof[va] = apply_centered(model, features[va])
        key = (float(np.nanmedian(pearson_columns(truth_residual, oof))), -float(np.mean((truth_residual-oof)**2)))
        oof_by_ridge[float(ridge)] = (key, oof)
    ridge = max(oof_by_ridge, key=lambda r: oof_by_ridge[r][0])
    oof = oof_by_ridge[ridge][1]
    candidates = np.stack([gain * oof for gain in GAIN_GRID], axis=0)
    selection, guard, meta = temporal_guard(np.zeros_like(oof), candidates, truth_residual)
    selected_oof = np.column_stack([candidates[selection[j], :, j] for j in range(truth_residual.shape[1])])
    selected_oof[:, ~guard] = 0
    model = fit_centered_ridge_rank(features, truth_residual, ridge, RANK)
    return {"ridge": ridge, "selection": selection, "guard": guard, "meta": meta,
            "oof": selected_oof, "model": model,
            "output_low": np.quantile(truth_residual,.005,axis=0).astype(np.float32),
            "output_high": np.quantile(truth_residual,.995,axis=0).astype(np.float32)}


def apply_temporal(fitted, features):
    raw = apply_centered(fitted["model"], features)
    candidates = np.stack([gain * raw for gain in GAIN_GRID], axis=0)
    out = np.column_stack([candidates[fitted["selection"][j], :, j] for j in range(raw.shape[1])])
    out[:, ~fitted["guard"]] = 0
    out=np.clip(out,fitted["output_low"][None,:],fitted["output_high"][None,:])
    out[:, ~fitted["guard"]] = 0
    return out.astype(np.float32)


def prediction_hashes(fold_dir, prediction_paths):
    rows = [{"method": name, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in prediction_paths.items()]
    manifest = fold_dir / "PRE_UNBLINDING_PREDICTION_HASHES.tsv"
    write_tsv_new(manifest, rows)
    sentinel = {"status": "predictions_frozen_target_truth_may_now_be_opened",
                "prediction_hash_manifest_sha256": sha256(manifest), "timestamp_utc": pd.Timestamp.utcnow().isoformat()}
    json_dump_new(fold_dir / "UNBLINDING_SENTINEL.json", sentinel)


def save_model(path, fitted, feature_names):
    np.savez_compressed(path, x_mean=fitted["model"]["x_mean"], x_scale=fitted["model"]["x_scale"],
                        y_mean=fitted["model"]["y_mean"], weight=fitted["model"]["weight"],
                        selection=fitted["selection"], guard=fitted["guard"], ridge=fitted["ridge"],
                        output_low=fitted["output_low"],output_high=fitted["output_high"],
                        feature_names=np.asarray(feature_names))


def evaluation_rows(fold, genes, truth, predictions, lineage_early, xyz, anchor_indices, thresholds, cohorts, source_svg):
    eval_genes = np.asarray([i for i in range(len(genes)) if i not in set(anchor_indices)], np.int64)
    rows, gene_rows, domain_rows = [], [], []
    for cohort, mask in cohorts.items():
        cohort_xyz=xyz[mask]
        neighbors=cKDTree(cohort_xyz).query(cohort_xyz,k=min(13,len(cohort_xyz)))[1][:,1:]
        full_neighbors=cKDTree(xyz).query(xyz,k=min(13,len(xyz)))[1][:,1:] if cohort == "full" else None
        def moran_fast(values):
            centered=values-values.mean(); denom=np.sum(centered*centered)
            return np.nan if denom<1e-12 else float(np.sum(centered[:,None]*centered[neighbors])/(max(neighbors.shape[1],1)*denom))
        for method, pred in predictions.items():
            rtrue, rpred = future_residuals(truth[mask][:, eval_genes], pred[mask][:, eval_genes], lineage_early[mask][:, eval_genes])
            pg = pearson_columns(rtrue, rpred)
            fp = pearson_columns(truth[mask][:, eval_genes], pred[mask][:, eval_genes])
            sp = spearman_columns(truth[mask][:, eval_genes], pred[mask][:, eval_genes])
            rows.append({"fold": fold, "cohort": cohort, "method": method, "cells": int(mask.sum()),
                         "ancestors": int(len(np.unique(cohorts[cohort+"__families"])) if cohort+"__families" in cohorts else 0),
                         "genes": len(eval_genes), "rho_future_pooled": pooled_pearson(rtrue, rpred),
                         "rho_future_macro": float(np.nanmean(pg)), "rho_future_median": float(np.nanmedian(pg)),
                         "final_pearson_macro": float(np.nanmean(fp)), "final_spearman_macro": float(np.nanmean(sp)),
                         "rmse": float(np.sqrt(np.mean((truth[mask][:, eval_genes]-pred[mask][:, eval_genes])**2))),
                         "normalized_rmse": float(np.sqrt(np.mean((truth[mask][:, eval_genes]-pred[mask][:, eval_genes])**2)) /
                                                  max(np.std(truth[mask][:, eval_genes]), 1e-8)),
                         "moran_error": float(np.nanmedian([abs(moran_fast(truth[mask][:, g])-moran_fast(pred[mask][:, g])) for g in source_svg[:50]]))})
            if cohort == "full":
                for local, gene in enumerate(eval_genes):
                    gene_rows.append({"fold": fold, "method": method, "gene": genes[gene],
                                      "future_residual_pearson": pg[local], "final_pearson": fp[local],
                                      "final_spearman": sp[local],
                                      "rmse": float(np.sqrt(np.mean((truth[:, gene]-pred[:, gene])**2)))})
                for gene in source_svg[:100]:
                    if gene in set(anchor_indices): continue
                    dm = domain_metrics(truth[:, gene], pred[:, gene], xyz, thresholds[gene],full_neighbors)
                    domain_rows.append({"fold": fold, "method": method, "gene": genes[gene], **dm})
    return rows, gene_rows, domain_rows


def family_bootstrap(fold, method_predictions, truth, lineage_early, ancestors, eval_genes, cohorts, n=N_BOOT):
    rng = np.random.default_rng(42000 + ord(fold))
    records = []
    for cohort, allowed in (("full", np.unique(ancestors)),
                            ("shared", np.unique(ancestors[cohorts["shared"]])),
                            ("disjoint", np.unique(ancestors[cohorts["disjoint"]]))):
        family_indices = {int(a): np.flatnonzero(ancestors == a) for a in allowed}
        sufficient={}
        for method,pred in method_predictions.items():
            fam=[]
            for a in allowed:
                idx=family_indices[int(a)]
                x=(truth[idx][:,eval_genes]-lineage_early[idx][:,eval_genes]).ravel().astype(np.float64)
                y=(pred[idx][:,eval_genes]-lineage_early[idx][:,eval_genes]).ravel().astype(np.float64)
                fam.append((len(x),x.sum(),y.sum(),np.dot(x,x),np.dot(y,y),np.dot(x,y)))
            sufficient[method]=np.asarray(fam,np.float64)
        distributions = {m: [] for m in method_predictions}
        for _ in range(n):
            sampled = rng.integers(0,len(allowed),len(allowed))
            for method in method_predictions:
                count,sx,sy,sxx,syy,sxy=sufficient[method][sampled].sum(0)
                cov=sxy-sx*sy/count; vx=sxx-sx*sx/count; vy=syy-sy*sy/count
                distributions[method].append(float(cov/np.sqrt(max(vx*vy,1e-16))))
        for method, values in distributions.items():
            values = np.asarray(values)
            records.append({"fold": fold, "cohort": cohort, "method": method,
                            "estimate": float(np.mean(values)), "ci95_low": float(np.quantile(values,.025)),
                            "ci95_high": float(np.quantile(values,.975)), "bootstrap_unit": "TM0_ancestor_family",
                            "replicates": n})
    return records


def run_fold(fold, spec, genes, tracking_by_embryo, anchors_by_source, overlap):
    source, target = spec["source"], spec["target"]
    fold_dir = ROOT / spec["dir"]
    early_source_path = INPUTS / f"matched_A_50p_{source}_sphere_logFC.h5ad"
    early_target_path = INPUTS / f"matched_A_50p_{target}_sphere_logFC.h5ad"
    late_source_path = INPUTS / f"matched_B_75p_{source}_sphere_logFC.h5ad"
    late_target_path = INPUTS / f"matched_B_75p_{target}_sphere_logFC.h5ad"
    es_ids, et_ids = read_ids(early_source_path), read_ids(early_target_path)
    ls_ids, lt_ids = read_ids(late_source_path), read_ids(late_target_path)
    source_track, target_track = tracking_by_embryo[source], tracking_by_embryo[target]
    source_counts = read_counts(early_source_path)
    source_xyz = canonicalize(read_geometry(early_source_path))[0]
    target_xyz = canonicalize(read_geometry(early_target_path))[0]
    anchor_indices = anchors_by_source[source]
    target_anchor = read_counts(early_target_path, anchor_indices)
    early = early_map_fit_predict(source_counts, source_xyz, target_anchor, target_xyz, genes, anchor_indices)
    source_L, source_index = lineage_transport(early["source_oof_map"], source_track["ancestor"], es_ids["Amat_id"])
    source_L_idw, _ = lineage_transport(early["source_oof_base"], source_track["ancestor"], es_ids["Amat_id"])
    target_L, target_index = lineage_transport(early["target_map"], target_track["ancestor"], et_ids["Amat_id"])
    target_L_idw, _ = lineage_transport(early["target_base"], target_track["ancestor"], et_ids["Amat_id"])
    source_late_counts = read_counts(late_source_path)
    source_truth = np.log1p(normalize_counts(source_late_counts, early["depth_target"])).astype(np.float32)
    D = source_truth - source_L
    source_depth = np.maximum(source_counts.sum(1), 1)
    source_molecular, target_molecular, molecular_names, context_basis, residual_basis = feature_matrices(
        early, source_track, target_track, source_index, target_index, source_depth)
    traj_names = source_track["feature_names"]
    assert traj_names == target_track["feature_names"]
    traj_mean, traj_scale = source_track["features"].mean(0), np.maximum(source_track["features"].std(0), 1e-5)
    source_traj = (source_track["features"] - traj_mean) / traj_scale
    target_traj_raw = (target_track["features"] - traj_mean) / traj_scale
    # Conservative source-support guard fixed before target-late unblinding.
    traj_low=np.min(source_traj,axis=0); traj_high=np.max(source_traj,axis=0)
    target_traj=np.clip(target_traj_raw,traj_low,traj_high)
    full_names = molecular_names + traj_names
    assert_no_identifier_features(full_names)
    source_full = np.c_[source_molecular, source_traj].astype(np.float32)
    target_full = np.c_[target_molecular, target_traj].astype(np.float32)
    source_anchor_abs = early["source_log"][source_index][:, anchor_indices]
    target_anchor_abs = early["target_anchor_log"][target_index]
    blocks_by_family = family_blocks(canonicalize(read_geometry(early_source_path))[0])
    block_lookup = {int(a): int(b) for a,b in zip(es_ids["Amat_id"], blocks_by_family)}
    descendant_blocks = np.asarray([block_lookup[int(a)] for a in source_track["ancestor"]], np.int64)
    assert_family_split(source_track["ancestor"], descendant_blocks)
    fits = {
        "B4": crossfit_temporal(source_anchor_abs, D, descendant_blocks),
        "B5": crossfit_temporal(source_molecular, D, descendant_blocks),
        "B6": crossfit_temporal(source_full, D, descendant_blocks),
    }
    for name, fit in fits.items(): fit["guard"][anchor_indices] = False
    uncertainty_gene=np.std(D-fits["B6"]["oof"],axis=0).astype(np.float32)
    pred_residuals = {"B4": apply_temporal(fits["B4"], target_anchor_abs),
                      "B5": apply_temporal(fits["B5"], target_molecular),
                      "B6": apply_temporal(fits["B6"], target_full)}
    mean_D = D.mean(0, keepdims=True)
    predictions = {
        "B1_lineage_registered_idw": target_L_idw,
        "B2_lineage_early_map": target_L,
        "B3_mean_developmental_correction": target_L + mean_D,
        "B4_anchor_only_temporal": target_L + pred_residuals["B4"],
        "B5_trajectory_free_map_t": target_L + pred_residuals["B5"],
        "B6_trajectory_aware_map_t": target_L + pred_residuals["B6"],
    }
    predictions = {k: np.maximum(v, 0).astype(np.float32) for k,v in predictions.items()}
    # Frozen cohorts based solely on complete tracking node sets.
    shared_families, disjoint_families = overlap[(source, target)]
    cohorts = {"full": np.ones(len(target_track["ancestor"]), bool),
               "shared": np.isin(target_track["ancestor"], list(shared_families)),
               "disjoint": np.isin(target_track["ancestor"], list(disjoint_families))}
    cohort_rows=[]
    for name, families in (("full", set(map(int,np.unique(target_track["ancestor"])))),
                           ("shared", shared_families), ("disjoint", disjoint_families)):
        mask=cohorts[name]
        cohort_rows.append({"fold":fold,"cohort":name,"ancestor_count":len(families),"descendant_count":int(mask.sum()),
                            "division_fraction":float(target_track["division"][mask].mean()) if mask.any() else np.nan,
                            "median_path_length":float(np.median(target_track["path_length"][mask])) if mask.any() else np.nan})
    write_tsv_new(ROOT / f"06_tracking_overlap/fold_{fold}_cohorts.tsv", cohort_rows)
    np.savez_compressed(ROOT / f"06_tracking_overlap/fold_{fold}_membership.npz",
                        ancestor=target_track["ancestor"], shared=cohorts["shared"], disjoint=cohorts["disjoint"])
    # Save models/guards and predictions before any late-target count access.
    feature_sets={"B4": [f"early_anchor_value_{i}" for i in range(ANCHORS)], "B5":molecular_names,"B6":full_names}
    for key, fit in fits.items():
        save_model(ROOT / f"13_models/fold_{fold}_{key}.npz", fit, feature_sets[key])
    guard_rows=[]
    for key, fit in fits.items():
        for j,gene in enumerate(genes):
            guard_rows.append({"fold":fold,"model":key,"gene":gene,"is_anchor":j in set(anchor_indices),
                               "guard_pass":bool(fit["guard"][j]),"candidate_index":int(fit["selection"][j]),
                               "source_cv_mse":float(fit["meta"]["selected_mse"][j]),
                               "source_cv_residual_pearson":float(fit["meta"]["selected_corr"][j])})
    write_tsv_new(ROOT / f"02_source_validation/fold_{fold}_temporal_guards.tsv", guard_rows)
    pred_dir = fold_dir / "predictions_pre_unblinding"; pred_dir.mkdir(exist_ok=True)
    prediction_paths={}
    np.savez_compressed(pred_dir / "decomposition.npz", registered_reference=target_L_idw,
                        early_map_lineage=target_L, predicted_residual=pred_residuals["B6"],
                        final_prediction=predictions["B6_trajectory_aware_map_t"],
                        predictive_uncertainty_gene=uncertainty_gene)
    prediction_paths["decomposition"] = pred_dir / "decomposition.npz"
    for method,pred in predictions.items():
        path=pred_dir/f"{method}.npy"; np.save(path,pred); prediction_paths[method]=path
    # Shuffled dynamic trajectory controls saved compactly as float16 arrays.
    perm_dir = ROOT / f"09_permutations/fold_{fold}"; perm_dir.mkdir(exist_ok=True)
    rng=np.random.default_rng(9042+ord(fold)); dynamic_start=len(molecular_names)
    disp_q=np.digitize(target_track["displacement"],np.quantile(target_track["displacement"],[.25,.5,.75]))
    strata=(target_track["division"].astype(int)*4+disp_q).astype(int)
    perm_paths=[]
    for permutation in range(N_PERM):
        shuffled=target_full.copy()
        for stratum in np.unique(strata):
            idx=np.flatnonzero(strata==stratum); shuffled[idx,dynamic_start:]=target_full[rng.permutation(idx),dynamic_start:]
        correction=apply_temporal(fits["B6"],shuffled)
        path=perm_dir/f"perm_{permutation:03d}.npy"
        np.save(path,(target_L+correction).astype(np.float32)); perm_paths.append(path)
    for i,path in enumerate(perm_paths): prediction_paths[f"B7_perm_{i:03d}"]=path
    config={"fold":fold,"source":source,"target":target,"rank":RANK,"anchors":genes[anchor_indices].tolist(),
            "anchor_indices":anchor_indices.tolist(),"early_map":early["config"],"temporal_feature_names":full_names,
            "trajectory_normalization":{"mean":traj_mean.tolist(),"scale":traj_scale.tolist()},
            "source_support_guard":{"feature_low":traj_low.tolist(),"feature_high":traj_high.tolist(),
                                    "temporal_output_quantiles":[.005,.995]},
            "temporal_guard_margin":.03,"near_optimal_mse_tolerance":1.05,"ridge_grid":list(RIDGE_GRID),
            "gain_grid":list(GAIN_GRID),"target_75_expression_accessed":False}
    json_dump_new(fold_dir/"FROZEN_FOLD_CONFIG.json",config)
    prediction_paths["frozen_config"] = fold_dir/"FROZEN_FOLD_CONFIG.json"
    prediction_hashes(fold_dir,prediction_paths)
    # Unblind only now.
    target_counts = LateTruthVault(late_target_path, fold_dir).open()
    target_truth=np.log1p(normalize_counts(target_counts,early["depth_target"])).astype(np.float32)
    config["target_75_expression_access"]="evaluation_only_after_sentinel"
    json_dump_new(fold_dir/"UNBLINDED_EVALUATION_METADATA.json",{"late_truth_shape":list(target_truth.shape),
                  "late_truth_sha256":sha256(late_target_path),"normalization_depth_from_source":early["depth_target"]})
    source_svg,_=source_svg_indices([source_truth],[canonicalize(source_track["endpoints"])[0]],count=150)
    thresholds=np.quantile(source_truth,.8,axis=0)
    rows,gene_rows,domain_rows=evaluation_rows(fold,genes,target_truth,predictions,target_L,
                 canonicalize(target_track["endpoints"])[0],anchor_indices,thresholds,cohorts,source_svg)
    write_tsv_new(fold_dir/"PRIMARY_AND_SECONDARY_ENDPOINTS.tsv",rows)
    write_tsv_new(fold_dir/"PER_GENE_METRICS.tsv",gene_rows)
    write_tsv_new(fold_dir/"DOMAIN_METRICS.tsv",domain_rows)
    error=np.abs(target_truth-predictions["B6_trajectory_aware_map_t"])
    coverage=np.mean(error<=1.96*uncertainty_gene[None,:],axis=0)
    uncertainty_rows=[{"fold":fold,"gene":genes[j],"is_anchor":j in set(anchor_indices),
                       "source_cv_sigma":float(uncertainty_gene[j]),"target_95pct_coverage":float(coverage[j]),
                       "target_mae":float(error[:,j].mean())} for j in range(len(genes))]
    write_tsv_new(fold_dir/"UNCERTAINTY_CALIBRATION.tsv",uncertainty_rows)
    # Permutation effects, after prediction freeze.
    eval_genes=np.asarray([i for i in range(len(genes)) if i not in set(anchor_indices)],np.int64)
    perm_rows=[]
    for cohort,mask in cohorts.items():
        rt=target_truth[mask][:,eval_genes]-target_L[mask][:,eval_genes]
        real=pooled_pearson(rt,predictions["B6_trajectory_aware_map_t"][mask][:,eval_genes]-target_L[mask][:,eval_genes])
        null=[]
        for path in perm_paths:
            pp=np.asarray(np.load(path,mmap_mode="r"),np.float32)
            null.append(pooled_pearson(rt,pp[mask][:,eval_genes]-target_L[mask][:,eval_genes]))
        perm_rows.append({"fold":fold,"cohort":cohort,"observed":real,"null_mean":float(np.mean(null)),
                          "null_ci_low":float(np.quantile(null,.025)),"null_ci_high":float(np.quantile(null,.975)),
                          "empirical_p":float((1+np.sum(np.asarray(null)>=real))/(1+len(null))),"permutations":len(null)})
    write_tsv_new(ROOT/f"09_permutations/fold_{fold}_results.tsv",perm_rows)
    boot=family_bootstrap(fold,predictions,target_truth,target_L,target_track["ancestor"],eval_genes,cohorts)
    write_tsv_new(ROOT/f"08_bootstrap/fold_{fold}_hierarchical_bootstrap.tsv",boot)
    # Overlap-excluded retraining sensitivity.
    shared_source=set(source_track["node_sets"]) & set(target_track["node_sets"])
    keep=~np.isin(source_track["ancestor"],list(shared_source))
    sensitivity={"eligible":int(len(np.unique(source_track["ancestor"][keep])))>=300,
                 "remaining_ancestors":int(len(np.unique(source_track["ancestor"][keep]))),
                 "remaining_descendants":int(keep.sum()),"removed_fraction":float(1-keep.mean())}
    if sensitivity["eligible"]:
        sf=crossfit_temporal(source_full[keep],D[keep],descendant_blocks[keep])
        sf["guard"][anchor_indices]=False
        sp=np.maximum(target_L+apply_temporal(sf,target_full),0)
        path=ROOT/f"07_overlap_excluded_retraining/fold_{fold}_prediction.npy"; np.save(path,sp)
        sensitivity["prediction_sha256"]=sha256(path)
        for cohort,mask in cohorts.items():
            rt=target_truth[mask][:,eval_genes]-target_L[mask][:,eval_genes]
            rp=sp[mask][:,eval_genes]-target_L[mask][:,eval_genes]
            sensitivity[f"rho_future_{cohort}"]=pooled_pearson(rt,rp)
        sensitivity["guard_stability_fraction"]=float(np.mean(sf["guard"]==fits["B6"]["guard"]))
    json_dump_new(ROOT/f"07_overlap_excluded_retraining/fold_{fold}_sensitivity.json",sensitivity)
    return {"fold":fold,"source":source,"target":target,"anchors":genes[anchor_indices].tolist(),
            "anchor_indices":anchor_indices,"rows":rows,"gene_rows":gene_rows,"domain_rows":domain_rows,
            "bootstrap":boot,"permutation":perm_rows,"sensitivity":sensitivity,
            "truth":target_truth,"predictions":predictions,"lineage":target_L,"xyz":canonicalize(target_track["endpoints"])[0],
            "target_track":target_track,"source_svg":source_svg}


def make_figures(results, genes):
    out=ROOT/"10_figures_main"; sup=ROOT/"11_figures_supplement"
    method_labels={m:m.split("_",1)[1].replace("_"," ") for m in METHODS}
    fig=plt.figure(figsize=(13.2,10.5),layout="constrained")
    gs=fig.add_gridspec(3,3,height_ratios=[.85,1,1])
    ax=fig.add_subplot(gs[0,:]); ax.axis("off")
    ax.text(.01,.96,"a",fontweight="bold",fontsize=13,va="top")
    for i,(textc,color) in enumerate((("50% source embryo\n495-gene field","#4C78A8"),("source-trained\n16-anchor MAP","#59A14F"),
                                      ("TM0–TM230\ntracking atlas","#B279A2"),("guarded rank-16\ntemporal residual","#D62728"),
                                      ("withheld 75%\nmolecular field","#222222"))):
        x=.05+i*.19
        ax.add_patch(plt.Rectangle((x,.25),.15,.43,facecolor="white",edgecolor=color,lw=2,transform=ax.transAxes))
        ax.text(x+.075,.465,textc,ha="center",va="center",fontsize=9,transform=ax.transAxes)
        if i<4: ax.annotate("",xy=(x+.19,.465),xytext=(x+.15,.465),xycoords=ax.transAxes,
                            arrowprops=dict(arrowstyle="->",lw=1.5,color="#555555"))
    ax.text(.05,.10,"Fold A: E1 transition → E2 forecast     Fold B: E2 transition → E1 forecast; no molecular pooling",
            fontsize=9,transform=ax.transAxes)
    metrics=pd.concat([pd.DataFrame(r["rows"]) for r in results],ignore_index=True)
    for col,(metric,label) in enumerate((("rho_future_pooled","Future-residual Pearson"),("rmse","log1p RMSE"))):
        ax=fig.add_subplot(gs[1,col]); ax.text(-.16,1.06,chr(ord('b')+col),transform=ax.transAxes,fontweight="bold",fontsize=13)
        for fi,fold in enumerate(("A","B")):
            d=metrics[(metrics.fold==fold)&(metrics.cohort=="full")]
            xs=np.arange(len(METHODS))+.12*(fi-.5)
            ax.scatter(xs,[d[d.method==m][metric].iloc[0] for m in METHODS],s=35,label=f"Fold {fold}",zorder=3)
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([f"B{i+1}" for i in range(len(METHODS))])
        ax.set_ylabel(label); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,fontsize=8)
    ax=fig.add_subplot(gs[1,2]); ax.text(-.16,1.06,"d",transform=ax.transAxes,fontweight="bold",fontsize=13)
    for r in results:
        p=pd.DataFrame(r["permutation"])
        x=0 if r["fold"]=="A" else 1
        q=p[p.cohort=="full"].iloc[0]
        ax.errorbar(x,q.null_mean,yerr=[[q.null_mean-q.null_ci_low],[q.null_ci_high-q.null_mean]],fmt='o',color="#888888",label="shuffled" if x==0 else None)
        ax.scatter(x,q.observed,color="#D62728",s=50,label="real" if x==0 else None,zorder=4)
    ax.set_xticks([0,1],["Fold A","Fold B"]); ax.set_ylabel("Future-residual Pearson"); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[2,0]); ax.text(-.16,1.06,"e",transform=ax.transAxes,fontweight="bold",fontsize=13)
    for fi,fold in enumerate(("A","B")):
        d=metrics[(metrics.fold==fold)&(metrics.method=="B6_trajectory_aware_map_t")]
        ax.plot([0,1,2],[d[d.cohort==c].rho_future_pooled.iloc[0] for c in ("full","shared","disjoint")],'-o',label=f"Fold {fold}")
    ax.axhline(0,color="#999999",ls=':',lw=1); ax.set_xticks([0,1,2],["Full","Shared","Disjoint"]); ax.set_ylabel("Future-residual Pearson"); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[2,1]); ax.text(-.16,1.06,"f",transform=ax.transAxes,fontweight="bold",fontsize=13)
    for fi,r in enumerate(results):
        g=pd.DataFrame(r["gene_rows"]); a=g[g.method=="B6_trajectory_aware_map_t"].future_residual_pearson.dropna(); b=g[g.method=="B5_trajectory_free_map_t"].future_residual_pearson.dropna()
        ax.scatter(b,a,s=4,alpha=.25,label=f"Fold {r['fold']}")
    lim=[-.5,1]; ax.plot(lim,lim,ls=':',color='#777777'); ax.set(xlim=lim,ylim=lim,xlabel="Trajectory-free",ylabel="Trajectory-aware"); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[2,2]); ax.text(-.16,1.06,"g",transform=ax.transAxes,fontweight="bold",fontsize=13)
    for fi,r in enumerate(results):
        full=[row for row in r["rows"] if row["cohort"]=="full" and row["method"]=="B6_trajectory_aware_map_t"][0]
        ax.scatter(full["rho_future_pooled"],-full["rmse"],s=80,label=f"Fold {r['fold']}")
    ax.set_xlabel("Future-residual Pearson →"); ax.set_ylabel("− log1p RMSE →"); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"Figure_MAP_T_crossfit.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)
    # Supplement: trajectory, maps, guards and failure distributions.
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for col,r in enumerate(results):
        ax=axes[0,col]; tr=r["target_track"]
        sample=np.linspace(0,len(tr["starts"])-1,min(500,len(tr["starts"]))).astype(int)
        for i in sample: ax.plot([tr["starts"][i,0],tr["endpoints"][i,0]],[tr["starts"][i,1],tr["endpoints"][i,1]],color="#4C78A8",alpha=.12,lw=.5)
        ax.set_title(f"Fold {r['fold']} tracking trajectories"); ax.set_aspect('equal'); ax.axis('off')
        ax=axes[1,col]; g=pd.DataFrame(r["gene_rows"]); vals=[]
        for m in ("B1_lineage_registered_idw","B4_anchor_only_temporal","B5_trajectory_free_map_t","B6_trajectory_aware_map_t"):
            vals.append(g[g.method==m].future_residual_pearson.dropna().values)
        parts=ax.violinplot(vals,showmedians=True); ax.set_xticks(range(1,5),["IDW","Anchor","No traj.","MAP-T"],rotation=20); ax.set_ylabel("Per-gene residual Pearson"); ax.spines[['top','right']].set_visible(False)
    ax=axes[0,2]; ax.axis('off'); ax.text(.02,.95,"Source-only temporal guards",fontweight='bold',va='top')
    for i,r in enumerate(results):
        guard=pd.read_csv(ROOT/f"02_source_validation/fold_{r['fold']}_temporal_guards.tsv",sep='\t')
        sub=guard[guard.model=='B6']; ax.text(.05,.75-i*.22,f"Fold {r['fold']}: {sub.guard_pass.sum()} / {len(sub)-ANCHORS} non-anchor genes corrected",fontsize=10)
    axes[1,2].axis('off'); axes[1,2].text(.02,.95,"Applicability boundary",fontweight='bold',va='top'); axes[1,2].text(.02,.78,"Failures are retained in the per-gene source tables.\nFolds are independent molecular embryos but share\npart of one published tracking reference.",va='top')
    for ext in ("pdf","svg","png"):
        fig.savefig(sup/f"Supplementary_MAP_T_diagnostics.{ext}",dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)


def finalize(results, genes):
    all_rows=[row for r in results for row in r["rows"]]
    all_gene=[row for r in results for row in r["gene_rows"]]
    all_domain=[row for r in results for row in r["domain_rows"]]
    all_boot=[row for r in results for row in r["bootstrap"]]
    write_tsv_new(ROOT/"12_source_data/PRIMARY_ENDPOINTS_ALL_FOLDS.tsv",all_rows)
    write_tsv_new(ROOT/"12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv",all_gene)
    write_tsv_new(ROOT/"12_source_data/DOMAIN_METRICS_ALL_FOLDS.tsv",all_domain)
    write_tsv_new(ROOT/"08_bootstrap/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv",all_boot)
    table=pd.DataFrame(all_rows)
    gates={}
    g1=g2=g3=g4=g5=True
    gate_rows=[]
    for r in results:
        fold=r["fold"]
        full=table[(table.fold==fold)&(table.cohort=="full")].set_index("method")
        b6=full.loc["B6_trajectory_aware_map_t"]
        ci=[x for x in r["bootstrap"] if x["cohort"]=="full" and x["method"]=="B6_trajectory_aware_map_t"][0]
        fg1=b6.rho_future_pooled>0 and ci["ci95_low"]>0
        fg2=b6.rho_future_pooled>full.loc["B1_lineage_registered_idw"].rho_future_pooled
        dom=pd.DataFrame(r["domain_rows"]); med=dom.groupby('method').median(numeric_only=True)
        fg3=(med.loc['B6_trajectory_aware_map_t','dice']>med.loc['B1_lineage_registered_idw','dice'] or
             med.loc['B6_trajectory_aware_map_t','boundary_f1']>med.loc['B1_lineage_registered_idw','boundary_f1'] or
             med.loc['B6_trajectory_aware_map_t','centroid_error']<med.loc['B1_lineage_registered_idw','centroid_error'])
        perm=[x for x in r['permutation'] if x['cohort']=='full'][0]
        fg4=perm['observed']>perm['null_ci_high']
        fg5=b6.rho_future_pooled>full.loc['B5_trajectory_free_map_t'].rho_future_pooled
        g1&=fg1;g2&=fg2;g3&=fg3;g4&=fg4;g5&=fg5
        gate_rows.append({"fold":fold,"G1_positive_CI":fg1,"G2_beats_IDW":fg2,"G3_localization":fg3,"G4_real_beats_shuffled":fg4,"G5_beats_trajectory_free":fg5})
    g6=True # all reported primary metrics excluded frozen anchor indices by construction
    g7=True # test suite is run and recorded before finalization
    if not (g1 and g2): classification="FAIL"
    elif all((g1,g2,g3,g4,g5,g6,g7)): classification="PASS"
    else: classification="PARTIAL"
    # Tracking grade.
    directions=[]; comparisons=[]; shuffles=[]; sensitivities=[]
    for r in results:
        full=table[(table.fold==r['fold'])].set_index(['cohort','method'])
        directions.append(full.loc[('disjoint','B6_trajectory_aware_map_t')].rho_future_pooled>0)
        comparisons.append(full.loc[('disjoint','B6_trajectory_aware_map_t')].rho_future_pooled>full.loc[('disjoint','B5_trajectory_free_map_t')].rho_future_pooled)
        p=[x for x in r['permutation'] if x['cohort']=='disjoint'][0]; shuffles.append(p['observed']>p['null_ci_high'])
        sensitivities.append(r['sensitivity'].get('rho_future_full',-1)>0)
    if all(directions+comparisons+shuffles+sensitivities): grade='A'
    elif all(directions) and all(sensitivities): grade='B'
    elif classification in ('PASS','PARTIAL'): grade='C'
    else: grade='D'
    write_tsv_new(ROOT/"12_source_data/CORE_GATE_TABLE.tsv",gate_rows)
    claim=("Early molecular measurements conditioned on a developmental tracking atlas forecast later three-dimensional molecular organization across individual embryos."
           if classification=='PASS' else "No headline forecasting claim is authorized by the frozen gates.")
    (ROOT/"FINAL_DECISION.md").open('x').write(f"# Final decision\n\n**Core classification: {classification}.**\n\n**Tracking-generalization grade: {grade}.**\n\n{claim}\n")
    (ROOT/"TRACKING_PROVENANCE.md").open('x').write("# Tracking provenance\n\nE1 and E2 are independent molecular embryos mapped to a partly shared published live-tracking reference (about 60% early-node and 52% late-node overlap). The experiment is tracking-map-conditioned cross-embryo molecular forecasting, not two independently live-imaged trajectory systems.\n")
    (ROOT/"TRACKING_GENERALIZATION.md").open('x').write(f"# Tracking generalization\n\nGrade **{grade}** under the frozen full/shared/disjoint and overlap-excluded-retraining criteria. See `12_source_data/PRIMARY_ENDPOINTS_ALL_FOLDS.tsv`, `09_permutations/`, and `07_overlap_excluded_retraining/`.\n")
    (ROOT/"SUPPORTED_CLAIMS.md").open('x').write(f"# Supported claims\n\n{claim if classification=='PASS' else 'The experiment provides cross-fitted descriptive evidence only; the frozen headline gates did not all pass.'}\n")
    (ROOT/"UNSUPPORTED_CLAIMS.md").open('x').write("# Unsupported claims\n\nIndependent longitudinal imaging systems; forecasting before trajectories are observed; continuous intermediate molecular truth; future morphology; prospective validation; genome-wide forecasting; or generalization beyond zebrafish 50%–75% epiboly.\n")
    (ROOT/"LIMITATIONS.md").open('x').write("# Limitations\n\nOnly two reciprocal molecular embryos were evaluated, and they partly reuse one tracking reference. Target trajectories are observed inputs. The assay measures 495 genes. Cross-fitted folds are not independent live-tracking systems. A prior eligibility schema audit materialized late-target obs tables, but no expression values entered fitting or decisions; this execution enforced selective access and prediction sentinels.\n")
    (ROOT/"RUN_SUMMARY.md").open('x').write(f"# Run summary\n\nCompleted two cross-fitted folds, six locked baselines, {N_PERM} trajectory permutations per fold, full/shared/disjoint cohorts, overlap-excluded retraining and {N_BOOT} ancestor-family bootstraps. Core={classification}; tracking grade={grade}.\n")
    (ROOT/"REPRODUCE.sh").open('x').write("#!/usr/bin/env bash\nset -euo pipefail\nMPLCONFIGDIR=/tmp/mplconfig_map_t python3 run_hyperspatial_map_t_4d_frozen_20260804.py --root results_v10/hyperspatial_map_t_4d_frozen_20260804_112617\n")
    os.chmod(ROOT/"REPRODUCE.sh",0o755)
    captions="# Figure captions\n\n**Figure MAP-T.** Cross-fitted tracking-map-conditioned molecular forecasting. E1 and E2 are independent molecular embryos but partly share one published tracking reference. Points and intervals use TM0 ancestor families; target 75% expression was opened only after predictions were frozen and hashed.\n\n**Supplementary diagnostics.** Tracking paths, per-gene residual recovery and source-only temporal guards, retaining all successes and failures.\n"
    (ROOT/"10_figures_main/FIGURE_CAPTIONS.md").open('x').write(captions)
    environment={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__,
                 "cuda_available":torch.cuda.is_available(),"git_commit":None,"peak_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    json_dump_new(ROOT/"14_logs/SOFTWARE_ENVIRONMENT.json",environment)
    return classification,grade,claim


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT); args=parser.parse_args()
    if args.root != ROOT: raise RuntimeError("This frozen runner is authorized only for the active result root")
    ensure_layout(); started=time.time()
    test_files=[str(p) for p in sorted(Path('tests').glob('test_4d_*.py'))]
    test_run=subprocess.run(['pytest','-q',*test_files],env={**os.environ,'PYTHONPATH':'.'},capture_output=True,text=True)
    (ROOT/'14_logs/PRE_EXECUTION_TESTS.txt').open('x').write(test_run.stdout+test_run.stderr)
    if test_run.returncode:
        raise RuntimeError('Frozen leakage/provenance tests failed; see PRE_EXECUTION_TESTS.txt')
    genes=audit_inputs()
    early_ids={e:read_ids(INPUTS/f"matched_A_50p_{e}_sphere_logFC.h5ad") for e in ('E1','E2')}
    late_ids={e:read_ids(INPUTS/f"matched_B_75p_{e}_sphere_logFC.h5ad") for e in ('E1','E2')}
    relevant=set(map(int,np.r_[early_ids['E1']['Amat_id'],early_ids['E2']['Amat_id']]))
    frame=read_tracking(relevant)
    tracking={e:build_tracking_data(frame,e,early_ids[e],late_ids[e]) for e in ('E1','E2')}
    overlap={}
    for source,target in (('E1','E2'),('E2','E1')):
        overlap[(source,target)]=tracking_overlap(tracking[source]['node_sets'],tracking[target]['node_sets'])
    provenance={"frozen_interpretation":"tracking-map-conditioned cross-embryo molecular forecasting",
                "independent_live_tracking_systems":False,"early_node_overlap":2953,"early_jaccard":.5963,
                "late_node_overlap":4737,"late_jaccard":.5207,
                "cohort_definition":"complete TM0-TM230 family node-set intersection, frozen before target truth"}
    json_dump_new(ROOT/"01_protocol_lock/TRACKING_PROVENANCE_LOCK.json",provenance)
    # Materialize cohort membership before any predictive fitting or late truth access.
    for fold,spec in FOLDS.items():
        shared,disjoint=overlap[(spec['source'],spec['target'])]
        target_anc=tracking[spec['target']]['ancestor']
        np.savez_compressed(ROOT/f"06_tracking_overlap/fold_{fold}_PREMODEL_membership.npz",
                            ancestor=target_anc,shared=np.isin(target_anc,list(shared)),
                            disjoint=np.isin(target_anc,list(disjoint)))
        json_dump_new(ROOT/f"06_tracking_overlap/fold_{fold}_PREMODEL_definition.json",
                      {"fold":fold,"defined_before_model_fitting":True,"defined_before_unblinding":True,
                       "shared_ancestors":len(shared),"disjoint_ancestors":len(disjoint),
                       "rule":"any node intersection across complete TM0-TM230 family node sets"})
    # Freeze source-only panels before either target late truth is accessed.
    anchors={}; panel_rows=[]
    for embryo in ('E1','E2'):
        counts=read_counts(INPUTS/f"matched_A_50p_{embryo}_sphere_logFC.h5ad")
        depth=float(np.median(np.maximum(counts.sum(1),1))); log=np.log1p(normalize_counts(counts,depth))
        idx,meta=select_non_axial_anchors(log,genes,ANCHORS); anchors[embryo]=idx
        for rank,index in enumerate(idx): panel_rows.append({"source_embryo":embryo,"rank":rank+1,"gene":genes[index],"gene_index":int(index),"source_only":True})
    write_tsv_new(ROOT/"01_protocol_lock/FROZEN_SOURCE_ONLY_ANCHOR_PANELS.tsv",panel_rows)
    protocol={"experiment":"HyperSpatial-MAP-T","status":"frozen_before_target_late_expression","folds":FOLDS,
              "architecture":{"registered_idw_k":12,"anchors":16,"multiscale_k":[12,32],"rank":16,
                              "ridge_grid":list(RIDGE_GRID),"gain_grid":list(GAIN_GRID),"guard_margin":.03,
                              "near_optimal_mse":1.05},"bootstraps":N_BOOT,"permutations":N_PERM,
              "pre_unblinding_numerical_amendment":"source-min/max trajectory feature clipping, source 0.5%-99.5% developmental-residual bounds, float32 permutation serialization",
              "primary_endpoint":"corr(Y_target75-L_target75, Yhat_target75-L_target75), non-anchor genes",
              "interpretation":provenance["frozen_interpretation"],"gates":"G1-G7 and tracking grades A-D exactly as user-specified"}
    json_dump_new(ROOT/"01_protocol_lock/PROTOCOL_LOCK.json",protocol)
    code_files=[Path(__file__),Path('hyperspatial_map_t_core_20260804.py'),Path('run_hyperspatial_map_stage1b_20260803.py'),Path('run_hyperspatial_map_stage1_20260803.py')]
    write_tsv_new(ROOT/"01_protocol_lock/CODE_AND_CONFIG_HASHES.tsv",
                  [{"path":str(p.resolve()),"sha256":sha256(p)} for p in code_files]+[{"path":str((ROOT/'01_protocol_lock/PROTOCOL_LOCK.json').resolve()),"sha256":sha256(ROOT/'01_protocol_lock/PROTOCOL_LOCK.json')}])
    results=[]
    for fold,spec in FOLDS.items(): results.append(run_fold(fold,spec,genes,tracking,anchors,overlap))
    make_figures(results,genes)
    classification,grade,claim=finalize(results,genes)
    (ROOT/"14_logs/RUNTIME.txt").open('x').write(f"wall_seconds\t{time.time()-started:.3f}\npeak_rss_kb\t{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\n")
    print(json.dumps({"core_classification":classification,"tracking_grade":grade,"supported_claim":claim,"output":str(ROOT.resolve())},indent=2))


if __name__=='__main__': main()
