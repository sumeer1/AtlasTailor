#!/usr/bin/env python3
"""Fit the frozen 2-D MAP source model and lock geometry for one SPATCH pair.

This process can see Visium-HD expression and Xenium coordinates, but it has no
code path to the sealed Xenium expression matrix.  All model selection and the
16-gene panel are therefore source-only.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_spatch")

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "analysis" / "hyperspatial_map_2d_dlpfc"))
sys.path.insert(0, str(PROJECT))

from dlpfc_common import (  # noqa: E402
    ANCHOR_COUNT, FUSION_GRID, GAIN_GRID, K_LOCAL, RANK, REGISTRATION_EPOCHS,
    RIDGE_GRID, SEEDS, apply_centered, canonicalize, fit_centered_ridge_rank,
    idw, normalized_source, pearson_columns, quadrants, residual_features,
    save_json, select_anchors, sha256, source_svg_indices,
)
from run_wemerfish_temporal_holdout import registered_coordinates  # noqa: E402


DATA = ROOT / "official_data" / "extracted"
SEALED = ROOT / "SEALED_TARGET_8UM"
WORK = ROOT / "work"


def pack_model(output, prefix, model):
    for key, value in model.items():
        output[f"{prefix}_{key}"] = value


def load_source(tumor: str, common_genes: np.ndarray):
    path = DATA / f"VisiumHD_FFPE_{tumor}" / "adata.h5ad"
    data = ad.read_h5ad(path, backed="r")
    source_names = data.var_names.astype(str).to_numpy()
    by_name = {name: i for i, name in enumerate(source_names)}
    columns = np.asarray([by_name[name] for name in common_genes], np.int64)
    rows = np.flatnonzero(np.asarray(data.obs["high_quality"]) == 1)
    coordinates_raw = np.asarray(data.obsm["spatial"])[rows].astype(np.float32)
    # Only the official source tissue/QC flag is used. Molecular/pathology labels
    # present in obs are intentionally never loaded here.
    counts = data[rows, columns].X
    if hasattr(counts, "to_memory"):
        counts = counts.to_memory()
    counts = counts.tocsr()
    data.file.close()
    return path, counts, coordinates_raw, rows


def load_target_geometry(tumor: str):
    path = SEALED / f"Xenium_{tumor}_8um_shared_counts.h5ad"
    target = ad.read_h5ad(path, backed="r")
    coordinates_raw = np.asarray(target.obsm["spatial"], np.float32).copy()
    ids = target.obs_names.astype(str).to_numpy()
    genes = target.var_names.astype(str).to_numpy()
    target.file.close()
    return path, coordinates_raw, ids, genes


def chunked_gate(truth, base, global_base, local_base, anchor_log, gain, chunk=128):
    """Exact source gate from per-gene moments, without materializing 11 cubes."""
    n_genes = truth.shape[1]
    selection = np.empty(n_genes, np.int16)
    predictable = np.zeros(n_genes, bool)
    selected_mse = np.empty(n_genes, np.float32)
    selected_r = np.empty(n_genes, np.float32)
    base_mse = np.empty(n_genes, np.float32)
    base_r = np.empty(n_genes, np.float32)
    selected_mse_sum = 0.0
    for begin in range(0, n_genes, chunk):
        end = min(begin + chunk, n_genes)
        t = truth[:, begin:end]
        b = base[:, begin:end]
        a = anchor_log[:, begin:end]
        tm = np.mean(t, axis=0, dtype=np.float64)
        t2 = np.mean(np.square(t), axis=0, dtype=np.float64)
        am = np.mean(a, axis=0, dtype=np.float64)
        a2 = np.mean(np.square(a), axis=0, dtype=np.float64)
        at = np.mean(a * t, axis=0, dtype=np.float64)
        errors, correlations = [], []
        for residual in (b + gain * global_base[:, begin:end],
                         b + gain * local_base[:, begin:end]):
            pm = np.mean(residual, axis=0, dtype=np.float64)
            p2 = np.mean(np.square(residual), axis=0, dtype=np.float64)
            pt = np.mean(residual * t, axis=0, dtype=np.float64)
            pa = np.mean(residual * a, axis=0, dtype=np.float64)
            for weight in FUSION_GRID:
                one = 1.0 - float(weight)
                ym = one * pm + weight * am
                y2 = one * one * p2 + weight * weight * a2 + 2.0 * one * weight * pa
                yt = one * pt + weight * at
                errors.append(np.maximum(y2 + t2 - 2.0 * yt, 0.0))
                denom = np.sqrt(np.maximum((y2 - ym * ym) * (t2 - tm * tm), 0.0))
                correlations.append(np.divide(yt - ym * tm, denom,
                                               out=np.full_like(denom, np.nan), where=denom > 1e-12))
        bm = np.mean(np.square(b - t), axis=0, dtype=np.float64)
        br = pearson_columns(t, b)
        errors.append(bm)
        correlations.append(br)
        errors = np.stack(errors)
        correlations = np.stack(correlations)
        eligible = errors <= 1.05 * np.maximum(errors.min(axis=0)[None], 1e-10)
        chosen = np.argmax(np.where(eligible, correlations, -np.inf), axis=0)
        local = np.arange(end - begin)
        sm = errors[chosen, local]
        sr = correlations[chosen, local]
        selection[begin:end] = chosen
        selected_mse[begin:end] = sm
        selected_r[begin:end] = sr
        base_mse[begin:end] = bm
        base_r[begin:end] = br
        predictable[begin:end] = (sm < bm) & (sr > br + 0.03)
        selected_mse_sum += float(sm.sum())
    return selection, predictable, {
        "selected_validation_mse": selected_mse,
        "selected_validation_pearson": selected_r,
        "base_validation_mse": base_mse,
        "base_validation_pearson": base_r,
        "median_spatial_pearson": float(np.nanmedian(selected_r)),
        "log1p_rmse": float(np.sqrt(selected_mse_sum / n_genes)),
    }


def chunked_gate_grid(truth, base, global_base, local_base, anchor_log, gains, chunk=128):
    """Evaluate all fixed gains in one pass over each validation gene block."""
    n_genes = truth.shape[1]
    states = {
        float(gain): {
            "selection": np.empty(n_genes, np.int16),
            "predictable": np.zeros(n_genes, bool),
            "selected_mse": np.empty(n_genes, np.float32),
            "selected_r": np.empty(n_genes, np.float32),
            "base_mse": np.empty(n_genes, np.float32),
            "base_r": np.empty(n_genes, np.float32),
        } for gain in gains
    }
    for begin in range(0, n_genes, chunk):
        end = min(begin + chunk, n_genes)
        t = truth[:, begin:end]; b = base[:, begin:end]; a = anchor_log[:, begin:end]
        g = global_base[:, begin:end]; l = local_base[:, begin:end]
        tm = np.mean(t, 0, dtype=np.float64); t2 = np.mean(np.square(t), 0, dtype=np.float64)
        bm = np.mean(b, 0, dtype=np.float64); b2 = np.mean(np.square(b), 0, dtype=np.float64)
        bt = np.mean(b*t, 0, dtype=np.float64); ba = np.mean(b*a, 0, dtype=np.float64)
        am = np.mean(a, 0, dtype=np.float64); a2 = np.mean(np.square(a), 0, dtype=np.float64)
        at = np.mean(a*t, 0, dtype=np.float64)
        base_error = np.maximum(b2 + t2 - 2.0*bt, 0.0)
        base_denom = np.sqrt(np.maximum((b2-bm*bm)*(t2-tm*tm), 0.0))
        base_corr = np.divide(bt-bm*tm, base_denom, out=np.full_like(base_denom, np.nan), where=base_denom>1e-12)
        residual_moments = []
        for residual in (g, l):
            residual_moments.append({
                "m": np.mean(residual, 0, dtype=np.float64),
                "s2": np.mean(np.square(residual), 0, dtype=np.float64),
                "bt": np.mean(residual*t, 0, dtype=np.float64),
                "bb": np.mean(residual*b, 0, dtype=np.float64),
                "ba": np.mean(residual*a, 0, dtype=np.float64),
            })
        local_index = np.arange(end-begin)
        for gain in gains:
            gain = float(gain)
            errors, correlations = [], []
            for moments in residual_moments:
                pm = bm + gain*moments["m"]
                p2 = b2 + 2.0*gain*moments["bb"] + gain*gain*moments["s2"]
                pt = bt + gain*moments["bt"]
                pa = ba + gain*moments["ba"]
                for weight in FUSION_GRID:
                    weight = float(weight); one = 1.0-weight
                    ym = one*pm + weight*am
                    y2 = one*one*p2 + weight*weight*a2 + 2.0*one*weight*pa
                    yt = one*pt + weight*at
                    errors.append(np.maximum(y2+t2-2.0*yt, 0.0))
                    denom = np.sqrt(np.maximum((y2-ym*ym)*(t2-tm*tm), 0.0))
                    correlations.append(np.divide(yt-ym*tm, denom,
                                                  out=np.full_like(denom, np.nan), where=denom>1e-12))
            errors.append(base_error); correlations.append(base_corr)
            errors = np.stack(errors); correlations = np.stack(correlations)
            eligible = errors <= 1.05*np.maximum(errors.min(0)[None], 1e-10)
            chosen = np.argmax(np.where(eligible, correlations, -np.inf), axis=0)
            selected_error = errors[chosen, local_index]
            selected_corr = correlations[chosen, local_index]
            state = states[gain]
            state["selection"][begin:end] = chosen
            state["selected_mse"][begin:end] = selected_error
            state["selected_r"][begin:end] = selected_corr
            state["base_mse"][begin:end] = base_error
            state["base_r"][begin:end] = base_corr
            state["predictable"][begin:end] = ((selected_error < base_error) &
                                                 (selected_corr > base_corr + 0.03))
    results = {}
    for gain, state in states.items():
        results[gain] = (state["selection"], state["predictable"], {
            "selected_validation_mse": state["selected_mse"],
            "selected_validation_pearson": state["selected_r"],
            "base_validation_mse": state["base_mse"],
            "base_validation_pearson": state["base_r"],
            "median_spatial_pearson": float(np.nanmedian(state["selected_r"])),
            "log1p_rmse": float(np.sqrt(np.mean(state["selected_mse"], dtype=np.float64))),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor", choices=("COAD", "HCC", "OV"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    tumor = args.tumor
    output = WORK / tumor
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()

    target_path, target_raw, target_ids, common_genes = load_target_geometry(tumor)
    source_path, counts, source_raw, source_rows = load_source(tumor, common_genes)
    source_xy, source_transform = canonicalize(source_raw)
    target_xy, target_transform = canonicalize(target_raw)
    source_values, source_log, source_depth, depth_target = normalized_source(counts)
    print(f"{tumor}: loaded source {source_log.shape} and target geometry {target_xy.shape}", flush=True)

    anchors, eligible, anchor_metadata = select_anchors(source_log, common_genes)
    if len(anchors) != ANCHOR_COUNT:
        raise AssertionError("Frozen anchor count changed")
    anchor_set = set(anchors.tolist())
    evaluation = np.asarray([i for i in eligible if i not in anchor_set], np.int64)
    print(f"{tumor}: source-only anchors selected; {len(evaluation)} hidden genes", flush=True)

    labels = quadrants(source_xy)
    if set(np.unique(labels)) != {0, 1, 2, 3}:
        raise RuntimeError("All four source-only median quadrants are required")
    pseudo_base = np.empty_like(source_values)
    for label in range(4):
        query = np.flatnonzero(labels == label)
        reference = np.flatnonzero(labels != label)
        pseudo_base[query] = idw(source_xy[reference], source_values[reference], source_xy[query], 12)
    pseudo_log = np.log1p(pseudo_base).astype(np.float32)
    source_residual = source_log - pseudo_log
    print(f"{tumor}: four-quadrant source pseudo-target prior complete", flush=True)
    raw_features = source_residual[:, anchors]
    local_features = residual_features(raw_features, source_xy, True)
    validation = np.isin(labels, [0, 3])
    train = ~validation

    grid, best = [], None
    for ridge in RIDGE_GRID:
        global_part = fit_centered_ridge_rank(raw_features[train], source_residual[train], ridge, RANK)
        local_part = fit_centered_ridge_rank(local_features[train], source_residual[train], ridge, RANK)
        anchor_part = fit_centered_ridge_rank(source_log[train][:, anchors], source_log[train], ridge, RANK)
        global_validation = apply_centered(global_part, raw_features[validation])
        local_validation = apply_centered(local_part, local_features[validation])
        anchor_validation = apply_centered(anchor_part, source_log[validation][:, anchors])
        gain_results = chunked_gate_grid(
            source_log[validation], pseudo_log[validation], global_validation,
            local_validation, anchor_validation, GAIN_GRID,
        )
        for gain in GAIN_GRID:
            selection, predictable, gate = gain_results[float(gain)]
            record = {
                "ridge": float(ridge), "gain": float(gain),
                "median_spatial_pearson": gate["median_spatial_pearson"],
                "log1p_rmse": gate["log1p_rmse"],
                "predictable_genes": int(predictable.sum()),
            }
            grid.append(record)
            print(f"{tumor}: source gate ridge={ridge} gain={gain} r={record['median_spatial_pearson']:.4f}", flush=True)
            key = (record["median_spatial_pearson"], -record["log1p_rmse"], record["predictable_genes"])
            if best is None or key > best[0]:
                best = (key, record, selection.copy(), predictable.copy(), {
                    k: v.copy() if isinstance(v, np.ndarray) else v for k, v in gate.items()
                })
        del global_validation, local_validation, anchor_validation

    chosen, fusion_selection, predictable, gate = best[1:]
    ridge = chosen["ridge"]
    global_model = fit_centered_ridge_rank(raw_features, source_residual, ridge, RANK)
    local_model = fit_centered_ridge_rank(local_features, source_residual, ridge, RANK)
    anchor_model = fit_centered_ridge_rank(source_log[:, anchors], source_log, ridge, RANK)
    predictable[anchors] = False
    anchor_sum = np.asarray(counts[:, anchors].sum(axis=1)).ravel()
    total_sum = np.asarray(counts.sum(axis=1)).ravel()
    depth_x = np.c_[np.ones(len(anchor_sum)), np.log1p(anchor_sum)]
    depth_weight = np.linalg.solve(
        depth_x.T @ depth_x + np.diag([0.0, 1.0]), depth_x.T @ np.log1p(total_sum)
    ).astype(np.float32)
    svg_pool, svg_score = source_svg_indices(source_log, source_xy, count=220)
    evaluation_set = set(evaluation.tolist())
    representative_pool = np.asarray(
        [index for index in svg_pool[np.argsort(svg_score[svg_pool])[::-1]]
         if int(index) in evaluation_set][:20], np.int64
    )

    arrays = {
        "anchor_indices": anchors, "eligible_indices": eligible,
        "evaluation_indices": evaluation, "predictable": predictable,
        "fusion_selection": fusion_selection, "depth_weight": depth_weight,
        "source_depth_target": np.asarray(depth_target, np.float32),
        "source_coordinates": source_xy, "source_rows": source_rows,
        "gene_names": common_genes,
        "source_spatial_scores": svg_score,
        "source_only_representative_gene_pool": representative_pool,
        "gate_selected_validation_mse": gate["selected_validation_mse"],
        "gate_selected_validation_pearson": gate["selected_validation_pearson"],
        "gate_base_validation_mse": gate["base_validation_mse"],
        "gate_base_validation_pearson": gate["base_validation_pearson"],
    }
    pack_model(arrays, "global", global_model)
    pack_model(arrays, "local", local_model)
    pack_model(arrays, "anchor", anchor_model)
    model_path = output / "source_model.npz"
    np.savez(model_path, **arrays)
    print(f"{tumor}: source model serialized", flush=True)

    registrations = {}
    mapped_by_seed = {}
    target_tree = cKDTree(target_xy)
    target_spacing = float(np.median(target_tree.query(target_xy, k=2)[0][:, 1]))
    for seed in SEEDS:
        mapped, rigid, metadata = registered_coordinates(
            source_xy, target_xy, REGISTRATION_EPOCHS, seed, args.device
        )
        mapped_path = output / f"seed{seed}_registered_source_xy.npy"
        rigid_path = output / f"seed{seed}_rigid_source_xy.npy"
        np.save(mapped_path, mapped)
        np.save(rigid_path, rigid)
        distance_to_target = target_tree.query(mapped)[0]
        overlap = float(np.mean(distance_to_target <= 2.0 * target_spacing))
        metadata.update({
            "seed": int(seed), "device": args.device,
            "source_within_2x_target_spacing": overlap,
            "registered_source_xy_sha256": sha256(mapped_path),
            "rigid_source_xy_sha256": sha256(rigid_path),
        })
        registrations[str(seed)] = metadata
        mapped_by_seed[seed] = mapped
        print(f"{tumor}: registration seed {seed} complete", flush=True)

    primary_tree = cKDTree(mapped_by_seed[42])
    target_to_source = primary_tree.query(target_xy)[0]
    roi_rows = np.flatnonzero(target_to_source <= 2.0 * target_spacing).astype(np.int64)
    if len(roi_rows) < 1000:
        raise RuntimeError("Geometry-only shared ROI unexpectedly small")
    roi_path = output / "geometry_only_target_roi_rows.npy"
    np.save(roi_path, roi_rows)
    stability = {}
    for left, right in ((42, 43), (42, 44), (43, 44)):
        stability[f"{left}_vs_{right}_median_source_displacement"] = float(
            np.median(np.linalg.norm(mapped_by_seed[left] - mapped_by_seed[right], axis=1))
        )

    request = {
        "status": "SOURCE_MODEL_AND_GEOMETRY_LOCKED_TARGET_EXPRESSION_UNSEEN",
        "pair_id": f"{tumor}_VisiumHD_FFPE_to_Xenium5K",
        "tumor": tumor,
        "source_platform": "Visium HD FFPE",
        "target_platform": "Xenium 5K",
        "source_tissue_bin_count": int(len(source_xy)),
        "target_dapi_tissue_bin_count": int(len(target_xy)),
        "geometry_only_common_roi_bin_count": int(len(roi_rows)),
        "shared_gene_count": int(len(common_genes)),
        "hidden_source_eligible_gene_count": int(len(evaluation)),
        "anchor_gene_names": common_genes[anchors].tolist(),
        "anchor_indices": anchors.tolist(),
        "anchor_metadata": anchor_metadata,
        "source_prevalence": np.mean(source_log > 0, axis=0)[anchors].tolist(),
        "selected_source_configuration": chosen,
        "source_selection_grid": grid,
        "fusion_candidate_names": [
            *[f"global_anchor_w{x:.2f}" for x in FUSION_GRID],
            *[f"local_anchor_w{x:.2f}" for x in FUSION_GRID],
            "registered_idw",
        ],
        "source_coordinate_transform": source_transform,
        "target_coordinate_transform": target_transform,
        "target_spacing_canonical": target_spacing,
        "common_roi_rule": "target bin nearest registered seed-42 source bin <= 2x median target nearest-neighbor spacing",
        "registration": registrations,
        "registration_seed_stability": stability,
        "source_path": str(source_path), "source_sha256": sha256(source_path),
        "sealed_target_geometry_container": str(target_path),
        "sealed_target_container_sha256": sha256(target_path),
        "source_model_sha256": sha256(model_path),
        "geometry_only_target_roi_rows_sha256": sha256(roi_path),
        "target_nonanchor_expression_values_read": 0,
        "target_anchor_expression_values_read": 0,
        "target_annotations_used": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    save_json(output / "source_geometry_lock.json", request)
    print(json.dumps({
        "tumor": tumor, "anchors": request["anchor_gene_names"],
        "hidden": len(evaluation), "roi_bins": len(roi_rows), "chosen": chosen,
        "seconds": request["runtime_seconds"],
    }))


if __name__ == "__main__":
    main()
