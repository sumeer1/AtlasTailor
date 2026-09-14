#!/usr/bin/env python3
"""Expose only 16 Xenium anchors, predict hidden genes, then hash the lock."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "analysis" / "hyperspatial_map_2d_dlpfc"))

from dlpfc_common import (  # noqa: E402
    apply_centered, neighbor_index_weights, normalized_source, predicted_depth,
    pearson_columns, residual_features, save_json, sha256,
)

DATA = ROOT / "official_data" / "extracted"
SEALED = ROOT / "SEALED_TARGET_8UM"
WORK = ROOT / "work"


def model_part(model, prefix):
    return {key: model[f"{prefix}_{key}"] for key in ("x_mean", "x_scale", "y_mean", "weight")}


def load_source(tumor, genes, rows):
    path = DATA / f"VisiumHD_FFPE_{tumor}" / "adata.h5ad"
    data = ad.read_h5ad(path, backed="r")
    by_name = {name: i for i, name in enumerate(data.var_names.astype(str))}
    columns = np.asarray([by_name[g] for g in genes], np.int64)
    counts = data[rows, columns].X
    if hasattr(counts, "to_memory"):
        counts = counts.to_memory()
    counts = counts.tocsr()
    data.file.close()
    return path, counts


def extract_anchor_object(tumor, model, roi, path):
    target_path = SEALED / f"Xenium_{tumor}_8um_shared_counts.h5ad"
    target = ad.read_h5ad(target_path, backed="r")
    genes = target.var_names.astype(str).to_numpy()
    if not np.array_equal(genes, model["gene_names"]):
        raise RuntimeError("Target shared-gene axis changed")
    anchors = model["anchor_indices"].astype(np.int64)
    # This backed sparse slice materializes values only for the 16 requested
    # columns and the geometry-locked ROI rows.
    selected = target[roi, anchors].X
    if hasattr(selected, "to_memory"):
        selected = selected.to_memory()
    anchor_counts = selected.toarray().astype(np.float32)
    coordinates = np.asarray(target.obsm["spatial"], np.float32)[roi]
    ids = target.obs_names.astype(str).to_numpy()[roi]
    target.file.close()
    np.savez(
        path, anchor_counts=anchor_counts, target_coordinates_raw=coordinates,
        target_ids=ids, target_rows=roi, anchor_indices=anchors,
        anchor_gene_names=genes[anchors],
    )
    return target_path, anchor_counts, coordinates, ids


def indexed_idw(source_values, index, weight, columns):
    return np.einsum(
        "qk,qkg->qg", weight,
        source_values[index[:, :, None], columns[None, None, :]], optimize=True,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor", choices=("COAD", "HCC", "OV"), required=True)
    parser.add_argument("--gene-chunk", type=int, default=128)
    args = parser.parse_args()
    tumor = args.tumor
    pair = WORK / tumor
    output = pair / "predictions"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()

    lock = json.loads((pair / "source_geometry_lock.json").read_text())
    model_path = pair / "source_model.npz"
    model = np.load(model_path, allow_pickle=True)
    genes = model["gene_names"].astype(str)
    rows = model["source_rows"].astype(np.int64)
    source_path, source_counts = load_source(tumor, genes, rows)
    source_values, _, _, depth_target = normalized_source(source_counts)
    if not np.isclose(depth_target, float(model["source_depth_target"])):
        raise RuntimeError("Source normalization depth changed")

    roi_path = pair / "geometry_only_target_roi_rows.npy"
    roi = np.load(roi_path)
    anchor_path = pair / "target_anchor_only_object.npz"
    target_path, anchor_counts, target_raw, target_ids = extract_anchor_object(
        tumor, model, roi, anchor_path
    )
    transform = lock["target_coordinate_transform"]
    target_xy = ((target_raw - np.asarray(transform["center"], np.float32)) /
                 float(transform["scale"])).astype(np.float32)
    anchor_indices = model["anchor_indices"].astype(np.int64)
    evaluation = model["evaluation_indices"].astype(np.int64)
    predicted_target_depth = predicted_depth(anchor_counts, model["depth_weight"])
    target_anchor_log = np.log1p(
        anchor_counts * (depth_target / predicted_target_depth)[:, None]
    ).astype(np.float32)

    mapped_path = pair / "seed42_registered_source_xy.npy"
    mapped = np.load(mapped_path, mmap_mode="r")
    index12, weight12 = neighbor_index_weights(mapped, target_xy, 12)
    base_anchor = indexed_idw(source_values, index12, weight12, anchor_indices)
    rigid_path = pair / "seed42_rigid_source_xy.npy"
    rigid = np.load(rigid_path, mmap_mode="r")
    rigid_index, rigid_weight = neighbor_index_weights(rigid, target_xy, 12)
    rigid_anchor = indexed_idw(source_values, rigid_index, rigid_weight, anchor_indices)
    platform_control = {
        "definition": "gene-wise spatial Pearson of target anchors versus the 12-NN atlas expectation",
        "before_nonrigid_registration_median": float(np.nanmedian(pearson_columns(target_anchor_log, np.log1p(rigid_anchor)))),
        "after_nonrigid_registration_median": float(np.nanmedian(pearson_columns(target_anchor_log, np.log1p(base_anchor)))),
        "per_anchor_before": pearson_columns(target_anchor_log, np.log1p(rigid_anchor)).tolist(),
        "per_anchor_after": pearson_columns(target_anchor_log, np.log1p(base_anchor)).tolist(),
    }
    save_json(pair / "platform_anchor_correlation_control.json", platform_control)
    anchor_residual = target_anchor_log - np.log1p(base_anchor)
    local_features = residual_features(anchor_residual, target_xy, True)

    shapes = (len(roi), len(evaluation))
    paths = {
        "registered_idw_k12": output / "registered_idw_k12_log1p.npy",
        "anchor_only_decoder": output / "anchor_only_decoder_log1p.npy",
        "hyperspatial_map": output / "hyperspatial_map_log1p.npy",
    }
    arrays = {
        name: np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shapes)
        for name, path in paths.items()
    }
    global_model = model_part(model, "global")
    local_model = model_part(model, "local")
    anchor_model = model_part(model, "anchor")
    gain = float(lock["selected_source_configuration"]["gain"])
    selection = model["fusion_selection"].astype(np.int64)[evaluation]
    predictable = model["predictable"].astype(bool)[evaluation]

    for begin in range(0, len(evaluation), args.gene_chunk):
        end = min(begin + args.gene_chunk, len(evaluation))
        columns = evaluation[begin:end]
        base = np.log1p(indexed_idw(source_values, index12, weight12, columns)).astype(np.float32)
        global_log = base + gain * apply_centered(global_model, anchor_residual, columns)
        local_log = base + gain * apply_centered(local_model, local_features, columns)
        anchor_log = apply_centered(anchor_model, target_anchor_log, columns)
        chosen = selection[begin:end]
        map_log = base.copy()
        for local, candidate in enumerate(chosen):
            if candidate < 5:
                weight = 0.25 * candidate
                map_log[:, local] = (1.0 - weight) * global_log[:, local] + weight * anchor_log[:, local]
            elif candidate < 10:
                weight = 0.25 * (candidate - 5)
                map_log[:, local] = (1.0 - weight) * local_log[:, local] + weight * anchor_log[:, local]
            elif candidate != 10:
                raise RuntimeError(f"Unexpected frozen fusion index {candidate}")
        unsupported = ~predictable[begin:end]
        map_log[:, unsupported] = base[:, unsupported]
        arrays["registered_idw_k12"][:, begin:end] = base
        arrays["anchor_only_decoder"][:, begin:end] = np.maximum(anchor_log, 0)
        arrays["hyperspatial_map"][:, begin:end] = np.maximum(map_log, 0)
        for array in arrays.values():
            array.flush()

    del arrays
    prediction_records = {
        name: {"path": str(path), "sha256": sha256(path), "shape": list(shapes),
               "representation": "log1p source-depth-normalized expression"}
        for name, path in paths.items()
    }
    target_depth_path = pair / "source_model_predicted_target_depth.npy"
    np.save(target_depth_path, predicted_target_depth)
    manifest = {
        "status": "PREDICTIONS_LOCKED_AND_HASHED_HIDDEN_TARGET_EXPRESSION_STILL_UNOPENED",
        "pair_id": lock["pair_id"], "tumor": tumor,
        "source_model": str(model_path), "source_model_sha256": sha256(model_path),
        "registration": str(mapped_path), "registration_sha256": sha256(mapped_path),
        "rigid_registration": str(rigid_path), "rigid_registration_sha256": sha256(rigid_path),
        "geometry_only_roi": str(roi_path), "geometry_only_roi_sha256": sha256(roi_path),
        "selected_anchor_gene_names": genes[anchor_indices].tolist(),
        "anchor_only_object": str(anchor_path), "anchor_only_object_sha256": sha256(anchor_path),
        "anchor_values_materialized": int(anchor_counts.size),
        "nonanchor_target_values_materialized_before_lock": 0,
        "model_configuration": lock["selected_source_configuration"],
        "operator_assignment_storage": "source_model.npz:fusion_selection,predictable",
        "guard_supported_hidden_genes": int(predictable.sum()),
        "registered_idw_fallback_hidden_genes": int((~predictable).sum()),
        "hidden_gene_count": int(len(evaluation)), "target_roi_bin_count": int(len(roi)),
        "predicted_target_depth": str(target_depth_path),
        "predicted_target_depth_sha256": sha256(target_depth_path),
        "B_T": prediction_records["registered_idw_k12"],
        "predictions": prediction_records,
        "source_input_sha256": sha256(source_path),
        "sealed_target_container_sha256": sha256(target_path),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    save_json(pair / "prediction_lock_manifest.json", manifest)
    print(json.dumps({
        "tumor": tumor, "status": manifest["status"], "shape": shapes,
        "supported": manifest["guard_supported_hidden_genes"],
        "seconds": manifest["runtime_seconds"],
    }))


if __name__ == "__main__":
    main()
