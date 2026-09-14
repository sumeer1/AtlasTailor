#!/usr/bin/env python3
"""Prediction-only process for one frozen DLPFC direction.

There is intentionally no target HDF5 or target metadata argument. The only
target object accepted contains XY, barcodes and the requested 16 anchor values.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from hyperspatial_molecular_hologram import RegistrationConfig, fit_geometry_field

from dlpfc_common import (
    METHODS, SEEDS, apply_centered, candidate_fusion, load_10x_h5,
    neighbor_index_weights, predicted_depth, residual_features, save_json, sha256,
)


def model_from_npz(archive, prefix):
    return {
        key: archive[f"{prefix}_{key}"]
        for key in ("x_mean", "x_scale", "y_mean", "weight")
    }


def similarity_icp(source, target, iterations=30):
    mapped = source.copy()
    for _ in range(iterations):
        match = target[cKDTree(target).query(mapped)[1]]
        x0, y0 = mapped.mean(axis=0), match.mean(axis=0)
        x, y = mapped - x0, match - y0
        u, singular, vt = np.linalg.svd(x.T @ y)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        scale = singular.sum() / max(np.square(x).sum(), 1e-8)
        mapped = scale * (mapped - x0) @ rotation.T + y0
    return mapped.astype(np.float32)


def chamfer(a, b):
    return float(0.5 * (
        cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()
    ))


def best_rigid(source, target):
    candidates = {}
    choices = {
        "none": (1, 1), "reflect_x": (-1, 1),
        "reflect_y": (1, -1), "reflect_xy": (-1, -1),
    }
    for name, signs in choices.items():
        reflected = source.copy()
        reflected[:, :2] = 0.5 + (reflected[:, :2] - 0.5) * np.asarray(signs)
        mapped = similarity_icp(reflected, target, iterations=30)
        candidates[name] = (chamfer(mapped, target), mapped)
    selected = min(candidates, key=lambda key: candidates[key][0])
    return candidates[selected][1], selected, {
        key: float(value[0]) for key, value in candidates.items()
    }


def register(source, target, seed):
    rigid, reflection, candidates = best_rigid(source, target)
    field, history = fit_geometry_field(
        rigid,
        target,
        "cpu",
        RegistrationConfig(epochs=30, batch_size=min(768, len(source)), seed=seed),
    )
    with torch.no_grad():
        mapped = field(torch.as_tensor(rigid, dtype=torch.float32)).cpu().numpy()
    return mapped.astype(np.float32), {
        "seed": seed,
        "device": "cpu",
        "reflection": reflection,
        "reflection_candidate_chamfers": candidates,
        "raw_chamfer": chamfer(source, target),
        "rigid_chamfer": chamfer(rigid, target),
        "registered_chamfer": chamfer(mapped, target),
        "final_geometry_loss": float(history["loss"][-1]),
        "final_fold_loss": float(history["fold"][-1]),
    }


def apply_indexed(values, index, weight, columns, gene_chunk=1024):
    output = np.empty((len(index), len(columns)), np.float32)
    for begin in range(0, len(columns), gene_chunk):
        end = min(begin + gene_chunk, len(columns))
        block = columns[begin:end]
        output[:, begin:end] = np.einsum(
            "qk,qkg->qg", weight, values[index[:, :, None], block[None, None, :]], optimize=True
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--target-anchor-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    request = json.loads(args.request.read_text())
    model = np.load(args.model, allow_pickle=False)
    target = np.load(args.target_anchor_object, allow_pickle=False)
    if not np.array_equal(target["anchor_gene_ids"], model["gene_ids"][model["anchor_indices"]]):
        raise RuntimeError("Target anchor object does not match source model")

    source_counts, source_barcodes, source_gene_ids, source_gene_names = load_10x_h5(args.source_h5)
    if not np.array_equal(source_barcodes, model["source_barcodes"]):
        raise RuntimeError("Source barcode order changed")
    if not np.array_equal(source_gene_ids, model["gene_ids"]):
        raise RuntimeError("Source feature axis changed")
    source_depth = np.maximum(np.asarray(source_counts.sum(axis=1)).ravel(), 1.0)
    depth_target = float(model["source_depth_target"])
    source_values = source_counts.toarray().astype(np.float32)
    source_values *= (depth_target / source_depth)[:, None]

    source_coordinates = model["source_coordinates"].astype(np.float32)
    target_coordinates = target["target_coordinates"].astype(np.float32)
    anchor_indices = model["anchor_indices"].astype(np.int64)
    evaluation_indices = model["evaluation_indices"].astype(np.int64)
    predictable = model["predictable"].astype(bool)[evaluation_indices]
    fusion_selection = model["fusion_selection"].astype(np.int64)[evaluation_indices]
    global_model = model_from_npz(model, "global")
    local_model = model_from_npz(model, "local")
    anchor_model = model_from_npz(model, "anchor")
    gain = float(request["selected_source_configuration"]["gain"])
    target_anchor_counts = target["anchor_counts"].astype(np.float32)
    target_depth = predicted_depth(target_anchor_counts, model["depth_weight"])
    target_anchor_log = np.log1p(
        target_anchor_counts * (depth_target / target_depth)[:, None]
    ).astype(np.float32)

    manifest = {
        "status": "PREDICTION_IN_PROGRESS_TARGET_NONANCHOR_UNAVAILABLE",
        "direction": request["direction"],
        "source": request["source"], "target": request["target"],
        "donor": request["donor"], "adjacent_pair": request["adjacent_pair"],
        "source_model_sha256": sha256(args.model),
        "target_anchor_object_sha256": sha256(args.target_anchor_object),
        "source_h5_sha256": sha256(args.source_h5),
        "target_hidden_expression_path_argument": None,
        "target_layer_argument": None,
        "eligible_nonanchor_gene_count": int(len(evaluation_indices)),
        "guard_supported_count": int(predictable.sum()),
        "guard_fallback_count": int((~predictable).sum()),
        "methods": {},
    }
    for seed in SEEDS:
        seed_started = time.perf_counter()
        torch.manual_seed(seed)
        mapped, geometry = register(source_coordinates, target_coordinates, seed)
        union = np.unique(np.r_[evaluation_indices, anchor_indices])
        union_position = {int(gene): local for local, gene in enumerate(union)}
        index12, weight12 = neighbor_index_weights(mapped, target_coordinates, 12)
        index1, weight1 = neighbor_index_weights(mapped, target_coordinates, 1)
        prior_union = apply_indexed(source_values, index12, weight12, union)
        eval_positions = np.asarray([union_position[int(gene)] for gene in evaluation_indices])
        anchor_positions = np.asarray([union_position[int(gene)] for gene in anchor_indices])
        base_log = np.log1p(prior_union[:, eval_positions]).astype(np.float32)
        base_anchor_log = np.log1p(prior_union[:, anchor_positions]).astype(np.float32)
        nearest_log = np.log1p(
            apply_indexed(source_values, index1, weight1, evaluation_indices)
        ).astype(np.float32)
        anchor_residual = target_anchor_log - base_anchor_log
        global_log = base_log + gain * apply_centered(
            global_model, anchor_residual, evaluation_indices
        )
        local_log = base_log + gain * apply_centered(
            local_model,
            residual_features(anchor_residual, target_coordinates, True),
            evaluation_indices,
        )
        anchor_log = apply_centered(anchor_model, target_anchor_log, evaluation_indices)
        candidates, names = candidate_fusion(
            base_log, global_log, local_log, anchor_log
        )
        if names != request["fusion_candidate_names"]:
            raise RuntimeError("Fusion candidate order changed")
        map_log = candidates[
            fusion_selection, :, np.arange(len(evaluation_indices))
        ].T
        map_log[:, ~predictable] = base_log[:, ~predictable]
        predictions = {
            "registered_nearest_neighbor": nearest_log,
            "registered_idw_k12": base_log,
            "atlas_plus_global_anchor_residual": np.maximum(global_log, 0),
            "atlas_plus_local_anchor_residual": np.maximum(local_log, 0),
            "anchor_only_calibrated": np.maximum(anchor_log, 0),
            "hyperspatial_map_calibrated_fusion": np.maximum(map_log, 0),
        }
        seed_manifest = {"geometry": geometry, "predictions": {}}
        mapped_path = args.output / f"seed{seed}_registered_source_xy.npy"
        np.save(mapped_path, mapped)
        seed_manifest["registered_source_xy"] = {
            "path": str(mapped_path), "sha256": sha256(mapped_path)
        }
        for method in METHODS:
            path = args.output / f"seed{seed}_{method}.npy"
            np.save(path, predictions[method].astype(np.float32))
            seed_manifest["predictions"][method] = {
                "path": str(path), "sha256": sha256(path),
                "shape": list(predictions[method].shape),
            }
        seed_manifest["runtime_seconds"] = time.perf_counter() - seed_started
        manifest["methods"][str(seed)] = seed_manifest
        save_json(args.output / "prediction_manifest.in_progress.json", manifest)
        del candidates, predictions, prior_union, base_log, nearest_log

    manifest["status"] = "ALL_PREDICTIONS_SERIALIZED_AND_HASHED_EVALUATION_FORBIDDEN_UNTIL_GLOBAL_HASH_TABLE"
    manifest["runtime_seconds"] = time.perf_counter() - started
    save_json(args.output / "prediction_manifest.json", manifest)
    (args.output / "prediction_manifest.in_progress.json").unlink()
    print(json.dumps({
        "direction": manifest["direction"], "status": manifest["status"],
        "seconds": manifest["runtime_seconds"],
        "guard_supported": manifest["guard_supported_count"],
        "fallback": manifest["guard_fallback_count"],
    }))


if __name__ == "__main__":
    main()
