#!/usr/bin/env python3
"""Fit one source-only DLPFC direction and emit an anchor request.

This program receives no target expression path. Target information is limited
to the section identifier used to name the later anchor-only request.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from dlpfc_common import (
    ANCHOR_COUNT, GAIN_GRID, RIDGE_GRID, RANK, apply_centered,
    candidate_fusion, canonicalize, fit_centered_ridge_rank,
    idw, load_10x_h5, load_section_metadata, normalized_source, pearson_columns,
    quadrants, residual_features, save_json, select_anchors, sha256,
    source_gate, source_svg_indices,
)


def pack_model(output, prefix, model):
    for key, value in model.items():
        output[f"{prefix}_{key}"] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--spot-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    counts_sparse, barcodes, gene_ids, gene_names = load_10x_h5(args.source_h5)
    metadata = load_section_metadata(args.spot_metadata, args.source, barcodes, include_layers=False)
    if not np.all(metadata["donor"].to_numpy() == args.donor):
        raise RuntimeError("Donor lock mismatch")
    if not np.all(metadata["adjacent_pair"].to_numpy() == args.pair):
        raise RuntimeError("Pair lock mismatch")
    coordinates_raw = metadata[["x_fullres_px", "y_fullres_px"]].to_numpy(np.float32)
    coordinates, coordinate_transform = canonicalize(coordinates_raw)
    source_values, source_log, source_depth, depth_target = normalized_source(counts_sparse)

    anchor_indices, eligible_indices, anchor_metadata = select_anchors(source_log, gene_names)
    if len(anchor_indices) != ANCHOR_COUNT:
        raise AssertionError("Anchor count changed")
    anchor_set = set(anchor_indices.tolist())
    evaluation_indices = np.asarray(
        [index for index in eligible_indices if index not in anchor_set], np.int64
    )

    labels = quadrants(coordinates)
    if set(np.unique(labels)) != {0, 1, 2, 3}:
        raise RuntimeError("All four frozen 2D median quadrants are required")
    pseudo_base = np.empty_like(source_values)
    quadrant_counts = {}
    for label in range(4):
        query = np.flatnonzero(labels == label)
        reference = np.flatnonzero(labels != label)
        quadrant_counts[str(label)] = int(len(query))
        pseudo_base[query] = idw(
            coordinates[reference], source_values[reference], coordinates[query], 12
        )
    pseudo_base_log = np.log1p(pseudo_base).astype(np.float32)
    source_residual = source_log - pseudo_base_log
    raw_features = source_residual[:, anchor_indices]
    local_features = residual_features(raw_features, coordinates, True)
    validation = np.isin(labels, [0, 3])
    train = ~validation

    source_selection = []
    best = None
    for ridge in RIDGE_GRID:
        global_model = fit_centered_ridge_rank(
            raw_features[train], source_residual[train], ridge, RANK
        )
        local_model = fit_centered_ridge_rank(
            local_features[train], source_residual[train], ridge, RANK
        )
        anchor_model = fit_centered_ridge_rank(
            source_log[train][:, anchor_indices], source_log[train], ridge, RANK
        )
        global_base = apply_centered(global_model, raw_features[validation])
        local_base = apply_centered(local_model, local_features[validation])
        anchor_log = apply_centered(anchor_model, source_log[validation][:, anchor_indices])
        for gain in GAIN_GRID:
            global_log = pseudo_base_log[validation] + gain * global_base
            local_log = pseudo_base_log[validation] + gain * local_base
            candidates, names = candidate_fusion(
                pseudo_base_log[validation], global_log, local_log, anchor_log
            )
            selection, predictable, gate = source_gate(
                candidates, source_log[validation], pseudo_base_log[validation]
            )
            selected = candidates[selection, :, np.arange(source_log.shape[1])].T
            correlation = float(np.nanmedian(pearson_columns(source_log[validation], selected)))
            rmse = float(np.sqrt(np.mean((source_log[validation] - selected) ** 2)))
            record = {
                "ridge": float(ridge), "gain": float(gain),
                "median_spatial_pearson": correlation, "log1p_rmse": rmse,
                "predictable_genes": int(predictable.sum()),
            }
            source_selection.append(record)
            key = (correlation, -rmse, int(predictable.sum()))
            if best is None or key > best[0]:
                best = (key, record, names, selection.copy(), predictable.copy(), gate)
            del candidates, selected, global_log, local_log

    chosen, fusion_names, fusion_selection, predictable, gate_metadata = best[1:]
    ridge, gain = chosen["ridge"], chosen["gain"]
    global_model = fit_centered_ridge_rank(raw_features, source_residual, ridge, RANK)
    local_model = fit_centered_ridge_rank(local_features, source_residual, ridge, RANK)
    anchor_model = fit_centered_ridge_rank(
        source_log[:, anchor_indices], source_log, ridge, RANK
    )
    predictable[anchor_indices] = False
    anchor_sum = np.asarray(counts_sparse[:, anchor_indices].sum(axis=1)).ravel()
    total_sum = np.asarray(counts_sparse.sum(axis=1)).ravel()
    depth_x = np.c_[np.ones(len(anchor_sum)), np.log1p(anchor_sum)]
    depth_weight = np.linalg.solve(
        depth_x.T @ depth_x + np.diag([0.0, 1.0]),
        depth_x.T @ np.log1p(total_sum),
    ).astype(np.float32)
    svg_pool, svg_scores = source_svg_indices(source_log, coordinates, count=220)
    eligible_set = set(evaluation_indices.tolist())
    evaluation_svg = np.asarray(
        [index for index in svg_pool if index in eligible_set][:100], np.int64
    )
    thresholds = np.quantile(source_values[:, evaluation_svg], 0.8, axis=0).astype(np.float32)

    arrays = {
        "anchor_indices": anchor_indices,
        "eligible_indices": eligible_indices,
        "evaluation_indices": evaluation_indices,
        "evaluation_svg": evaluation_svg,
        "evaluation_svg_thresholds": thresholds,
        "predictable": predictable,
        "fusion_selection": fusion_selection,
        "depth_weight": depth_weight,
        "source_depth_target": np.asarray(depth_target, np.float32),
        "source_coordinates": coordinates,
        "source_barcodes": barcodes,
        "gene_ids": gene_ids,
        "gene_names": gene_names,
        "gate_selected_validation_mse": gate_metadata["selected_validation_mse"],
        "gate_selected_validation_pearson": gate_metadata["selected_validation_pearson"],
        "gate_base_validation_mse": gate_metadata["base_validation_mse"],
        "gate_base_validation_pearson": gate_metadata["base_validation_pearson"],
    }
    pack_model(arrays, "global", global_model)
    pack_model(arrays, "local", local_model)
    pack_model(arrays, "anchor", anchor_model)
    model_path = args.output / "source_model.npz"
    np.savez(model_path, **arrays)

    request = {
        "status": "SOURCE_ONLY_FIT_COMPLETE_TARGET_ANCHORS_REQUESTED",
        "direction": f"{args.source}_to_{args.target}",
        "source": args.source, "target": args.target, "donor": args.donor,
        "adjacent_pair": args.pair,
        "target_allowed_fields": ["barcode", "x_fullres_px", "y_fullres_px", "16 requested anchor counts"],
        "target_forbidden_fields": ["all non-anchor counts", "logcounts", "cortical_layer"],
        "anchor_gene_ids": gene_ids[anchor_indices].tolist(),
        "anchor_gene_names": gene_names[anchor_indices].tolist(),
        "anchor_indices_on_common_axis": anchor_indices.tolist(),
        "eligible_nonanchor_gene_count": int(len(evaluation_indices)),
        "source_defined_predictable_nonanchor_count": int(predictable[evaluation_indices].sum()),
        "anchor_metadata": anchor_metadata,
        "coordinate_transform": coordinate_transform,
        "quadrant_counts": quadrant_counts,
        "source_validation_quadrants": [0, 3],
        "source_training_quadrants": [1, 2],
        "selected_source_configuration": chosen,
        "source_selection_grid": source_selection,
        "fusion_candidate_names": fusion_names,
        "evaluation_svg_count": int(len(evaluation_svg)),
        "source_h5_sha256": sha256(args.source_h5),
        "source_model_path": str(model_path),
        "source_model_sha256": sha256(model_path),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    save_json(args.output / "anchor_request.json", request)
    with (args.output / "anchors.tsv").open("w") as handle:
        handle.write("direction\tposition\tgene_id\tgene_name\n")
        for position, index in enumerate(anchor_indices, 1):
            handle.write(
                f"{args.source}_to_{args.target}\t{position}\t{gene_ids[index]}\t{gene_names[index]}\n"
            )
    print(json.dumps({
        "direction": request["direction"], "anchors": request["anchor_gene_names"],
        "eligible": request["eligible_nonanchor_gene_count"],
        "predictable": request["source_defined_predictable_nonanchor_count"],
        "chosen": chosen, "seconds": request["runtime_seconds"],
    }))


if __name__ == "__main__":
    main()
