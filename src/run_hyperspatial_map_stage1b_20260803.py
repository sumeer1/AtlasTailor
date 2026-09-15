#!/usr/bin/env python3
"""Calibrated source-gated HyperSpatial-MAP development cycle.

This cycle fixes the missing absolute-decoder intercept identified in Stage 1
and uses source pseudo-blocks to choose a per-gene atlas/anchor fusion. Target
unmeasured genes remain unopened until all predictions are serialized.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from run_hyperspatial_map_stage1_20260803 import (
    AXIAL,
    GAIN_GRID,
    K_LOCAL,
    RANK,
    RIDGE_GRID,
    apply_linear as _old_apply_linear,
    depth_predictor,
    features,
    idw,
    octants,
    predicted_depth,
    pseudo_block_prediction,
    residual_pearson,
    sha256,
    target_anchor_counts,
)
from run_mouse_temporal_transport_gate import source_svg_indices
from run_wemerfish_temporal_holdout import (
    canonicalize,
    coordinates_only,
    evaluate,
    measured_counts,
    normalize_counts,
    pearson_columns,
    registered_coordinates,
)


SEEDS = (42, 43, 44)
ANCHOR_COUNT = 16
FUSION_GRID = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), np.float32)
EXPANDED_EXCLUDE = {
    "admp", "bambia", "foxa", "foxa2", "foxa3", "gata6", "gbx1", "gli1",
    "lhx1a", "msgn1", "noto", "otx1", "shha", "sox19a", "sp5l", "tbx1",
    "tbxta", "tbxtb", "zic2b",
}


def fit_centered_ridge_rank(x, y, ridge, rank):
    x_mean = x.mean(0)
    x_scale = np.maximum(x.std(0), 1e-5)
    y_mean = y.mean(0)
    xs = (x - x_mean) / x_scale
    yc = y - y_mean
    weight = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ yc)
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    keep = min(rank, len(s))
    weight = (u[:, :keep] * s[:keep]) @ vt[:keep]
    return {
        "x_mean": x_mean.astype(np.float32), "x_scale": x_scale.astype(np.float32),
        "y_mean": y_mean.astype(np.float32), "weight": weight.astype(np.float32),
    }


def apply_centered(model, x):
    return model["y_mean"] + ((x - model["x_mean"]) / model["x_scale"]) @ model["weight"]


def select_non_axial_anchors(source_log, genes, count):
    prevalence = np.mean(source_log > 0, axis=0)
    variance = np.var(source_log, axis=0)
    candidates = np.asarray([
        index for index, gene in enumerate(genes)
        if 0.05 <= prevalence[index] <= 0.98 and variance[index] > 1e-5
        and gene.lower() not in EXPANDED_EXCLUDE
        and not gene.lower().startswith(("mt-", "rpl", "rps"))
    ], np.int64)
    rng = np.random.default_rng(42116)
    sample = np.sort(rng.choice(len(source_log), min(5000, len(source_log)), replace=False))
    x = source_log[sample]
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-5)
    correlation = np.abs((x[:, candidates].T @ x) / max(len(x) - 1, 1))
    axial_indices = [int(np.flatnonzero(genes == gene)[0]) for gene in AXIAL if np.any(genes == gene)]
    if axial_indices:
        allowed = np.max(correlation[:, axial_indices], axis=1) < 0.40
        candidates, correlation = candidates[allowed], correlation[allowed]
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} strictly non-axial anchor candidates")
    coverage = np.zeros(len(genes), np.float32)
    chosen = []
    for _ in range(count):
        gain = np.mean(np.maximum(coverage[None, :], correlation**2) - coverage[None, :], axis=1)
        for index in chosen:
            local = np.flatnonzero(candidates == index)
            if len(local):
                gain[local[0]] = -np.inf
        local = int(np.argmax(gain))
        chosen.append(int(candidates[local]))
        coverage = np.maximum(coverage, correlation[local] ** 2)
    return np.asarray(chosen, np.int64), {
        "candidate_count": int(len(candidates)),
        "mean_squared_gene_coverage": float(coverage.mean()),
        "maximum_allowed_source_correlation_to_axial": 0.40,
    }


def candidate_fusion(base_log, global_log, local_log, anchor_log):
    candidates, names = [], []
    for residual_name, residual_prediction in (("global", global_log), ("local", local_log)):
        for weight in FUSION_GRID:
            candidates.append((1 - weight) * residual_prediction + weight * anchor_log)
            names.append(f"{residual_name}_anchor_w{weight:.2f}")
    candidates.append(base_log)
    names.append("registered_idw")
    return np.stack(candidates, axis=0).astype(np.float32), names


def source_gate(candidates, truth, base, tolerance=1.05):
    # Choose the highest-correlation source candidate among candidates whose
    # gene-wise MSE is within 5% of the best source-validation MSE.
    errors = np.mean((candidates - truth[None, :, :]) ** 2, axis=1)
    correlations = np.stack([pearson_columns(truth, candidate) for candidate in candidates])
    best_error = errors.min(0)
    eligible = errors <= tolerance * np.maximum(best_error[None, :], 1e-10)
    score = np.where(eligible, correlations, -np.inf)
    selection = np.argmax(score, axis=0)
    selected = np.column_stack([candidates[selection[gene], :, gene] for gene in range(truth.shape[1])])
    base_error = np.mean((base - truth) ** 2, axis=0)
    selected_error = np.mean((selected - truth) ** 2, axis=0)
    selected_correlation = correlations[selection, np.arange(truth.shape[1])]
    base_correlation = pearson_columns(truth, base)
    predictable = (selected_error < base_error) & (selected_correlation > base_correlation + 0.03)
    return selection.astype(np.int64), predictable, {
        "selected_validation_mse": selected_error.tolist(),
        "selected_validation_pearson": selected_correlation.tolist(),
        "base_validation_mse": base_error.tolist(),
        "base_validation_pearson": base_correlation.tolist(),
    }


def write_tsv(path, records):
    columns = list(records[0])
    with path.open("x") as handle:
        handle.write("\t".join(columns) + "\n")
        for record in records:
            handle.write("\t".join(str(record[column]) for column in columns) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", choices=("E1", "E2"), required=True)
    parser.add_argument("--target", choices=("E1", "E2"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.source == args.target or args.output.exists():
        raise RuntimeError("Invalid direction or output already exists")
    lock = json.loads(args.lock.read_text())
    if lock["runner_sha256"] != sha256(Path(__file__).resolve()):
        raise RuntimeError("Runner differs from protocol lock")
    direction = f"{args.source}_to_{args.target}"
    if direction != lock["authorized_direction"]:
        raise RuntimeError(f"Direction {direction} is not authorized")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    args.output.mkdir(parents=True)
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir()
    started = time.perf_counter()

    source_path = args.data / f"weMERFISH_measured_B_75p_{args.source}.h5ad"
    target_path = args.data / f"weMERFISH_measured_B_75p_{args.target}.h5ad"
    source_counts, source_coordinates_raw, genes = measured_counts(source_path)
    source_coordinates = canonicalize(source_coordinates_raw)[0]
    target_coordinates_raw, target_genes, target_ids, coordinate_key = coordinates_only(target_path)
    target_coordinates = canonicalize(target_coordinates_raw)[0]
    if not np.array_equal(genes, target_genes):
        raise AssertionError("Gene axes differ")
    source_depth = np.maximum(source_counts.sum(1), 1.0)
    depth_target = float(np.median(source_depth))
    source_values = normalize_counts(source_counts, depth_target).astype(np.float32)
    source_log = np.log1p(source_values).astype(np.float32)
    anchor_indices, anchor_metadata = select_non_axial_anchors(source_log, genes, ANCHOR_COUNT)
    anchor_genes = genes[anchor_indices]
    target_anchor_raw = target_anchor_counts(target_path, anchor_indices)
    depth_model = depth_predictor(source_counts, anchor_indices)
    target_depth = predicted_depth(target_anchor_raw, depth_model)
    target_anchor_log = np.log1p(target_anchor_raw * (depth_target / target_depth)[:, None]).astype(np.float32)

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
        global_model = fit_centered_ridge_rank(raw_features[train], source_residual[train], ridge, RANK)
        local_model = fit_centered_ridge_rank(local_features[train], source_residual[train], ridge, RANK)
        anchor_model = fit_centered_ridge_rank(source_log[train][:, anchor_indices], source_log[train], ridge, RANK)
        for gain in GAIN_GRID:
            global_log = pseudo_base_log[validation] + gain * apply_centered(global_model, raw_features[validation])
            local_log = pseudo_base_log[validation] + gain * apply_centered(local_model, local_features[validation])
            anchor_log = apply_centered(anchor_model, source_log[validation][:, anchor_indices])
            candidates, names = candidate_fusion(pseudo_base_log[validation], global_log, local_log, anchor_log)
            selection, predictable, metadata = source_gate(candidates, source_log[validation], pseudo_base_log[validation])
            selected = np.column_stack([candidates[selection[gene], :, gene] for gene in range(len(genes))])
            correlation = float(np.nanmedian(pearson_columns(source_log[validation], selected)))
            rmse = float(np.sqrt(np.mean((source_log[validation] - selected) ** 2)))
            record = {"ridge": ridge, "gain": gain, "median_spatial_pearson": correlation,
                      "log1p_rmse": rmse, "predictable_genes": int(predictable.sum())}
            source_selection.append(record)
            key = (correlation, -rmse, int(predictable.sum()))
            if best is None or key > best[0]:
                best = (key, record, names, selection, predictable, metadata)
    chosen, fusion_names, fusion_selection, predictable, gate_metadata = best[1:]
    ridge, gain = chosen["ridge"], chosen["gain"]
    global_model = fit_centered_ridge_rank(raw_features, source_residual, ridge, RANK)
    local_model = fit_centered_ridge_rank(local_features, source_residual, ridge, RANK)
    anchor_model = fit_centered_ridge_rank(source_log[:, anchor_indices], source_log, ridge, RANK)
    predictable[anchor_indices] = False
    svg_order, svg_scores = source_svg_indices([source_log], [source_coordinates], count=min(220, len(genes)))
    evaluation_svg = np.asarray([index for index in svg_order if index not in set(anchor_indices)][:100], np.int64)
    thresholds = np.quantile(source_values, 0.8, axis=0).astype(np.float32)

    manifest = {
        "status": "target_unmeasured_genes_unopened",
        "direction": direction,
        "anchor_genes": anchor_genes.tolist(), "anchor_indices": anchor_indices.tolist(),
        "anchor_metadata": anchor_metadata, "expanded_biological_exclusion": sorted(EXPANDED_EXCLUDE),
        "selected_source_configuration": chosen, "source_selection": source_selection,
        "fusion_candidate_names": fusion_names, "per_gene_fusion_selection": fusion_selection.tolist(),
        "source_defined_predictable_genes": genes[predictable].tolist(),
        "source_gate_metadata": gate_metadata, "evaluation_svg_indices": evaluation_svg.tolist(),
        "evaluation_svg_scores": svg_scores[:len(evaluation_svg)].tolist(),
        "target_allowed_before_prediction": ["global_sphere geometry", "IDs", "gene names", "16 locked anchor columns"],
        "target_coordinate_key": coordinate_key, "target_ids": target_ids.tolist(),
        "methods": {},
    }
    for seed in SEEDS:
        torch.manual_seed(seed)
        mapped, _, geometry = registered_coordinates(source_coordinates, target_coordinates, 30, seed, device)
        base_log = np.log1p(idw(mapped, source_values, target_coordinates, K_LOCAL)).astype(np.float32)
        nearest_log = np.log1p(idw(mapped, source_values, target_coordinates, 1)).astype(np.float32)
        anchor_residual = target_anchor_log - base_log[:, anchor_indices]
        global_log = base_log + gain * apply_centered(global_model, anchor_residual)
        local_log = base_log + gain * apply_centered(local_model, features(anchor_residual, target_coordinates, True))
        anchor_log = apply_centered(anchor_model, target_anchor_log)
        candidates, names_check = candidate_fusion(base_log, global_log, local_log, anchor_log)
        if names_check != fusion_names:
            raise AssertionError("Fusion candidate order changed")
        map_log = np.column_stack([candidates[fusion_selection[gene], :, gene] for gene in range(len(genes))])
        # Genes not source-predictable retain the registered atlas.
        map_log[:, ~predictable] = base_log[:, ~predictable]
        predictions = {
            "registered_idw_k12": base_log,
            "registered_nearest_neighbor": nearest_log,
            "anchor_only_calibrated": np.maximum(anchor_log, 0),
            "atlas_plus_global_anchor_residual": np.maximum(global_log, 0),
            "atlas_plus_local_anchor_residual": np.maximum(local_log, 0),
            "hyperspatial_map_calibrated_fusion": np.maximum(map_log, 0),
        }
        manifest["methods"][str(seed)] = {"geometry": geometry, "predictions": {}}
        for method, prediction in predictions.items():
            path = prediction_dir / f"seed{seed}_{method}.npy"
            np.save(path, prediction.astype(np.float32))
            manifest["methods"][str(seed)]["predictions"][method] = {"path": str(path), "sha256": sha256(path)}
    manifest["status"] = "all_predictions_serialized_evaluation_pending"
    (args.output / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    leakage = {
        "status": manifest["status"], "target_unmeasured_expression_accessed_before_prediction": False,
        "target_anchor_genes_accessed": anchor_genes.tolist(), "target_annotations_accessed": False,
        "all_selection_calibration_and_fusion_source_only": True,
        "qualification": "second retrospective E1/E2 development cycle after Stage-1 failure; not independent confirmation",
    }
    (args.output / "LEAKAGE_AUDIT.json").write_text(json.dumps(leakage, indent=2) + "\n")

    target_counts, _, target_genes_check = measured_counts(target_path)
    if not np.array_equal(target_genes_check, genes):
        raise AssertionError("Target gene axis changed")
    truth_log = np.log1p(normalize_counts(target_counts, depth_target)).astype(np.float32)
    predictable_eval = np.asarray([index for index in np.flatnonzero(predictable) if index not in set(anchor_indices)], np.int64)
    rows, residual_rows, axial_rows = [], [], []
    for seed in SEEDS:
        paths = manifest["methods"][str(seed)]["predictions"]
        base_log = np.asarray(np.load(paths["registered_idw_k12"]["path"], mmap_mode="r"), np.float32)
        for method_index, (method, metadata) in enumerate(paths.items()):
            prediction = np.asarray(np.load(metadata["path"], mmap_mode="r"), np.float32)
            result = evaluate(method, truth_log, prediction, target_coordinates, evaluation_svg, thresholds, seed * 100 + method_index)
            result.pop("per_gene_pearson")
            ci = result.pop("svg_spatial_pearson_bootstrap_ci95")
            rows.append({"direction": direction, "seed": seed, **result, "pearson_ci_low": ci[0], "pearson_ci_high": ci[1]})
            correlation = residual_pearson(truth_log, base_log, prediction, predictable_eval)
            rng = np.random.default_rng(seed * 1000 + method_index)
            bootstrap = [np.nanmedian(rng.choice(correlation, len(correlation), replace=True)) for _ in range(2000)]
            residual_rows.append({"direction": direction, "seed": seed, "method": method,
                                  "predictable_genes": len(predictable_eval),
                                  "residual_spatial_pearson_median": float(np.nanmedian(correlation)),
                                  "ci95_low": float(np.nanquantile(bootstrap, 0.025)),
                                  "ci95_high": float(np.nanquantile(bootstrap, 0.975))})
            for gene in AXIAL:
                index = int(np.flatnonzero(genes == gene)[0])
                gene_result = evaluate(method, truth_log, prediction, target_coordinates, np.asarray([index]), thresholds, seed * 10000 + method_index)
                axial_rows.append({"direction": direction, "seed": seed, "method": method, "gene": gene,
                                   "spatial_pearson": gene_result["svg_spatial_pearson_median"],
                                   "topk_dice": gene_result["topk_dice_median"],
                                   "centroid_error": gene_result["weighted_centroid_error_median"]})
    write_tsv(args.output / "all_metrics.tsv", rows)
    write_tsv(args.output / "residual_metrics.tsv", residual_rows)
    write_tsv(args.output / "axial_metrics.tsv", axial_rows)

    map_name = "hyperspatial_map_calibrated_fusion"
    methods = sorted(set(row["method"] for row in rows))
    endpoints = (("svg_spatial_pearson_median", True), ("auprc_enrichment_median", True),
                 ("topk_dice_median", True), ("weighted_centroid_error_median", False),
                 ("spatial_emd_median", False), ("log1p_rmse", False))
    means = {method: {endpoint: float(np.mean([row[endpoint] for row in rows if row["method"] == method])) for endpoint, _ in endpoints} for method in methods}
    residual_means = {method: float(np.mean([row["residual_spatial_pearson_median"] for row in residual_rows if row["method"] == method])) for method in methods}
    residual_best = residual_means[map_name] >= max(value for method, value in residual_means.items() if method != map_name) - 1e-8
    endpoint_best = {}
    for endpoint, higher in endpoints:
        comparator = [means[method][endpoint] for method in methods if method != map_name]
        endpoint_best[endpoint] = means[map_name][endpoint] >= max(comparator) - 1e-8 if higher else means[map_name][endpoint] <= min(comparator) + 1e-8
    moran_map = float(np.mean([row["moran_error_median"] for row in rows if row["method"] == map_name]))
    moran_best = min(float(np.mean([row["moran_error_median"] for row in rows if row["method"] == method])) for method in methods if method != map_name)
    moran_guard = moran_map <= 1.05 * moran_best
    axial_improvements = 0
    for gene in AXIAL:
        map_rows = [row for row in axial_rows if row["method"] == map_name and row["gene"] == gene]
        base_rows = [row for row in axial_rows if row["method"] == "registered_idw_k12" and row["gene"] == gene]
        axial_improvements += int(np.mean([x["topk_dice"] for x in map_rows]) > np.mean([x["topk_dice"] for x in base_rows])
                                  and np.mean([x["centroid_error"] for x in map_rows]) < np.mean([x["centroid_error"] for x in base_rows]))
    passed = bool(residual_best and all(endpoint_best.values()) and moran_guard and axial_improvements >= 2 and len(predictable_eval) >= 50)
    decision = {
        "status": "completed", "direction": direction, "gate_passed": passed,
        "residual_pearson_best_or_tied": residual_best, "endpoint_best_or_tied": endpoint_best,
        "moran_within_5pct_best": moran_guard, "axial_genes_improved_dice_and_centroid": axial_improvements,
        "source_defined_predictable_genes": int(len(predictable_eval)),
        "method_means": means, "residual_pearson_means": residual_means,
        "next_step_authorized": "reciprocal frozen development" if passed else "none",
        "runtime_seconds": time.perf_counter() - started,
        "peak_cpu_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    (args.output / "STAGE_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    leakage["status"] = "completed"
    leakage["target_unmeasured_expression_access"] = "evaluation_only_after_prediction_manifest"
    (args.output / "LEAKAGE_AUDIT_COMPLETED.json").write_text(json.dumps(leakage, indent=2) + "\n")


if __name__ == "__main__":
    main()
