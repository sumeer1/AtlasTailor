#!/usr/bin/env python3
"""Stage-1 gate for HyperSpatial-MAP using frozen weMERFISH development embryos.

This is retrospective development evidence only. It reads target geometry and the
locked anchor columns before prediction; all other target genes remain unopened
until every prediction is serialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
import torch

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
ANCHORS = 16
K_LOCAL = 12
K_BROAD = 32
RANK = 16
RIDGE_GRID = (0.1, 1.0, 10.0)
GAIN_GRID = (0.25, 0.5, 0.75, 1.0)
AXIAL = ("tbxta", "noto", "shha")
AXIAL_EXCLUDE = {"tbxta", "tbxtb", "noto", "shha", "foxa", "foxa2", "foxa3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(values) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def target_anchor_counts(path: Path, columns: np.ndarray) -> np.ndarray:
    """Read only the preselected anchor columns from target molecular layers."""
    order = np.argsort(columns)
    sorted_columns = columns[order]
    inverse = np.argsort(order)
    with h5py.File(path, "r") as handle:
        spliced = np.asarray(handle["layers/spliced"][:, sorted_columns], np.float32)
        unspliced = np.asarray(handle["layers/unspliced"][:, sorted_columns], np.float32)
    return (spliced + unspliced)[:, inverse]


def idw(source_coordinates, source_values, query_coordinates, k, chunk=4096):
    tree = cKDTree(source_coordinates)
    output = np.empty((len(query_coordinates), source_values.shape[1]), np.float32)
    for begin in range(0, len(query_coordinates), chunk):
        query = query_coordinates[begin : begin + chunk]
        distance, index = tree.query(query, k=min(k, len(source_coordinates)))
        if np.ndim(distance) == 1:
            distance, index = distance[:, None], index[:, None]
        weight = np.maximum(distance, 1e-5) ** -2
        weight /= np.maximum(weight.sum(1, keepdims=True), 1e-12)
        output[begin : begin + len(query)] = np.einsum("bk,bkg->bg", weight, source_values[index])
    return output


def smooth(values, coordinates, k):
    index = cKDTree(coordinates).query(coordinates, k=min(k + 1, len(coordinates)))[1][:, 1:]
    return values[index].mean(1).astype(np.float32)


def features(anchor_residual, coordinates, multiscale=True):
    blocks = [anchor_residual]
    if multiscale:
        blocks.extend((smooth(anchor_residual, coordinates, K_LOCAL), smooth(anchor_residual, coordinates, K_BROAD)))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def octants(coordinates):
    median = np.median(coordinates, axis=0)
    return ((coordinates[:, 0] > median[0]).astype(int)
            + 2 * (coordinates[:, 1] > median[1]).astype(int)
            + 4 * (coordinates[:, 2] > median[2]).astype(int))


def pseudo_block_prediction(values, coordinates, labels):
    prediction = np.empty_like(values)
    for label in range(8):
        query = np.flatnonzero(labels == label)
        reference = np.flatnonzero(labels != label)
        if len(query) == 0 or len(reference) < K_LOCAL:
            raise RuntimeError(f"Invalid source pseudo-block {label}")
        prediction[query] = idw(coordinates[reference], values[reference], coordinates[query], K_LOCAL)
    return prediction


def fit_ridge_rank(x, y, ridge, rank):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-5)
    standardized = (x - mean) / scale
    weight = np.linalg.solve(
        standardized.T @ standardized + ridge * np.eye(standardized.shape[1]),
        standardized.T @ y,
    )
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    keep = min(rank, len(s))
    weight = (u[:, :keep] * s[:keep]) @ vt[:keep]
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32), "weight": weight.astype(np.float32)}


def apply_linear(model, x):
    return ((x - model["mean"]) / model["scale"]) @ model["weight"]


def select_anchors(source_log, coordinates, genes, count):
    prevalence = np.mean(source_log > 0, axis=0)
    variance = np.var(source_log, axis=0)
    candidates = np.asarray([
        index for index, gene in enumerate(genes)
        if 0.05 <= prevalence[index] <= 0.98
        and variance[index] > 1e-5
        and gene.lower() not in AXIAL_EXCLUDE
        and not gene.lower().startswith(("mt-", "rpl", "rps"))
    ], np.int64)
    rng = np.random.default_rng(42016)
    sample = np.sort(rng.choice(len(source_log), min(5000, len(source_log)), replace=False))
    x = source_log[sample]
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-5)
    correlation = np.abs((x[:, candidates].T @ x) / max(len(x) - 1, 1))
    axial_indices = [np.flatnonzero(genes == gene)[0] for gene in AXIAL if np.any(genes == gene)]
    if axial_indices:
        allowed = np.max(correlation[:, axial_indices], axis=1) < 0.65
        candidates, correlation = candidates[allowed], correlation[allowed]
    coverage = np.zeros(len(genes), np.float32)
    chosen = []
    for _ in range(count):
        gains = np.mean(np.maximum(coverage[None, :], correlation**2) - coverage[None, :], axis=1)
        gains[[np.flatnonzero(candidates == index)[0] for index in chosen if np.any(candidates == index)]] = -np.inf
        local = int(np.argmax(gains))
        chosen.append(int(candidates[local]))
        coverage = np.maximum(coverage, correlation[local] ** 2)
    return np.asarray(chosen, np.int64), {"mean_squared_gene_coverage": float(coverage.mean()), "candidate_count": int(len(candidates))}


def depth_predictor(source_counts, anchor_indices):
    x = np.c_[np.ones(len(source_counts)), np.log1p(source_counts[:, anchor_indices].sum(1))]
    y = np.log1p(source_counts.sum(1))
    weight = np.linalg.solve(x.T @ x + np.diag([0.0, 1.0]), x.T @ y)
    return weight.astype(np.float32)


def predicted_depth(anchor_counts, weight):
    x = np.c_[np.ones(len(anchor_counts)), np.log1p(anchor_counts.sum(1))]
    return np.maximum(np.expm1(x @ weight), 1.0).astype(np.float32)


def residual_pearson(truth_log, base_log, prediction_log, indices):
    truth_residual = truth_log - base_log
    predicted_residual = prediction_log - base_log
    return pearson_columns(truth_residual[:, indices], predicted_residual[:, indices])


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
    if direction != lock["stage_1_direction"] and not (lock.get("stage_1_passed") and direction == lock["stage_2_direction"]):
        raise RuntimeError(f"Direction {direction} is not currently authorized by the gate")
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
    anchor_indices, anchor_selection = select_anchors(source_log, source_coordinates, genes, ANCHORS)
    anchor_genes = genes[anchor_indices]
    target_anchors_raw = target_anchor_counts(target_path, anchor_indices)
    depth_model = depth_predictor(source_counts, anchor_indices)
    target_depth = predicted_depth(target_anchors_raw, depth_model)
    target_anchor_values = target_anchors_raw * (depth_target / target_depth)[:, None]
    target_anchor_log = np.log1p(target_anchor_values).astype(np.float32)

    labels = octants(source_coordinates)
    pseudo_base = pseudo_block_prediction(source_values, source_coordinates, labels)
    pseudo_base_log = np.log1p(pseudo_base).astype(np.float32)
    source_residual = source_log - pseudo_base_log
    raw_source_features = source_residual[:, anchor_indices]
    local_source_features = features(raw_source_features, source_coordinates, True)
    validation = np.isin(labels, [0, 7])
    train = ~validation
    selection_records = []
    best = None
    for ridge in RIDGE_GRID:
        global_model = fit_ridge_rank(raw_source_features[train], source_residual[train], ridge, RANK)
        local_model = fit_ridge_rank(local_source_features[train], source_residual[train], ridge, RANK)
        for gain in GAIN_GRID:
            for name, model, x in (("global", global_model, raw_source_features), ("local_multiscale", local_model, local_source_features)):
                residual = gain * apply_linear(model, x[validation])
                predicted = pseudo_base_log[validation] + residual
                correlation = pearson_columns(source_residual[validation], residual)
                score = float(np.nanmedian(correlation))
                rmse = float(np.sqrt(np.mean((source_log[validation] - predicted) ** 2)))
                record = {"model": name, "ridge": ridge, "gain": gain, "residual_pearson": score, "log1p_rmse": rmse}
                selection_records.append(record)
                key = (score, -rmse, name == "local_multiscale")
                if best is None or key > best[0]:
                    best = (key, record)
    chosen = best[1]
    chosen_features = local_source_features if chosen["model"] == "local_multiscale" else raw_source_features
    residual_model = fit_ridge_rank(chosen_features, source_residual, chosen["ridge"], RANK)
    global_model = fit_ridge_rank(raw_source_features, source_residual, chosen["ridge"], RANK)
    anchor_absolute_model = fit_ridge_rank(source_log[:, anchor_indices], source_log, chosen["ridge"], RANK)
    cv_model = fit_ridge_rank(chosen_features[train], source_residual[train], chosen["ridge"], RANK)
    cv_residual = chosen["gain"] * apply_linear(cv_model, chosen_features[validation])
    cv_correlation = pearson_columns(source_residual[validation], cv_residual)
    cv_base_error = np.mean((source_log[validation] - pseudo_base_log[validation]) ** 2, axis=0)
    cv_map_error = np.mean((source_log[validation] - (pseudo_base_log[validation] + cv_residual)) ** 2, axis=0)
    predictable = (cv_correlation > 0.1) & (cv_map_error < cv_base_error)
    predictable[anchor_indices] = False
    source_svg_order, source_svg_scores = source_svg_indices([source_log], [source_coordinates], count=min(200, len(genes)))
    evaluation_svg = np.asarray([index for index in source_svg_order if index not in set(anchor_indices)][:100], np.int64)
    thresholds = np.quantile(source_values, 0.8, axis=0).astype(np.float32)

    manifest = {
        "status": "target_unmeasured_genes_unopened",
        "direction": direction,
        "source": source_path.name,
        "target": target_path.name,
        "target_allowed_before_prediction": ["geometry", "IDs", "gene names", "16 locked anchor columns"],
        "anchor_genes": anchor_genes.tolist(),
        "anchor_indices": anchor_indices.tolist(),
        "anchor_selection": anchor_selection,
        "excluded_axial_genes": sorted(AXIAL_EXCLUDE),
        "target_coordinate_key": coordinate_key,
        "target_ids": target_ids.tolist(),
        "source_depth_target": depth_target,
        "source_depth_predictor": depth_model.tolist(),
        "selected_residual_model": chosen,
        "source_only_selection_records": selection_records,
        "source_defined_predictable_indices": np.flatnonzero(predictable).tolist(),
        "source_defined_predictable_genes": genes[predictable].tolist(),
        "evaluation_svg_indices": evaluation_svg.tolist(),
        "evaluation_svg_scores": source_svg_scores[:len(evaluation_svg)].tolist(),
        "methods": {},
    }
    for seed in SEEDS:
        torch.manual_seed(seed)
        mapped, rigid, geometry = registered_coordinates(source_coordinates, target_coordinates, 30, seed, device)
        base = idw(mapped, source_values, target_coordinates, K_LOCAL)
        nearest = idw(mapped, source_values, target_coordinates, 1)
        base_log = np.log1p(base).astype(np.float32)
        nearest_log = np.log1p(nearest).astype(np.float32)
        anchor_residual = target_anchor_log - base_log[:, anchor_indices]
        raw_target_features = anchor_residual
        local_target_features = features(anchor_residual, target_coordinates, True)
        map_features = local_target_features if chosen["model"] == "local_multiscale" else raw_target_features
        map_residual = chosen["gain"] * apply_linear(residual_model, map_features)
        map_residual[:, ~predictable] = 0
        map_log = np.maximum(base_log + map_residual, 0).astype(np.float32)
        global_residual = chosen["gain"] * apply_linear(global_model, raw_target_features)
        global_residual[:, ~predictable] = 0
        global_log = np.maximum(base_log + global_residual, 0).astype(np.float32)
        anchor_only_log = np.maximum(apply_linear(anchor_absolute_model, target_anchor_log), 0).astype(np.float32)
        predictions = {
            "registered_idw_k12": base_log,
            "registered_nearest_neighbor": nearest_log,
            "anchor_only_low_rank": anchor_only_log,
            "atlas_plus_global_anchor_residual": global_log,
            "hyperspatial_map": map_log,
        }
        manifest["methods"][str(seed)] = {"geometry": geometry, "predictions": {}}
        for method, prediction in predictions.items():
            path = prediction_dir / f"seed{seed}_{method}.npy"
            np.save(path, prediction)
            manifest["methods"][str(seed)]["predictions"][method] = {"path": str(path), "sha256": sha256(path)}
    manifest["status"] = "all_predictions_serialized_evaluation_pending"
    (args.output / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    leakage = {
        "status": manifest["status"],
        "target_unmeasured_expression_accessed_before_prediction": False,
        "target_anchor_columns_accessed": anchor_genes.tolist(),
        "target_annotations_accessed": False,
        "anchor_selection_source_only": True,
        "model_and_predictable_subspace_source_only": True,
        "qualification": "retrospective E1/E2 development; not independent confirmation; geometry is global_sphere rather than an independently acquired morphology image",
    }
    (args.output / "LEAKAGE_AUDIT.json").write_text(json.dumps(leakage, indent=2) + "\n")

    # Evaluation-only access to the full target panel starts here.
    target_counts, target_coordinates_check, target_genes_check = measured_counts(target_path)
    if not np.array_equal(target_genes_check, genes):
        raise AssertionError("Target genes changed")
    truth_values = normalize_counts(target_counts, depth_target).astype(np.float32)
    truth_log = np.log1p(truth_values).astype(np.float32)
    unmeasured_svg = np.asarray([index for index in evaluation_svg if index not in set(anchor_indices)], np.int64)
    predictable_eval = np.asarray([index for index in np.flatnonzero(predictable) if index not in set(anchor_indices)], np.int64)
    rows, residual_rows, axial_rows = [], [], []
    for seed in SEEDS:
        paths = manifest["methods"][str(seed)]["predictions"]
        base_log = np.asarray(np.load(paths["registered_idw_k12"]["path"], mmap_mode="r"), np.float32)
        for method_index, (method, metadata) in enumerate(paths.items()):
            prediction = np.asarray(np.load(metadata["path"], mmap_mode="r"), np.float32)
            result = evaluate(method, truth_log, prediction, target_coordinates, unmeasured_svg, thresholds, seed * 100 + method_index)
            result.pop("per_gene_pearson")
            ci = result.pop("svg_spatial_pearson_bootstrap_ci95")
            rows.append({"direction": direction, "seed": seed, **result, "pearson_ci_low": ci[0], "pearson_ci_high": ci[1]})
            if len(predictable_eval):
                correlation = residual_pearson(truth_log, base_log, prediction, predictable_eval)
                rng = np.random.default_rng(seed * 1000 + method_index)
                bootstrap = [np.nanmedian(rng.choice(correlation, len(correlation), replace=True)) for _ in range(2000)]
                residual_rows.append({
                    "direction": direction, "seed": seed, "method": method,
                    "predictable_genes": len(predictable_eval),
                    "residual_spatial_pearson_median": float(np.nanmedian(correlation)),
                    "ci95_low": float(np.nanquantile(bootstrap, 0.025)),
                    "ci95_high": float(np.nanquantile(bootstrap, 0.975)),
                })
            for gene in AXIAL:
                if not np.any(genes == gene):
                    continue
                index = int(np.flatnonzero(genes == gene)[0])
                gene_result = evaluate(method, truth_log, prediction, target_coordinates, np.asarray([index]), thresholds, seed * 10000 + method_index)
                axial_rows.append({
                    "direction": direction, "seed": seed, "method": method, "gene": gene,
                    "spatial_pearson": gene_result["svg_spatial_pearson_median"],
                    "topk_dice": gene_result["topk_dice_median"],
                    "centroid_error": gene_result["weighted_centroid_error_median"],
                })

    def write_tsv(path, records):
        columns = list(records[0])
        with path.open("x") as handle:
            handle.write("\t".join(columns) + "\n")
            for record in records:
                handle.write("\t".join(str(record[column]) for column in columns) + "\n")
    write_tsv(args.output / "all_metrics.tsv", rows)
    write_tsv(args.output / "residual_metrics.tsv", residual_rows)
    write_tsv(args.output / "axial_metrics.tsv", axial_rows)

    methods = sorted(set(row["method"] for row in rows))
    endpoints = (
        ("svg_spatial_pearson_median", True), ("auprc_enrichment_median", True),
        ("topk_dice_median", True), ("weighted_centroid_error_median", False),
        ("spatial_emd_median", False), ("log1p_rmse", False),
    )
    means = {method: {endpoint: float(np.mean([row[endpoint] for row in rows if row["method"] == method])) for endpoint, _ in endpoints} for method in methods}
    residual_means = {method: float(np.mean([row["residual_spatial_pearson_median"] for row in residual_rows if row["method"] == method])) for method in methods}
    map_name = "hyperspatial_map"
    residual_best = residual_means[map_name] > max(value for method, value in residual_means.items() if method != map_name)
    endpoint_best = {}
    for endpoint, higher in endpoints:
        comparator = [means[method][endpoint] for method in methods if method != map_name]
        endpoint_best[endpoint] = means[map_name][endpoint] > max(comparator) if higher else means[map_name][endpoint] < min(comparator)
    moran_map = float(np.mean([row["moran_error_median"] for row in rows if row["method"] == map_name]))
    moran_best = min(float(np.mean([row["moran_error_median"] for row in rows if row["method"] == method])) for method in methods if method != map_name)
    moran_guard = moran_map <= 1.05 * moran_best
    axial_improvements = 0
    for gene in AXIAL:
        map_dice = np.mean([row["topk_dice"] for row in axial_rows if row["method"] == map_name and row["gene"] == gene])
        base_dice = np.mean([row["topk_dice"] for row in axial_rows if row["method"] == "registered_idw_k12" and row["gene"] == gene])
        map_centroid = np.mean([row["centroid_error"] for row in axial_rows if row["method"] == map_name and row["gene"] == gene])
        base_centroid = np.mean([row["centroid_error"] for row in axial_rows if row["method"] == "registered_idw_k12" and row["gene"] == gene])
        axial_improvements += int(map_dice > base_dice and map_centroid < base_centroid)
    gate = bool(residual_best and all(endpoint_best.values()) and moran_guard and axial_improvements >= 2 and len(predictable_eval) >= 50)
    decision = {
        "status": "completed", "direction": direction, "stage_1_gate_passed": gate,
        "residual_pearson_best": residual_best, "endpoint_best": endpoint_best,
        "moran_within_5pct_best": moran_guard, "axial_genes_improved_dice_and_centroid": axial_improvements,
        "source_defined_predictable_genes": int(len(predictable_eval)), "method_means": means,
        "residual_pearson_means": residual_means, "next_step_authorized": "reciprocal E2_to_E1" if gate and direction == lock["stage_1_direction"] else "none",
        "runtime_seconds": time.perf_counter() - started,
        "peak_cpu_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    (args.output / "STAGE_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    leakage["status"] = "completed"
    leakage["target_unmeasured_expression_access"] = "evaluation_only_after_prediction_manifest"
    (args.output / "LEAKAGE_AUDIT_COMPLETED.json").write_text(json.dumps(leakage, indent=2) + "\n")


if __name__ == "__main__":
    main()
