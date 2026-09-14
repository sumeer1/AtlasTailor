#!/usr/bin/env python3
"""Leakage-free mouse complete-stage transport feasibility gate.

Sources: E10.5 and E12.5.  Target: complete E11.5 serial-section volume.
Target expression, depth, labels and annotations are opened only after every
prediction file is complete.  This is a serial-section 3D pilot, not true xyz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree, distance
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
import torch

from hyperspatial_molecular_hologram import RegistrationConfig, fit_geometry_field
from run_molecular_hologram_pilot import similarity_icp, strings


SOURCES = ("E10.5", "E12.5")
TARGET = "E11.5"
METHODS = ("raw_temporal_idw", "hyperspatial_registered_transport", "anchor_gp_kriging")


def categorical(h, name):
    return strings(h[f"obs/{name}/categories"][:]), np.asarray(h[f"obs/{name}/codes"][:])


def stage_rows(h, stage):
    categories, codes = categorical(h, "time_point")
    return np.flatnonzero(codes == int(np.flatnonzero(categories == stage)[0]))


def read_csr_rows(h, rows):
    group = h["layers/counts"] if "counts" in h["layers"] else h["layers/count"]
    indptr = group["indptr"][:]
    starts, stops = indptr[rows], indptr[rows + 1]
    lengths = stops - starts
    output_indptr = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    data = np.concatenate([group["data"][int(a) : int(b)] for a, b in zip(starts, stops)])
    indices = np.concatenate([group["indices"][int(a) : int(b)] for a, b in zip(starts, stops)])
    return sp.csr_matrix(
        (data, indices, output_indptr), shape=(len(rows), int(group.attrs["shape"][1]))
    )


def read_contiguous_csr(h, first, stop):
    group = h["layers/counts"] if "counts" in h["layers"] else h["layers/count"]
    global_indptr = group["indptr"][first : stop + 1]
    begin, end = int(global_indptr[0]), int(global_indptr[-1])
    indptr = global_indptr - begin
    return sp.csr_matrix(
        (group["data"][begin:end], group["indices"][begin:end], indptr),
        shape=(stop - first, int(group.attrs["shape"][1])),
    )


def stratified_sample(rows, slices, limit, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    unique = np.sort(np.unique(slices[rows]))
    allocation = max(1, limit // len(unique))
    for slice_id in unique:
        candidates = rows[slices[rows] == slice_id]
        chosen.append(rng.choice(candidates, min(allocation, len(candidates)), replace=False))
    return np.sort(np.concatenate(chosen))


def serial_coordinates(xy, slices, rows):
    values = np.asarray(xy[rows, :2], dtype=np.float32)
    lo, hi = np.quantile(values, 0.01, axis=0), np.quantile(values, 0.99, axis=0)
    normalized = (values - lo) / np.maximum(hi - lo, 1e-6)
    z = slices[rows].astype(np.float32)
    z = (z - z.min()) / max(float(z.max() - z.min()), 1.0)
    return np.c_[normalized, z].astype(np.float32), {"xy_q01": lo.tolist(), "xy_q99": hi.tolist()}


def select_genes(source_counts, source_depth, n_genes):
    normalized = source_counts.multiply(
        (np.median(source_depth[source_depth > 0]) / np.maximum(source_depth, 1.0))[:, None]
    ).tocsr()
    mean = np.asarray(normalized.mean(0)).ravel()
    variance = np.asarray(normalized.power(2).mean(0)).ravel() - mean**2
    prevalence = np.asarray((source_counts > 0).mean(0)).ravel()
    eligible = (prevalence >= 0.01) & (prevalence <= 0.98) & (mean > 0)
    score = np.where(eligible, variance / (mean + 1e-3), -np.inf)
    selected = np.sort(np.argsort(score)[-n_genes:])
    return selected, {
        "method": "source_only_normalized_dispersion",
        "eligible": int(eligible.sum()),
        "selected": int(len(selected)),
    }


def normalized_values(counts, full_depth, target_depth):
    return counts.multiply((target_depth / np.maximum(full_depth, 1.0))[:, None]).toarray().astype(np.float32)


def source_svg_indices(values_by_stage, coordinates_by_stage, count=200):
    scores = []
    for values, coordinates in zip(values_by_stage, coordinates_by_stage):
        take = min(3500, len(values))
        subset = np.linspace(0, len(values) - 1, take, dtype=np.int64)
        x = np.log1p(values[subset])
        neighbors = cKDTree(coordinates[subset]).query(coordinates[subset], k=min(13, take))[1][:, 1:]
        centered = x - x.mean(0, keepdims=True)
        score = (centered * centered[neighbors].mean(1)).sum(0) / np.maximum(
            np.square(centered).sum(0), 1e-8
        )
        scores.append(score)
    combined = np.nanmean(np.stack(scores), axis=0)
    return np.sort(np.argsort(combined)[-min(count, len(combined)) :]), combined


def idw_numpy(train_coordinates, train_values, query_coordinates, k):
    distances, indices = cKDTree(train_coordinates).query(query_coordinates, k=min(k, len(train_coordinates)))
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-5) ** 2
    weights /= weights.sum(1, keepdims=True)
    return np.einsum("qk,qkg->qg", weights, train_values[indices]).astype(np.float32)


def tune_k_source_only(values_by_stage, coordinates_by_stage):
    records = {}
    for k in (8, 16, 32):
        losses = []
        for values, coordinates in zip(values_by_stage, coordinates_by_stage):
            holdout = (coordinates[:, 0] >= 0.45) & (coordinates[:, 0] < 0.55)
            prediction = idw_numpy(coordinates[~holdout], values[~holdout], coordinates[holdout], k)
            losses.append(float(np.mean((np.log1p(values[holdout]) - np.log1p(prediction)) ** 2)))
        records[str(k)] = float(np.mean(losses))
    chosen = min((8, 16, 32), key=lambda k: records[str(k)])
    return chosen, records


def chamfer(a, b):
    return float(0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()))


def best_rigid(source, target):
    candidates = {}
    for name, signs in {"none": (1, 1), "reflect_x": (-1, 1), "reflect_y": (1, -1), "reflect_xy": (-1, -1)}.items():
        reflected = source.copy()
        reflected[:, :2] = 0.5 + (reflected[:, :2] - 0.5) * np.asarray(signs)
        mapped = similarity_icp(reflected, target, iterations=30)
        candidates[name] = (chamfer(mapped, target), mapped)
    name = min(candidates, key=lambda key: candidates[key][0])
    return candidates[name][1], name, {key: float(value[0]) for key, value in candidates.items()}


def register_source(source, target_anchor, epochs, seed, device):
    rigid, reflection, candidates = best_rigid(source, target_anchor)
    start = time.perf_counter()
    field, history = fit_geometry_field(
        rigid,
        target_anchor,
        device,
        RegistrationConfig(epochs=epochs, batch_size=min(768, len(source)), seed=seed),
    )
    with torch.no_grad():
        mapped = field(torch.as_tensor(rigid, dtype=torch.float32, device=device)).cpu().numpy()
    return mapped, field, {
        "reflection": reflection,
        "reflection_candidate_chamfers": candidates,
        "raw_chamfer": chamfer(source, target_anchor),
        "rigid_chamfer": chamfer(rigid, target_anchor),
        "registered_chamfer": chamfer(mapped, target_anchor),
        "runtime_seconds": time.perf_counter() - start,
        "final_loss": history["loss"][-1],
    }


def predict_transport(path, query_coordinates, source_coordinates, source_values, k, device, chunk=2048):
    output = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(query_coordinates), source_values[0].shape[1]))
    trees = [cKDTree(x) for x in source_coordinates]
    values_gpu = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in source_values]
    for begin in range(0, len(query_coordinates), chunk):
        end = min(begin + chunk, len(query_coordinates))
        stage_predictions = []
        for tree, values in zip(trees, values_gpu):
            distances, indices = tree.query(query_coordinates[begin:end], k=min(k, len(tree.data)))
            weights = 1.0 / np.maximum(distances, 1e-5) ** 2
            weights /= weights.sum(1, keepdims=True)
            idx = torch.as_tensor(indices, dtype=torch.long, device=device)
            weight = torch.as_tensor(weights, dtype=torch.float32, device=device)
            stage_predictions.append((values[idx] * weight[:, :, None]).sum(1))
        prediction = torch.log1p(0.5 * stage_predictions[0] + 0.5 * stage_predictions[1])
        output[begin:end] = prediction.cpu().numpy()
    output.flush()


def fit_anchor_gp(source_coordinates, source_log_values, seed, anchors=384, latent=32):
    combined_coordinates = np.vstack(source_coordinates).astype(np.float32)
    combined_values = np.vstack(source_log_values).astype(np.float32)
    svd = TruncatedSVD(n_components=latent, random_state=seed)
    latent_values = svd.fit_transform(sp.csr_matrix(combined_values))
    rng = np.random.default_rng(seed)
    anchor_indices = np.sort(rng.choice(len(combined_coordinates), min(anchors, len(combined_coordinates)), replace=False))
    anchor_coordinates = combined_coordinates[anchor_indices]
    anchor_latent = latent_values[anchor_indices]
    pairwise = distance.pdist(anchor_coordinates)
    lengthscale = max(float(np.median(pairwise[pairwise > 0])), 1e-3)
    kernel = np.exp(-0.5 * distance.cdist(anchor_coordinates, anchor_coordinates, "sqeuclidean") / lengthscale**2)
    kernel.flat[:: len(kernel) + 1] += 0.05
    coefficients = cho_solve(cho_factor(kernel, check_finite=False), anchor_latent, check_finite=False)
    return anchor_coordinates, coefficients.astype(np.float32), svd.components_.astype(np.float32), lengthscale


def predict_anchor_gp(path, query_4d, fit, device, chunk=4096):
    anchor_coordinates, coefficients, components, lengthscale = fit
    output = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(query_4d), components.shape[1]))
    anchor = torch.as_tensor(anchor_coordinates, dtype=torch.float32, device=device)
    coefficients = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
    components = torch.as_tensor(components, dtype=torch.float32, device=device)
    for begin in range(0, len(query_4d), chunk):
        end = min(begin + chunk, len(query_4d))
        query = torch.as_tensor(query_4d[begin:end], dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * torch.cdist(query, anchor).square() / lengthscale**2)
        prediction = torch.clamp((kernel @ coefficients) @ components, min=0)
        output[begin:end] = prediction.cpu().numpy()
    output.flush()


def normalized_emd(true, predicted, coordinates, seed):
    true = np.clip(true, 0, None)
    predicted = np.clip(predicted, 0, None)
    if true.sum() <= 0 or predicted.sum() <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = min(192, len(coordinates))
    a = rng.choice(len(coordinates), n, replace=True, p=true / true.sum())
    b = rng.choice(len(coordinates), n, replace=True, p=predicted / predicted.sum())
    from scipy.optimize import linear_sum_assignment

    costs = distance.cdist(coordinates[a], coordinates[b])
    row, col = linear_sum_assignment(costs)
    diameter = max(float(np.linalg.norm(coordinates.max(0) - coordinates.min(0))), 1e-8)
    return float(costs[row, col].mean() / diameter)


def evaluate_predictions(
    h,
    target_rows,
    selected,
    genes,
    svg,
    marker_local,
    thresholds,
    target_coordinates,
    prediction_paths,
    target_depth,
    output,
    chunk=1024,
):
    n, g = len(target_rows), len(selected)
    if not np.all(np.diff(target_rows) == 1):
        raise AssertionError("Fast leakage-safe evaluator requires contiguous target-stage rows.")
    truth_svg = np.lib.format.open_memmap(output / "observed_svg_log.npy", mode="w+", dtype="float32", shape=(n, len(svg)))
    sums = {
        method: {key: np.zeros(g, np.float64) for key in ("y", "y2", "xy")}
        for method in prediction_paths
    }
    x_sum = np.zeros(g, np.float64)
    x2_sum = np.zeros(g, np.float64)
    squared_error = {method: 0.0 for method in prediction_paths}
    prediction_maps = {method: np.load(path, mmap_mode="r") for method, path in prediction_paths.items()}
    truth_markers = np.empty((n, len(marker_local)), np.float32)
    predicted_markers = {method: np.empty((n, len(marker_local)), np.float32) for method in prediction_paths}
    cursor = 0
    for first in range(int(target_rows[0]), int(target_rows[-1]) + 1, chunk):
        stop = min(first + chunk, int(target_rows[-1]) + 1)
        counts = read_contiguous_csr(h, first, stop)
        depths = np.asarray(counts.sum(1)).ravel()
        true_values = counts[:, selected].multiply((target_depth / np.maximum(depths, 1.0))[:, None]).toarray().astype(np.float32)
        true_log = np.log1p(true_values)
        size = stop - first
        truth_svg[cursor : cursor + size] = true_log[:, svg]
        truth_markers[cursor : cursor + size] = true_values[:, marker_local]
        x_sum += true_log.sum(0)
        x2_sum += np.square(true_log).sum(0)
        for method, predictions in prediction_maps.items():
            predicted = np.asarray(predictions[cursor : cursor + size])
            sums[method]["y"] += predicted.sum(0)
            sums[method]["y2"] += np.square(predicted).sum(0)
            sums[method]["xy"] += (true_log * predicted).sum(0)
            squared_error[method] += float(np.square(true_log - predicted).sum())
            predicted_markers[method][cursor : cursor + size] = np.expm1(predicted[:, marker_local])
        cursor += size
    truth_svg.flush()
    results = {}
    x_var = x2_sum - x_sum * x_sum / n
    for method, values in sums.items():
        y_var = values["y2"] - values["y"] * values["y"] / n
        covariance = values["xy"] - x_sum * values["y"] / n
        correlations = covariance / np.sqrt(np.maximum(x_var * y_var, 1e-16))
        spatial_pearson = correlations[svg]
        rng = np.random.default_rng(42)
        bootstrap = np.asarray(
            [
                np.nanmedian(rng.choice(spatial_pearson, len(spatial_pearson), replace=True))
                for _ in range(1000)
            ]
        )
        spearman = []
        prediction = prediction_maps[method]
        for local in svg:
            spearman.append(spearmanr(truth_svg[:, np.flatnonzero(svg == local)[0]], prediction[:, local]).statistic)
        marker_metrics = []
        for j, local in enumerate(marker_local):
            true = truth_markers[:, j]
            predicted = predicted_markers[method][:, j]
            positive = true >= thresholds[j]
            prevalence = float(positive.mean())
            k = int(positive.sum())
            top = np.zeros(n, bool)
            if k:
                top[np.argpartition(predicted, -k)[-k:]] = True
            true_weight, pred_weight = np.clip(true, 0, None), np.clip(predicted, 0, None)
            diameter = max(float(np.linalg.norm(target_coordinates.max(0) - target_coordinates.min(0))), 1e-8)
            if true_weight.sum() > 0 and pred_weight.sum() > 0:
                true_centroid = (target_coordinates * true_weight[:, None]).sum(0) / true_weight.sum()
                pred_centroid = (target_coordinates * pred_weight[:, None]).sum(0) / pred_weight.sum()
                centroid = float(np.linalg.norm(true_centroid - pred_centroid) / diameter)
            else:
                centroid = float("nan")
            marker_metrics.append(
                {
                    "gene": str(genes[local]),
                    "prevalence": prevalence,
                    "auprc_enrichment": float(average_precision_score(positive, predicted) / prevalence)
                    if prevalence > 0 and positive.any()
                    else float("nan"),
                    "topk_dice": float(2 * np.logical_and(positive, top).sum() / max(positive.sum() + top.sum(), 1)),
                    "centroid_error": centroid,
                    "spatial_emd": normalized_emd(true, predicted, target_coordinates, 42 + j),
                }
            )
        mean = lambda key: float(np.nanmean([item[key] for item in marker_metrics]))
        results[method] = {
            "svg_spatial_pearson_median": float(np.nanmedian(spatial_pearson)),
            "svg_spatial_pearson_bootstrap_ci95": [
                float(np.nanquantile(bootstrap, 0.025)),
                float(np.nanquantile(bootstrap, 0.975)),
            ],
            "svg_spatial_spearman_median": float(np.nanmedian(spearman)),
            "log1p_normalized_rmse": float(np.sqrt(squared_error[method] / (n * g))),
            "auprc_enrichment_mean": mean("auprc_enrichment"),
            "topk_dice_mean": mean("topk_dice"),
            "centroid_error_mean": mean("centroid_error"),
            "spatial_emd_mean": mean("spatial_emd"),
            "marker_metrics": marker_metrics,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_v10/mouse_3d_4d_validation/temporal_transport_gate_E11.5"),
    )
    parser.add_argument("--source-spots-per-stage", type=int, default=5000)
    parser.add_argument("--target-geometry-anchors", type=int, default=3000)
    parser.add_argument("--genes", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions").mkdir(exist_ok=True)
    (args.output / "models").mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; no CPU fallback launched.")
    device = "cuda"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    with h5py.File(args.cache, "r") as h:
        slices = np.asarray(h["obs/slice"][:])
        xy = np.asarray(h["obsm/spatial"][:], dtype=np.float32)
        gene_key = "var/_index" if "var/_index" in h else "var/gene_short_name"
        all_genes = strings(h[gene_key][:])
        rows_by_stage = {stage: stage_rows(h, stage) for stage in (*SOURCES, TARGET)}
        source_rows = {
            stage: stratified_sample(rows_by_stage[stage], slices, args.source_spots_per_stage, args.seed + i)
            for i, stage in enumerate(SOURCES)
        }
        target_anchor_rows = stratified_sample(
            rows_by_stage[TARGET], slices, args.target_geometry_anchors, args.seed + 10
        )
        source_coordinates, coordinate_meta = {}, {}
        for stage in SOURCES:
            full_stage_coordinates, coordinate_meta[stage] = serial_coordinates(
                xy, slices, rows_by_stage[stage]
            )
            source_coordinates[stage] = full_stage_coordinates[
                np.searchsorted(rows_by_stage[stage], source_rows[stage])
            ]
        target_coordinates, coordinate_meta[TARGET] = serial_coordinates(xy, slices, rows_by_stage[TARGET])
        target_anchor_coordinates = target_coordinates[
            np.searchsorted(rows_by_stage[TARGET], target_anchor_rows)
        ]
        source_counts = {stage: read_csr_rows(h, source_rows[stage]) for stage in SOURCES}

    assert not np.intersect1d(np.concatenate(list(source_rows.values())), rows_by_stage[TARGET]).size
    combined_counts = sp.vstack([source_counts[stage] for stage in SOURCES], format="csr")
    source_depth = {
        stage: np.asarray(source_counts[stage].sum(1)).ravel().astype(np.float32) for stage in SOURCES
    }
    combined_depth = np.concatenate([source_depth[stage] for stage in SOURCES])
    selected, selection_meta = select_genes(combined_counts, combined_depth, args.genes)
    genes = all_genes[selected]
    depth_target = max(float(np.median(combined_depth[combined_depth > 0])), 1.0)
    source_values = {
        stage: normalized_values(source_counts[stage][:, selected], source_depth[stage], depth_target)
        for stage in SOURCES
    }
    svg, svg_scores = source_svg_indices(
        [source_values[stage] for stage in SOURCES], [source_coordinates[stage] for stage in SOURCES]
    )
    marker_local = svg[np.argsort(svg_scores[svg])[-min(12, len(svg)) :]]
    thresholds = []
    combined_values = np.vstack([source_values[stage] for stage in SOURCES])
    for local in marker_local:
        positive = combined_values[:, local][combined_values[:, local] > 0]
        thresholds.append(float(np.quantile(positive, 0.5)) if len(positive) else float("inf"))
    k, k_tuning = tune_k_source_only(
        [source_values[stage] for stage in SOURCES], [source_coordinates[stage] for stage in SOURCES]
    )

    manifest = {
        "status": "prediction_in_progress",
        "source_stages": list(SOURCES),
        "target_stage": TARGET,
        "evaluation_geometry": "serial-section 3D (x/y plus normalized slice order), not true xyz",
        "source_sample_rows": {stage: source_rows[stage].tolist() for stage in SOURCES},
        "target_rows": {
            "first": int(rows_by_stage[TARGET][0]),
            "last": int(rows_by_stage[TARGET][-1]),
            "count": int(len(rows_by_stage[TARGET])),
            "sha256": hashlib.sha256(rows_by_stage[TARGET].tobytes()).hexdigest(),
        },
        "all_target_slices_included": [float(x) for x in np.unique(slices[rows_by_stage[TARGET]])],
        "target_expression_accessed_before_predictions": False,
        "target_labels_or_annotations_accessed": False,
        "target_library_size_used_for_fitting": False,
        "target_coordinates_used_as_query_geometry": True,
        "gene_selection": selection_meta,
        "selected_genes": genes.tolist(),
        "source_selected_svg_genes": genes[svg].tolist(),
        "source_selected_marker_genes": genes[marker_local].tolist(),
        "source_only_k_tuning": {"chosen": k, "losses": k_tuning},
        "source_depth_target": depth_target,
        "coordinate_normalization": coordinate_meta,
        "seed": args.seed,
    }
    (args.output / "leakage_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(
        args.output / "split_manifest.npz",
        E10_5_rows=source_rows["E10.5"],
        E12_5_rows=source_rows["E12.5"],
        E11_5_rows=rows_by_stage[TARGET],
        selected_gene_indices=selected,
        svg_local_indices=svg,
        marker_local_indices=marker_local,
    )

    registered, geometry_meta = {}, {}
    fields = {}
    for i, stage in enumerate(SOURCES):
        registered[stage], fields[stage], geometry_meta[stage] = register_source(
            source_coordinates[stage], target_anchor_coordinates, args.epochs, args.seed + i, device
        )
        torch.save(fields[stage].state_dict(), args.output / "models" / f"geometry_{stage}.pt")

    prediction_paths = {method: args.output / "predictions" / f"{method}.npy" for method in METHODS}
    start = time.perf_counter()
    predict_transport(
        prediction_paths["raw_temporal_idw"],
        target_coordinates,
        [source_coordinates[stage] for stage in SOURCES],
        [source_values[stage] for stage in SOURCES],
        k,
        device,
    )
    raw_runtime = time.perf_counter() - start
    start = time.perf_counter()
    predict_transport(
        prediction_paths["hyperspatial_registered_transport"],
        target_coordinates,
        [registered[stage] for stage in SOURCES],
        [source_values[stage] for stage in SOURCES],
        k,
        device,
    )
    hyper_runtime = time.perf_counter() - start
    source_4d = [
        np.c_[source_coordinates["E10.5"], np.zeros(len(source_coordinates["E10.5"]), np.float32)],
        np.c_[source_coordinates["E12.5"], np.ones(len(source_coordinates["E12.5"]), np.float32)],
    ]
    gp_start = time.perf_counter()
    gp_fit = fit_anchor_gp(
        source_4d,
        [np.log1p(source_values["E10.5"]), np.log1p(source_values["E12.5"])],
        args.seed,
    )
    predict_anchor_gp(
        prediction_paths["anchor_gp_kriging"],
        np.c_[target_coordinates, np.full(len(target_coordinates), 0.5, np.float32)],
        gp_fit,
        device,
    )
    gp_runtime = time.perf_counter() - gp_start

    for path in prediction_paths.values():
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Prediction serialization failed: {path}")
    manifest["status"] = "predictions_complete_evaluation_pending"
    manifest["target_expression_accessed_before_predictions"] = False
    manifest["prediction_files"] = {method: str(path) for method, path in prediction_paths.items()}
    (args.output / "leakage_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with h5py.File(args.cache, "r") as h:
        results = evaluate_predictions(
            h,
            rows_by_stage[TARGET],
            selected,
            genes,
            svg,
            marker_local,
            np.asarray(thresholds),
            target_coordinates,
            prediction_paths,
            depth_target,
            args.output,
        )
    runtimes = {
        "raw_temporal_idw": raw_runtime,
        "hyperspatial_registered_transport": hyper_runtime + sum(x["runtime_seconds"] for x in geometry_meta.values()),
        "anchor_gp_kriging": gp_runtime,
    }
    for method in results:
        results[method]["runtime_seconds"] = runtimes[method]

    hyper = results["hyperspatial_registered_transport"]
    gp = results["anchor_gp_kriging"]
    localization_keys = (
        ("auprc_enrichment_mean", True),
        ("topk_dice_mean", True),
        ("centroid_error_mean", False),
        ("spatial_emd_mean", False),
    )
    localization_wins = sum(
        hyper[key] >= gp[key] if higher else hyper[key] <= gp[key] for key, higher in localization_keys
    )
    gate = {
        "svg_spatial_pearson_beats_or_ties_gp": hyper["svg_spatial_pearson_median"] >= gp["svg_spatial_pearson_median"],
        "at_least_two_localization_metrics_improve": localization_wins >= 2,
        "rmse_within_5_percent_of_gp": hyper["log1p_normalized_rmse"] <= 1.05 * gp["log1p_normalized_rmse"],
        "localization_metrics_won": int(localization_wins),
    }
    gate["pass"] = all(value for key, value in gate.items() if key != "localization_metrics_won")
    payload = {
        "status": "completed",
        "task": "E10.5 + E12.5 -> complete held-out E11.5 serial-section volume",
        "results": results,
        "geometry": geometry_meta,
        "gate": gate,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cpu_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "command": " ".join(sys.argv)},
    }
    (args.output / "temporal_transport_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    columns = [
        "method",
        "svg_spatial_pearson_median",
        "svg_spatial_spearman_median",
        "auprc_enrichment_mean",
        "topk_dice_mean",
        "centroid_error_mean",
        "spatial_emd_mean",
        "log1p_normalized_rmse",
        "runtime_seconds",
    ]
    lines = ["\t".join(columns)]
    for method, values in results.items():
        lines.append("\t".join([method] + [str(values[key]) for key in columns[1:]]))
    (args.output / "metric_table.tsv").write_text("\n".join(lines) + "\n")
    manifest["status"] = "completed"
    manifest["target_expression_access"] = "evaluation_only_after_all_prediction_files_completed"
    manifest["gate_pass"] = gate["pass"]
    (args.output / "leakage_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "Complete E11.5 serial-section expression holdout; seed 42.",
        "All target expression and depth were unavailable until prediction serialization completed.",
        f"Gate pass: {gate['pass']}.",
        "v9 and Spateo-adapted expansion is authorized only if this gate passes.",
    ]
    (args.output / "report.txt").write_text("\n".join(report) + "\n")
    print(json.dumps({"result": str(args.output / "temporal_transport_results.json"), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
