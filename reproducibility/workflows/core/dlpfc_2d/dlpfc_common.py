#!/usr/bin/env python3
"""Shared frozen operations for the isolated 2D DLPFC extension."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree


ANCHOR_COUNT = 16
K_LOCAL = 12
K_BROAD = 32
RANK = 16
RIDGE_GRID = (0.1, 1.0, 10.0)
GAIN_GRID = (0.25, 0.5, 0.75, 1.0)
FUSION_GRID = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), np.float32)
SEEDS = (42, 43, 44)
REGISTRATION_EPOCHS = 30
EXPANDED_EXCLUDE = {
    "admp", "bambia", "foxa", "foxa2", "foxa3", "gata6", "gbx1", "gli1",
    "lhx1a", "msgn1", "noto", "otx1", "shha", "sox19a", "sp5l", "tbx1",
    "tbxta", "tbxtb", "zic2b",
}
AXIAL = ("tbxta", "noto", "shha")
METHODS = (
    "registered_nearest_neighbor",
    "registered_idw_k12",
    "atlas_plus_global_anchor_residual",
    "atlas_plus_local_anchor_residual",
    "anchor_only_calibrated",
    "hyperspatial_map_calibrated_fusion",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def load_10x_h5(path: Path):
    """Load an official 10x filtered matrix as spots x genes CSR."""
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(x) for x in group["shape"][:])
        matrix = csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        ).T.tocsr()
        barcodes = decode(group["barcodes"][:])
        gene_ids = decode(group["features/id"][:])
        gene_names = decode(group["features/name"][:])
    return matrix, barcodes, gene_ids, gene_names


def load_section_metadata(path: Path, section: str, barcodes: np.ndarray, include_layers: bool = False):
    metadata = pd.read_csv(path, sep="\t", dtype={"section": str, "barcode": str})
    section_data = metadata.loc[metadata["section"] == str(section)].copy()
    if len(section_data) != len(barcodes):
        raise RuntimeError(f"{section}: metadata rows {len(section_data)} != matrix spots {len(barcodes)}")
    index = pd.Series(np.arange(len(section_data)), index=section_data["barcode"])
    if set(index.index) != set(barcodes):
        raise RuntimeError(f"{section}: barcode sets differ")
    section_data = section_data.iloc[index.loc[barcodes].to_numpy()].reset_index(drop=True)
    allowed = [
        "section", "barcode", "donor", "adjacent_pair", "position_um",
        "replicate_within_pair", "x_fullres_px", "y_fullres_px",
    ]
    if include_layers:
        allowed.append("cortical_layer")
    return section_data[allowed]


def canonicalize(coordinates: np.ndarray):
    center = np.median(coordinates, axis=0)
    centered = coordinates - center
    scale = max(float(np.quantile(np.linalg.norm(centered, axis=1), 0.99)), 1e-6)
    return (centered / scale).astype(np.float32), {"center": center.tolist(), "scale": scale}


def normalized_source(counts):
    depth = np.maximum(np.asarray(counts.sum(axis=1)).ravel().astype(np.float32), 1.0)
    depth_target = float(np.median(depth[depth > 0]))
    values = counts.toarray().astype(np.float32)
    values *= (depth_target / depth)[:, None]
    return values, np.log1p(values).astype(np.float32), depth, depth_target


def quadrants(coordinates: np.ndarray):
    """Frozen d-dimensional median orthants; four labels when d=2."""
    median = np.median(coordinates, axis=0)
    labels = np.zeros(len(coordinates), np.int8)
    for dim in range(coordinates.shape[1]):
        labels += (2**dim) * (coordinates[:, dim] > median[dim]).astype(np.int8)
    return labels


def neighbor_index_weights(source_coordinates, query_coordinates, k):
    distance, index = cKDTree(source_coordinates).query(
        query_coordinates, k=min(k, len(source_coordinates))
    )
    if np.ndim(distance) == 1:
        distance, index = distance[:, None], index[:, None]
    weight = np.maximum(distance, 1e-5) ** -2
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return index, weight.astype(np.float32)


def apply_neighbors(values, index, weight, gene_chunk=1024):
    output = np.empty((len(index), values.shape[1]), np.float32)
    for begin in range(0, values.shape[1], gene_chunk):
        end = min(begin + gene_chunk, values.shape[1])
        output[:, begin:end] = np.einsum(
            "qk,qkg->qg", weight, values[index, begin:end], optimize=True
        )
    return output


def idw(source_coordinates, source_values, query_coordinates, k):
    index, weight = neighbor_index_weights(source_coordinates, query_coordinates, k)
    return apply_neighbors(source_values, index, weight)


def smooth(values, coordinates, k):
    index = cKDTree(coordinates).query(
        coordinates, k=min(k + 1, len(coordinates))
    )[1][:, 1:]
    return values[index].mean(axis=1).astype(np.float32)


def residual_features(anchor_residual, coordinates, multiscale=True):
    blocks = [anchor_residual]
    if multiscale:
        blocks.extend((
            smooth(anchor_residual, coordinates, K_LOCAL),
            smooth(anchor_residual, coordinates, K_BROAD),
        ))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def pearson_columns(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.square(x).sum(axis=0) * np.square(y).sum(axis=0))
    result = np.full(x.shape[1], np.nan, np.float64)
    valid = denom > 1e-12
    result[valid] = (x[:, valid] * y[:, valid]).sum(axis=0) / denom[valid]
    return result


def fit_centered_ridge_rank(x, y, ridge, rank):
    x_mean = x.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0), 1e-5)
    y_mean = y.mean(axis=0)
    xs = (x - x_mean) / x_scale
    yc = y - y_mean
    weight = np.linalg.solve(
        xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ yc
    )
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    keep = min(rank, len(s))
    weight = (u[:, :keep] * s[:keep]) @ vt[:keep]
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_scale": x_scale.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "weight": weight.astype(np.float32),
    }


def apply_centered(model, x, columns=None):
    weight = model["weight"] if columns is None else model["weight"][:, columns]
    mean = model["y_mean"] if columns is None else model["y_mean"][columns]
    return mean + ((x - model["x_mean"]) / model["x_scale"]) @ weight


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
    errors = np.mean((candidates - truth[None, :, :]) ** 2, axis=1)
    correlations = np.stack([pearson_columns(truth, candidate) for candidate in candidates])
    best_error = errors.min(axis=0)
    eligible = errors <= tolerance * np.maximum(best_error[None, :], 1e-10)
    score = np.where(eligible, correlations, -np.inf)
    selection = np.argmax(score, axis=0)
    selected = candidates[selection, :, np.arange(truth.shape[1])].T
    base_error = np.mean((base - truth) ** 2, axis=0)
    selected_error = np.mean((selected - truth) ** 2, axis=0)
    selected_correlation = correlations[selection, np.arange(truth.shape[1])]
    base_correlation = pearson_columns(truth, base)
    predictable = (selected_error < base_error) & (
        selected_correlation > base_correlation + 0.03
    )
    return selection.astype(np.int16), predictable, {
        "selected_validation_mse": selected_error.astype(np.float32),
        "selected_validation_pearson": selected_correlation.astype(np.float32),
        "base_validation_mse": base_error.astype(np.float32),
        "base_validation_pearson": base_correlation.astype(np.float32),
    }


def depth_predictor(source_counts_dense, anchor_indices):
    x = np.c_[
        np.ones(len(source_counts_dense)),
        np.log1p(source_counts_dense[:, anchor_indices].sum(axis=1)),
    ]
    y = np.log1p(source_counts_dense.sum(axis=1))
    return np.linalg.solve(x.T @ x + np.diag([0.0, 1.0]), x.T @ y).astype(np.float32)


def predicted_depth(anchor_counts, weight):
    x = np.c_[np.ones(len(anchor_counts)), np.log1p(anchor_counts.sum(axis=1))]
    return np.maximum(np.expm1(x @ weight), 1.0).astype(np.float32)


def select_anchors(source_log, gene_names):
    prevalence = np.mean(source_log > 0, axis=0)
    variance = np.var(source_log, axis=0)
    lowered = np.char.lower(gene_names.astype(str))
    candidate_mask = np.asarray([
        0.05 <= prevalence[index] <= 0.98
        and variance[index] > 1e-5
        and lowered[index] not in EXPANDED_EXCLUDE
        and not lowered[index].startswith(("mt-", "rpl", "rps"))
        for index in range(len(gene_names))
    ])
    candidates = np.flatnonzero(candidate_mask).astype(np.int64)
    rng = np.random.default_rng(42116)
    sample = np.sort(rng.choice(len(source_log), min(5000, len(source_log)), replace=False))
    x = source_log[sample].astype(np.float32, copy=True)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-5)
    correlation = np.abs((x[:, candidates].T @ x) / max(len(x) - 1, 1)).astype(np.float32)
    axial_indices = [int(np.flatnonzero(lowered == gene)[0]) for gene in AXIAL if np.any(lowered == gene)]
    axial_filter_applied = bool(axial_indices)
    if axial_indices:
        allowed = np.max(correlation[:, axial_indices], axis=1) < 0.40
        candidates, correlation = candidates[allowed], correlation[allowed]
    if len(candidates) < ANCHOR_COUNT:
        raise RuntimeError(f"Only {len(candidates)} frozen-rule anchor candidates")
    coverage = np.zeros(source_log.shape[1], np.float32)
    chosen = []
    gains_at_selection = []
    for _ in range(ANCHOR_COUNT):
        gains = np.mean(
            np.maximum(coverage[None, :], correlation**2) - coverage[None, :], axis=1
        )
        for index in chosen:
            local = np.flatnonzero(candidates == index)
            if len(local):
                gains[local[0]] = -np.inf
        local = int(np.argmax(gains))
        chosen.append(int(candidates[local]))
        gains_at_selection.append(float(gains[local]))
        coverage = np.maximum(coverage, correlation[local] ** 2)
    return np.asarray(chosen, np.int64), candidates, {
        "candidate_count": int(len(candidates)),
        "mean_squared_gene_coverage": float(coverage.mean()),
        "maximum_allowed_source_correlation_to_axial": 0.40,
        "exact_axial_symbols_present": [gene_names[i] for i in axial_indices],
        "axial_filter_applied": axial_filter_applied,
        "marginal_coverage_gains": gains_at_selection,
    }


def source_svg_indices(source_log, coordinates, count=100):
    take = min(3500, len(source_log))
    subset = np.linspace(0, len(source_log) - 1, take, dtype=np.int64)
    x = source_log[subset]
    neighbors = cKDTree(coordinates[subset]).query(
        coordinates[subset], k=min(13, take)
    )[1][:, 1:]
    centered = x - x.mean(axis=0, keepdims=True)
    score = (centered * centered[neighbors].mean(axis=1)).sum(axis=0) / np.maximum(
        np.square(centered).sum(axis=0), 1e-8
    )
    order = np.argsort(score)[-min(count, len(score)):]
    return np.sort(order), score.astype(np.float32)


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
