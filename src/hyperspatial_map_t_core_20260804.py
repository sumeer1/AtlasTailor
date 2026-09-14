#!/usr/bin/env python3
"""Frozen utilities for the HyperSpatial-MAP-T cross-fitted experiment.

This module deliberately keeps identifiers outside every numerical feature
matrix.  Late target expression is accessible only through ``LateTruthVault``
after a prediction manifest, checksums and an unblinding sentinel exist.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata


IDENTIFIER_TOKENS = (
    "id", "index", "row", "ancestor", "descendant", "lineage", "amat",
    "wemerfish", "node", "family",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump_new(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def decode(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def read_gene_names(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return decode(handle["var/_index"][:])


def read_ids(path: Path):
    """Read join identifiers only; never reads X, layers or logFC columns."""
    with h5py.File(path, "r") as handle:
        return {
            "Amat_id": np.asarray(handle["obs/Amat_id"][:], np.int64),
            "weMERFISH_id": np.asarray(handle["obs/weMERFISH_id"][:], np.int64),
        }


def read_geometry(path: Path, key="global_sphere") -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[f"obsm/{key}"][:], np.float32)


def read_counts(path: Path, columns=None) -> np.ndarray:
    """Read count layers, optionally restricted to frozen columns."""
    with h5py.File(path, "r") as handle:
        if columns is None:
            return (np.asarray(handle["layers/spliced"][:], np.float32)
                    + np.asarray(handle["layers/unspliced"][:], np.float32))
        columns = np.asarray(columns, np.int64)
        order = np.argsort(columns)
        reverse = np.argsort(order)
        cols = columns[order]
        result = (np.asarray(handle["layers/spliced"][:, cols], np.float32)
                  + np.asarray(handle["layers/unspliced"][:, cols], np.float32))
        return result[:, reverse]


class LateTruthVault:
    def __init__(self, late_path: Path, fold_dir: Path):
        self.path = late_path
        self.fold_dir = fold_dir

    def open(self):
        sentinel = self.fold_dir / "UNBLINDING_SENTINEL.json"
        manifest = self.fold_dir / "PRE_UNBLINDING_PREDICTION_HASHES.tsv"
        if not sentinel.exists() or not manifest.exists():
            raise RuntimeError("Late target truth is sealed until predictions are hashed and sentinel exists")
        record = json.loads(sentinel.read_text())
        if record.get("prediction_hash_manifest_sha256") != sha256(manifest):
            raise RuntimeError("Prediction hash manifest changed after sentinel")
        return read_counts(self.path)


def assert_no_identifier_features(feature_names) -> None:
    def forbidden(name):
        words = set(filter(None, re.split(r"[^a-z0-9]+", name.lower())))
        return bool(words & set(IDENTIFIER_TOKENS)) or name.lower() in {"cellid", "amatid", "wemerfishid"}
    bad = [name for name in feature_names if forbidden(name)]
    if bad:
        raise AssertionError(f"Identifier-like predictive features forbidden: {bad}")


def canonicalize(x, reference=None):
    x = np.asarray(x, np.float32)
    if reference is None:
        center = np.median(x, axis=0)
        scale = max(float(np.quantile(np.linalg.norm(x - center, axis=1), .99)), 1e-6)
    else:
        center, scale = np.asarray(reference["center"], np.float32), float(reference["scale"])
    return ((x - center) / scale).astype(np.float32), {"center": center.tolist(), "scale": scale}


def normalize_counts(counts, depth_target):
    depth = np.maximum(np.sum(counts, axis=1), 1.0)
    return counts * (float(depth_target) / depth)[:, None]


def depth_model(source_counts, anchor_indices):
    x = np.c_[np.ones(len(source_counts)), np.log1p(source_counts[:, anchor_indices].sum(1))]
    y = np.log1p(np.maximum(source_counts.sum(1), 1))
    return np.linalg.solve(x.T @ x + np.diag([0., 1.]), x.T @ y).astype(np.float32)


def apply_depth(anchor_counts, weights):
    x = np.c_[np.ones(len(anchor_counts)), np.log1p(anchor_counts.sum(1))]
    return np.maximum(np.expm1(x @ weights), 1.0).astype(np.float32)


def idw(source_xyz, source_values, query_xyz, k=12, chunk=2048):
    tree = cKDTree(source_xyz)
    out = np.empty((len(query_xyz), source_values.shape[1]), np.float32)
    for start in range(0, len(query_xyz), chunk):
        q = query_xyz[start:start + chunk]
        distance, index = tree.query(q, k=min(k, len(source_xyz)))
        if np.ndim(distance) == 1:
            distance, index = distance[:, None], index[:, None]
        weight = np.maximum(distance, 1e-5) ** -2
        weight /= np.maximum(weight.sum(1, keepdims=True), 1e-12)
        out[start:start + len(q)] = np.einsum("bk,bkg->bg", weight, source_values[index])
    return out


def smooth(values, coordinates, k):
    indices = cKDTree(coordinates).query(coordinates, k=min(k + 1, len(coordinates)))[1][:, 1:]
    return values[indices].mean(1).astype(np.float32)


def multiscale(residual, coordinates):
    return np.c_[residual, smooth(residual, coordinates, 12), smooth(residual, coordinates, 32)].astype(np.float32)


def fit_centered_ridge_rank(x, y, ridge=1.0, rank=16):
    xm, xs, ym = x.mean(0), np.maximum(x.std(0), 1e-5), y.mean(0)
    z = (x - xm) / xs
    weight = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ (y - ym))
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    keep = min(rank, len(s))
    weight = (u[:, :keep] * s[:keep]) @ vt[:keep]
    return {"x_mean": xm.astype(np.float32), "x_scale": xs.astype(np.float32),
            "y_mean": ym.astype(np.float32), "weight": weight.astype(np.float32)}


def apply_centered(model, x):
    return (model["y_mean"] + ((x - model["x_mean"]) / model["x_scale"]) @ model["weight"]).astype(np.float32)


def pearson_columns(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    denom = np.sqrt(np.maximum(np.sum(xc * xc, 0) * np.sum(yc * yc, 0), 1e-16))
    return np.sum(xc * yc, 0) / denom


def pooled_pearson(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_columns(x, y):
    return np.asarray([np.corrcoef(rankdata(x[:, j]), rankdata(y[:, j]))[0, 1]
                       if np.std(x[:, j]) > 1e-10 and np.std(y[:, j]) > 1e-10 else np.nan
                       for j in range(x.shape[1])])


def family_blocks(start_xyz, n_blocks=8):
    med = np.median(start_xyz, axis=0)
    block = ((start_xyz[:, 0] > med[0]).astype(int)
             + 2 * (start_xyz[:, 1] > med[1]).astype(int)
             + 4 * (start_xyz[:, 2] > med[2]).astype(int))
    if len(np.unique(block)) < min(n_blocks, 8):
        order = np.argsort(start_xyz[:, 0] + .37 * start_xyz[:, 1] + .13 * start_xyz[:, 2])
        block = np.empty(len(start_xyz), np.int64)
        for b, indices in enumerate(np.array_split(order, n_blocks)):
            block[indices] = b
    return block.astype(np.int64)


def assert_family_split(families, blocks):
    seen = {}
    for family, block in zip(families, blocks):
        old = seen.setdefault(int(family), int(block))
        if old != int(block):
            raise AssertionError("Descendants of one ancestor cross validation blocks")


def lineage_transport(early, descendant_ancestor, early_ancestors):
    lookup = {int(a): i for i, a in enumerate(early_ancestors)}
    missing = np.asarray([int(a) not in lookup for a in descendant_ancestor])
    if np.any(missing):
        raise KeyError(f"{int(missing.sum())} descendant ancestors absent from early state")
    indices = np.asarray([lookup[int(a)] for a in descendant_ancestor], np.int64)
    return early[indices].astype(np.float32), indices


def temporal_guard(base, candidates, truth, margin=.03, tolerance=1.05):
    """Per-gene source-only guard. Candidate axis is first."""
    mse = np.mean((candidates - truth[None]) ** 2, axis=1)
    corr = np.stack([pearson_columns(truth, pred) for pred in candidates])
    base_mse = np.mean((base - truth) ** 2, axis=0)
    base_corr = pearson_columns(truth, base)
    best = np.min(mse, axis=0)
    eligible = mse <= tolerance * np.maximum(best[None], 1e-10)
    selection = np.argmax(np.where(eligible, corr, -np.inf), axis=0)
    selected_mse = mse[selection, np.arange(truth.shape[1])]
    selected_corr = corr[selection, np.arange(truth.shape[1])]
    guard = (selected_mse < base_mse) & (selected_corr > base_corr + margin)
    return selection.astype(np.int64), guard, {
        "selected_mse": selected_mse, "selected_corr": selected_corr,
        "base_mse": base_mse, "base_corr": base_corr,
    }


def future_residuals(truth, forecast, lineage_early):
    return truth - lineage_early, forecast - lineage_early


def tracking_overlap(source_sets, target_sets):
    all_source = set().union(*source_sets.values()) if source_sets else set()
    shared = {family for family, nodes in target_sets.items() if not nodes.isdisjoint(all_source)}
    disjoint = set(target_sets) - shared
    return shared, disjoint


def moran(values, xyz, k=12):
    neighbors = cKDTree(xyz).query(xyz, k=min(k + 1, len(xyz)))[1][:, 1:]
    centered = values - values.mean()
    denom = np.sum(centered * centered)
    if denom < 1e-12:
        return np.nan
    return float(np.sum(centered[:, None] * centered[neighbors]) / (max(neighbors.shape[1], 1) * denom))


def domain_metrics(truth, pred, xyz, threshold, neighbors=None):
    positive = truth > threshold
    k = int(positive.sum())
    if k < 5 or k >= len(truth):
        return {k: np.nan for k in ("dice", "iou", "boundary_f1", "boundary_displacement", "centroid_error")}
    guess = np.zeros(len(pred), bool)
    guess[np.argpartition(pred, -k)[-k:]] = True
    inter = int(np.sum(positive & guess))
    dice = 2 * inter / max(int(positive.sum() + guess.sum()), 1)
    iou = inter / max(int(np.sum(positive | guess)), 1)
    diameter = max(float(np.linalg.norm(xyz.max(0) - xyz.min(0))), 1e-8)
    tc, pc = xyz[positive].mean(0), xyz[guess].mean(0)
    centroid = float(np.linalg.norm(tc - pc) / diameter)
    if neighbors is None:
        neighbors = cKDTree(xyz).query(xyz, k=min(13, len(xyz)))[1][:, 1:]
    tb = positive & np.any(~positive[neighbors], axis=1)
    pb = guess & np.any(~guess[neighbors], axis=1)
    if tb.any() and pb.any():
        dt = cKDTree(xyz[tb]).query(xyz[pb])[0]
        dp = cKDTree(xyz[pb]).query(xyz[tb])[0]
        boundary = float(.5 * (dt.mean() + dp.mean()) / diameter)
        tolerance = .03 * diameter
        precision, recall = np.mean(dt <= tolerance), np.mean(dp <= tolerance)
        bf1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    else:
        boundary, bf1 = np.nan, np.nan
    return {"dice": dice, "iou": iou, "boundary_f1": bf1,
            "boundary_displacement": boundary, "centroid_error": centroid}
