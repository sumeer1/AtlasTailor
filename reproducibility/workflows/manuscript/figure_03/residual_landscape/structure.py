#!/usr/bin/env python3
"""Is the target-specific residual a spatially coherent field or counting noise?

The frozen endpoint reports that the predicted residual correlates with the
truth residual. A correlation of 0.5 against a residual that is mostly Poisson
noise would mean something very different from the same correlation against a
residual that is mostly coherent tissue structure. This module separates the
two by decomposing the truth-residual variance of every predictable gene into

* a counting-noise floor, estimated by Poisson resampling the measured target
  counts at their observed rate and per-cell normalisation factor;
* the remaining structured variance;
* the part of that structured variance each method linearly recovers.

It also measures spatial coherence directly (Moran's I of the residual against a
cell-permutation null), because a residual can exceed the noise floor for purely
technical reasons without being spatially organised.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import AUDITED_METHODS
from .io_frozen import TargetData
from .residual_fields import pearson_columns

K_NEIGHBOURS = 12  # matches the frozen K_LOCAL used by registered IDW
GENE_CHUNK = 64
MORAN_PERMUTATIONS = 20


def neighbour_index(coordinates: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """k nearest neighbours of every cell, self excluded."""
    tree = cKDTree(coordinates)
    return tree.query(coordinates, k=min(k + 1, len(coordinates)))[1][:, 1:]


def moran_columns(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Moran's I per column under uniform row-normalised k-NN weights."""
    n, k = neighbours.shape
    centered = values - values.mean(0)
    denominator = np.sum(centered**2, axis=0)
    out = np.empty(values.shape[1], np.float64)
    for start in range(0, values.shape[1], GENE_CHUNK):
        stop = min(start + GENE_CHUNK, values.shape[1])
        block = centered[:, start:stop]
        lagged = block[neighbours].sum(axis=1)  # (cells, chunk)
        out[start:stop] = np.sum(block * lagged, axis=0)
    return out / np.maximum(k * denominator, 1e-16)


def neighbourhood_variance_fraction(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Var(local mean) / Var(raw), per column.

    Under spatially independent noise this converges to ``1 / (k + 1)``; a value
    well above that indicates the field varies smoothly in space.
    """
    n, k = neighbours.shape
    out = np.empty(values.shape[1], np.float64)
    for start in range(0, values.shape[1], GENE_CHUNK):
        stop = min(start + GENE_CHUNK, values.shape[1])
        block = values[:, start:stop]
        smoothed = (block[neighbours].sum(axis=1) + block) / (k + 1)
        out[start:stop] = smoothed.var(axis=0) / np.maximum(block.var(axis=0), 1e-16)
    return out


def counting_noise_variance(data: TargetData, seed: int = 20260805) -> np.ndarray:
    """Per-gene variance of log1p measurement noise at the observed count rate.

    Draws an independent Poisson realisation of every measured count, applies the
    *observed* per-cell normalisation factor so depth is held fixed, and uses the
    fact that two independent realisations of the same rate differ by twice the
    single-measurement noise variance.
    """
    rng = np.random.default_rng(seed)
    idx = data.predictable_indices
    observed_normalised = data.target_counts_normalised[:, idx]
    factor = data.depth_target / np.maximum(data.target_depth, 1.0)
    raw_counts = observed_normalised / factor[:, None]
    resampled = rng.poisson(np.maximum(raw_counts, 0)).astype(np.float32) * factor[:, None]
    difference = np.log1p(resampled) - np.log1p(observed_normalised)
    return 0.5 * np.mean(difference**2, axis=0)


def structure_table(
    data: TargetData,
    truth_residual: np.ndarray,
    predicted_residual: dict[str, np.ndarray],
) -> list[dict]:
    """Per-gene residual variance decomposition and spatial-coherence test."""
    neighbours = neighbour_index(data.coordinates)
    idx = data.predictable_indices
    genes = data.genes[idx]

    total_variance = truth_residual.var(axis=0)
    noise_variance = counting_noise_variance(data)
    structured = np.maximum(total_variance - noise_variance, 0.0)
    truth_variance = data.truth_log[:, idx].var(axis=0)

    moran = moran_columns(truth_residual, neighbours)
    smooth_fraction = neighbourhood_variance_fraction(truth_residual, neighbours)

    rng = np.random.default_rng(20260805)
    null_moran = np.empty((MORAN_PERMUTATIONS, truth_residual.shape[1]), np.float64)
    for p in range(MORAN_PERMUTATIONS):
        order = rng.permutation(truth_residual.shape[0])
        null_moran[p] = moran_columns(truth_residual[order], neighbours)
    null_mean = null_moran.mean(0)
    null_sd = np.maximum(null_moran.std(0, ddof=1), 1e-12)

    method_stats = {}
    for method in AUDITED_METHODS:
        residual = predicted_residual[method]
        correlation = pearson_columns(truth_residual, residual)
        # Variance of the best linear predictor of the truth residual from the
        # predicted residual, i.e. what the method actually pins down.
        explained = np.clip(correlation, -1, 1) ** 2 * total_variance
        method_stats[method] = (correlation, explained)

    rows = []
    for j, gene in enumerate(genes):
        row = {
            "target_key": data.run.key,
            "stage": data.run.stage,
            "target_embryo": data.run.target,
            "gene": gene,
            "gene_index": int(idx[j]),
            "truth_log_variance": float(truth_variance[j]),
            "residual_variance": float(total_variance[j]),
            # Ratio, not a share: the residual subtracts a field that carries its
            # own variance, so it can exceed the measured truth variance.
            "residual_to_truth_variance_ratio": float(
                total_variance[j] / max(truth_variance[j], 1e-16)
            ),
            "counting_noise_variance": float(noise_variance[j]),
            "structured_residual_variance": float(structured[j]),
            "structured_fraction_of_residual": float(
                structured[j] / max(total_variance[j], 1e-16)
            ),
            "residual_moran_i": float(moran[j]),
            "residual_moran_null_mean": float(null_mean[j]),
            "residual_moran_z": float((moran[j] - null_mean[j]) / null_sd[j]),
            "residual_neighbourhood_variance_fraction": float(smooth_fraction[j]),
            "independent_noise_neighbourhood_expectation": 1.0 / (K_NEIGHBOURS + 1),
        }
        for method, (correlation, explained) in method_stats.items():
            row[f"{method}__residual_pearson"] = float(correlation[j])
            row[f"{method}__explained_residual_variance"] = float(explained[j])
            row[f"{method}__fraction_of_structured_recovered"] = float(
                explained[j] / max(structured[j], 1e-16)
            )
        rows.append(row)
    return rows
