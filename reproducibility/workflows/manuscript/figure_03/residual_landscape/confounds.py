#!/usr/bin/env python3
"""Non-biological explanations for the recovered residual, tested one by one.

A positive residual correlation is consistent with several stories that are not
"the method recovers specimen-specific biology":

* **Sequencing depth.** The frozen pipeline predicts target depth from the anchor
  panel, so it can reproduce any residual component that is really a per-cell
  depth rescaling. Tested by partialling per-cell log depth out of both residual
  fields.
* **Registration quality.** Residuals are largest where the atlas transfer is
  unstable. If recovery lives only there, the correction is geometric rather
  than biological. Tested by stratifying on the disagreement between the k=12
  IDW transfer and the k=1 nearest-neighbour transfer, both of which are frozen
  saved artefacts.
* **Anchor propagation.** A gene strongly correlated with an anchor in the source
  can be recovered without any spatial learning. Tested by relating per-gene
  recovery to the gene's maximum source correlation with the anchor panel.
* **Coarse global offset.** A single low-frequency shift would inflate the
  cell-wise correlation without describing tissue-scale structure. Tested by
  splitting the residual into a broad component and a fine local component.
* **6-somite coordinate choice.** The frozen loader used raw ``spatial`` at
  6-somite rather than the available ``spatial_rescaled_z``. No method is re-run
  here; the audit only measures whether the residual landscape is sensitive to
  that choice.
"""
from __future__ import annotations

import numpy as np

from .config import AUDITED_METHODS, IDW, NEAREST
from .io_frozen import TargetData, load_prediction
from .residual_fields import pearson_columns
from .structure import moran_columns, neighbour_index

BROAD_NEIGHBOURS = 64
INSTABILITY_BINS = 5


def _residualize(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Remove the least-squares projection of each column onto one covariate."""
    z = covariate - covariate.mean()
    denominator = float(np.dot(z, z))
    if denominator <= 1e-12:
        return values - values.mean(0)
    centered = values - values.mean(0)
    slope = (z @ centered) / denominator
    return centered - np.outer(z, slope)


def _correlate_with_vector(values: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column with one shared vector."""
    z = vector - vector.mean()
    centered = values - values.mean(0)
    denominator = np.sqrt(np.maximum(float(np.dot(z, z)) * np.sum(centered**2, axis=0), 1e-16))
    return (z @ centered) / denominator


def depth_confound(
    data: TargetData,
    truth_residual: np.ndarray,
    predicted_residual: dict[str, np.ndarray],
) -> list[dict]:
    """Residual recovery before and after partialling out per-cell log depth."""
    log_depth = np.log1p(data.target_depth).astype(np.float64)
    truth_adjusted = _residualize(truth_residual.astype(np.float64), log_depth)
    depth_correlation = _correlate_with_vector(truth_residual.astype(np.float64), log_depth)
    rows = []
    for method in AUDITED_METHODS:
        residual = predicted_residual[method].astype(np.float64)
        raw = pearson_columns(truth_residual.astype(np.float64), residual)
        partial = pearson_columns(truth_adjusted, _residualize(residual, log_depth))
        rows.append({
            "target_key": data.run.key,
            "stage": data.run.stage,
            "method": method,
            "median_residual_pearson": float(np.nanmedian(raw)),
            "median_depth_partialled_residual_pearson": float(np.nanmedian(partial)),
            "median_change_after_depth_removal": float(np.nanmedian(partial - raw)),
            "median_truth_residual_correlation_with_log_depth": float(
                np.nanmedian(depth_correlation)
            ),
            "median_absolute_truth_residual_correlation_with_log_depth": float(
                np.nanmedian(np.abs(depth_correlation))
            ),
            "n_genes": int(truth_residual.shape[1]),
        })
    return rows


def transfer_instability(data: TargetData, seed: int) -> np.ndarray:
    """Per-cell disagreement between the k=12 and k=1 registered transfers."""
    idx = data.predictable_indices
    base = load_prediction(data.run, seed, IDW, data.manifest)[:, idx]
    nearest = load_prediction(data.run, seed, NEAREST, data.manifest)[:, idx]
    return np.mean(np.abs(base - nearest), axis=1).astype(np.float64)


def registration_confound(
    data: TargetData,
    truth_residual: np.ndarray,
    predicted_residual: dict[str, np.ndarray],
    seed: int,
) -> list[dict]:
    """Residual magnitude and recovery stratified by local transfer instability."""
    instability = transfer_instability(data, seed)
    edges = np.quantile(instability, np.linspace(0, 1, INSTABILITY_BINS + 1))
    edges[-1] = np.inf
    rows = []
    for b in range(INSTABILITY_BINS):
        mask = (instability >= edges[b]) & (instability < edges[b + 1])
        if mask.sum() < 50:
            continue
        truth_block = truth_residual[mask]
        row = {
            "target_key": data.run.key,
            "stage": data.run.stage,
            "instability_quintile": b + 1,
            "n_cells": int(mask.sum()),
            "mean_transfer_instability": float(instability[mask].mean()),
            "median_truth_residual_variance": float(np.median(truth_block.var(axis=0))),
        }
        for method in AUDITED_METHODS:
            within = pearson_columns(truth_block, predicted_residual[method][mask])
            row[f"{method}__median_residual_pearson"] = float(np.nanmedian(within))
        rows.append(row)
    return rows


def anchor_propagation(
    data: TargetData,
    truth_residual: np.ndarray,
    predicted_residual: dict[str, np.ndarray],
) -> list[dict]:
    """Per-gene recovery against the gene's maximum source correlation to an anchor."""
    idx = data.predictable_indices
    source = data.source_log
    centered = source - source.mean(0)
    scale = np.maximum(np.sqrt((centered**2).sum(0)), 1e-12)
    normalised = centered / scale
    anchor_block = normalised[:, data.anchor_indices]
    correlation = np.abs(normalised[:, idx].T @ anchor_block)  # (genes, anchors)
    max_anchor_correlation = correlation.max(axis=1)

    recovery = {
        method: pearson_columns(truth_residual, predicted_residual[method])
        for method in AUDITED_METHODS
    }
    rows = []
    for j, gene in enumerate(data.genes[idx]):
        row = {
            "target_key": data.run.key,
            "stage": data.run.stage,
            "gene": gene,
            "gene_index": int(idx[j]),
            "max_source_correlation_to_anchor_panel": float(max_anchor_correlation[j]),
        }
        for method, values in recovery.items():
            row[f"{method}__residual_pearson"] = float(values[j])
        rows.append(row)
    return rows


def spatial_scale(
    data: TargetData,
    truth_residual: np.ndarray,
    predicted_residual: dict[str, np.ndarray],
) -> list[dict]:
    """Split the residual into a broad component and a fine local component.

    The broad component is the mean over a wide neighbourhood; the fine component
    is what remains. Recovery concentrated entirely in the broad component would
    indicate a global offset correction rather than tissue-scale structure.
    """
    neighbours = neighbour_index(data.coordinates, BROAD_NEIGHBOURS)
    def split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        broad = np.empty_like(values)
        for start in range(0, values.shape[1], 32):
            stop = min(start + 32, values.shape[1])
            block = values[:, start:stop]
            broad[:, start:stop] = (block[neighbours].sum(axis=1) + block) / (
                neighbours.shape[1] + 1
            )
        return broad, values - broad

    truth_broad, truth_fine = split(truth_residual)
    total = truth_residual.var(axis=0)
    rows = []
    for method in AUDITED_METHODS:
        pred_broad, pred_fine = split(predicted_residual[method])
        rows.append({
            "target_key": data.run.key,
            "stage": data.run.stage,
            "method": method,
            "neighbourhood_size": BROAD_NEIGHBOURS,
            "median_broad_variance_share_of_truth_residual": float(
                np.median(truth_broad.var(axis=0) / np.maximum(total, 1e-16))
            ),
            "median_broad_component_pearson": float(
                np.nanmedian(pearson_columns(truth_broad, pred_broad))
            ),
            "median_fine_component_pearson": float(
                np.nanmedian(pearson_columns(truth_fine, pred_fine))
            ),
            "median_full_residual_pearson": float(
                np.nanmedian(pearson_columns(truth_residual, predicted_residual[method]))
            ),
        })
    return rows


def coordinate_sensitivity(data: TargetData, truth_residual: np.ndarray) -> list[dict]:
    """How much the residual landscape depends on the frozen coordinate choice.

    Only descriptive: the frozen predictions are never recomputed. At 6-somite
    the alternative ``spatial_rescaled_z`` field exists and the frozen run
    flagged that it was not used, so the audit reports how different the two
    neighbourhood structures are and whether residual coherence changes under
    the alternative.
    """
    obsm = data.annotations.get("__obsm__", {})
    rows = []
    base_neighbours = neighbour_index(data.coordinates)
    base_moran = moran_columns(truth_residual, base_neighbours)
    for name, coordinates in obsm.items():
        if name == data.coordinate_key:
            continue
        center = np.median(coordinates, axis=0)
        centered = coordinates - center
        scale = max(float(np.quantile(np.linalg.norm(centered, axis=1), 0.99)), 1e-6)
        alternative = (centered / scale).astype(np.float32)
        alt_neighbours = neighbour_index(alternative)
        overlap = np.mean([
            len(set(a) & set(b)) / a.size
            for a, b in zip(base_neighbours, alt_neighbours)
        ])
        alt_moran = moran_columns(truth_residual, alt_neighbours)
        rows.append({
            "target_key": data.run.key,
            "stage": data.run.stage,
            "frozen_coordinate_key": data.coordinate_key,
            "alternative_coordinate_key": name,
            "mean_knn_overlap_fraction": float(overlap),
            "median_residual_moran_i_frozen": float(np.median(base_moran)),
            "median_residual_moran_i_alternative": float(np.median(alt_moran)),
            "median_moran_change": float(np.median(alt_moran - base_moran)),
        })
    if not rows:
        rows.append({
            "target_key": data.run.key,
            "stage": data.run.stage,
            "frozen_coordinate_key": data.coordinate_key,
            "alternative_coordinate_key": "none_available",
            "mean_knn_overlap_fraction": float("nan"),
            "median_residual_moran_i_frozen": float(np.median(base_moran)),
            "median_residual_moran_i_alternative": float("nan"),
            "median_moran_change": float("nan"),
        })
    return rows
