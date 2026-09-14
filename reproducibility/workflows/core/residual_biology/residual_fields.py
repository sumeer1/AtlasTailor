#!/usr/bin/env python3
"""Reconstruct the target-specific residual fields the Class A claim rests on.

The frozen endpoint is ``residual_spatial_pearson_median``: the per-gene Pearson
correlation, across target cells, between the *truth residual*
``truth_log - registered_idw`` and the *predicted residual*
``prediction - registered_idw``, taken over the source-defined predictable genes
with anchors removed, then median-reduced over genes.

This module rebuilds those two residual fields and refuses to continue unless it
reproduces the frozen per-seed numbers. Everything downstream in the audit is a
statement about these exact arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AUDITED_METHODS, IDW, SEEDS, TargetRun
from .io_frozen import TargetData, load_prediction

# The frozen metrics were written from float32 arrays by the same code path, so
# agreement should be at float32 round-off. Anything larger means the
# reconstruction is not the frozen object.
REPRODUCTION_TOLERANCE = 1e-5


def pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Frozen column-wise Pearson correlation."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    return np.sum(xc * yc, axis=0) / np.sqrt(
        np.maximum(np.sum(xc**2, axis=0) * np.sum(yc**2, axis=0), 1e-16)
    )


def residual_pearson(truth_log, base_log, prediction_log, indices) -> np.ndarray:
    """Frozen residual endpoint, per gene."""
    return pearson_columns(
        (truth_log - base_log)[:, indices], (prediction_log - base_log)[:, indices]
    )


@dataclass
class ResidualFields:
    """Residual fields for one target volume at one seed.

    ``truth`` and each entry of ``predicted`` are ``(cells, predictable_genes)``
    arrays already restricted to the frozen predictable-gene set, so gene column
    ``j`` everywhere refers to ``genes[predictable_indices[j]]``.
    """

    run: TargetRun
    seed: int
    gene_indices: np.ndarray
    gene_names: np.ndarray
    truth: np.ndarray
    predicted: dict[str, np.ndarray]

    @property
    def n_genes(self) -> int:
        return int(self.truth.shape[1])


def build_fields(data: TargetData, seed: int) -> ResidualFields:
    """Construct truth and predicted residual fields for one seed."""
    manifest = data.manifest
    base = load_prediction(data.run, seed, IDW, manifest)
    idx = data.predictable_indices
    truth_residual = (data.truth_log[:, idx] - base[:, idx]).astype(np.float32)
    predicted = {}
    for method in AUDITED_METHODS:
        prediction = load_prediction(data.run, seed, method, manifest)
        predicted[method] = (prediction[:, idx] - base[:, idx]).astype(np.float32)
    return ResidualFields(
        run=data.run,
        seed=seed,
        gene_indices=idx,
        gene_names=data.genes[idx],
        truth=truth_residual,
        predicted=predicted,
    )


def frozen_residual_table(run: TargetRun) -> pd.DataFrame:
    """The frozen per-seed residual metrics written by the locked runner."""
    return pd.read_csv(run.prediction_dir / "residual_metrics.tsv", sep="\t")


def reproduction_check(data: TargetData, fields: ResidualFields) -> list[dict]:
    """Compare recomputed residual medians against the frozen record.

    Returns one row per method. The caller must abort if any row fails: an audit
    of a residual field that is not the frozen residual field would describe a
    different object than the one the decision was made on.
    """
    frozen = frozen_residual_table(data.run)
    frozen = frozen[frozen.seed == fields.seed].set_index("method")
    rows = []
    for method, residual in fields.predicted.items():
        recomputed = float(np.nanmedian(pearson_columns(fields.truth, residual)))
        recorded = float(frozen.loc[method, "residual_spatial_pearson_median"])
        rows.append({
            "target_key": data.run.key,
            "stage": data.run.stage,
            "seed": fields.seed,
            "method": method,
            "frozen_residual_spatial_pearson_median": recorded,
            "recomputed_residual_spatial_pearson_median": recomputed,
            "absolute_difference": abs(recomputed - recorded),
            "reproduced": abs(recomputed - recorded) <= REPRODUCTION_TOLERANCE,
            "frozen_predictable_genes": int(frozen.loc[method, "predictable_genes"]),
            "audit_predictable_genes": fields.n_genes,
        })
    return rows


def seed_mean_truth(data: TargetData) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Seed-averaged truth and predicted residual fields.

    The base field depends on the seed (registration is re-fit per seed), so the
    truth residual does too. Averaging over seeds gives the residual landscape
    that is stable across registration draws, which is the object the biological
    questions are about.
    """
    truth_sum = None
    predicted_sum: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        fields = build_fields(data, seed)
        truth_sum = fields.truth if truth_sum is None else truth_sum + fields.truth
        for method, residual in fields.predicted.items():
            predicted_sum[method] = predicted_sum.get(method, 0) + residual
    n = float(len(SEEDS))
    return truth_sum / n, {method: value / n for method, value in predicted_sum.items()}
