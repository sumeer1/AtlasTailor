#!/usr/bin/env python3
"""Does the recovered residual align with independent biological annotation?

The dataset authors supplied per-cell ``germlayer``, ``type``, ``tissue`` and
``clusters`` labels. The frozen runner never read them — its leakage audit
records ``target_annotations_accessed: false``, and the anchor-panel note states
that no ontology mapping was used. That makes these labels an independent
yardstick: nothing in the locked pipeline could have been fitted to them.

Three questions are asked of every predictable gene:

1. How much of the truth-residual variance is between annotation classes rather
   than within them (eta-squared, against a label-permutation null)?
2. Does the method reproduce the *class profile* — the per-class mean residual —
   or does it merely correlate cell-wise for other reasons?
3. Which biological compartments does the residual correction actually land in?
"""
from __future__ import annotations

import numpy as np

from .config import AUDITED_METHODS, MAP
from .io_frozen import TargetData
from .residual_fields import pearson_columns

MIN_CLASS_CELLS = 20
LABEL_PERMUTATIONS = 20


def usable_annotations(data: TargetData) -> dict[str, np.ndarray]:
    """Annotation columns with at least two classes that clear the size floor."""
    usable = {}
    for column, labels in data.annotations.items():
        if column == "__obsm__":
            continue
        values, counts = np.unique(labels, return_counts=True)
        keep = values[counts >= MIN_CLASS_CELLS]
        if len(keep) >= 2:
            usable[column] = labels
    return usable


def _design(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class names, per-cell class code and class sizes, dropping tiny classes.

    Cells in dropped classes get code ``-1`` and are excluded from the
    between-class statistics.
    """
    values, counts = np.unique(labels, return_counts=True)
    keep = values[counts >= MIN_CLASS_CELLS]
    lookup = {name: i for i, name in enumerate(keep)}
    codes = np.asarray([lookup.get(label, -1) for label in labels], np.int64)
    sizes = np.asarray([int((codes == i).sum()) for i in range(len(keep))], np.int64)
    return keep, codes, sizes


def class_means(values: np.ndarray, codes: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class column means, shape ``(n_classes, n_columns)``."""
    out = np.zeros((n_classes, values.shape[1]), np.float64)
    for c in range(n_classes):
        mask = codes == c
        if mask.any():
            out[c] = values[mask].mean(axis=0)
    return out


def eta_squared(values: np.ndarray, codes: np.ndarray, n_classes: int) -> np.ndarray:
    """Fraction of per-column variance lying between annotation classes."""
    mask = codes >= 0
    subset = values[mask]
    sub_codes = codes[mask]
    grand = subset.mean(axis=0)
    means = class_means(subset, sub_codes, n_classes)
    sizes = np.asarray([(sub_codes == c).sum() for c in range(n_classes)], np.float64)
    between = np.sum(sizes[:, None] * (means - grand) ** 2, axis=0)
    total = np.sum((subset - grand) ** 2, axis=0)
    return between / np.maximum(total, 1e-16)


def annotation_table(
    data: TargetData,
    truth_residual: np.ndarray,
    predicted_residual: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    """Per-gene annotation alignment and per-class recovery.

    Returns ``(gene_rows, class_rows)``.
    """
    idx = data.predictable_indices
    genes = data.genes[idx]
    gene_rows: list[dict] = []
    class_rows: list[dict] = []
    rng = np.random.default_rng(20260805)

    for column, labels in usable_annotations(data).items():
        names, codes, sizes = _design(labels)
        n_classes = len(names)
        truth_eta = eta_squared(truth_residual, codes, n_classes)

        null = np.empty((LABEL_PERMUTATIONS, truth_residual.shape[1]), np.float64)
        for p in range(LABEL_PERMUTATIONS):
            null[p] = eta_squared(truth_residual, rng.permutation(codes), n_classes)
        null_mean = null.mean(0)
        null_sd = np.maximum(null.std(0, ddof=1), 1e-12)

        truth_profile = class_means(truth_residual, codes, n_classes)
        method_stats = {}
        for method in AUDITED_METHODS:
            residual = predicted_residual[method]
            profile = class_means(residual, codes, n_classes)
            method_stats[method] = (
                eta_squared(residual, codes, n_classes),
                pearson_columns(truth_profile, profile),
            )

        for j, gene in enumerate(genes):
            row = {
                "target_key": data.run.key,
                "stage": data.run.stage,
                "annotation": column,
                "n_classes": n_classes,
                "gene": gene,
                "gene_index": int(idx[j]),
                "truth_residual_eta_squared": float(truth_eta[j]),
                "permuted_label_eta_squared": float(null_mean[j]),
                "truth_residual_eta_squared_z": float((truth_eta[j] - null_mean[j]) / null_sd[j]),
            }
            for method, (eta, profile_r) in method_stats.items():
                row[f"{method}__eta_squared"] = float(eta[j])
                row[f"{method}__class_profile_pearson"] = float(profile_r[j])
            gene_rows.append(row)

        # Per-class recovery: restrict the residual correlation to the cells of
        # one compartment, so a strong global correlation cannot mask a
        # compartment where the correction points the wrong way.
        for c, name in enumerate(names):
            mask = codes == c
            truth_block = truth_residual[mask]
            row = {
                "target_key": data.run.key,
                "stage": data.run.stage,
                "annotation": column,
                "class": name,
                "n_cells": int(mask.sum()),
                "cell_fraction": float(mask.mean()),
                "median_absolute_truth_residual": float(np.median(np.abs(truth_block))),
                "median_within_class_truth_residual_variance": float(
                    np.median(truth_block.var(axis=0))
                ),
            }
            for method in AUDITED_METHODS:
                within = pearson_columns(truth_block, predicted_residual[method][mask])
                row[f"{method}__median_within_class_residual_pearson"] = float(
                    np.nanmedian(within)
                )
                row[f"{method}__genes_with_positive_within_class_recovery"] = int(
                    np.nansum(within > 0)
                )
            row["n_genes"] = int(truth_residual.shape[1])
            row["primary_method"] = MAP
            class_rows.append(row)

    return gene_rows, class_rows
