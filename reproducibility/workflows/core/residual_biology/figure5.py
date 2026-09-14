#!/usr/bin/env python3
"""Main Figure 5: geometry-corrected residuals beyond anatomy.

Every scalar in this figure is read from the completed residual-landscape audit
tables. Nothing is re-derived with a different estimator and HyperSpatial-MAP is
never re-run. The only frozen arrays opened are those needed to *draw* the panel-a
maps and the panel-e inset profile, and they are loaded through the audit's own
read-only helpers.

Usage::

    python -m analysis.residual_landscape.figure5 \
        --audit-dir results_v10/hyperspatial_map_residual_landscape_20260805_135217
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from . import annotation as annotation_module
from . import structure as structure_module
from .config import (
    ANCHOR,
    MAP,
    OUTPUT_PARENT,
    OUTPUT_PREFIX,
    RUNS,
    SEEDS,
    assert_writable,
)
from .io_frozen import load_prediction, load_target
from .report import load_tables
from .residual_fields import pearson_columns

FIGURE_TITLE = (
    "Geometry-corrected residuals reveal fine-scale developmental variation "
    "beyond anatomy"
)

# Okabe-Ito, colour-blind safe. Stage colour is constant across every panel.
STAGE_COLOURS = {
    "50% epiboly": "#0072B2",
    "75% epiboly": "#E69F00",
    "6-somite": "#009E73",
}
STAGE_ORDER = ("50% epiboly", "75% epiboly", "6-somite")
# Reciprocal atlas constructions, not independent replicates.
DIRECTION_MARKERS = {"E1_to_E2": "o", "E2_to_E1": "^"}
DIRECTION_LABELS = {"E1_to_E2": "E1 to E2", "E2_to_E1": "E2 to E1"}

ANNOTATION_ORDER = ("germlayer", "type", "tissue", "clusters")
ANNOTATION_LABELS = {
    "germlayer": "Germ layer",
    "type": "Cell type",
    "tissue": "Tissue",
    "clusters": "Cluster",
}

MM = 1.0 / 25.4
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 232.0

BASE_FONT = 7.0
LABEL_FONT = 7.5
TITLE_FONT = 8.0
PANEL_FONT = 9.0

# Panel-a selection rule. Every threshold is fixed here and serialised to
# Figure_5_gene_selection.json; the chosen gene is never overridden by hand.
SELECTION = {
    "stage": "6-somite",
    "normalised_residual_variability_percentile": 0.90,
    "counting_noise_null_percentile": 0.95,
    "counting_noise_null_draws": 20,
    "spatial_permutation_null_percentile": 0.95,
    "spatial_permutation_null_z": 1.645,
    "registration_perturbation_seeds": list(SEEDS),
    "registration_perturbation_min_residual_pearson": 0.5,
    "min_map_residual_pearson": 0.5,
    "min_detection_fraction": 0.10,
    "max_detection_fraction": 0.98,
    "tie_break": "highest tissue-scale annotation eta squared",
}


def direction_of(target_key: str) -> str:
    """``50p_E1_to_E2`` -> ``E1_to_E2``."""
    return target_key.split("_", 1)[1]


def run_for(target_key: str):
    for run in RUNS:
        if run.key == target_key:
            return run
    raise KeyError(target_key)


# ----------------------------------------------------------------------------
# Selection


@dataclass
class Selection:
    """Outcome of the machine-readable panel-a example rule."""

    target_key: str
    stage: str
    direction: str
    gene: str
    gene_index: int
    n_cells: int
    statistics: dict
    survivors: list[dict] = field(default_factory=list)
    runner_up: dict | None = None
    criteria_applied: list[dict] = field(default_factory=list)


def _detection_fraction(data, gene_index: int) -> float:
    """Share of target cells with a non-zero measured count for this gene."""
    return float((data.target_counts_normalised[:, gene_index] > 0).mean())


def _noise_floor_percentile(data, draws: int, percentile: float) -> np.ndarray:
    """Upper percentile of the audit's own counting-noise estimator over draws.

    Uses ``structure.counting_noise_variance`` unchanged; only the number of
    Poisson draws differs, which turns the audit's point estimate into the null
    distribution the selection rule asks for.
    """
    samples = np.stack([
        structure_module.counting_noise_variance(data, seed=20260805 + i)
        for i in range(draws)
    ])
    return np.quantile(samples, percentile, axis=0)


def _per_seed_recovery(data, gene_positions: np.ndarray) -> np.ndarray:
    """MAP residual Pearson at each registration seed, shape ``(seeds, genes)``.

    Registration is re-fit per seed, so spread across seeds is the audit's
    available registration perturbation. The metric is the audit's own
    ``pearson_columns`` on the audit's own residual fields.
    """
    idx = data.predictable_indices
    out = np.empty((len(SEEDS), len(gene_positions)), np.float64)
    for s, seed in enumerate(SEEDS):
        base = load_prediction(data.run, seed, "registered_idw_k12", data.manifest)
        prediction = load_prediction(data.run, seed, MAP, data.manifest)
        truth_residual = data.truth_log[:, idx] - base[:, idx]
        predicted_residual = prediction[:, idx] - base[:, idx]
        out[s] = pearson_columns(
            truth_residual[:, gene_positions], predicted_residual[:, gene_positions]
        )
    return out


def select_example_gene(tables: dict[str, pd.DataFrame], cache: dict) -> Selection:
    """Apply the fixed panel-a rule and return the winning gene."""
    structure = tables["structure"]
    annotation = tables["annotation_genes"]
    criteria: list[dict] = []

    candidates = structure[structure.stage == SELECTION["stage"]].copy()
    criteria.append({
        "step": "restrict to 6-somite evaluations",
        "source_column": "stage",
        "surviving_gene_by_target": int(len(candidates)),
    })

    threshold = float(
        candidates.residual_to_truth_variance_ratio.quantile(
            SELECTION["normalised_residual_variability_percentile"]
        )
    )
    candidates = candidates[candidates.residual_to_truth_variance_ratio >= threshold]
    criteria.append({
        "step": "top 10% normalised residual variability",
        "source_column": "residual_to_truth_variance_ratio",
        "threshold": threshold,
        "surviving_gene_by_target": int(len(candidates)),
    })

    candidates = candidates[
        candidates[f"{MAP}__residual_pearson"] > SELECTION["min_map_residual_pearson"]
    ]
    criteria.append({
        "step": "MAP residual Pearson above 0.5",
        "source_column": f"{MAP}__residual_pearson",
        "threshold": SELECTION["min_map_residual_pearson"],
        "surviving_gene_by_target": int(len(candidates)),
    })

    candidates = candidates[candidates.residual_moran_z > SELECTION["spatial_permutation_null_z"]]
    criteria.append({
        "step": "above 95th percentile of the spatial-permutation null",
        "source_column": "residual_moran_z",
        "threshold": SELECTION["spatial_permutation_null_z"],
        "note": "audit stored a 20-permutation null; z is the normal-approximation percentile",
        "surviving_gene_by_target": int(len(candidates)),
    })

    # Remaining criteria need the measured arrays, so load only the 6-somite
    # targets that still carry candidates.
    keep_rows = []
    for target_key, block in candidates.groupby("target_key"):
        data = cache.setdefault(target_key, load_target(run_for(target_key)))
        positions = {int(g): p for p, g in enumerate(data.predictable_indices)}
        floor = _noise_floor_percentile(
            data,
            SELECTION["counting_noise_null_draws"],
            SELECTION["counting_noise_null_percentile"],
        )
        gene_positions = np.asarray([positions[int(i)] for i in block.gene_index], np.int64)
        per_seed = _per_seed_recovery(data, gene_positions)
        for local, (_, row) in enumerate(block.iterrows()):
            position = gene_positions[local]
            detection = _detection_fraction(data, int(row.gene_index))
            record = dict(row)
            record["counting_noise_null_p95"] = float(floor[position])
            record["above_counting_noise_null_p95"] = bool(
                row.residual_variance > floor[position]
            )
            record["min_residual_pearson_across_registration_seeds"] = float(per_seed[:, local].min())
            record["detection_fraction"] = detection
            keep_rows.append(record)
    if not keep_rows:
        raise RuntimeError("Panel-a selection rule eliminated every candidate")

    frame = pd.DataFrame(keep_rows)
    frame = frame[frame.above_counting_noise_null_p95]
    criteria.append({
        "step": "above 95th percentile of the counting-noise null",
        "source_column": "residual_variance vs counting_noise_variance",
        "note": (
            f"null built from {SELECTION['counting_noise_null_draws']} draws of the "
            "audit's counting_noise_variance estimator"
        ),
        "surviving_gene_by_target": int(len(frame)),
    })

    frame = frame[
        frame.min_residual_pearson_across_registration_seeds
        > SELECTION["registration_perturbation_min_residual_pearson"]
    ]
    criteria.append({
        "step": "stable under registration perturbation",
        "source_column": "min residual Pearson across registration seeds 42/43/44",
        "threshold": SELECTION["registration_perturbation_min_residual_pearson"],
        "surviving_gene_by_target": int(len(frame)),
    })

    frame = frame[
        (frame.detection_fraction >= SELECTION["min_detection_fraction"])
        & (frame.detection_fraction <= SELECTION["max_detection_fraction"])
    ]
    criteria.append({
        "step": "adequate detection support",
        "source_column": "share of target cells with non-zero measured count",
        "threshold": [SELECTION["min_detection_fraction"], SELECTION["max_detection_fraction"]],
        "surviving_gene_by_target": int(len(frame)),
    })
    if frame.empty:
        raise RuntimeError("Panel-a selection rule eliminated every candidate")

    tissue = annotation[annotation.annotation == "tissue"][
        ["target_key", "gene", "truth_residual_eta_squared"]
    ].rename(columns={"truth_residual_eta_squared": "tissue_eta_squared"})
    frame = frame.merge(tissue, on=["target_key", "gene"], how="inner")
    criteria.append({
        "step": "tie-break on highest tissue-scale annotation eta squared",
        "source_column": "truth_residual_eta_squared (annotation == tissue)",
        "surviving_gene_by_target": int(len(frame)),
    })
    if frame.empty:
        raise RuntimeError("No candidate has a tissue-scale annotation score")

    ordered = frame.sort_values("tissue_eta_squared", ascending=False).reset_index(drop=True)
    winner = ordered.iloc[0]
    data = cache[winner.target_key]

    keys = [
        "residual_to_truth_variance_ratio", "residual_variance", "counting_noise_variance",
        "counting_noise_null_p95", "structured_fraction_of_residual", "residual_moran_i",
        "residual_moran_z", f"{MAP}__residual_pearson", f"{ANCHOR}__residual_pearson",
        "min_residual_pearson_across_registration_seeds", "detection_fraction",
        "tissue_eta_squared",
    ]
    return Selection(
        target_key=str(winner.target_key),
        stage=str(winner.stage),
        direction=direction_of(str(winner.target_key)),
        gene=str(winner.gene),
        gene_index=int(winner.gene_index),
        n_cells=int(data.n_cells),
        statistics={k: float(winner[k]) for k in keys},
        survivors=ordered.head(10)[
            ["target_key", "gene", "tissue_eta_squared", f"{MAP}__residual_pearson"]
        ].to_dict("records"),
        runner_up=(
            ordered.iloc[1][["target_key", "gene", "tissue_eta_squared"]].to_dict()
            if len(ordered) > 1 else None
        ),
        criteria_applied=criteria,
    )


# ----------------------------------------------------------------------------
# Panel drawing helpers


def _style_axes(ax, title: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=BASE_FONT, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT, pad=3)


def _panel_label(ax, letter: str, dx: float = -0.20, dy: float = 1.24) -> None:
    """Place the panel letter clear of the panel title."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=PANEL_FONT,
            fontweight="bold", va="top", ha="left")


def _stage_direction_legend(ax, loc="lower right", ncol=1) -> None:
    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3.2,
                   color=STAGE_COLOURS[s], label=s)
        for s in STAGE_ORDER
    ]
    handles += [
        plt.Line2D([], [], marker=m, linestyle="none", markersize=3.2,
                   color="#444444", markerfacecolor="none", label=DIRECTION_LABELS[d])
        for d, m in DIRECTION_MARKERS.items()
    ]
    legend = ax.legend(handles=handles, fontsize=BASE_FONT - 0.5, loc=loc, ncol=ncol,
                       frameon=False, handletextpad=0.4, borderpad=0.2,
                       labelspacing=0.25, columnspacing=0.7)
    return legend


# ----------------------------------------------------------------------------
# Panels


def panel_a(axes, selection: Selection, cache: dict, source_dir: Path) -> pd.DataFrame:
    """Measured target, registered IDW and measured residual for one gene."""
    data = cache[selection.target_key]
    idx = selection.gene_index

    bases = [
        load_prediction(data.run, seed, "registered_idw_k12", data.manifest)[:, idx]
        for seed in SEEDS
    ]
    base = np.mean(bases, axis=0)
    measured = data.truth_log[:, idx].astype(np.float64)
    residual = measured - base

    # Project onto the two widest canonical axes, longest one horizontal, so the
    # specimen fills the panel instead of rendering as a thin sliver.
    coordinates = data.coordinates
    extents = coordinates.max(0) - coordinates.min(0)
    horizontal, vertical = np.argsort(extents)[::-1][:2]
    x, y = coordinates[:, horizontal], coordinates[:, vertical]

    # One identical expression scale for measured and registered IDW.
    vmax = float(np.quantile(np.concatenate([measured, base]), 0.995))
    vmin = 0.0
    rlim = float(np.quantile(np.abs(residual), 0.99))

    # Box the axes to the specimen aspect so the colour bars sit flush against
    # the map instead of across empty padding.
    pad = 0.03 * float(np.ptp(x))
    xlim = (x.min() - pad, x.max() + pad)
    ylim = (y.min() - pad, y.max() + pad)
    box_aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

    fields = [
        ("Measured target", measured, "viridis", dict(vmin=vmin, vmax=vmax), False),
        ("Registered IDW", base, "viridis", dict(vmin=vmin, vmax=vmax), True),
        ("Measured residual", residual, "RdBu_r", dict(vmin=-rlim, vmax=rlim), True),
    ]
    for ax, (title, values, cmap, limits, show_bar) in zip(axes, fields):
        order = np.argsort(np.abs(values - np.median(values)))
        scatter = ax.scatter(
            x[order], y[order], c=values[order], s=1.1, cmap=cmap, linewidths=0,
            rasterized=True, **limits,
        )
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_box_aspect(box_aspect)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=TITLE_FONT, pad=2)
        if not show_bar:
            continue
        bar = plt.colorbar(scatter, ax=ax, fraction=0.038, pad=0.02)
        bar.ax.tick_params(labelsize=BASE_FONT - 1, length=2, width=0.5)
        bar.outline.set_linewidth(0.5)
        bar.set_label(
            "log1p normalised" if cmap == "viridis" else "residual",
            fontsize=BASE_FONT - 1,
        )

    axes[1].text(
        0.5, -0.34,
        f"{selection.stage}, {DIRECTION_LABELS[selection.direction]}, "
        f"{selection.gene}, n = {selection.n_cells:,} cells",
        transform=axes[1].transAxes, fontsize=BASE_FONT, va="top", ha="center",
    )
    frame = pd.DataFrame({
        "cell_index": np.arange(len(x)),
        "plot_x": x, "plot_y": y,
        "canonical_x": coordinates[:, 0], "canonical_y": coordinates[:, 1],
        "canonical_z": coordinates[:, 2],
        "measured_target_log1p": measured,
        "registered_idw_log1p": base,
        "measured_residual": residual,
    })
    frame.to_csv(source_dir / "panel_a_residual_maps.csv", index=False)
    return frame


def panel_b(ax, structure: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """Observed residual variance against the Poisson counting-noise floor."""
    for stage in STAGE_ORDER:
        for direction, marker in DIRECTION_MARKERS.items():
            block = structure[
                (structure.stage == stage)
                & (structure.target_key.map(direction_of) == direction)
            ]
            if block.empty:
                continue
            ax.scatter(
                block.counting_noise_variance, block.residual_variance,
                s=2.0, marker=marker, linewidths=0.25, alpha=0.55,
                facecolors="none", edgecolors=STAGE_COLOURS[stage], rasterized=True,
            )
    lo = float(min(structure.counting_noise_variance.min(), structure.residual_variance.min()))
    hi = float(max(structure.counting_noise_variance.max(), structure.residual_variance.max()))
    ax.plot([lo, hi], [lo, hi], color="#444444", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Poisson counting-noise floor", fontsize=LABEL_FONT)
    ax.set_ylabel("Observed residual variance", fontsize=LABEL_FONT)
    _style_axes(ax, "Exceeds counting noise")

    median_fraction = float(structure.structured_fraction_of_residual.median())
    above = float((structure.residual_variance > structure.counting_noise_variance).mean())
    # Points sit above the diagonal, so the free space is the lower right.
    ax.text(
        0.02, 0.98,
        f"{median_fraction * 100:.1f}% of residual variance above\n"
        f"Poisson floor (median gene)\n"
        f"{above * 100:.0f}% of points above y = x\nn = {len(structure):,}",
        transform=ax.transAxes, fontsize=BASE_FONT - 1, va="top", ha="left",
    )
    _stage_direction_legend(ax, loc="lower right")
    frame = structure[[
        "target_key", "stage", "gene", "counting_noise_variance", "residual_variance",
        "structured_fraction_of_residual",
    ]].copy()
    frame.to_csv(source_dir / "panel_b_variance_vs_noise.csv", index=False)
    return frame


def panel_c(ax, structure: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """Observed residual Moran's I minus the spatial-permutation null."""
    keys = sorted(structure.target_key.unique(), key=lambda k: (STAGE_ORDER.index(
        structure[structure.target_key == k].stage.iloc[0]), k))
    rows = []
    for position, key in enumerate(keys):
        block = structure[structure.target_key == key]
        stage = block.stage.iloc[0]
        delta = (block.residual_moran_i - block.residual_moran_null_mean).to_numpy()
        parts = ax.violinplot([delta], positions=[position], widths=0.72,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(STAGE_COLOURS[stage])
            body.set_alpha(0.35)
            body.set_linewidth(0)
        ax.scatter([position], [np.median(delta)], s=9,
                   marker=DIRECTION_MARKERS[direction_of(key)],
                   color=STAGE_COLOURS[stage], zorder=3, linewidths=0)
        rows.append({
            "target_key": key, "stage": stage, "direction": direction_of(key),
            "n_genes": int(len(delta)),
            "median_observed_moran_i": float(block.residual_moran_i.median()),
            "median_permutation_null_moran_i": float(block.residual_moran_null_mean.median()),
            "median_observed_minus_null": float(np.median(delta)),
            "median_permutation_z": float(block.residual_moran_z.median()),
        })
    ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle="--")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_to_", "→").replace("_", " ") for k in keys],
                       rotation=45, ha="right", fontsize=BASE_FONT - 0.5)
    ax.set_ylabel("Moran's I $-$ null", fontsize=LABEL_FONT)
    _style_axes(ax, "Spatially organised")
    frame = pd.DataFrame(rows)
    smallest = frame.median_permutation_z.min()
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    ax.text(0.02, 0.98,
            f"6 evaluations; genes pooled within each\n"
            f"lowest median permutation z = {smallest:.0f}",
            transform=ax.transAxes, fontsize=BASE_FONT - 1, va="top", ha="left")
    frame.to_csv(source_dir / "panel_c_moran_vs_null.csv", index=False)
    return frame


def panel_d(ax, annotation: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """Annotation-associated eta squared against the permuted-label null."""
    rows = []
    for position, column in enumerate(ANNOTATION_ORDER):
        block = annotation[annotation.annotation == column]
        if block.empty:
            continue
        for key, group in block.groupby("target_key"):
            stage = group.stage.iloc[0]
            marker = DIRECTION_MARKERS[direction_of(key)]
            observed = float(group.truth_residual_eta_squared.median())
            null = float(group.permuted_label_eta_squared.median())
            ax.scatter([position - 0.17], [observed], s=11, marker=marker,
                       color=STAGE_COLOURS[stage], linewidths=0, zorder=3)
            ax.scatter([position + 0.17], [null], s=11, marker=marker,
                       facecolors="none", edgecolors=STAGE_COLOURS[stage],
                       linewidths=0.7, zorder=3)
            rows.append({
                "annotation": column, "target_key": key, "stage": stage,
                "direction": direction_of(key), "n_classes": int(group.n_classes.iloc[0]),
                "n_genes": int(len(group)),
                "median_observed_eta_squared": observed,
                "median_permuted_label_eta_squared": null,
            })
    ax.set_yscale("log")
    ax.set_xticks(range(len(ANNOTATION_ORDER)))
    ax.set_xticklabels([ANNOTATION_LABELS[c] for c in ANNOTATION_ORDER],
                       rotation=30, ha="right", fontsize=BASE_FONT)
    ax.set_ylabel("Annotation $\\eta^2$", fontsize=LABEL_FONT)
    _style_axes(ax, "Aligned with annotation")
    ratio = (annotation.truth_residual_eta_squared.median()
             / annotation.permuted_label_eta_squared.median())
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 12.0)
    ax.text(0.02, 0.98,
            f"filled observed, open permuted labels\noverall observed/null = {ratio:.0f}x",
            transform=ax.transAxes, fontsize=BASE_FONT - 1, va="top", ha="left")
    frame = pd.DataFrame(rows)
    frame.to_csv(source_dir / "panel_d_annotation_eta_squared.csv", index=False)
    return frame


def panel_e(ax, annotation: pd.DataFrame, selection: Selection, cache: dict,
            source_dir: Path) -> pd.DataFrame:
    """Per-gene compartment-profile agreement, with an inset for the example gene."""
    values = annotation[f"{MAP}__class_profile_pearson"].dropna().to_numpy()
    ax.hist(values, bins=45, color="#0072B2", alpha=0.75, linewidth=0)
    median = float(np.median(values))
    ax.axvline(median, color="#C83E4D", linewidth=1.0)
    ax.set_xlabel("Compartment-profile Pearson r\n(measured vs MAP-predicted)",
                  fontsize=LABEL_FONT)
    ax.set_ylabel("Gene $\\times$ annotation", fontsize=LABEL_FONT)
    _style_axes(ax, "MAP recovers compartment profiles")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.30)
    # Histogram mass sits at high r, so the free space is the upper left.
    ax.text(0.02, 0.98,
            f"median r = {median:.3f}   n = {len(values):,}\n"
            f"annotations were not used in model fitting",
            transform=ax.transAxes, fontsize=BASE_FONT - 1, va="top", ha="left")

    # Inset: standardised tissue-scale compartment residuals for the panel-a gene.
    data = cache[selection.target_key]
    labels = data.annotations.get("tissue")
    inset_rows = []
    if labels is not None:
        names, codes, _ = annotation_module._design(labels)
        position = int(np.flatnonzero(data.predictable_indices == selection.gene_index)[0])
        bases, predictions = [], []
        for seed in SEEDS:
            base = load_prediction(data.run, seed, "registered_idw_k12", data.manifest)
            prediction = load_prediction(data.run, seed, MAP, data.manifest)
            bases.append(base[:, selection.gene_index])
            predictions.append(prediction[:, selection.gene_index])
        base = np.mean(bases, axis=0)
        measured_residual = data.truth_log[:, selection.gene_index] - base
        predicted_residual = np.mean(predictions, axis=0) - base
        observed = annotation_module.class_means(measured_residual[:, None], codes, len(names))[:, 0]
        predicted = annotation_module.class_means(predicted_residual[:, None], codes, len(names))[:, 0]
        zo = (observed - observed.mean()) / max(observed.std(), 1e-12)
        zp = (predicted - predicted.mean()) / max(predicted.std(), 1e-12)
        inset = ax.inset_axes([0.11, 0.34, 0.27, 0.38])
        inset.scatter(zo, zp, s=5, color=STAGE_COLOURS[selection.stage], linewidths=0)
        span = [min(zo.min(), zp.min()), max(zo.max(), zp.max())]
        inset.plot(span, span, color="#444444", linewidth=0.6, linestyle="--")
        inset.set_xlabel("measured (z)", fontsize=BASE_FONT - 1)
        inset.set_ylabel("MAP (z)", fontsize=BASE_FONT - 1)
        inset.tick_params(labelsize=BASE_FONT - 1.5, length=2, width=0.5)
        inset.set_title(f"{selection.gene}, tissue classes (n = {len(names)})",
                        fontsize=BASE_FONT - 1, pad=2)
        for spine in ("top", "right"):
            inset.spines[spine].set_visible(False)
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)
        inset_rows = [
            {"tissue_class": n, "n_cells": int((codes == i).sum()),
             "measured_compartment_residual": float(observed[i]),
             "map_compartment_residual": float(predicted[i]),
             "measured_z": float(zo[i]), "map_z": float(zp[i])}
            for i, n in enumerate(names)
        ]

    frame = annotation[[
        "target_key", "stage", "annotation", "gene", f"{MAP}__class_profile_pearson",
        f"{ANCHOR}__class_profile_pearson",
    ]].copy()
    frame.to_csv(source_dir / "panel_e_compartment_profile_pearson.csv", index=False)
    if inset_rows:
        pd.DataFrame(inset_rows).to_csv(
            source_dir / "panel_e_inset_example_gene_profile.csv", index=False
        )
    return frame


def panel_f(ax, annotation: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """Six-somite eta squared across annotation granularity."""
    block = annotation[annotation.stage == "6-somite"]
    rows = []
    for position, column in enumerate(ANNOTATION_ORDER):
        sub = block[block.annotation == column]
        if sub.empty:
            continue
        for offset, (key, group) in zip((-0.13, 0.13), sub.groupby("target_key")):
            observed = float(group.truth_residual_eta_squared.median())
            ax.scatter([position + offset], [observed], s=16,
                       marker=DIRECTION_MARKERS[direction_of(key)],
                       color=STAGE_COLOURS["6-somite"], linewidths=0, zorder=3)
            rows.append({
                "annotation": column, "target_key": key, "direction": direction_of(key),
                "n_classes": int(group.n_classes.iloc[0]), "n_genes": int(len(group)),
                "median_observed_eta_squared": observed,
                "median_permuted_label_eta_squared": float(
                    group.permuted_label_eta_squared.median()
                ),
            })
        pooled = float(sub.truth_residual_eta_squared.median())
        ax.hlines(pooled, position - 0.28, position + 0.28, color="#444444",
                  linewidth=0.9, zorder=2)
        if column in ("germlayer", "tissue"):
            ax.annotate(f"{pooled:.3f}", (position, pooled), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=BASE_FONT - 0.5,
                        fontweight="bold")
    ax.set_xticks(range(len(ANNOTATION_ORDER)))
    ax.set_xticklabels([ANNOTATION_LABELS[c] for c in ANNOTATION_ORDER],
                       rotation=30, ha="right", fontsize=BASE_FONT)
    ax.set_ylabel("Annotation-associated $\\eta^2$", fontsize=LABEL_FONT)
    _style_axes(ax, "Finer than germ layer")
    handles = [
        plt.Line2D([], [], marker=m, linestyle="none", markersize=3.2,
                   color=STAGE_COLOURS["6-somite"], label=DIRECTION_LABELS[d])
        for d, m in DIRECTION_MARKERS.items()
    ]
    handles.append(plt.Line2D([], [], color="#444444", linewidth=0.9, label="pooled median"))
    ax.legend(handles=handles, fontsize=BASE_FONT - 0.5, loc="upper left", frameon=False,
              handletextpad=0.4, borderpad=0.2, labelspacing=0.25)
    ax.text(0.97, 0.03, "reciprocal atlas constructions,\nnot independent replicates",
            transform=ax.transAxes, fontsize=BASE_FONT - 1, va="bottom", ha="right",
            color="#555555")
    frame = pd.DataFrame(rows)
    frame.to_csv(source_dir / "panel_f_6somite_eta_by_granularity.csv", index=False)
    return frame


def panel_g(axes, depth: pd.DataFrame, registration: pd.DataFrame,
            source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Depth adjustment and registration-stability strata."""
    left, right = axes

    block = depth[depth.method == MAP].copy()
    block["direction"] = block.target_key.map(direction_of)
    for _, row in block.iterrows():
        colour = STAGE_COLOURS[row.stage]
        left.plot([0, 1], [row.median_residual_pearson,
                           row.median_depth_partialled_residual_pearson],
                  color=colour, linewidth=0.7, alpha=0.8, zorder=2)
        left.scatter([0, 1], [row.median_residual_pearson,
                              row.median_depth_partialled_residual_pearson],
                     s=11, marker=DIRECTION_MARKERS[row.direction], color=colour,
                     linewidths=0, zorder=3)
    left.set_xticks([0, 1])
    left.set_xticklabels(["raw", "depth\nadjusted"], fontsize=BASE_FONT)
    left.set_xlim(-0.35, 1.35)
    left.set_ylabel("Median residual Pearson r", fontsize=LABEL_FONT)
    _style_axes(left, "Depth adjustment")
    delta = float(block.median_change_after_depth_removal.median())
    lo, hi = left.get_ylim()
    left.set_ylim(lo, hi + 0.26 * (hi - lo))
    left.text(0.5, 0.98, f"$\\Delta r$ = {delta:.3f}", transform=left.transAxes,
              fontsize=BASE_FONT, ha="center", va="top")

    rows = []
    for key, group in registration.groupby("target_key"):
        group = group.sort_values("instability_quintile")
        stage = group.stage.iloc[0]
        right.plot(group.instability_quintile,
                   group[f"{MAP}__median_residual_pearson"],
                   color=STAGE_COLOURS[stage], linewidth=0.7, alpha=0.8,
                   marker=DIRECTION_MARKERS[direction_of(key)], markersize=2.8,
                   markeredgewidth=0)
        for _, row in group.iterrows():
            rows.append({
                "target_key": key, "stage": stage, "direction": direction_of(key),
                "instability_quintile": int(row.instability_quintile),
                "n_cells": int(row.n_cells),
                "map_median_residual_pearson": float(row[f"{MAP}__median_residual_pearson"]),
            })
    right.set_xticks([1, 2, 3, 4, 5])
    right.set_xlabel("Registration-instability quintile", fontsize=LABEL_FONT)
    right.set_ylabel("Median residual Pearson r", fontsize=LABEL_FONT)
    _style_axes(right, "Registration stability")
    gradient = (registration.sort_values("instability_quintile")
                .groupby("target_key")[f"{MAP}__median_residual_pearson"]
                .agg(lambda s: s.iloc[-1] - s.iloc[0]))
    lo, hi = right.get_ylim()
    right.set_ylim(lo, hi + 0.30 * (hi - lo))
    right.text(0.02, 0.98,
               f"Q5 - Q1 = {gradient.median():.3f}\n(modest, not absent)",
               transform=right.transAxes, fontsize=BASE_FONT - 1, va="top", ha="left")

    depth_frame = block[[
        "target_key", "stage", "direction", "median_residual_pearson",
        "median_depth_partialled_residual_pearson", "median_change_after_depth_removal",
        "n_genes",
    ]]
    registration_frame = pd.DataFrame(rows)
    depth_frame.to_csv(source_dir / "panel_g1_depth_adjustment.csv", index=False)
    registration_frame.to_csv(source_dir / "panel_g2_registration_strata.csv", index=False)
    return depth_frame, registration_frame


def panel_h(ax, anchor: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    """Gene-level recovery against molecular similarity to the anchor panel."""
    x = anchor.max_source_correlation_to_anchor_panel.to_numpy()
    y = anchor[f"{MAP}__residual_pearson"].to_numpy()
    hexes = ax.hexbin(x, y, gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
    bar = plt.colorbar(hexes, ax=ax, fraction=0.040, pad=0.015)
    bar.ax.tick_params(labelsize=BASE_FONT - 0.5, length=2, width=0.5)
    bar.outline.set_linewidth(0.5)
    bar.set_label("non-anchor genes", fontsize=BASE_FONT - 0.5)

    frame = anchor.copy()
    frame["quartile"] = pd.qcut(frame.max_source_correlation_to_anchor_panel, 4,
                                labels=["Q1", "Q2", "Q3", "Q4"])
    summary = frame.groupby("quartile", observed=True).agg(
        centre=("max_source_correlation_to_anchor_panel", "median"),
        recovery=(f"{MAP}__residual_pearson", "median"),
        n=("gene", "size"),
    ).reset_index()
    ax.scatter(summary.centre, summary.recovery, s=20, marker="D", color="#C83E4D",
               zorder=4, linewidths=0, label="quartile median")
    ax.set_xlabel("Maximum source correlation to anchor panel", fontsize=LABEL_FONT)
    ax.set_ylabel("MAP residual Pearson r", fontsize=LABEL_FONT)
    _style_axes(ax, "Recovery beyond anchor similarity")

    correlation = float(np.corrcoef(x, y)[0, 1])
    q1 = float(summary.loc[summary.quartile == "Q1", "recovery"].iloc[0])
    ax.text(0.97, 0.28,
            f"r = {correlation:.3f}\nQ1 median recovery = {q1:.3f}\n"
            f"n = {len(frame):,} non-anchor gene x target",
            transform=ax.transAxes, fontsize=BASE_FONT - 1, va="top", ha="right")
    ax.legend(fontsize=BASE_FONT - 1, loc="upper left", frameon=False,
              handletextpad=0.4, borderpad=0.2)

    out = frame[[
        "target_key", "stage", "gene", "max_source_correlation_to_anchor_panel",
        f"{MAP}__residual_pearson", "quartile",
    ]].copy()
    out["is_anchor_gene"] = False
    out.to_csv(source_dir / "panel_h_anchor_similarity.csv", index=False)
    summary.to_csv(source_dir / "panel_h_quartile_summary.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# Assembly


def build_figure(tables: dict[str, pd.DataFrame], selection: Selection, cache: dict,
                 source_dir: Path) -> tuple[plt.Figure, dict]:
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "font.family": "sans-serif",
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # vector text
        "svg.fonttype": "none",
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM))
    grid = figure.add_gridspec(
        4, 6, height_ratios=[0.55, 1.0, 1.0, 1.0],
        hspace=0.85, wspace=1.35, left=0.085, right=0.955, top=0.935, bottom=0.055,
    )

    axes_a = [figure.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(3)]
    ax_b = figure.add_subplot(grid[1, 0:2])
    ax_c = figure.add_subplot(grid[1, 2:4])
    ax_d = figure.add_subplot(grid[1, 4:6])
    ax_e = figure.add_subplot(grid[2, 0:3])
    ax_f = figure.add_subplot(grid[2, 3:6])
    g_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[3, 0:3], wspace=0.55)
    axes_g = [figure.add_subplot(g_grid[0, 0]), figure.add_subplot(g_grid[0, 1])]
    ax_h = figure.add_subplot(grid[3, 3:6])

    frames = {}
    frames["a"] = panel_a(axes_a, selection, cache, source_dir)
    frames["b"] = panel_b(ax_b, tables["structure"], source_dir)
    frames["c"] = panel_c(ax_c, tables["structure"], source_dir)
    frames["d"] = panel_d(ax_d, tables["annotation_genes"], source_dir)
    frames["e"] = panel_e(ax_e, tables["annotation_genes"], selection, cache, source_dir)
    frames["f"] = panel_f(ax_f, tables["annotation_genes"], source_dir)
    frames["g1"], frames["g2"] = panel_g(
        axes_g, tables["depth"], tables["registration"], source_dir
    )
    frames["h"] = panel_h(ax_h, tables["anchor_propagation"], source_dir)

    _panel_label(axes_a[0], "a", dx=-0.10, dy=1.30)
    for ax, letter in ((ax_b, "b"), (ax_c, "c"), (ax_d, "d"),
                       (axes_g[0], "g")):
        _panel_label(ax, letter)
    # Wider panels need a proportionally smaller horizontal offset.
    for ax, letter in ((ax_e, "e"), (ax_f, "f"), (ax_h, "h")):
        _panel_label(ax, letter, dx=-0.11, dy=1.22)

    figure.suptitle(FIGURE_TITLE, fontsize=TITLE_FONT + 1, y=0.982)
    return figure, frames


def write_caption(path: Path, selection: Selection, tables: dict[str, pd.DataFrame],
                  frames: dict, audit_dir: Path) -> None:
    structure = tables["structure"]
    annotation = tables["annotation_genes"]
    anchor = tables["anchor_propagation"]
    depth = tables["depth"][tables["depth"].method == MAP]
    registration = tables["registration"]
    gradient = (registration.sort_values("instability_quintile")
                .groupby("target_key")[f"{MAP}__median_residual_pearson"]
                .agg(lambda s: s.iloc[-1] - s.iloc[0]))
    ratio = (annotation.truth_residual_eta_squared.median()
             / annotation.permuted_label_eta_squared.median())
    six = annotation[annotation.stage == "6-somite"]

    text = f"""# Figure 5

**{FIGURE_TITLE}**

All statistics are read from the completed residual-landscape audit
(`{audit_dir.name}`). HyperSpatial-MAP was not re-run and no frozen input was
modified. Residuals are defined as measured target expression minus the
registered inverse-distance-weighted (IDW) atlas transfer,
R_true = Y_target - B_registered_IDW, on the log1p depth-normalised scale used by
the frozen protocol, averaged over the three registration seeds.

**a**, Residual decomposition for one measured target volume. Measured target
expression, the registered IDW transfer and the measured residual for
*{selection.gene}* at {selection.stage}, {DIRECTION_LABELS[selection.direction]}
(n = {selection.n_cells:,} cells). Measured and registered panels share one
expression scale; the residual uses a symmetric blue-white-red scale centred at
zero. The displayed gene was chosen by a fixed machine-readable rule recorded in
`Figure_5_gene_selection.json` and was not substituted by hand.

**b**, Observed residual variance against the Poisson counting-noise floor for
every eligible gene x target volume (n = {len(structure):,}; log axes; dashed line
y = x). A median of
{structure.structured_fraction_of_residual.median() * 100:.1f}% of residual
variance lies above the counting-noise floor. Colour denotes developmental stage
and marker denotes the direction of atlas construction.

**c**, Residual spatial autocorrelation. Per-gene Moran's I minus the
cell-permutation null, shown as a distribution over genes within each of the six
evaluations; points give the evaluation median. Genes within an evaluation are not
independent biological replicates and are not treated as such. The lowest median
permutation z across evaluations is
{structure.groupby('target_key').residual_moran_z.median().min():.0f}.

**d**, Annotation-associated eta squared for the measured residual (filled) and
for permuted labels (open), at germ-layer, cell-type, tissue and cluster
granularity. Overall observed-to-null ratio {ratio:.0f}x. Author-supplied
annotations were never read by the frozen pipeline
(`target_annotations_accessed: false`).

**e**, Distribution of per-gene compartment-profile Pearson correlations between
measured and MAP-predicted per-class mean residuals
(median r = {annotation[f'{MAP}__class_profile_pearson'].median():.3f},
n = {annotation[f'{MAP}__class_profile_pearson'].notna().sum():,}). Inset, measured
versus MAP-predicted standardised tissue-class residuals for *{selection.gene}*.
Annotations were not used during model fitting.

**f**, Six-somite annotation-associated eta squared across granularity levels.
Germ-layer eta squared
{six[six.annotation == 'germlayer'].truth_residual_eta_squared.median():.3f}
versus tissue eta squared
{six[six.annotation == 'tissue'].truth_residual_eta_squared.median():.3f}, so the
organisation is finer than germ layer. The two reciprocal atlas constructions are
shown separately and are not independent replicates.

**g**, Confound controls. Left, median residual Pearson before and after removing
per-cell log sequencing depth
(delta r = {depth.median_change_after_depth_removal.median():.3f}). Right, median
residual Pearson across registration-instability quintiles
(Q5 - Q1 = {gradient.median():.3f}). The association with registration stability
is modest rather than absent.

**h**, Gene-level MAP residual recovery against maximum source correlation to the
16-gene anchor panel (r = {np.corrcoef(anchor.max_source_correlation_to_anchor_panel, anchor[f'{MAP}__residual_pearson'])[0, 1]:.3f};
n = {len(anchor):,} non-anchor gene x target). Diamonds give quartile medians.
Recovery is strongest for anchor-related genes but remains substantial in the
lowest-similarity quartile
({frames['h'].groupby('quartile', observed=True)[f'{MAP}__residual_pearson'].median().loc['Q1']:.3f}).
Anchor genes are excluded from the frozen source-defined predictable set by
construction, so they carry no residual-recovery statistic and cannot be shown.

## Interpretation limits

- Cells are not treated as biological replicates and reciprocal directions are not
  independent replicates; they are reciprocal atlas constructions of the same
  specimen pairs.
- Findings describe geometry-corrected molecular variation and its consistency
  under reciprocal atlas construction. No claim is made about canalization,
  population-wide variability or causality.
- Residual variance above the Poisson floor is structured variance, not
  necessarily biological variance; the counting-noise floor uses observed counts
  as the Poisson rate and is therefore conservative.
- This is a retrospective characterisation of frozen outputs and does not
  establish prospective confirmation.
"""
    assert_writable(path)
    path.write_text(_rewrap(text))


def _rewrap(text: str, width: int = 80) -> str:
    """Reflow prose paragraphs; leave headings and list items alone."""
    blocks = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        # "**a**," is bold prose, not a bullet, so bullets need the trailing space.
        is_list = stripped.startswith(("- ", "* ", "1. "))
        if not stripped or is_list or stripped.startswith(("#", "|", "```")):
            blocks.append(block.rstrip())
            continue
        blocks.append(textwrap.fill(
            " ".join(stripped.split()), width=width, break_long_words=False,
            break_on_hyphens=False,
        ))
    return "\n\n".join(blocks) + "\n"


def resolve_audit_dir(argument: str | None) -> Path:
    if argument:
        path = Path(argument)
        if path.exists():
            return path.resolve()
        # Tolerate a truncated timestamp such as ..._20260805_1.
        matches = sorted(OUTPUT_PARENT.glob(f"{path.name}*"))
        if len(matches) == 1:
            return matches[0].resolve()
        raise SystemExit(
            f"Audit directory {argument} not found; candidates: {[m.name for m in matches]}"
        )
    matches = sorted(OUTPUT_PARENT.glob(f"{OUTPUT_PREFIX}_*"))
    if not matches:
        raise SystemExit("No residual-landscape audit directory found")
    return matches[-1].resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default=None,
                        help="Completed residual-landscape audit directory.")
    args = parser.parse_args(argv)

    audit_dir = resolve_audit_dir(args.audit_dir)
    figure_dir = assert_writable(audit_dir / "06_figures")
    source_dir = assert_writable(audit_dir / "07_source_data/figure5")
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(audit_dir)
    cache: dict = {}
    selection = select_example_gene(tables, cache)

    selection_path = figure_dir / "Figure_5_gene_selection.json"
    assert_writable(selection_path)
    selection_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audit_dir": str(audit_dir),
        "rule": SELECTION,
        "criteria_applied": selection.criteria_applied,
        "selected": {
            "target_key": selection.target_key, "stage": selection.stage,
            "direction": selection.direction, "gene": selection.gene,
            "gene_index": selection.gene_index, "n_cells": selection.n_cells,
            "statistics": selection.statistics,
        },
        "runner_up": selection.runner_up,
        "top_candidates": selection.survivors,
        "manual_override": False,
    }, indent=2) + "\n")

    figure, frames = build_figure(tables, selection, cache, source_dir)
    stem = figure_dir / "Figure_5_residual_biology"
    for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 300})):
        assert_writable(stem.with_suffix(suffix))
        figure.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(figure)

    caption_path = figure_dir / "Figure_5_caption.md"
    write_caption(caption_path, selection, tables, frames, audit_dir)

    _print_summary(audit_dir, figure_dir, source_dir, stem, selection, tables, frames)
    return 0


def _print_summary(audit_dir, figure_dir, source_dir, stem, selection, tables, frames):
    structure = tables["structure"]
    annotation = tables["annotation_genes"]
    anchor = tables["anchor_propagation"]

    print("=" * 78)
    print("FIGURE 5 GENERATED")
    print("=" * 78)
    print(f"audit directory : {audit_dir}")
    print()
    print("--- selected panel-a example (machine-readable rule, no manual override)")
    print(f"  gene       : {selection.gene}")
    print(f"  stage      : {selection.stage}")
    print(f"  direction  : {DIRECTION_LABELS[selection.direction]} ({selection.target_key})")
    print(f"  cells      : {selection.n_cells:,}")
    for key, value in selection.statistics.items():
        print(f"    {key:<52s} {value:.4f}")
    if selection.runner_up:
        print(f"  runner-up  : {selection.runner_up}")
    print()
    print("--- audit tables used by each panel")
    usage = {
        "a": "frozen predictions + measured target (drawing only); values in 02_structure",
        "b": "02_structure/RESIDUAL_VARIANCE_DECOMPOSITION.tsv",
        "c": "02_structure/RESIDUAL_VARIANCE_DECOMPOSITION.tsv",
        "d": "03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv",
        "e": "03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv (+ inset from frozen arrays)",
        "f": "03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv (6-somite rows)",
        "g": "04_confounds/DEPTH_PARTIAL_CORRELATION.tsv, 04_confounds/REGISTRATION_INSTABILITY_STRATA.tsv",
        "h": "04_confounds/ANCHOR_PROPAGATION_BY_GENE.tsv",
    }
    for panel, table in usage.items():
        print(f"  {panel}: {table}")
    print()
    print("--- panel sample sizes")
    sizes = {
        "a": f"{selection.n_cells:,} cells, 1 gene, 1 target volume",
        "b": f"{len(frames['b']):,} gene x target points, 6 evaluations",
        "c": f"6 evaluations, {int(frames['c'].n_genes.sum()):,} gene observations pooled",
        "d": f"{len(frames['d'])} evaluation x annotation medians from {len(tables['annotation_genes']):,} gene rows",
        "e": f"{annotation[f'{MAP}__class_profile_pearson'].notna().sum():,} gene x annotation x evaluation",
        "f": f"{len(frames['f'])} evaluation x annotation medians (6-somite only)",
        "g": f"{len(frames['g1'])} evaluations (depth), {len(frames['g2'])} evaluation x quintile rows",
        "h": f"{len(frames['h']):,} non-anchor gene x target points",
    }
    for panel, size in sizes.items():
        print(f"  {panel}: {size}")
    print()
    print("--- outputs")
    for suffix in (".pdf", ".svg", ".png"):
        path = stem.with_suffix(suffix)
        print(f"  {path}  ({path.stat().st_size:,} bytes)")
    print(f"  {figure_dir / 'Figure_5_caption.md'}")
    print(f"  {figure_dir / 'Figure_5_gene_selection.json'}")
    print(f"  source data: {source_dir}")
    for path in sorted(source_dir.glob("*.csv")):
        print(f"    {path.name}  ({path.stat().st_size:,} bytes)")
    print()
    print("--- values that differ from the rounded manuscript summaries")
    above = float((structure.residual_variance > structure.counting_noise_variance).mean())
    median_fraction = float(structure.structured_fraction_of_residual.median())
    print(
        f"  '80.2% above Poisson floor' is the MEDIAN SHARE OF RESIDUAL VARIANCE above the\n"
        f"    counting-noise floor ({median_fraction:.4f}), not a proportion of genes.\n"
        f"    {above * 100:.0f}% of gene x target points lie above y = x in panel b, so panel b\n"
        f"    annotates both quantities with distinct wording."
    )
    ratio = (annotation.truth_residual_eta_squared.median()
             / annotation.permuted_label_eta_squared.median())
    print(f"  annotation observed/null ratio: exact {ratio:.2f}x (manuscript rounds to 161x)")
    print(f"  compartment-profile median r: exact "
          f"{annotation[f'{MAP}__class_profile_pearson'].median():.6f} (manuscript 0.772)")
    depth_delta = tables["depth"][tables["depth"].method == MAP].median_change_after_depth_removal.median()
    print(f"  depth delta r: exact {depth_delta:.6f} (manuscript -0.014)")
    gradient = (tables["registration"].sort_values("instability_quintile")
                .groupby("target_key")[f"{MAP}__median_residual_pearson"]
                .agg(lambda s: s.iloc[-1] - s.iloc[0]).median())
    print(f"  registration Q5-Q1: exact {gradient:.6f} (manuscript 0.048)")
    correlation = float(np.corrcoef(
        anchor.max_source_correlation_to_anchor_panel, anchor[f"{MAP}__residual_pearson"]
    )[0, 1])
    print(f"  anchor-similarity correlation: exact {correlation:.6f} (manuscript 0.769)")
    q1 = frames["h"].groupby("quartile", observed=True)[f"{MAP}__residual_pearson"].median().loc["Q1"]
    print(f"  Q1 median recovery: exact {q1:.6f} (manuscript 0.397)")
    print(
        "  panel h shows no anchor genes: the frozen predictable set excludes all 16 anchors\n"
        "    by construction, so no residual-recovery statistic exists for them."
    )
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
