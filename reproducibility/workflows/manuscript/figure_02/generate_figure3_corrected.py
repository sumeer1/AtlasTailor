#!/usr/bin/env python3
"""Presentation-only correction of frozen HyperSpatial-MAP Figure 3.

The scientific inputs, projection, layout, and point coordinates are inherited
from summarize_hyperspatial_map_cross_stage_frozen_20260803.py.  This script
only adds the requested labels and shared colour scales.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import summarize_hyperspatial_map_cross_stage_frozen_20260803 as frozen


OUT = Path("results_v10/hyperspatial_map_figure3_correction_20260804_164109")
FIGURE_STEM = OUT / "figure/Figure_3_specimen_specific_residual_recovery_corrected"
GENE = "wnt11f2"
DISPLAY_RUNS = [frozen.RUNS[0], frozen.RUNS[2], frozen.RUNS[4]]
TRANSFORMATION = "log1p(counts normalized to source median total molecular count)"
FROZEN_RESIDUAL_PEARSON = {
    "50p_E1_to_E2": 0.260810,
    "75p_E1_to_E2": 0.320200,
    "6s_E1_to_E2": 0.590460,
}


def load_row(run: dict) -> dict:
    manifest = json.loads((run["path"] / "prediction_manifest.json").read_text())
    source_counts, _, genes = frozen.measured_counts(run["source_file"])
    source_depth = float(np.median(np.maximum(source_counts.sum(1), 1)))
    target_counts, coords, _ = frozen.measured_counts(run["target_file"])
    coords = frozen.canonicalize(coords)[0]
    truth = np.log1p(frozen.normalize_counts(target_counts, source_depth))
    gene_index = int(np.flatnonzero(genes == GENE)[0])
    paths = manifest["methods"]["42"]["predictions"]
    baseline = np.asarray(np.load(paths[frozen.IDW]["path"], mmap_mode="r")[:, gene_index])
    prediction = np.asarray(np.load(paths[frozen.MAP]["path"], mmap_mode="r")[:, gene_index])
    measured = truth[:, gene_index]
    return {
        "run": run,
        "coords": coords,
        "measured": measured,
        "registered_idw": baseline,
        "true_residual": measured - baseline,
        "predicted_residual": prediction - baseline,
        "hyperspatial_map": prediction,
        "absolute_error": np.abs(measured - prediction),
        "source_depth": source_depth,
        "gene_index": gene_index,
        "prediction_manifest": run["path"] / "prediction_manifest.json",
        "map_prediction": Path(paths[frozen.MAP]["path"]),
        "idw_prediction": Path(paths[frozen.IDW]["path"]),
    }


def add_inset_colorbar(ax, mappable, label: str, above: bool = False) -> None:
    cax = inset_axes(
        ax,
        width="76%",
        height="4%",
        loc="lower center",
        bbox_to_anchor=(0, 1.10 if above else -0.13, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=4, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label(label, fontsize=4.4, labelpad=1)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    rows = [load_row(run) for run in DISPLAY_RUNS]

    # One residual and one absolute-error scale across every displayed stage.
    residual_limit = float(np.quantile(np.abs(np.concatenate([
        item["true_residual"] for item in rows
    ] + [item["predicted_residual"] for item in rows])), 0.98))
    error_limit = float(np.quantile(np.concatenate([
        item["absolute_error"] for item in rows
    ]), 0.99))

    fig, axes = plt.subplots(3, 6, figsize=(7.2, 5.5), constrained_layout=True)
    titles = [
        "Measured", "Registered IDW", "True residual", "Predicted residual",
        "HyperSpatial-MAP", "Absolute error",
    ]
    source_rows = []

    for row_index, item in enumerate(rows):
        run = item["run"]
        values = [
            item["measured"], item["registered_idw"], item["true_residual"],
            item["predicted_residual"], item["hyperspatial_map"], item["absolute_error"],
        ]
        expression_limit = float(np.quantile(np.concatenate([
            item["measured"], item["registered_idw"], item["hyperspatial_map"]
        ]), 0.99))
        artists = []
        for col_index, (value, title) in enumerate(zip(values, titles)):
            if col_index in (2, 3):
                cmap = "RdBu_r"
                limits = {"vmin": -residual_limit, "vmax": residual_limit}
            elif col_index == 5:
                cmap = "viridis"
                limits = {"vmin": 0, "vmax": error_limit}
            else:
                cmap = "viridis"
                limits = {"vmin": 0, "vmax": expression_limit}
            artist = axes[row_index, col_index].scatter(
                item["coords"][:, 0], item["coords"][:, 1], c=value,
                s=0.35, cmap=cmap, rasterized=True, **limits,
            )
            artists.append(artist)
            axes[row_index, col_index].set_aspect("equal")
            axes[row_index, col_index].axis("off")
            if row_index == 0:
                axes[row_index, col_index].set_title(title, fontsize=7)

        direction = f"{run['source']}→{run['target']}"
        axes[row_index, 0].text(
            -0.12, 0.5, f"{run['stage']}\n{direction}", rotation=90,
            va="center", ha="right", transform=axes[row_index, 0].transAxes,
            fontsize=7,
        )
        key = f"{run['tag']}_{run['source']}_to_{run['target']}"
        axes[row_index, 3].text(
            0.96, 0.04, f"residual r = {FROZEN_RESIDUAL_PEARSON[key]:.3f}",
            transform=axes[row_index, 3].transAxes, ha="right", va="bottom",
            fontsize=5.2, color="#222222",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

        # Row-specific shared expression scale; residual and absolute-error
        # limits are identical across all rows.
        add_inset_colorbar(
            axes[row_index, 4], artists[4],
            "log1p(source-depth-normalized counts)",
        )
        if row_index == 1:
            add_inset_colorbar(axes[row_index, 3], artists[3], "residual (log1p units)")
            add_inset_colorbar(axes[row_index, 5], artists[5], "absolute error (log1p units)")

        for cell_index in range(item["coords"].shape[0]):
            source_rows.append({
                "stage": run["stage"],
                "direction": direction,
                "gene": GENE,
                "gene_index": item["gene_index"],
                "cell_index": cell_index,
                "x": item["coords"][cell_index, 0],
                "y": item["coords"][cell_index, 1],
                "z": item["coords"][cell_index, 2],
                "measured": item["measured"][cell_index],
                "registered_idw": item["registered_idw"][cell_index],
                "true_residual": item["true_residual"][cell_index],
                "predicted_residual": item["predicted_residual"][cell_index],
                "hyperspatial_map": item["hyperspatial_map"][cell_index],
                "absolute_error": item["absolute_error"][cell_index],
                "expression_vmin": 0.0,
                "expression_vmax": expression_limit,
                "residual_vmin": -residual_limit,
                "residual_vmax": residual_limit,
                "absolute_error_vmin": 0.0,
                "absolute_error_vmax": error_limit,
                "residual_spatial_pearson": FROZEN_RESIDUAL_PEARSON[key],
                "expression_transformation": TRANSFORMATION,
                "source_depth": item["source_depth"],
                "source_file": str(run["source_file"]),
                "target_file": str(run["target_file"]),
                "map_prediction_file": str(item["map_prediction"]),
                "idw_prediction_file": str(item["idw_prediction"]),
            })

    axes[0, 0].text(-0.25, 1.16, "a", fontweight="bold", transform=axes[0, 0].transAxes)
    fig.suptitle("wnt11f2", fontsize=11, fontweight="bold")
    fig.text(
        0.5, -0.095,
        "wnt11f2 was predeclared using source-only criteria and passed the spatial predictability guard at all displayed stages.",
        ha="center", fontsize=6.3,
    )

    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    source_path = OUT / "source_data/Figure_3_source_data.tsv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(source_path, sep="\t", index=False)

    report = OUT / "FIGURE_3_CORRECTION_REPORT.md"
    report.write_text(
        "# Figure 3 correction report\n\n"
        "- Status: presentation-only correction from frozen predictions.\n"
        "- Authoritative script: `summarize_hyperspatial_map_cross_stage_frozen_20260803.py` (`figure3`).\n"
        "- Authoritative original: `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/08_figures_main/Figure_3_specimen_specific_residual_recovery.png`.\n"
        "- Gene: `wnt11f2`, predeclared using source-only criteria.\n"
        "- Rows: 50% epiboly E1→E2; 75% epiboly E1→E2; 6-somite E1→E2.\n"
        f"- Expression scale: `{TRANSFORMATION}`; shared across Measured, Registered IDW and HyperSpatial-MAP within each row.\n"
        "- Residual scale: one symmetric scale shared by True residual and Predicted residual across all rows.\n"
        "- Absolute-error scale: one shared scale across all rows.\n"
        "- Residual correlations are copied from the frozen machine-readable per-gene metric table.\n"
        "- No predictions, coordinates, or scientific results were changed.\n"
    )


if __name__ == "__main__":
    main()
