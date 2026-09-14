#!/usr/bin/env python3
"""Add frozen Spateo panels to S1 without altering the authoritative S1 directory."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "results_v10/supplement_nature_methods_20260809/01_registration_qc"
SPATEO = ROOT / "PI_revision/02A_alignment_benchmark_20260816_182905"

CASES = [
    "50p_E1_to_E2", "50p_E2_to_E1", "75p_E1_to_E2",
    "75p_E2_to_E1", "6s_E1_to_E2", "6s_E2_to_E1",
]
LABELS = ["50% E1→E2", "50% E2→E1", "75% E1→E2", "75% E2→E1", "6-s E1→E2", "6-s E2→E1"]
SHORT = ["50 E2", "50 E1", "75 E2", "75 E1", "6s E2", "6s E1"]
MAP_REG = "HyperSpatial_MAP_frozen_registration"
SPATEO_REG = "Spateo_Morpho_geometry_adapter"
BLUE = "#2474A6"
ORANGE = "#D98324"
GRAY = "#AEB4B8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.3,
        "axes.titlesize": 8.1, "axes.labelsize": 7.3,
        "xtick.labelsize": 6.1, "ytick.labelsize": 6.2,
        "legend.fontsize": 6.5, "axes.linewidth": .8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E4E7E9", lw=.55, zorder=0)
    ax.tick_params(direction="out", length=2.5)


def panel_label(ax, text: str, x: float = -.12, y: float = 1.08) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left", clip_on=False)


def main() -> None:
    style()
    original_files = sorted(p for p in ORIGINAL.iterdir() if p.is_file())
    hashes_before = {p: sha256(p) for p in original_files}

    align = pd.read_csv(SPATEO / "ALIGNMENT_RESULTS.csv")
    downstream = pd.read_csv(SPATEO / "DOWNSTREAM_IDW_RESULTS.csv")
    residual = pd.read_csv(SPATEO / "RESIDUAL_AFTER_ALIGNMENT.csv")
    if set(align.target) != set(CASES) or set(downstream.case) != set(CASES) or set(residual.case) != set(CASES):
        raise AssertionError("The Spateo audit does not contain the exact six frozen directions")

    # Copy the authoritative source tables into the new output as read-only provenance snapshots.
    align.to_csv(OUT / "external_alignment_chamfer.tsv", sep="\t", index=False)
    downstream.to_csv(OUT / "external_alignment_downstream_idw.tsv", sep="\t", index=False)
    residual.to_csv(OUT / "external_alignment_anchor_residual.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(12.0, 9.1))
    outer = fig.add_gridspec(2, 1, height_ratios=[6.15, 2.55], hspace=.055,
                             left=.025, right=.985, top=.985, bottom=.065)

    # Panels A-C are retained pixel-for-pixel from the authoritative rendered S1.
    ax_top = fig.add_subplot(outer[0, 0])
    ax_top.imshow(plt.imread(ORIGINAL / "Supp_Fig_S1_registration_qc.png"), interpolation="nearest")
    ax_top.axis("off")

    lower = outer[1, 0].subgridspec(1, 3, width_ratios=[1.05, 2.15, 1.05], wspace=.34)
    x = np.arange(6)

    # D: exact final Chamfer values from the frozen geometry-only alignment audit.
    ax = fig.add_subplot(lower[0, 0])
    for method, label, color in [
        ("HyperSpatial-MAP frozen registration", "HyperSpatial-MAP registration", BLUE),
        ("Spateo Morpho geometry adapter", "Spateo Morpho", ORANGE),
    ]:
        q = align[align.method == method].set_index("target").loc[CASES]
        ax.plot(x, q.final_chamfer, "-o", color=color, lw=1.25, ms=4.0,
                markerfacecolor="white", markeredgewidth=1.1, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(.0095, .065)
    ax.set_yticks([.01, .02, .05]); ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(x, SHORT, rotation=42, ha="right")
    ax.set_ylabel("Final symmetric Chamfer\n(canonical units; lower is better)")
    ax.set_title("External geometry-only alignment benchmark", fontweight="bold", pad=5)
    ax.legend(frameon=False, loc="upper left")
    clean(ax); panel_label(ax, "D", -.17, 1.13)

    # E: absolute registered-IDW endpoint values after each frozen alignment.
    sub = lower[0, 1].subgridspec(3, 3, height_ratios=[.55, 1, 1], hspace=.48, wspace=.42)
    header = fig.add_subplot(sub[0, :]); header.axis("off")
    header.text(.5, .92, "Downstream registered 12-NN IDW after alignment",
                transform=header.transAxes, ha="center", va="top", fontsize=8.1, fontweight="bold")
    panel_label(header, "E", -.055, 1.05)
    endpoints = [
        ("spatial_pearson", "Spatial Pearson ↑"),
        ("auprc_enrichment", "AUPRC enrichment ↑"),
        ("dice", "Dice ↑"),
        ("centroid_error", "Centroid error ↓"),
        ("spatial_emd", "Spatial EMD ↓"),
        ("log1p_rmse", "log1p RMSE ↓"),
    ]
    for j, (metric, title) in enumerate(endpoints):
        a = fig.add_subplot(sub[1 + j // 3, j % 3])
        for method, label, color in [
            (MAP_REG, "MAP registration", BLUE),
            (SPATEO_REG, "Spateo", ORANGE),
        ]:
            q = downstream[downstream.method == method].set_index("case").loc[CASES]
            a.plot(x, q[metric], "-o", color=color, lw=1.0, ms=3.0,
                   markerfacecolor="white", markeredgewidth=.9, label=label, zorder=3)
        a.set_title(title, fontsize=6.8, pad=2)
        if j < 3:
            a.set_xticks([])
        else:
            a.set_xticks(x, SHORT, rotation=42, ha="right", fontsize=5.1)
        clean(a)
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", markerfacecolor="white", label="MAP registration"),
        plt.Line2D([], [], color=ORANGE, marker="o", markerfacecolor="white", label="Spateo Morpho"),
    ]
    header.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
                  bbox_to_anchor=(.5, .00), fontsize=6.1)

    # F: exact median anchor-residual Moran's I after both frozen alignments.
    ax = fig.add_subplot(lower[0, 2])
    for method, label, color in [
        (MAP_REG, "MAP registration", BLUE),
        (SPATEO_REG, "Spateo Morpho", ORANGE),
    ]:
        q = residual[residual.method == method].set_index("case").loc[CASES]
        ax.plot(x, q.anchor_residual_moran_median, "-o", color=color, lw=1.25, ms=4.0,
                markerfacecolor="white", markeredgewidth=1.1, label=label, zorder=3)
    ax.axhline(0, color="#777", ls=":", lw=.75)
    ax.set_xticks(x, SHORT, rotation=42, ha="right")
    ax.set_ylabel("Median anchor-residual Moran’s I")
    ax.set_title("Structured residuals persist after either alignment", fontweight="bold", pad=5)
    ax.legend(frameon=False, loc="upper left")
    clean(ax); panel_label(ax, "F", -.17, 1.13)

    fig.text(.5, .018,
             "Six directional embryo-pair registrations are shown; reciprocal directions are not independent biological replicates. "
             "Target expression was excluded from both geometry-only alignments; anchors were evaluated only after alignment was frozen.",
             ha="center", va="bottom", fontsize=6.5, color="#555B60")

    stem = OUT / "Supp_Fig_S1_registration_qc_with_Spateo"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=.08, facecolor="white")
    plt.close(fig)

    hashes_after = {p: sha256(p) for p in original_files}
    if hashes_before != hashes_after:
        raise RuntimeError("The authoritative 01_registration_qc directory changed")
    audit_rows = [{
        "file": str(p.relative_to(ROOT)), "sha256_before": hashes_before[p],
        "sha256_after": hashes_after[p], "unchanged": hashes_before[p] == hashes_after[p],
    } for p in original_files]
    pd.DataFrame(audit_rows).to_csv(OUT / "ORIGINAL_S1_HASH_AUDIT.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
