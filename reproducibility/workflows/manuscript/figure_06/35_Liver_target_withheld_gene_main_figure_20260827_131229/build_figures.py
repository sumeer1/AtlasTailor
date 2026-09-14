#!/usr/bin/env python3
"""Render target-withheld liver gene figures from frozen predictions only."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[1]
LOCKED = PROJECT / "PI_revision/34_Liver_target_withheld_gene_interpretation_20260827_094120"
MASLD = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
PSC = PROJECT / "PI_revision/28_Liver_PSC_MAP_20260825_175802"
AH = PROJECT / "PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254"
LOADER_PATH = LOCKED / "scripts/03_render_and_report.py"

COLORS = {"MASLD": "#C7832B", "PSC": "#16858C", "AH": "#B54A69"}
UNITS = {"MASLD": "VLP115_A_C1", "PSC": "PSC011_A1", "AH": "Sah73"}
MAIN = [
    ("MASLD", "DCN", "ECM / fibrotic remodeling"),
    ("PSC", "KRT8", "Ductular reaction"),
    ("AH", "NFKBIA", "Inflammatory response"),
]
SUPP = [
    ("MASLD", "TAT", "Hepatocyte zonation"),
    ("PSC", "TAT", "Hepatocyte zonation"),
]
ROWS = [
    "Measured target",
    "Registered healthy prior",
    "HyperSpatial-MAP",
    "Measured residual",
    "Predicted residual",
]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_loader():
    spec = importlib.util.spec_from_file_location("frozen_gene_loader", LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate_inputs(summary: pd.DataFrame, lock: pd.DataFrame) -> None:
    expected_lock_hash = "4d23e9a3c3079ef8cc78d0fe0d8200bdedc80da161a0f0f6ecb7cc4c60dcc0f1"
    actual = sha256(LOCKED / "CANDIDATE_GENE_LOCK.tsv")
    if actual != expected_lock_hash:
        raise RuntimeError(f"Candidate lock hash mismatch: {actual}")
    for disease, gene, _ in MAIN + SUPP:
        if not ((lock.disease == disease) & (lock.gene == gene)).any():
            raise RuntimeError(f"{disease}/{gene} is absent from the locked candidate set")
        if not ((summary.disease == disease) & (summary.gene == gene)).any():
            raise RuntimeError(f"{disease}/{gene} is absent from frozen result summaries")
    for disease, root in {"MASLD": MASLD, "PSC": PSC, "AH": AH}.items():
        anchors = pd.read_csv(root / "01_ANCHORS.tsv", sep="\t")
        if len(anchors) != 16:
            raise RuntimeError(f"{disease} does not have exactly 16 frozen anchors")


def result_row(summary: pd.DataFrame, disease: str, gene: str):
    return summary[(summary.disease == disease) & (summary.gene == gene)].iloc[0]


def metric_text(record) -> str:
    consistency = str(record.consistency_across_units)
    consistency = consistency.replace("spatial MAP>IDW ", "")
    consistency = consistency.replace("RMSE MAP<IDW ", "")
    consistency = consistency.replace("positive residual r ", "")
    bits = [x.strip() for x in consistency.split(";")]
    compact = f"{bits[0]} spatial · {bits[1]} RMSE · {bits[2]} residual+"
    return (
        f"Spatial r  {record.IDW_spatial_Pearson:.3f} → {record.MAP_spatial_Pearson:.3f}\n"
        f"RMSE       {record.IDW_RMSE:.3f} → {record.MAP_RMSE:.3f}\n"
        f"Residual r  {record.residual_Pearson:.3f}\n"
        f"{compact}"
    )


def draw_schematic(ax):
    ax.set_axis_off()
    ax.text(0.015, 0.62, "a", fontsize=15, fontweight="bold", va="center")
    ax.text(0.065, 0.62, "Healthy liver atlas", fontsize=11.5, fontweight="bold", va="center")
    ax.text(0.223, 0.62, "+", fontsize=15, color="#555A5E", ha="center", va="center")
    ax.text(0.285, 0.62, "16 measured target genes", fontsize=11.5, fontweight="bold", va="center")
    ax.annotate("", xy=(0.594, 0.62), xytext=(0.505, 0.62),
                arrowprops=dict(arrowstyle="-|>", color="#555A5E", lw=1.5))
    ax.text(0.63, 0.62, "Target-withheld disease-state expression", fontsize=11.5,
            fontweight="bold", va="center")
    ax.text(0.065, 0.20,
            "One fixed healthy reference and the same frozen 16-gene target panel were used across all three diseases.",
            fontsize=8.2, color="#666B70", va="center")


def render_grid(entries, maps, summary, stem: str, supplementary: bool = False):
    ncol = len(entries)
    width = 10.5 if ncol == 3 else 7.8
    fig = plt.figure(figsize=(width, 12.0), facecolor="white")
    gs = fig.add_gridspec(
        7, ncol + 1,
        height_ratios=[0.72, 1.48, 2.0, 2.0, 2.0, 2.0, 2.0],
        width_ratios=[0.62] + [1.0] * ncol,
        left=0.045, right=0.975, top=0.985, bottom=0.035,
        wspace=0.08, hspace=0.09,
    )

    schematic = fig.add_subplot(gs[0, :])
    if supplementary:
        schematic.set_axis_off()
        schematic.text(0.015, 0.66, "Target-withheld TAT reconstruction",
                       fontsize=13, fontweight="bold", va="center")
        schematic.text(0.015, 0.20,
                       "Healthy liver atlas + 16 measured target genes → target-withheld disease-state expression",
                       fontsize=8.4, color="#666B70", va="center")
        fig.text(0.022, 0.872, "a", fontsize=15, fontweight="bold", va="top")
    else:
        draw_schematic(schematic)
        fig.text(0.022, 0.872, "b", fontsize=15, fontweight="bold", va="top")

    for col, (disease, gene, biology) in enumerate(entries, start=1):
        head = fig.add_subplot(gs[1, col])
        head.set_axis_off()
        head.add_patch(plt.Rectangle((0.0, 0.90), 1.0, 0.045, transform=head.transAxes,
                                     color=COLORS[disease], lw=0))
        head.text(0.5, 0.78, disease, ha="center", va="center", fontsize=12.2,
                  color=COLORS[disease], fontweight="bold")
        head.text(0.5, 0.57, rf"$\it{{{gene}}}$", ha="center", va="center",
                  fontsize=14.0, fontweight="bold")
        head.text(0.5, 0.40, biology, ha="center", va="center", fontsize=8.8)
        record = result_row(summary, disease, gene)
        head.text(0.5, 0.15, metric_text(record), ha="center", va="center", fontsize=7.0,
                  linespacing=1.30, color="#373B3E",
                  bbox=dict(boxstyle="round,pad=0.30", fc="#F6F7F7", ec="#D6DADC", lw=0.65))

    label_ax = fig.add_subplot(gs[1, 0])
    label_ax.set_axis_off()
    label_ax.text(0.0, 0.55, "Frozen summary\n(IDW → MAP)", fontsize=7.4,
                  color="#666B70", ha="left", va="center")

    for row_index, row_label in enumerate(ROWS):
        la = fig.add_subplot(gs[row_index + 2, 0])
        la.set_axis_off()
        la.text(0.98, 0.5, row_label, ha="right", va="center", fontsize=8.4,
                fontweight="bold" if row_index in (0, 2) else "normal",
                color="#303437")
        for col, (disease, gene, _) in enumerate(entries, start=1):
            z = maps[(disease, gene)]
            values = [z["truth"], z["prior"], z["pred"],
                      z["measured_residual"], z["predicted_residual"]]
            expr_low = min(float(np.nanmin(x)) for x in values[:3])
            expr_high = max(float(np.nanmax(x)) for x in values[:3])
            resid_max = max(float(np.nanmax(np.abs(values[3]))),
                            float(np.nanmax(np.abs(values[4]))), 1e-6)
            ax = fig.add_subplot(gs[row_index + 2, col])
            norm = Normalize(expr_low, expr_high) if row_index < 3 else TwoSlopeNorm(
                vmin=-resid_max, vcenter=0, vmax=resid_max)
            ax.scatter(z["xy"][:, 0], z["xy"][:, 1], c=values[row_index],
                       s=4.0 if disease != "MASLD" else 3.5,
                       cmap="viridis" if row_index < 3 else "coolwarm",
                       norm=norm, linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 4:
                ax.text(0.5, -0.07, f"Expression 0–{expr_high:.2f}  ·  Residual ±{resid_max:.2f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=6.7,
                        color="#666B70")

    for ext, dpi in [("pdf", 300), ("svg", 300), ("png", 360)]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


def write_documents(summary: pd.DataFrame, lock: pd.DataFrame):
    requested = MAIN + SUPP
    rows = []
    for disease, gene, biology in requested:
        rec = result_row(summary, disease, gene)
        rows.append({
            "figure": "main" if (disease, gene, biology) in MAIN else "supplementary",
            "disease": disease,
            "unit": UNITS[disease],
            "gene": gene,
            "biological_label": biology,
            "number_of_biological_units": rec.number_of_biological_units,
            "IDW_spatial_Pearson": rec.IDW_spatial_Pearson,
            "MAP_spatial_Pearson": rec.MAP_spatial_Pearson,
            "delta_spatial_Pearson": rec.delta_spatial_Pearson,
            "IDW_RMSE": rec.IDW_RMSE,
            "MAP_RMSE": rec.MAP_RMSE,
            "delta_RMSE": rec.delta_RMSE,
            "residual_Pearson": rec.residual_Pearson,
            "consistency_across_units": rec.consistency_across_units,
        })
    pd.DataFrame(rows).to_csv(OUT / "04_FROZEN_METRICS_USED.tsv", sep="\t", index=False)

    anchors = pd.read_csv(MASLD / "01_ANCHORS.tsv", sep="\t")["gene"].astype(str).tolist()
    provenance = f"""# Selected units and gene provenance

## Main figure

- MASLD: **VLP115_A_C1**, target-withheld **DCN** (prelocked extracellular-matrix/fibrosis candidate).
- PSC: **PSC011_A1**, target-withheld **KRT8** (prelocked ductular-reaction candidate).
- AH: **Sah73**, target-withheld **NFKBIA** (prelocked inflammatory/immune candidate).

These fixed display units were already used by the locked target-withheld analysis and were not selected by comparing candidate-level MAP performance. The three genes were fixed by the present request and are all present in the hashed candidate lock (`{sha256(LOCKED / 'CANDIDATE_GENE_LOCK.tsv')}`).

## Supplementary figure

- MASLD: **VLP115_A_C1**, target-withheld **TAT**.
- PSC: **PSC011_A1**, target-withheld **TAT**.

## Frozen information boundary

All three disease analyses used the same 16 target anchors: {', '.join(anchors)}. DCN, KRT8, NFKBIA and TAT are not anchors. Their target disease-state expression was withheld during inference, although the genes were represented in the fixed healthy source atlas.

## MASLD source-directory resolution

No directory matching `PI_revision/27_Liver_MASLD_MAP_*` exists in the project. The locked interpretation analysis points to the canonical frozen MASLD run `PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD`, which was therefore used without modification.
"""
    (OUT / "01_SELECTED_UNITS_AND_GENE_PROVENANCE.md").write_text(provenance)

    caption = """# Figure caption draft

**Target-withheld disease genes reconstructed from sparse liver measurements.** A fixed healthy liver atlas and the same frozen panel of 16 measured target genes were used to reconstruct target disease-state expression across MASLD, primary sclerosing cholangitis (PSC) and alcohol-associated hepatitis (AH). The displayed genes were not among the 16 target anchors: *DCN* represents extracellular-matrix/fibrotic remodeling in MASLD, *KRT8* represents ductular reaction in PSC, and *NFKBIA* represents inflammatory signaling in AH. Rows show measured target expression, the registered healthy-atlas prior generated by 12-nearest-neighbor inverse-distance weighting (IDW), the HyperSpatial-MAP reconstruction, the measured atlas-relative residual and the predicted residual. HyperSpatial-MAP did not observe the target disease-state expression of these genes during inference; “target-withheld” does not imply that the genes were absent from the healthy source atlas. Expression scales are shared across measured target, IDW and MAP maps within each gene, and residual scales are symmetric and shared within each gene. Summary values are medians across 33 MASLD biopsy evaluations from a 32-patient cohort, four PSC sections from one patient, and two independent AH patients; section-level PSC consistency is not patient replication.

**Supplementary figure.** Target-withheld *TAT* reconstruction in MASLD and PSC, shown with the same map order and within-gene scaling conventions.
"""
    (OUT / "02_FIGURE_CAPTION_DRAFT.md").write_text(caption)

    text = """# Manuscript text draft

Beyond aggregate recovery of predeclared disease programs, HyperSpatial-MAP reconstructed specific biologically relevant genes whose disease-state expression was withheld from the target input. Using the fixed healthy reference and only 16 measured target genes, MAP recovered spatial expression and atlas-relative deviations for *DCN* in MASLD, *KRT8* in PSC and *NFKBIA* in alcohol-associated hepatitis, providing concrete examples of fibrotic, cholangiopathic and inflammatory target-state adaptation, respectively. These examples concern prediction of target disease-state expression; the genes themselves were represented in the healthy source atlas.
"""
    (OUT / "03_MANUSCRIPT_TEXT_DRAFT.md").write_text(text)


def main():
    summary = pd.read_csv(LOCKED / "TARGET_WITHHELD_GENE_RESULTS.tsv", sep="\t")
    lock = pd.read_csv(LOCKED / "CANDIDATE_GENE_LOCK.tsv", sep="\t")
    validate_inputs(summary, lock)
    loader = load_frozen_loader()
    maps = {}
    for disease, genes in {
        "MASLD": ["DCN", "TAT"],
        "PSC": ["KRT8", "TAT"],
        "AH": ["NFKBIA"],
    }.items():
        loaded = loader.load_gene_maps(disease, genes)
        for gene, values in loaded.items():
            maps[(disease, gene)] = values
    render_grid(MAIN, maps, summary, "Liver_target_withheld_genes_main")
    render_grid(SUPP, maps, summary, "Liver_target_withheld_genes_TAT_supplementary", supplementary=True)
    write_documents(summary, lock)


if __name__ == "__main__":
    main()
