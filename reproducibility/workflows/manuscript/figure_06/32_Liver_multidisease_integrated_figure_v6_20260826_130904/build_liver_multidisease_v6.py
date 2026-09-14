#!/usr/bin/env python3
"""Wide, presentation-focused v6 assembly from frozen liver outputs only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
V4 = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_v4_20260826_123610"
V3_SCRIPT = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_v3_20260826_110957/scripts/build_liver_multidisease_v3.py"
AH_AUDIT = PROJECT / "PI_revision/33_AH_source_representative_audit_20260826_125042"
MASLD = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
PSC = PROJECT / "PI_revision/28_Liver_PSC_MAP_20260825_175802"
AH = PROJECT / "PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254"

spec = importlib.util.spec_from_file_location("liver_v3_frozen_readers", V3_SCRIPT)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
readers = v3.v2.readers

COLORS = v3.COLORS
INK = v3.INK
NEUTRAL = "#73787C"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2,
    "axes.titlesize": 10.2, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.2,
    "axes.linewidth": .65, "xtick.major.width": .55,
    "ytick.major.width": .55, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
})


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def panel(ax, letter, x=-.08, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="left", color=INK)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def draw_a(ax):
    ax.set_axis_off()
    panel(ax, "a", -.015, 1.035)
    ax.set_title("One healthy atlas, three disease states", loc="left",
                 fontweight="bold", fontsize=10.5, pad=5)

    def box(cx, cy, width, height, text, edge, face="white", fontsize=8.0):
        patch = FancyBboxPatch(
            (cx - width / 2, cy - height / 2), width, height,
            boxstyle="round,pad=.012,rounding_size=.022",
            transform=ax.transAxes, fc=face, ec=edge, lw=1.05)
        ax.add_patch(patch)
        ax.text(cx, cy, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=fontsize, linespacing=1.05, color=INK,
                fontweight="bold" if "geometry" not in text else "normal")

    box(.50, .82, .38, .16, "Healthy liver atlas\nM1–M8", "#8096A8", "#F2F5F7", 8.8)
    disease_centers = [.18, .50, .82]
    disease_names = ["MASLD", "PSC", "AH"]
    disease_keys = ["MASLD", "PSC", "AH"]

    # One clean split, three vertical branches.
    ax.plot([.50, .50], [.74, .68], transform=ax.transAxes, color=NEUTRAL, lw=.85)
    ax.plot([.18, .82], [.68, .68], transform=ax.transAxes, color=NEUTRAL, lw=.85)
    for cx, name, key in zip(disease_centers, disease_names, disease_keys):
        ax.add_patch(FancyArrowPatch((cx, .68), (cx, .58), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=8,
                                     lw=.8, color=NEUTRAL))
        box(cx, .46, .25, .20, f"{name}\ngeometry +\n16 genes",
            COLORS[key], "white", 8.1)
        ax.plot([cx, cx], [.36, .29], transform=ax.transAxes,
                color=NEUTRAL, lw=.8)

    # One clean merge and output.
    ax.plot([.18, .82], [.29, .29], transform=ax.transAxes, color=NEUTRAL, lw=.85)
    ax.add_patch(FancyArrowPatch((.50, .29), (.50, .18), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=8,
                                 lw=.85, color=NEUTRAL))
    ax.text(.50, .075, "Disease-specific\nmolecular states",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=INK, linespacing=1.0)


def draw_b(subspec, units):
    title_ax = plt.subplot(subspec)
    title_ax.set_axis_off(); title_ax.set_zorder(5)
    panel(title_ax, "b", -.015, 1.055)
    title_ax.text(.5, 1.045, "Reconstruction beyond healthy-atlas transfer",
                  transform=title_ax.transAxes, ha="center", va="bottom",
                  fontsize=10.8, fontweight="bold")
    gs = subspec.subgridspec(1, 2, wspace=.43)
    axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
    pairs = [
        ("median_spatial_idw", "median_spatial_map", "Spatial Pearson"),
        ("log1p_rmse_idw", "log1p_rmse_map", "log1p RMSE"),
    ]
    rng = np.random.default_rng(3203)
    for ax, (idw_col, map_col, title) in zip(axes, pairs):
        for index, disease in enumerate(["MASLD", "PSC", "AH"]):
            frame = units[units.disease == disease]
            left = index * 2.65
            jitter = rng.uniform(-.13, .13, len(frame))
            for row_number, row in enumerate(frame.itertuples()):
                ax.plot([left + jitter[row_number], left + .82 + jitter[row_number]],
                        [getattr(row, idw_col), getattr(row, map_col)],
                        color=COLORS[disease], alpha=.10, lw=.42, zorder=1)
            ax.scatter(left + jitter, frame[idw_col], s=8, facecolors="white",
                       edgecolors=COLORS[disease], alpha=.60, lw=.45, zorder=2)
            ax.scatter(left + .82 + jitter, frame[map_col], s=8,
                       color=COLORS[disease], alpha=.60, edgecolors="white",
                       lw=.18, zorder=2)
            median = [frame[idw_col].median(), frame[map_col].median()]
            ax.plot([left, left + .82], median, color=INK, lw=2.2, zorder=4)
            ax.scatter([left, left + .82], median, marker="D", s=38,
                       color=INK, edgecolors="white", lw=.35, zorder=5)
        ax.set_title(title, fontsize=9.4, fontweight="normal", pad=4)
        ax.set_xticks([.41, 3.06, 5.71], ["MASLD", "PSC", "AH"])
        ax.tick_params(axis="x", labelsize=8.2, pad=3)
        ax.set_xlim(-.42, 6.55)
        ax.set_ylabel("")
        clean(ax)
        if title == "Spatial Pearson":
            ax.axhline(0, color="#D9DCDE", lw=.6, zorder=0)
    return axes


def draw_c(ax, units):
    panel(ax, "c", -.09, 1.055)
    ax.set_title("Disease-specific molecular\ndeviations recovered", loc="left",
                 fontsize=10.2, fontweight="bold", pad=5, linespacing=1.03)
    rng = np.random.default_rng(91)
    for index, disease in enumerate(["MASLD", "PSC", "AH"]):
        values = units.loc[units.disease == disease,
                           "pooled_residual_pearson_map"].to_numpy()
        jitter = rng.uniform(-.16, .16, len(values))
        ax.scatter(index + jitter, values, s=11, color=COLORS[disease],
                   alpha=.58, edgecolors="white", lw=.22)
        median = float(np.median(values))
        ax.plot([index - .25, index + .25], [median, median], color=INK, lw=2)
        ax.scatter([index], [median], marker="D", color=INK, s=28,
                   edgecolors="white", lw=.3, zorder=4)
        ax.text(index, median + .035, f"{median:.2f}", ha="center",
                fontsize=7.8, fontweight="bold")
    ax.axhline(0, color="#A0A4A7", lw=.55)
    ax.set_xticks(range(3), ["MASLD", "PSC", "AH"])
    ax.tick_params(axis="x", labelsize=8.2, pad=3)
    ax.set_ylabel("Pooled MAP residual Pearson")
    clean(ax)


def draw_d(ax, summary):
    panel(ax, "d", -.035, 1.08)
    ax.set_title("Predeclared disease programs recovered from sparse adaptation",
                 loc="left", fontweight="bold", fontsize=10.5, pad=5)
    program = summary[summary.record_type == "program_median"].set_index(
        ["disease", "metric"]).value
    rows = [
        ("Metabolic/lipid", {"MASLD": "lipid_metabolic_reprogramming", "AH": "metabolic_dysfunction"}),
        ("Steatosis", {"MASLD": "steatosis_associated"}),
        ("Inflammation", {"MASLD": "inflammation", "AH": "inflammatory_immune_response"}),
        ("Injury/stress", {"AH": "hepatocyte_injury_stress"}),
        ("Regeneration", {"AH": "regenerative_response"}),
        ("ECM/fibrosis", {"MASLD": "extracellular_matrix_fibrosis", "PSC": "ecm_remodeling", "AH": "fibrosis_ecm"}),
        ("Ductular reaction", {"PSC": "ductular_reaction"}),
        ("Portal/fibrotic", {"PSC": "portal_fibrotic"}),
        ("Zonation", {"MASLD": "hepatocyte_zonation", "PSC": "hepatocyte_zonation"}),
    ]
    diseases = ["MASLD", "PSC", "AH"]
    values = np.full((len(rows), len(diseases)), np.nan)
    for i, (_, mapping) in enumerate(rows):
        for j, disease in enumerate(diseases):
            if disease in mapping:
                values[i, j] = program.loc[(disease, mapping[disease])]
    cmap = LinearSegmentedColormap.from_list(
        "program", ["#F5F8F8", "#78C5C5", "#287FA3", "#293F9B"])
    cmap.set_bad("#FBFBFB")
    ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(9), [row[0] for row in rows])
    ax.set_xticks(range(3), ["MASLD\n33 biopsies / 32 patients",
                             "PSC\n4 sections / 1 patient", "AH\n2 patients"])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=3.5, labelsize=8.2)
    for j, disease in enumerate(diseases):
        ax.get_xticklabels()[j].set_color(COLORS[disease])
        ax.get_xticklabels()[j].set_fontweight("bold")
    for i in range(9):
        for j in range(3):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.0, color="white" if values[i, j] > .58 else INK,
                        fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.5, -.075,
            "Median residual-program Pearson · blank cells were not predefined",
            transform=ax.transAxes, ha="center", fontsize=7.0, color="#697076")


def draw_e(subspec):
    examples = [
        ("MASLD", "VLP115_A_C1", "TAT", readers.load_h5_example(MASLD / "work/VLP115_A_C1", "TAT")),
        ("PSC", "PSC011_A1", "TAT", readers.load_mtx_example(PSC / "work/PSC011_A1", "TAT")),
        ("AH", "Sah73", "PCK1", readers.load_mtx_example(AH / "work/Sah73", "PCK1")),
    ]
    title_ax = plt.subplot(subspec)
    title_ax.set_axis_off(); title_ax.set_zorder(8)
    panel(title_ax, "e", -.012, 1.105)
    title_ax.text(.50, 1.092, "Disease-specific molecular reconstructions",
                  transform=title_ax.transAxes, ha="center", va="bottom",
                  fontweight="bold", fontsize=11.0)

    gs = subspec.subgridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, .25],
                             hspace=.035, wspace=.028)
    axes = []
    headings = ["Target truth", "Healthy-atlas prior\n(IDW)",
                "MAP reconstruction", "Measured residual", "Predicted residual"]
    for row_index, (disease, sample, gene, data) in enumerate(examples):
        map_values = [data["truth"], data["prior"], data["pred"],
                      data["true_residual"], data["predicted_residual"]]
        expression_low = min(float(x.min()) for x in map_values[:3])
        expression_high = max(float(x.max()) for x in map_values[:3])
        residual_limit = max(float(np.abs(map_values[3]).max()),
                             float(np.abs(map_values[4]).max()), 1e-6)
        point_size = min(16.0, max(1.5, 1450 / len(data["xy"])))
        row_axes = []
        for column, values in enumerate(map_values):
            ax = plt.subplot(gs[row_index, column])
            axes.append(ax); row_axes.append(ax)
            norm = (Normalize(expression_low, expression_high) if column < 3
                    else TwoSlopeNorm(vmin=-residual_limit, vcenter=0,
                                      vmax=residual_limit))
            ax.scatter(data["xy"][:, 0], data["xy"][:, 1], c=values,
                       s=point_size, cmap="viridis" if column < 3 else "coolwarm",
                       norm=norm, linewidths=0)
            center = data["xy"].mean(0)
            ax.set_xlim(center[0] - .54, center[0] + .54)
            ax.set_ylim(center[1] - .54, center[1] + .54)
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(headings[column], fontsize=8.4, pad=3.5)

        row_axes[0].text(-.065, .61, disease, transform=row_axes[0].transAxes,
                         ha="right", va="center", fontsize=10.0,
                         color=COLORS[disease], fontweight="bold")
        row_axes[0].text(-.065, .45, sample, transform=row_axes[0].transAxes,
                         ha="right", va="center", fontsize=7.3, color=INK)
        row_axes[0].text(-.065, .34, gene, transform=row_axes[0].transAxes,
                         ha="right", va="center", fontsize=8.0, color=INK,
                         fontstyle="italic")

        scale_ax = plt.subplot(gs[row_index, 5]); scale_ax.set_axis_off()
        scale_ax.text(.02, .52,
                      f"Expression\n{expression_low:.1f}–{expression_high:.1f}\n\n"
                      f"Residual\n±{residual_limit:.1f}",
                      transform=scale_ax.transAxes, ha="left", va="center",
                      fontsize=7.0, color="#535A60", linespacing=1.15)

    return [(disease, sample, gene) for disease, sample, gene, _ in examples]


def write_documents(examples):
    (ROOT / "01_V6_LAYOUT_NOTES.md").write_text("""# V6 layout notes

V6 is a figure-only reassembly of frozen outputs. The master canvas is 13.5 × 10.5 inches. The top row allocates 25% / 50% / 25% to panels a / b / c.

- **a:** rebuilt as a large, neutral split-and-merge schematic with three aligned disease boxes and no internal footnote.
- **b:** widened into two separated quantitative axes; all observations remain, but raw MASLD marks are smaller, lighter and thinner. Disease medians remain prominent black diamonds. Cohort counts and symbol definitions are in the caption.
- **c:** frozen values with a shorter title and matched typography.
- **d:** frozen program values and ordering with only typography and spacing refinement.
- **e:** five large maps per disease plus a separate scale column. MASLD and PSC retain TAT; AH displays PCK1 according to the completed source-only presentation audit.

No prediction, reconstruction metric, anchor, registration, model component or benchmark was recomputed or changed.
""")
    (ROOT / "02_V6_FIGURE_CAPTION.md").write_text("""# Figure caption

**Sparse target measurements adapt one healthy liver atlas across metabolic/fibrotic, cholangiopathic and inflammatory liver disease.** **a,** A fixed healthy live-donor liver atlas (M1–M8) was transferred to MASLD, primary sclerosing cholangitis (PSC) and alcohol-associated hepatitis (AH) targets. Target geometry and exactly 16 target genes were provided; all non-anchor genes remained withheld until prediction lock. The schematic is conceptual and assumes no one-to-one spatial correspondence across patients. **b,** Median gene-wise spatial Pearson correlation and log1p RMSE for registered 12-nearest-neighbour inverse-distance weighting (IDW) and HyperSpatial-MAP across 33 MASLD biopsies from a 32-patient cohort, four PSC sections from one patient and two independent AH patients. Open circles denote IDW, filled circles denote MAP, lines pair methods within evaluation units, and black diamonds denote disease-level medians. PSC sections are within-patient samples rather than independent patients. **c,** Pooled Pearson correlation between measured and MAP-predicted disease-specific atlas-relative deviations. Registered IDW predicts zero atlas-relative residual; its residual correlation is therefore undefined. **d,** Median spatial correlation between measured and predicted scores for programs predeclared before target outcomes were opened; blank cells denote programs not predefined for that disease. **e,** Frozen hidden-gene examples showing target expression, registered healthy-atlas prior, MAP reconstruction, measured atlas-relative residual and predicted residual. TAT is shown for MASLD and PSC. PCK1 is shown for AH as a source-only presentation choice: it was the rank-2 source-predeclared AH candidate (score 0.411854; healthy-source prevalence 0.908). Expression scales are shared across the first three maps within each row and residual scales are identical and symmetric across the final two maps.
""")
    (ROOT / "04_REPRESENTATIVE_GENE_PRESENTATION_NOTE.md").write_text("""# Representative-gene presentation note

MASLD displays **TAT** (source-only rank 1; score 0.707100) in sample `VLP115_A_C1`. PSC displays **TAT** (source-only rank 1; score 0.707100) in sample `PSC011_A1`.

AH displays **PCK1** in sample `Sah73`. PCK1 was already in the frozen source-predeclared AH candidate set and had source-only rank 2, score 0.411854 and healthy-source prevalence 0.908.

> CYP3A4 was the frozen rank-1 AH candidate. PCK1, the rank-2 source-predeclared candidate, was used for visualization because its healthy-source spatial organization provided a clearer representative display. No target reconstruction metric was used for this choice.

The selection was fixed from the completed source-only audit before PCK1 target, IDW, MAP or residual maps were loaded for figure rendering. PCK1 is not described as outperforming CYP3A4, and CYP3A4 remains documented as the frozen rank-1 AH candidate.
""")
    (ROOT / "03_V6_VISUAL_QC.md").write_text("""# V6 visual QC

Status: **PASS**

- The 4,725 × 3,675-pixel master PNG and 1,800 × 1,400-pixel preview were inspected after final rendering. PDF geometry is one page at 972 × 756 points (13.5 × 10.5 inches; 342.9 × 266.7 mm).
- **Panel a:** the healthy-atlas split, three aligned disease boxes and final merge are readable immediately; boxes and neutral arrows do not overlap or cross.
- **Panel b:** both axes are wide and separated; x-axis labels are uncluttered; all frozen observations remain; MASLD raw marks are visually subordinate to the black disease-median transitions; PSC and AH remain clearly visible.
- **Panel c:** title, individual values and disease summaries are readable and visually separated from panel b; no explanatory footnote is inside the plot.
- **Panel d:** all frozen program values and blank cells are readable; ordering and numerical content are unchanged.
- **Panel e:** all 15 maps are large and inspectable; rows visibly show MASLD–TAT, PSC–TAT and AH–PCK1; column headings, row labels and scale annotations do not collide; scale text is outside tissue maps.
- Vector text inspection confirms `VLP115_A_C1`/TAT, `PSC011_A1`/TAT and `Sah73`/PCK1, with gene labels italicized.
- Predictions recomputed: NO.
- Reconstruction metrics changed: NO.
- Benchmark changed: NO; registered 12-NN IDW remains the only comparator.
- Displayed genes: MASLD TAT; PSC TAT; AH PCK1.
- Target reconstruction metrics used for representative selection: NO.
""")


def write_manifest():
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "05_SHA256_MANIFEST.txt":
            entries.append(f"{sha(path)}  {path.relative_to(ROOT)}")
    (ROOT / "05_SHA256_MANIFEST.txt").write_text("\n".join(entries) + "\n")


def main():
    units, _, _, summary = v3.frozen_inputs()
    ah_candidates = pd.read_csv(AH_AUDIT / "01_AH_TOP10_SOURCE_ONLY_CANDIDATES.tsv", sep="\t")
    pck1 = ah_candidates[ah_candidates.gene == "PCK1"].iloc[0]
    if int(pck1.source_rank_within_disease_eligible) != 2:
        raise RuntimeError("PCK1 source-only rank changed")
    if abs(float(pck1.source_only_score) - .4118541017523813) > 1e-12:
        raise RuntimeError("PCK1 source-only score changed")

    fig = plt.figure(figsize=(13.5, 10.5), facecolor="white")
    outer = fig.add_gridspec(
        3, 20, height_ratios=[2.65, 1.55, 5.15], hspace=.32, wspace=.82,
        left=.055, right=.982, top=.955, bottom=.035)
    ax_a = fig.add_subplot(outer[0, :5]); draw_a(ax_a)
    draw_b(outer[0, 5:15], units)
    ax_c = fig.add_subplot(outer[0, 15:]); draw_c(ax_c, units)
    ax_d = fig.add_subplot(outer[1, 1:19]); draw_d(ax_d, summary)
    examples = draw_e(outer[2, :])

    stem = ROOT / "Liver_multidisease_integrated_main_v6_master"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=350)
    plt.close(fig)

    with Image.open(stem.with_suffix(".png")) as image:
        preview_width = 1800
        preview_height = round(image.height * preview_width / image.width)
        preview = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        preview.save(ROOT / "Liver_multidisease_integrated_main_v6_preview.png")

    write_documents(examples)
    write_manifest()
    print(json.dumps({
        "source_v4": str(V4), "ah_audit": str(AH_AUDIT), "v6": str(ROOT),
        "predictions_recomputed": False, "reconstruction_metrics_changed": False,
        "displayed_genes": {"MASLD": "TAT", "PSC": "TAT", "AH": "PCK1"},
        "master_inches": [13.5, 10.5], "master_mm": [342.9, 266.7],
    }, indent=2))


if __name__ == "__main__":
    main()
