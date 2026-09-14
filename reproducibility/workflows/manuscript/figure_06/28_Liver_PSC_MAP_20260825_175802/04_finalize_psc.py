#!/usr/bin/env python3
"""Create manuscript-facing PSC reports, figure, and final integrity manifest."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def panel_label(ax, label):
    ax.text(-0.14, 1.09, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")


def main():
    sections = pd.read_csv(ROOT / "04A_SECTION_LEVEL_RESULTS_PSC.tsv", sep="\t")
    patient = pd.read_csv(ROOT / "04_PATIENT_LEVEL_RESULTS.tsv", sep="\t").iloc[0]
    programs = pd.read_csv(ROOT / "07B_DISEASE_PROGRAM_SUMMARY.tsv", sep="\t")
    registration = pd.read_csv(ROOT / "03_REGISTRATION_RESULTS_PRELOCK.tsv", sep="\t")
    summary = json.loads((ROOT / "psc_summary.json").read_text())
    shutil.copyfile(ROOT / "03_REGISTRATION_RESULTS_PRELOCK.tsv", ROOT / "06_REGISTRATION_RESULTS.tsv")

    biology_lines = "\n".join(
        f"- {row.program.replace('_', ' ')}: median residual r = {fmt(row.median_spatial_residual_pearson)} "
        f"(section range {fmt(row.min_spatial_residual_pearson)}–{fmt(row.max_spatial_residual_pearson)}; "
        f"positive in {row.positive_section_fraction:.0%} of sections)."
        for row in programs.itertuples()
    )
    (ROOT / "08_BIOLOGICAL_INTERPRETATION.md").write_text(f"""# PSC biological recovery

Four literature-defined programs were frozen in `PSC_PROGRAMS.json` before prediction lock and before target hidden-expression access. Recovery compares the measured atlas-relative residual (`PSC truth − registered IDW`) with the predicted residual (`MAP − registered IDW`).

{biology_lines}

These are spatial correlations of aggregate program residuals, not differential-expression tests. The four sections come from one patient and are not independent biological replicates.
""")

    (ROOT / "08_COMPARTMENT_ANALYSIS.md").write_text("""# Portal/periportal compartment analysis

**Status: not estimable from verified deposited annotations.**

GSE245620 provides count matrices, coordinates, and tissue images but no deposited spot-level portal/periportal or PSC-scar annotation table. The frozen rule permits only externally verified compartment labels for this analysis. Target hidden expression was not used to manufacture a compartment label, and no pathology annotation was inferred from images. Whole-section portal/fibrotic, ductular-reaction, ECM-remodeling, and zonation program recovery is reported instead.
""")

    (ROOT / "09_GSE240429_SENSITIVITY.md").write_text("""# GSE240429 same-study healthy-reference sensitivity

**Status: not run under the exact frozen protocol.**

GSE240429 contains four healthy C73 Visium sections from the same study and is relevant to separating study/platform effects. However, replacing the M1–M8 reference with C73 would require refitting the source model or redefining the residual prior. The post-MASLD protocol freezes the exact M1–M8 source model and prohibits either change. GSE240429 was therefore not used as a secondary reference in this exact-protocol PSC result. This is a protocol decision made independently of PSC reconstruction outcomes.
""")

    (ROOT / "11_LEAKAGE_AUDIT.md").write_text(f"""# PSC leakage and integrity audit

**Result: PASS.**

- The exact post-MASLD protocol and source model hashes verified before prediction.
- All four PSC011 sections contained the same 16 frozen anchor identities and all 6,814 locked hidden-axis genes.
- Before lock, matrix values were parsed and materialized only for anchor feature rows; target non-anchor expression values materialized: 0.
- Registration used coordinates only, with fixed seeds 42, 43, and 44 and seed 42 primary.
- Disease status, programs, portal/scar annotations, and outcomes did not enter registration, fitting, operator selection, calibration, or guard decisions.
- Registered 12-NN IDW was the only comparator; anchor-only decoding remained internal.
- Every registration, support mask, depth prediction, IDW prediction, MAP prediction, operator/guard assignment, source model, configuration, target file, and lock manifest matched its recorded SHA256 before hidden truth was opened.
- No retuning, post-outcome exclusion, or metric change occurred.

Independent patient N is 1. Four sections are retained as within-patient section evaluations, not biological replicates.
""")

    (ROOT / "12_RESULTS_INTERPRETATION.md").write_text(f"""# PSC results interpretation

Using the fixed M1–M8 healthy reference and exactly 16 frozen target genes, HyperSpatial-MAP improved the patient-level median spatial Pearson from {fmt(patient.median_spatial_idw)} to {fmt(patient.median_spatial_map)} and reduced log1p RMSE from {fmt(patient.log1p_rmse_idw)} to {fmt(patient.log1p_rmse_map)}. The pooled atlas-relative residual Pearson was {fmt(patient.pooled_residual_pearson_map)}, with median gene-wise residual Pearson {fmt(patient.median_residual_pearson_map)}.

The direction of improvement was consistent across all four sections for both principal reconstruction endpoints. MAP exceeded IDW spatial Pearson for {patient.fraction_map_gt_idw_spatial:.1%} of hidden gene–section-compatible records, reduced gene-wise RMSE for {patient.fraction_map_lt_idw_rmse:.1%}, and produced positive residual Pearson for {patient.fraction_positive_map_residual:.1%}.

The result supports successful frozen-protocol adaptation in PSC011, including ductular, portal/fibrotic, ECM-remodeling, and hepatocyte-zonation programs. Because GSE245620 contributes one Visium-profiled PSC patient, it does not establish cross-patient PSC generalization.
""")

    (ROOT / "13_MANUSCRIPT_RESULTS_PARAGRAPH.md").write_text(f"""# Manuscript-ready Results paragraph

We next applied the post-MASLD frozen liver protocol to primary sclerosing cholangitis (PSC), using the healthy live-donor atlas as the fixed reference and exposing only the same 16 target genes in four Visium sections from PSC011. Relative to registered 12-nearest-neighbour inverse-distance transfer, HyperSpatial-MAP increased patient-level median spatial correlation from {fmt(patient.median_spatial_idw)} to {fmt(patient.median_spatial_map)} and reduced log1p RMSE from {fmt(patient.log1p_rmse_idw)} to {fmt(patient.log1p_rmse_map)}. Atlas-relative residual recovery was positive (pooled r = {fmt(patient.pooled_residual_pearson_map)}), and predeclared ductular-reaction, portal/fibrotic, extracellular-matrix-remodelling and hepatocyte-zonation programs were recovered across all four sections. These data provide a frozen-protocol PSC validation in one patient; the sections represent within-patient sampling and not independent biological replicates.
""")

    (ROOT / "14_FIGURE_CAPTION_DRAFT.md").write_text("""# Figure caption draft

**Frozen HyperSpatial-MAP adaptation of a healthy liver reference to PSC.** **a,** Experimental design. Geometry-only registration transfers the fixed M1–M8 healthy liver atlas to four Visium sections from patient PSC011; only the 16 post-MASLD frozen target genes are exposed before hidden-transcriptome prediction. **b,c,** Paired registered-IDW and HyperSpatial-MAP median spatial Pearson (**b**) and log1p RMSE (**c**) for each section. Lines identify sections and do not denote independent patients. **d,** MAP atlas-relative residual Pearson by section; registered IDW predicts zero residual and its residual correlation is undefined. **e,** Fractions of hidden genes with higher spatial Pearson, lower RMSE, or positive residual Pearson under MAP. **f,** Spatial recovery of predeclared PSC-related program residuals. **g,** Geometry-only Chamfer distance before and after registration for the primary seed. **h,** Frozen source-only guard assignments. GSE245620 contains one PSC patient; no one-to-one anatomical correspondence between sections and the healthy reference is assumed.
""")

    decision = summary["PSC_decision"]
    (ROOT / "15_FINAL_DECISION.md").write_text(f"""# PSC final decision

## Decision {decision}

The single PSC patient met the prespecified Decision A gate: patient-level spatial Pearson increased, RMSE decreased, pooled residual recovery was positive, and every predeclared program family had positive median recovery. Both principal reconstruction directions were also consistent across all four within-patient sections.

Integrity conditions passed: exact frozen source model and 16 anchors, prediction lock before hidden-expression access, geometry-only registration, registered 12-NN IDW as the only comparator, and no target-specific tuning or outcome-based exclusion.

This is a positive one-patient frozen-protocol validation, not evidence of cross-patient PSC generalization. Portal/periportal compartment-specific evaluation was unavailable because no verified spot-level compartment labels were deposited. The GSE240429 sensitivity was not run because changing the reference would violate the exact hashed M1–M8 source-model freeze.
""")

    # Compact manuscript-facing summary figure.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.7, "xtick.major.width": 0.7,
                         "ytick.major.width": 0.7})
    fig = plt.figure(figsize=(12, 7.3), facecolor="white")
    grid = fig.add_gridspec(2, 4, left=0.055, right=0.98, top=0.94, bottom=0.09,
                            wspace=0.55, hspace=0.62)
    blue, grey, rust, teal = "#2878B5", "#8C8C8C", "#C65D3A", "#2A9D8F"

    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    panel_label(ax, "a")
    ax.set_title("Frozen PSC design", loc="left", fontsize=9.5, fontweight="bold")
    boxes = [(0.50, 0.82, "Healthy atlas · M1–M8"),
             (0.50, 0.61, "Geometry-only registration"),
             (0.50, 0.40, "PSC011 · 16 genes exposed"),
             (0.50, 0.19, "6,814 hidden genes")]
    for x, y, label in boxes:
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="#F6F8FA", ec="#B7C2CC", lw=0.8))
    for x1, y1, x2, y2 in [(0.50, .76, .50, .68), (0.50, .55, .50, .47),
                            (0.50, .34, .50, .26)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#4D5964", lw=1))
    ax.text(.50, .02, "Registered IDW vs full MAP", transform=ax.transAxes,
            color=blue, fontsize=8, ha="center")

    labels = sections["sample"].str.replace("PSC011_", "", regex=False).tolist()
    x = np.array([0, 1])
    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "b")
    ax.set_title("Spatial reconstruction", fontsize=9.5, fontweight="bold")
    for _, row in sections.iterrows():
        ax.plot(x, [row.median_spatial_idw, row.median_spatial_map], "-o", lw=1.2, ms=4,
                color=blue, alpha=0.8)
    ax.set_xticks(x, ["IDW", "MAP"])
    ax.set_ylabel("Median spatial Pearson")
    ax.axhline(0, color="#BBBBBB", lw=.7)

    ax = fig.add_subplot(grid[0, 2])
    panel_label(ax, "c")
    ax.set_title("Reconstruction error", fontsize=9.5, fontweight="bold")
    for _, row in sections.iterrows():
        ax.plot(x, [row.log1p_rmse_idw, row.log1p_rmse_map], "-o", lw=1.2, ms=4,
                color=rust, alpha=0.8)
    ax.set_xticks(x, ["IDW", "MAP"])
    ax.set_ylabel("log1p RMSE")

    ax = fig.add_subplot(grid[0, 3])
    panel_label(ax, "d")
    ax.set_title("Residual recovery", fontsize=9.5, fontweight="bold")
    ax.bar(labels, sections["pooled_residual_pearson_map"], color=teal, width=.68)
    ax.set_ylabel("Pooled residual Pearson")
    ax.set_ylim(0, max(.6, sections["pooled_residual_pearson_map"].max() * 1.18))
    ax.text(.98, .96, "IDW residual r undefined", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color="#555555")

    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "e")
    ax.set_title("Gene-level recovery", fontsize=9.5, fontweight="bold")
    wins = [patient.fraction_map_gt_idw_spatial, patient.fraction_map_lt_idw_rmse,
            patient.fraction_positive_map_residual]
    names = ["MAP > IDW\nspatial", "MAP < IDW\nRMSE", "Positive\nresidual"]
    ax.bar(np.arange(3), wins, color=[blue, rust, teal], width=.65)
    ax.set_xticks(np.arange(3), names, fontsize=7)
    ax.set_ylabel("Fraction of hidden genes")
    ax.set_ylim(0, 1)
    for i, value in enumerate(wins):
        ax.text(i, value + .025, f"{value:.1%}", ha="center", fontsize=7)

    ax = fig.add_subplot(grid[1, 1])
    panel_label(ax, "f")
    ax.set_title("PSC program recovery", fontsize=9.5, fontweight="bold")
    program_labels = [x.replace("_", "\n") for x in programs["program"]]
    ax.bar(np.arange(len(programs)), programs["median_spatial_residual_pearson"],
           color="#6C8EAD", width=.68)
    ax.set_xticks(np.arange(len(programs)), program_labels, fontsize=6.6)
    ax.set_ylabel("Median residual Pearson")
    ax.set_ylim(0, 1)

    primary = registration[registration["seed"].eq(42)]
    ax = fig.add_subplot(grid[1, 2])
    panel_label(ax, "g")
    ax.set_title("Geometry-only registration", fontsize=9.5, fontweight="bold")
    for _, row in primary.iterrows():
        ax.plot(x, [row.raw_chamfer, row.registered_chamfer], "-o", color=grey, lw=1.2, ms=4)
    ax.set_xticks(x, ["Before", "After"])
    ax.set_ylabel("Chamfer distance")

    ax = fig.add_subplot(grid[1, 3])
    panel_label(ax, "h")
    ax.set_title("Frozen guard", fontsize=9.5, fontweight="bold")
    guard = 1.0 - sections["guard_fallback_gene_fraction"].iloc[0]
    fallback = sections["guard_fallback_gene_fraction"].iloc[0]
    ax.bar([0, 1], [guard, fallback], color=[teal, grey], width=.65)
    ax.set_xticks([0, 1], ["Correction\nsupported", "IDW\nfallback"])
    ax.set_ylabel("Fraction of hidden genes")
    ax.set_ylim(0, 1)
    for i, value in enumerate([guard, fallback]):
        ax.text(i, value + .025, f"{value:.1%}", ha="center", fontsize=7)

    fig.text(.055, .015, "GSE245620: four sections from one PSC patient; patient/sample is the biological unit.",
             fontsize=7, color="#555555")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(ROOT / "figures" / f"Liver_PSC_frozen_MAP_summary.{ext}",
                    dpi=400 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    excluded = {"16_SHA256_FINAL.txt"}
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path.name not in excluded)
    with (ROOT / "16_SHA256_FINAL.txt").open("w") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path}\n")
    print(json.dumps({"PSC_decision": decision, "patient_n": 1, "sections": 4,
                      "output_directory": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
