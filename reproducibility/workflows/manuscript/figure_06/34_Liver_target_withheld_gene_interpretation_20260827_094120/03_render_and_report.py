#!/usr/bin/env python3
"""Render frozen target-withheld maps and write manuscript support text."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MASLD = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
PSC = PROJECT / "PI_revision/28_Liver_PSC_MAP_20260825_175802"
AH = PROJECT / "PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254"
EXPECTED_LOCK_SHA256 = "4d23e9a3c3079ef8cc78d0fe0d8200bdedc80da161a0f0f6ecb7cc4c60dcc0f1"
COLORS = {"MASLD": "#C7832B", "PSC": "#16858C", "AH": "#B54A69"}
FIXED_UNITS = {"MASLD": "VLP115_A_C1", "PSC": "PSC011_A1", "AH": "Sah73"}
ROOTS = {"MASLD": MASLD, "PSC": PSC, "AH": AH}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def load_matrix(pair: Path, kind: str):
    lock = json.loads((pair / "prediction_lock_manifest.json").read_text())
    if kind == "h5":
        with h5py.File(lock["target_matrix_path"], "r") as handle:
            group = handle["matrix"]
            matrix = csc_matrix(
                (group["data"][:], group["indices"][:], group["indptr"][:]),
                shape=tuple(int(x) for x in group["shape"][:]),
            ).tocsr()
            ids = decode(group["features/id"][:])
            names = decode(group["features/name"][:])
    else:
        matrix_path = Path(lock["target_matrix"])
        feature_path = Path(str(matrix_path).replace("_matrix.mtx.gz", "_features.tsv.gz"))
        features = pd.read_csv(feature_path, sep="\t", header=None,
                               names=["gene_id", "gene", "feature_type"])
        with gzip.open(matrix_path, "rb") as handle:
            matrix = mmread(handle).tocsr()
        ids = features["gene_id"].astype(str).to_numpy()
        names = features["gene"].astype(str).to_numpy()

    spots = matrix.T.tocsr()
    nfeature = np.diff(spots.indptr)
    total = np.asarray(spots.sum(axis=1)).ravel()
    mito = np.char.startswith(names.astype(str), "MT-")
    mito_total = np.asarray(matrix[mito].sum(axis=0)).ravel() if mito.any() else np.zeros(matrix.shape[1])
    qc = (nfeature > 10) & (100 * mito_total / np.maximum(total, 1) < 10)
    return lock, matrix, ids, qc


def load_gene_maps(disease: str, genes: list[str]):
    root = ROOTS[disease]
    pair = root / "work" / FIXED_UNITS[disease]
    lock, matrix, feature_ids, spot_qc = load_matrix(pair, "h5" if disease == "MASLD" else "mtx")
    hidden_names = np.asarray(lock["hidden_gene_names"])
    hidden_ids = np.asarray(lock["hidden_gene_ids"])
    hidden_lookup = {gene: i for i, gene in enumerate(hidden_names)}
    row_lookup = {gene_id: i for i, gene_id in enumerate(feature_ids)}
    obj = np.load(pair / "target_anchor_only_object.npz", allow_pickle=True)
    tissue_rows = obj["target_rows"].astype(int)
    roi = np.load(pair / "geometry_only_target_roi_rows.npy").astype(int)
    target_columns = tissue_rows[roi]
    local_qc = spot_qc[target_columns]
    target_columns = target_columns[local_qc]
    depth = np.load(pair / "source_model_predicted_target_depth.npy")[local_qc]
    source = np.load(lock["source_model"], allow_pickle=True)
    depth_target = float(np.asarray(source["source_depth_target"]).reshape(-1)[0])
    prior_all = np.load(lock["registered_idw_prediction"], mmap_mode="r")
    map_all = np.load(lock["map_prediction"], mmap_mode="r")
    xy = obj["target_coordinates_raw"][roi][local_qc].astype(float)
    xy -= np.nanmean(xy, axis=0)
    xy /= max(float(np.nanmax(np.ptp(xy, axis=0))), 1.0)
    output = {}
    for gene in genes:
        j = hidden_lookup[gene]
        row = row_lookup[hidden_ids[j]]
        counts = np.asarray(matrix[row, target_columns].toarray()).ravel().astype(np.float32)
        truth = np.log1p(counts * depth_target / depth)
        prior = np.asarray(prior_all[local_qc, j])
        pred = np.asarray(map_all[local_qc, j])
        output[gene] = {
            "xy": xy, "truth": truth, "prior": prior, "pred": pred,
            "measured_residual": truth - prior,
            "predicted_residual": pred - prior,
        }
    return output


def render(rows: pd.DataFrame, maps: dict, stem: str, title: str):
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(14.8, 1.72 * n + 1.0), facecolor="white", squeeze=False)
    headers = ["Target truth", "Healthy-atlas prior (IDW)", "MAP reconstruction",
               "Measured residual", "Predicted residual"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontweight="bold", pad=7)
    for row_index, record in enumerate(rows.itertuples(index=False)):
        z = maps[(record.disease, record.gene)]
        values = [z["truth"], z["prior"], z["pred"], z["measured_residual"], z["predicted_residual"]]
        expression_low = min(float(np.min(v)) for v in values[:3])
        expression_high = max(float(np.max(v)) for v in values[:3])
        residual_max = max(float(np.max(np.abs(values[3]))), float(np.max(np.abs(values[4]))), 1e-6)
        for col, values_column in enumerate(values):
            ax = axes[row_index, col]
            norm = (Normalize(expression_low, expression_high) if col < 3 else
                    TwoSlopeNorm(vmin=-residual_max, vcenter=0, vmax=residual_max))
            ax.scatter(z["xy"][:, 0], z["xy"][:, 1], c=values_column,
                       s=3.1, cmap="viridis" if col < 3 else "coolwarm",
                       norm=norm, linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(
                    f"{record.disease} · {FIXED_UNITS[record.disease]}\n{record.gene} · {record.predeclared_program}",
                    rotation=0, ha="right", va="center", labelpad=18,
                    fontsize=7.4, color=COLORS[record.disease], fontweight="bold"
                )
            if col == 4:
                ax.text(1.03, .5, f"Expression {expression_low:.2f}–{expression_high:.2f}\nResidual ±{residual_max:.2f}",
                        transform=ax.transAxes, va="center", fontsize=6.1, color="#5B6166")
    fig.suptitle(title, x=.075, y=.997, ha="left", fontweight="bold", fontsize=13)
    fig.text(.075, .006,
             "Fixed display units were chosen before this evaluation. Expression scales are shared across truth, IDW and MAP within each row; residual scales are symmetric and shared within each row.",
             ha="left", fontsize=7, color="#5B6166")
    fig.subplots_adjust(left=.19, right=.925, top=.965, bottom=.025, hspace=.10, wspace=.025)
    for extension, dpi in [("pdf", 300), ("svg", 300), ("png", 300)]:
        fig.savefig(ROOT / "figures" / f"{stem}.{extension}", dpi=dpi)
    plt.close(fig)


def write_text(lock: pd.DataFrame, summary: pd.DataFrame, examples: pd.DataFrame):
    def q(disease, gene):
        return summary[(summary.disease == disease) & (summary.gene == gene)].iloc[0]

    m_tat, m_dcn, m_nf, m_scd = [q("MASLD", g) for g in ["TAT", "DCN", "NFKBIA", "SCD"]]
    p_tat, p_dcn, p_krt = [q("PSC", g) for g in ["TAT", "DCN", "KRT8"]]
    a_dcn, a_krt, a_nf, a_cyp = [q("AH", g) for g in ["DCN", "KRT8", "NFKBIA", "CYP3A4"]]

    per_unit = pd.read_csv(ROOT / "TARGET_WITHHELD_GENE_RESULTS_PER_UNIT.tsv", sep="\t")
    scd_paired_delta = per_unit[(per_unit.disease == "MASLD") & (per_unit.gene == "SCD")].delta_RMSE.median()

    manuscript = f"""# Target-withheld disease-gene manuscript text

## A. Concise Results paragraph

Using only 16 measured genes from each target, HyperSpatial-MAP reconstructed the disease-state spatial expression of genes that were withheld from the target input but represented in the healthy source atlas. In MASLD, the prelocked zonation gene **TAT** increased from a median IDW spatial Pearson correlation of {m_tat.IDW_spatial_Pearson:.3f} to {m_tat.MAP_spatial_Pearson:.3f}, with RMSE decreasing from {m_tat.IDW_RMSE:.3f} to {m_tat.MAP_RMSE:.3f} and positive residual recovery (median r={m_tat.residual_Pearson:.3f}); the fibrosis-associated matrix gene **DCN** and inflammatory regulator **NFKBIA** showed concordant improvements. Across all four sections from the single PSC patient, **TAT**, **DCN**, **KRT8** and **TIMP1** each improved over registered atlas transfer for both spatial correlation and RMSE and showed positive atlas-relative residual recovery. In AH, **DCN**, **KRT8**, **TUBB** and **NFKBIA** improved in both independent patients, capturing fibrotic, injury, regenerative and inflammatory components of the withheld disease state. Not all predefined examples were uniformly recovered: MASLD **SCD** did not improve median RMSE, and AH **CYP3A4** improved spatial correlation in only one of two patients.

## B. Interpretation

These results connect the genome-wide and program-level benchmarks to specific biological transcripts without selecting genes on prediction performance. They show that sparse target measurements can adapt a healthy atlas toward spatially localized metabolic, fibrotic, epithelial-injury and inflammatory states, while the mixed SCD and CYP3A4 results define the limits of individual-gene recovery.

## C. Abstract/discussion sentence

Using only 16 measured target genes, HyperSpatial-MAP recovered specimen-specific spatial states of predeclared disease-relevant genes that were withheld from the target input across metabolic/fibrotic, cholangiopathic and inflammatory liver disease.

## D. Concrete examples

### MASLD

- **TAT** — withheld from the 16 target anchors; represents periportal amino-acid metabolism and hepatocyte zonation; MAP increased median spatial r from {m_tat.IDW_spatial_Pearson:.3f} to {m_tat.MAP_spatial_Pearson:.3f} and recovered residual structure (r={m_tat.residual_Pearson:.3f}).
- **DCN** — withheld; represents stromal matrix/fibrotic remodeling; MAP increased median spatial r from {m_dcn.IDW_spatial_Pearson:.3f} to {m_dcn.MAP_spatial_Pearson:.3f}, reduced RMSE from {m_dcn.IDW_RMSE:.3f} to {m_dcn.MAP_RMSE:.3f}, and yielded residual r={m_dcn.residual_Pearson:.3f}.
- **NFKBIA** — withheld; represents inflammatory NF-kappaB feedback; MAP increased median spatial r from {m_nf.IDW_spatial_Pearson:.3f} to {m_nf.MAP_spatial_Pearson:.3f} and yielded residual r={m_nf.residual_Pearson:.3f}.
- **SCD** — withheld lipid/steatosis candidate; spatial correlation and residual recovery were positive. Although the separate method-wise RMSE medians were {m_scd.IDW_RMSE:.3f} (IDW) and {m_scd.MAP_RMSE:.3f} (MAP), RMSE improved in only 14/33 biopsies and the median within-biopsy paired ΔRMSE was {scd_paired_delta:+.3f}; it is retained as a mixed predefined result.

### PSC

- **TAT** — withheld zonation gene; MAP increased section-median spatial r from {p_tat.IDW_spatial_Pearson:.3f} to {p_tat.MAP_spatial_Pearson:.3f} with residual r={p_tat.residual_Pearson:.3f}.
- **DCN** — withheld portal/fibrotic gene; MAP increased spatial r from {p_dcn.IDW_spatial_Pearson:.3f} to {p_dcn.MAP_spatial_Pearson:.3f} and reduced RMSE from {p_dcn.IDW_RMSE:.3f} to {p_dcn.MAP_RMSE:.3f}.
- **KRT8** — withheld ductular-reaction gene; MAP increased spatial r from {p_krt.IDW_spatial_Pearson:.3f} to {p_krt.MAP_spatial_Pearson:.3f} with residual r={p_krt.residual_Pearson:.3f}. These are four sections from one patient, not four patient replicates.

### Alcohol-associated hepatitis

- **DCN** — withheld fibrosis/ECM gene; MAP improved both endpoints in both patients, with balanced median residual r={a_dcn.residual_Pearson:.3f}.
- **KRT8** — withheld hepatocyte injury/stress gene; MAP increased balanced median spatial r from {a_krt.IDW_spatial_Pearson:.3f} to {a_krt.MAP_spatial_Pearson:.3f} and yielded residual r={a_krt.residual_Pearson:.3f}.
- **NFKBIA** — withheld inflammatory/immune gene; MAP increased spatial r from {a_nf.IDW_spatial_Pearson:.3f} to {a_nf.MAP_spatial_Pearson:.3f} and reduced RMSE from {a_nf.IDW_RMSE:.3f} to {a_nf.MAP_RMSE:.3f}.
- **CYP3A4** — withheld metabolic gene; RMSE and residual recovery were favorable in both patients, but spatial correlation improved in only one of two patients, so it is not proposed as a main-text success example.

## Relationship to the current integrated figure

- **Panel b** reports genome-wide reconstruction accuracy across all hidden genes.
- **Panel c** reports recovery of atlas-relative, target-specific molecular deviations.
- **Panel d** aggregates recovery across predeclared disease programs.
- **Panel e** provides individual target-withheld gene examples.

This analysis supplies the biological gene-level interpretation requested for the manuscript and complements rather than duplicates the program-level evidence in Panel d.
"""
    (ROOT / "TARGET_WITHHELD_GENE_MANUSCRIPT_TEXT.md").write_text(manuscript)

    anchor_lines = []
    for disease, root in ROOTS.items():
        a = pd.read_csv(root / "01_ANCHORS.tsv", sep="\t")
        anchor_lines.append(f"- {disease}: " + ", ".join(a["gene"].astype(str)))
    candidate_lines = []
    for disease in ["MASLD", "PSC", "AH"]:
        x = lock[lock.disease == disease]
        candidate_lines.append("- " + disease + ": " + ", ".join(f"{r.gene} [{r.predeclared_program}]" for r in x.itertuples()))
    success = examples[examples.appropriate_for_main_text_yes_no == "yes"]
    mixed = examples[examples.appropriate_for_main_text_yes_no == "no"]
    summary_md = f"""# Run summary

## Integrity

- Predictions recomputed: **NO**
- Model, anchors, registration, operators, guard and metrics changed: **NO**
- Candidate selection preceded candidate-level target evaluation: **YES**
- Candidate-lock SHA256: `{EXPECTED_LOCK_SHA256}`
- Frozen prediction files were read only after the lock was written and verified.

## Exact frozen anchors

{chr(10).join(anchor_lines)}

## Hidden-gene availability

- Each canonical prediction manifest locked the same **6,814** hidden genes.
- After the frozen post-lock per-unit expression/QC filters, evaluated counts were MASLD **6,806–6,813**, PSC **6,813–6,814**, and AH **6,793–6,804** hidden genes per unit.

## Locked candidates

{chr(10).join(candidate_lines)}

## Main-text-suitable prelocked examples

- MASLD: **TAT, DCN, NFKBIA**.
- PSC: **TAT, DCN, KRT8**; section consistency is within one patient.
- AH: **DCN, KRT8, NFKBIA**.

## Mixed prelocked candidates retained transparently

- MASLD **SCD**: spatial Pearson improved in 31/33 biopsies and residual r was positive in 33/33. The separate method-wise RMSE medians were 0.894 (IDW) and 0.881 (MAP), but RMSE improved in only 14/33 biopsies and the median within-biopsy paired ΔRMSE was +0.018.
- AH **CYP3A4**: RMSE improved and residual r was positive in both patients, but spatial Pearson improved in only 1/2.

All 13 locked candidates remain in `TARGET_WITHHELD_GENE_RESULTS.tsv`, `TARGET_WITHHELD_GENE_RESULTS_PER_UNIT.tsv` and the all-candidate exploratory figure.
"""
    (ROOT / "RUN_SUMMARY.md").write_text(summary_md)


def main():
    if sha256(ROOT / "CANDIDATE_GENE_LOCK.tsv") != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Candidate lock changed before rendering")
    lock = pd.read_csv(ROOT / "CANDIDATE_GENE_LOCK.tsv", sep="\t")
    summary = pd.read_csv(ROOT / "TARGET_WITHHELD_GENE_RESULTS.tsv", sep="\t")
    examples = pd.read_csv(ROOT / "MANUSCRIPT_GENE_EXAMPLES.tsv", sep="\t")

    maps = {}
    for disease in ["MASLD", "PSC", "AH"]:
        genes = lock.loc[lock.disease == disease, "gene"].tolist()
        for gene, values in load_gene_maps(disease, genes).items():
            maps[(disease, gene)] = values

    render(lock, maps, "TARGET_WITHHELD_ALL_LOCKED_CANDIDATES",
           "Target-withheld disease genes: all prelocked candidates")
    compact_genes = {
        "MASLD": ["TAT", "DCN", "NFKBIA"],
        "PSC": ["TAT", "DCN", "KRT8"],
        "AH": ["DCN", "KRT8", "NFKBIA"],
    }
    compact = pd.concat([
        lock[(lock.disease == disease) & (lock.gene.isin(genes))]
        for disease, genes in compact_genes.items()
    ], ignore_index=True)
    compact["_order"] = [compact_genes[d].index(g) for d, g in zip(compact.disease, compact.gene)]
    compact["_dorder"] = compact.disease.map({"MASLD": 0, "PSC": 1, "AH": 2})
    compact = compact.sort_values(["_dorder", "_order"]).drop(columns=["_order", "_dorder"])
    render(compact, maps, "TARGET_WITHHELD_MANUSCRIPT_EXAMPLES_COMPACT",
           "Target-withheld disease-gene reconstruction examples")
    write_text(lock, summary, examples)
    print("Rendered all locked candidates and compact manuscript examples")


if __name__ == "__main__":
    main()
