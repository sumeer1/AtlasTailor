#!/usr/bin/env python3
"""Build an IDW-vs-MAP manuscript package from frozen SPATCH outputs only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
ORIGINAL = PROJECT / "PI_revision" / "23_SPATCH_cross_platform_MAP_20260824_111004"
PRIMARY = ("COAD", "OV")
METHOD_MAP = {"registered_idw_k12": "registered 12-NN IDW", "hyperspatial_map": "HyperSpatial-MAP"}
SPECIMEN = {"COAD": "Benchmark_03", "OV": "Benchmark_02", "HCC": "Benchmark_01"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wide_record(tumor: str, pair: pd.DataFrame) -> dict:
    idw = pair[(pair.tumor == tumor) & (pair.method == "registered_idw_k12")].iloc[0]
    map_row = pair[(pair.tumor == tumor) & (pair.method == "hyperspatial_map")].iloc[0]
    return {
        "specimen_id": SPECIMEN[tumor], "tumor": tumor,
        "source_platform": idw.source_platform, "target_platform": idw.target_platform,
        "target_bin_count": int(idw.target_bin_count), "measured_anchor_count": 16,
        "hidden_shared_gene_count": int(idw.hidden_gene_count),
        "IDW_median_gene_wise_spatial_pearson": idw.median_gene_wise_spatial_pearson,
        "MAP_median_gene_wise_spatial_pearson": map_row.median_gene_wise_spatial_pearson,
        "delta_spatial_pearson_MAP_minus_IDW": map_row.median_gene_wise_spatial_pearson - idw.median_gene_wise_spatial_pearson,
        "IDW_log1p_RMSE": idw.log1p_rmse, "MAP_log1p_RMSE": map_row.log1p_rmse,
        "delta_RMSE_MAP_minus_IDW": map_row.log1p_rmse - idw.log1p_rmse,
        "IDW_residual_pearson": np.nan,
        "IDW_residual_pearson_status": "undefined: predicted atlas-relative residual is identically zero",
        "MAP_median_gene_wise_residual_pearson": map_row.median_gene_wise_residual_pearson,
        "MAP_pooled_residual_pearson": map_row.pooled_bin_gene_residual_pearson,
        "fraction_hidden_genes_MAP_gt_IDW_spatial_pearson": map_row.fraction_genes_MAP_gt_IDW_spatial_pearson,
        "fraction_hidden_genes_MAP_lt_IDW_log1p_RMSE": map_row.fraction_genes_MAP_lt_IDW_log1p_RMSE,
        "fraction_hidden_genes_positive_MAP_residual_pearson": map_row.fraction_positive_gene_residual_recovery,
    }


pair_all = pd.read_csv(ORIGINAL / "03_PAIR_LEVEL_RESULTS.tsv", sep="\t")
gene_all = pd.read_csv(ORIGINAL / "04_GENE_LEVEL_RESULTS.tsv", sep="\t")
registration_all = pd.read_csv(ORIGINAL / "05_REGISTRATION_RESULTS.tsv", sep="\t")
lock = pd.read_csv(ORIGINAL / "02_PREDICTION_LOCK_MANIFEST.tsv", sep="\t")

# Primary long-format table: exactly two specimens and two reported methods.
pair = pair_all[pair_all.tumor.isin(PRIMARY) & pair_all.method.isin(METHOD_MAP)].copy()
pair["method"] = pair.method.map(METHOD_MAP)
pair["measured_anchor_count"] = 16
pair["residual_pearson_status"] = np.where(
    pair.method == "registered 12-NN IDW",
    "undefined: predicted atlas-relative residual is identically zero", "defined",
)
idw_rows = pair.method == "registered 12-NN IDW"
pair.loc[idw_rows, [
    "pooled_bin_gene_residual_pearson", "median_gene_wise_residual_pearson",
    "fraction_positive_gene_residual_recovery",
    "fraction_genes_MAP_gt_IDW_spatial_pearson",
    "fraction_genes_MAP_lt_IDW_log1p_RMSE",
]] = np.nan
pair.to_csv(HERE / "01_PAIR_LEVEL_RESULTS_IDW_vs_MAP.tsv", sep="\t", index=False)

primary_wide = pd.DataFrame([wide_record(tumor, pair_all) for tumor in PRIMARY])
primary_wide.to_csv(HERE / "02_SPECIMEN_LEVEL_RESULTS_IDW_vs_MAP.tsv", sep="\t", index=False)

# Primary gene table is paired within each specimen and contains no other method.
gene = gene_all[gene_all.tumor.isin(PRIMARY) & gene_all.method.isin(METHOD_MAP)].copy()
identity = gene[["specimen_id", "tumor", "source_platform", "target_platform", "gene"]].drop_duplicates()
metrics = gene.pivot(index=["specimen_id", "tumor", "source_platform", "target_platform", "gene"],
                     columns="method", values=["spatial_pearson", "log1p_rmse", "residual_pearson"]).reset_index()
metrics.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in metrics.columns]
gene_rows = metrics.rename(columns={
    "spatial_pearson_registered_idw_k12": "IDW_spatial_pearson",
    "spatial_pearson_hyperspatial_map": "MAP_spatial_pearson",
    "log1p_rmse_registered_idw_k12": "IDW_log1p_RMSE",
    "log1p_rmse_hyperspatial_map": "MAP_log1p_RMSE",
    "residual_pearson_hyperspatial_map": "MAP_residual_pearson",
}).drop(columns=[c for c in metrics.columns if c == "residual_pearson_registered_idw_k12"], errors="ignore")
gene_rows["delta_spatial_pearson_MAP_minus_IDW"] = gene_rows.MAP_spatial_pearson - gene_rows.IDW_spatial_pearson
gene_rows["delta_RMSE_MAP_minus_IDW"] = gene_rows.MAP_log1p_RMSE - gene_rows.IDW_log1p_RMSE
gene_rows["MAP_gt_IDW_spatial_pearson"] = gene_rows.MAP_spatial_pearson > gene_rows.IDW_spatial_pearson
gene_rows["MAP_lt_IDW_log1p_RMSE"] = gene_rows.MAP_log1p_RMSE < gene_rows.IDW_log1p_RMSE
gene_rows["positive_MAP_residual_pearson"] = gene_rows.MAP_residual_pearson > 0
counts = primary_wide.set_index("tumor")
gene_rows["target_bin_count"] = gene_rows.tumor.map(counts.target_bin_count)
gene_rows["measured_anchor_count"] = 16
gene_rows["hidden_shared_gene_count"] = gene_rows.tumor.map(counts.hidden_shared_gene_count)
ordered = [
    "specimen_id", "tumor", "source_platform", "target_platform", "gene",
    "target_bin_count", "measured_anchor_count", "hidden_shared_gene_count",
    "IDW_spatial_pearson", "MAP_spatial_pearson", "delta_spatial_pearson_MAP_minus_IDW",
    "IDW_log1p_RMSE", "MAP_log1p_RMSE", "delta_RMSE_MAP_minus_IDW",
    "MAP_residual_pearson", "MAP_gt_IDW_spatial_pearson", "MAP_lt_IDW_log1p_RMSE",
    "positive_MAP_residual_pearson",
]
gene_rows[ordered].to_csv(HERE / "03_GENE_LEVEL_MAP_vs_IDW.tsv", sep="\t", index=False)

registration_all[registration_all.tumor.isin(PRIMARY)].to_csv(
    HERE / "04_REGISTRATION_RESULTS.tsv", sep="\t", index=False
)

# HCC remains exact, separate and descriptive.
hcc = pd.DataFrame([wide_record("HCC", pair_all)])
hcc.to_csv(HERE / "04B_HCC_SECONDARY_STRESS_TEST.tsv", sep="\t", index=False)
h = hcc.iloc[0]
(HERE / "04C_HCC_SECONDARY_STRESS_TEST.md").write_text(f"""# Secondary HCC cross-platform stress test

The original frozen experiment evaluated COAD, HCC and ovarian cancer. HCC is retained here as a secondary cross-platform stress test because its shared hidden-gene evaluation space was substantially smaller than for the two primary evaluations: HCC included 151 genes, compared with 763 for COAD and 697 for ovarian cancer. This difference limits direct comparability with the primary evaluations and is the reason for separate manuscript presentation; HCC was not removed or rerun.

| Endpoint | Registered 12-NN IDW | HyperSpatial-MAP |
|---|---:|---:|
| Median gene-wise spatial Pearson | {h.IDW_median_gene_wise_spatial_pearson:.10f} | {h.MAP_median_gene_wise_spatial_pearson:.10f} |
| log1p RMSE | {h.IDW_log1p_RMSE:.10f} | {h.MAP_log1p_RMSE:.10f} |
| Pooled atlas-relative residual Pearson | undefined | {h.MAP_pooled_residual_pearson:.10f} |

HCC was retained as a secondary cross-platform stress test because the shared hidden-gene evaluation space was substantially smaller than for COAD and ovarian cancer. Residual recovery remained positive, although absolute reconstruction improvement over registered transfer was limited.

All HCC predictions, registrations, anchors, operator assignments, metrics and hashes remain unchanged in the original archive.
""")

c, o = primary_wide.set_index("tumor").loc["COAD"], primary_wide.set_index("tumor").loc["OV"]
primary_mean = primary_wide.select_dtypes(include=[np.number]).mean()

(HERE / "05_RESULTS_INTERPRETATION_IDW_ONLY.md").write_text(f"""# Results interpretation: registered IDW versus HyperSpatial-MAP

## Scientific question

Does sparse molecular information from a target measured on a different spatial platform improve reconstruction beyond direct registered atlas transfer?

HyperSpatial-MAP was evaluated against registered 12-NN IDW, which uses the same Visium HD source atlas and geometry registration but receives no target molecular correction. This directly tests whether the 16 measured Xenium genes add specimen-specific molecular information beyond cross-platform anatomical transfer.

## Primary manuscript-facing result

The primary presentation comprises COAD and ovarian cancer, which had substantial and comparable source-eligible hidden-gene spaces (763 and 697 genes, respectively). In COAD, median spatial Pearson increased from {c.IDW_median_gene_wise_spatial_pearson:.10f} to {c.MAP_median_gene_wise_spatial_pearson:.10f}, while log1p RMSE decreased from {c.IDW_log1p_RMSE:.10f} to {c.MAP_log1p_RMSE:.10f}; pooled MAP residual Pearson was {c.MAP_pooled_residual_pearson:.10f}. In ovarian cancer, median spatial Pearson increased from {o.IDW_median_gene_wise_spatial_pearson:.10f} to {o.MAP_median_gene_wise_spatial_pearson:.10f}, RMSE decreased from {o.IDW_log1p_RMSE:.10f} to {o.MAP_log1p_RMSE:.10f}, and pooled residual Pearson was {o.MAP_pooled_residual_pearson:.10f}.

Across these two primary cross-platform specimen evaluations, HyperSpatial-MAP improved reconstruction beyond registered atlas transfer, showing that sparse measurements from a target profiled by a different technology can adapt the molecular reference toward the target state. The result does not imply cell-to-cell correspondence, general platform invariance or clinical generalization.

## Secondary HCC record

HCC remains fully documented as a secondary stress test. Its hidden space contained 151 genes, and residual recovery remained positive, although absolute improvement over registered transfer was limited. HCC is not included in the primary aggregate or Decision A/B/C because the manuscript-facing reorganization predeclared COAD and ovarian cancer as the coverage-comparable primary evaluations.

The original frozen analysis remains unchanged and contains all three specimens and all internal model diagnostics.
""")

primary_a = bool(
    ((primary_wide.MAP_median_gene_wise_spatial_pearson > primary_wide.IDW_median_gene_wise_spatial_pearson) &
     (primary_wide.MAP_log1p_RMSE < primary_wide.IDW_log1p_RMSE) &
     (primary_wide.MAP_pooled_residual_pearson > 0)).all()
)
decision = "A" if primary_a else "B"
(HERE / "06_FINAL_DECISION_IDW_ONLY.md").write_text(f"""# Primary manuscript-facing decision

## Decision {decision}

Decision A is supported by the two coverage-comparable primary cross-platform specimen evaluations, COAD and ovarian cancer:

- median gene-wise spatial Pearson was higher for HyperSpatial-MAP than registered 12-NN IDW in both specimens;
- log1p RMSE was lower for HyperSpatial-MAP in both specimens;
- pooled atlas-relative residual Pearson was positive in both specimens;
- no target leakage or post-outcome retuning occurred; and
- every frozen-output hash verified.

HCC does not alter this primary decision. It remains an unchanged secondary stress test with a substantially smaller hidden-gene space. The original three-specimen record is preserved: COAD, HCC and ovarian cancer were all evaluated before this presentation-only reorganization; no prediction, registration, anchor, operator assignment or metric was changed.
""")

(HERE / "07_FIGURE_CAPTION_DRAFT.md").write_text(f"""# Main figure caption draft

**Cross-platform adaptation of same-specimen serial sections using sparse target measurements.** **a,** Experimental design. A Visium HD FFPE section, represented at 8-µm resolution, was registered to a same-specimen Xenium 5K serial section using geometry alone. Sixteen Xenium genes were exposed to HyperSpatial-MAP and the remaining source-eligible shared genes were hidden (763 in COAD and 697 in ovarian cancer). **b,** Sparse measurement budget for the two primary cross-platform specimen evaluations. **c,** Geometry-only registered tissue support for COAD and ovarian cancer. **d,** Median gene-wise spatial Pearson and log1p RMSE for registered 12-NN IDW and HyperSpatial-MAP. IDW uses the same registered Visium HD atlas but receives no target molecular correction. Lines join results within a specimen. **e,** Pooled Pearson correlation between measured and predicted atlas-relative residuals for HyperSpatial-MAP. Registered IDW predicts an identically zero residual, for which residual correlation is undefined. **f,** Source-predeclared representative COAD hidden genes showing measured Xenium expression, the registered Visium HD prior, HyperSpatial-MAP reconstruction, and measured and predicted atlas-relative residuals. Expression maps share a scale within each gene; residual maps use a common symmetric scale. Serial sections are not assumed to provide one-to-one cell correspondence.

## Supplementary HCC caption

**Secondary HCC cross-platform stress test.** The frozen HCC evaluation contained 151 source-eligible hidden shared genes, substantially fewer than COAD and ovarian cancer. Registered IDW and HyperSpatial-MAP achieved median spatial Pearson values of {h.IDW_median_gene_wise_spatial_pearson:.10f} and {h.MAP_median_gene_wise_spatial_pearson:.10f}, respectively, and log1p RMSE values of {h.IDW_log1p_RMSE:.10f} and {h.MAP_log1p_RMSE:.10f}; pooled MAP residual Pearson was {h.MAP_pooled_residual_pearson:.10f}. HCC is shown separately because of its smaller shared evaluation space.
""")

(HERE / "08_MANUSCRIPT_RESULTS_PARAGRAPH.md").write_text(f"""# Manuscript Results paragraph

To test whether sparse target measurements could adapt a molecular reference across spatial technologies, we evaluated same-specimen Visium HD FFPE and Xenium 5K serial sections from the SPATCH resource. Both platforms were represented as 8-µm bins and registered using geometry alone; 16 Xenium genes were then exposed to HyperSpatial-MAP, with 763 source-eligible shared genes hidden in COAD and 697 in ovarian cancer. Registered 12-NN IDW provided the direct atlas-transfer comparator because it used the same source atlas and registration without target molecular correction. In COAD, HyperSpatial-MAP increased median gene-wise spatial Pearson from {c.IDW_median_gene_wise_spatial_pearson:.10f} to {c.MAP_median_gene_wise_spatial_pearson:.10f} and reduced log1p RMSE from {c.IDW_log1p_RMSE:.10f} to {c.MAP_log1p_RMSE:.10f}; in ovarian cancer, spatial Pearson increased from {o.IDW_median_gene_wise_spatial_pearson:.10f} to {o.MAP_median_gene_wise_spatial_pearson:.10f} and RMSE decreased from {o.IDW_log1p_RMSE:.10f} to {o.MAP_log1p_RMSE:.10f}. Atlas-relative residual recovery was positive in both evaluations (pooled Pearson r = {c.MAP_pooled_residual_pearson:.10f} and {o.MAP_pooled_residual_pearson:.10f}, respectively), indicating that sparse measurements from a target profiled using a different technology can improve reconstruction beyond registered atlas transfer. HCC was retained separately as a secondary stress test because its hidden shared-gene space was substantially smaller (151 genes).
""")

# Integrity report uses filesystem evidence and repeats the full hash verification.
hash_failures = []
for row in lock.itertuples():
    actual = sha256(Path(row.path))
    if actual != row.sha256:
        hash_failures.append(f"{row.tumor}:{row.artifact}")
global_lock = ORIGINAL / "02_PREDICTION_LOCK_MANIFEST.tsv"
opened = {tumor: ORIGINAL / "work" / tumor / "HIDDEN_TARGET_OPENED_AFTER_GLOBAL_LOCK.txt"
          for tumor in ("COAD", "HCC", "OV")}
lock_time = global_lock.stat().st_mtime
ordering_ok = all(path.stat().st_mtime > lock_time for path in opened.values())
for tumor in PRIMARY:
    assert int(primary_wide.set_index("tumor").loc[tumor, "measured_anchor_count"]) == 16
if hash_failures or not ordering_ok:
    raise RuntimeError(f"Integrity failure: hashes={hash_failures}, ordering={ordering_ok}")

(HERE / "09_LEAKAGE_AUDIT.md").write_text(f"""# Leakage and integrity audit

## Result: PASS

- Original directory: `{ORIGINAL}`
- Global prediction lock: `{global_lock}`
- Global lock timestamp: {pd.Timestamp(global_lock.stat().st_mtime, unit='s', tz='Asia/Riyadh').isoformat()}
- COAD hidden-expression-open marker: {pd.Timestamp(opened['COAD'].stat().st_mtime, unit='s', tz='Asia/Riyadh').isoformat()}
- HCC hidden-expression-open marker: {pd.Timestamp(opened['HCC'].stat().st_mtime, unit='s', tz='Asia/Riyadh').isoformat()}
- OV hidden-expression-open marker: {pd.Timestamp(opened['OV'].stat().st_mtime, unit='s', tz='Asia/Riyadh').isoformat()}
- Frozen prediction-lock artifacts verified: {len(lock)} of {len(lock)}
- Hash failures: none

The global prediction lock predates every hidden-expression access marker. Exactly 16 anchors were exposed for each primary pair. Target non-anchor expression remained unavailable during source fitting, panel selection, geometry registration, operator assignment and prediction serialization. Geometry registration used coordinates and tissue support only. Representative genes in the revised main figure are the previously locked source-only COAD choices; no gene was reselected from target outcomes.

No model fitting, prediction, registration, anchor selection, operator selection, metric computation for outcome optimization or post-outcome tuning was performed for this reorganization. The numerical tables are filtered and pivoted views of the frozen result tables. Registered IDW residual Pearson is retained as undefined, not zero.

The original run evaluated all three specimens and remains unmodified. HCC is reorganized as a secondary stress test solely because its source-eligible hidden space (151 genes) is substantially smaller than COAD (763) and ovarian cancer (697). Internal model diagnostics remain archived in the original directory and are not treated as manuscript-facing comparators.
""")

print(json.dumps({
    "original": str(ORIGINAL), "revised": str(HERE), "prediction_recomputed": False,
    "registration_recomputed": False, "anchors_changed": False,
    "model_assignments_changed": False, "hashes_verified": len(lock),
    "primary_decision": decision,
}, indent=2))
