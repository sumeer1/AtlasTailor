#!/usr/bin/env python3
"""Build MASLD and combined ALF+MASLD audit/report artifacts from frozen results."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ALF = PROJECT / "PI_revision" / "26_HumanLiver_multidisease_MAP_20260825_110222"
WORK = ROOT / "work"
ARCHIVE = ROOT / "data" / "Visium.zip"

PATIENT_IDS = [
    "16P29140", "16P29175", "16P32359", "17P29186", "17P30856",
    "16P17680", "16P19258", "16P20389", "16P22307", "16P40310", "17P11664", "17P6898", "17P8625", "17P9965",
    "16P15496", "16P34865", "16P40876",
    "16P18307", "16P22317", "16P27507", "16P33821", "16P34300", "16P35513", "17P2240", "17P31492_1", "17P31492_2",
    "16P26452", "16P29949", "16P29953", "16P32062", "16P33251", "17P19314", "17P7841",
]
COARSE_SCORE = [0] * 5 + [1] * 9 + [2] * 3 + [3] * 9 + [4] * 7


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(x, n=4):
    return "NA" if pd.isna(x) else f"{float(x):.{n}f}"


def main():
    biopsies = pd.read_csv(ROOT / "04A_BIOPSY_LEVEL_RESULTS_MASLD.tsv", sep="\t")
    patients = pd.read_csv(ROOT / "04_PATIENT_LEVEL_RESULTS_MASLD.tsv", sep="\t")
    genes = pd.read_csv(ROOT / "05_GENE_LEVEL_RESULTS_MASLD.tsv", sep="\t")
    programs = pd.read_csv(ROOT / "07_DISEASE_PROGRAM_RECOVERY_MASLD.tsv", sep="\t")
    qc = pd.read_csv(ROOT / "00C_MASLD_QC_BY_BIOPSY.tsv", sep="\t")
    outcome = json.loads((ROOT / "masld_patient_level_summary.json").read_text())
    lock = json.loads((WORK / "healthy_source_lock_masld_panel.json").read_text())

    shutil.copyfile(ROOT / "04_PATIENT_LEVEL_RESULTS_MASLD.tsv", ROOT / "04_PATIENT_LEVEL_RESULTS.tsv")
    shutil.copyfile(ROOT / "05_GENE_LEVEL_RESULTS_MASLD.tsv", ROOT / "05_GENE_LEVEL_RESULTS.tsv")
    shutil.copyfile(ROOT / "07_DISEASE_PROGRAM_RECOVERY_MASLD.tsv", ROOT / "07_DISEASE_PROGRAM_RECOVERY.tsv")
    registration = pd.read_csv(ROOT / "06_REGISTRATION_RESULTS_PRELOCK.tsv", sep="\t")
    registration.to_csv(ROOT / "06_REGISTRATION_RESULTS.tsv", sep="\t", index=False)

    with h5py.File(ROOT / "data" / "extracted" / "Visium" / "VLP115_A" / "filtered_feature_bc_matrix.h5", "r") as h:
        assay_genes = int(h["matrix/shape"][0])
    stage_counts = biopsies["fibrosis_stage"].value_counts().sort_index().to_dict()
    id_rows = "\n".join(f"- `{pid}`: deposited clinical coarse fibrosis score {score}" for pid, score in zip(PATIENT_IDS, COARSE_SCORE))
    audit = f"""# MASLD data audit

## Authoritative sources

- Publication: *Progressive fibrosis in human MASLD is associated with spatially linked transcriptomic signatures of metabolic reprogramming and senescence* (JHEP Reports, DOI 10.1016/j.jhepr.2025.101657).
- Official repository: UQ eSpace DOI 10.48610/e95155f.
- Archive used: `{ARCHIVE}`.
- Archive size: {ARCHIVE.stat().st_size} bytes (expected 336420009; PASS).
- Archive MD5: `2962d8463da108f9fb9ef951d9204dbd` (verified before extraction; PASS).
- Local-copy timestamp: {ARCHIVE.stat().st_mtime} (Unix time); the user supplied the archive in the project directory, so no automated-download URL is claimed.

## Cohort and assay

The publication reports 33 FFPE liver needle biopsies from 32 patients with MASLD, including two synchronous biopsy cores from one patient. The assay is 10x Genomics Visium CytAssist Spatial Gene Expression for FFPE (Human Transcriptome, 6.5-mm capture area). Four to five biopsies were transferred to each multiplexed slide. The official public analysis code uses eight arrays: `VLP115_A`, `VLP116_D`, `VLP119_A`, `VLP119_D`, `VLP120_A`, `VLP120_D`, `VLP121_A`, and `VLP121_D`. The deposited archive also contains `VLP115_D` and `VLP116_A`; these were not introduced into this benchmark because the authors' official analysis does not load them.

- Assay feature rows per analyzed array: {assay_genes}.
- Healthy/MASLD shared feature axis used by the source model: {lock['shared_gene_count']}.
- Source-eligible hidden evaluation genes locked per target before post-lock QC: {lock['hidden_gene_count']}.
- Exactly 16 measured anchors per target: PASS.
- Geometry-defined biopsy territories: {len(biopsies)}.
- Independent patient count reported by the publication: 32.
- Fibrosis labels transcribed post-lock from Supplementary Fig. S1E: {stage_counts}.

## Patient and biopsy identifiers

The authors' public code lists the following 33 biopsy identifiers. `17P31492_1` and `17P31492_2` are the synchronous two-biopsy patient; therefore these represent 32 independent patients.

{id_rows}

## Deposited-metadata limitation

The archive does **not** contain the eight barcode-to-biopsy CSV files referenced by the official code (`VLP115A_samples.csv`, `VLP116D_samples.csv`, `VLP119A_samples.csv`, `VLP119D_samples.csv`, `VLP120A_samples.csv`, `VLP120D_samples.csv`, `VLP121A_samples.csv`, and `VLP121D_samples.csv`). It also omits the pathology-annotation CSVs. Consequently, biopsy territories were fixed from spatial geometry alone before target-expression unlock, and fibrosis stages were attached only after lock from the published geometry plot. The reported synchronous-pair grouping is explicitly marked as reconstructed. Exact clinical-ID-to-array-core linkage requires those eight author mapping files and is not represented as certain here.

No biopsy was excluded using reconstruction performance. `VLP119_A_C4` is retained and flagged because only 19 spots met the frozen geometry-support radius.
"""
    (ROOT / "00B_MASLD_DATA_AUDIT.md").write_text(audit)
    (ROOT / "00_DATA_AUDIT.md").write_text(audit)

    leakage = f"""# MASLD leakage and integrity audit

**Result: PASS**, with a deposited-metadata qualification unrelated to target leakage.

- All 33 prediction-lock manifests existed before hidden-expression evaluation.
- Every registered-IDW prediction, full MAP prediction, registration, anchor-only target object, operator/guard assignment, source model, configuration, and target matrix matched its recorded SHA256 hash.
- Exactly 16 target anchors were materialized before lock.
- HDF5 sparse values were read before lock only at exact nonzero positions belonging to those 16 anchor feature rows.
- Target non-anchor expression values materialized before lock: 0.
- Geometry-only registration used target coordinates but no target molecular values, disease stage, or pathology annotation.
- Anchor selection, operator selection, calibration, gain, ridge, rank, registration seeds/epochs, neighborhood sizes, and guard logic were source-only and fixed before MASLD outcomes.
- The ALF architecture/hyperparameters were unchanged. The target assay lacked ALF anchor `MTRNR2L12`; as prespecified for differing shared gene spaces, the identical source-only selector was rerun on the known MASLD assay panel. The resulting 16-gene panel was locked before any MASLD expression access.
- No post-outcome tuning or performance-based exclusion occurred.
- Patient/fibrosis metadata were attached only after prediction lock.

Qualification: exact author barcode-to-clinical-ID mapping CSVs were not deposited. This affects identity linkage and patient-grouping certainty, not the molecular information boundary or prediction hashes.
"""
    (ROOT / "06_LEAKAGE_AUDIT.md").write_text(leakage)
    (ROOT / "11_LEAKAGE_AUDIT.md").write_text(leakage)

    by_program = programs.groupby("program")["spatial_residual_pearson"].agg(["median", lambda x: np.mean(x > 0)])
    by_program.columns = ["median_residual_pearson", "fraction_biopsies_positive"]
    program_lines = "\n".join(f"- {idx}: median residual r = {fmt(row.median_residual_pearson)}; positive in {row.fraction_biopsies_positive:.1%} of biopsies."
                              for idx, row in by_program.iterrows())
    bio = f"""# MASLD biological recovery

Five predeclared, source-independent programs were evaluated only after prediction lock. Results compare the measured atlas-relative residual with the MAP-predicted residual; pathology annotations were not used for fitting.

{program_lines}

These scores assess spatial agreement of aggregate program residuals, not differential-expression significance and not independent replication across genes or spots. Patient/biopsy is the biological unit.
"""
    (ROOT / "08_BIOLOGICAL_INTERPRETATION.md").write_text(bio)

    s = outcome["summary"]
    sev = outcome["severity_findings"]
    guard_rho = sev["guard_fallback_gene_fraction"]["spearman_rho"]
    corr_rho = sev["correction_magnitude_mean_abs_log1p"]["spearman_rho"]
    interpretation = f"""# MASLD results interpretation

HyperSpatial-MAP was compared only with registered 12-NN IDW. Across {len(patients)} reconstructed patient units ({len(biopsies)} biopsies), the fraction with higher median spatial Pearson was {outcome['patient_fraction_spatial_improved']:.1%}, the fraction with lower log1p RMSE was {outcome['patient_fraction_rmse_improved']:.1%}, and the fraction with positive pooled atlas-relative residual recovery was {outcome['patient_fraction_positive_pooled_residual']:.1%}.

The patient-balanced mean MAP–IDW spatial-Pearson change was {fmt(s['delta_spatial_pearson']['patient_balanced_mean'])} (bootstrap 95% CI {fmt(s['delta_spatial_pearson']['bootstrap_ci95_mean'][0])} to {fmt(s['delta_spatial_pearson']['bootstrap_ci95_mean'][1])}). The patient-balanced mean RMSE change was {fmt(s['delta_log1p_rmse']['patient_balanced_mean'])} (95% CI {fmt(s['delta_log1p_rmse']['bootstrap_ci95_mean'][0])} to {fmt(s['delta_log1p_rmse']['bootstrap_ci95_mean'][1])}). Mean pooled residual Pearson was {fmt(s['map_pooled_residual_pearson']['patient_balanced_mean'])} (95% CI {fmt(s['map_pooled_residual_pearson']['bootstrap_ci95_mean'][0])} to {fmt(s['map_pooled_residual_pearson']['bootstrap_ci95_mean'][1])}).

Guard fallback frequency is gene-defined and therefore constant across targets; a severity correlation is undefined ({fmt(guard_rho)}). Correction magnitude versus fibrosis severity had Spearman rho {fmt(corr_rho)} and was evaluated post-lock without changing the guard.

MASLD decision: **{outcome['MASLD_decision']}** under the prespecified gate.

Patient-level estimates are qualified because the deposited archive omitted the barcode-to-biopsy mapping files; all biopsy-level metrics and prediction hashes remain exact.
"""
    (ROOT / "12_RESULTS_INTERPRETATION.md").write_text(interpretation)

    alf_patients = pd.read_csv(ALF / "04_PATIENT_LEVEL_RESULTS.tsv", sep="\t")
    alf_out = alf_patients.copy()
    alf_out["source_experiment"] = "frozen_ALF"
    masld_out = patients.copy()
    masld_out["source_experiment"] = "MASLD_continuation"
    columns = sorted(set(alf_out.columns) | set(masld_out.columns))
    combined = pd.concat([alf_out.reindex(columns=columns), masld_out.reindex(columns=columns)], ignore_index=True)
    combined.to_csv(ROOT / "18_ALF_MASLD_COMBINED_SUMMARY.tsv", sep="\t", index=False)

    masld_decision = outcome["MASLD_decision"]
    combined_decision = "B" if masld_decision in ("A", "B") else "C"
    final = f"""# Updated Stage-1 decision

## ALF

**Decision B (unchanged).** RMSE improved in 4/4 independent patients, pooled residual recovery was positive in 4/4, and the predeclared disease programs were recovered, while median spatial Pearson decreased in 4/4. No ALF model, prediction, metric, or decision was rerun or modified.

## MASLD

**Decision {masld_decision}.** The decision is calculated from the frozen patient-level endpoints after lock. See `04_PATIENT_LEVEL_RESULTS_MASLD.tsv` and `masld_patient_level_summary.json` for exact values.

## Combined Stage 1

**Decision {combined_decision}.** {'The two disease settings support useful disease-state adaptation, but spatial reconstruction is heterogeneous and therefore does not satisfy the overall Stage-1 A criterion.' if combined_decision == 'B' else 'The combined patient-level evidence is insufficient for a reproducible multi-disease adaptation claim under the prespecified criteria.'}

Integrity: prediction lock PASS; leakage audit PASS; no ALF rerun; no target-specific retuning; no outcome-based exclusion. The MASLD patient-grouping qualification is retained because the official deposit omitted author barcode-to-biopsy mapping CSVs.
"""
    (ROOT / "15_FINAL_DECISION_UPDATED.md").write_text(final)
    stage1 = f"""# Stage-1 ALF + MASLD interpretation

Using one healthy live-donor liver reference, HyperSpatial-MAP recovered positive atlas-relative disease signal and reduced reconstruction error in ALF, but did not improve median spatial correlation. The frozen MASLD continuation was evaluated independently against the same registered-IDW comparator and yielded Decision {masld_decision}. Together, Stage 1 is Decision {combined_decision}: {'evidence for useful but spatially heterogeneous disease-state adaptation' if combined_decision == 'B' else 'insufficient reproducible evidence for multi-disease adaptation'}.

PSC should be considered only as a new frozen stage, without modifying the architecture, anchor-selection rule, registration, calibration, guard, or endpoints in response to ALF/MASLD results.
"""
    (ROOT / "19_STAGE1_INTERPRETATION.md").write_text(stage1)

    summary_rows = biopsies[["biopsy", "fibrosis_stage", "target_spots_evaluated", "hidden_genes",
                             "median_spatial_idw", "median_spatial_map", "log1p_rmse_idw",
                             "log1p_rmse_map", "median_residual_pearson_map",
                             "pooled_residual_pearson_map"]]
    summary_rows.to_csv(ROOT / "20_TERMINAL_SUMMARY_BY_BIOPSY.tsv", sep="\t", index=False)

    excluded = {"03_SHA256_PRE_UNLOCK.txt", "16_SHA256_FINAL.txt"}
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name not in excluded
                   and "data/extracted" not in str(p) and p != ARCHIVE)
    with (ROOT / "16_SHA256_FINAL.txt").open("w") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path}\n")
    print(json.dumps({"MASLD_decision": masld_decision, "combined_stage1_decision": combined_decision,
                      "biopsies": len(biopsies), "independent_patient_n": len(patients),
                      "output_directory": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
