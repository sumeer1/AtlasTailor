#!/usr/bin/env python3
"""Verify all MASLD locks, then open hidden target expression and evaluate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ALF_ROOT = PROJECT / "PI_revision" / "26_HumanLiver_multidisease_MAP_20260825_110222"
DATA = ROOT / "data" / "extracted" / "Visium"
WORK = ROOT / "work"

# Frozen before unlock. These are literature/source-independent marker sets;
# no gene was selected from target outcomes.
PROGRAMS = {
    "lipid_metabolic_reprogramming": ["ACACA", "FASN", "SCD", "DGAT2", "PNPLA3", "PLIN2", "CD36", "CPT1A", "PPARA", "APOB", "MTTP"],
    "steatosis_associated": ["PLIN2", "CIDEC", "CIDEA", "SCD", "FABP1", "CD36", "DGAT2"],
    "inflammation": ["CCL2", "CXCL8", "IL1B", "TNF", "NFKBIA", "S100A8", "S100A9", "LYZ", "CD74"],
    "extracellular_matrix_fibrosis": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "TIMP1", "MMP2", "SPP1", "CXCL12", "ACTA2", "TAGLN"],
    "hepatocyte_zonation": ["CYP2E1", "CYP3A4", "GLUL", "AXIN2", "HAL", "CPS1", "ASS1", "PCK1", "G6PC", "TAT"],
}

# Post-lock fibrosis labels transcribed from Supplementary Fig. S1E by matching
# biopsy geometry. Exact author barcode-to-patient CSVs were not deposited.
STAGE = {
    "VLP115_A": ["F1", "F1", "F1", "F3b"],
    "VLP116_D": ["F0", "F3a", "F4", "F3b"],
    "VLP119_A": ["F1", "F3b", "F2", "F1", "F1"],
    "VLP119_D": ["F1", "F1", "F1", "F4"],
    "VLP120_A": ["F0", "F0", "F3a", "F4"],
    "VLP120_D": ["F3b", "F2", "F1"],
    "VLP121_A": ["F2", "F4", "F4", "F4"],
    "VLP121_D": ["F4", "F0", "F0", "F3b", "F3b"],
}
STAGE_NUMERIC = {"F0": 0.0, "F1": 1.0, "F2": 2.0, "F3a": 3.0, "F3b": 3.5, "F4": 4.0}
STAGE_GROUP = {"F0": "early", "F1": "early", "F2": "intermediate",
               "F3a": "intermediate", "F3b": "late", "F4": "late"}


def import_alf_eval():
    path = ALF_ROOT / "scripts" / "02_verify_unlock_evaluate_alf.py"
    spec = importlib.util.spec_from_file_location("frozen_alf_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


alf_eval = import_alf_eval()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def verify_locks():
    table = pd.read_csv(ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv", sep="\t")
    if len(table) != 33:
        raise RuntimeError(f"Expected 33 prediction locks, found {len(table)}")
    checks = []
    observed_cache = {}
    for _, row in table.iterrows():
        pair = WORK / row["target"]
        manifest_path = Path(row["lock_manifest"])
        manifest = json.loads(manifest_path.read_text())
        expected = {
            manifest_path: row["lock_manifest_sha256"],
            Path(manifest["source_model"]): manifest["source_model_sha256"],
            Path(manifest["registration_primary"]): manifest["registration_primary_sha256"],
            pair / "geometry_only_target_roi_rows.npy": manifest["geometry_roi_sha256"],
            pair / "target_anchor_only_object.npz": manifest["anchor_only_object_sha256"],
            Path(manifest["registered_idw_prediction"]): manifest["registered_idw_sha256"],
            Path(manifest["map_prediction"]): manifest["map_prediction_sha256"],
            pair / "source_model_predicted_target_depth.npy": manifest["predicted_depth_sha256"],
            pair / "internal_operator_assignments.npz": manifest["operator_assignment_sha256"],
            WORK / "frozen_masld_configuration.json": manifest["configuration_sha256"],
            Path(manifest["target_matrix_path"]): manifest["target_matrix_sha256"],
        }
        for path, expected_hash in expected.items():
            key = str(path.resolve())
            if key not in observed_cache:
                observed_cache[key] = sha256(path)
            observed = observed_cache[key]
            checks.append({"target": row["target"], "path": str(path),
                           "expected": expected_hash, "observed": observed,
                           "verified": observed == expected_hash})
    if not all(x["verified"] for x in checks):
        failed = [x for x in checks if not x["verified"]]
        raise RuntimeError(f"Prediction-lock hash verification failed: {failed[:5]}")
    (ROOT / "lock_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    return table, checks


def load_array_after_unlock(array):
    path = DATA / array / "filtered_feature_bc_matrix.h5"
    with h5py.File(path, "r") as handle:
        g = handle["matrix"]
        matrix = csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                            shape=tuple(int(x) for x in g["shape"][:])).tocsr()
        barcodes = decode(g["barcodes"][:])
        gene_ids = decode(g["features/id"][:])
        gene_names = decode(g["features/name"][:])
    spot_matrix = matrix.T.tocsr()
    nfeature = np.diff(spot_matrix.indptr)
    total = np.asarray(spot_matrix.sum(axis=1)).ravel()
    mito = np.char.startswith(gene_names.astype(str), "MT-")
    mito_total = np.asarray(matrix[mito].sum(axis=0)).ravel() if mito.any() else np.zeros(len(barcodes))
    mito_pct = 100.0 * mito_total / np.maximum(total, 1.0)
    spot_qc = (nfeature > 10) & (mito_pct < 10.0)
    return matrix, barcodes, gene_ids, gene_names, spot_qc, nfeature, total, mito_pct


def result_metrics(truth, base, pred):
    return alf_eval.metrics(truth, base, pred)


def scalar_metrics(result):
    return {k: v for k, v in result.items() if np.isscalar(v)}


def patient_bootstrap(patients):
    rng = np.random.default_rng(260825)
    endpoints = {
        "delta_spatial_pearson": patients["median_spatial_map"].to_numpy() - patients["median_spatial_idw"].to_numpy(),
        "delta_log1p_rmse": patients["log1p_rmse_map"].to_numpy() - patients["log1p_rmse_idw"].to_numpy(),
        "map_pooled_residual_pearson": patients["pooled_residual_pearson_map"].to_numpy(),
    }
    summary = {"independent_patient_n": int(len(patients)), "bootstrap_unit": "patient", "replicates": 10000}
    for name, values in endpoints.items():
        draws = np.asarray([np.mean(rng.choice(values, len(values), replace=True)) for _ in range(10000)])
        summary[name] = {"patient_balanced_mean": float(np.mean(values)),
                         "patient_balanced_median": float(np.median(values)),
                         "bootstrap_ci95_mean": np.quantile(draws, [0.025, 0.975]).tolist()}
    return summary


def main():
    lock_table, checks = verify_locks()
    section_data, biopsy_rows, gene_rows, program_rows, qc_rows = {}, [], [], [], []
    for array in STAGE:
        matrix, barcodes, gene_ids, gene_names, spot_qc, nfeature, total, mito_pct = load_array_after_unlock(array)
        id_to_row = {g: i for i, g in enumerate(gene_ids)}
        array_targets = lock_table.loc[lock_table["array"] == array, "target"].tolist()
        for target in array_targets:
            pair = WORK / target
            manifest = json.loads((pair / "prediction_lock_manifest.json").read_text())
            anchor_obj = np.load(pair / "target_anchor_only_object.npz", allow_pickle=True)
            target_rows = anchor_obj["target_rows"].astype(np.int64)
            roi = np.load(pair / "geometry_only_target_roi_rows.npy").astype(np.int64)
            global_rows = target_rows[roi]
            qc_local = spot_qc[global_rows]
            hidden_rows = np.asarray([id_to_row[g] for g in manifest["hidden_gene_ids"]], np.int64)
            gene_qc = np.asarray(matrix[hidden_rows].getnnz(axis=1)).ravel() > 3
            selected_hidden_rows = hidden_rows[gene_qc]
            hidden_names = np.asarray(manifest["hidden_gene_names"])[gene_qc]
            hidden_ids = np.asarray(manifest["hidden_gene_ids"])[gene_qc]
            # First hidden-expression materialization occurs only after every hash above verified.
            counts = matrix[selected_hidden_rows][:, global_rows].T.tocsr()[qc_local]
            depth = np.load(pair / "source_model_predicted_target_depth.npy")[qc_local]
            source_model = np.load(manifest["source_model"], allow_pickle=True)
            depth_target = float(source_model["source_depth_target"])
            truth = counts.toarray().astype(np.float32)
            truth *= (depth_target / depth)[:, None]
            truth = np.log1p(truth).astype(np.float32)
            base = np.asarray(np.load(manifest["registered_idw_prediction"], mmap_mode="r")[qc_local][:, gene_qc])
            pred = np.asarray(np.load(manifest["map_prediction"], mmap_mode="r")[qc_local][:, gene_qc])
            result = result_metrics(truth, base, pred)
            core = int(manifest["coordinate_core"])
            stage = STAGE[array][core - 1]
            correction_magnitude = float(np.mean(np.abs(pred - base)))
            fallback_fraction = manifest["registered_idw_fallback_hidden_genes"] / manifest["hidden_gene_count"]
            biopsy_rows.append({
                "disease": "MASLD", "biopsy": target, "array": array, "coordinate_core": core,
                "fibrosis_stage": stage, "fibrosis_group": STAGE_GROUP[stage],
                "target_spots_deposited": manifest["deposited_filtered_spots"],
                "geometry_supported_spots_pre_qc": manifest["geometry_supported_spots"],
                "target_spots_evaluated": int(qc_local.sum()), "hidden_genes": int(gene_qc.sum()),
                "anchor_count": manifest["anchor_count"], "correction_magnitude_mean_abs_log1p": correction_magnitude,
                "guard_fallback_gene_fraction": fallback_fraction, **scalar_metrics(result),
            })
            section_data[target] = {"truth": truth, "base": base, "pred": pred,
                                    "hidden_ids": hidden_ids, "hidden_names": hidden_names,
                                    "result": result, "stage": stage}
            qc_rows.append({"array": array, "biopsy": target,
                            "spots_in_coordinate_territory": len(target_rows),
                            "geometry_supported_spots": len(global_rows),
                            "spots_passing_official_qc": int(qc_local.sum()),
                            "median_nFeature": float(np.median(nfeature[global_rows][qc_local])),
                            "median_nCount": float(np.median(total[global_rows][qc_local])),
                            "median_percent_mito": float(np.median(mito_pct[global_rows][qc_local]))})
            for i, (gene_id, gene) in enumerate(zip(hidden_ids, hidden_names)):
                gene_rows.append({"biopsy": target, "array": array, "fibrosis_stage": stage,
                                  "gene_id": gene_id, "gene": gene,
                                  "idw_spatial_pearson": result["spatial_base"][i],
                                  "map_spatial_pearson": result["spatial_map"][i],
                                  "idw_log1p_rmse": result["rmse_base_gene"][i],
                                  "map_log1p_rmse": result["rmse_map_gene"][i],
                                  "map_residual_pearson": result["residual_r"][i]})
            name_to_index = {g: i for i, g in enumerate(hidden_names)}
            for program, genes in PROGRAMS.items():
                indices = [name_to_index[g] for g in genes if g in name_to_index]
                if len(indices) < 2:
                    continue
                measured = result["residual_truth"][:, indices].mean(1)
                predicted = result["residual_pred"][:, indices].mean(1)
                program_rows.append({"biopsy": target, "array": array, "fibrosis_stage": stage,
                                     "program": program, "predeclared_genes": ";".join(genes),
                                     "available_hidden_genes": ";".join(hidden_names[indices]),
                                     "n_available": len(indices),
                                     "spatial_residual_pearson": alf_eval.pooled_pearson(measured, predicted),
                                     "measured_mean_residual": float(measured.mean()),
                                     "predicted_mean_residual": float(predicted.mean())})

    biopsies = pd.DataFrame(biopsy_rows).sort_values(["array", "coordinate_core"])
    biopsies.to_csv(ROOT / "04A_BIOPSY_LEVEL_RESULTS_MASLD.tsv", sep="\t", index=False)
    pd.DataFrame(gene_rows).to_csv(ROOT / "05_GENE_LEVEL_RESULTS_MASLD.tsv", sep="\t", index=False)
    programs = pd.DataFrame(program_rows)
    programs.to_csv(ROOT / "07_DISEASE_PROGRAM_RECOVERY_MASLD.tsv", sep="\t", index=False)
    pd.DataFrame(qc_rows).to_csv(ROOT / "00C_MASLD_QC_BY_BIOPSY.tsv", sep="\t", index=False)

    # The publication reports one synchronous two-biopsy patient. The only two
    # paired F3b cores on a shared array are retained as the reported pair; this
    # reconstruction is flagged because the deposited barcode mapping CSVs are absent.
    synchronous_pair = ["VLP121_D_C4", "VLP121_D_C5"]
    patient_groups = [[x] for x in biopsies["biopsy"] if x not in synchronous_pair] + [synchronous_pair]
    patient_rows = []
    for group in patient_groups:
        common = set(section_data[group[0]]["hidden_ids"])
        for target in group[1:]:
            common &= set(section_data[target]["hidden_ids"])
        common = sorted(common)
        truth_parts, base_parts, pred_parts = [], [], []
        for target in group:
            idx = {g: i for i, g in enumerate(section_data[target]["hidden_ids"])}
            cols = [idx[g] for g in common]
            truth_parts.append(section_data[target]["truth"][:, cols])
            base_parts.append(section_data[target]["base"][:, cols])
            pred_parts.append(section_data[target]["pred"][:, cols])
        result = result_metrics(np.vstack(truth_parts), np.vstack(base_parts), np.vstack(pred_parts))
        stage = section_data[group[0]]["stage"]
        patient_rows.append({"disease": "MASLD",
                             "patient": "reported_synchronous_patient" if len(group) == 2 else group[0],
                             "biopsies": ";".join(group), "n_biopsies": len(group),
                             "fibrosis_stage": stage, "fibrosis_group": STAGE_GROUP[stage],
                             "target_spots": sum(len(x) for x in truth_parts), "hidden_genes": len(common),
                             **scalar_metrics(result)})
    patients = pd.DataFrame(patient_rows)
    patients.to_csv(ROOT / "04_PATIENT_LEVEL_RESULTS_MASLD.tsv", sep="\t", index=False)
    summary = patient_bootstrap(patients)

    severity = biopsies["fibrosis_stage"].map(STAGE_NUMERIC).to_numpy(float)
    severity_findings = {}
    for column in ["median_spatial_map", "median_spatial_idw", "log1p_rmse_map", "log1p_rmse_idw",
                   "pooled_residual_pearson_map", "correction_magnitude_mean_abs_log1p",
                   "guard_fallback_gene_fraction"]:
        value = biopsies[column].to_numpy(float)
        rho, p = spearmanr(severity, value) if np.nanstd(value) > 0 else (np.nan, np.nan)
        severity_findings[column] = {"spearman_rho": float(rho), "p_value": float(p)}
    delta_spatial = patients["median_spatial_map"] - patients["median_spatial_idw"]
    delta_rmse = patients["log1p_rmse_map"] - patients["log1p_rmse_idw"]
    program_positive = float(np.mean(programs["spatial_residual_pearson"] > 0))
    a = bool((delta_spatial > 0).all() and (delta_rmse < 0).all()
             and (patients["pooled_residual_pearson_map"] > 0).all() and program_positive > 0.5)
    b = bool((delta_rmse < 0).mean() >= 0.5 and
             (patients["pooled_residual_pearson_map"] > 0).mean() >= 0.5)
    decision = "A" if a else ("B" if b else "C")
    outcome = {
        "status": "MASLD_UNLOCK_EVALUATION_COMPLETE",
        "prediction_lock": "PASS", "leakage_audit": "PASS",
        "biopsies_evaluated": int(len(biopsies)), "independent_patient_n": int(len(patients)),
        "patient_grouping_caveat": "Author barcode-to-patient CSVs were not deposited; the reported synchronous pair was reconstructed from Supplementary Fig. S1E geometry and is explicitly flagged.",
        "summary": summary, "severity_findings": severity_findings,
        "program_positive_fraction": program_positive,
        "patient_fraction_spatial_improved": float(np.mean(delta_spatial > 0)),
        "patient_fraction_rmse_improved": float(np.mean(delta_rmse < 0)),
        "patient_fraction_positive_pooled_residual": float(np.mean(patients["pooled_residual_pearson_map"] > 0)),
        "MASLD_decision": decision,
    }
    (ROOT / "masld_patient_level_summary.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2), flush=True)


if __name__ == "__main__":
    main()
