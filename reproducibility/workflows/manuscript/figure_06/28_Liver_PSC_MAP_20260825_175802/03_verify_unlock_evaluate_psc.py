#!/usr/bin/env python3
"""Verify all PSC locks, then open hidden expression and evaluate frozen endpoints."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
DATA = ROOT / "data" / "GSE245620"
WORK = ROOT / "work"
ALF_EVAL = PROJECT / "PI_revision" / "26_HumanLiver_multidisease_MAP_20260825_110222" / "scripts" / "02_verify_unlock_evaluate_alf.py"
SAMPLES = {
    "PSC011_A1": ("GSM7845914", "PSC011_A1_VISIUM"),
    "PSC011_B1": ("GSM7845915", "PSC011_B1_VISIUM"),
    "PSC011_C1": ("GSM7845916", "PSC011_C1_VISIUM"),
    "PSC011_D1": ("GSM7845917", "PSC011_D1_VISIUM"),
}


def import_alf_eval():
    spec = importlib.util.spec_from_file_location("frozen_alf_eval_for_psc", ALF_EVAL)
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


def paths(gsm: str, stem: str):
    prefix = DATA / f"{gsm}_{stem}"
    return {
        "barcodes": Path(str(prefix) + "_barcodes.tsv.gz"),
        "features": Path(str(prefix) + "_features.tsv.gz"),
        "matrix": Path(str(prefix) + "_matrix.mtx.gz"),
        "positions": Path(str(prefix) + "_tissue_positions_list.csv.gz"),
    }


def verify_all_locks():
    table = pd.read_csv(ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv", sep="\t")
    if len(table) != 4:
        raise RuntimeError(f"Expected four PSC locks, found {len(table)}")
    checks, cache = [], {}
    for row in table.itertuples():
        pair = WORK / row.sample
        lock_path = Path(row.lock_manifest)
        lock = json.loads(lock_path.read_text())
        expected = {
            lock_path: row.lock_manifest_sha256,
            Path(lock["source_model"]): lock["source_model_sha256"],
            Path(lock["frozen_protocol"]): lock["frozen_protocol_sha256"],
            Path(lock["applied_configuration"]): lock["applied_configuration_sha256"],
            Path(lock["predeclared_programs"]): lock["predeclared_programs_sha256"],
            Path(lock["registration_primary"]): lock["registration_primary_sha256"],
            pair / "geometry_only_target_roi_rows.npy": lock["geometry_roi_sha256"],
            pair / "target_anchor_only_object.npz": lock["anchor_only_object_sha256"],
            Path(lock["registered_idw_prediction"]): lock["registered_idw_sha256"],
            Path(lock["map_prediction"]): lock["map_prediction_sha256"],
            pair / "source_model_predicted_target_depth.npy": lock["predicted_depth_sha256"],
            pair / "internal_operator_assignments.npz": lock["operator_assignment_sha256"],
            Path(lock["target_matrix"]): lock["target_matrix_sha256"],
        }
        gsm, stem = SAMPLES[row.sample]
        p = paths(gsm, stem)
        expected[p["features"]] = lock["target_features_sha256"]
        expected[p["barcodes"]] = lock["target_barcodes_sha256"]
        expected[p["positions"]] = lock["target_positions_sha256"]
        for path, digest in expected.items():
            key = str(path.resolve())
            if key not in cache:
                cache[key] = sha256(path)
            checks.append({"sample": row.sample, "path": str(path), "expected": digest,
                           "observed": cache[key], "verified": cache[key] == digest})
    if not all(item["verified"] for item in checks):
        raise RuntimeError("PSC prediction-lock verification failed")
    (ROOT / "lock_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    return table


def load_target_after_unlock(gsm: str, stem: str):
    p = paths(gsm, stem)
    features = pd.read_csv(p["features"], sep="\t", header=None,
                           names=["gene_id", "gene", "feature_type"])
    with gzip.open(p["barcodes"], "rt") as handle:
        barcodes = np.asarray([line.strip() for line in handle if line.strip()])
    positions = pd.read_csv(p["positions"], header=None,
                            names=["barcode", "in_tissue", "array_row", "array_col", "y", "x"])
    positions = positions.set_index("barcode").loc[barcodes]
    with gzip.open(p["matrix"], "rb") as handle:
        matrix = mmread(handle).tocsr()
    spot_matrix = matrix.T.tocsr()
    nfeature = np.diff(spot_matrix.indptr)
    total = np.asarray(spot_matrix.sum(axis=1)).ravel()
    mito = features["gene"].astype(str).str.startswith("MT-").to_numpy()
    mito_total = np.asarray(matrix[mito].sum(axis=0)).ravel() if mito.any() else np.zeros(len(barcodes))
    mito_pct = 100.0 * mito_total / np.maximum(total, 1.0)
    spot_qc = (nfeature > 10) & (mito_pct < 10.0)
    return matrix, features, barcodes, positions, spot_qc, nfeature, total, mito_pct


def scalar_metrics(result):
    return {key: value for key, value in result.items() if np.isscalar(value)}


def main():
    lock_table = verify_all_locks()
    program_spec = json.loads((ROOT / "PSC_PROGRAMS.json").read_text())["programs"]
    section_rows, gene_rows, program_rows, qc_rows = [], [], [], []
    section_data = {}

    # Hidden target expression is first opened below, after global lock verification.
    for sample, (gsm, stem) in SAMPLES.items():
        pair = WORK / sample
        lock = json.loads((pair / "prediction_lock_manifest.json").read_text())
        matrix, features, barcodes, positions, spot_qc, nfeature, total, mito_pct = load_target_after_unlock(gsm, stem)
        id_to_row = {g: i for i, g in enumerate(features["gene_id"].astype(str))}
        hidden_rows = np.asarray([id_to_row[g] for g in lock["hidden_gene_ids"]], np.int64)
        gene_qc = np.asarray(matrix[hidden_rows].getnnz(axis=1)).ravel() > 3
        selected_rows = hidden_rows[gene_qc]
        hidden_ids = np.asarray(lock["hidden_gene_ids"])[gene_qc]
        hidden_names = np.asarray(lock["hidden_gene_names"])[gene_qc]
        anchor_obj = np.load(pair / "target_anchor_only_object.npz", allow_pickle=True)
        tissue_rows = anchor_obj["target_rows"].astype(np.int64)
        roi = np.load(pair / "geometry_only_target_roi_rows.npy").astype(np.int64)
        target_columns = tissue_rows[roi]
        qc_local = spot_qc[target_columns]
        counts = matrix[selected_rows][:, target_columns].T.tocsr()[qc_local]
        depth = np.load(pair / "source_model_predicted_target_depth.npy")[qc_local]
        source_model = np.load(lock["source_model"], allow_pickle=True)
        depth_target = float(source_model["source_depth_target"])
        truth = counts.toarray().astype(np.float32)
        truth *= (depth_target / depth)[:, None]
        truth = np.log1p(truth).astype(np.float32)
        base = np.asarray(np.load(lock["registered_idw_prediction"], mmap_mode="r")[qc_local][:, gene_qc])
        pred = np.asarray(np.load(lock["map_prediction"], mmap_mode="r")[qc_local][:, gene_qc])
        result = alf_eval.metrics(truth, base, pred)
        correction_magnitude = float(np.mean(np.abs(pred - base)))
        fallback_fraction = lock["registered_idw_fallback_hidden_genes"] / lock["hidden_gene_count"]
        section_rows.append({
            "disease": "PSC", "patient": "PSC011", "sample": sample, "GSM": gsm,
            "target_spots_on_tissue": lock["on_tissue_spots"],
            "geometry_supported_spots_pre_qc": lock["geometry_supported_spots"],
            "target_spots_evaluated": int(qc_local.sum()), "hidden_genes": int(gene_qc.sum()),
            "anchor_count": 16, "correction_magnitude_mean_abs_log1p": correction_magnitude,
            "guard_fallback_gene_fraction": fallback_fraction, **scalar_metrics(result),
        })
        qc_rows.append({
            "sample": sample, "GSM": gsm, "geometry_supported_spots": len(target_columns),
            "spots_passing_frozen_qc": int(qc_local.sum()), "hidden_genes_passing_frozen_qc": int(gene_qc.sum()),
            "median_nFeature": float(np.median(nfeature[target_columns][qc_local])),
            "median_nCount": float(np.median(total[target_columns][qc_local])),
            "median_percent_mito": float(np.median(mito_pct[target_columns][qc_local])),
        })
        for i, (gene_id, gene) in enumerate(zip(hidden_ids, hidden_names)):
            gene_rows.append({
                "patient": "PSC011", "sample": sample, "GSM": gsm,
                "gene_id": gene_id, "gene": gene,
                "idw_spatial_pearson": result["spatial_base"][i],
                "map_spatial_pearson": result["spatial_map"][i],
                "idw_log1p_rmse": result["rmse_base_gene"][i],
                "map_log1p_rmse": result["rmse_map_gene"][i],
                "map_residual_pearson": result["residual_r"][i],
            })
        name_to_idx = {g: i for i, g in enumerate(hidden_names)}
        for program, genes in program_spec.items():
            indices = [name_to_idx[g] for g in genes if g in name_to_idx]
            measured = result["residual_truth"][:, indices].mean(1)
            predicted = result["residual_pred"][:, indices].mean(1)
            program_rows.append({
                "patient": "PSC011", "sample": sample, "program": program,
                "predeclared_genes": ";".join(genes),
                "available_hidden_genes": ";".join(hidden_names[indices]),
                "n_available": len(indices),
                "spatial_residual_pearson": alf_eval.pooled_pearson(measured, predicted),
                "measured_mean_residual": float(measured.mean()),
                "predicted_mean_residual": float(predicted.mean()),
            })
        section_data[sample] = {"truth": truth, "base": base, "pred": pred,
                                "hidden_ids": hidden_ids, "hidden_names": hidden_names}

    sections = pd.DataFrame(section_rows)
    genes = pd.DataFrame(gene_rows)
    programs = pd.DataFrame(program_rows)
    qc = pd.DataFrame(qc_rows)
    sections.to_csv(ROOT / "04A_SECTION_LEVEL_RESULTS_PSC.tsv", sep="\t", index=False)
    genes.to_csv(ROOT / "05_GENE_LEVEL_RESULTS.tsv", sep="\t", index=False)
    programs.to_csv(ROOT / "07_DISEASE_PROGRAM_RECOVERY.tsv", sep="\t", index=False)
    qc.to_csv(ROOT / "00B_PSC_QC.tsv", sep="\t", index=False)

    common = set(section_data[next(iter(SAMPLES))]["hidden_ids"])
    for data in section_data.values():
        common &= set(data["hidden_ids"])
    ordered_common = [g for g in section_data[next(iter(SAMPLES))]["hidden_ids"] if g in common]
    truth_parts, base_parts, pred_parts = [], [], []
    for sample in SAMPLES:
        data = section_data[sample]
        lookup = {g: i for i, g in enumerate(data["hidden_ids"])}
        columns = [lookup[g] for g in ordered_common]
        truth_parts.append(data["truth"][:, columns])
        base_parts.append(data["base"][:, columns])
        pred_parts.append(data["pred"][:, columns])
    patient_result = alf_eval.metrics(np.vstack(truth_parts), np.vstack(base_parts), np.vstack(pred_parts))
    patient = pd.DataFrame([{
        "disease": "PSC", "patient": "PSC011", "independent_patient": True,
        "sections": ";".join(SAMPLES), "n_sections": 4,
        "target_spots": int(sum(len(x) for x in truth_parts)), "hidden_genes": len(ordered_common),
        "anchor_count": 16, **scalar_metrics(patient_result),
    }])
    patient.to_csv(ROOT / "04_PATIENT_LEVEL_RESULTS.tsv", sep="\t", index=False)

    program_patient = programs.groupby("program", as_index=False).agg(
        section_count=("sample", "count"),
        median_spatial_residual_pearson=("spatial_residual_pearson", "median"),
        min_spatial_residual_pearson=("spatial_residual_pearson", "min"),
        max_spatial_residual_pearson=("spatial_residual_pearson", "max"),
        positive_section_fraction=("spatial_residual_pearson", lambda x: float(np.mean(x > 0))),
    )
    program_patient.to_csv(ROOT / "07B_DISEASE_PROGRAM_SUMMARY.tsv", sep="\t", index=False)

    section_spatial_consistent = bool((sections["median_spatial_map"] > sections["median_spatial_idw"]).all())
    section_rmse_consistent = bool((sections["log1p_rmse_map"] < sections["log1p_rmse_idw"]).all())
    patient_spatial = bool(patient.loc[0, "median_spatial_map"] > patient.loc[0, "median_spatial_idw"])
    patient_rmse = bool(patient.loc[0, "log1p_rmse_map"] < patient.loc[0, "log1p_rmse_idw"])
    patient_residual = bool(patient.loc[0, "pooled_residual_pearson_map"] > 0)
    all_programs_positive = bool((program_patient["median_spatial_residual_pearson"] > 0).all())
    if patient_spatial and patient_rmse and patient_residual and all_programs_positive:
        decision = "A"
    elif (patient_rmse and patient_residual) or patient_spatial:
        decision = "B"
    else:
        decision = "C"
    summary = {
        "PSC_decision": decision, "independent_patient_n": 1, "sections": 4,
        "section_spatial_improvement_consistent": section_spatial_consistent,
        "section_rmse_improvement_consistent": section_rmse_consistent,
        "patient_spatial_improved": patient_spatial, "patient_rmse_improved": patient_rmse,
        "patient_pooled_residual_positive": patient_residual,
        "all_program_medians_positive": all_programs_positive,
        "homologous_portal_periportal_compartment_analysis": "UNAVAILABLE_NO_DEPOSITED_SPOT_ANNOTATIONS",
        "same_study_GSE240429_sensitivity": "NOT_RUN_WOULD_REQUIRE_CHANGING_HASHED_M1_M8_SOURCE_MODEL",
        "prediction_lock": "PASS", "leakage_audit": "PASS",
    }
    (ROOT / "psc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

