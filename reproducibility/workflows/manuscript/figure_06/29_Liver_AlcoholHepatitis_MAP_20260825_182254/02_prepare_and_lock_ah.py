#!/usr/bin/env python3
"""Apply the exact post-MASLD frozen model to AH anchors and geometry, then lock."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_liver_ah_map")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
DATA = ROOT / "data" / "GSE278662"
WORK = ROOT / "work"
FROZEN = PROJECT / "PI_revision" / "27_Liver_multidisease_frozen_protocol"
SOURCE_RUN = PROJECT / "PI_revision" / "26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
SOURCE_MODEL = SOURCE_RUN / "work" / "healthy_source_model_masld_panel.npz"
ALF_SCRIPT = PROJECT / "PI_revision" / "26_HumanLiver_multidisease_MAP_20260825_110222" / "scripts" / "01_prepare_and_lock_alf.py"
EXPECTED_SOURCE_SHA = "28c9c83f77b3c95c285b63e562e6b8d38e44523329976b8d3fefa788d1c130d5"
SEEDS = (42, 43, 44)
SAMPLES = {
    "Sah73": ("GSM8552412", "Sah73"),
    "Sah80": ("GSM8552413", "Sah80"),
}


def import_alf():
    spec = importlib.util.spec_from_file_location("frozen_alf_prepare_for_psc", ALF_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.WORK = WORK
    return module


alf = import_alf()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def paths(gsm: str, stem: str):
    prefix = DATA / f"GSE278662_{stem}"
    return {
        "barcodes": Path(str(prefix) + "_barcodes.tsv.gz"),
        "features": Path(str(prefix) + "_features.tsv.gz"),
        "matrix": Path(str(prefix) + "_matrix.mtx.gz"),
        "positions": Path(str(prefix) + "_tissue_positions.csv.gz"),
    }


def target_metadata(gsm: str, stem: str):
    p = paths(gsm, stem)
    features = pd.read_csv(p["features"], sep="\t", header=None,
                           names=["gene_id", "gene", "feature_type"])
    with gzip.open(p["barcodes"], "rt") as handle:
        barcodes = np.asarray([line.strip() for line in handle if line.strip()])
    positions = pd.read_csv(p["positions"]).rename(columns={
        "pxl_row_in_fullres": "y", "pxl_col_in_fullres": "x"})
    positions = positions.set_index("barcode").loc[barcodes]
    tissue_rows = np.flatnonzero(positions["in_tissue"].to_numpy() == 1).astype(np.int64)
    coordinates = positions.iloc[tissue_rows][["x", "y"]].to_numpy(np.float32)
    return p, features, barcodes, tissue_rows, coordinates


def read_anchor_rows_only(matrix_path: Path, anchor_feature_rows: np.ndarray,
                          tissue_rows: np.ndarray, n_barcodes: int):
    """Scan compressed MTX but parse/materialize values only for the 16 anchor rows."""
    anchor_by_one_based = {int(row) + 1: i for i, row in enumerate(anchor_feature_rows)}
    tissue_local = np.full(n_barcodes, -1, np.int64)
    tissue_local[tissue_rows] = np.arange(len(tissue_rows))
    output = np.zeros((len(tissue_rows), len(anchor_feature_rows)), np.float32)
    anchor_values = 0
    dimensions_seen = False
    with gzip.open(matrix_path, "rt") as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            if not dimensions_seen:
                n_features, n_columns, _ = map(int, line.split())
                if n_columns != n_barcodes or np.max(anchor_feature_rows) >= n_features:
                    raise RuntimeError("Matrix metadata mismatch")
                dimensions_seen = True
                continue
            first_space = line.find(" ")
            feature = int(line[:first_space])
            local_gene = anchor_by_one_based.get(feature)
            if local_gene is None:
                continue
            _, column, value = line.split()
            local_spot = tissue_local[int(column) - 1]
            if local_spot >= 0:
                output[local_spot, local_gene] = float(value)
                anchor_values += 1
    return output, anchor_values


def model_part(model, prefix):
    return {key: model[f"{prefix}_{key}"] for key in ("x_mean", "x_scale", "y_mean", "weight")}


def indexed_idw(values, index, weight, columns):
    return np.einsum("qk,qkg->qg", weight,
                     values[index[:, :, None], columns[None, None, :]],
                     optimize=True).astype(np.float32)


def main():
    started = time.time()
    WORK.mkdir(exist_ok=True)
    protocol_path = FROZEN / "FROZEN_CONFIG.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "FROZEN_NO_POST_OUTCOME_CHANGES_PERMITTED":
        raise RuntimeError("Frozen protocol invalid")
    if sha256(SOURCE_MODEL) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Frozen source model hash mismatch")
    programs_path = ROOT / "AH_PROGRAMS.json"

    model = np.load(SOURCE_MODEL, allow_pickle=True)
    gene_ids = model["gene_ids"].astype(str)
    gene_names = model["gene_names"].astype(str)
    anchors = model["anchor_indices"].astype(np.int64)
    evaluation = model["evaluation_indices"].astype(np.int64)
    predictable = model["predictable"].astype(bool)
    selection = model["fusion_selection"].astype(np.int16)
    source_xy = model["source_coordinates"].astype(np.float32)
    source_values = model["source_values"].astype(np.float32)
    depth_weight = model["depth_weight"].astype(np.float32)
    depth_target = float(model["source_depth_target"])
    global_model = model_part(model, "global")
    local_model = model_part(model, "local")
    anchor_model = model_part(model, "anchor")
    if gene_names[anchors].tolist() != protocol["anchor_panel"]["gene_symbols"]:
        raise RuntimeError("Source model anchors differ from frozen protocol")
    pd.DataFrame({"anchor_rank": np.arange(1, 17), "gene_id": gene_ids[anchors],
                  "gene": gene_names[anchors], "selection": "FROZEN_POST_MASLD_SOURCE_ONLY"}).to_csv(
        ROOT / "01_ANCHORS.tsv", sep="\t", index=False)

    applied = {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "source_model": str(SOURCE_MODEL), "source_model_sha256": sha256(SOURCE_MODEL),
        "source_reference": protocol["healthy_reference"],
        "anchor_count": 16, "anchors": gene_names[anchors].tolist(),
        "anchor_gene_ids": gene_ids[anchors].tolist(),
        "hidden_gene_count": int(len(evaluation)),
        "registration": protocol["geometry_registration"],
        "registered_atlas_prior": protocol["registered_atlas_prior"],
        "hyperspatial_map": protocol["hyperspatial_map"],
        "operator_selection_and_guard": protocol["source_only_operator_selection_and_guard"],
        "evaluation": protocol["evaluation"],
        "programs_path": str(programs_path), "programs_sha256": sha256(programs_path),
        "target_nonanchor_expression_values_materialized_before_lock": 0,
    }
    applied_path = ROOT / "FROZEN_CONFIG_APPLIED.json"
    save_json(applied_path, applied)

    manifest_rows, registration_rows = [], []
    for sample, (gsm, stem) in SAMPLES.items():
        pair_started = time.time()
        p, features, barcodes, tissue_rows, target_raw = target_metadata(gsm, stem)
        id_to_row = {g: i for i, g in enumerate(features["gene_id"].astype(str))}
        anchor_target_rows = np.asarray([id_to_row[g] for g in gene_ids[anchors]], np.int64)
        hidden_target_rows = np.asarray([id_to_row[g] for g in gene_ids[evaluation]], np.int64)
        anchor_counts, anchor_values_read = read_anchor_rows_only(
            p["matrix"], anchor_target_rows, tissue_rows, len(barcodes))
        if anchor_counts.shape != (len(tissue_rows), 16):
            raise RuntimeError(f"{sample}: anchor object shape mismatch")

        pair = WORK / sample
        if pair.exists():
            raise RuntimeError(f"Refusing to overwrite existing lock directory {pair}")
        pair.mkdir()
        target_object = pair / "target_anchor_only_object.npz"
        np.savez(target_object, anchor_counts=anchor_counts, target_rows=tissue_rows,
                 barcodes=barcodes[tissue_rows], target_coordinates_raw=target_raw,
                 anchor_gene_ids=gene_ids[anchors], anchor_gene_names=gene_names[anchors])

        target_xy, target_transform = alf.canonicalize(target_raw)
        registrations, mapped_by_seed = {}, {}
        for seed in SEEDS:
            mapped, rigid, metadata = alf.registered_coordinates(source_xy, target_xy, 30, seed, "cpu")
            mapped_path = pair / f"seed{seed}_registered_source_xy.npy"
            rigid_path = pair / f"seed{seed}_rigid_source_xy.npy"
            np.save(mapped_path, mapped)
            np.save(rigid_path, rigid)
            metadata.update({"seed": seed, "registered_sha256": sha256(mapped_path),
                             "rigid_sha256": sha256(rigid_path)})
            registrations[str(seed)] = metadata
            mapped_by_seed[seed] = mapped
        mapped = mapped_by_seed[42]
        spacing = float(np.median(cKDTree(target_xy).query(target_xy, k=2)[0][:, 1]))
        distance = cKDTree(mapped).query(target_xy)[0]
        roi = np.flatnonzero(distance <= 2.0 * spacing).astype(np.int64)
        if len(roi) < 15:
            raise RuntimeError(f"{sample}: only {len(roi)} frozen-rule supported spots")
        roi_path = pair / "geometry_only_target_roi_rows.npy"
        np.save(roi_path, roi)
        target_xy_roi = target_xy[roi]
        anchor_counts_roi = anchor_counts[roi]
        index12, weight12 = alf.neighbor_index_weights(mapped, target_xy_roi, 12)
        predicted_target_depth = alf.predicted_depth(anchor_counts_roi, depth_weight)
        target_anchor_log = np.log1p(
            anchor_counts_roi * (depth_target / predicted_target_depth)[:, None]).astype(np.float32)
        base_anchor = indexed_idw(source_values, index12, weight12, anchors)
        anchor_residual = target_anchor_log - np.log1p(base_anchor)
        local_target = alf.residual_features(anchor_residual, target_xy_roi, True)

        shape = (len(roi), len(evaluation))
        pred_dir = pair / "predictions"
        pred_dir.mkdir()
        base_path = pred_dir / "registered_idw_k12_log1p.npy"
        map_path = pred_dir / "hyperspatial_map_log1p.npy"
        base_out = np.lib.format.open_memmap(base_path, mode="w+", dtype="float32", shape=shape)
        map_out = np.lib.format.open_memmap(map_path, mode="w+", dtype="float32", shape=shape)
        eval_selection = selection[evaluation]
        eval_predictable = predictable[evaluation]
        for begin in range(0, len(evaluation), 128):
            end = min(begin + 128, len(evaluation))
            columns = evaluation[begin:end]
            base = np.log1p(indexed_idw(source_values, index12, weight12, columns)).astype(np.float32)
            global_log = base + alf.apply_centered(global_model, anchor_residual, columns)
            local_log = base + alf.apply_centered(local_model, local_target, columns)
            anchor_log = alf.apply_centered(anchor_model, target_anchor_log, columns)
            corrected = base.copy()
            for local, candidate in enumerate(eval_selection[begin:end]):
                if candidate < 5:
                    weight = 0.25 * candidate
                    corrected[:, local] = (1.0 - weight) * global_log[:, local] + weight * anchor_log[:, local]
                elif candidate < 10:
                    weight = 0.25 * (candidate - 5)
                    corrected[:, local] = (1.0 - weight) * local_log[:, local] + weight * anchor_log[:, local]
                elif candidate != 10:
                    raise RuntimeError(f"Unexpected frozen fusion candidate {candidate}")
            corrected[:, ~eval_predictable[begin:end]] = base[:, ~eval_predictable[begin:end]]
            base_out[:, begin:end] = base
            map_out[:, begin:end] = np.maximum(corrected, 0)
        base_out.flush()
        map_out.flush()
        del base_out, map_out

        depth_path = pair / "source_model_predicted_target_depth.npy"
        np.save(depth_path, predicted_target_depth)
        operator_path = pair / "internal_operator_assignments.npz"
        np.savez(operator_path, evaluation_gene_ids=gene_ids[evaluation],
                 source_selected_candidate=eval_selection, guard_supported=eval_predictable,
                 registered_idw_fallback=~eval_predictable)
        lock = {
            "status": "PREDICTIONS_LOCKED_AND_HASHED_HIDDEN_TARGET_EXPRESSION_UNOPENED",
            "patient": sample, "sample": sample, "GSM": gsm,
            "deposited_barcodes": int(len(barcodes)), "on_tissue_spots": int(len(tissue_rows)),
            "geometry_supported_spots": int(len(roi)),
            "geometry_support_flag": "LOW_SUPPORT_LT50" if len(roi) < 50 else "PASS",
            "target_coordinate_transform": target_transform,
            "anchor_count": 16, "anchors": gene_names[anchors].tolist(),
            "anchor_gene_ids": gene_ids[anchors].tolist(),
            "hidden_gene_count": int(len(evaluation)),
            "hidden_gene_ids": gene_ids[evaluation].tolist(),
            "hidden_gene_names": gene_names[evaluation].tolist(),
            "target_hidden_feature_rows": hidden_target_rows.tolist(),
            "source_model": str(SOURCE_MODEL), "source_model_sha256": sha256(SOURCE_MODEL),
            "frozen_protocol": str(protocol_path), "frozen_protocol_sha256": sha256(protocol_path),
            "applied_configuration": str(applied_path), "applied_configuration_sha256": sha256(applied_path),
            "predeclared_programs": str(programs_path), "predeclared_programs_sha256": sha256(programs_path),
            "registration_primary": str(pair / "seed42_registered_source_xy.npy"),
            "registration_primary_sha256": sha256(pair / "seed42_registered_source_xy.npy"),
            "geometry_roi_sha256": sha256(roi_path),
            "anchor_only_object_sha256": sha256(target_object),
            "registered_idw_prediction": str(base_path), "registered_idw_sha256": sha256(base_path),
            "map_prediction": str(map_path), "map_prediction_sha256": sha256(map_path),
            "predicted_depth_sha256": sha256(depth_path),
            "operator_assignment_sha256": sha256(operator_path),
            "guard_supported_hidden_genes": int(eval_predictable.sum()),
            "registered_idw_fallback_hidden_genes": int((~eval_predictable).sum()),
            "registration": registrations,
            "registration_seed_stability": {
                "42_vs_43_median_source_displacement": float(np.median(np.linalg.norm(mapped_by_seed[42] - mapped_by_seed[43], axis=1))),
                "42_vs_44_median_source_displacement": float(np.median(np.linalg.norm(mapped_by_seed[42] - mapped_by_seed[44], axis=1))),
                "43_vs_44_median_source_displacement": float(np.median(np.linalg.norm(mapped_by_seed[43] - mapped_by_seed[44], axis=1))),
            },
            "target_matrix": str(p["matrix"]), "target_matrix_sha256": sha256(p["matrix"]),
            "target_features_sha256": sha256(p["features"]),
            "target_barcodes_sha256": sha256(p["barcodes"]),
            "target_positions_sha256": sha256(p["positions"]),
            "target_anchor_sparse_values_materialized_before_lock": int(anchor_values_read),
            "target_nonanchor_expression_values_materialized_before_lock": 0,
            "target_disease_or_compartment_annotations_used_before_lock": False,
            "runtime_seconds": time.time() - pair_started,
        }
        lock_path = pair / "prediction_lock_manifest.json"
        save_json(lock_path, lock)
        manifest_rows.append({
            "patient": sample, "sample": sample, "GSM": gsm,
            "anchor_count": 16, "hidden_genes": len(evaluation),
            "on_tissue_spots": len(tissue_rows), "geometry_supported_spots": len(roi),
            "lock_manifest": str(lock_path), "lock_manifest_sha256": sha256(lock_path),
            "registered_idw_sha256": lock["registered_idw_sha256"],
            "map_sha256": lock["map_prediction_sha256"], "lock_status": lock["status"],
        })
        for seed, metadata in registrations.items():
            registration_rows.append({
                "patient": sample, "sample": sample, "seed": int(seed),
                "raw_chamfer": metadata["raw_chamfer"],
                "registered_chamfer": metadata["registered_chamfer"],
                "reflection": metadata["reflection"], "final_loss": metadata["final_loss"],
                "geometry_supported_spots_primary": len(roi) if seed == "42" else "",
            })
        print(json.dumps({"sample": sample, "status": lock["status"],
                          "supported_spots": len(roi), "hidden_genes": len(evaluation)}), flush=True)

    pd.DataFrame(manifest_rows).to_csv(ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv", sep="\t", index=False)
    pd.DataFrame(registration_rows).to_csv(ROOT / "03_REGISTRATION_RESULTS_PRELOCK.tsv", sep="\t", index=False)
    artifacts = [protocol_path, SOURCE_MODEL, applied_path, programs_path,
                 ROOT / "01_ANCHORS.tsv", ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv"]
    artifacts += [Path(row["lock_manifest"]) for row in manifest_rows]
    with (ROOT / "03_SHA256_PRE_UNLOCK.txt").open("w") as handle:
        for artifact in artifacts:
            handle.write(f"{sha256(artifact)}  {artifact}\n")
    print(json.dumps({"status": "ALL_AH_PREDICTIONS_LOCKED", "patients": len(manifest_rows),
                      "patient_n": 2, "anchors": 16, "hidden_genes": len(evaluation),
                      "runtime_seconds": time.time() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
