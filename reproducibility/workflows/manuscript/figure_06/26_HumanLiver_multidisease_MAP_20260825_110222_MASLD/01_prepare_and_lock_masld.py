#!/usr/bin/env python3
"""Prepare and lock MASLD predictions without opening target non-anchor values.

The target HDF5 expression dataset is accessed only at the exact sparse-value
positions corresponding to 16 source-selected anchors. Feature identifiers,
barcodes, sparse indices, and spatial coordinates are treated as assay/geometry
metadata. All non-anchor expression values remain unread until the separate
post-lock evaluator verifies every hash.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_liver_masld_map")

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ALF_ROOT = PROJECT / "PI_revision" / "26_HumanLiver_multidisease_MAP_20260825_110222"
DATA = ROOT / "data" / "extracted" / "Visium"
WORK = ROOT / "work"
OFFICIAL_ARRAYS = ("VLP115_A", "VLP116_D", "VLP119_A", "VLP119_D",
                   "VLP120_A", "VLP120_D", "VLP121_A", "VLP121_D")
CORE_COUNTS = {name: 4 for name in OFFICIAL_ARRAYS}
CORE_COUNTS.update({"VLP119_A": 5, "VLP120_D": 3, "VLP121_D": 5})
SEEDS = (42, 43, 44)
REGISTRATION_EPOCHS = 30
ANCHOR_COUNT = 16
RANK = 16


def import_alf_module():
    path = ALF_ROOT / "scripts" / "01_prepare_and_lock_alf.py"
    spec = importlib.util.spec_from_file_location("frozen_alf_prepare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.WORK = WORK
    return module


alf = import_alf_module()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def h5_metadata(path: Path):
    with h5py.File(path, "r") as handle:
        g = handle["matrix"]
        return (decode(g["barcodes"][:]), decode(g["features/id"][:]),
                decode(g["features/name"][:]), tuple(int(x) for x in g["shape"][:]))


def positions_for_barcodes(array: str, barcodes: np.ndarray):
    path = DATA / array / "spatial" / "tissue_positions.csv"
    frame = pd.read_csv(path).set_index("barcode")
    missing = set(barcodes) - set(frame.index)
    if missing:
        raise RuntimeError(f"{array}: {len(missing)} filtered barcodes lack coordinates")
    return frame.loc[barcodes].reset_index()


def core_assignment(array: str, positions: pd.DataFrame):
    """Predeclared coordinate-only partition of multiplexed biopsy territories."""
    row = positions["array_row"].to_numpy(float)
    col = positions["array_col"].to_numpy(float)
    if array == "VLP115_A":
        labels = np.select([col < 22, col < 62, col < 94], [1, 2, 3], default=4)
    elif array == "VLP116_D":
        labels = np.select([col < 19, col < 52, col < 97], [1, 2, 3], default=4)
    elif array == "VLP119_A":
        labels = np.select([col < 18, col < 47, col < 72, col < 84], [1, 2, 3, 4], default=5)
    elif array == "VLP119_D":
        labels = np.select([col < 21, col < 53, col < 82], [1, 2, 3], default=4)
    elif array == "VLP120_A":
        labels = np.select([col < 22, col < 55, col < 89], [1, 2, 3], default=4)
    elif array == "VLP120_D":
        labels = np.select([col < 58, col < 89], [1, 2], default=3)
    elif array == "VLP121_A":
        labels = np.select([col < 25, col < 57, col <= 90], [1, 2, 3], default=4)
    elif array == "VLP121_D":
        labels = np.select([col < 25, col < 49, col < 68, col < 91], [1, 2, 3, 4], default=5)
    else:
        raise KeyError(array)
    expected = set(range(1, CORE_COUNTS[array] + 1))
    if set(np.unique(labels)) != expected:
        raise RuntimeError(f"{array}: coordinate partition does not yield {sorted(expected)}")
    return labels.astype(np.int8)


def read_anchor_values_exact(path: Path, anchor_feature_rows: np.ndarray):
    """Read data values only at sparse positions belonging to anchor rows."""
    with h5py.File(path, "r") as handle:
        g = handle["matrix"]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        mask = np.isin(indices, anchor_feature_rows)
        data_positions = np.flatnonzero(mask)
        # h5py point selection reads only the requested scalar values.
        values = g["data"][data_positions]
        feature_rows = indices[data_positions]
        spots = np.searchsorted(indptr[1:], data_positions, side="right")
        out = np.zeros((len(indptr) - 1, len(anchor_feature_rows)), np.float32)
        row_to_local = {int(row): i for i, row in enumerate(anchor_feature_rows)}
        local = np.fromiter((row_to_local[int(x)] for x in feature_rows), np.int64,
                            count=len(feature_rows))
        out[spots, local] = values
    return out, int(len(values))


def model_part(model, prefix):
    return {key: model[f"{prefix}_{key}"] for key in ("x_mean", "x_scale", "y_mean", "weight")}


def indexed_idw(values, index, weight, columns):
    return np.einsum("qk,qkg->qg", weight,
                     values[index[:, :, None], columns[None, None, :]],
                     optimize=True).astype(np.float32)


def train_healthy_source_model(target_gene_ids, target_gene_names):
    """Same source-only training/configuration as frozen ALF, on known MASLD panel."""
    source_raw, source_xy, donor_labels = [], [], []
    source_gene_ids = source_gene_names = None
    source_file_hashes = {}
    for donor_index, donor in enumerate([f"M{i}" for i in range(1, 9)]):
        path = alf.HEALTHY_H5 / f"{donor}_filtered_feature_bc_matrix.h5"
        counts, barcodes, gene_ids, gene_names = alf.load_10x_h5(path)
        if source_gene_ids is None:
            source_gene_ids, source_gene_names = gene_ids, gene_names
        elif not np.array_equal(source_gene_ids, gene_ids):
            raise RuntimeError("Healthy donor gene axes differ")
        coords, _ = alf.canonicalize(alf.healthy_coordinates(donor, barcodes))
        source_raw.append(counts)
        source_xy.append(coords)
        donor_labels.extend([donor_index] * len(coords))
        source_file_hashes[donor] = sha256(path)

    source_id_to_col = {g: i for i, g in enumerate(source_gene_ids)}
    target_id_to_name = dict(zip(target_gene_ids, target_gene_names))
    common_ids = np.asarray([g for g in target_gene_ids if g in source_id_to_col])
    common_names = np.asarray([target_id_to_name[g] for g in common_ids])
    source_columns = np.asarray([source_id_to_col[g] for g in common_ids], np.int64)
    counts = vstack([matrix[:, source_columns] for matrix in source_raw], format="csr")
    source_xy = np.vstack(source_xy).astype(np.float32)
    donor_labels = np.asarray(donor_labels, np.int8)
    depth = alf.normalize_sparse(counts)
    depth_target = float(np.median(depth))
    source_values = counts.toarray().astype(np.float32)
    source_values *= (depth_target / depth)[:, None]
    source_log = np.log1p(source_values).astype(np.float32)
    anchors, eligible, anchor_metadata = alf.scalable_source_only_anchors(source_log, common_names)
    if len(anchors) != ANCHOR_COUNT:
        raise RuntimeError("Source-only selector did not return exactly 16 anchors")
    anchor_set = set(anchors.tolist())
    evaluation = np.asarray([i for i in eligible if int(i) not in anchor_set], np.int64)

    pseudo_values = alf.pseudo_atlas(source_values, source_xy, donor_labels)
    pseudo_log = np.log1p(pseudo_values).astype(np.float32)
    source_residual = source_log - pseudo_log
    raw_features = source_residual[:, anchors]
    local_features = np.empty((len(source_log), ANCHOR_COUNT * 3), np.float32)
    for donor in np.unique(donor_labels):
        rows = np.flatnonzero(donor_labels == donor)
        local_features[rows] = alf.residual_features(raw_features[rows], source_xy[rows], True)

    validation = np.isin(donor_labels, [1, 6])
    train = ~validation
    best, grid = None, []
    for ridge in alf.RIDGE_GRID:
        global_model = alf.fit_centered_ridge_rank(raw_features[train], source_residual[train], ridge, RANK)
        local_model = alf.fit_centered_ridge_rank(local_features[train], source_residual[train], ridge, RANK)
        anchor_model = alf.fit_centered_ridge_rank(source_log[train][:, anchors], source_log[train], ridge, RANK)
        global_val = alf.apply_centered(global_model, raw_features[validation])
        local_val = alf.apply_centered(local_model, local_features[validation])
        anchor_val = alf.apply_centered(anchor_model, source_log[validation][:, anchors])
        for gain in alf.GAIN_GRID:
            selection, predictable, gate = alf.chunked_gate(
                source_log[validation], pseudo_log[validation], global_val,
                local_val, anchor_val, float(gain))
            record = {"ridge": float(ridge), "gain": float(gain),
                      "median_spatial_pearson": gate["median_spatial_pearson"],
                      "log1p_rmse": gate["rmse"],
                      "predictable_genes": int(predictable.sum())}
            grid.append(record)
            key = (record["median_spatial_pearson"], -record["log1p_rmse"], record["predictable_genes"])
            if best is None or key > best[0]:
                best = (key, record, selection.copy(), predictable.copy(), gate)
    chosen, selection, predictable, gate = best[1:]
    ridge = chosen["ridge"]
    global_model = alf.fit_centered_ridge_rank(raw_features, source_residual, ridge, RANK)
    local_model = alf.fit_centered_ridge_rank(local_features, source_residual, ridge, RANK)
    anchor_model = alf.fit_centered_ridge_rank(source_log[:, anchors], source_log, ridge, RANK)
    predictable[anchors] = False
    anchor_sum = np.asarray(counts[:, anchors].sum(axis=1)).ravel()
    total_sum = np.asarray(counts.sum(axis=1)).ravel()
    depth_x = np.c_[np.ones(len(anchor_sum)), np.log1p(anchor_sum)]
    depth_weight = np.linalg.solve(depth_x.T @ depth_x + np.diag([0.0, 1.0]),
                                   depth_x.T @ np.log1p(total_sum)).astype(np.float32)
    arrays = {
        "gene_ids": common_ids, "gene_names": common_names,
        "anchor_indices": anchors, "evaluation_indices": evaluation,
        "predictable": predictable, "fusion_selection": selection,
        "depth_weight": depth_weight,
        "source_depth_target": np.asarray(depth_target, np.float32),
        "source_coordinates": source_xy, "source_values": source_values,
        "source_donor_labels": donor_labels,
        "gate_selected_validation_mse": gate["selected_mse"],
        "gate_selected_validation_pearson": gate["selected_r"],
        "gate_base_validation_mse": gate["base_mse"],
        "gate_base_validation_pearson": gate["base_r"],
    }
    alf.pack_model(arrays, "global", global_model)
    alf.pack_model(arrays, "local", local_model)
    alf.pack_model(arrays, "anchor", anchor_model)
    model_path = WORK / "healthy_source_model_masld_panel.npz"
    np.savez(model_path, **arrays)
    lock = {
        "status": "MASLD_PANEL_HEALTHY_SOURCE_MODEL_LOCKED_TARGET_NONANCHOR_EXPRESSION_UNOPENED",
        "source_reference": "healthy live-donor liver atlas, Zenodo 10.5281/zenodo.17735506",
        "source_donors": [f"M{i}" for i in range(1, 9)],
        "shared_gene_count": int(len(common_ids)),
        "hidden_gene_count": int(len(evaluation)),
        "anchors": common_names[anchors].tolist(),
        "anchor_gene_ids": common_ids[anchors].tolist(),
        "anchor_metadata": anchor_metadata,
        "validation_donors": ["M2", "M7"],
        "selected_source_configuration": chosen,
        "source_selection_grid": grid,
        "source_model_sha256": sha256(model_path),
        "source_input_sha256": source_file_hashes,
        "reason_for_panel_specific_source_lock": "The MASLD Visium FFPE panel lacks ALF anchor MTRNR2L12; the unchanged prespecified source-only selector was rerun on assay-known shared genes.",
        "target_nonanchor_expression_values_read": 0,
    }
    save_json(WORK / "healthy_source_lock_masld_panel.json", lock)
    return model_path, lock


def main():
    lock_path = WORK / "healthy_source_lock_masld_panel.json"
    WORK.mkdir(parents=True, exist_ok=True)
    started = time.time()

    axes = {}
    for array in OFFICIAL_ARRAYS:
        h5 = DATA / array / "filtered_feature_bc_matrix.h5"
        barcodes, ids, names, shape = h5_metadata(h5)
        axes[array] = (barcodes, ids, names, shape)
    first_ids, first_names = axes[OFFICIAL_ARRAYS[0]][1:3]
    for array in OFFICIAL_ARRAYS[1:]:
        if not np.array_equal(first_ids, axes[array][1]):
            raise RuntimeError(f"{array}: feature axis differs from primary MASLD assay axis")

    model_path = WORK / "healthy_source_model_masld_panel.npz"
    if lock_path.exists():
        source_lock = json.loads(lock_path.read_text())
        if not model_path.exists() or sha256(model_path) != source_lock["source_model_sha256"]:
            raise RuntimeError("Existing MASLD source lock/model hash verification failed")
    else:
        model_path, source_lock = train_healthy_source_model(first_ids, first_names)
    model = np.load(model_path, allow_pickle=True)
    common_ids, common_names = model["gene_ids"], model["gene_names"]
    anchors = model["anchor_indices"]
    evaluation = model["evaluation_indices"]
    predictable = model["predictable"]
    selection = model["fusion_selection"]
    source_xy = model["source_coordinates"]
    source_values = model["source_values"]
    depth_weight = model["depth_weight"]
    depth_target = float(model["source_depth_target"])
    chosen = source_lock["selected_source_configuration"]
    global_model = model_part(model, "global")
    local_model = model_part(model, "local")
    anchor_model = model_part(model, "anchor")
    target_id_to_row = {g: i for i, g in enumerate(first_ids)}
    target_anchor_rows = np.asarray([target_id_to_row[common_ids[i]] for i in anchors], np.int64)
    target_hidden_rows = np.asarray([target_id_to_row[common_ids[i]] for i in evaluation], np.int64)
    config = {
        "scientific_model": "frozen HyperSpatial-MAP architecture used for ALF",
        "comparison": ["registered 12-NN IDW", "full HyperSpatial-MAP"],
        "anchor_count": 16, "rank": RANK, "registration_seeds": list(SEEDS),
        "registration_epochs": REGISTRATION_EPOCHS, "idw_k": 12,
        "residual_neighborhoods": [12, 32],
        "source_only_selected_configuration": chosen,
        "coordinate_only_core_partition": True,
        "clinical_metadata_used_before_lock": False,
    }
    config_path = WORK / "frozen_masld_configuration.json"
    save_json(config_path, config)

    manifest_rows = []
    geometry_rows = []
    array_hashes = {a: sha256(DATA / a / "filtered_feature_bc_matrix.h5") for a in OFFICIAL_ARRAYS}
    for array in OFFICIAL_ARRAYS:
        h5 = DATA / array / "filtered_feature_bc_matrix.h5"
        barcodes = axes[array][0]
        positions = positions_for_barcodes(array, barcodes)
        assignments = core_assignment(array, positions)
        anchor_counts_all, anchor_values_read = read_anchor_values_exact(h5, target_anchor_rows)
        if anchor_counts_all.shape != (len(barcodes), ANCHOR_COUNT):
            raise RuntimeError(f"{array}: anchor-only object has wrong shape")
        for core in range(1, CORE_COUNTS[array] + 1):
            target_name = f"{array}_C{core}"
            pair_started = time.time()
            pair = WORK / target_name
            lock_file = pair / "prediction_lock_manifest.json"
            if lock_file.exists():
                lock = json.loads(lock_file.read_text())
                expected = {
                    Path(lock["source_model"]): lock["source_model_sha256"],
                    Path(lock["registration_primary"]): lock["registration_primary_sha256"],
                    pair / "geometry_only_target_roi_rows.npy": lock["geometry_roi_sha256"],
                    pair / "target_anchor_only_object.npz": lock["anchor_only_object_sha256"],
                    Path(lock["registered_idw_prediction"]): lock["registered_idw_sha256"],
                    Path(lock["map_prediction"]): lock["map_prediction_sha256"],
                    pair / "source_model_predicted_target_depth.npy": lock["predicted_depth_sha256"],
                    pair / "internal_operator_assignments.npz": lock["operator_assignment_sha256"],
                    config_path: lock["configuration_sha256"],
                }
                if not all(path.exists() and sha256(path) == digest for path, digest in expected.items()):
                    raise RuntimeError(f"{target_name}: existing prediction lock verification failed")
                manifest_rows.append({
                    "target": target_name, "array": array, "coordinate_core": core,
                    "anchor_count": 16, "hidden_genes": len(evaluation),
                    "deposited_filtered_spots": lock["deposited_filtered_spots"],
                    "geometry_supported_spots": lock["geometry_supported_spots"],
                    "lock_manifest": str(lock_file), "lock_manifest_sha256": sha256(lock_file),
                    "registered_idw_sha256": lock["registered_idw_sha256"],
                    "map_sha256": lock["map_prediction_sha256"], "lock_status": lock["status"],
                })
                for seed, md in lock["registration"].items():
                    geometry_rows.append({"target": target_name, "seed": seed,
                                          "raw_chamfer": md["raw_chamfer"],
                                          "registered_chamfer": md["registered_chamfer"],
                                          "reflection": md["reflection"],
                                          "final_loss": md["final_loss"],
                                          "geometry_supported_spots_primary": lock["geometry_supported_spots"] if seed == "42" else ""})
                print(json.dumps({"target": target_name, "status": "EXISTING_LOCK_VERIFIED",
                                  "roi": lock["geometry_supported_spots"], "hidden": len(evaluation)}), flush=True)
                continue
            pair.mkdir(exist_ok=False)
            rows = np.flatnonzero(assignments == core).astype(np.int64)
            target_raw = positions.iloc[rows][["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(np.float32)
            target_xy, target_transform = alf.canonicalize(target_raw)
            anchor_counts = anchor_counts_all[rows]
            target_object = pair / "target_anchor_only_object.npz"
            np.savez(target_object, anchor_counts=anchor_counts, target_rows=rows,
                     barcodes=barcodes[rows], target_coordinates_raw=target_raw,
                     anchor_gene_ids=common_ids[anchors], anchor_gene_names=common_names[anchors])
            registrations, mapped_by_seed = {}, {}
            target_spacing = float(np.median(cKDTree(target_xy).query(target_xy, k=2)[0][:, 1]))
            for seed in SEEDS:
                mapped, rigid, metadata = alf.registered_coordinates(
                    source_xy, target_xy, REGISTRATION_EPOCHS, seed, "cpu")
                mapped_path = pair / f"seed{seed}_registered_source_xy.npy"
                rigid_path = pair / f"seed{seed}_rigid_source_xy.npy"
                np.save(mapped_path, mapped); np.save(rigid_path, rigid)
                metadata.update({"seed": seed, "registered_sha256": sha256(mapped_path),
                                 "rigid_sha256": sha256(rigid_path)})
                registrations[str(seed)] = metadata
                mapped_by_seed[seed] = mapped
            mapped = mapped_by_seed[42]
            distance = cKDTree(mapped).query(target_xy)[0]
            roi = np.flatnonzero(distance <= 2.0 * target_spacing).astype(np.int64)
            if len(roi) < 15:
                raise RuntimeError(f"{target_name}: geometry-only supported ROI has only {len(roi)} spots")
            roi_path = pair / "geometry_only_target_roi_rows.npy"
            np.save(roi_path, roi)
            target_xy_roi = target_xy[roi]
            anchor_counts_roi = anchor_counts[roi]
            index12, weight12 = alf.neighbor_index_weights(mapped, target_xy_roi, 12)
            target_depth = alf.predicted_depth(anchor_counts_roi, depth_weight)
            target_anchor_log = np.log1p(anchor_counts_roi * (depth_target / target_depth)[:, None]).astype(np.float32)
            base_anchor = indexed_idw(source_values, index12, weight12, anchors)
            anchor_residual = target_anchor_log - np.log1p(base_anchor)
            local_target = alf.residual_features(anchor_residual, target_xy_roi, True)

            shape = (len(roi), len(evaluation))
            pred_dir = pair / "predictions"; pred_dir.mkdir()
            base_path = pred_dir / "registered_idw_k12_log1p.npy"
            map_path = pred_dir / "hyperspatial_map_log1p.npy"
            base_out = np.lib.format.open_memmap(base_path, mode="w+", dtype="float32", shape=shape)
            map_out = np.lib.format.open_memmap(map_path, mode="w+", dtype="float32", shape=shape)
            gain = float(chosen["gain"])
            eval_selection = selection[evaluation]
            eval_predictable = predictable[evaluation]
            for begin in range(0, len(evaluation), 128):
                end = min(begin + 128, len(evaluation))
                columns = evaluation[begin:end]
                base = np.log1p(indexed_idw(source_values, index12, weight12, columns)).astype(np.float32)
                global_log = base + gain * alf.apply_centered(global_model, anchor_residual, columns)
                local_log = base + gain * alf.apply_centered(local_model, local_target, columns)
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
                        raise RuntimeError(f"Unexpected fusion candidate {candidate}")
                corrected[:, ~eval_predictable[begin:end]] = base[:, ~eval_predictable[begin:end]]
                base_out[:, begin:end] = base
                map_out[:, begin:end] = np.maximum(corrected, 0)
            base_out.flush(); map_out.flush(); del base_out, map_out
            depth_path = pair / "source_model_predicted_target_depth.npy"
            np.save(depth_path, target_depth)
            operator_path = pair / "internal_operator_assignments.npz"
            np.savez(operator_path, evaluation_gene_ids=common_ids[evaluation],
                     source_selected_candidate=eval_selection,
                     guard_supported=eval_predictable,
                     registered_idw_fallback=~eval_predictable)
            lock = {
                "status": "PREDICTIONS_LOCKED_AND_HASHED_HIDDEN_TARGET_EXPRESSION_UNOPENED",
                "target": target_name, "array": array, "coordinate_core": core,
                "deposited_filtered_spots": int(len(rows)),
                "geometry_supported_spots": int(len(roi)),
                "geometry_support_flag": "LOW_SUPPORT_LT50" if len(roi) < 50 else "PASS",
                "anchor_count": 16, "anchors": common_names[anchors].tolist(),
                "anchor_gene_ids": common_ids[anchors].tolist(),
                "hidden_gene_count": int(len(evaluation)),
                "hidden_gene_ids": common_ids[evaluation].tolist(),
                "hidden_gene_names": common_names[evaluation].tolist(),
                "target_hidden_feature_rows": target_hidden_rows.tolist(),
                "source_model": str(model_path), "source_model_sha256": sha256(model_path),
                "registration_primary": str(pair / "seed42_registered_source_xy.npy"),
                "registration_primary_sha256": sha256(pair / "seed42_registered_source_xy.npy"),
                "geometry_roi_sha256": sha256(roi_path),
                "anchor_only_object_sha256": sha256(target_object),
                "registered_idw_prediction": str(base_path), "registered_idw_sha256": sha256(base_path),
                "map_prediction": str(map_path), "map_prediction_sha256": sha256(map_path),
                "predicted_depth_sha256": sha256(depth_path),
                "operator_assignment_sha256": sha256(operator_path),
                "configuration_sha256": sha256(config_path),
                "selected_source_configuration": chosen,
                "guard_supported_hidden_genes": int(eval_predictable.sum()),
                "registered_idw_fallback_hidden_genes": int((~eval_predictable).sum()),
                "registration": registrations,
                "registration_seed_stability": {
                    "42_vs_43_median_source_displacement": float(np.median(np.linalg.norm(mapped_by_seed[42] - mapped_by_seed[43], axis=1))),
                    "42_vs_44_median_source_displacement": float(np.median(np.linalg.norm(mapped_by_seed[42] - mapped_by_seed[44], axis=1))),
                    "43_vs_44_median_source_displacement": float(np.median(np.linalg.norm(mapped_by_seed[43] - mapped_by_seed[44], axis=1))),
                },
                "target_matrix_path": str(h5), "target_matrix_sha256": array_hashes[array],
                "target_anchor_sparse_values_read_before_lock_array_total": anchor_values_read,
                "target_nonanchor_expression_values_materialized_before_lock": 0,
                "clinical_or_fibrosis_metadata_used_before_lock": False,
                "runtime_seconds": time.time() - pair_started,
            }
            lock_file = pair / "prediction_lock_manifest.json"
            save_json(lock_file, lock)
            manifest_rows.append({
                "target": target_name, "array": array, "coordinate_core": core,
                "anchor_count": 16, "hidden_genes": len(evaluation),
                "deposited_filtered_spots": len(rows), "geometry_supported_spots": len(roi),
                "lock_manifest": str(lock_file), "lock_manifest_sha256": sha256(lock_file),
                "registered_idw_sha256": lock["registered_idw_sha256"],
                "map_sha256": lock["map_prediction_sha256"],
                "lock_status": lock["status"],
            })
            for seed, md in registrations.items():
                geometry_rows.append({"target": target_name, "seed": seed,
                                      "raw_chamfer": md["raw_chamfer"],
                                      "registered_chamfer": md["registered_chamfer"],
                                      "reflection": md["reflection"],
                                      "final_loss": md["final_loss"],
                                      "geometry_supported_spots_primary": len(roi) if seed == "42" else ""})
            print(json.dumps({"target": target_name, "status": lock["status"],
                              "roi": len(roi), "hidden": len(evaluation)}), flush=True)

    pd.DataFrame(manifest_rows).to_csv(ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv", sep="\t", index=False)
    pd.DataFrame(geometry_rows).to_csv(ROOT / "06_REGISTRATION_RESULTS_PRELOCK.tsv", sep="\t", index=False)
    with (ROOT / "01_ANCHORS.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["anchor_order", "gene_id", "gene", "selection_information"])
        for i, idx in enumerate(anchors, 1):
            writer.writerow([i, common_ids[idx], common_names[idx], "healthy_source_only"])
    artifacts = [model_path, WORK / "healthy_source_lock_masld_panel.json", config_path,
                 ROOT / "01_ANCHORS.tsv", ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv"]
    artifacts += [Path(row["lock_manifest"]) for row in manifest_rows]
    with (ROOT / "03_SHA256_PRE_UNLOCK.txt").open("w") as handle:
        for path in artifacts:
            handle.write(f"{sha256(path)}  {path}\n")
    print(json.dumps({"status": "ALL_33_MASLD_BIOPSY_PREDICTIONS_LOCKED",
                      "target_count": len(manifest_rows),
                      "hidden_genes": len(evaluation),
                      "anchors": common_names[anchors].tolist(),
                      "runtime_seconds": time.time() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
