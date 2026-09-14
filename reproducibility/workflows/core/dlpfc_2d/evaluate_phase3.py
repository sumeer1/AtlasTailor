#!/usr/bin/env python3
"""Evaluation-only process; target hidden expression is opened here, post-hash."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.metrics import average_precision_score

from dlpfc_common import METHODS, SEEDS, load_10x_h5, load_section_metadata, pearson_columns, sha256


def validate_global_hashes(path: Path):
    table = pd.read_csv(path, sep="\t")
    if len(table) != 12 * 3 * 6:
        raise RuntimeError(f"Expected 216 molecular prediction hashes, found {len(table)}")
    for row in table.itertuples(index=False):
        file_path = Path(row.path)
        if sha256(file_path) != row.sha256:
            raise RuntimeError(f"Prediction hash mismatch: {file_path}")
    return table


def normalized_emd(true, predicted, coordinates, seed):
    true = np.clip(true, 0, None)
    predicted = np.clip(predicted, 0, None)
    if true.sum() <= 0 or predicted.sum() <= 0:
        return np.nan
    rng = np.random.default_rng(seed)
    n = min(192, len(coordinates))
    a = rng.choice(len(coordinates), n, replace=True, p=true / true.sum())
    b = rng.choice(len(coordinates), n, replace=True, p=predicted / predicted.sum())
    costs = distance.cdist(coordinates[a], coordinates[b])
    rows, cols = linear_sum_assignment(costs)
    diameter = max(float(np.linalg.norm(coordinates.max(0) - coordinates.min(0))), 1e-8)
    return float(costs[rows, cols].mean() / diameter)


def localization_metrics(truth_log, prediction_log, coordinates, thresholds, seed):
    truth = np.expm1(truth_log)
    predicted = np.clip(np.expm1(prediction_log), 0, None)
    rows = []
    diameter = max(float(np.linalg.norm(coordinates.max(0) - coordinates.min(0))), 1e-8)
    for gene in range(truth.shape[1]):
        binary = truth[:, gene] > thresholds[gene]
        if binary.sum() < 10 or binary.sum() == len(binary):
            continue
        score = predicted[:, gene]
        prevalence = float(binary.mean())
        auprc = average_precision_score(binary, score)
        k = int(binary.sum())
        top = np.zeros(len(binary), bool)
        top[np.argpartition(score, -k)[-k:]] = True
        dice = 2 * np.sum(top & binary) / max(int(top.sum() + binary.sum()), 1)
        true_centroid = np.average(coordinates, axis=0, weights=np.maximum(truth[:, gene], 1e-8))
        pred_centroid = np.average(coordinates, axis=0, weights=np.maximum(score, 1e-8))
        rows.append((
            auprc / max(prevalence, 1e-8),
            dice,
            float(np.linalg.norm(true_centroid - pred_centroid) / diameter),
            normalized_emd(truth[:, gene], score, coordinates, seed + gene),
        ))
    return np.asarray(rows, np.float64)


def layer_profile_recovery(truth_log, prediction_log, layers):
    valid = layers != "NA"
    labels = sorted(set(layers[valid]))
    if len(labels) < 3:
        return np.full(truth_log.shape[1], np.nan), labels
    truth_profiles = np.stack([truth_log[(layers == label) & valid].mean(0) for label in labels])
    prediction_profiles = np.stack([prediction_log[(layers == label) & valid].mean(0) for label in labels])
    return pearson_columns(truth_profiles, prediction_profiles), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--spot-metadata", type=Path, required=True)
    args = parser.parse_args()
    hash_path = args.root / "PREDICTION_HASHES.tsv"
    validate_global_hashes(hash_path)
    lock = json.loads(args.lock.read_text())
    all_component_rows = []
    all_gene_rows = []
    all_layer_rows = []

    for direction_index, direction in enumerate(lock["directions"]):
        source, target = direction.split("_to_")
        fit_dir = args.root / "03_pairs" / direction / "source_fit"
        prediction_dir = args.root / "04_predictions" / direction
        request = json.loads((fit_dir / "anchor_request.json").read_text())
        model = np.load(fit_dir / "source_model.npz", allow_pickle=False)
        manifest = json.loads((prediction_dir / "prediction_manifest.json").read_text())
        target_h5 = args.downloads / f"{target}_filtered_feature_bc_matrix.h5"
        target_counts, barcodes, gene_ids, gene_names = load_10x_h5(target_h5)
        if not np.array_equal(gene_ids, model["gene_ids"]):
            raise RuntimeError(f"{direction}: target feature axis changed")
        metadata = load_section_metadata(
            args.spot_metadata, target, barcodes, include_layers=True
        )
        coordinates = np.load(
            args.root / "03_pairs" / direction / "target_anchor_only.npz",
            allow_pickle=False,
        )["target_coordinates"].astype(np.float32)
        layers = metadata["cortical_layer"].fillna("NA").astype(str).to_numpy()
        evaluation_indices = model["evaluation_indices"].astype(np.int64)
        predictable = model["predictable"].astype(bool)[evaluation_indices]
        evaluation_svg = model["evaluation_svg"].astype(np.int64)
        eval_position = {int(gene): local for local, gene in enumerate(evaluation_indices)}
        svg_positions = np.asarray([eval_position[int(gene)] for gene in evaluation_svg], np.int64)
        thresholds = model["evaluation_svg_thresholds"].astype(np.float32)
        target_depth = np.maximum(np.asarray(target_counts.sum(axis=1)).ravel(), 1.0)
        source_depth_target = float(model["source_depth_target"])
        truth_values = target_counts[:, evaluation_indices].toarray().astype(np.float32)
        truth_values *= (source_depth_target / target_depth)[:, None]
        truth_log = np.log1p(truth_values).astype(np.float32)

        seed_predictions = {method: [] for method in METHODS}
        direction_gene_rows = []
        for seed in SEEDS:
            paths = manifest["methods"][str(seed)]["predictions"]
            base = np.load(paths["registered_idw_k12"]["path"], mmap_mode="r")
            truth_residual = truth_log[:, predictable] - base[:, predictable]
            for method_index, method in enumerate(METHODS):
                prediction = np.asarray(np.load(paths[method]["path"], mmap_mode="r"), np.float32)
                seed_predictions[method].append(prediction)
                spatial = pearson_columns(truth_log, prediction)
                residual = np.full(len(evaluation_indices), np.nan)
                if method != "registered_idw_k12" and predictable.sum() > 0:
                    residual[predictable] = pearson_columns(
                        truth_residual, prediction[:, predictable] - base[:, predictable]
                    )
                rmse_gene = np.sqrt(np.mean((truth_log - prediction) ** 2, axis=0))
                localization = localization_metrics(
                    truth_log[:, svg_positions], prediction[:, svg_positions],
                    coordinates, thresholds, seed * 10000 + method_index * 100,
                )
                all_component_rows.append({
                    "direction": direction, "source_section": source,
                    "target_section": target, "donor": request["donor"],
                    "adjacent_pair": request["adjacent_pair"], "seed": seed,
                    "method": method,
                    "eligible_nonanchor_genes": len(evaluation_indices),
                    "guard_supported_genes": int(predictable.sum()),
                    "all_gene_spatial_pearson_median": float(np.nanmedian(spatial)),
                    "source_spatial_subset_pearson_median": float(np.nanmedian(spatial[svg_positions])),
                    "residual_spatial_pearson_median": (
                        float(np.nanmedian(residual[predictable]))
                        if method != "registered_idw_k12" and predictable.any() else np.nan
                    ),
                    "log1p_rmse": float(np.sqrt(np.mean((truth_log - prediction) ** 2))),
                    "auprc_enrichment_median": float(np.nanmedian(localization[:, 0])),
                    "prevalence_matched_dice_median": float(np.nanmedian(localization[:, 1])),
                    "centroid_error_median": float(np.nanmedian(localization[:, 2])),
                    "spatial_emd_2d_median": float(np.nanmedian(localization[:, 3])),
                    "localization_gene_count": int(len(localization)),
                })
                if method == "hyperspatial_map_calibrated_fusion":
                    map_spatial = spatial
                    map_residual = residual
                    map_rmse = rmse_gene
                elif method == "registered_idw_k12":
                    idw_spatial = spatial
                    idw_rmse = rmse_gene
            for local, gene_index in enumerate(evaluation_indices):
                direction_gene_rows.append({
                    "direction": direction, "source_section": source,
                    "target_section": target, "donor": request["donor"],
                    "adjacent_pair": request["adjacent_pair"], "seed": seed,
                    "gene_id": gene_ids[gene_index], "gene_name": gene_names[gene_index],
                    "guard_supported": bool(predictable[local]),
                    "map_spatial_pearson": map_spatial[local],
                    "idw_spatial_pearson": idw_spatial[local],
                    "map_minus_idw_spatial_pearson": map_spatial[local] - idw_spatial[local],
                    "map_residual_spatial_pearson": map_residual[local],
                    "map_log1p_rmse": map_rmse[local],
                    "idw_log1p_rmse": idw_rmse[local],
                    "idw_minus_map_log1p_rmse": idw_rmse[local] - map_rmse[local],
                })
        all_gene_rows.extend(direction_gene_rows)

        for method in ("registered_idw_k12", "hyperspatial_map_calibrated_fusion"):
            mean_prediction = np.mean(np.stack(seed_predictions[method]), axis=0)
            layer_recovery, labels = layer_profile_recovery(truth_log, mean_prediction, layers)
            for local, gene_index in enumerate(evaluation_indices):
                all_layer_rows.append({
                    "direction": direction, "donor": request["donor"],
                    "adjacent_pair": request["adjacent_pair"], "method": method,
                    "gene_id": gene_ids[gene_index], "gene_name": gene_names[gene_index],
                    "layer_profile_pearson": layer_recovery[local],
                    "available_layer_labels": ";".join(labels),
                    "spots_with_layer": int(np.sum(layers != "NA")),
                })
        metrics_dir = args.root / "05_metrics" / direction
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r for r in all_component_rows if r["direction"] == direction]).to_csv(
            metrics_dir / "component_metrics.tsv", sep="\t", index=False
        )
        pd.DataFrame(direction_gene_rows).to_csv(
            metrics_dir / "gene_map_vs_idw.tsv", sep="\t", index=False
        )
        pd.DataFrame([r for r in all_layer_rows if r["direction"] == direction]).to_csv(
            metrics_dir / "layer_localization.tsv", sep="\t", index=False
        )
        print(json.dumps({"evaluated": direction, "genes": len(evaluation_indices)}))

    tables = args.root / "07_tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_component_rows).to_csv(
        tables / "all_component_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(all_gene_rows).to_csv(
        tables / "all_gene_map_vs_idw.tsv", sep="\t", index=False
    )
    pd.DataFrame(all_layer_rows).to_csv(
        tables / "all_layer_localization.tsv", sep="\t", index=False
    )
    (args.root / "05_metrics" / "TARGET_EVALUATION_COMPLETE.marker").write_text(
        "Target non-anchor expression and cortical layers were opened only in evaluate_phase3.py after PREDICTION_HASHES.tsv validation.\n"
    )


if __name__ == "__main__":
    main()
