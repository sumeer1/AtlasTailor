#!/usr/bin/env python3
"""Leakage-free true-3D complete-stage benchmark on whole-embryo weMERFISH.

Primary task: 50% epiboly + 6-somite -> complete 75% epiboly (E1).
The target file is first opened in backed mode for geometry only.  Its measured
expression is opened only after every prediction has been serialized.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from hyperspatial_correspondence_transport import (
    TransportMixtureConfig,
    local_source_expert,
    mix_local_experts,
    temporal_barycentric_priors,
)
from hyperspatial_modelv9 import HyperSpatialModel
from hyperspatial_molecular_hologram import RegistrationConfig, fit_geometry_field
from run_molecular_hologram_pilot import run_official_spateo_alignment
from run_mouse_temporal_transport_gate import (
    best_rigid,
    fit_anchor_gp,
    normalized_emd,
    predict_anchor_gp,
    source_svg_indices,
)


STAGES = {"A_50p": 5.3, "B_75p": 8.0, "C_6s": 12.0}
def stage_files(replicate):
    return {
        "A_50p": f"weMERFISH_measured_A_50p_{replicate}.h5ad",
        "B_75p": f"weMERFISH_measured_B_75p_{replicate}.h5ad",
        "C_6s": f"weMERFISH_measured_C_6s_{replicate}_rescaled_z.h5ad",
    }


def coordinates_only(path: Path):
    backed = ad.read_h5ad(path, backed="r")
    key = "global_sphere" if "global_sphere" in backed.obsm else "spatial"
    coordinates = np.asarray(backed.obsm[key], dtype=np.float32).copy()
    genes = np.asarray(backed.var_names.astype(str))
    ids = np.asarray(backed.obs_names.astype(str))
    backed.file.close()
    return coordinates, genes, ids, key


def measured_counts(path: Path):
    data = ad.read_h5ad(path)
    counts = np.asarray(data.layers["spliced"], dtype=np.float32)
    counts += np.asarray(data.layers["unspliced"], dtype=np.float32)
    if not np.allclose(counts, np.rint(counts)):
        raise AssertionError(f"{path.name}: spliced + unspliced is not integer count data")
    key = "global_sphere" if "global_sphere" in data.obsm else "spatial"
    return counts, np.asarray(data.obsm[key], np.float32), np.asarray(data.var_names.astype(str))


def canonicalize(coordinates):
    center = np.median(coordinates, axis=0)
    centered = coordinates - center
    scale = max(float(np.quantile(np.linalg.norm(centered, axis=1), 0.99)), 1e-6)
    return (centered / scale).astype(np.float32), {"center": center.tolist(), "scale": scale}


def normalize_counts(counts, depth_target):
    depth = counts.sum(1)
    return counts * (depth_target / np.maximum(depth, 1.0))[:, None]


def registered_coordinates(source, target, epochs, seed, device):
    rigid, reflection, candidates = best_rigid(source, target)
    rng = np.random.default_rng(seed)
    source_fit = rigid[
        np.sort(rng.choice(len(rigid), min(5000, len(rigid)), replace=False))
    ]
    target_fit = target[
        np.sort(rng.choice(len(target), min(5000, len(target)), replace=False))
    ]
    start = time.perf_counter()
    field, history = fit_geometry_field(
        source_fit,
        target_fit,
        device,
        RegistrationConfig(epochs=epochs, batch_size=768, seed=seed),
    )
    mapped = []
    with torch.no_grad():
        for begin in range(0, len(rigid), 4096):
            x = torch.as_tensor(rigid[begin : begin + 4096], dtype=torch.float32, device=device)
            mapped.append(field(x).cpu().numpy())
    mapped = np.vstack(mapped).astype(np.float32)
    raw_chamfer = 0.5 * (
        cKDTree(target).query(source)[0].mean() + cKDTree(source).query(target)[0].mean()
    )
    final_chamfer = 0.5 * (
        cKDTree(target).query(mapped)[0].mean() + cKDTree(mapped).query(target)[0].mean()
    )
    return mapped, rigid.astype(np.float32), {
        "reflection": reflection,
        "reflection_candidates": candidates,
        "raw_chamfer": float(raw_chamfer),
        "registered_chamfer": float(final_chamfer),
        "runtime_seconds": time.perf_counter() - start,
        "final_loss": float(history["loss"][-1]),
    }


def geometry_features(coordinates):
    x = np.asarray(coordinates, np.float32)
    radius = np.linalg.norm(x, axis=1, keepdims=True)
    # Spateo treats representations like molecular abundances in parts of its
    # alignment stack; keep geometry-only pseudofeatures strictly nonnegative.
    features = np.c_[x, x**2, np.sin(np.pi * x), np.cos(np.pi * x), radius]
    features -= features.min(0, keepdims=True)
    features /= np.maximum(features.max(0, keepdims=True), 1e-6)
    return (features + 1e-4).astype(np.float32)


def v9_prediction(path, source_coordinates, source_values, source_times, query_coordinates, query_time, epochs, seed, device):
    torch.manual_seed(seed)
    x = np.vstack(
        [
            np.c_[coordinates, np.full(len(coordinates), time, np.float32)]
            for coordinates, time in zip(source_coordinates, source_times)
        ]
    ).astype(np.float32)
    y = np.vstack(source_values).astype(np.float32)
    model = HyperSpatialModel(
        y.shape[1],
        n_cell_types=1,
        latent_dim=128,
        n_pos_freqs=8,
        gene_emb_dim=64,
        use_zinb=False,
        has_temporal=True,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    start = time.perf_counter()
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for coordinates, observed in loader:
            coordinates, observed = coordinates.to(device), observed.to(device)
            predicted = model(coordinates)["mu"]
            loss = torch.mean((torch.log1p(predicted) - torch.log1p(observed)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(coordinates)
        history.append(total / len(x))
    query = np.c_[query_coordinates, np.full(len(query_coordinates), query_time, np.float32)].astype(np.float32)
    output = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(query), y.shape[1]))
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(query), 4096):
            predicted = model(torch.from_numpy(query[begin : begin + 4096]).to(device))["mu"]
            output[begin : begin + len(predicted)] = torch.log1p(predicted).cpu().numpy()
    output.flush()
    return {
        "implementation": "v9_coordinate_backbone_adapted_to_fixed_495_gene_panel",
        "epochs": epochs,
        "fit_predict_seconds": time.perf_counter() - start,
        "final_training_log1p_mse": history[-1],
    }


def pearson_columns(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    return np.sum(xc * yc, axis=0) / np.sqrt(
        np.maximum(np.sum(xc**2, axis=0) * np.sum(yc**2, axis=0), 1e-16)
    )


def spearman_columns(x, y):
    values = []
    for j in range(x.shape[1]):
        values.append(
            np.corrcoef(rankdata(x[:, j]), rankdata(y[:, j]))[0, 1]
            if np.std(x[:, j]) > 0 and np.std(y[:, j]) > 0
            else np.nan
        )
    return np.asarray(values)


def moran(values, coordinates, k=12):
    neighbors = cKDTree(coordinates).query(coordinates, k=min(k + 1, len(coordinates)))[1][:, 1:]
    centered = values - values.mean()
    denom = np.sum(centered**2)
    if denom <= 1e-12:
        return np.nan
    return float(len(values) * np.sum(centered[:, None] * centered[neighbors]) / (len(values) * k * denom))


def evaluate(method, truth_log, predicted_log, coordinates, svg, thresholds, seed):
    truth = np.expm1(truth_log)
    predicted = np.clip(np.expm1(predicted_log), 0, None)
    pearson = pearson_columns(truth_log, predicted_log)
    spearman = spearman_columns(truth_log[:, svg], predicted_log[:, svg])
    selected_pearson = pearson[svg]
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [np.nanmedian(rng.choice(selected_pearson, len(selected_pearson), replace=True)) for _ in range(1000)]
    )
    localization = []
    moran_errors = []
    for local, gene in enumerate(svg):
        binary = truth[:, gene] > thresholds[gene]
        prevalence = float(binary.mean())
        if binary.sum() < 10 or binary.sum() == len(binary):
            continue
        score = predicted[:, gene]
        auprc = average_precision_score(binary, score)
        k = int(binary.sum())
        top = np.zeros(len(binary), bool)
        top[np.argpartition(score, -k)[-k:]] = True
        dice = 2 * np.sum(top & binary) / max(int(top.sum() + binary.sum()), 1)
        diameter = max(float(np.linalg.norm(coordinates.max(0) - coordinates.min(0))), 1e-8)
        true_centroid = np.average(coordinates, axis=0, weights=np.maximum(truth[:, gene], 1e-8))
        pred_centroid = np.average(coordinates, axis=0, weights=np.maximum(score, 1e-8))
        localization.append(
            [
                auprc,
                prevalence,
                auprc / max(prevalence, 1e-8),
                dice,
                np.linalg.norm(true_centroid - pred_centroid) / diameter,
                normalized_emd(truth[:, gene], score, coordinates, seed + int(gene)),
            ]
        )
        moran_errors.append(abs(moran(truth_log[:, gene], coordinates) - moran(predicted_log[:, gene], coordinates)))
    loc = np.asarray(localization)
    return {
        "method": method,
        "svg_spatial_pearson_median": float(np.nanmedian(selected_pearson)),
        "svg_spatial_pearson_bootstrap_ci95": np.nanquantile(bootstrap, [0.025, 0.975]).tolist(),
        "svg_spatial_spearman_median": float(np.nanmedian(spearman)),
        "auprc_enrichment_median": float(np.nanmedian(loc[:, 2])),
        "topk_dice_median": float(np.nanmedian(loc[:, 3])),
        "weighted_centroid_error_median": float(np.nanmedian(loc[:, 4])),
        "spatial_emd_median": float(np.nanmedian(loc[:, 5])),
        "log1p_rmse": float(np.sqrt(np.mean((truth_log - predicted_log) ** 2))),
        "moran_error_median": float(np.nanmedian(moran_errors)),
        "eligible_localization_genes": int(len(loc)),
        "per_gene_pearson": pearson.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data_v10/wemerfish_zebrafish_3stage"))
    parser.add_argument("--output", type=Path, default=Path("results_v10/wemerfish_3stage/primary_holdout"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--registration-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicate", choices=("E1", "E2"), default="E2")
    parser.add_argument("--skip-v9", action="store_true")
    parser.add_argument("--skip-spateo", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; no experiment launched")
    device = "cuda"
    files = stage_files(args.replicate)

    # Phase 1: target geometry only. No target matrix/layer/annotation is read here.
    target_path = args.data_dir / files["B_75p"]
    target_coordinates_raw, target_genes, target_ids, target_coordinate_key = coordinates_only(target_path)
    target_coordinates, target_transform = canonicalize(target_coordinates_raw)

    source_counts, source_coordinates, source_genes, source_transforms = [], [], [], []
    for stage in ("A_50p", "C_6s"):
        counts, coordinates, genes = measured_counts(args.data_dir / files[stage])
        coordinates, transform = canonicalize(coordinates)
        source_counts.append(counts)
        source_coordinates.append(coordinates)
        source_genes.append(genes)
        source_transforms.append(transform)
    if not np.array_equal(source_genes[0], source_genes[1]) or not np.array_equal(source_genes[0], target_genes):
        raise AssertionError("The fixed measured panel is not identical across stages")

    pooled_depth = np.concatenate([counts.sum(1) for counts in source_counts])
    depth_target = float(np.median(pooled_depth[pooled_depth > 0]))
    source_values = [normalize_counts(counts, depth_target).astype(np.float32) for counts in source_counts]
    source_logs = [np.log1p(values).astype(np.float32) for values in source_values]
    svg, svg_scores = source_svg_indices(source_logs, source_coordinates, count=100)
    thresholds = np.quantile(np.vstack(source_values), 0.8, axis=0)
    priors = temporal_barycentric_priors(
        [STAGES["A_50p"], STAGES["C_6s"]], STAGES["B_75p"]
    )

    alignment = {}
    registered, rigid = [], []
    for i, stage in enumerate(("A_50p", "C_6s")):
        mapped, rigid_coordinates, status = registered_coordinates(
            source_coordinates[i], target_coordinates, args.registration_epochs, args.seed + i, device
        )
        registered.append(mapped)
        rigid.append(rigid_coordinates)
        alignment[stage] = status

    configuration = TransportMixtureConfig(k=12, visibility_scale=3.0)
    methods = {}
    for name, coordinates in (
        ("rigid_temporal_idw", rigid),
        ("hyperspatial_correspondence_transport", registered),
    ):
        experts = [
            local_source_expert(coordinates[i], source_values[i], target_coordinates, configuration)
            for i in range(2)
        ]
        fallback = np.median(np.vstack(source_values), axis=0)
        prediction, weights, support = mix_local_experts(experts, priors, fallback)
        path = predictions_dir / f"{name}.npy"
        np.save(path, np.log1p(prediction).astype(np.float32))
        np.savez_compressed(
            predictions_dir / f"{name}_transport_metadata.npz",
            weights=weights,
            support=support,
            priors=priors,
        )
        methods[name] = {"prediction": str(path), "mean_support": float(support.mean())}

    combined_4d = [
        np.c_[rigid[i], np.full(len(rigid[i]), STAGES[stage], np.float32)]
        for i, stage in enumerate(("A_50p", "C_6s"))
    ]
    gp_fit = fit_anchor_gp(combined_4d, source_logs, args.seed, anchors=512, latent=32)
    gp_path = predictions_dir / "anchor_gp_kriging.npy"
    query_4d = np.c_[target_coordinates, np.full(len(target_coordinates), STAGES["B_75p"], np.float32)]
    predict_anchor_gp(gp_path, query_4d, gp_fit, device)
    methods["anchor_gp_kriging"] = {"prediction": str(gp_path)}

    # E1 development selected beta=0.30: highest SVG Pearson subject to <=5%
    # RMSE degradation relative to pure transport. E2 remains the locked test.
    baseline = np.load(
        predictions_dir / "hyperspatial_correspondence_transport.npy", mmap_mode="r"
    )
    gp_prediction = np.load(gp_path, mmap_mode="r")
    gp_scale = np.maximum(np.std(gp_prediction, axis=0), 1e-8)
    gp_z = (gp_prediction - np.mean(gp_prediction, axis=0)) / gp_scale
    desired_std = (
        priors[0] * np.std(source_logs[0], axis=0)
        + priors[1] * np.std(source_logs[1], axis=0)
    )
    hybrid = np.maximum(baseline + 0.30 * gp_z * desired_std[None, :], 0).astype(np.float32)
    hybrid_path = predictions_dir / "hyperspatial_transport_gp_residual.npy"
    np.save(hybrid_path, hybrid)
    methods["hyperspatial_transport_gp_residual"] = {
        "prediction": str(hybrid_path),
        "implementation": "local_transport_plus_source_scaled_centered_GP_spatial_residual",
        "residual_beta": 0.30,
        "selection": "E1_development_maximum_SVG_Pearson_subject_to_RMSE_within_5_percent_of_transport",
    }

    if not args.skip_v9:
        v9_path = predictions_dir / "v9_coordinate_backbone.npy"
        methods["v9_coordinate_backbone"] = {
            "prediction": str(v9_path),
            **v9_prediction(
                v9_path,
                rigid,
                source_values,
                [STAGES["A_50p"], STAGES["C_6s"]],
                target_coordinates,
                STAGES["B_75p"],
                args.epochs,
                args.seed,
                device,
            ),
        }

    if not args.skip_spateo:
        try:
            spateo_coordinates = []
            target_features = geometry_features(target_coordinates)
            names = np.asarray([f"geometry_{i}" for i in range(target_features.shape[1])])
            for i in range(2):
                mapped, _ = run_official_spateo_alignment(
                    source_coordinates[i],
                    target_coordinates,
                    geometry_features(source_coordinates[i]),
                    target_features,
                    names,
                    Path("vendor/spateo-release"),
                    "1",
                )
                spateo_coordinates.append(mapped.astype(np.float32))
            experts = [
                local_source_expert(spateo_coordinates[i], source_values[i], target_coordinates, configuration)
                for i in range(2)
            ]
            prediction, weights, support = mix_local_experts(
                experts, priors, np.median(np.vstack(source_values), axis=0)
            )
            spateo_path = predictions_dir / "spateo_adapted_geometry_transport.npy"
            np.save(spateo_path, np.log1p(prediction).astype(np.float32))
            methods["spateo_adapted_geometry_transport"] = {
                "prediction": str(spateo_path),
                "implementation": "official_Spateo_Morpho_alignment_with_geometry_only_pseudofeatures_then_shared_HyperSpatial_local_transport",
                "honesty_label": "adapted_not_official_complete_stage_prediction",
                "mean_support": float(support.mean()),
            }
        except Exception as error:
            methods["spateo_adapted_geometry_transport"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }

    prediction_manifest = {
        "target_spot_ids": target_ids.tolist(),
        "genes": target_genes.tolist(),
        "methods": methods,
    }
    (args.output / "prediction_manifest.json").write_text(json.dumps(prediction_manifest, indent=2) + "\n")
    leakage = {
        "target": f"B_75p_{args.replicate}_complete_embryo",
        "target_geometry_opened_before_prediction": True,
        "target_expression_opened_before_prediction": False,
        "target_annotations_opened_before_prediction": False,
        "count_source": "layers/spliced + layers/unspliced",
        "fixed_gene_panel": 495,
        "normalization_depth_target_source_only": depth_target,
        "svg_selection": "source_stages_only",
        "threshold_selection": "source_stages_only_80th_percentile",
        "hyperparameters": {"k": 12, "visibility_scale": 3.0, "registration_epochs": args.registration_epochs},
        "development_status": "E1_development" if args.replicate == "E1" else "E2_locked_confirmation",
        "target_coordinate_key": target_coordinate_key,
        "target_transform": target_transform,
        "source_transforms": source_transforms,
    }
    (args.output / "leakage_audit.json").write_text(json.dumps(leakage, indent=2) + "\n")

    # Evaluation phase begins only after prediction and leakage manifests exist.
    target_counts, _, evaluation_genes = measured_counts(target_path)
    if not np.array_equal(evaluation_genes, target_genes):
        raise AssertionError("target gene order changed between prediction and evaluation")
    truth_log = np.log1p(normalize_counts(target_counts, depth_target)).astype(np.float32)
    metrics = []
    per_gene = {"gene": target_genes.tolist(), "source_svg_score": svg_scores.tolist(), "selected_svg": np.isin(np.arange(len(target_genes)), svg).tolist()}
    for method, metadata in methods.items():
        if "prediction" not in metadata:
            continue
        predicted = np.load(metadata["prediction"], mmap_mode="r")
        result = evaluate(method, truth_log, predicted, target_coordinates, svg, thresholds, args.seed)
        per_gene[method] = result.pop("per_gene_pearson")
        metrics.append(result)
    (args.output / "corrected_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output / "corrected_metric_table.tsv").open("w") as handle:
        columns = [key for key in metrics[0] if key != "svg_spatial_pearson_bootstrap_ci95"]
        handle.write("\t".join(columns + ["pearson_ci_low", "pearson_ci_high"]) + "\n")
        for result in metrics:
            handle.write(
                "\t".join(str(result[key]) for key in columns)
                + "\t"
                + "\t".join(map(str, result["svg_spatial_pearson_bootstrap_ci95"]))
                + "\n"
            )
    with (args.output / "per_gene_metrics.tsv").open("w") as handle:
        columns = list(per_gene)
        handle.write("\t".join(columns) + "\n")
        for row in range(len(target_genes)):
            handle.write("\t".join(str(per_gene[column][row]) for column in columns) + "\n")
    summary = {
        "metrics": metrics,
        "alignment": alignment,
        "primary_method": "hyperspatial_correspondence_transport",
        "best_by_svg_pearson": max(metrics, key=lambda x: x["svg_spatial_pearson_median"])["method"],
        "best_by_rmse": min(metrics, key=lambda x: x["log1p_rmse"])["method"],
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "completed", "output": str(args.output), "best_svg": summary["best_by_svg_pearson"], "best_rmse": summary["best_by_rmse"]}))


if __name__ == "__main__":
    main()
