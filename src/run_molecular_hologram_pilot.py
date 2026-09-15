#!/usr/bin/env python3
"""Bounded HyperSpatial/official-Spateo feasibility pilot.

The pilot uses real E11.5 serial-section coordinates and counts, but applies a
known synthetic partial non-rigid deformation so alignment accuracy has ground
truth.  It also performs a leakage-free contiguous coordinate holdout for
continuous expression querying.  It is not a complete-stage benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree, distance
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader, TensorDataset

from hyperspatial_molecular_hologram import (
    MolecularField,
    RegistrationConfig,
    fit_geometry_field,
    jacobian_determinants,
    poisson_deviance_loss,
)


STAGE = "E11.5"
OFFICIAL_SPATEO_COMMIT = "1bd8a35e6e7b0dfc712d76b1c32987dc06652774"


def strings(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def categorical(h: h5py.File, name: str):
    return strings(h[f"obs/{name}/categories"][:]), h[f"obs/{name}/codes"][:]


def read_csr_rows(h: h5py.File, rows: np.ndarray) -> sp.csr_matrix:
    group = h["layers/counts"] if "counts" in h["layers"] else h["layers/count"]
    indptr = group["indptr"][:]
    starts, stops = indptr[rows], indptr[rows + 1]
    lengths = stops - starts
    out_indptr = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    data = np.concatenate([group["data"][int(a) : int(b)] for a, b in zip(starts, stops)])
    indices = np.concatenate([group["indices"][int(a) : int(b)] for a, b in zip(starts, stops)])
    return sp.csr_matrix((data, indices, out_indptr), shape=(len(rows), int(group.attrs["shape"][1])))


def load_stage_sample(cache: Path, n_spots: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    with h5py.File(cache, "r") as h:
        stages, stage_codes = categorical(h, "time_point")
        stage_code = int(np.flatnonzero(stages == STAGE)[0])
        stage_rows = np.flatnonzero(stage_codes == stage_code)
        slices = np.asarray(h["obs/slice"][:])
        coordinates = np.asarray(h["obsm/spatial"][:], dtype=np.float32)
        gene_key = "var/_index" if "var/_index" in h else "var/gene_short_name"
        genes = strings(h[gene_key][:])
        selected = []
        unique_slices = np.sort(np.unique(slices[stage_rows]))
        for slice_id in unique_slices:
            candidates = stage_rows[slices[stage_rows] == slice_id]
            take = min(len(candidates), max(1, n_spots // len(unique_slices)))
            selected.append(rng.choice(candidates, take, replace=False))
        rows = np.sort(np.concatenate(selected))
        counts = read_csr_rows(h, rows)
    xy = coordinates[rows]
    lo, hi = np.quantile(xy, 0.01, axis=0), np.quantile(xy, 0.99, axis=0)
    xy = (xy - lo) / np.maximum(hi - lo, 1e-6)
    z = slices[rows].astype(np.float32)
    z = (z - z.min()) / max(float(z.max() - z.min()), 1.0)
    xyz = np.c_[xy, z].astype(np.float32)
    return xyz, counts, genes, rows


def select_genes_training_only(counts: sp.csr_matrix, train: np.ndarray, n_genes: int):
    x = counts[train]
    mean = np.asarray(x.mean(0)).ravel()
    var = np.asarray(x.power(2).mean(0)).ravel() - mean**2
    prevalence = np.asarray((x > 0).mean(0)).ravel()
    score = np.where((prevalence >= 0.02) & (mean > 0), var / (mean + 1e-3), -np.inf)
    return np.sort(np.argsort(score)[-n_genes:])


def known_deformation(x: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(18.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32
    )
    y = 1.06 * ((x - 0.5) @ rotation.T) + 0.5 + np.asarray([0.08, -0.05, 0.02], np.float32)
    warp = np.c_[
        0.07 * np.sin(np.pi * y[:, 1]) * np.sin(np.pi * y[:, 2]),
        0.05 * np.sin(2 * np.pi * y[:, 0]) * np.sin(np.pi * y[:, 2]),
        0.025 * np.sin(np.pi * y[:, 0]) * np.sin(np.pi * y[:, 1]),
    ]
    return (y + warp).astype(np.float32)


def similarity_icp(source: np.ndarray, target: np.ndarray, iterations: int = 30) -> np.ndarray:
    mapped = source.copy()
    for _ in range(iterations):
        match = target[cKDTree(target).query(mapped)[1]]
        x0, y0 = mapped.mean(0), match.mean(0)
        x, y = mapped - x0, match - y0
        u, s, vt = np.linalg.svd(x.T @ y)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        scale = s.sum() / max(np.square(x).sum(), 1e-8)
        mapped = scale * (mapped - x0) @ rotation.T + y0
    return mapped


def geometry_metrics(predicted: np.ndarray, truth: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    diameter = max(float(np.linalg.norm(truth.max(0) - truth.min(0))), 1e-8)
    chamfer = 0.5 * (
        np.mean(cKDTree(target).query(predicted)[0]) + np.mean(cKDTree(predicted).query(target)[0])
    )
    return {
        "endpoint_error_mean": float(np.linalg.norm(predicted - truth, axis=1).mean() / diameter),
        "endpoint_error_median": float(np.median(np.linalg.norm(predicted - truth, axis=1)) / diameter),
        "boundary_proxy_chamfer": float(chamfer / diameter),
    }


def run_official_spateo_alignment(
    source: np.ndarray,
    target: np.ndarray,
    source_x: np.ndarray,
    target_x: np.ndarray,
    gene_names: np.ndarray,
    vendor: Path,
    device: str,
) -> Tuple[np.ndarray, Dict]:
    sys.path.insert(0, str(vendor.resolve()))
    import spateo as st

    source_adata = ad.AnnData(X=sp.csr_matrix(source_x), var={"gene": gene_names})
    target_adata = ad.AnnData(X=sp.csr_matrix(target_x), var={"gene": gene_names})
    source_adata.var_names = gene_names
    target_adata.var_names = gene_names
    source_adata.obsm["spatial"] = source
    target_adata.obsm["spatial"] = target
    start = time.perf_counter()
    aligned, _ = st.align.morpho_align(
        [target_adata, source_adata],
        rep_layer="X",
        rep_field="layer",
        spatial_key="spatial",
        mode="SN-N",
        dissimilarity="euclidean",
        max_iter=15,
        device=device,
        verbose=False,
        use_hvg=False,
        nn_init=True,
        allow_flip=True,
        nonrigid_start_iter=5,
        K=15,
        batch_size=min(512, len(source)),
        pre_compute_dist=True,
    )
    elapsed = time.perf_counter() - start
    return np.asarray(aligned[1].obsm["align_spatial"]), {
        "status": "completed",
        "implementation": "official_spateo_source",
        "commit": OFFICIAL_SPATEO_COMMIT,
        "bounded_settings": {"max_iter": 15, "K": 15, "mode": "SN-N"},
        "runtime_seconds": elapsed,
    }


def train_molecular_field(
    coords: np.ndarray,
    counts: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    epochs: int,
    seed: int,
    device: str,
):
    torch.manual_seed(seed)
    model = MolecularField(counts.shape[1], dim=3, rank=min(64, counts.shape[1]), hidden=128).to(device)
    source_depth = counts[train].sum(1)
    depth_target = max(float(np.median(source_depth[source_depth > 0])), 1.0)
    log_depth = np.log(np.maximum(source_depth / depth_target, 1e-3)).astype(np.float32)
    ds = TensorDataset(
        torch.as_tensor(coords[train], dtype=torch.float32),
        torch.zeros(int(train.sum()), dtype=torch.float32),
        torch.as_tensor(counts[train], dtype=torch.float32),
        torch.as_tensor(log_depth, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=min(512, len(ds)), shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    start = time.perf_counter()
    history = []
    for _ in range(epochs):
        total = 0.0
        for xyz, t, observed, offset in loader:
            xyz, t, observed, offset = xyz.to(device), t.to(device), observed.to(device), offset.to(device)
            rate = model(xyz, t, offset)
            loss = poisson_deviance_loss(observed, rate)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xyz)
        history.append(total / len(ds))
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    with torch.no_grad():
        prediction = model(
            torch.as_tensor(coords[query], dtype=torch.float32, device=device),
            torch.zeros(int(query.sum()), dtype=torch.float32, device=device),
            torch.zeros(int(query.sum()), dtype=torch.float32, device=device),
        ).cpu().numpy()
    predict_time = time.perf_counter() - start
    return model, prediction, history, {
        "fit_time_seconds": fit_time,
        "prediction_time_seconds": predict_time,
        "query_throughput_spots_per_second": int(query.sum()) / max(predict_time, 1e-8),
        "training_depth_target": depth_target,
    }


def run_official_spateo_gp(
    coords: np.ndarray,
    counts: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    genes: np.ndarray,
    vendor: Path,
    device: str,
):
    sys.path.insert(0, str(vendor.resolve()))
    from spateo.tdr.interpolations import gp_interpolation

    source = ad.AnnData(X=sp.csr_matrix(counts[train]), var={"gene": genes})
    source.var_names = genes
    source.obsm["spatial"] = coords[train]
    start = time.perf_counter()
    interpolated = gp_interpolation(
        source_adata=source,
        target_points=coords[query],
        keys=genes.tolist(),
        spatial_key="spatial",
        layer="X",
        training_iter=10,
        device=device,
        method="SVGP",
        batch_size=min(512, int(train.sum())),
        inducing_num=min(128, int(train.sum())),
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    return np.clip(np.asarray(interpolated.X), 0, None), {
        "status": "completed",
        "implementation": "official_spateo_gp_interpolation",
        "commit": OFFICIAL_SPATEO_COMMIT,
        "bounded_settings": {"training_iter_per_gene": 10, "method": "SVGP", "inducing_num": min(128, int(train.sum()))},
        "fit_and_prediction_time_seconds": elapsed,
    }


def expression_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    correlations = []
    for j in range(observed.shape[1]):
        if np.std(observed[:, j]) > 1e-8 and np.std(predicted[:, j]) > 1e-8:
            correlations.append(pearsonr(observed[:, j], predicted[:, j]).statistic)
    return {
        "log1p_rmse": float(np.sqrt(np.mean((np.log1p(observed) - np.log1p(np.clip(predicted, 0, None))) ** 2))),
        "per_gene_spatial_pearson_median": float(np.median(correlations)) if correlations else float("nan"),
        "per_spot_cosine_mean": float(
            np.mean(
                np.sum(observed * predicted, 1)
                / np.maximum(np.linalg.norm(observed, axis=1) * np.linalg.norm(predicted, axis=1), 1e-8)
            )
        ),
    }


def write_overlay(path: Path, truth: np.ndarray, distorted: np.ndarray, predictions: Dict[str, np.ndarray]):
    fig = plt.figure(figsize=(4 * (2 + len(predictions)), 4))
    panels = {"target": truth, "distorted": distorted, **predictions}
    for i, (name, values) in enumerate(panels.items(), 1):
        ax = fig.add_subplot(1, len(panels), i, projection="3d")
        ax.scatter(values[:, 0], values[:, 1], values[:, 2], s=2, alpha=0.5)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_v10/molecular_hologram_pilot"))
    parser.add_argument("--vendor-spateo", type=Path, default=Path("vendor/spateo-release"))
    parser.add_argument("--spots", type=int, default=2400)
    parser.add_argument("--genes", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-spateo", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions").mkdir(exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this pilot; no CPU fallback was launched.")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    coords, counts, all_genes, rows = load_stage_sample(args.cache, args.spots, args.seed)
    query = (coords[:, 0] >= 0.62) & (coords[:, 0] < 0.82)
    train = ~query
    gene_indices = select_genes_training_only(counts, train, args.genes)
    genes = all_genes[gene_indices]
    dense_counts = counts[:, gene_indices].toarray().astype(np.float32)
    target_keep = coords[:, 0] < 0.88
    target = coords[target_keep]
    distorted = known_deformation(coords)
    geometry_results = {}
    predictions = {}

    start = time.perf_counter()
    rigid = similarity_icp(distorted, target)
    geometry_results["rigid_similarity_icp"] = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - start,
        **geometry_metrics(rigid[target_keep], coords[target_keep], target),
    }
    predictions["rigid"] = rigid

    start = time.perf_counter()
    field, history = fit_geometry_field(
        rigid,
        target,
        device,
        RegistrationConfig(epochs=args.epochs, batch_size=min(768, len(coords)), seed=args.seed),
    )
    with torch.no_grad():
        hyper_aligned = field(torch.as_tensor(rigid, dtype=torch.float32, device=device)).cpu().numpy()
    determinants = jacobian_determinants(
        field, torch.as_tensor(rigid[: min(256, len(rigid))], dtype=torch.float32, device=device)
    ).detach().cpu().numpy()
    geometry_results["hyperspatial_geometry"] = {
        "status": "completed",
        "implementation": "independent_similarity_prealignment_plus_stationary_neural_velocity_field",
        "runtime_seconds": time.perf_counter() - start,
        "jacobian_fold_fraction": float((determinants <= 0).mean()),
        "jacobian_determinant_median": float(np.median(determinants)),
        **geometry_metrics(hyper_aligned[target_keep], coords[target_keep], target),
    }
    predictions["hyperspatial"] = hyper_aligned

    if not args.skip_spateo:
        try:
            spateo_aligned, status = run_official_spateo_alignment(
                distorted,
                target,
                dense_counts,
                dense_counts[target_keep],
                genes,
                args.vendor_spateo,
                "0",
            )
            geometry_results["official_spateo_morpho"] = {
                **status,
                **geometry_metrics(spateo_aligned[target_keep], coords[target_keep], target),
            }
            predictions["official_spateo"] = spateo_aligned
        except Exception as error:
            geometry_results["official_spateo_morpho"] = {
                "status": "failed",
                "implementation": "official_spateo_source",
                "commit": OFFICIAL_SPATEO_COMMIT,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }

    model, molecular_prediction, molecular_history, molecular_runtime = train_molecular_field(
        coords, dense_counts, train, query, args.epochs, args.seed, device
    )
    molecular_results = {
        "hyperspatial_count_aware_field": {
            "status": "completed",
            **molecular_runtime,
            **expression_metrics(dense_counts[query], molecular_prediction),
        },
    }
    spateo_gp_prediction = None
    if not args.skip_spateo:
        try:
            spateo_gp_prediction, status = run_official_spateo_gp(
                coords, dense_counts, train, query, genes, args.vendor_spateo, "0"
            )
            molecular_results["official_spateo_gp"] = {
                **status,
                **expression_metrics(dense_counts[query], spateo_gp_prediction),
            }
        except Exception as error:
            molecular_results["official_spateo_gp"] = {
                "status": "failed",
                "implementation": "official_spateo_gp_interpolation",
                "commit": OFFICIAL_SPATEO_COMMIT,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
    else:
        molecular_results["official_spateo_gp"] = {"status": "skipped_by_flag"}

    np.savez_compressed(
        args.output / "predictions" / "geometry_alignment_predictions.npz",
        rows=rows,
        target_coordinates=coords,
        distorted_coordinates=distorted,
        target_keep=target_keep,
        rigid_aligned=rigid,
        hyperspatial_aligned=hyper_aligned,
        **({"official_spateo_aligned": predictions["official_spateo"]} if "official_spateo" in predictions else {}),
    )
    molecular_arrays = {
        "rows": rows[query],
        "coordinates": coords[query],
        "genes": genes,
        "observed": dense_counts[query],
        "hyperspatial_prediction": molecular_prediction,
    }
    if spateo_gp_prediction is not None:
        molecular_arrays["official_spateo_gp_prediction"] = spateo_gp_prediction
    np.savez_compressed(
        args.output / "predictions" / "molecular_query_predictions.npz",
        **molecular_arrays,
    )
    torch.save({"state_dict": field.state_dict(), "config": vars(RegistrationConfig())}, args.output / "geometry_field.pt")
    torch.save({"state_dict": model.state_dict(), "genes": genes.tolist()}, args.output / "molecular_field.pt")
    write_overlay(args.output / "figures" / "alignment_overlay.png", coords, distorted, predictions)

    task_status = {
        "serial_section_3d_reconstruction": "pilot_proxy_completed_known_deformation",
        "partial_nonrigid_alignment": "completed",
        "runtime_and_memory": "completed",
        "arbitrary_3d_expression_querying": (
            "hyperspatial_and_official_spateo_gp_completed"
            if molecular_results["official_spateo_gp"]["status"] == "completed"
            else "hyperspatial_completed_spateo_gp_failed_or_skipped"
        ),
        "morphometric_vector_fields": "geometry_velocity_field_completed_differential_geometry_pending",
        "held_out_complete_molecular_stage": "not_tested_requires_three_stage_correspondent_data",
        "reaction_diffusion_dynamics": "component_implemented_unit_tested_not_fit_single_stage",
    }
    audit = {
        "experiment_type": "feasibility_pilot_not_paper_benchmark",
        "real_data": "mouse E11.5 serial-section coordinates and raw counts",
        "synthetic_component": "known partial non-rigid deformation applied to real coordinates",
        "cache": str(args.cache.resolve()),
        "cache_copied": False,
        "stage": STAGE,
        "serial_section_3d": True,
        "seed": args.seed,
        "sampled_rows": rows.tolist(),
        "genes_selected_from_training_coordinates_only": genes.tolist(),
        "target_query_expression_used_for_selection_or_training": False,
        "spateo_is_external_only": True,
        "spateo_used_inside_hyperspatial": False,
        "official_spateo_commit": OFFICIAL_SPATEO_COMMIT,
    }
    result = {
        "audit": audit,
        "task_status": task_status,
        "geometry": geometry_results,
        "molecular_query": molecular_results,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cpu_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "geometry_training_history": history,
        "molecular_training_history": molecular_history,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "command": " ".join(sys.argv),
        },
    }
    (args.output / "pilot_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "leakage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = ["method\ttask\tstatus\tendpoint_error_mean\tchamfer\tlog1p_rmse\tspatial_pearson\truntime_seconds"]
    for name, values in geometry_results.items():
        lines.append(
            f"{name}\talignment\t{values['status']}\t{values.get('endpoint_error_mean','')}\t"
            f"{values.get('boundary_proxy_chamfer','')}\t\t\t{values.get('runtime_seconds','')}"
        )
    for name, values in molecular_results.items():
        lines.append(
            f"{name}\texpression_query\t{values['status']}\t\t\t{values.get('log1p_rmse','')}\t"
            f"{values.get('per_gene_spatial_pearson_median','')}\t{values.get('fit_time_seconds','')}"
        )
    (args.output / "metric_table.tsv").write_text("\n".join(lines) + "\n")
    report = [
        "This is a bounded feasibility pilot, not evidence for a paper claim.",
        "Alignment uses a known synthetic partial non-rigid perturbation of real E11.5 serial-section coordinates.",
        "HyperSpatial never imports or teaches from Spateo; the official checkout is an external comparator only.",
        "The expression query split is contiguous and its counts were unavailable to gene selection and fitting.",
        "Complete-stage prediction and fitted reaction-diffusion dynamics require a correspondent >=3-stage dataset.",
    ]
    (args.output / "pilot_report.txt").write_text("\n".join(report) + "\n")
    print(json.dumps({"result": str(args.output / "pilot_results.json"), "task_status": task_status}, indent=2))


if __name__ == "__main__":
    main()
