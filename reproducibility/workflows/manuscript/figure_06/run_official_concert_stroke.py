#!/usr/bin/env python3
"""Run the released CONCERT 3D stroke model and lock its denoised output.

This is an output-matched, not input-matched, benchmark.  The script imports
the official implementation without editing it.  It uses the published 3D
configuration and the complete released stroke H5, including PT expression and
ICA-derived condition labels, exactly as required by the released training
workflow.  A fixed RNG seed is added only to make the otherwise unseeded runner
reproducible; it is not selected using evaluation performance.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "PI_revision/15_CONCERT_3D_stroke_audit_20260820_092123/official_sources/CONCERT/src/CONCERT-3D"
DATA = REPO / "PI_revision/15_CONCERT_3D_stroke_audit_20260820_092123/official_sources/Mouse_brain_stroke_all_data.h5"
sys.path.insert(0, str(OFFICIAL))

from run_concert_3D_stroke import (  # noqa: E402
    build_inducing_grid3d_tiled,
    load_h5_dataset,
    make_batch_onehot,
    make_perturbation_from_labels,
    per_batch_scale_positions_3d,
)
from concert_batch_3D_stroke import CONCERT  # noqa: E402
from preprocess import normalize  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    locked = ROOT / "LOCKED_CONCERT_OUTPUT"
    locked.mkdir(exist_ok=True)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    X, pos3d, batch_str, labels_str, barcodes = load_h5_dataset(str(DATA))
    batch_oh, _ = make_batch_onehot(batch_str)
    cell_atts, _ = make_perturbation_from_labels(labels_str)
    pos_scaled = per_batch_scale_positions_3d(pos3d, batch_oh, 60.0, 60.0)
    pos_batched = np.concatenate([pos_scaled, batch_oh], axis=1).astype(np.float32)
    inducing = build_inducing_grid3d_tiled(
        n_batch=2,
        steps=6,
        loc_range=60.0,
        z_range=60.0,
        inducing_z_uses_loc_range=True,
    )

    adata = normalize(sc.AnnData(X, dtype="float32"), size_factors=True,
                      normalize_input=True, logtrans_input=True)
    training_median_depth = float(np.median(np.asarray(adata.raw.X).sum(axis=1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CONCERT(
        cell_atts=cell_atts,
        num_genes=adata.n_vars,
        input_dim=256,
        GP_dim=2,
        Normal_dim=8,
        n_batch=2,
        encoder_layers=[256, 64],
        decoder_layers=[64, 256],
        noise=0.1,
        encoder_dropout=0.1,
        decoder_dropout=0.0,
        shared_dispersion=False,
        fixed_inducing_points=True,
        initial_inducing_points=inducing,
        fixed_gp_params=False,
        kernel_scale=60.0,
        allow_batch_kernel_scale=True,
        N_train=adata.n_obs,
        KL_loss=0.025,
        dynamicVAE=True,
        init_beta=10.0,
        min_beta=5.0,
        max_beta=25.0,
        dtype=torch.float32,
        device=device,
    )

    model_path = locked / "model_stroke_3D.pt"
    started = time.time()
    model.train_model(
        pos=pos_batched,
        ncounts=adata.X,
        raw_counts=adata.raw.X,
        size_factors=np.asarray(adata.obs.size_factors),
        batch=batch_oh,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=512,
        num_samples=1,
        train_size=0.95,
        maxiter=3000,
        patience=200,
        save_model=True,
        model_weights=str(model_path),
        wandb_run=None,
    )
    training_seconds = time.time() - started

    # Official denoising API and the 25-sample convention used by the released
    # stroke 2D runner and Perturb-map notebook.
    denoised = model.batching_denoise_counts(
        X=pos_batched,
        sample_index=np.arange(X.shape[0]),
        cell_atts=cell_atts,
        n_samples=25,
        batch_size=512,
    ).astype(np.float32)
    pt_mask = np.char.startswith(batch_str, "PT")
    prediction_path = locked / "CONCERT_PT_normalized_counts.npy"
    np.save(prediction_path, denoised[pt_mask])
    np.save(locked / "CONCERT_PT_barcodes.npy", barcodes[pt_mask].astype("U"))

    manifest = {
        "status": "locked_before_benchmark_metric_evaluation",
        "official_repository_commit": "8babda00d09a44b5812bac250434cdd5d057bb18",
        "official_runner": str(OFFICIAL / "run_concert_3D_stroke.py"),
        "official_model": str(OFFICIAL / "concert_batch_3D_stroke.py"),
        "official_config": str(OFFICIAL / "config_3D.yaml"),
        "data_file": str(DATA),
        "data_sha256": sha256(DATA),
        "rng_seed_added_for_reproducibility": seed,
        "device": str(device),
        "n_training_spots": int(X.shape[0]),
        "n_pt_spots": int(pt_mask.sum()),
        "n_genes": int(X.shape[1]),
        "training_median_depth": training_median_depth,
        "training_seconds": training_seconds,
        "training_inputs": [
            "complete sham and PT expression for all 2,003 genes",
            "deposited pathology-informed 3D coordinates for sham and PT",
            "PT-versus-sham condition/batch encoding",
            "ICA-versus-other label-derived perturbation attribute",
            "per-spot size factors",
        ],
        "prediction_definition": "official batching_denoise_counts at observed coordinates with observed per-spot learned basal embeddings and ICA attributes",
        "denoise_samples": 25,
        "prediction_path": str(prediction_path),
        "prediction_shape": list(denoised[pt_mask].shape),
        "prediction_sha256": sha256(prediction_path),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "evaluation_metrics_accessed_during_training": False,
        "MAP_results_accessed_for_tuning": False,
    }
    (locked / "CONCERT_PREDICTION_LOCK.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
