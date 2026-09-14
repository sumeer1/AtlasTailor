#!/usr/bin/env python3
"""Open Xenium hidden expression only after the global prediction lock exists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "analysis" / "hyperspatial_map_2d_dlpfc"))
from dlpfc_common import pearson_columns, sha256  # noqa: E402

METHODS = ("registered_idw_k12", "anchor_only_decoder", "hyperspatial_map")


class PooledPearson:
    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.n += x.size
        self.sx += float(x.sum()); self.sy += float(y.sum())
        self.sxx += float(np.square(x).sum()); self.syy += float(np.square(y).sum())
        self.sxy += float((x * y).sum())

    def value(self):
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor", choices=("COAD", "HCC", "OV"), required=True)
    parser.add_argument("--gene-chunk", type=int, default=96)
    args = parser.parse_args()
    tumor = args.tumor
    global_lock = ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv"
    if not global_lock.exists():
        raise RuntimeError("Global all-pair lock is absent; hidden target opening is forbidden")
    pair = ROOT / "work" / tumor
    manifest = json.loads((pair / "prediction_lock_manifest.json").read_text())
    locked = pd.read_csv(global_lock, sep="\t")
    rows = locked.loc[locked.tumor == tumor]
    if len(rows) != 9 or not np.all(rows.hidden_target_opened_before_hash == "NO"):
        raise RuntimeError("Incomplete pair lock table")
    for record in rows.itertuples():
        if sha256(Path(record.path)) != record.sha256:
            raise RuntimeError(f"Post-lock hash mismatch: {record.artifact}")

    model = np.load(pair / "source_model.npz", allow_pickle=True)
    evaluation = model["evaluation_indices"].astype(np.int64)
    genes = model["gene_names"].astype(str)[evaluation]
    prevalence_all = None
    # prevalence is source-only and reconstructed from the already frozen model
    # metadata only where needed by the gene-level output.
    roi = np.load(pair / "geometry_only_target_roi_rows.npy")
    depth = np.load(pair / "source_model_predicted_target_depth.npy", mmap_mode="r")
    depth_target = float(model["source_depth_target"])
    predictions = {
        method: np.load(Path(manifest["predictions"][method]["path"]), mmap_mode="r")
        for method in METHODS
    }
    base = predictions["registered_idw_k12"]
    target_path = ROOT / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_shared_counts.h5ad"
    target = ad.read_h5ad(target_path, backed="r")
    if not np.array_equal(target.var_names.astype(str).to_numpy(), model["gene_names"].astype(str)):
        raise RuntimeError("Target gene axis changed after lock")

    per_gene = []
    pooled = {method: PooledPearson() for method in METHODS}
    squared_error = {method: 0.0 for method in METHODS}
    total_values = 0
    for begin in range(0, len(evaluation), args.gene_chunk):
        end = min(begin + args.gene_chunk, len(evaluation))
        columns = evaluation[begin:end]
        selected = target[roi, columns].X
        if hasattr(selected, "to_memory"):
            selected = selected.to_memory()
        counts = selected.toarray().astype(np.float32)
        truth = np.log1p(counts * (depth_target / np.asarray(depth)[:, None])).astype(np.float32)
        base_block = np.asarray(base[:, begin:end])
        true_residual = truth - base_block
        total_values += truth.size
        for method in METHODS:
            pred = np.asarray(predictions[method][:, begin:end])
            pred_residual = pred - base_block
            spatial = pearson_columns(truth, pred)
            residual = pearson_columns(true_residual, pred_residual)
            rmse = np.sqrt(np.mean(np.square(truth - pred), axis=0, dtype=np.float64))
            pooled[method].add(true_residual, pred_residual)
            squared_error[method] += float(np.square(truth - pred).sum(dtype=np.float64))
            for local, gene in enumerate(genes[begin:end]):
                per_gene.append({
                    "pair_id": manifest["pair_id"], "tumor": tumor,
                    "source_platform": "Visium HD FFPE", "target_platform": "Xenium 5K",
                    "gene": gene, "method": method,
                    "spatial_pearson": spatial[local],
                    "residual_pearson": residual[local],
                    "log1p_rmse": rmse[local],
                    "source_guard_supported": bool(model["predictable"][columns[local]]),
                    "source_spatial_score": float(model["source_spatial_scores"][columns[local]]),
                    "representative_source_only_pool": bool(
                        columns[local] in set(model["source_only_representative_gene_pool"].tolist())
                    ),
                })
    target.file.close()
    gene_frame = pd.DataFrame(per_gene)
    pair_rows = []
    for method in METHODS:
        subset = gene_frame.loc[gene_frame.method == method]
        pair_rows.append({
            "pair_id": manifest["pair_id"], "tumor": tumor,
            "source_platform": "Visium HD FFPE", "target_platform": "Xenium 5K",
            "method": method, "hidden_gene_count": len(evaluation),
            "target_bin_count": len(roi),
            "median_gene_wise_spatial_pearson": subset.spatial_pearson.median(skipna=True),
            "pooled_bin_gene_residual_pearson": pooled[method].value(),
            "median_gene_wise_residual_pearson": subset.residual_pearson.median(skipna=True),
            "log1p_rmse": np.sqrt(squared_error[method] / total_values),
            "fraction_positive_gene_residual_recovery": float((subset.residual_pearson > 0).mean()),
        })
    pair_frame = pd.DataFrame(pair_rows)
    pivot = gene_frame.pivot(index="gene", columns="method")
    map_spatial = pivot["spatial_pearson"]["hyperspatial_map"]
    idw_spatial = pivot["spatial_pearson"]["registered_idw_k12"]
    valid_spatial = map_spatial.notna() & idw_spatial.notna()
    fraction_spatial = float((map_spatial[valid_spatial] > idw_spatial[valid_spatial]).mean())
    fraction_rmse = float((pivot["log1p_rmse"]["hyperspatial_map"] <
                           pivot["log1p_rmse"]["registered_idw_k12"]).mean())
    pair_frame["fraction_genes_MAP_gt_IDW_spatial_pearson"] = fraction_spatial
    pair_frame["fraction_genes_MAP_lt_IDW_log1p_RMSE"] = fraction_rmse
    pair_frame.to_csv(pair / "pair_level_results.tsv", sep="\t", index=False)
    gene_frame.to_csv(pair / "gene_level_results.tsv", sep="\t", index=False)
    (pair / "HIDDEN_TARGET_OPENED_AFTER_GLOBAL_LOCK.txt").write_text(
        "Xenium non-anchor expression was first materialized by 05_open_hidden_and_evaluate.py "
        "after verification of 02_PREDICTION_LOCK_MANIFEST.tsv.\n"
    )
    print(pair_frame.to_json(orient="records"))


if __name__ == "__main__":
    main()
