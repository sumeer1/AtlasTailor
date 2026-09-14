#!/usr/bin/env python3
"""Build sealed Xenium 8-um count matrices using the published SPATCH rules.

This preprocessing step is deliberately outcome-agnostic.  It aggregates every
quality-filtered Xenium gene call and applies the DAPI tissue mask before any
HyperSpatial-MAP source model or target panel is fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tmp/spatch_deps")

import anndata as ad
import cv2
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
import tifffile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "official_data" / "extracted"
OUT = ROOT / "SEALED_TARGET_8UM"
TMP = ROOT / "work" / "xenium_8um"
TUMORS = ("COAD", "HCC", "OV")
DAPI_UM_PER_PIXEL = 0.2125
BIN_UM = 8.0
KERNEL_WIDTH = 500


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dapi_mask_and_bins(path: Path, output: Path) -> dict:
    """Reproduce SPATCH dapi_mask and sample its mask on the 8-um grid."""
    started = time.perf_counter()
    image = tifffile.imread(path)
    maximum = float(image.max())
    if maximum <= 0:
        raise RuntimeError(f"No positive DAPI signal in {path}")

    image8 = np.empty(image.shape, np.uint8)
    scale = np.float32(255.0 / maximum)
    for first in range(0, image.shape[0], 512):
        block = image[first:first + 512].astype(np.float32)
        image8[first:first + len(block)] = np.floor(block * scale).astype(np.uint8)
    del image

    otsu_value, threshold = cv2.threshold(
        image8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    del image8
    kernel = np.ones((KERNEL_WIDTH, KERNEL_WIDTH), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=2)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(f"No DAPI contour in {path}")
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(threshold)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    del threshold, kernel, contours

    height, width = mask.shape
    max_bx = int(np.floor((width - 1) * DAPI_UM_PER_PIXEL / BIN_UM))
    max_by = int(np.floor((height - 1) * DAPI_UM_PER_PIXEL / BIN_UM))
    bx = np.arange(max_bx + 1, dtype=np.int32)
    by = np.arange(max_by + 1, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(bx, by, indexing="xy")
    pixel_x = np.floor(grid_x.ravel() * BIN_UM / DAPI_UM_PER_PIXEL).astype(np.int32)
    pixel_y = np.floor(grid_y.ravel() * BIN_UM / DAPI_UM_PER_PIXEL).astype(np.int32)
    keep = mask[pixel_y, pixel_x] == 255
    tissue = pd.DataFrame({
        "bin_x": grid_x.ravel()[keep],
        "bin_y": grid_y.ravel()[keep],
    })
    pq.write_table(pa.Table.from_pandas(tissue, preserve_index=False), output,
                   compression="zstd")
    return {
        "dapi_shape_pixels": [int(height), int(width)],
        "dapi_um_per_pixel": DAPI_UM_PER_PIXEL,
        "kernel_width_pixels": KERNEL_WIDTH,
        "otsu_threshold_uint8": float(otsu_value),
        "largest_contour_area_pixels": float(cv2.contourArea(contour)),
        "candidate_tissue_grid_bins": int(len(tissue)),
        "runtime_seconds": time.perf_counter() - started,
    }


def shared_gene_table(tumor: str, path: Path) -> np.ndarray:
    source = ad.read_h5ad(DATA / f"VisiumHD_FFPE_{tumor}" / "adata.h5ad", backed="r")
    target = ad.read_h5ad(DATA / f"Xenium_{tumor}" / "adata.h5ad", backed="r")
    source_genes = source.var_names.astype(str).to_numpy()
    target_genes = target.var_names.astype(str).to_numpy()
    source.file.close()
    target.file.close()
    shared = np.asarray([g for g in target_genes if g in set(source_genes)], dtype=str)
    if len(shared) != len(np.unique(shared)):
        raise RuntimeError("Shared gene symbols are not unique")
    pd.DataFrame({"feature_name": shared, "gene_index": np.arange(len(shared))}).to_csv(
        path, index=False
    )
    return shared


def aggregate(tumor: str, transcript_path: Path, tissue_path: Path,
              genes_path: Path, aggregated_path: Path, bins_path: Path) -> dict:
    started = time.perf_counter()
    db_path = TMP / f"{tumor}.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("SET threads=12")
    con.execute("SET memory_limit='100GB'")
    con.execute(f"SET temp_directory='{(TMP / ('ducktmp_' + tumor)).as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    query = f"""
        CREATE OR REPLACE TABLE aggregated AS
        SELECT
            CAST(floor(t.x_location / {BIN_UM}) AS INTEGER) AS bin_x,
            CAST(floor(t.y_location / {BIN_UM}) AS INTEGER) AS bin_y,
            g.gene_index,
            CAST(count(*) AS INTEGER) AS count
        FROM read_parquet('{transcript_path.as_posix()}') t
        INNER JOIN read_csv_auto('{genes_path.as_posix()}') g
          ON t.feature_name = g.feature_name
        INNER JOIN read_parquet('{tissue_path.as_posix()}') m
          ON CAST(floor(t.x_location / {BIN_UM}) AS INTEGER) = m.bin_x
         AND CAST(floor(t.y_location / {BIN_UM}) AS INTEGER) = m.bin_y
        WHERE t.is_gene = true AND t.qv > 20
        GROUP BY 1, 2, 3
    """
    con.execute(query)
    n_records = int(con.execute("SELECT count(*) FROM aggregated").fetchone()[0])
    n_bins = int(con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT bin_x, bin_y FROM aggregated)"
    ).fetchone()[0])
    con.execute(f"""
        COPY (SELECT * FROM aggregated ORDER BY bin_x, bin_y, gene_index)
        TO '{aggregated_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
        COPY (SELECT DISTINCT bin_x, bin_y FROM aggregated ORDER BY bin_x, bin_y)
        TO '{bins_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    return {
        "retained_8um_bins_with_gene_signal": n_bins,
        "nonzero_bin_gene_records": n_records,
        "aggregation_runtime_seconds": time.perf_counter() - started,
    }


def write_h5ad(tumor: str, shared: np.ndarray, aggregated_path: Path,
               bins_path: Path, output: Path) -> dict:
    bins = pq.read_table(bins_path).to_pandas()
    bin_keys = (bins.bin_x.to_numpy(np.int64) << 32) | bins.bin_y.to_numpy(np.int64)
    rows, cols, values = [], [], []
    parquet = pq.ParquetFile(aggregated_path)
    for batch in parquet.iter_batches(batch_size=2_000_000):
        table = pa.Table.from_batches([batch])
        bx = table["bin_x"].to_numpy().astype(np.int64, copy=False)
        by = table["bin_y"].to_numpy().astype(np.int64, copy=False)
        keys = (bx << 32) | by
        local_rows = np.searchsorted(bin_keys, keys).astype(np.int32)
        if np.any(local_rows >= len(bin_keys)) or not np.array_equal(bin_keys[local_rows], keys):
            raise RuntimeError("Aggregated bin absent from ordered bin table")
        rows.append(local_rows)
        cols.append(table["gene_index"].to_numpy().astype(np.int32, copy=False))
        values.append(table["count"].to_numpy().astype(np.int32, copy=False))
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(bins), len(shared)), dtype=np.int32,
    ).tocsr()
    if matrix.nnz != sum(len(x) for x in values):
        raise RuntimeError("Unexpected duplicate bin-gene records")
    obs = pd.DataFrame(index=[f"{tumor}_{x}_{y}" for x, y in zip(bins.bin_x, bins.bin_y)])
    obs["bin_x"] = bins.bin_x.to_numpy(np.int32)
    obs["bin_y"] = bins.bin_y.to_numpy(np.int32)
    var = pd.DataFrame(index=pd.Index(shared, name="gene"))
    obj = ad.AnnData(matrix, obs=obs, var=var)
    obj.obsm["spatial"] = bins[["bin_x", "bin_y"]].to_numpy(np.float32) * BIN_UM
    obj.uns["preprocessing"] = {
        "source": "official SPATCH Xenium transcript calls",
        "quality_filter": "is_gene == true and qv > 20",
        "aggregation": "floor(x_location/8), floor(y_location/8), integer call sum",
        "tissue_mask": "official SPATCH DAPI Otsu/largest-contour rule",
        "bin_size_um": BIN_UM,
        "hidden_expression_status": "sealed preprocessing output; not used for panel/model selection",
    }
    obj.write_h5ad(output, compression="gzip")
    return {
        "shape": [int(obj.n_obs), int(obj.n_vars)],
        "nnz": int(obj.X.nnz),
        "total_retained_calls": int(obj.X.sum()),
        "output_sha256": sha256(output),
    }


def run(tumor: str) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    local = TMP / tumor
    local.mkdir(parents=True, exist_ok=True)
    xenium = DATA / f"Xenium_{tumor}"
    transcript_path = xenium / "transcripts.parquet"
    dapi_path = xenium / "DAPI.tif"
    tissue_path = local / "dapi_tissue_bins.parquet"
    genes_path = local / "shared_genes.csv"
    aggregated_path = local / "aggregated_counts.parquet"
    bins_path = local / "retained_bins.parquet"
    output = OUT / f"Xenium_{tumor}_8um_shared_counts.h5ad"

    metadata = {
        "tumor": tumor,
        "status": "OUTCOME_AGNOSTIC_PREPROCESSING",
        "official_spatch_code_commit": "c20108f60660b86bc2740607d84fd825beaaceca",
        "transcript_parquet": str(transcript_path),
        "transcript_parquet_sha256": sha256(transcript_path),
        "dapi": str(dapi_path),
        "dapi_sha256": sha256(dapi_path),
        "target_nonanchor_used_for_selection_or_tuning": False,
    }
    shared = shared_gene_table(tumor, genes_path)
    metadata["shared_gene_count"] = int(len(shared))
    metadata["mask"] = dapi_mask_and_bins(dapi_path, tissue_path)
    metadata["aggregation"] = aggregate(
        tumor, transcript_path, tissue_path, genes_path, aggregated_path, bins_path
    )
    metadata["sealed_output"] = write_h5ad(
        tumor, shared, aggregated_path, bins_path, output
    )
    (OUT / f"Xenium_{tumor}_8um_preprocessing.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata["sealed_output"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor", choices=TUMORS, required=True)
    args = parser.parse_args()
    run(args.tumor)
