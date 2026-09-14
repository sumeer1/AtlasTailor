#!/usr/bin/env python3
"""Construct sealed CosMx 8-um count matrices using official SPATCH rules."""
from __future__ import annotations

import argparse
import hashlib
import json
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
PACKAGES = ROOT / "official_data" / "packages"
OUT = ROOT / "SEALED_TARGET_8UM"
TMP = ROOT / "work" / "cosmx_8um"
TUMORS = ("COAD", "HCC", "OV")
PIXEL_UM = 0.12
BIN_UM = 8.0
FOV_SIZE_PIXELS = 4256.0
KERNEL_WIDTH = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def common_genes(tumor: str, transcript_path: Path, output: Path) -> tuple[np.ndarray, dict]:
    source_path = DATA / f"VisiumHD_FFPE_{tumor}" / "adata.h5ad"
    source = ad.read_h5ad(source_path, backed="r")
    source_names = source.var_names.astype(str).to_numpy()
    source.file.close()
    con = duckdb.connect()
    target_names = np.asarray([
        row[0] for row in con.execute(
            "SELECT DISTINCT target FROM read_parquet(?) ORDER BY target", [str(transcript_path)]
        ).fetchall()
    ], dtype=str)
    con.close()
    source_set = set(source_names)
    shared = np.asarray([gene for gene in target_names if gene in source_set], dtype=str)
    if len(shared) != len(np.unique(shared)):
        raise RuntimeError("Shared gene names are not unique")
    pd.DataFrame({"target": shared, "gene_index": np.arange(len(shared))}).to_csv(output, index=False)
    return shared, {
        "source_gene_axis_count": int(len(source_names)),
        "target_call_identity_count_including_controls": int(len(target_names)),
        "shared_gene_identity_count": int(len(shared)),
    }


def aggregate_calls(transcript_path: Path, genes_path: Path, output: Path, bins_path: Path,
                    database: Path) -> dict:
    started = time.perf_counter()
    con = duckdb.connect(str(database))
    con.execute("SET threads=16")
    con.execute("SET memory_limit='110GB'")
    con.execute(f"SET temp_directory='{(database.parent / 'ducktmp').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
        CREATE OR REPLACE TABLE aggregated AS
        SELECT
          CAST(floor(t.x_global_px * {PIXEL_UM} / {BIN_UM}) AS INTEGER) AS bin_x,
          CAST(floor(t.y_global_px * {PIXEL_UM} / {BIN_UM}) AS INTEGER) AS bin_y,
          g.gene_index,
          CAST(count(*) AS INTEGER) AS count
        FROM read_parquet('{transcript_path.as_posix()}') t
        INNER JOIN read_csv_auto('{genes_path.as_posix()}') g ON t.target = g.target
        GROUP BY 1, 2, 3
    """)
    con.execute(f"""
        COPY (SELECT * FROM aggregated ORDER BY bin_x, bin_y, gene_index)
        TO '{output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
        COPY (SELECT DISTINCT bin_x, bin_y FROM aggregated ORDER BY bin_x, bin_y)
        TO '{bins_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    records = int(con.execute("SELECT count(*) FROM aggregated").fetchone()[0])
    bins = int(con.execute("SELECT count(*) FROM (SELECT DISTINCT bin_x,bin_y FROM aggregated)").fetchone()[0])
    calls = int(con.execute("SELECT sum(count) FROM aggregated").fetchone()[0])
    con.close()
    return {
        "nonzero_bin_gene_records_before_tissue_mask": records,
        "nonempty_8um_bins_before_tissue_mask": bins,
        "shared_gene_calls_before_tissue_mask": calls,
        "aggregation_runtime_seconds": time.perf_counter() - started,
    }


def tissue_bins(dapi_path: Path, candidate_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    image = tifffile.imread(dapi_path)
    if image.ndim != 2 or float(image.max()) <= 0:
        raise RuntimeError(f"Unexpected CosMx DAPI image: {dapi_path} {image.shape}")
    height, width = image.shape
    maximum = float(image.max())
    image8 = np.empty(image.shape, np.uint8)
    factor = np.float32(255.0 / maximum)
    for first in range(0, height, 512):
        block = image[first:first + 512].astype(np.float32)
        image8[first:first + len(block)] = np.floor(block * factor).astype(np.uint8)
    del image
    otsu, threshold = cv2.threshold(image8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    del image8
    kernel = np.ones((KERNEL_WIDTH, KERNEL_WIDTH), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=2)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(f"No CosMx tissue contour found in {dapi_path}")
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(threshold)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    del threshold, kernel, contours

    candidates = pq.read_table(candidate_path).to_pandas()
    pixel_x = np.floor(candidates.bin_x.to_numpy(np.float64) * BIN_UM / PIXEL_UM).astype(np.int64)
    pixel_y = np.floor(
        height - candidates.bin_y.to_numpy(np.float64) * BIN_UM / PIXEL_UM - FOV_SIZE_PIXELS
    ).astype(np.int64)
    in_bounds = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
    keep = np.zeros(len(candidates), bool)
    keep[in_bounds] = mask[pixel_y[in_bounds], pixel_x[in_bounds]] == 255
    retained = candidates.loc[keep].copy()
    pq.write_table(pa.Table.from_pandas(retained, preserve_index=False), output, compression="zstd")
    result = {
        "dapi_shape_pixels": [int(height), int(width)],
        "cosmx_pixel_size_um": PIXEL_UM,
        "cosmx_fov_size_pixels": FOV_SIZE_PIXELS,
        "kernel_width_pixels": KERNEL_WIDTH,
        "otsu_threshold_uint8": float(otsu),
        "largest_contour_area_pixels": float(cv2.contourArea(contour)),
        "candidate_bins_in_dapi_bounds": int(in_bounds.sum()),
        "retained_tissue_bins": int(keep.sum()),
        "mask_runtime_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_h5ad(shared: np.ndarray, aggregate_path: Path, tissue_path: Path,
               output: Path, tumor: str) -> dict:
    tissue = pq.read_table(tissue_path).to_pandas()
    keys = (tissue.bin_x.to_numpy(np.int64) << 32) | (tissue.bin_y.to_numpy(np.int64) & 0xffffffff)
    order = np.argsort(keys, kind="stable")
    tissue = tissue.iloc[order].reset_index(drop=True)
    keys = keys[order]
    con = duckdb.connect()
    filtered = output.parent / f".{tumor}_filtered_records.parquet"
    con.execute(f"""
        COPY (
          SELECT a.* FROM read_parquet('{aggregate_path.as_posix()}') a
          INNER JOIN read_parquet('{tissue_path.as_posix()}') t USING (bin_x, bin_y)
          ORDER BY a.bin_x, a.bin_y, a.gene_index
        ) TO '{filtered.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    row_parts, col_parts, value_parts = [], [], []
    parquet = pq.ParquetFile(filtered)
    for batch in parquet.iter_batches(batch_size=2_000_000):
        table = pa.Table.from_batches([batch])
        bx = table["bin_x"].to_numpy().astype(np.int64, copy=False)
        by = table["bin_y"].to_numpy().astype(np.int64, copy=False)
        local_keys = (bx << 32) | (by & 0xffffffff)
        rows = np.searchsorted(keys, local_keys).astype(np.int32)
        if np.any(rows >= len(keys)) or not np.array_equal(keys[rows], local_keys):
            raise RuntimeError("Filtered CosMx bin absent from tissue table")
        row_parts.append(rows)
        col_parts.append(table["gene_index"].to_numpy().astype(np.int32, copy=False))
        value_parts.append(table["count"].to_numpy().astype(np.int32, copy=False))
    matrix = sparse.coo_matrix(
        (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(len(tissue), len(shared)), dtype=np.int32,
    ).tocsr()
    obs = pd.DataFrame(index=[f"{tumor}_{x}_{y}" for x, y in zip(tissue.bin_x, tissue.bin_y)])
    obs["bin_x"] = tissue.bin_x.to_numpy(np.int32)
    obs["bin_y"] = tissue.bin_y.to_numpy(np.int32)
    obj = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=pd.Index(shared, name="gene")))
    obj.obsm["spatial"] = tissue[["bin_x", "bin_y"]].to_numpy(np.float32) * BIN_UM
    obj.uns["preprocessing"] = {
        "source": "official SPATCH CosMx transcript calls",
        "call_filter": "official distributed calls; negative controls removed by source gene-identity intersection",
        "aggregation": "floor(x_global_px*0.12/8), floor(y_global_px*0.12/8), integer call sum",
        "tissue_mask": "official SPATCH DAPI Otsu/largest-contour rule; kernel width 1000",
        "dapi_coordinate_transform": "x=x_um/0.12; y=image_height-y_um/0.12-4256",
        "bin_size_um": BIN_UM,
        "hidden_expression_status": "sealed preprocessing output; not used for panel/model selection",
    }
    obj.write_h5ad(output, compression="gzip")
    filtered.unlink()
    return {
        "shape": [int(obj.n_obs), int(obj.n_vars)],
        "nnz": int(obj.X.nnz),
        "total_retained_shared_gene_calls": int(obj.X.sum()),
        "output_sha256": sha256(output),
    }


def run(tumor: str, resume: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local = TMP / tumor
    local.mkdir(parents=True, exist_ok=True)
    transcript_path = DATA / f"CosMx_{tumor}" / "transcripts" / "tx_file.parquet"
    if not transcript_path.exists():
        transcript_path = DATA / f"CosMx_{tumor}" / "tx_file.parquet"
    dapi_path = PACKAGES / f"{tumor}_dapi" / "DAPI.tif"
    output = OUT / f"CosMx_{tumor}_8um_shared_counts.h5ad"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    genes_path = local / "shared_genes.csv"
    aggregate_path = local / "aggregated.parquet"
    candidates_path = local / "candidate_bins.parquet"
    tissue_path = local / "tissue_bins.parquet"
    if resume and genes_path.exists() and aggregate_path.exists() and candidates_path.exists():
        table = pd.read_csv(genes_path)
        shared = table.target.astype(str).to_numpy()
        con = duckdb.connect()
        target_count = int(con.execute(
            "SELECT count(DISTINCT target) FROM read_parquet(?)", [str(transcript_path)]
        ).fetchone()[0])
        records, calls = con.execute(
            "SELECT count(*), sum(count) FROM read_parquet(?)", [str(aggregate_path)]
        ).fetchone()
        con.close()
        identity = {
            "source_gene_axis_count": 18085,
            "target_call_identity_count_including_controls": target_count,
            "shared_gene_identity_count": int(len(shared)),
        }
        aggregate = {
            "nonzero_bin_gene_records_before_tissue_mask": int(records),
            "nonempty_8um_bins_before_tissue_mask": int(pq.read_table(candidates_path).num_rows),
            "shared_gene_calls_before_tissue_mask": int(calls),
            "aggregation_runtime_seconds": None,
        }
    else:
        shared, identity = common_genes(tumor, transcript_path, genes_path)
        aggregate = aggregate_calls(transcript_path, genes_path, aggregate_path, candidates_path,
                                    local / "aggregation.duckdb")
    metadata_path = tissue_path.with_suffix(".metadata.json")
    if resume and tissue_path.exists():
        if metadata_path.exists():
            mask = json.loads(metadata_path.read_text())
        else:
            with tifffile.TiffFile(dapi_path) as tif:
                shape = tif.pages[0].shape
            mask = {
                "dapi_shape_pixels": [int(shape[0]), int(shape[1])],
                "cosmx_pixel_size_um": PIXEL_UM,
                "cosmx_fov_size_pixels": FOV_SIZE_PIXELS,
                "kernel_width_pixels": KERNEL_WIDTH,
                "otsu_threshold_uint8": None,
                "largest_contour_area_pixels": None,
                "candidate_bins_in_dapi_bounds": None,
                "retained_tissue_bins": int(pq.read_table(tissue_path).num_rows),
                "mask_runtime_seconds": None,
                "resume_note": "Existing fixed mask reused after downstream sparse-index ordering assertion; mask unchanged",
            }
            metadata_path.write_text(json.dumps(mask, indent=2) + "\n")
    else:
        mask = tissue_bins(dapi_path, candidates_path, tissue_path)
    container = write_h5ad(shared, aggregate_path, tissue_path, output, tumor)
    report = {
        "tumor": tumor,
        "target_platform": "CosMx 6K",
        "transcript_path": str(transcript_path),
        "transcript_sha256": sha256(transcript_path),
        "dapi_path": str(dapi_path),
        "dapi_sha256": sha256(dapi_path),
        **identity, **aggregate, **mask, **container,
    }
    (OUT / f"CosMx_{tumor}_8um_preprocessing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor", choices=TUMORS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.tumor, args.resume)


if __name__ == "__main__":
    main()
