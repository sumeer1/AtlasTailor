#!/usr/bin/env python3
"""Geometry/count-axis audit of the deposited multiplexed Visium arrays.

This script reads feature identifiers and sparse-matrix structure for audit only.
It does not materialize expression values.
"""
from pathlib import Path
import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "extracted" / "Visium"


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def components(coords, radius=2.1):
    tree = cKDTree(coords)
    pairs = np.array(list(tree.query_pairs(radius)), dtype=int)
    if len(pairs) == 0:
        return np.arange(len(coords))
    rows = np.r_[pairs[:, 0], pairs[:, 1]]
    cols = np.r_[pairs[:, 1], pairs[:, 0]]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(coords), len(coords)))
    return connected_components(graph, directed=False, return_labels=True)[1]


out = []
for arr in sorted(p for p in DATA.iterdir() if p.is_dir()):
    h5 = arr / "filtered_feature_bc_matrix.h5"
    pos = arr / "spatial" / "tissue_positions.csv"
    with h5py.File(h5, "r") as f:
        g = f["matrix"]
        barcodes = dec(g["barcodes"][:])
        names = dec(g["features"]["name"][:])
        ids = dec(g["features"]["id"][:])
        shape = tuple(g["shape"][:])
        nnz = len(g["data"])
    positions = {}
    with pos.open(newline="") as fh:
        for row in csv.DictReader(fh):
            positions[row["barcode"]] = (
                float(row["array_row"]), float(row["array_col"]),
                float(row["pxl_row_in_fullres"]), float(row["pxl_col_in_fullres"]),
            )
    keep = np.array([b in positions for b in barcodes])
    b = barcodes[keep]
    xy_grid = np.array([[positions[x][0], positions[x][1]] for x in b])
    xy_px = np.array([[positions[x][2], positions[x][3]] for x in b])
    labels = components(xy_grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xy_grid[:, 1], -xy_grid[:, 0], c=labels, s=6, cmap="tab20")
    for lab in np.unique(labels):
        idx = labels == lab
        if idx.sum() >= 10:
            ax.text(xy_grid[idx, 1].mean(), -xy_grid[idx, 0].mean(), str(lab), fontsize=8)
    ax.set_title(arr.name)
    ax.set_aspect("equal")
    fig.tight_layout()
    plot_dir = ROOT / "work" / "component_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"{arr.name}.png", dpi=180)
    plt.close(fig)
    counts = [(lab, int(np.sum(labels == lab))) for lab in np.unique(labels)]
    counts.sort(key=lambda z: -z[1])
    for rank, (lab, n) in enumerate(counts, 1):
        idx = labels == lab
        if n < 10:
            continue
        out.append({
            "array": arr.name,
            "component_rank": rank,
            "spots": n,
            "grid_row_min": xy_grid[idx, 0].min(),
            "grid_row_max": xy_grid[idx, 0].max(),
            "grid_col_min": xy_grid[idx, 1].min(),
            "grid_col_max": xy_grid[idx, 1].max(),
            "pixel_row_centroid": xy_px[idx, 0].mean(),
            "pixel_col_centroid": xy_px[idx, 1].mean(),
            "feature_rows": shape[0],
            "matrix_spots": shape[1],
            "unique_gene_names": len(set(names)),
            "unique_feature_ids": len(set(ids)),
            "nnz": nnz,
        })

dest = ROOT / "work" / "visium_geometry_components.tsv"
dest.parent.mkdir(parents=True, exist_ok=True)
with dest.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(dest)
for row in out:
    print("\t".join(str(row[k]) for k in ("array", "component_rank", "spots", "grid_row_min", "grid_row_max", "grid_col_min", "grid_col_max")))
