#!/usr/bin/env python3
"""Privileged simulator of the 16-gene target measurement.

Only values at source-requested anchor indices are read from the target HDF5
data vector. Non-anchor values are not loaded, summarized, filtered or emitted.
The resulting object is the sole target molecular input accepted by prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from dlpfc_common import canonicalize, decode, load_section_metadata, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--target-h5", type=Path, required=True)
    parser.add_argument("--spot-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite {args.output}")
    request = json.loads(args.request.read_text())
    requested_ids = np.asarray(request["anchor_gene_ids"])
    requested_names = np.asarray(request["anchor_gene_names"])

    with h5py.File(args.target_h5, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(x) for x in group["shape"][:])
        barcodes = decode(group["barcodes"][:])
        gene_ids = decode(group["features/id"][:])
        gene_names = decode(group["features/name"][:])
        index_by_id = {gene_id: index for index, gene_id in enumerate(gene_ids)}
        anchor_rows = np.asarray([index_by_id[gene_id] for gene_id in requested_ids], np.int64)
        if not np.array_equal(gene_names[anchor_rows], requested_names):
            raise RuntimeError("Requested target anchor identities do not match target feature axis")
        indptr = group["indptr"][:]
        sparse_indices = group["indices"][:]
        anchor_lookup = {int(row): local for local, row in enumerate(anchor_rows)}
        counts = np.zeros((shape[1], len(anchor_rows)), np.float32)
        selected_positions = []
        selected_spots = []
        selected_locals = []
        for spot in range(shape[1]):
            begin, end = int(indptr[spot]), int(indptr[spot + 1])
            for position in range(begin, end):
                local = anchor_lookup.get(int(sparse_indices[position]))
                if local is not None:
                    selected_positions.append(position)
                    selected_spots.append(spot)
                    selected_locals.append(local)
        selected_positions = np.asarray(selected_positions, np.int64)
        # HDF5 fancy indexing reads expression values only at anchor positions.
        selected_values = group["data"][selected_positions]
        counts[np.asarray(selected_spots), np.asarray(selected_locals)] = selected_values

    metadata = load_section_metadata(
        args.spot_metadata, request["target"], barcodes, include_layers=False
    )
    coordinates_raw = metadata[["x_fullres_px", "y_fullres_px"]].to_numpy(np.float32)
    coordinates, transform = canonicalize(coordinates_raw)
    np.savez(
        args.output,
        target_section=np.asarray(request["target"]),
        target_barcodes=barcodes,
        target_coordinates=coordinates,
        target_coordinates_fullres_px=coordinates_raw,
        anchor_gene_ids=requested_ids,
        anchor_gene_names=requested_names,
        anchor_counts=counts,
    )
    audit = {
        "status": "ANCHOR_ONLY_TARGET_OBJECT_SERIALIZED",
        "direction": request["direction"],
        "target_section": request["target"],
        "target_h5_sha256": sha256(args.target_h5),
        "requested_anchor_count": int(len(anchor_rows)),
        "anchor_nonzero_values_read": int(len(selected_positions)),
        "nonanchor_expression_values_read": 0,
        "nonanchor_expression_values_serialized": 0,
        "target_layer_labels_accessed": False,
        "target_detection_or_variance_filtering": False,
        "coordinate_transform": transform,
        "anchor_object": str(args.output),
        "anchor_object_sha256": sha256(args.output),
        "qualification": "Sparse gene-index structure was scanned only to locate requested anchors; non-anchor data values were not read.",
    }
    save_json(args.output.with_suffix(".audit.json"), audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
