#!/usr/bin/env python3
"""Read-only access to frozen predictions, frozen manifests and measured data.

Normalisation here mirrors the locked runner exactly (median source depth,
``log1p`` of depth-normalised counts, frozen coordinate key). Any divergence
would produce a residual field that is not the one the Class A decision rests
on, so :func:`load_target` is gated by a reproduction check in
``residual_fields``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from .config import ANNOTATION_COLUMNS, IDW, PROJECT_ROOT, SEEDS, TargetRun, is_protected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_only(path: Path) -> Path:
    """Assert a frozen input exists and is readable before it is opened."""
    if not path.exists():
        raise FileNotFoundError(f"Frozen input missing: {path}")
    return path


@dataclass
class TargetData:
    """Everything the audit needs for one frozen source-to-target transfer."""

    run: TargetRun
    genes: np.ndarray
    truth_log: np.ndarray
    coordinates: np.ndarray
    coordinate_key: str
    target_depth: np.ndarray
    target_counts_normalised: np.ndarray
    annotations: dict[str, np.ndarray]
    anchor_indices: np.ndarray
    anchor_genes: list[str]
    predictable_indices: np.ndarray
    evaluation_svg_indices: np.ndarray
    manifest: dict
    depth_target: float
    source_log: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.truth_log.shape[0])


def _canonicalize(coordinates: np.ndarray) -> np.ndarray:
    """Frozen coordinate canonicalisation (median-centred, 99th-percentile scaled)."""
    center = np.median(coordinates, axis=0)
    centered = coordinates - center
    scale = max(float(np.quantile(np.linalg.norm(centered, axis=1), 0.99)), 1e-6)
    return (centered / scale).astype(np.float32)


def _measured(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """Frozen loader: spliced + unspliced integer counts, coordinates, genes.

    Also returns the author-supplied annotation columns, which the frozen runner
    never read.
    """
    data = ad.read_h5ad(read_only(path))
    counts = np.asarray(data.layers["spliced"], dtype=np.float32)
    counts += np.asarray(data.layers["unspliced"], dtype=np.float32)
    if not np.allclose(counts, np.rint(counts)):
        raise AssertionError(f"{path.name}: spliced + unspliced is not integer count data")
    key = "global_sphere" if "global_sphere" in data.obsm else "spatial"
    coordinates = np.asarray(data.obsm[key], np.float32)
    genes = np.asarray(data.var_names.astype(str))
    annotations = {
        column: np.asarray(data.obs[column].astype(str))
        for column in ANNOTATION_COLUMNS
        if column in data.obs.columns
    }
    extra_obsm = {
        name: np.asarray(data.obsm[name], np.float32)
        for name in ("spatial", "spatial_rescaled_z")
        if name in data.obsm
    }
    annotations["__obsm__"] = extra_obsm  # type: ignore[assignment]
    return counts, coordinates, genes, key, annotations


def _normalize(counts: np.ndarray, depth_target: float) -> np.ndarray:
    depth = counts.sum(1)
    return counts * (depth_target / np.maximum(depth, 1.0))[:, None]


def verify_frozen_predictions(run: TargetRun) -> list[dict]:
    """Recompute sha256 of every frozen prediction and compare to its manifest.

    A mismatch means the frozen tree changed since the run was sealed, which
    invalidates the audit's premise.
    """
    manifest = json.loads(read_only(run.manifest_path).read_text())
    records = []
    for seed in SEEDS:
        for method, meta in manifest["methods"][str(seed)]["predictions"].items():
            path = _resolve_manifest_path(meta["path"])
            observed = sha256(read_only(path))
            records.append({
                "target_key": run.key,
                "seed": seed,
                "method": method,
                "path": str(path),
                "recorded_sha256": meta["sha256"],
                "observed_sha256": observed,
                "matches": observed == meta["sha256"],
            })
    return records


def _resolve_manifest_path(recorded: str) -> Path:
    """Manifest paths are recorded relative to the project root."""
    path = Path(recorded)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prediction_path(run: TargetRun, seed: int, method: str, manifest: dict) -> Path:
    path = _resolve_manifest_path(manifest["methods"][str(seed)]["predictions"][method]["path"])
    if not is_protected(path):
        raise RuntimeError(f"Prediction path is outside the frozen trees: {path}")
    return read_only(path)


def load_prediction(run: TargetRun, seed: int, method: str, manifest: dict) -> np.ndarray:
    return np.asarray(np.load(prediction_path(run, seed, method, manifest), mmap_mode="r"), np.float32)


def load_target(run: TargetRun) -> TargetData:
    """Rebuild the frozen evaluation state for one target volume."""
    manifest = json.loads(read_only(run.manifest_path).read_text())

    source_counts, _, source_genes, _, _ = _measured(run.source_file)
    depth_target = float(np.median(np.maximum(source_counts.sum(1), 1.0)))
    source_log = np.log1p(_normalize(source_counts, depth_target)).astype(np.float32)
    del source_counts

    target_counts, target_coordinates_raw, genes, coordinate_key, annotations = _measured(run.target_file)
    if not np.array_equal(genes, source_genes):
        raise AssertionError(f"{run.key}: gene axes differ between source and target")
    if len(set(genes)) != len(genes):
        raise AssertionError(f"{run.key}: duplicate gene names break name-based gene selection")

    target_depth = target_counts.sum(1).astype(np.float32)
    normalised = _normalize(target_counts, depth_target).astype(np.float32)
    truth_log = np.log1p(normalised).astype(np.float32)
    del target_counts

    anchor_indices = np.asarray(manifest["anchor_indices"], np.int64)
    predictable_names = set(manifest["source_defined_predictable_genes"])
    anchor_set = set(anchor_indices.tolist())
    predictable_indices = np.asarray(
        [i for i, gene in enumerate(genes) if gene in predictable_names and i not in anchor_set],
        np.int64,
    )

    obsm = annotations.pop("__obsm__", {})
    return TargetData(
        run=run,
        genes=genes,
        truth_log=truth_log,
        coordinates=_canonicalize(target_coordinates_raw),
        coordinate_key=coordinate_key,
        target_depth=target_depth,
        target_counts_normalised=normalised,
        annotations={**annotations, "__obsm__": obsm},  # type: ignore[dict-item]
        anchor_indices=anchor_indices,
        anchor_genes=list(manifest["anchor_genes"]),
        predictable_indices=predictable_indices,
        evaluation_svg_indices=np.asarray(manifest["evaluation_svg_indices"], np.int64),
        manifest=manifest,
        depth_target=depth_target,
        source_log=source_log,
    )


def base_field(run: TargetRun, seed: int, manifest: dict) -> np.ndarray:
    """The registered atlas transfer that defines the residual origin."""
    return load_prediction(run, seed, IDW, manifest)
