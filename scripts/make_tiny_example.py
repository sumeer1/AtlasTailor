#!/usr/bin/env python3
"""Create deterministic, license-clean fixtures for automated software tests."""
from pathlib import Path
import anndata as ad
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def specimen(dim: int, seed: int, n: int = 128, g: int = 48):
    rng = np.random.default_rng(seed)
    coordinates = rng.normal(size=(n, dim)).astype(np.float32)
    latent = np.c_[np.ones(n), coordinates, np.square(coordinates)]
    weights = rng.normal(0, 0.35, size=(latent.shape[1], g))
    rate = np.exp(np.clip(latent @ weights + rng.normal(0.4, 0.15, g), -1.5, 2.2))
    counts = rng.poisson(rate).astype(np.float32)
    return coordinates, counts

def write_pair(dim: int):
    out = ROOT / "tests" / "fixtures" / f"minimal_{dim}d"
    out.mkdir(parents=True, exist_ok=True)
    coordinates, counts = specimen(dim, 110 + dim)
    rng = np.random.default_rng(220 + dim)
    target_counts = rng.poisson(np.maximum(counts * np.exp(0.10 * coordinates[:, [0]]), 0.05)).astype(np.float32)
    genes = np.asarray([f"gene_{j:02d}" for j in range(counts.shape[1])])
    for name, matrix in (("reference", counts), ("target", target_counts)):
        a = ad.AnnData(X=matrix)
        a.var_names = genes
        a.obs_names = [f"{name}_{j:03d}" for j in range(len(matrix))]
        a.obsm["spatial"] = coordinates.copy()
        a.uns["fixture"] = {"synthetic": True, "license": "BSD-3-Clause", "seed": 110 + dim}
        a.write_h5ad(out / f"{name}.h5ad")
    (out / "config.yaml").write_text(yaml.safe_dump({"adapt": {"registration": "pre_registered", "seeds": [42]}}))
    (out / "README.md").write_text(
        f"# Minimal {dim}D fixture\n\nDeterministic synthetic counts and coordinates for API/CLI tests. "
        "These files are not biological evidence and are distributed under the software license.\n"
    )

def tracking_fixture():
    early_dir = ROOT / "tests" / "fixtures" / "minimal_2d"
    early = ad.read_h5ad(early_dir / "target.h5ad")
    n, g = early.shape
    rng = np.random.default_rng(412)
    temporal = rng.normal(0, 0.03, size=(n, g)).astype(np.float32)
    np.savez_compressed(
        early_dir / "tracking_bundle.npz",
        descendant_ancestor=np.arange(n),
        early_ancestors=np.arange(n),
        late_coordinates=np.asarray(early.obsm["spatial"], np.float32),
        temporal_residual_prediction=temporal,
        supported=np.arange(g) % 3 != 0,
    )

if __name__ == "__main__":
    write_pair(2)
    write_pair(3)
    tracking_fixture()
