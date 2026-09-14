#!/usr/bin/env python3
"""Validate the presentation-only package, write hashes, and print its summary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
ORIGINAL = PROJECT / "PI_revision" / "23_SPATCH_cross_platform_MAP_20260824_111004"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


lock = pd.read_csv(ORIGINAL / "02_PREDICTION_LOCK_MANIFEST.tsv", sep="\t")
failures = []
for row in lock.itertuples():
    actual = sha256(Path(row.path))
    if actual != row.sha256:
        failures.append(f"{row.tumor}:{row.artifact}")
for tumor in ("COAD", "HCC", "OV"):
    metadata = json.loads((ORIGINAL / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_preprocessing.json").read_text())
    target = ORIGINAL / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_shared_counts.h5ad"
    if sha256(target) != metadata["sealed_output"]["output_sha256"]:
        failures.append(f"{tumor}:sealed_target")
if failures:
    raise RuntimeError(f"Frozen hash failure: {failures}")

open_markers = {
    tumor: ORIGINAL / "work" / tumor / "HIDDEN_TARGET_OPENED_AFTER_GLOBAL_LOCK.txt"
    for tumor in ("COAD", "HCC", "OV")
}
global_lock = ORIGINAL / "02_PREDICTION_LOCK_MANIFEST.tsv"
assert all(path.stat().st_mtime > global_lock.stat().st_mtime for path in open_markers.values())
prediction_rows = lock[lock.artifact.str.startswith("prediction:")]
assert all(Path(path).stat().st_mtime < min(x.stat().st_mtime for x in open_markers.values())
           for path in prediction_rows.path)

pair = pd.read_csv(HERE / "01_PAIR_LEVEL_RESULTS_IDW_vs_MAP.tsv", sep="\t")
specimen = pd.read_csv(HERE / "02_SPECIMEN_LEVEL_RESULTS_IDW_vs_MAP.tsv", sep="\t")
gene = pd.read_csv(HERE / "03_GENE_LEVEL_MAP_vs_IDW.tsv", sep="\t")
registration = pd.read_csv(HERE / "04_REGISTRATION_RESULTS.tsv", sep="\t")
hcc = pd.read_csv(HERE / "04B_HCC_SECONDARY_STRESS_TEST.tsv", sep="\t").iloc[0]

assert set(pair.tumor) == {"COAD", "OV"} and set(pair.method) == {"registered 12-NN IDW", "HyperSpatial-MAP"}
assert len(pair) == 4 and len(specimen) == 2 and len(gene) == 763 + 697 and len(registration) == 6
assert set(specimen.tumor) == {"COAD", "OV"} and set(gene.tumor) == {"COAD", "OV"}
assert set(registration.tumor) == {"COAD", "OV"}
assert (specimen.measured_anchor_count == 16).all()
assert specimen.set_index("tumor").hidden_shared_gene_count.to_dict() == {"COAD": 763, "OV": 697}
assert int(hcc.hidden_shared_gene_count) == 151
assert np.isclose(hcc.IDW_median_gene_wise_spatial_pearson, -0.0011823562781821)
assert np.isclose(hcc.MAP_median_gene_wise_spatial_pearson, -0.0001079184557561)
assert np.isclose(hcc.IDW_log1p_RMSE, 0.5275201332191763)
assert np.isclose(hcc.MAP_log1p_RMSE, 0.5248553609452737)
assert np.isclose(hcc.MAP_pooled_residual_pearson, 0.192481178711276)
assert specimen.IDW_residual_pearson.isna().all()

decision_a = bool(((specimen.MAP_median_gene_wise_spatial_pearson > specimen.IDW_median_gene_wise_spatial_pearson) &
                   (specimen.MAP_log1p_RMSE < specimen.IDW_log1p_RMSE) &
                   (specimen.MAP_pooled_residual_pearson > 0)).all())
assert decision_a

primary_files = [
    HERE / "01_PAIR_LEVEL_RESULTS_IDW_vs_MAP.tsv",
    HERE / "02_SPECIMEN_LEVEL_RESULTS_IDW_vs_MAP.tsv",
    HERE / "03_GENE_LEVEL_MAP_vs_IDW.tsv",
    HERE / "04_REGISTRATION_RESULTS.tsv",
    HERE / "figures" / "SPATCH_cross_platform_IDW_vs_MAP_main.pdf",
    HERE / "figures" / "SPATCH_cross_platform_IDW_vs_MAP_main.png",
    HERE / "figures" / "SPATCH_cross_platform_IDW_vs_MAP_main.svg",
]
for path in primary_files:
    text = path.read_text(errors="ignore").lower() if path.suffix in {".tsv", ".md", ".txt"} else ""
    if text:
        assert "hcc" not in text and "liver" not in text
        assert "gimvi" not in text and "anchor-only" not in text and "anchor_only" not in text

s = specimen.set_index("tumor")
coad, ov = s.loc["COAD"], s.loc["OV"]
summary = f"""SPATCH manuscript-facing IDW-vs-MAP reorganization

1. Original directory: {ORIGINAL}
2. Revised directory: {HERE}
3. Predictions recomputed: NO
4. Registration recomputed: NO
5. Anchors changed: NO
6. Model assignments changed: NO
7. Frozen hashes verified: YES (27/27 prediction-lock artifacts; 3/3 sealed target matrices)
8. Primary COAD: hidden genes=763; bins=629321; spatial Pearson IDW={coad.IDW_median_gene_wise_spatial_pearson:.10f}, MAP={coad.MAP_median_gene_wise_spatial_pearson:.10f}; log1p RMSE IDW={coad.IDW_log1p_RMSE:.10f}, MAP={coad.MAP_log1p_RMSE:.10f}; MAP pooled residual r={coad.MAP_pooled_residual_pearson:.10f}
9. Primary OV: hidden genes=697; bins=582568; spatial Pearson IDW={ov.IDW_median_gene_wise_spatial_pearson:.10f}, MAP={ov.MAP_median_gene_wise_spatial_pearson:.10f}; log1p RMSE IDW={ov.IDW_log1p_RMSE:.10f}, MAP={ov.MAP_log1p_RMSE:.10f}; MAP pooled residual r={ov.MAP_pooled_residual_pearson:.10f}
10. Secondary HCC: hidden genes=151; bins={int(hcc.target_bin_count)}; spatial Pearson IDW={hcc.IDW_median_gene_wise_spatial_pearson:.10f}, MAP={hcc.MAP_median_gene_wise_spatial_pearson:.10f}; log1p RMSE IDW={hcc.IDW_log1p_RMSE:.10f}, MAP={hcc.MAP_log1p_RMSE:.10f}; MAP pooled residual r={hcc.MAP_pooled_residual_pearson:.10f}
11. Separately displayed internal operators absent from new primary figures/tables: YES
12. HCC absent from main figure/primary tables and preserved in secondary outputs: YES
13. Primary manuscript-facing decision: A
"""
(HERE / "11_TERMINAL_SUMMARY.txt").write_text(summary)

# Hash all frozen artifacts plus every revised file except the manifest itself.
lines = [
    "# SHA-256 manifest",
    f"# Original frozen directory: {ORIGINAL}",
    f"# Revised presentation directory: {HERE}",
    "# Predictions/registrations recomputed: NO",
    "",
    "# Original prediction-lock artifacts (verified against original recorded hashes)",
]
for row in lock.sort_values(["tumor", "artifact"]).itertuples():
    lines.append(f"{row.sha256}  {row.path}")
for tumor in ("COAD", "HCC", "OV"):
    metadata = json.loads((ORIGINAL / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_preprocessing.json").read_text())
    path = ORIGINAL / "SEALED_TARGET_8UM" / f"Xenium_{tumor}_8um_shared_counts.h5ad"
    lines.append(f"{metadata['sealed_output']['output_sha256']}  {path}")
lines.extend(["", "# Revised manuscript-facing files"])
revised = sorted(path for path in HERE.rglob("*") if path.is_file() and
                 path.name != "10_SHA256_MANIFEST.txt" and "__pycache__" not in path.parts)
for path in revised:
    lines.append(f"{sha256(path)}  {path}")
(HERE / "10_SHA256_MANIFEST.txt").write_text("\n".join(lines) + "\n")

print(summary)
print("14. Newly created files:")
for path in sorted(path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts):
    print(f"   - {path.relative_to(HERE)}")
