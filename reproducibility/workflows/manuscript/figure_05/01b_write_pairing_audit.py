#!/usr/bin/env python3
"""Write the pre-model SPATCH data/pairing audit from sealed preprocessing records."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "SEALED_TARGET_8UM"
SPECIMENS = {"HCC": "Benchmark_01", "OV": "Benchmark_02", "COAD": "Benchmark_03"}
PORTAL_CELLS = {"HCC": 213297, "OV": 255628, "COAD": 261731}

reports = {}
for tumor in ("COAD", "HCC", "OV"):
    path = SEALED / f"CosMx_{tumor}_8um_preprocessing.json"
    if not path.exists():
        raise RuntimeError(f"Pre-model audit cannot be finalized without {path}")
    reports[tumor] = json.loads(path.read_text())

rows = []
for tumor in ("COAD", "HCC", "OV"):
    report = reports[tumor]
    rows.append(
        f"| {SPECIMENS[tumor]} | {tumor} | Visium HD FFPE section | CosMx 6K section | "
        f"same FFPE block; serial order verified | 18,085 source-axis genes | 6,175-gene CosMx panel "
        f"({report['target_call_identity_count_including_controls']:,} distributed call identities including controls) | "
        f"{report['shared_gene_identity_count']:,} | {report['retained_tissue_bins']:,} non-empty tissue 8-µm bins "
        f"(portal cell object: {PORTAL_CELLS[tumor]:,} cells) | YES |"
    )

text = f"""# SPATCH data and pairing audit

## Gate result

**Visium HD FFPE → CosMx 6K: PASS (3 pairs).** The official SPATCH study assigns Benchmark_01, Benchmark_02 and Benchmark_03 to HCC, ovarian cancer and colon adenocarcinoma, respectively. Visium HD FFPE and CosMx were generated from serial sections cut from the same FFPE block. The published cutting order is `Visium HD FFPE → CODEX → CosMx 6K → CODEX → Xenium 5K → CODEX`; therefore the assays are described as **same-specimen serial sections measured by different technologies**, not immediately adjacent sections and not cell-matched sections.

**Visium HD FFPE → Stereo-seq v1.3: STOPPED at pairing gate (0 valid pairs).** The official Methods state that each tumor was divided into separate portions: the FFPE block supplied Visium HD FFPE, CosMx and Xenium, whereas Stereo-seq used a different fresh-frozen OCT-embedded portion. The samples share patient/tumor identity, but the requested Visium-HD-FFPE→Stereo relationship is not a same-block serial-section relationship. This is a predefined specimen-identity/section-pairing exclusion, made before any model run or target outcome inspection. No Stereo-seq prediction was generated.

## Candidate-pair inventory

| specimen ID | tumor | source section | target section | relationship | source genes | target genes | shared genes | usable target units after QC | exact same specimen identity verified |
|---|---|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Coordinates, resolution and QC

- Source: official SPATCH Visium HD FFPE 8-µm output; platform-local spatial coordinates; only the official `high_quality=1` source tissue/QC flag is used.
- CosMx target: official global transcript pixel coordinates (`x_global_px`, `y_global_px`) at 0.12 µm per pixel. Calls were aggregated by `floor(pixel × 0.12 / 8)` and summed per target to produce 8-µm bins.
- Negative probes/codes are not genes in the Visium source and therefore are removed by the pre-known source/target gene-identity intersection.
- Tissue support follows the official SPATCH CosMx rule: stitched DAPI image, Otsu threshold, two dilations and one erosion with a 1,000-pixel square kernel, largest filled contour, and the published CosMx coordinate transform with a 4,256-pixel FOV offset.
- The common source/target ROI is not defined here from molecular data. It is fixed later from geometry alone using the same frozen Xenium rule: target bins within twice the median target-bin spacing of the seed-42 registered source support.
- No cell type, niche, tumor-region or pathological annotation is available to model fitting.

## Molecular information boundary

Target gene identities and coordinates are design metadata and may be used before prediction. All CosMx shared-gene counts were aggregated once into sealed containers. Before prediction lock, model code may materialize only the 16 source-selected anchor columns. Non-anchor target expression is forbidden until all three pair predictions and their SHA-256 hashes are recorded.

## Official sources

- SPATCH resource portal: `https://spatch.pku-genomics.org`
- Official code: `https://github.com/zenglab-pku/SPATCH`, frozen template checkout `c20108f60660b86bc2740607d84fd825beaaceca`
- Resource article: *Systematic benchmarking of high-throughput subcellular spatial transcriptomics platforms across human tumors*, Nature Communications (2025), DOI `10.1038/s41467-025-64292-3`.

This audit was finalized before any HyperSpatial-MAP source fit, target-anchor extraction, registered IDW prediction or MAP prediction in this extension.
"""
(ROOT / "00_DATA_PAIRING_AUDIT.md").write_text(text)
print(ROOT / "00_DATA_PAIRING_AUDIT.md")
