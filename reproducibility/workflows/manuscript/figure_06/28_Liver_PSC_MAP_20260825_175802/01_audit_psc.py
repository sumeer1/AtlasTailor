#!/usr/bin/env python3
"""Audit GSE245620 using assay/geometry metadata only; do not open expression values."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
DATA = ROOT / "data" / "GSE245620"
FROZEN = PROJECT / "PI_revision" / "27_Liver_multidisease_frozen_protocol"
SOURCE_RUN = PROJECT / "PI_revision" / "26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
SOURCE_MODEL = SOURCE_RUN / "work" / "healthy_source_model_masld_panel.npz"
SAMPLES = {
    "PSC011_A1": ("GSM7845914", "PSC011_A1_VISIUM"),
    "PSC011_B1": ("GSM7845915", "PSC011_B1_VISIUM"),
    "PSC011_C1": ("GSM7845916", "PSC011_C1_VISIUM"),
    "PSC011_D1": ("GSM7845917", "PSC011_D1_VISIUM"),
}
EXPECTED_ANCHORS = [
    "IGFBP1", "VIM", "C7", "ORM1", "NNMT", "MT1M", "FOS", "CYP1A2",
    "MYL9", "JUND", "IGKC", "SERPINE1", "CD74", "GSTA2", "ADIRF", "SDS",
]
EXPECTED_SOURCE_SHA = "28c9c83f77b3c95c285b63e562e6b8d38e44523329976b8d3fefa788d1c130d5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paths(gsm: str, stem: str):
    prefix = DATA / f"{gsm}_{stem}"
    return {
        "barcodes": Path(str(prefix) + "_barcodes.tsv.gz"),
        "features": Path(str(prefix) + "_features.tsv.gz"),
        "matrix": Path(str(prefix) + "_matrix.mtx.gz"),
        "positions": Path(str(prefix) + "_tissue_positions_list.csv.gz"),
    }


def matrix_header(path: Path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.startswith("%"):
                return tuple(map(int, line.split()))
    raise RuntimeError(f"No Matrix Market dimensions in {path}")


def main():
    protocol = json.loads((FROZEN / "FROZEN_CONFIG.json").read_text())
    if protocol["status"] != "FROZEN_NO_POST_OUTCOME_CHANGES_PERMITTED":
        raise RuntimeError("Frozen protocol status invalid")
    if sha256(SOURCE_MODEL) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Frozen source model hash mismatch")
    model = np.load(SOURCE_MODEL, allow_pickle=True)
    source_ids = model["gene_ids"].astype(str)
    source_names = model["gene_names"].astype(str)
    anchor_idx = model["anchor_indices"].astype(int)
    eval_idx = model["evaluation_indices"].astype(int)
    if source_names[anchor_idx].tolist() != EXPECTED_ANCHORS:
        raise RuntimeError("Frozen model anchor identities differ from protocol")

    audit_rows, download_rows = [], []
    for sample, (gsm, stem) in SAMPLES.items():
        p = paths(gsm, stem)
        if not all(x.exists() for x in p.values()):
            raise FileNotFoundError(f"Missing processed files for {sample}")
        features = pd.read_csv(p["features"], sep="\t", header=None,
                               names=["gene_id", "gene", "feature_type"])
        with gzip.open(p["barcodes"], "rt") as handle:
            barcodes = [line.strip() for line in handle if line.strip()]
        positions = pd.read_csv(p["positions"], header=None,
                                names=["barcode", "in_tissue", "array_row", "array_col", "y", "x"])
        dims = matrix_header(p["matrix"])
        if dims[:2] != (len(features), len(barcodes)):
            raise RuntimeError(f"{sample}: matrix dimensions disagree with metadata")
        if set(barcodes) != set(positions["barcode"]):
            raise RuntimeError(f"{sample}: barcode/position sets differ")
        target_ids = set(features["gene_id"])
        anchors_present = [g for g in source_ids[anchor_idx] if g in target_ids]
        hidden_present = [g for g in source_ids[eval_idx] if g in target_ids]
        if len(anchors_present) != 16:
            raise RuntimeError(f"{sample}: only {len(anchors_present)}/16 frozen anchors present")
        if len(hidden_present) != len(eval_idx):
            raise RuntimeError(f"{sample}: frozen hidden axis incomplete ({len(hidden_present)}/{len(eval_idx)})")
        audit_rows.append({
            "patient": "PSC011", "sample": sample, "GSM": gsm,
            "diagnosis": "primary sclerosing cholangitis", "platform": "10x Genomics Visium",
            "tissue_preparation": "OCT-frozen, 16-um cryosection",
            "matrix_genes": len(features), "matrix_barcodes": len(barcodes),
            "on_tissue_spots": int(positions["in_tissue"].eq(1).sum()),
            "matrix_nonzeros_metadata": dims[2], "frozen_source_genes_present": len(set(source_ids) & target_ids),
            "frozen_anchors_present": len(anchors_present), "frozen_hidden_genes_locked": len(hidden_present),
            "coordinates_available": True, "full_target_expression_available": True,
            "eligible_pre_outcome": True,
        })
        for kind, path in p.items():
            download_rows.append({"sample": sample, "GSM": gsm, "file_role": kind,
                                  "path": str(path), "bytes": path.stat().st_size,
                                  "sha256": sha256(path),
                                  "source": f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7845nnn/{gsm}/suppl/{path.name}"})

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(ROOT / "00_TARGET_METADATA.tsv", sep="\t", index=False)
    pd.DataFrame(download_rows).to_csv(ROOT / "00_DOWNLOAD_MANIFEST.tsv", sep="\t", index=False)
    anchor_table = pd.DataFrame({
        "anchor_order": np.arange(1, 17), "gene_id": source_ids[anchor_idx],
        "gene": source_names[anchor_idx], "selection_information": "frozen_post_MASLD_source_only_panel",
    })
    anchor_table.to_csv(ROOT / "01_ANCHORS.tsv", sep="\t", index=False)
    section_lines = "\n".join(
        f"- `{r.sample}` / `{r.GSM}`: {r.on_tissue_spots} on-tissue spots; "
        f"{r.matrix_genes} assayed genes; 16/16 frozen anchors; {r.frozen_hidden_genes_locked} locked hidden genes."
        for r in audit.itertuples()
    )
    text = f"""# PSC data and eligibility audit

## Authoritative sources

- Study: *Single-cell, single-nucleus, and spatial transcriptomics characterization of the immunological landscape in the healthy and PSC human liver*.
- Publication: Journal of Hepatology 80, 730–743 (2024); DOI `10.1016/j.jhep.2023.12.023`; PMID `38199298`.
- Primary PSC spatial series: GEO `GSE245620`, BioProject `PRJNA1029172`.
- Optional healthy same-study series: GEO `GSE240429`, not used as the primary reference.
- Primary reference: healthy live-donor atlas M1–M8, Zenodo `10.5281/zenodo.17735506`.

## Biological structure

GSE245620 contains four Visium sections (`A1`–`D1`) from the single PSC donor/sample identifier `PSC011`. The four sections are evaluated separately for transparency and aggregated once at the patient level. They are not treated as four independent patients; independent patient N is 1.

{section_lines}

All four targets passed the predefined technical eligibility checks before outcome evaluation. No performance-based exclusion was possible or used.

## Information boundary

This audit inspected filenames, file hashes, feature identifiers, barcodes, matrix dimensions, tissue flags, and coordinates only. It did not materialize any target expression value. The exact post-MASLD 16-gene panel is present in every section. All {len(eval_idx)} genes on the frozen hidden axis are present in GSE245620 and remain sealed until prediction hashes verify.

## Same-study sensitivity decision

GSE240429 contains four healthy C73 Visium sections and is scientifically relevant as a platform/study control. It is not run as a replacement reference in this exact-protocol analysis because doing so would require refitting or changing the post-MASLD hashed M1–M8 source model. That would violate `PI_revision/27_Liver_multidisease_frozen_protocol/FROZEN_CONFIG.json`. This branch is therefore documented but not performed; the primary PSC result remains the exact frozen M1–M8-reference evaluation.

## Compartment metadata

The GEO deposit provides raw count matrices, spot coordinates, and images but no deposited spot-level portal/periportal or scar annotation table. Homologous-compartment metrics will be reported only if a source-independent annotation can be verified; no target-expression-derived compartment will be invented for fitting or used to optimize registration.
"""
    (ROOT / "00_DATA_AUDIT.md").write_text(text)
    print(json.dumps({"status": "PASS", "sections": len(audit), "patient_n": 1,
                      "anchors": 16, "hidden_genes": int(len(eval_idx))}, indent=2))


if __name__ == "__main__":
    main()

