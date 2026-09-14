#!/usr/bin/env python3
"""Write the final integrity manifest and compact terminal summary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "12_SHA256_FINAL.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


required = [
    ROOT / "00_DATA_PAIRING_AUDIT.md", ROOT / "01_ANCHORS.tsv",
    ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv", ROOT / "02_SHA256_PRE_UNLOCK.txt",
    ROOT / "03_PAIR_LEVEL_RESULTS.tsv", ROOT / "04_GENE_LEVEL_RESULTS.tsv",
    ROOT / "05_REGISTRATION_RESULTS.tsv", ROOT / "06_LEAKAGE_AUDIT.md",
    ROOT / "08_RESULTS_INTERPRETATION.md", ROOT / "09_FINAL_DECISION.md",
    ROOT / "10_MANUSCRIPT_PARAGRAPH.md", ROOT / "11_FIGURE_CAPTIONS.md",
    ROOT / "07_FIGURES" / "SPATCH_VisiumHD_to_CosMx_IDW_vs_MAP.pdf",
    ROOT / "07_FIGURES" / "SPATCH_VisiumHD_to_CosMx_IDW_vs_MAP.png",
    ROOT / "07_FIGURES" / "SPATCH_VisiumHD_to_CosMx_IDW_vs_MAP.svg",
]
required.extend(sorted((ROOT / "SEALED_TARGET_8UM").glob("*preprocessing.json")))
required.extend(sorted((ROOT / "scripts").glob("*.py")))
missing = [path for path in required if not path.exists()]
if missing:
    raise RuntimeError("Missing final files: " + ", ".join(map(str, missing)))
OUTPUT.write_text("\n".join(f"{sha256(path)}  {path}" for path in required) + "\n")

pair = pd.read_csv(ROOT / "03_PAIR_LEVEL_RESULTS.tsv", sep="\t")
decision_text = (ROOT / "09_FINAL_DECISION.md").read_text()
cosmx_decision = next(letter for letter in "ABC" if f"Decision {letter}" in decision_text)
lines = [
    "SPATCH COSMX/STEREO EXTENSION — FINAL SUMMARY",
    f"output_directory: {ROOT}",
    "valid_VisiumHD_FFPE_to_CosMx_pairs: 3 (COAD, HCC, OV)",
    "valid_VisiumHD_FFPE_to_Stereo_seq_pairs: 0 (pairing gate stopped: separate OCT portion)",
    "predictions_locked_before_hidden_target_evaluation: YES",
]
for row in pair.itertuples():
    lines.append(
        f"{row.tumor}: hidden={row.hidden_gene_count}; target_8um_bins={row.target_units_8um_bins}; "
        f"spatial IDW={row.IDW_median_gene_wise_spatial_pearson:.6f}, MAP={row.MAP_median_gene_wise_spatial_pearson:.6f}; "
        f"RMSE IDW={row.IDW_log1p_RMSE:.6f}, MAP={row.MAP_log1p_RMSE:.6f}; "
        f"MAP residual median={row.MAP_median_gene_wise_residual_pearson:.6f}, "
        f"pooled={row.MAP_pooled_residual_pearson:.6f}"
    )
lines.extend([
    f"CosMx_decision: {cosmx_decision}",
    "Stereo_seq_decision: NOT ASSIGNED — predefined pairing gate STOP",
    "leakage_audit: PASS",
    "manuscript_facing_comparator: registered 12-NN IDW only",
    "model_retuned_or_extended: NO",
    "previous_Xenium_experiment_modified: NO",
])
summary = "\n".join(lines) + "\n"
(ROOT / "TERMINAL_SUMMARY.txt").write_text(summary)
print(summary)
