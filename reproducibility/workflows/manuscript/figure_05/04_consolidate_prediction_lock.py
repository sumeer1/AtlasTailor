#!/usr/bin/env python3
"""Verify all frozen prediction artifacts and write the global lock table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "analysis" / "hyperspatial_map_2d_dlpfc"))
from dlpfc_common import sha256  # noqa: E402

records = []
for tumor in ("COAD", "HCC", "OV"):
    path = ROOT / "work" / tumor / "prediction_lock_manifest.json"
    if not path.exists():
        raise RuntimeError(f"Missing pair lock: {path}")
    manifest = json.loads(path.read_text())
    if "HIDDEN_TARGET_EXPRESSION_STILL_UNOPENED" not in manifest["status"]:
        raise RuntimeError(f"Unexpected lock status for {tumor}")
    artifacts = {
        "source_model": (manifest["source_model"], manifest["source_model_sha256"]),
        "registration": (manifest["registration"], manifest["registration_sha256"]),
        "rigid_registration": (manifest["rigid_registration"], manifest["rigid_registration_sha256"]),
        "geometry_only_roi": (manifest["geometry_only_roi"], manifest["geometry_only_roi_sha256"]),
        "target_anchor_only_object": (manifest["anchor_only_object"], manifest["anchor_only_object_sha256"]),
        "source_model_predicted_target_depth": (manifest["predicted_target_depth"], manifest["predicted_target_depth_sha256"]),
    }
    for method, item in manifest["predictions"].items():
        artifacts[f"prediction:{method}"] = (item["path"], item["sha256"])
    for artifact, (filename, expected) in artifacts.items():
        actual = sha256(Path(filename))
        if actual != expected:
            raise RuntimeError(f"Hash mismatch: {tumor} {artifact}")
        records.append({
            "pair_id": manifest["pair_id"], "tumor": tumor,
            "artifact": artifact, "path": filename, "sha256": actual,
            "hidden_gene_count": manifest["hidden_gene_count"],
            "target_bin_count": manifest["target_roi_bin_count"],
            "hidden_target_opened_before_hash": "NO",
        })

output = ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv"
if output.exists():
    raise RuntimeError(f"Refusing to overwrite {output}")
pd.DataFrame(records).to_csv(output, sep="\t", index=False)
print(f"LOCKED\t{len(records)} artifacts\t{output}")
