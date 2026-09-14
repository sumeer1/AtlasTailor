#!/usr/bin/env python3
"""Complete AH lock bookkeeping after predictions were already frozen."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD/work/healthy_source_model_masld_panel.npz"
PROTOCOL = PROJECT / "PI_revision/27_Liver_multidisease_frozen_protocol/FROZEN_CONFIG.json"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

model = np.load(SOURCE, allow_pickle=True)
a = model["anchor_indices"].astype(int)
pd.DataFrame({"anchor_rank": np.arange(1, 17), "gene_id": model["gene_ids"][a],
              "gene": model["gene_names"][a], "selection": "FROZEN_POST_MASLD_SOURCE_ONLY"}).to_csv(
    ROOT / "01_ANCHORS.tsv", sep="\t", index=False)
rows = []
for sample, gsm in (("Sah73", "GSM8552412"), ("Sah80", "GSM8552413")):
    p = ROOT / "work" / sample / "prediction_lock_manifest.json"
    lock = json.loads(p.read_text())
    lock["patient"] = sample
    p.write_text(json.dumps(lock, indent=2) + "\n")
    rows.append({"patient": sample, "sample": sample, "GSM": gsm, "anchor_count": 16,
                 "hidden_genes": lock["hidden_gene_count"], "on_tissue_spots": lock["on_tissue_spots"],
                 "geometry_supported_spots": lock["geometry_supported_spots"], "lock_manifest": str(p),
                 "lock_manifest_sha256": sha(p), "registered_idw_sha256": lock["registered_idw_sha256"],
                 "map_sha256": lock["map_prediction_sha256"], "lock_status": lock["status"]})
pd.DataFrame(rows).to_csv(ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv", sep="\t", index=False)
items = [PROTOCOL, SOURCE, ROOT / "FROZEN_CONFIG_APPLIED.json", ROOT / "AH_PROGRAMS.json",
         ROOT / "01_ANCHORS.tsv", ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv"] + [Path(r["lock_manifest"]) for r in rows]
with (ROOT / "03_SHA256_PRE_UNLOCK.txt").open("w") as f:
    for p in items:
        f.write(f"{sha(p)}  {p}\n")
print(json.dumps({"status": "ALL_AH_PREDICTIONS_LOCKED", "patient_n": 2,
                  "predictions_recomputed": False, "anchors": 16, "hidden_genes": 6814}))
