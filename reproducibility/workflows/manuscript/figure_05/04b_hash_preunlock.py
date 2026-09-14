#!/usr/bin/env python3
"""Verify the global CosMx prediction lock and record pre-unlock hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv"
OUTPUT = ROOT / "02_SHA256_PRE_UNLOCK.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if not LOCK.exists() or OUTPUT.exists():
    raise RuntimeError("Expected one new global lock and no prior pre-unlock manifest")
records = pd.read_csv(LOCK, sep="\t")
if not (records.hidden_target_opened_before_hash == "NO").all():
    raise RuntimeError("Global lock contains a pre-hash hidden-target access record")
lines = []
for record in records.itertuples():
    path = Path(record.path)
    if not path.exists():
        raise RuntimeError(f"Verified artifact disappeared before manifest write: {path}")
    # 04_consolidate_prediction_lock.py has just recomputed this value from
    # every artifact.  Record that verified value without a redundant second
    # multi-gigabyte read before unlock.
    lines.append(f"{record.sha256}  {path}")
lines.append(f"{sha256(LOCK)}  {LOCK}")
OUTPUT.write_text("\n".join(lines) + "\n")
print(f"PRE-UNLOCK HASH PASS: {len(records)} frozen artifacts plus global lock")
