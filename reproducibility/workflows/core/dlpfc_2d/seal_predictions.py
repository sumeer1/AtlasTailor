#!/usr/bin/env python3
"""Validate all 12 prediction manifests and create the global hash barrier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlpfc_common import METHODS, SEEDS, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "PREDICTION_HASHES.tsv"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    lock = json.loads(args.lock.read_text())
    rows = []
    for direction in lock["directions"]:
        manifest_path = args.root / "04_predictions" / direction / "prediction_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        expected_status = "ALL_PREDICTIONS_SERIALIZED_AND_HASHED_EVALUATION_FORBIDDEN_UNTIL_GLOBAL_HASH_TABLE"
        if manifest["status"] != expected_status:
            raise RuntimeError(f"{direction}: incomplete prediction status")
        for seed in SEEDS:
            predictions = manifest["methods"][str(seed)]["predictions"]
            if set(predictions) != set(METHODS):
                raise RuntimeError(f"{direction} seed {seed}: method set changed")
            for method in METHODS:
                metadata = predictions[method]
                path = Path(metadata["path"])
                observed = sha256(path)
                if observed != metadata["sha256"]:
                    raise RuntimeError(f"{path}: manifest hash mismatch")
                rows.append((direction, manifest["source"], manifest["target"],
                             manifest["donor"], manifest["adjacent_pair"], seed,
                             method, str(path), observed, metadata["shape"][0], metadata["shape"][1]))
    if len(rows) != 216:
        raise RuntimeError(f"Expected 216 prediction records, found {len(rows)}")
    with output.open("x") as handle:
        handle.write("direction\tsource_section\ttarget_section\tdonor\tadjacent_pair\tseed\tmethod\tpath\tsha256\tn_spots\tn_genes\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")
    marker = args.root / "04_predictions" / "ALL_12_DIRECTIONS_SEALED.marker"
    marker.write_text(
        "216 molecular predictions (12 directions x 3 seeds x 6 methods) validated and globally hashed before target non-anchor evaluation.\n"
    )
    print(json.dumps({"status": "SEALED", "directions": 12, "records": len(rows), "path": str(output)}))


if __name__ == "__main__":
    main()
