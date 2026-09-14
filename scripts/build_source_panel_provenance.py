#!/usr/bin/env python3
"""Match bundled generated SVGs to byte-identical research artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    research_root = args.research_root.resolve()
    release_root = args.release_root.resolve()
    output = args.output or release_root / "manuscript/SOURCE_PANEL_PROVENANCE.tsv"

    bundled = sorted((release_root / "manuscript/source_panels").rglob("*.svg"))
    wanted = {sha256(path) for path in bundled}
    matches: dict[str, list[Path]] = defaultdict(list)
    for path in research_root.rglob("*.svg"):
        if release_root in path.parents or ".git" in path.parts:
            continue
        digest = sha256(path)
        if digest in wanted:
            matches[digest].append(path)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["bundled_path", "sha256", "bytes", "identical_original_paths", "match_status"])
        for path in bundled:
            digest = sha256(path)
            originals = sorted(p.relative_to(research_root).as_posix() for p in matches[digest])
            writer.writerow(
                [
                    path.relative_to(release_root).as_posix(),
                    digest,
                    path.stat().st_size,
                    ";".join(originals),
                    "MATCHED_IDENTICAL" if originals else "NO_IDENTICAL_SOURCE_FOUND",
                ]
            )
    print(f"Wrote {len(bundled)} source-panel records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
