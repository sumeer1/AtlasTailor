#!/usr/bin/env python3
"""Match bundled workflow snapshots to identical files in a research checkout.

This release-maintainer utility is not needed by ordinary users. Matching is by
SHA-256, so it records provenance without rewriting executed scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


SUFFIXES = {".py", ".R", ".sh"}


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
    output = args.output or release_root / "reproducibility/provenance/EXECUTED_SCRIPT_PROVENANCE.tsv"

    bundled_root = release_root / "reproducibility/workflows"
    bundled = [path for path in bundled_root.rglob("*") if path.is_file() and path.suffix in SUFFIXES]
    wanted = {sha256(path) for path in bundled}

    matches: dict[str, list[Path]] = defaultdict(list)
    for path in research_root.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if release_root in path.parents or ".git" in path.parts:
            continue
        digest = sha256(path)
        if digest in wanted:
            matches[digest].append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["bundled_path", "sha256", "bytes", "identical_original_paths", "match_status"])
        for path in sorted(bundled):
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
    print(f"Wrote {len(bundled)} workflow records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
