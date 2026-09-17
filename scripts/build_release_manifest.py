#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for the release tree.

The manifest deliberately excludes itself, validation reports, VCS metadata,
and disposable Python/build caches. It never edits scientific artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "build", "dist", "site"}
EXCLUDED_NAMES = {"SHA256SUMS.tsv", "RELEASE_VALIDATION.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    return path.name not in EXCLUDED_NAMES and not any(part in EXCLUDED_PARTS for part in path.parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = (args.output or root / "provenance" / "SHA256SUMS.tsv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in root.rglob("*") if path.is_file() and included(path.relative_to(root)))

    lines = ["sha256\tbytes\tpath"]
    for path in files:
        lines.append(f"{sha256(path)}\t{path.stat().st_size}\t{path.relative_to(root).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
