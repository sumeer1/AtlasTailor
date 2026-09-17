#!/usr/bin/env python3
"""Read-only integrity and scope checks for the software repository."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    required = [
        "README.md",
        "docs/repository_scope.md",
        "pyproject.toml",
        "environment.yml",
        "src/hyperspatial/__init__.py",
        "tests/fixtures/minimal_2d/config.yaml",
        "notebooks/00_run_atlastailor_dlpfc.ipynb",
        "notebooks/01_run_atlastailor_zebrafish.ipynb",
        "notebooks/04_run_atlastailor_on_your_data.ipynb",
        "notebooks/10_figure2_benchmark_results.ipynb",
        "notebooks/11_cross_platform_benchmark_results.ipynb",
        "notebooks/12_liver_disease_validation_results.ipynb",
        "docs/assets/Figure_1.pdf",
        "docs/assets/Figure_1_preview.png",
        "provenance/SHA256SUMS.tsv",
    ]
    for name in required:
        if not (root / name).is_file():
            failures.append(f"required file missing: {name}")

    for excluded in ("manuscript", "reproducibility", "environment-manuscript.yml"):
        if (root / excluded).exists():
            failures.append(f"manuscript-scale content present in software repository: {excluded}")

    checked = 0
    manifest = root / "provenance/SHA256SUMS.tsv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                path = root / row["path"]
                checked += 1
                if not path.is_file():
                    failures.append(f"missing: {row['path']}")
                elif path.stat().st_size != int(row["bytes"]):
                    failures.append(f"size mismatch: {row['path']}")
                elif sha256(path) != row["sha256"]:
                    failures.append(f"sha256 mismatch: {row['path']}")

    report = {
        "repository_role": "software",
        "manifest_entries_checked": checked,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    report_path = root / "provenance/RELEASE_VALIDATION.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
