#!/usr/bin/env python3
"""Read-only integrity and policy checks for the public release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


MANUAL_MASTER_PATTERNS = (
    "Figure_*_FINAL.*",
    "Figure_2_Jesper_restructured*",
    "Figure_3_Jesper_restructured*",
    "Figure_5_Jesper_restructured*",
    "Figure_6_Jesper_restructured*",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = (args.report or root / "RELEASE_VALIDATION.json").resolve()

    checks: dict[str, object] = {}
    failures: list[str] = []

    manifest = root / "SHA256SUMS.tsv"
    checked = 0
    if not manifest.exists():
        failures.append("SHA256SUMS.tsv is missing")
    else:
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
    checks["manifest_entries_checked"] = checked

    required = [
        "README.md",
        "pyproject.toml",
        "environment.yml",
        "environment-manuscript.yml",
        "manuscript/FIGURE_PROVENANCE.tsv",
        "manuscript/SUPPLEMENTARY_PROVENANCE.tsv",
        "reproducibility/WORKFLOW_INDEX.tsv",
        "data/DATASETS.tsv",
    ]
    missing_required = [name for name in required if not (root / name).is_file()]
    failures.extend(f"required file missing: {name}" for name in missing_required)
    checks["required_files"] = {"count": len(required), "missing": missing_required}

    manual_masters: list[str] = []
    for pattern in MANUAL_MASTER_PATTERNS:
        manual_masters.extend(path.relative_to(root).as_posix() for path in root.rglob(pattern))
    if manual_masters:
        failures.append("manual final-assembly masters are present")
    checks["manual_final_assemblies"] = sorted(set(manual_masters))

    provenance_missing: list[str] = []
    figure_provenance = root / "manuscript/FIGURE_PROVENANCE.tsv"
    if figure_provenance.is_file():
        with figure_provenance.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                for field in ("generated_source_asset", "analysis_or_render_workflow"):
                    value = row[field]
                    if value.startswith(("Not bundled", "Not applicable")):
                        continue
                    for item in (part.strip() for part in value.split(";")):
                        if item and not (root / item).is_file():
                            provenance_missing.append(f"{row['figure']} {row['new_panel']}: {item}")
    if provenance_missing:
        failures.append("figure provenance references missing files")
    checks["figure_provenance_missing_files"] = provenance_missing

    identity_tables = [
        root / "manuscript/SOURCE_PANEL_PROVENANCE.tsv",
        root / "reproducibility/provenance/EXECUTED_SCRIPT_PROVENANCE.tsv",
    ]
    identity_failures: list[str] = []
    for table in identity_tables:
        if not table.is_file():
            identity_failures.append(f"missing identity table: {table.relative_to(root)}")
            continue
        with table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["match_status"] != "MATCHED_IDENTICAL":
                    identity_failures.append(row["bundled_path"])
    if identity_failures:
        failures.append("one or more copied artifacts lack a byte-identical source match")
    checks["byte_identical_source_failures"] = identity_failures

    source_svgs = list((root / "manuscript/source_panels").rglob("*.svg"))
    workflow_files = [
        path
        for path in (root / "reproducibility/workflows").rglob("*")
        if path.is_file() and path.suffix in {".py", ".R", ".sh"}
    ]
    checks["generated_source_svg_count"] = len(source_svgs)
    checks["archived_workflow_file_count"] = len(workflow_files)
    if not source_svgs:
        failures.append("no generated source SVGs found")
    if not workflow_files:
        failures.append("no archived workflows found")

    checks["status"] = "PASS" if not failures else "FAIL"
    checks["failures"] = failures
    report_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
