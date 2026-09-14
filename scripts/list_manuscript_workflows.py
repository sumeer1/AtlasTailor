#!/usr/bin/env python3
"""Print the manuscript workflow index grouped by figure."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (root / "reproducibility/WORKFLOW_INDEX.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            groups[row["scope"]].append(row)
    for scope, rows in groups.items():
        print(f"\n{scope}")
        for row in rows:
            print(f"  - {row['workflow']}: {row['entry_point']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
