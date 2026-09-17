#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import nbformat
from nbclient import NotebookClient

p = ArgumentParser()
p.add_argument("--smoke", action="store_true")
p.add_argument(
    "--external-data",
    action="store_true",
    help="execute notebooks that require separately downloaded public datasets",
)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
paths = sorted((root / "notebooks").glob("*.ipynb"))
if args.smoke:
    # Exercise one bundled frozen-results notebook; external-data tutorials are
    # syntax-checked by the full notebook validation and cannot download data in CI.
    paths = [path for path in paths if path.name.startswith("10_")][:1]
for path in paths:
    nb = nbformat.read(path, as_version=4)
    requires_external = nb.get("metadata", {}).get("atlastailor", {}).get("requires_external_data", False)
    if requires_external and not args.external_data:
        for cell in nb.cells:
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell", "exec")
        print(f"PASS-SYNTAX {path.name} (external data required; execution skipped)")
        continue
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
    print(f"PASS {path.name}")
