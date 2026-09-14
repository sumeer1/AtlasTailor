#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import nbformat
from nbclient import NotebookClient

p = ArgumentParser(); p.add_argument("--smoke", action="store_true"); args = p.parse_args()
root = Path(__file__).resolve().parents[1]
paths = sorted((root / "notebooks").glob("*.ipynb"))
if args.smoke: paths = paths[:1]
for path in paths:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
    print(f"PASS {path.name}")

