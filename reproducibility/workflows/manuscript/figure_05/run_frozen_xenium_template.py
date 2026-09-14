#!/usr/bin/env python3
"""Execute an unmodified frozen Xenium pipeline stage for CosMx paths/labels.

The source text is loaded from the authoritative completed run.  Only platform
path and label tokens are changed; the HyperSpatial-MAP computations and fixed
parameters are not rewritten here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


HERE = Path(__file__).resolve()
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_spatch24")
ROOT = HERE.parents[1]
PROJECT = ROOT.parents[1]
TEMPLATE_ROOT = PROJECT / "PI_revision" / "23_SPATCH_cross_platform_MAP_20260824_111004"
STAGES = {
    "02": "02_freeze_source_and_geometry.py",
    "03": "03_extract_anchors_predict_and_lock.py",
    "04": "04_consolidate_prediction_lock.py",
    "05": "05_open_hidden_and_evaluate.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("stage", choices=STAGES)
    known, remaining = parser.parse_known_args()
    source_path = TEMPLATE_ROOT / "scripts" / STAGES[known.stage]
    source = source_path.read_text()
    source = source.replace("Xenium 5K", "CosMx 6K")
    source = source.replace("Xenium5K", "CosMx6K")
    source = source.replace("Xenium_", "CosMx_")
    source = source.replace("Xenium ", "CosMx ")
    source = source.replace("Xenium", "CosMx")
    namespace = {
        "__name__": "__main__",
        "__file__": str(HERE),
        "__package__": None,
    }
    import sys
    sys.argv = [str(HERE), *remaining]
    exec(compile(source, str(source_path), "exec"), namespace)


if __name__ == "__main__":
    main()
