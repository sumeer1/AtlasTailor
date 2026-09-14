#!/usr/bin/env python3
"""Enlarge compact Figure 2 panel d through layout space only.

All panel values, labels, scales, point clouds and frozen inputs are delegated
to the existing Figure 2 builder.  This revision gives panel d a wider cell,
compresses its title/legend header and fixes the metric-axis box aspect so the
plots enlarge without geometric stretching.  Vector-container point clouds
retain the existing 600-dpi render fix.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "build_figure2_external_revision.py"
spec = importlib.util.spec_from_file_location("figure2_existing", SOURCE)
figure2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(figure2)


def high_resolution_export(fig, stem: str, png: bool = True) -> None:
    exts = ("pdf", "svg", "png") if png else ("pdf", "svg")
    for ext in exts:
        dpi = 300 if ext == "png" else 600
        fig.savefig(HERE / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight",
                    pad_inches=.10, facecolor="white")
    plt.close(fig)


def main() -> None:
    figure2.apply_style()
    values = figure2.load_external_values()
    wnt = figure2.read_wnt11f2_maps()
    figure2.export = high_resolution_export
    figure2.compact_main_figure(
        values,
        wnt,
        stem="Figure_2_revised_compact_with_registered_IDW_panel_d_enlarged_render_fixed",
        roomy_panel_d=True,
    )


if __name__ == "__main__":
    main()
