#!/usr/bin/env python3
"""Re-export Figure 2 with high-resolution rasterized point clouds in PDF/SVG.

Scientific data, map scale, point geometry and layout are delegated unchanged to
the existing frozen Figure 2 plotting module.  Only the rasterization resolution
used inside vector containers is raised from Matplotlib's default 100 dpi to
600 dpi.
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
    """Match the existing export, using 600 dpi for rasterized vector artists."""
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

    # The existing builders call their module-level exporter. Replacing only
    # that callback preserves the complete figure construction verbatim.
    figure2.export = high_resolution_export
    figure2.compact_main_figure(
        values, wnt,
        stem="Figure_2_revised_compact_with_registered_IDW_render_fixed",
    )

    # Match the existing compact standalone panel-e width.
    fig = plt.figure(figsize=(3.20, 1.85))
    figure2.draw_wnt11f2(
        fig, wnt, standalone=True, include_idw=True,
    )
    high_resolution_export(
        fig, "Fig2e_wnt11f2_with_registered_IDW_render_fixed", png=True,
    )


if __name__ == "__main__":
    main()
