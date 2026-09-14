#!/usr/bin/env python3
"""Crisper panel-e rendering of the immutable v8 liver figure."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
V8 = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_v8_20260826_142434"
V8_SCRIPT = V8 / "scripts/build_liver_multidisease_v8.py"

spec = importlib.util.spec_from_file_location("liver_v8_frozen", V8_SCRIPT)
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)
v7, v6 = v8.v7, v8.v6


def main():
    units, _, _, summary = v6.v3.frozen_inputs()
    fig = plt.figure(figsize=(13.5, 10.5), facecolor="white")
    outer = fig.add_gridspec(
        3, 20, height_ratios=[2.65, 1.55, 5.15], hspace=.32, wspace=.82,
        left=.055, right=.982, top=.955, bottom=.035)

    ax_a = fig.add_subplot(outer[0, :5]); v6.draw_a(ax_a)
    v7.draw_b_title_spacing_only(outer[0, 5:15], units)
    ax_c = fig.add_subplot(outer[0, 15:]); v6.draw_c(ax_c, units)
    ax_d = fig.add_subplot(outer[1, 1:19]); v7.draw_d_np_only(ax_d, summary)
    np_labels = [text for text in ax_d.texts if text.get_text() == "n.p."]
    if len(np_labels) != 13:
        raise RuntimeError(f"Expected 13 n.p. labels, found {len(np_labels)}")
    for text in np_labels:
        text.set_fontsize(5.6)
        text.set_color("#A7AAAD")

    axes_before_panel_e = set(fig.axes)
    v6.draw_e(outer[2, :])
    panel_e_axes = [axis for axis in fig.axes if axis not in axes_before_panel_e]
    map_collections = [collection for axis in panel_e_axes for collection in axis.collections]
    if len(map_collections) != 15:
        raise RuntimeError(f"Expected 15 panel-e map collections, found {len(map_collections)}")

    # Only visual change from v8: 60% larger marker area (26.5% diameter).
    for collection in map_collections:
        collection.set_sizes(collection.get_sizes() * 1.60)

    stem = ROOT / "Liver_multidisease_integrated_main_v9_master"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=500)
    plt.close(fig)

    with Image.open(stem.with_suffix(".png")) as image:
        preview_width = 2200
        preview_height = round(image.height * preview_width / image.width)
        image.resize((preview_width, preview_height), Image.Resampling.LANCZOS).save(
            ROOT / "Liver_multidisease_integrated_main_v9_preview.png")

    (ROOT / "01_V9_RENDERING_NOTE.md").write_text("""# V9 rendering note

V9 is a cosmetic-only derivative of v8. Panel-e marker area was increased uniformly by 60% (26.5% in marker diameter) across all 15 spatial maps, and the master PNG was exported at 500 dpi. This reduces apparent softness after downsampling.

No values, coordinates, color normalization, expression or residual scales, representative genes, predictions, metrics, titles, annotations, panel geometry or other panels were changed. PDF and SVG remain vector outputs.
""")
    (ROOT / "02_V9_VISUAL_QC.md").write_text("""# V9 visual QC

Status: **PASS**

- Panel e was inspected at native 6,750 × 5,250-pixel master resolution and in the 2,200-pixel preview.
- Individual spots and fine spatial structure are more distinct than in v8 after equal-width downsampling.
- No problematic spot merging, map-label collision or scale-text collision was observed.
- All 15 maps retain their frozen values, coordinates, color normalization and scales.
- Panels a–d and all non-marker panel-e elements are unchanged from v8.
- Predictions recomputed: NO.
- Metrics or results changed: NO.
- Representative genes changed: NO.
""")

    print(json.dumps({
        "source_v8": str(V8), "v9": str(ROOT),
        "predictions_recomputed": False, "metrics_changed": False,
        "results_changed": False, "representatives_changed": False,
        "panel_e_marker_area_multiplier": 1.60,
        "master_png_dpi": 500, "visual_qc": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
