#!/usr/bin/env python3
"""One cosmetic-only n.p.-label refinement of the immutable v7 figure."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
V7 = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_v7_20260826_135955"
V7_SCRIPT = V7 / "scripts/build_liver_multidisease_v7.py"

spec = importlib.util.spec_from_file_location("liver_v7_frozen", V7_SCRIPT)
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
v6 = v7.v6


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

    # The only v8 edit: make the 13 n.p. labels smaller and lighter.
    np_labels = [text for text in ax_d.texts if text.get_text() == "n.p."]
    if len(np_labels) != 13:
        raise RuntimeError(f"Expected 13 n.p. labels, found {len(np_labels)}")
    for text in np_labels:
        text.set_fontsize(5.6)       # 20% reduction from v7's 7.0 pt
        text.set_color("#A7AAAD")   # lighter neutral gray; numeric labels untouched

    v6.draw_e(outer[2, :])

    stem = ROOT / "Liver_multidisease_integrated_main_v8_master"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=350)
    plt.close(fig)

    print(json.dumps({
        "source_v7": str(V7), "v8": str(ROOT),
        "predictions_recomputed": False, "metrics_changed": False,
        "representatives_changed": False,
        "numerical_heatmap_values_changed": False,
        "np_label_count": 13, "np_font_size_pt": 5.6,
        "np_color": "#A7AAAD", "visual_qc": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
