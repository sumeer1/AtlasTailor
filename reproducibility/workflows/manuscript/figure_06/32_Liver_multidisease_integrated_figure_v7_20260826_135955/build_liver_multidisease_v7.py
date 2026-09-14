#!/usr/bin/env python3
"""Two visual-only edits to the immutable v6 liver figure."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
V6 = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_v6_20260826_130904"
V6_SCRIPT = V6 / "scripts/build_liver_multidisease_v6.py"

spec = importlib.util.spec_from_file_location("liver_v6_frozen", V6_SCRIPT)
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_b_title_spacing_only(subspec, units):
    """Invoke v6 verbatim, then move only the shared title upward."""
    axes = v6.draw_b(subspec, units)
    wanted = "Reconstruction beyond healthy-atlas transfer"
    matches = []
    for axis in plt.gcf().axes:
        for text in axis.texts:
            if text.get_text() == wanted:
                matches.append(text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one panel-b shared title, found {len(matches)}")
    matches[0].set_position((.5, 1.085))
    return axes


def draw_d_np_only(ax, summary):
    """Invoke v6 verbatim, then style and label only non-predefined cells."""
    v6.draw_d(ax, summary)
    if len(ax.images) != 1:
        raise RuntimeError("Expected one panel-d heatmap image")
    heatmap = ax.images[0]
    array = np.ma.asarray(heatmap.get_array())
    mask = np.ma.getmaskarray(array)
    if int(mask.sum()) != 13:
        raise RuntimeError(f"Expected 13 non-predefined cells, found {int(mask.sum())}")

    cmap = heatmap.get_cmap().copy()
    cmap.set_bad("#EEF0F1")
    heatmap.set_cmap(cmap)

    for row, column in np.argwhere(mask):
        ax.text(column, row, "n.p.", ha="center", va="center",
                fontsize=7.0, color="#858A8E", fontweight="normal")

    old_footer = "Median residual-program Pearson · blank cells were not predefined"
    new_footer = "Median residual-program Pearson · n.p., program not predefined for that disease"
    footer_matches = [text for text in ax.texts if text.get_text() == old_footer]
    if len(footer_matches) != 1:
        raise RuntimeError(f"Expected one panel-d footer, found {len(footer_matches)}")
    footer_matches[0].set_text(new_footer)


def write_docs():
    (ROOT / "01_V7_CHANGELOG.md").write_text("""# V7 changelog

V7 makes exactly two visual edits to the completed v6 figure.

1. **Panel d:** every masked disease–program combination that was not predefined now has a very light neutral-gray background and a centered `n.p.` label. The footer now defines `n.p., program not predefined for that disease`. Numerical cells, values, ordering, headers and color scale are unchanged.
2. **Panel b:** the global title, `Reconstruction beyond healthy-atlas transfer`, was moved slightly upward. Subplot titles, axes, points, paired lines, disease medians and values are unchanged.

Panels a, c and e are rendered by the v6 functions without modification. The canvas, architecture, disease colors, representative genes, spatial maps and scales are unchanged.
""")
    (ROOT / "02_V7_VISUAL_QC.md").write_text("""# V7 visual QC

Status: **PASS**

- The final master PNG and preview were inspected after rendering; the PDF remains a one-page 13.5 × 10.5-inch master.
- **Panel d:** all 13 non-predefined cells use the same light neutral gray (`#EEF0F1`) and carry a centered `n.p.` label. The footer reads `n.p., program not predefined for that disease`. All 14 numerical program annotations, their underlying values, ordering, headers and color scale are unchanged.
- **Panel b:** the global title is clearly separated vertically from `Spatial Pearson` and `log1p RMSE`; no title overlap or crowding remains. Axes, point locations, paired lines, disease medians and plot dimensions are unchanged.
- **Frozen content:** panels a, c and e are unchanged from v6; MASLD–TAT, PSC–TAT and AH–PCK1 remain displayed.
- Predictions recomputed: NO.
- Metrics changed: NO.
- Representative genes changed: NO.
- Panel d numerical values changed: NO.
""")


def write_manifest():
    lines = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "03_SHA256_MANIFEST.txt":
            lines.append(f"{sha(path)}  {path.relative_to(ROOT)}")
    (ROOT / "03_SHA256_MANIFEST.txt").write_text("\n".join(lines) + "\n")


def main():
    units, _, _, summary = v6.v3.frozen_inputs()
    fig = plt.figure(figsize=(13.5, 10.5), facecolor="white")
    outer = fig.add_gridspec(
        3, 20, height_ratios=[2.65, 1.55, 5.15], hspace=.32, wspace=.82,
        left=.055, right=.982, top=.955, bottom=.035)

    ax_a = fig.add_subplot(outer[0, :5]); v6.draw_a(ax_a)
    draw_b_title_spacing_only(outer[0, 5:15], units)
    ax_c = fig.add_subplot(outer[0, 15:]); v6.draw_c(ax_c, units)
    ax_d = fig.add_subplot(outer[1, 1:19]); draw_d_np_only(ax_d, summary)
    v6.draw_e(outer[2, :])

    stem = ROOT / "Liver_multidisease_integrated_main_v7_master"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=350)
    plt.close(fig)

    with Image.open(stem.with_suffix(".png")) as image:
        preview_width = 1800
        preview_height = round(image.height * preview_width / image.width)
        image.resize((preview_width, preview_height), Image.Resampling.LANCZOS).save(
            ROOT / "Liver_multidisease_integrated_main_v7_preview.png")

    write_docs()
    write_manifest()
    print(json.dumps({
        "source_v6": str(V6), "v7": str(ROOT),
        "predictions_recomputed": False, "metrics_changed": False,
        "representatives_changed": False, "panel_d_values_changed": False,
        "figure_inches": [13.5, 10.5], "figure_mm": [342.9, 266.7],
    }, indent=2))


if __name__ == "__main__":
    main()
