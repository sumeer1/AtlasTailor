#!/usr/bin/env python3
"""Render fixed Figure 3 PDF/SVG copies without changing scientific content.

The original mixed raster/vector exports combined constrained layout with
dynamically located inset axes.  Vector backends consequently evaluated the
scatter-image and vector-annotation positions at different layout states.  This
renderer executes the authoritative figure builder in memory, freezes every
resolved axes position, and writes new, explicitly named exports.  Existing
Figure 3 files are never overwritten.
"""
from __future__ import annotations

import runpy
import tempfile
from pathlib import Path

import matplotlib.figure


HERE = Path(__file__).resolve().parent
ORIGINAL_SAVEFIG = matplotlib.figure.Figure.savefig
ORIGINAL_CLOSE = None
CAPTURED = []


def capture_savefig(self, fname, *args, **kwargs):
    """Capture the completed figure instead of writing original filenames."""
    if self not in CAPTURED:
        CAPTURED.append(self)


def keep_open(*args, **kwargs):
    """Prevent the authoritative builder from closing the captured figure."""


def main() -> None:
    import matplotlib.pyplot as plt

    global ORIGINAL_CLOSE
    ORIGINAL_CLOSE = plt.close
    matplotlib.figure.Figure.savefig = capture_savefig
    plt.close = keep_open
    try:
        namespace = runpy.run_path(str(HERE / "generate_figure3_corrected.py"))
        # Redirect the authoritative builder's ancillary report/source-table
        # writes to a temporary directory. Only the two new vector exports
        # below are permitted to persist in the frozen figure folder.
        with tempfile.TemporaryDirectory(prefix="figure3-render-") as temporary:
            temporary_root = Path(temporary)
            namespace["OUT"] = temporary_root
            namespace["FIGURE_STEM"] = temporary_root / "captured"
            namespace["main"]()
    finally:
        matplotlib.figure.Figure.savefig = ORIGINAL_SAVEFIG
        plt.close = ORIGINAL_CLOSE

    if len(CAPTURED) != 1:
        raise RuntimeError(f"Expected one captured Figure 3 object, found {len(CAPTURED)}")

    fig = CAPTURED[0]
    fig.canvas.draw()
    resolved = [(axis, axis.get_position().frozen()) for axis in fig.axes]
    fig.set_layout_engine("none")
    for axis, position in resolved:
        axis.set_axes_locator(None)
        axis.set_position(position)

    stem = HERE / "figure/Figure_3_specimen_specific_residual_recovery_corrected_render_fixed"
    stem.parent.mkdir(parents=True, exist_ok=True)
    ORIGINAL_SAVEFIG(fig, stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    ORIGINAL_SAVEFIG(fig, stem.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    ORIGINAL_CLOSE(fig)
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
