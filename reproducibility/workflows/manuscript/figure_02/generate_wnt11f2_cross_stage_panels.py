#!/usr/bin/env python3
"""Render frozen wnt11f2 method maps at 50% epiboly and 6-somite.

The two requested panels use the E1-to-E2 direction to match the original
75% E1-to-E2 panel.  No prediction or metric is recomputed.  Within each
stage, Measured, IDW, gimVI, TransImpSpa and MAP use one common scale under
the exact existing panel-e rule.  A supplementary combined rendering applies
one pooled common scale across both stages and every displayed method.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "build_figure2_external_revision.py"
spec = importlib.util.spec_from_file_location("figure2_existing", SOURCE)
figure2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(figure2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        dpi = 300 if ext == "png" else 600
        fig.savefig(HERE / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight",
                    pad_inches=.08, facecolor="white")
    plt.close(fig)


def draw_one(data: dict, stem: str) -> None:
    fig = plt.figure(figsize=(3.55, 1.75))
    figure2.draw_wnt11f2(fig, data, standalone=True, include_idw=True)
    export(fig, stem)


def draw_combined(data_50: dict, data_6s: dict, stem: str) -> tuple[float, float]:
    # One scale across stages and methods. Include all five displayed maps;
    # measured values and four frozen predictions contribute symmetrically.
    all_values = np.concatenate([
        data["maps"][key]
        for data in (data_50, data_6s)
        for key in ("measured", figure2.IDW, "gimvi_adapted", "transimpspa", figure2.MAP)
    ])
    vmin, vmax = 0.0, float(np.nanquantile(all_values, .99))
    for data in (data_50, data_6s):
        data["vmin"], data["vmax"] = vmin, vmax

    fig = plt.figure(figsize=(3.75, 3.60))
    top = fig.add_axes([.02, .51, .96, .39])
    bottom = fig.add_axes([.02, .04, .96, .39])
    top.axis("off"); bottom.axis("off")

    def row(host, data):
        names = ["Measured", "IDW", "gimVI", "TransImpSpa", "MAP"]
        keys = ["measured", figure2.IDW, "gimvi_adapted", "transimpspa", figure2.MAP]
        step = 1 / len(names)
        for i, (name, key) in enumerate(zip(names, keys)):
            ax = host.inset_axes([i * step, .06, .94 * step, .82])
            ax.scatter(data["coords"][:, 0], data["coords"][:, 1],
                       c=data["maps"][key], s=.30, cmap="viridis",
                       vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(name, fontsize=6.5, pad=2)
        host.set_title(f"{data['stage']} {data['source']}→{data['target']}", fontsize=7.2)

    row(top, data_50); row(bottom, data_6s)
    fig.suptitle("Source-predeclared wnt11f2 · one shared cross-stage scale",
                 fontsize=8, y=.985)
    export(fig, stem)
    return vmin, vmax


def main() -> None:
    figure2.apply_style()
    # AUTH.RUNS indices 0 and 4 are fixed E1-to-E2 evaluations at 50% and 6s.
    data_50 = figure2.read_wnt11f2_maps(run_index=0, gene="wnt11f2")
    data_6s = figure2.read_wnt11f2_maps(run_index=4, gene="wnt11f2")
    draw_one(data_50, "Fig2e_wnt11f2_50p_E1_to_E2_all_methods_shared_scale")
    draw_one(data_6s, "Fig2e_wnt11f2_6s_E1_to_E2_all_methods_shared_scale")

    stage_limits = {
        data_50["target_key"]: {"vmin": data_50["vmin"], "vmax": data_50["vmax"]},
        data_6s["target_key"]: {"vmin": data_6s["vmin"], "vmax": data_6s["vmax"]},
    }
    global_vmin, global_vmax = draw_combined(
        data_50, data_6s,
        "Fig2e_wnt11f2_50p_and_6s_all_methods_one_cross_stage_scale",
    )

    paths = []
    for data in (data_50, data_6s):
        paths.append({
            "target_key": data["target_key"], "stage": data["stage"],
            "source": data["source"], "target": data["target"],
            "gene": data["gene"], "gene_index": data["gene_index"],
            "is_hidden_non_anchor": True,
            "stage_specific_vmin": stage_limits[data["target_key"]]["vmin"],
            "stage_specific_vmax": stage_limits[data["target_key"]]["vmax"],
            "measured_target": str(data["target_file"]),
            "registered_idw": str(data["paths"][figure2.IDW]),
            "adapted_gimvi": str(data["paths"]["gimvi_adapted"]),
            "transimpspa": str(data["paths"]["transimpspa"]),
            "hyperspatial_map": str(data["paths"][figure2.MAP]),
            "input_hashes": {
                key: sha256(Path(path)) for key, path in {
                    "target": data["target_file"], **data["paths"]
                }.items()
            },
        })
    audit = {
        "selection": "wnt11f2 retained; E1-to-E2 fixed at each requested stage to match original 75% E1-to-E2 panel",
        "stage_specific_scale_rule": (
            "vmin=0; pooled q99 of measured, MAP, adapted gimVI and TransImpSpa; "
            "registered IDW displayed under the same limits without changing the original panel-e rule"
        ),
        "combined_scale_rule": "vmin=0; pooled q99 across all five displayed maps at both stages",
        "combined_vmin": global_vmin, "combined_vmax": global_vmax,
        "panels": paths,
        "predictions_modified": False,
    }
    (HERE / "WNT11F2_50P_6S_PANEL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
