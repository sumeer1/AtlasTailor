#!/usr/bin/env python3
"""Apply the liver-v9 marker-area/export strategy to immutable v36 figures."""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import matplotlib.figure
import pandas as pd
from PIL import Image


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parents[1]
V36 = PROJECT / "PI_revision/36_Liver_target_withheld_gene_figure_visual_refinement_20260827_132522"
V36_SCRIPT = V36 / "scripts/render_refined.py"
LOCKED = PROJECT / "PI_revision/34_Liver_target_withheld_gene_interpretation_20260827_094120"

AREA_MULTIPLIER = 1.60
DIAMETER_MULTIPLIER = math.sqrt(AREA_MULTIPLIER)
OLD_DPI = 360
NEW_DPI = 500
PREVIEW_WIDTH = 2200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v36():
    spec = importlib.util.spec_from_file_location("v36_frozen_renderer", V36_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def preview(master: Path, destination: Path):
    with Image.open(master) as image:
        height = round(image.height * PREVIEW_WIDTH / image.width)
        image.resize((PREVIEW_WIDTH, height), Image.Resampling.LANCZOS).save(destination)


def dimensions(path: Path):
    with Image.open(path) as image:
        return image.size


def main():
    v36 = load_v36()
    old_sizes = dict(v36.REFINED_SIZE)
    new_sizes = {disease: value * AREA_MULTIPLIER for disease, value in old_sizes.items()}

    # Reuse v36's immutable loader, layout, labels, metrics, alpha, colormaps and
    # normalization verbatim. Only the collection marker areas and PNG dpi change.
    v36.OUT = OUT
    v36.REFINED_SIZE = new_sizes
    summary = pd.read_csv(LOCKED / "TARGET_WITHHELD_GENE_RESULTS.tsv", sep="\t")
    lock = pd.read_csv(LOCKED / "CANDIDATE_GENE_LOCK.tsv", sep="\t")
    source = v36.load_source_module()
    source.validate_inputs(summary, lock)
    loader = source.load_frozen_loader()
    maps = {}
    for disease, genes in {
        "MASLD": ["DCN", "TAT"],
        "PSC": ["KRT8", "TAT"],
        "AH": ["NFKBIA"],
    }.items():
        for gene, values in loader.load_gene_maps(disease, genes).items():
            maps[(disease, gene)] = values

    original_savefig = matplotlib.figure.Figure.savefig

    def savefig_500dpi_for_png(figure, filename, *args, **kwargs):
        if Path(filename).suffix.lower() == ".png":
            kwargs["dpi"] = NEW_DPI
        return original_savefig(figure, filename, *args, **kwargs)

    matplotlib.figure.Figure.savefig = savefig_500dpi_for_png
    try:
        v36.render(source.MAIN, maps, summary, source,
                   "Liver_target_withheld_genes_main_v9style")
        v36.render(source.SUPP, maps, summary, source,
                   "Liver_target_withheld_genes_TAT_v9style", supplementary=True)
    finally:
        matplotlib.figure.Figure.savefig = original_savefig

    main_intermediate = OUT / "Liver_target_withheld_genes_main_v9style.png"
    main_master = OUT / "Liver_target_withheld_genes_main_v9style_500dpi.png"
    main_intermediate.rename(main_master)
    supp_intermediate = OUT / "Liver_target_withheld_genes_TAT_v9style.png"
    supp_master = OUT / "Liver_target_withheld_genes_TAT_v9style_500dpi.png"
    supp_intermediate.rename(supp_master)

    main_preview = OUT / "Liver_target_withheld_genes_main_v9style_preview_2200px.png"
    supp_preview = OUT / "Liver_target_withheld_genes_TAT_v9style_preview_2200px.png"
    preview(main_master, main_preview)
    preview(supp_master, supp_preview)

    old_main = V36 / "Liver_target_withheld_genes_main_refined.png"
    old_supp = V36 / "Liver_target_withheld_genes_TAT_supplementary_refined.png"

    entries = [("main", source.MAIN), ("supplementary", source.SUPP)]
    lines = [
        "# V9-style refinement log",
        "",
        "This is a cosmetic-only derivative of the canonical v36 target-withheld figures. It uses the same strategy as the integrated liver v9 refinement: every spatial marker area is multiplied by 1.60 and the master PNG is exported at 500 dpi.",
        "",
        "## Marker and scale audit",
        "",
        "| Disease / gene | Figure | Previous marker area | New marker area | Area increase | Diameter increase | Expression vmin before / after | Expression vmax before / after | Residual vmin before / after | Residual vmax before / after |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for figure_name, entries_for_figure in entries:
        for disease, gene, _ in entries_for_figure:
            _, expr_min, expr_max, residual_max = v36.scale_limits(maps[(disease, gene)])
            lines.append(
                f"| {disease} / {gene} | {figure_name} | {old_sizes[disease]:.2f} | "
                f"{new_sizes[disease]:.2f} | +60.0% | +{(DIAMETER_MULTIPLIER - 1) * 100:.1f}% | "
                f"{expr_min:.9g} / {expr_min:.9g} | {expr_max:.9g} / {expr_max:.9g} | "
                f"{-residual_max:.9g} / {-residual_max:.9g} | {residual_max:.9g} / {residual_max:.9g} |"
            )

    old_main_dim, new_main_dim = dimensions(old_main), dimensions(main_master)
    old_supp_dim, new_supp_dim = dimensions(old_supp), dimensions(supp_master)
    main_preview_dim, supp_preview_dim = dimensions(main_preview), dimensions(supp_preview)
    lines += [
        "",
        "## Export audit",
        "",
        f"- Previous PNG dpi: {OLD_DPI}.",
        f"- New master PNG dpi: {NEW_DPI}.",
        f"- Main PNG dimensions: {old_main_dim[0]} × {old_main_dim[1]} px → {new_main_dim[0]} × {new_main_dim[1]} px.",
        f"- TAT supplementary PNG dimensions: {old_supp_dim[0]} × {old_supp_dim[1]} px → {new_supp_dim[0]} × {new_supp_dim[1]} px.",
        f"- Main preview dimensions: {main_preview_dim[0]} × {main_preview_dim[1]} px; Lanczos downsampling.",
        f"- TAT supplementary preview dimensions: {supp_preview_dim[0]} × {supp_preview_dim[1]} px; Lanczos downsampling.",
        "- Physical dimensions and panel geometry: unchanged.",
        "- PDF/SVG workflow: unchanged vector export with the existing 300-dpi setting for rasterized spatial collections.",
        "",
        "## Integrity",
        "",
        f"- Canonical v36 renderer SHA256: `{sha256(V36_SCRIPT)}`.",
        "- Predictions recomputed: **NO**.",
        "- Metrics recomputed or changed: **NO**.",
        "- Genes, samples, coordinates or geometry changed: **NO**.",
        f"- Opacity changed: **NO** (retained alpha {v36.ALPHA:.2f}).",
        "- Normalization, colormaps, expression limits or residual limits changed: **NO**.",
        "- Text, labels and panel geometry changed: **NO**.",
    ]
    (OUT / "V9STYLE_REFINEMENT_LOG.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
