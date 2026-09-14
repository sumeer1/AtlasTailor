#!/usr/bin/env python3
"""Generate compact executable tutorial notebooks from reviewed source cells."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

def notebook(name, title, intro, cells):
    doc = nbf.v4.new_notebook()
    doc["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    doc["cells"] = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}")]
    for cell in cells:
        doc["cells"].append(nbf.v4.new_code_cell(cell) if not cell.startswith("MD:") else nbf.v4.new_markdown_cell(cell[3:]))
    nbf.write(doc, NB / name)

setup2 = """from pathlib import Path
import tempfile
import hyperspatial as hs
from hyperspatial.io import read_reference, read_target
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
EX = ROOT / 'examples' / 'minimal_2d'
reference = read_reference(EX / 'reference.h5ad')"""

notebook("00_quickstart.ipynb", "Quickstart", "Adapt a tiny target while target non-anchor columns remain inaccessible to prediction.", [
    setup2,
    "panel = hs.design_panel(reference, budget=8, config=hs.DesignConfig(budget=8, budgets=(8,)))\npanel.genes",
    "target = read_target(EX / 'target.h5ad', panel.genes)\nresult = hs.adapt(reference, target, panel.genes, config=hs.AdaptConfig(registration='pre_registered', seeds=(42,)))",
    "g = result.gene(result.genes[0])\n(g.gene, g.support['selected_predictor'], g.registered_atlas.shape, g.adapted.shape)",
    "out = Path(tempfile.mkdtemp()) / 'adaptation'\nresult.export(out)\nsorted(p.name for p in out.iterdir())",
])

notebook("01_design_sparse_panel.ipynb", "Design a sparse panel", "Source-only nested spatial cross-validation evaluates specified measurement budgets; it does not establish a universal optimum.", [
    setup2,
    "design = hs.design_panel(reference, budget=8, config=hs.DesignConfig(budget=8, budgets=(4,8,16)))\ndesign.ordered_panel",
    "design.budget_curve",
    "design.marker_utility.sort_values('loao_utility', ascending=False).head()",
    "design.conditional_utility.sort_values('conditional_utility').head()",
])

setup3 = setup2.replace("minimal_2d", "minimal_3d")
notebook("02_adapt_3d_specimen.ipynb", "Adapt a 3D specimen", "The target contributes geometry and measured anchors; anchors enter only after geometry registration.", [
    setup3,
    "panel = hs.design_panel(reference, budget=8, config=hs.DesignConfig(budget=8, budgets=(8,)))\ntarget = read_target(EX / 'target.h5ad', panel.genes)",
    "result = hs.adapt(reference, target, panel.genes, config=hs.AdaptConfig(registration='pre_registered', seeds=(42,)))\nresult.support.selected_predictor.value_counts()",
    "result.gene(result.genes[5])",
])

notebook("03_explore_target_specific_residuals.ipynb", "Explore target-specific residuals", "Residual structure is variation beyond the registered atlas expectation and is not automatically biological.", [
    setup2,
    "panel = hs.design_panel(reference, budget=8, config=hs.DesignConfig(budget=8, budgets=(8,)))\ntarget = read_target(EX / 'target.h5ad', panel.genes)\nresult = hs.adapt(reference, target, panel.genes, config=hs.AdaptConfig(registration='pre_registered', seeds=(42,)))",
    "ranking = hs.explore(result)\nranking.head(10)",
    "result.gene(ranking.iloc[0].gene)",
])

notebook("04_forecast_with_MAPT.ipynb", "Forecast with MAP-T", "A compatible tracking map transports the individualized early field; the frozen source-trained temporal residual supplies the supported correction.", [
    setup2,
    "panel = hs.design_panel(reference, budget=8, config=hs.DesignConfig(budget=8, budgets=(8,)))\ntarget = read_target(EX / 'target.h5ad', panel.genes)\nearly = hs.adapt(reference, target, panel.genes, config=hs.AdaptConfig(registration='pre_registered', seeds=(42,)))",
    "forecast = hs.forecast(early, EX / 'tracking_bundle.npz')\n(forecast.forecast.shape, forecast.support.selected_predictor.value_counts())",
])

notebook("05_adapt_2d_visium.ipynb", "Adapt 2D Visium-like data", "The 2D adapter changes coordinate dimensionality and spatial folds, not the core atlas-prior/residual/guard logic.", [
    setup2,
    "panel = hs.design_panel(reference, budget=8, config=hs.DesignConfig(budget=8, budgets=(8,)))\ntarget = read_target(EX / 'target.h5ad', panel.genes)",
    "result = hs.adapt(reference, target, panel.genes, config=hs.AdaptConfig(registration='pre_registered', seeds=(42,)))\n(result.coordinates.shape, result.adapted.shape)",
])

