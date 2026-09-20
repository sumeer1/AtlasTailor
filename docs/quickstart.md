# Quickstart

Prepare a deeply measured reference AnnData object and a target AnnData object with coordinates in `obsm["spatial"]`. Required fields are documented in [data conventions](data_formats.md).

## Python

```python
from hyperspatial import adapt, design_panel, validate
from hyperspatial.io import read_reference, read_target, read_truth

reference = read_reference("reference.h5ad", coordinate_key="spatial")
panel = design_panel(reference, budget=16, out="runs/panel")

# Only the declared panel is materialized from the target at prediction time.
target = read_target(
    "target.h5ad",
    measured_genes=panel.genes,
    coordinate_key="spatial",
)
result = adapt(reference, target, panel.genes, out="runs/adaptation")

# Optional retrospective evaluation follows the completed prediction lock.
truth, _ = read_truth("target.h5ad", result.genes, coordinate_key="spatial")
metrics = validate(result, truth, out="runs/adaptation/gene_metrics.tsv")
```

`runs/panel/` records the source-only measurement design. `runs/adaptation/` contains the registered atlas prior, adapted reconstruction, predicted atlas-relative residual, gene-level support decisions, registration metadata and a hashed run manifest.

## Command line

```bash
hyperspatial design \
  --reference reference.h5ad \
  --budget 16 \
  --out panel_run

hyperspatial adapt \
  --reference reference.h5ad \
  --target target_anchors_only.h5ad \
  --panel panel_run/recommended_panel.tsv \
  --config config.yaml \
  --out adaptation_run

hyperspatial inspect-run adaptation_run
```

The command-line interface enforces the same boundary. Use `hyperspatial inspect-run adaptation_run` to inspect run status and provenance.

## Real public datasets

For an end-to-end biological run, use the [zebrafish weMERFISH tutorial](https://github.com/sumeer1/AtlasTailor/blob/main/notebooks/01_run_atlastailor_zebrafish.ipynb), the [human DLPFC Visium tutorial](https://github.com/sumeer1/AtlasTailor/blob/main/notebooks/00_run_atlastailor_dlpfc.ipynb), or the [user-data template](https://github.com/sumeer1/AtlasTailor/blob/main/notebooks/04_run_atlastailor_on_your_data.ipynb). The exact frozen manuscript workflows remain in [AtlasTailor-reproducibility](https://github.com/sumeer1/AtlasTailor-reproducibility).
