# AtlasTailor

**Sparse measurements tailor a deeply measured spatial atlas to an individual specimen.**

AtlasTailor reconstructs a target spatial molecular state from a deeply measured reference, target geometry and a small, prespecified target panel. It is an atlas-adaptation framework rather than conventional within-sample imputation: the reference and target may be different specimens, developmental stages, tissue sections, disease states or spatial technologies.

```text
registered atlas prior + sparse target adaptation
                       = individualized molecular reconstruction
```

## Workflows

```text
DESIGN:   reference → source-only measurement panel
ADAPT:    reference + target geometry + measured anchors → adapted target
EXPLORE:  registered prior ↔ target-specific residual ↔ adapted field
FORECAST: adapted early state + tracking map → later molecular forecast
```

The software is a thin, regression-tested interface over the frozen manuscript implementation. It does not introduce a separate scientific model.

## Start here

- [Install AtlasTailor](installation.md).
- [Run the minimal Python workflow](quickstart.md).
- [Use a public zebrafish or DLPFC tutorial](tutorials.md).
- [Prepare an AnnData input](data_formats.md).
- [Understand the prediction-time information boundary](pipeline.md).

## Scientific boundary

Geometry registration does not receive target molecular measurements. Panel selection, predictor selection, calibration and fallback use source information. At prediction time the target loader materializes only the declared measured genes. Full target expression is accepted only by the separate post-lock validation workflow.

The main software repository contains the reusable package and tutorials. Exact executed manuscript workflows, frozen configurations, prediction locks and source-panel provenance are maintained in [AtlasTailor-reproducibility](https://github.com/sumeer1/AtlasTailor-reproducibility).
