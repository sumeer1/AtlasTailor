# Running AtlasTailor

Prepare a deeply measured reference AnnData object and a target AnnData object containing geometry plus only the declared measured target panel. Required fields are documented in [data formats](data_formats.md).

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

The target loader materializes only declared panel columns. Held-out truth is a separate validation input and is accepted only after prediction artifacts have been completed and hashed.

For worked biological examples, use the notebooks in the repository root and the exact workflows in [AtlasTailor-reproducibility](https://github.com/sumeer1/AtlasTailor-reproducibility).
