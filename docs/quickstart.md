# Quickstart

Generate or use the checked-in tiny fixtures, then run:

```bash
hyperspatial design --reference examples/minimal_2d/reference.h5ad --budget 8 --out panel_run
hyperspatial adapt --reference examples/minimal_2d/reference.h5ad --target examples/minimal_2d/target.h5ad --panel panel_run/recommended_panel.tsv --config examples/minimal_2d/config.yaml --out adaptation_run
hyperspatial report --run adaptation_run --out adaptation_report.tsv
hyperspatial inspect-run adaptation_run
```

The target loader materializes only declared panel columns. Held-out truth is a separate validation input and is accepted only after prediction artifacts have been completed and hashed.
