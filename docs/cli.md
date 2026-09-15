# Command-line interface

```text
hyperspatial design --reference REF.h5ad --budget 16 --out panel_run
hyperspatial adapt --reference REF.h5ad --target TARGET.h5ad --panel panel.tsv --out adaptation_run
hyperspatial report --run adaptation_run --out report.tsv
hyperspatial forecast --early adaptation_run --tracking tracking.npz --out forecast_run
hyperspatial validate --run adaptation_run --truth truth.h5ad --out metrics.tsv
hyperspatial inspect-run adaptation_run
```

Each scientific command accepts `--config CONFIG.yaml`. Validation is a distinct post-prediction command so hidden truth cannot enter fitting or selection.

