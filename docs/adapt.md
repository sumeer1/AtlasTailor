# Adapt a specimen

```text
reference and target geometry
        ↓
geometry-only registration
        ↓
registered 12-neighbour IDW prior
        ↓
measured anchor residuals at raw, local and broad scales
        ↓
source-trained global, local and anchor-only predictors
        ↓
source-only calibration and predictability guard
        ↓
registered prior + supported correction, otherwise registered prior
```

Default scientific settings are frozen: IDW `k=12`; residual contexts `k=12,32`; centered predictor rank 16; frozen ridge, gain and fusion grids; source-validation MSE tolerance 1.05 and Pearson margin 0.03. The public configuration rejects changes to these v1 scientific values. Seeds are algorithmic repeats, not biological replicates.

Target anchors do not enter registration. Target non-anchor values are not accepted by `adapt`.

