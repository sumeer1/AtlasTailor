# AtlasTailor pipeline

AtlasTailor separates experimental design, prediction and retrospective evaluation so that hidden target expression cannot influence the fitted reconstruction.

## 1. Design

```text
deeply measured reference → source-only 16-gene panel
```

`design_panel` receives a reference atlas only. It evaluates candidate measurements using spatial source validation. The primary manuscript budget was 16 genes; that number is a frozen experimental setting, not a claim of universal optimality.

## 2. Adapt

```text
reference expression + source geometry
                         target geometry
                         measured target panel
                                  ↓
registered atlas prior + predicted target-specific residual
                                  ↓
                         adapted molecular field
```

The stages are:

1. canonicalize source and target coordinates independently;
2. register source geometry to target geometry without molecular target input;
3. transfer the reference with registered 12-nearest-neighbour inverse-distance weighting;
4. calculate atlas-relative residuals for the measured target genes;
5. construct raw, 12-neighbour and 32-neighbour residual features;
6. apply source-trained centered reduced-rank predictors;
7. retain a correction only where source validation supports adaptation;
8. otherwise retain the registered atlas prior.

The persisted run contains input hashes, configuration, registration information, prediction artifacts, per-gene predictor assignments and an immutable completed manifest.

## 3. Explore

The adapted field is decomposed as:

```text
adapted target = registered atlas prior + predicted atlas-relative residual
```

Residual structure describes variation beyond the registered atlas expectation. It is not automatically biological and should be interpreted using independent annotations or prespecified programs.

## 4. Validate after locking

Retrospective truth is opened only after the adaptation manifest is complete. `validate` reports gene-level spatial Pearson, log1p RMSE, residual Pearson and spatial localization metrics. Registered IDW predicts zero atlas-relative correction, so its residual correlation is undefined rather than zero.

## 5. Forecast

AtlasTailor-T combines an individualized early state with a compatible, source-trained tracking bundle. It performs tracking-map-conditioned forecasting and does not treat predicted intermediate molecular states as experimentally validated observations.
