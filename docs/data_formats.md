# Data formats

## Reference AnnData

- observations: reference cells or spots;
- variables: unique gene identifiers;
- expression: `X`, an explicitly selected layer, or `spliced + unspliced` when both exist;
- coordinates: `obsm["global_sphere"]`, `obsm["spatial"]` or `obsm["X_spatial"]`;
- coordinates must be finite and have two or three columns.

Counts are median-depth normalized and transformed with `log1p` unless `already_log=True` is explicit.

## Prediction-time target AnnData

The target supplies coordinates and exactly the measured panel columns. `read_target` opens an AnnData object in backed mode, materializes only those columns and closes the file. It does not materialize target non-anchor expression. Target-wide molecular depth is not used; depth correction is predicted from the measured anchors using the source-fitted relationship.

## Outputs

`adapted.h5ad`, `registered_atlas.h5ad` and `predicted_residual.h5ad` contain points × genes in `X`, coordinates in `obsm["spatial"]` and run metadata in `uns["hyperspatial"]`. Tabular outputs record the measured panel, per-gene source support and selected predictor. Every persisted run includes SHA-256 input and artifact records.

MAP-T accepts a frozen source-trained tracking bundle with descendant-to-ancestor identifiers, early ancestor identifiers, later coordinates and temporal residual predictions. It does not accept later target expression.

