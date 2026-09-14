# Supplementary Figure S3 — full frozen component and ablation comparison

Status: **PASS**. All six requested components and all seven requested endpoints are serialized for each of the six directional target evaluations. The figure is a read-only rendering and descriptive ranking of frozen metrics; no model was rerun, refit, retuned, or selected from target outcomes.

## Authoritative inputs and SHA-256

| Input | SHA-256 | Use |
|---|---|---|
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv` | `eeb94cb07f89b3fc8efd6d913db1e378c2e0ba418e04857bd2ae7ed80999af31` | Six primary endpoint values for six methods × six target evaluations |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/05_metrics/RESIDUAL_ENDPOINTS_BY_EMBRYO.tsv` | `e65bc7375f6a8dbd7c3b04cd77086cc421d384b09a156cb63e813b57c9d8c355` | Frozen residual spatial Pearson and predictable-gene counts |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/05_metrics/COMPONENT_PARETO_RESULTS.tsv` | `ec89f44c740cd33230e5a85f99dd13ba6f1a71b010e479d0132c74747e09cbfd` | Frozen aggregate component metrics used in panel D |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/05_metrics/METHOD_COMPARISONS_BY_EMBRYO.tsv` | `0153d1fbdbeaeed08db3d4b98d1934d4a064992a06d33bddb85eeb23e837c819` | Frozen paired-comparison cross-check |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/ALL_TARGET_PREDICTIONS_MANIFEST.json` | `5472a774033d70bfdf29e7b15928159396f87944cd54b051f109f6033445d148` | Frozen prediction registry and component provenance |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/00_protocol_lock/PROTOCOL_LOCK.json` | `e579bf1f7e55f8223eab3513168cdcdbbff6213e2191c871371516495544bba5` | Locked architecture, methods, stages, directions, and target-access boundary |
| `summarize_hyperspatial_map_cross_stage_frozen_20260803.py` | `a3f19d5e00808bb503322e4ea8c20730e3e29ee7df026a7453b2c780e0f18b3f` | Frozen metric consolidation provenance |

## Evaluation design and unit

The six displayed evaluations preserve 50% epiboly, 75% epiboly, and 6-somite, separately for E1→E2 and E2→E1. The biological and evaluation unit is one directional target embryo/volume. Genes are technical inner units and are not biological replicates.

The frozen source-defined confirmatory gene sets exclude the 16 anchor genes. No target-selected genes or metrics were introduced. Target non-anchor expression remained unavailable during fitting and model selection and was opened only under the frozen evaluation protocol after prediction serialization.

## Component definitions

1. **Registered nearest-neighbour:** registered source expression transferred from the single nearest source cell.
2. **Registered 12-NN IDW:** registered reference-field transfer by inverse-distance weighting over 12 source neighbours; the reference comparator for oriented effects.
3. **Global residual operator:** registered IDW plus the source-trained global operator applied to target anchor residuals.
4. **Local multiscale residual operator:** registered IDW plus the source-trained operator using raw, 12-neighbour-smoothed, and 32-neighbour-smoothed target anchor residual features.
5. **Anchor-only decoder:** source-trained calibrated decoder from the 16 directly measured target anchors, without registered morphology transfer.
6. **Full HyperSpatial-MAP:** frozen source-gated fusion of the registered field and eligible anchor-residual/anchor-decoder candidates.

These labels describe the frozen artifacts; no component was reconstructed from another prediction.

## Endpoint direction and effects

| Endpoint | Better direction | Panel-B effect versus registered IDW |
|---|---|---|
| Spatial Pearson | Higher | method − IDW |
| Residual spatial Pearson | Higher | method − IDW |
| AUPRC enrichment | Higher | method − IDW |
| Prevalence-matched Dice | Higher | method − IDW |
| Weighted centroid error | Lower | IDW − method |
| Spatial EMD | Lower | IDW − method |
| log1p RMSE | Lower | IDW − method |

Thus every positive value in panel B favors the named method over registered 12-NN IDW. Endpoint values remain in their native units; no z-score normalization is used.

## Panel definitions

- **A — method × endpoint overview:** within-target ranks for all six methods, seven endpoints, and six target evaluations. Rank 1 is best after respecting endpoint direction. Ranks are descriptive transformations of frozen values.
- **B — paired effects versus registered IDW:** oriented target-level effects in native endpoint units. The figure subtitle states “Effect relative to registered 12-NN IDW; positive = better.” Every target is shown separately; the IDW self-comparison is zero and is omitted from the plotted method traces.
- **C — mean rank across seven endpoints (1 = best):** mean of the seven within-target endpoint ranks, displayed independently for every directional target evaluation.
- **D — residual-Pearson versus Dice/Pareto view:** exact frozen aggregate residual spatial Pearson and Dice for every component. Anchor-only, Full MAP, Local residual, and Global residual are identified with short colored leader-line labels. Upper right is favorable; the panel does not assert that one method must dominate all others.

## Interpretation

Anchor-only is permitted to win isolated—and in several targets, multiple—metrics. It provides a strong decoder-based reference and often ranks first for residual recovery, Dice, AUPRC, or RMSE. Those outcomes are retained without qualification or suppression.

Full HyperSpatial-MAP remains near the favorable end across the endpoint set and often leads morphology-sensitive or registered-field endpoints, but it is not universally superior. Its intended interpretation is balanced integration of registered morphology and specimen-specific anchor information. The comparison supports component specialization and trade-offs rather than a claim that full MAP wins every metric.

## Output table

`all_component_metrics.tsv` contains one row per method × directional target evaluation (36 rows). It preserves the frozen endpoint values and adds only display-derived within-target ranks, correctly oriented effects versus registered IDW, and explicit provenance labels for the gene set and evaluation unit.
