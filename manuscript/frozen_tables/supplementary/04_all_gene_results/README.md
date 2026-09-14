# Supplementary Figure S4 — all-gene performance and source-CV guard behavior

Status: **PASS** for the finalized frozen-evidence scope. No model was rerun, refit, retuned, or selected from target outcomes. Gene-level target MSE and predictive uncertainty remain unavailable and were not introduced.

## Authoritative inputs and SHA-256

| Input | SHA-256 | Use |
|---|---|---|
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/05_metrics/PER_GENE_SPATIAL_AND_RESIDUAL_METRICS.tsv` | `6c969bbeca3aafa1bcee0fe88bc7c94cff57370f92deade774f0a4558763b055` | Frozen non-anchor per-gene spatial and residual Pearson records |
| `results_v10/hyperspatial_map_full_article_20260804_132224/04_3d_consolidation/THREE_D_GUARD_DECISIONS.tsv` | `6726ed9a087c97d1c7af20c9c396954915f253a8658cb451fce2328252883b78` | Frozen gene-index/name map and four-direction guard consolidation |
| `results_v10/hyperspatial_map_full_article_20260804_132224/04_3d_consolidation/THREE_D_CANDIDATE_SELECTION.tsv` | `3659f565a45e32373cde38845e2facf6c44a1d2e43872be08934f623a69889ad` | Frozen candidate-label cross-check |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv` | `7d02422b8c89f00dacb010e2fb66922f27b4ce07c3c7cd8c804a7880fc47244f` | Exact anchor indices used for non-anchor confirmatory evaluation |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E1_to_E2/prediction_manifest.json` | `cc72962f0d0c566caf89cc2d102bab70a4c1a2724e2b320bd458f7ee4e461c0f` | Frozen source-CV arrays and candidate selection |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E2_to_E1/prediction_manifest.json` | `a1613b056f052f21a19d36d9168b28499d3c7c2fbe6734578f961eb3560ecc07` | Frozen source-CV arrays and candidate selection |
| `results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2/prediction_manifest.json` | `52b1c9bd3021655715dccd9b94e630d68b406ee4d40222e7cca52ec811d97658` | Frozen 75% E1→E2 source-CV arrays and candidate selection |
| `results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1/prediction_manifest.json` | `0fff2f37c5fbbe33c2421f7292dc3ab4d65c28de61be083e38b34a33286244c4` | Frozen 75% E2→E1 source-CV arrays and candidate selection |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E1_to_E2/prediction_manifest.json` | `34cdf2dee1d871e9b67d2edfd075c85d26399b2a287755e6443250b26e4e0bfb` | Frozen source-CV arrays and candidate selection |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E2_to_E1/prediction_manifest.json` | `03108cf23b75c8c513bd30de1f1083e09f447284b3aa3ab2e31d303191737dbd` | Frozen source-CV arrays and candidate selection |
| `run_hyperspatial_map_stage1b_20260803.py` | `317c5d642541def2e3384ec914c20067b6630dd535aecd4be880bd1a2a550bb6` | Hashed guard and candidate-fusion implementation |

## Frozen guard

Adaptation is supported only when the selected predictor has lower source-CV MSE than registered 12-NN IDW and source-CV ΔPearson greater than `0.03`. Otherwise the final predictor falls back to registered 12-NN IDW. Anchors are excluded from confirmatory prediction and evaluation. No target information enters the guard or candidate selection.

Frozen candidate labels are grouped as follows:

- **IDW fallback:** guard failure or `registered_idw`;
- **global residual:** `global_anchor_w0.00`;
- **local residual:** `local_anchor_w0.00`;
- **anchor-only:** anchor weight `1.00`;
- **calibrated mixture:** global/local candidate with anchor weight `0.25`, `0.50`, or `0.75`.

The guard therefore uses source validation to decide when target-specific correction is supported and otherwise retains registered atlas transfer.

## Residual-Pearson interpretation

Registered IDW predicts the atlas expectation and hence no target-specific residual. Its residual field is zero, so residual Pearson is undefined rather than a valid zero-valued comparator. Consequently:

- panel B left reports spatial-Pearson effect relative to registered 12-NN IDW, oriented so positive is better;
- panel B right reports **MAP residual spatial Pearson directly** under the title “Target-specific residual recovery”;
- residual Pearson is not described or interpreted as Δ/effect versus IDW anywhere in the finalized figure.

The existing supplement table contains a legacy display-derived column named `effect_vs_idw_residual_spatial_pearson`; because its frozen IDW placeholder is zero, its numbers equal `residual_spatial_pearson_map`. It must be interpreted only as MAP residual spatial Pearson, not as a comparative residual-Pearson effect.

## Panels

- **A — non-anchor residual recovery:** frozen MAP residual spatial Pearson for source-defined predictable non-anchor genes in all six target evaluations.
- **B — gene-level spatial performance and residual recovery:** left, MAP-minus-IDW spatial Pearson; right, direct MAP residual spatial Pearson. Values are unchanged from the frozen tables.
- **C — frozen source-CV predictor assignment:** category fractions among all 479 non-anchor measured genes in each source setting.
- **D — source validation versus target effect:** source-CV ΔPearson versus target MAP-minus-IDW spatial Pearson.
- **E — target spatial-Pearson effect by frozen selected predictor:** the same target spatial-Pearson effects stratified by the frozen source-only selected-predictor category. This is the former panel F, relabelled E because no target-ΔMSE panel can be constructed.

## Exact panel D/E eligibility and evaluation unit

Panels D and E use the complete frozen spatial-Pearson evaluation subset: **600 gene×target records**, comprising exactly **100 source-defined, non-anchor spatial-evaluation genes in each of six directional target evaluations**. Every one of these 600 records has a serialized source-CV ΔPearson, frozen target spatial-Pearson effect, and selected-predictor category. No record is removed for missing CV or target values, and no observation is added.

One point in panel D and one observation in panel E represent one gene evaluated in one directional target embryo: a **gene×target technical evaluation unit**. The 100 source-defined genes can differ across source settings, so 600 is not asserted to be 600 unique genes. It is also not a biological sample size.

Panel E category counts sum to 600:

| Frozen selected predictor | n gene×target records |
|---|---:|
| IDW fallback | 55 |
| Global residual | 11 |
| Local residual | 15 |
| Anchor-only | 30 |
| Calibrated mixture | 489 |

The unequal `n` values arise solely from the frozen source-CV guard and candidate assignments within the same complete 600-record subset. IDW-fallback target spatial-Pearson effects are exactly zero by definition because guarded MAP retains the registered 12-NN IDW prediction for those genes.

## Gene eligibility and biological/statistical units

All confirmatory target-performance panels use non-anchor genes only. Panel A residual-set sizes are 477, 478, 465, 467, 441, and 414 genes for 50% E1→E2, 50% E2→E1, 75% E1→E2, 75% E2→E1, 6-somite E1→E2, and 6-somite E2→E1, respectively. These are the frozen source-defined predictable non-anchor sets.

The embryo/directional target volume is the biological and statistical unit. Genes and gene×target records are technical evaluation units, not biological replicates; their distributions do not increase biological sample size.

## Unavailable analyses

Gene-level target MSE was not serialized, so source-CV ΔMSE versus target ΔMSE was not created. Gene-level AUPRC enrichment, Dice, centroid error, spatial EMD, and log1p RMSE were also not serialized. Three-dimensional predictive uncertainty and interval coverage were not evaluated because no frozen estimator existed. None of these quantities was reconstructed post hoc.
