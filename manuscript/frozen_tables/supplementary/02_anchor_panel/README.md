# Supplementary Figure S2 — source-only 16-anchor panel selection

Status: **PASS**. The finalized figure contains only quantities explicitly available in authoritative frozen source-only anchor-characterization outputs. No new analysis, anchor reselection, tuning, panel-size optimization, or target non-anchor expression access was performed.

## Authoritative inputs and SHA-256

| Input | SHA-256 | Use |
|---|---|---|
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv` | `7d02422b8c89f00dacb010e2fb66922f27b4ce07c3c7cd8c804a7880fc47244f` | Exact source-selected anchors, ranks, candidate counts, and source-only coverage for all six settings |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/09_figures_supplementary/source_data/Supplementary_Figure_5_anchor_expression_qc.tsv` | `ed62c350ee271c70ef1f1674ab85720fcf7ec5545d311edace1e4c44ced43c4b` | Frozen source detection prevalence and coverage values |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/03_anchor_panels/ANCHOR_PROGRAMME_ANNOTATION_LIMITATION.md` | `94eb06571c58698294cea26e89d72a3df9e3f82d5c791747b1986261974f3277` | Frozen limitation on post-hoc programme annotation |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/00_protocol_lock/PROTOCOL_LOCK.json` | `e579bf1f7e55f8223eab3513168cdcdbbff6213e2191c871371516495544bba5` | Locked 16-gene budget, source-only rule, target-access boundary, stages, and directions |
| `run_hyperspatial_map_stage1b_20260803.py` | `317c5d642541def2e3384ec914c20067b6630dd535aecd4be880bd1a2a550bb6` | Frozen selector provenance |

## Panel definitions

- **A — exact source-only 16-gene panels:** all anchors are shown in frozen rank order for 50% epiboly, 75% epiboly, and 6-somite, separately for E1→E2 and E2→E1.
- **B — source detection/prevalence:** exact frozen source-cell detection prevalence for each selected anchor. Points are within-embryo gene features, not biological replicates; horizontal lines are descriptive panel medians.
- **C — frozen source-only coverage criterion:** for each measured gene, the frozen selector takes the maximum squared absolute source-expression correlation to any selected anchor, then averages that coverage across measured genes: `mean_g max_a |r_source(g,a)|²`. Higher values indicate greater source-gene coverage by the fixed panel. One exact frozen panel-level value is shown per directional source. No separate redundancy metric was serialized.

The 16 genes are a fixed experimental budget, not an optimized panel size, biological optimum, or estimate of intrinsic dimension. The biological unit is the embryo; genes and cells are technical inner observations. Target non-anchor expression was never used in anchor selection or the displayed characterization.

## Serialization boundary

Anchor–anchor source-correlation matrices and per-gene anchor-similarity tables were not serialized and were therefore not reconstructed post hoc. No correlation, redundancy, spatial-variance, or informativeness statistic was inferred or invented for this figure.

Main Figure 5h remains the frozen anchor-similarity/recovery analysis; it was not duplicated or extended in Supplementary Figure S2.

## Output tables

- `anchor_panels.tsv` is a byte-for-byte supplement copy of the authoritative frozen 96-row panel table.
- `anchor_similarity_all_genes.tsv` is retained as a header-only schema because the requested per-gene similarities were not serialized; it contains no reconstructed values.
