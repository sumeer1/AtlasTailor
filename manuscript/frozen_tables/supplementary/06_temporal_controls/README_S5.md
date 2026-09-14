# Supplementary Figure S5 — MAP-T temporal and tracking controls

Status: **PASS**. Every requested panel was constructed from frozen evidence. This presentation-only finalization changed labels and annotations without changing any metric, cohort, model, or table. No primary model was rerun, refit, retuned, or selected using target outcomes. The only source refit shown is the already-frozen overlap-excluded sensitivity.

## Authoritative inputs and SHA-256

| Frozen input | SHA-256 | Use |
|---|---|---|
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/01_protocol_lock/PROTOCOL_LOCK.json` | `316ce73367a07ed1c7baeed07b2856fd9c16115417740e175cbd3b30569ec48a` | Fold definitions, primary endpoint, 100 permutations, 2,000 bootstraps and claim scope |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/01_protocol_lock/TRACKING_PROVENANCE_LOCK.json` | `a6da69c99915025f0b84eb2453bd3b5ce945a0ffb5024684747bfa28a6353b73` | Tracking-map overlap provenance and permitted interpretation |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/01_protocol_lock/FROZEN_SOURCE_ONLY_ANCHOR_PANELS.tsv` | `fcda5d77d6f4425e3f662223a1b4032b57d9a585b7667cd843b1f2a09bd80cc9` | Frozen 16-anchor panels excluded from confirmatory gene evaluation |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/03_fold_A/FROZEN_FOLD_CONFIG.json` | `53ef4d24b5a963701e1848e3264e371feb12e6156ee1a6f3f1f883d36cdae644` | Fold A source-only configuration; target-75% access flag is false |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/04_fold_B/FROZEN_FOLD_CONFIG.json` | `b87fc74e7c19ebef17d51cc45e330b8c4518b462cd45628cc916db0cbde3a92e` | Fold B source-only configuration; target-75% access flag is false |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/12_source_data/PRIMARY_ENDPOINTS_ALL_FOLDS.tsv` | `a8f6d99284b5c92dd08fd0b4aaf979e569f42c1365e4de8dffd8310c81416fa3` | Frozen method, fold and tracking-cohort endpoint summaries |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv` | `c4a3febb18f2459372b120613f9d3261dc7ad1e594e9ff4fc6e9ba243f9f56bd` | Frozen non-anchor gene-level trajectory-aware and trajectory-free comparisons |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/12_source_data/DOMAIN_METRICS_ALL_FOLDS.tsv` | `0f60fde1b46fce2d108d414d166edcacecc530ea0146d4c2606d9c3d6ab950ca` | Frozen source-defined domain localization metrics |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/08_bootstrap/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv` | `423c80960aa09c43dbf9c77cce2efad31a270ae7a9226e360261cc41ceef7319` | Frozen 2,000-replicate ancestor-family bootstrap estimates and intervals |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/09_permutations/fold_A_results.tsv` | `7fad6b0c1664c1a3f07404234ed6eac2b97815f648c374aca2a8ae443a5d868b` | Fold A 100-shuffle result |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/09_permutations/fold_B_results.tsv` | `ec925416271c7bf501ed4a54cb0520b89e82939f26d5e634ac5376eaeae47d9d` | Fold B 100-shuffle result |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/06_tracking_overlap/fold_A_cohorts.tsv` | `b533523a3781f91a1fed3d7ad3d1b0ff2ee11d5c776e772a0e26fdaa4120c4f2` | Fold A frozen full/shared/disjoint cohort sizes |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/06_tracking_overlap/fold_B_cohorts.tsv` | `1909adb734c6d816aff2816197f1ce32705be85d3bd3d8e7096104281402efdb` | Fold B frozen full/shared/disjoint cohort sizes |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/06_tracking_overlap/fold_A_PREMODEL_definition.json` | `2bd460de5a63cb385db922919949d39804cda2b9d72abc14e74ec986d0f5d742` | Fold A overlap rule frozen before fitting and unblinding |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/06_tracking_overlap/fold_B_PREMODEL_definition.json` | `4a9e4b79a5acbac6a7a4690c23f120ee898c8add84662e48e452f576c5dea961` | Fold B overlap rule frozen before fitting and unblinding |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/07_overlap_excluded_retraining/fold_A_sensitivity.json` | `9c09006bab2531b0128199ed15c52fec04bdfff421e5934d4a30da156e7c5a1d` | Frozen Fold A overlap-excluded source-refit sensitivity |
| `results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/07_overlap_excluded_retraining/fold_B_sensitivity.json` | `1e1a197e48537bfd1a1b796df28c0f97effe160abc524d8b30fd6bea63824895` | Frozen Fold B overlap-excluded source-refit sensitivity |
| `run_hyperspatial_map_t_4d_frozen_20260804.py` | `c2392d939d58d40166a1d54413b3dbcf3f4e9ea26b69e996004fdccb58f64d4c` | Frozen runner, shuffle procedure and feature-set definitions |
| `hyperspatial_map_t_core_20260804.py` | `3f308c8cf04cfa112eedb780b680fa65c770e1741c2c777046d26d361bde6655` | Frozen MAP-T core implementation |

## Fold and endpoint definitions

- **Fold A:** train/source E1, test/target E2.
- **Fold B:** train/source E2, test/target E1.
- **Primary endpoint:** pooled correlation between target-75% future residual and predicted future residual over confirmatory non-anchor genes. Target-75% expression remained hidden until evaluation (`target_75_expression_accessed: false` in both frozen fold configurations).
- **Secondary summaries shown:** late-state RMSE over all 479 non-anchor genes and median domain Dice over the frozen source-defined domain genes (98 in Fold A; 99 in Fold B). Lower RMSE and higher Pearson/Dice are better.

The biological unit is the directional target embryo. Genes and gene×fold records are technical evaluation units and do not increase the number of biological replicates.

## Frozen baseline definitions

| Code | Figure label | Definition |
|---|---|---|
| B1 | transported IDW | Lineage-transported registered 12-NN inverse-distance atlas transfer |
| B2 | transported early MAP | Lineage-transported individualized early HyperSpatial-MAP prediction |
| B3 | mean correction | Mean source developmental correction |
| B4 | early-anchor-only | Early-anchor-only temporal predictor |
| B5 | trajectory-free | Temporal residual model using the permitted molecular inputs without trajectory-history features |
| B6 | trajectory-aware | Full HyperSpatial-MAP-T using the same permitted molecular inputs as B5 plus the frozen trajectory-history features |

B5 and B6 therefore have identical permitted molecular inputs; their difference is the addition of the prespecified trajectory-history features to B6. The 16 anchor genes are molecular features within an embryo, not biological replicates. Neither model used target non-anchor expression for fitting or model selection.

B2 predicts the transported individualized early molecular state and therefore predicts an identically zero future-residual field. Its future-residual Pearson is mathematically undefined. The frozen endpoint table serialized a zero placeholder; the supplement TSV retains that value only in `rho_future_pooled_frozen_raw` and reports the displayed/analytic future-residual Pearson as `NA`, never as zero. B2 late-state RMSE and domain Dice remain valid and are shown.

## Panels and exact evaluation units

- **A — temporal-method comparison:** one frozen directional target-embryo endpoint per method and fold. Future-residual Pearson, late-state RMSE and median domain Dice are shown separately for Fold A and Fold B. All six methods remain visible; B2 future-residual Pearson is marked `NA`.
- **B — gene-level trajectory-aware minus trajectory-free improvement:** the axis reports “Trajectory-aware − trajectory-free future-residual Pearson.” One point is one confirmatory non-anchor gene within one directional target embryo. There are exactly 479 complete paired genes in each fold; no gene was filtered from the frozen paired tables.
- **C — genes improved by trajectory information:** trajectory-aware future-residual Pearson exceeds trajectory-free performance for 397/479 genes (82.9%) in Fold A and 372/479 (77.7%) in Fold B. These are technical gene counts, not biological sample sizes.
- **D — real versus shuffled histories:** full-cohort trajectory-aware MAP-T future-residual Pearson is compared with the frozen 100-permutation null in each fold. Fold A is 0.6205 versus null mean 0.4564 (95% null interval 0.4539–0.4588; empirical P=0.0099); Fold B is 0.6196 versus 0.4731 (0.4705–0.4756; P=0.0099). In both folds, 0/100 shuffled endpoints were at least as large as the real endpoint.
- **E — tracking-overlap cohorts:** full, shared-overlap and tracking-map-disjoint cohorts are shown without changing the frozen memberships. A point is one fold×cohort endpoint, not an embryo replicate.
- **F — overlap-excluded source-refit sensitivity:** the already-frozen source refit after excluding overlap-associated source families is compared with primary trajectory-aware MAP-T on the full target cohort. Fold A retains 954 source ancestor families/1,434 descendant cells and Fold B retains 1,045/1,647; guard stability is 89.7% and 94.5%, respectively.
- **G — lineage-family bootstrap:** frozen full-cohort future-residual Pearson estimates and 95% intervals from 2,000 hierarchical resamples of `TM0_ancestor_family`. Transported early MAP is retained as an explicit row annotated “NA — predicted future residual is identically zero”; no zero-valued point or interval is plotted for its undefined correlation.

## Shuffle procedure

There are exactly 100 prespecified permutations per fold. Molecular inputs, endpoint identities and lineage transport are unchanged. Only the dynamic trajectory-history feature columns are shuffled, within frozen strata formed by division status crossed with displacement-magnitude quartile. The displayed gray marker and interval are the serialized frozen null mean and 2.5th–97.5th percentile interval; no new permutation or threshold was introduced.

The run directory contains the 100 frozen permutation prediction arrays per fold, but it does **not** serialize the 100 individual permutation endpoint values. Recomputing those endpoints from prediction arrays would exceed this presentation-only task. Panel D therefore does not invent a jitter/violin distribution and instead retains the authoritative aggregate marker and interval. The exact annotation “0/100 shuffled ≥ real; empirical P = 0.010” follows the frozen empirical-P definition `(1 + count[shuffled ≥ real]) / (1 + 100)` and the serialized value `0.00990099`.

## Tracking overlap/disjoint definitions and exact cohort sizes

Shared overlap was frozen before model fitting and before target unblinding as **any node intersection across complete TM0–TM230 family node sets**. Disjoint families have no such node intersection. The definitions and memberships were not changed.

| Fold | Cohort | Ancestor families | Descendant cells |
|---|---|---:|---:|
| A | all families | 3,998 | 7,028 |
| A | shared overlap | 2,953 | 5,381 |
| A | tracking-map-disjoint | 1,045 | 1,647 |
| B | all families | 3,907 | 6,807 |
| B | shared overlap | 2,953 | 5,373 |
| B | tracking-map-disjoint | 954 | 1,434 |

Ancestor families are the bootstrap and tracking-cohort units. They are not independent embryos and are not presented as biological replication.

## Interpretation and claim scope

Within these frozen controls, trajectory history adds predictive value beyond the individualized early molecular state: trajectory-aware MAP-T exceeds the trajectory-free predictor for most evaluated genes, the real-history endpoint exceeds the frozen shuffled-history null in both folds, and the result remains positive in shared-overlap, tracking-map-disjoint and overlap-excluded-refit sensitivities.

The permitted framing is **tracking-map-conditioned cross-embryo forecasting**. The lineage maps condition transport and feature construction across embryos; they do not constitute independent longitudinal molecular imaging. No claim of independent live longitudinal molecular measurement is made.

## Supplement tables

`temporal_method_metrics.tsv` contains two explicitly labelled record types: 12 full-cohort method summaries and 958 paired gene×fold trajectory comparisons. `tracking_control_metrics.tsv` contains frozen shuffled-history summaries, tracking cohorts, overlap-excluded refits and full-cohort ancestor-family bootstrap records. Empty fields denote quantities not applicable to that record type; no value was inferred from manuscript text.
