# Supplementary Figure S6 — residual-biology robustness across reciprocal target embryos

Status: **PASS**. All eight panels use only serialized outputs from the immutable residual-landscape audit `results_v10/hyperspatial_map_residual_landscape_20260805_135217`. This redesign makes S6 a target-stratified/full-distribution robustness companion to authoritative main Figure 5. No prediction, residual field, registration, anchor panel, model, hyperparameter, biological example, or primary Figure 5 result was recomputed or changed.

## Authoritative inputs and SHA-256

| Frozen input | SHA-256 | Use |
|---|---|---|
| `00_provenance/AUDIT_PROVENANCE.json` | `11d6d733495e0235b48b801294058fffe7ab8763c460b8255447130f7a286813` | Immutable audit scope, targets and reference seed |
| `00_provenance/FROZEN_PREDICTION_HASHES.tsv` | `835efa1c75729b292c8614e2dbe0677863e5f3c10f0fb0cc08d1edce1577da98` | 108/108 frozen-prediction hash verification |
| `00_provenance/RESIDUAL_REPRODUCTION_CHECK.tsv` | `18d429518286fcfd4a6a05eb0542f0fa383d8312b5a7e36d561e8c8a96a6a049` | 72/72 frozen residual endpoints reproduced within `1e-05` by the completed audit |
| `02_structure/RESIDUAL_VARIANCE_DECOMPOSITION.tsv` | `1cf512eea14df95955d7b196fee43c8f9dbcc90b1358ba1e6a5661499a37e4e2` | Frozen per-gene Poisson and Moran statistics |
| `03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv` | `868e3cbfcabaa60e91c9666aa268185251a6bea16696719a996cc63dd40790e2` | Frozen per-gene annotation η² and compartment-profile recovery |
| `03_annotation/RESIDUAL_RECOVERY_BY_COMPARTMENT.tsv` | `57c9410b54949d0c49a458262ab93fc0215d74373ebe80d0a417db4025ab3446` | Frozen compartment-level recovery records included in the supplement table |
| `04_confounds/DEPTH_PARTIAL_CORRELATION.tsv` | `1009c2209a97fef4cb46939e1c081bdeea4859bf42362113c484a2f4bf704594` | Frozen raw/depth-adjusted recovery control |
| `04_confounds/ANCHOR_PROPAGATION_BY_GENE.tsv` | `a1ab511297bcec0e2a3c4bfc36b40c3c7843a809595c857396da121955d189ad` | Frozen source anchor similarity and target residual recovery |
| `05_report/RESIDUAL_LANDSCAPE_AUDIT.md` | `bd7091154eb3c15645253c03e6c8b81bbc3e26a5ea26582d1be2d4b3f22b5017` | Authoritative rounded summaries and interpretation limits |
| `07_source_data/figure5/panel_c_moran_vs_null.csv` | `dc458d999d48824f8467af8ebdd7be6505163dc169c9e44c9682e51c81aa8c4e` | Six frozen Moran summaries |
| `07_source_data/figure5/panel_d_annotation_eta_squared.csv` | `558db2c821bf4cdd9ee0d18f1e8f20a36440c16d347d272a784c1288bbce7672` | Frozen target×annotation medians |
| `07_source_data/figure5/panel_f_6somite_eta_by_granularity.csv` | `541b76194bcff5ff341867b261273396ebf9660e8290d0efe78243c309cf7e72` | Authoritative main-figure comparison source; not replotted in redesigned S6 |
| `07_source_data/figure5/panel_g1_depth_adjustment.csv` | `9e014ee3160f3e1dc754a0f9590617b00b53d2cd9b624ca452fa9f5f19701812` | Six frozen primary-method depth summaries |
| `07_source_data/figure5/panel_g2_registration_strata.csv` | `baf177cf28e3c7c8b6ce1584852912025ac567276a966f75879339f65a5975e5` | Thirty frozen target×transfer-disagreement-quintile summaries |
| `07_source_data/figure5/panel_h_anchor_similarity.csv` | `6d2521be5ac78dc503781d12be977b171ef385d3a3c3abe64f4840781d88426f` | 2,742 frozen non-anchor gene×target records |
| `07_source_data/figure5/panel_h_quartile_summary.csv` | `0a6ad0c26ca6a1c2d0958fc1d341cba45059ac3edc537915f51798846deaf4a0` | Frozen pooled anchor-similarity quartiles |

All paths in this table are relative to `results_v10/hyperspatial_map_residual_landscape_20260805_135217`. The authoritative audit code/config hashes are `analysis/residual_landscape/run_audit.py` = `dc85fa18cc3a9a862944c287d1fab86e8a2296fd49c155dc83c9fc6ada2206bd` and `analysis/residual_landscape/config.py` = `5a7d5e14a763ba8ac07bdf6a0c20c24af05a4deaf4ec5460fe188c4f7d2413ce`. The audit was not invoked while generating S6.

## Residual and eligibility

The measured target residual is measured target expression minus frozen registered 12-NN IDW atlas transfer on the frozen log1p depth-normalized scale, averaged over registration seeds 42–44 by the completed audit. Confirmatory recovery uses the frozen source-defined predictable set after excluding all 16 anchor genes.

There are exactly **2,742 eligible non-anchor gene×directional-target records**:

| Target evaluation | n genes |
|---|---:|
| 50% E1→E2 | 477 |
| 50% E2→E1 | 478 |
| 75% E1→E2 | 465 |
| 75% E2→E1 | 467 |
| 6-somite E1→E2 | 441 |
| 6-somite E2→E1 | 414 |

## Exact metric definitions

### Poisson counting-noise reference

For each target cell and eligible gene, the audit drew an independent Poisson realization at the observed count rate, retained the observed per-cell normalization factor, transformed observed and resampled values with log1p, and defined the single-measurement noise variance as one half of their mean squared difference. The structured-above-Poisson fraction is

`max(residual variance − counting-noise variance, 0) / residual variance`.

All 2,742 records lie above the variance reference; the median fraction of residual variance beyond that reference is `0.801683` (80.2%). This is a share of variance, not a percentage of genes. Panel A displays the serialized structured-fraction distribution and above-reference fraction separately for every target, replacing the pooled variance-versus-noise scatter used in main Figure 5. It does **not** establish that all residual variance is biological. The Poisson model omits biological and technical overdispersion; using observed counts as the rate also makes the reference itself noisy.

### Moran’s I and permutation null

Moran’s I is calculated per eligible gene using uniform row-normalized 12-nearest-neighbour weights in the exact frozen coordinate field, excluding self-neighbours. The completed audit used exactly **20 cell-label permutations per directional target** with frozen random seed `20260805`; each permutation applies one shuffled cell order to the residual matrix and recalculates Moran’s I for every eligible gene. No new permutation was generated for S6. For each gene, the null mean is the arithmetic mean of its 20 permuted Moran’s I values. **Moran permutation z = (observed I − mean null I) / null SD, using 20 frozen cell-label permutations per target.** The null SD is the sample SD (`ddof=1`) across those same 20 values. Panel B displays the complete frozen per-gene z distribution separately for each target rather than repeating main Figure 5’s Moran-I-minus-null view. The smallest target-median z is `31.699` (approximately 32).

### Annotation-associated η²

Author-supplied labels are `germlayer`, `type` (shown as cell type), `tissue`, and `clusters` (shown as cluster). Classes with fewer than 20 cells are excluded; an annotation is usable only if at least two classes remain. For each eligible gene, η² is between-class sum of squares divided by total sum of squares. The completed audit compared it with the mean of 20 label permutations. Across 8,126 gene×annotation×target rows, the ratio of the global median observed η² to the global median permuted-label η² is `160.960456`.

The frozen MAP fitting and anchor selection never accessed these target annotations (`target_annotations_accessed: false`). They are retrospective descriptive yardsticks, not fitting inputs and not ontology-level proof of a biological programme.

In Panel C, **“—” means annotation unavailable or ineligible for that target/stage; not zero.** No unavailable annotation value is inferred.

### Compartment-profile recovery

For each eligible gene and usable annotation, the audit forms vectors of per-class mean measured residual and per-class mean MAP-predicted residual, then calculates their Pearson correlation. Panel D shows all **8,126 gene×annotation×directional-target** records: 954, 956, 1,395, 1,401, 1,764 and 1,656 in the six displayed evaluations. The pooled median is `r = 0.772255`.

### Depth adjustment

Per-cell log sequencing depth is centered, and its least-squares projection is removed separately from the measured and MAP-predicted residual field for every gene. Recovery is then recalculated as their Pearson correlation and summarized by the per-target gene median. The displayed median change across the six target evaluations is depth-adjusted minus raw `Δr = −0.013662` (approximately −0.014).

### Registered-transfer disagreement

The authoritative metric is **registered-transfer disagreement**. For each cell it is the mean absolute disagreement, across eligible genes, between the frozen registered 12-NN IDW transfer and frozen registered 1-NN transfer at reference seed 42. The frozen script computes ascending quantile edges of this value and assigns bins from `b + 1`; mean registered-transfer disagreement increases monotonically from Q1 to Q5 in every target. Consequently:

- **Q1 = low registered-transfer disagreement**;
- **Q5 = high registered-transfer disagreement**.

Panel G is a **transfer-disagreement sensitivity** summary and shows one `Q5 − Q1` recovery contrast per target rather than repeating the five-quintile curves in main Figure 5. The six frozen-table contrasts are 0.027, 0.052, 0.028, 0.048, 0.082 and 0.049 in displayed target order; their median is `0.048410` (shown as 0.048). The values and signs come directly from the frozen Q5 and Q1 rows and were not inferred from the recovery trend.

### Source anchor similarity

For each eligible non-anchor gene, source-only similarity is the maximum absolute Pearson correlation between that gene and any member of the exact frozen 16-anchor panel, calculated in the source embryo. Recovery is the gene’s target MAP residual Pearson correlation. Across all 2,742 non-anchor gene×target records, their Pearson association is `r = 0.768707`. The frozen pooled lowest-similarity quartile contains 686 records and has median recovery `0.396558` (approximately 0.397). Panel H stratifies recovery by the already-frozen pooled quartile labels within each target and shows target-specific medians/counts, replacing main Figure 5’s gene-level scatter. No target non-anchor expression enters the similarity definition or panel selection.

## Panels

- **A — Poisson-reference result by target:** per-gene structured-fraction distributions and the fraction of records above the Poisson variance reference for each of the six directional targets. This is not the pooled main-Figure-5 scatter.
- **B — full Moran permutation evidence by target:** complete serialized per-gene Moran permutation-z distributions, with each target median shown. The panel defines the 20-permutation null and z calculation and does not reproduce the main-figure Moran-I-minus-null distribution.
- **C — target×annotation η² evidence:** paired observed/permuted median η² heatmap for every available target×germ-layer/cell-type/tissue/cluster combination. Gray cells indicate that the annotation level was not available for that target; they are not inferred.
- **D — compartment-profile recovery across all targets:** the requested retained panel, showing all 8,126 frozen gene×annotation×target recovery values separately for each directional target.
- **E — recovery distributions by annotation resolution:** full compartment-profile recovery distributions for germ layer, cell type, tissue and cluster. Colored points are medians for each directional target in which the annotation exists. This replaces the pooled 6-somite anatomical-scale plot.
- **F — depth-adjustment effect by target:** the six frozen `adjusted − raw` target summaries are shown directly as Δr bars, replacing raw→adjusted connecting lines.
- **G — transfer-disagreement sensitivity by target:** one Q5-high-disagreement minus Q1-low-disagreement recovery contrast per target, replacing the five-quintile curves.
- **H — recovery by source-anchor-similarity quartile:** target×frozen-pooled-quartile heatmap of median residual recovery and technical record count, replacing the gene-level anchor-similarity scatter. The authoritative pooled Q1 median 0.397 and overall gene-level association 0.769 remain annotated.

No target-selected example is displayed. No panel uses raw measured-expression arrays or reruns a primary residual statistic; S6 reads the completed audit’s serialized per-record or summary fields. Target-specific medians/counts in E and H and Q5−Q1 contrasts in G are display-only stratifications of those frozen values.

## Primary versus robustness status and claim scope

The residual-landscape audit is a retrospective characterization of already-frozen predictions. Main Figure 5 carries the headline biological evidence. S6 provides the target-stratified/full-distribution support behind those headlines without altering the main figure or analysis. Panels F–H are robustness/interpretability controls. None is used to reselect coordinates, anchors, genes, models, thresholds, or claims.

The retained genuinely complementary views are A–H as defined above. No requested panel was omitted. Panel D is intentionally retained at the user’s direction; although it uses the same frozen endpoint family as main Figure 5, it presents all six target-specific distributions rather than the pooled main-figure histogram. No panel duplicates a main-Figure-5 plotting design.

The supported wording is **“structured target-specific molecular variation beyond registered anatomy.”** The results do not show that all residual variance is biological and do not establish population-level variability, causality, independent replication, or prospective confirmation.

## Biological and statistical units

The directional target embryo/volume is the biological and statistical unit available in the frozen experiment. Genes, gene×target, gene×annotation×target, compartments, cells and transfer-disagreement strata are technical evaluation units, not biological replicates. Reciprocal E1→E2 and E2→E1 atlas constructions reuse the specimen pair structure and are not independent biological replicates.

## Supplement table

`full_residual_biology_metrics.tsv` is a long union of the frozen gene-structure, annotation, compartment, depth, transfer-disagreement, anchor-similarity, six-somite and reported-summary records. `record_type`, `evaluation_unit`, `analysis_status`, and `source_table` identify the meaning and provenance of every row. Blank fields are not applicable to that record type; no missing value was inferred from manuscript text.
