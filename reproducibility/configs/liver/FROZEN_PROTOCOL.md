# Frozen human-liver multi-disease HyperSpatial-MAP protocol

## Freeze declaration

This protocol was frozen on 25 August 2026 after completion of the MASLD/fibrosis analysis in `PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD/`. MASLD is the first **primary** human-liver disease validation and defines the post-MASLD configuration for every later disease evaluation.

No MASLD prediction, registration, anchor, operator assignment, guard decision, metric, or outcome was recomputed for this freeze. The exact source model and implementation inputs are identified by SHA256 in `SHA256_MANIFEST.txt`.

## Presentation hierarchy

- **Primary validation:** MASLD/fibrosis.
- **Supplementary stress test:** acute liver failure (ALF), Decision B, preserved unchanged.
- ALF is excluded from the main multi-disease figure and from the primary aggregate efficacy claim.
- ALF remains fully reported as a transparent supplementary stress test: RMSE improved in 4/4 patients and residual recovery was positive in 4/4, while median spatial Pearson decreased in 4/4.
- This is a presentation change only. ALF files are neither deleted nor rerun.

## Frozen scientific question

For each later disease, test whether a healthy live-donor liver atlas plus the target geometry and the same 16 measured genes can reconstruct the patient's hidden disease molecular state beyond geometry-only registered atlas transfer.

The only manuscript-facing comparison is:

`registered 12-NN IDW` versus `full HyperSpatial-MAP`.

Anchor-only decoding remains an internal candidate operator. It is not a baseline, comparator, or separate manuscript-facing method. Registered IDW predicts no atlas-relative correction; its residual Pearson is therefore undefined and must not be encoded or plotted as zero.

## Frozen reference and information boundary

The reference is the healthy live-donor human liver atlas deposited at Zenodo `10.5281/zenodo.17735506`, using donors M1–M8. Source validation donors are M2 and M7.

Before prediction lock, a disease target may contribute only:

1. spatial coordinates; and
2. counts for the 16 frozen anchors.

All non-anchor target expression, disease-program scores, severity, pathology annotations, anatomical labels, and reconstruction outcomes remain unavailable for registration, fitting, operator selection, calibration, or guard decisions. They may be opened only after every prediction artifact has been locked and its hash verified.

## Frozen 16-gene panel

The exact panel, in frozen order, is:

1. IGFBP1 (`ENSG00000146678`)
2. VIM (`ENSG00000026025`)
3. C7 (`ENSG00000112936`)
4. ORM1 (`ENSG00000229314`)
5. NNMT (`ENSG00000166741`)
6. MT1M (`ENSG00000205364`)
7. FOS (`ENSG00000170345`)
8. CYP1A2 (`ENSG00000140505`)
9. MYL9 (`ENSG00000101335`)
10. JUND (`ENSG00000130522`)
11. IGKC (`ENSG00000211592`)
12. SERPINE1 (`ENSG00000106366`)
13. CD74 (`ENSG00000019582`)
14. GSTA2 (`ENSG00000244067`)
15. ADIRF (`ENSG00000148671`)
16. SDS (`ENSG00000135094`)

These identities—not merely the panel size—are frozen. A later target missing any anchor is not eligible for an exact-protocol evaluation. Do not substitute genes or rerun source-only panel selection to rescue a branch.

The historical selection was source-only: eligible genes had healthy-source prevalence 0.05–0.98, variance above `1e-5`, and no `MT-`, `RPL`, or `RPS` prefix. The top 512 prevalence/variance candidates entered greedy squared-correlation coverage selection with row-subsampling seed 42116. This rule is documented for reproducibility but is not rerun for later diseases.

## Frozen processing and model

Source counts are scaled per spot to the median healthy-source depth and transformed with `log1p`. Target depth is predicted from the 16 anchor counts using the frozen source-trained depth model. No target non-anchor normalization or post-hoc calibration is allowed.

The registered prior is 12-nearest-neighbor inverse-distance weighting with squared inverse-distance weights and a distance floor of `1e-5`.

HyperSpatial-MAP uses:

- 16 raw anchor residuals relative to the registered prior;
- 12-NN local and 32-NN broader residual summaries;
- reduced-rank ridge models with rank 16 and ridge 10.0;
- correction gain 1.0;
- global, local, anchor-decoder fusion, and registered-IDW fallback candidates;
- fusion weights 0, 0.25, 0.5, 0.75, and 1.0.

For each gene, source validation first restricts candidates to those within 1.05 times the best MSE, then selects the candidate with the highest spatial Pearson. The correction passes the frozen guard only if its MSE is lower than IDW and its Pearson exceeds IDW by more than 0.03. Otherwise, the prediction falls back to registered IDW. Final log1p predictions are clipped at zero.

## Frozen geometry registration

Registration uses coordinates only. Coordinates are median-centered and divided by the 0.99 quantile of radial distance. Similarity ICP is run for 30 iterations under no reflection, x reflection, y reflection, and xy reflection; the lowest-Chamfer solution initializes the non-rigid stationary-velocity field.

Non-rigid parameters are fixed at 30 epochs, learning rate 0.002, batch size 768, Chamfer keep fraction 0.85, cycle penalty 0.2, smoothness penalty 0.02, fold penalty 0.2, AdamW weight decay `1e-6`, and gradient clipping at 5.0. Registration seeds are 42, 43, and 44; seed 42 is primary. No more than 5,000 source or target points are used for field fitting.

Target support is restricted to units whose nearest registered-source distance is no greater than twice the median target nearest-neighbor spacing. At least 15 supported target units are required. No molecular, severity, pathology, or anatomical target information may optimize registration.

## Frozen evaluation

Post-lock QC retains target units with more than 10 detected features and mitochondrial percentage below 10%, and hidden genes detected in more than three spots per target array. These rules cannot be changed after viewing performance.

The frozen metrics are:

- median gene-wise spatial Pearson;
- pooled log1p RMSE;
- median gene-wise atlas-relative residual Pearson;
- pooled target-unit-by-gene atlas-relative residual Pearson;
- fraction of hidden genes for which MAP spatial Pearson exceeds IDW;
- fraction for which MAP RMSE is below IDW; and
- fraction with positive MAP residual Pearson.

Measured residual is target truth minus the registered-IDW prior. Predicted residual is MAP minus that same prior. Patient is the biological unit; spots, genes, sections, and multiple biopsies are not biological replicates. Patient bootstrap intervals use 10,000 resamples, seed 260825, and the 2.5th and 97.5th percentiles of the patient-balanced mean.

## Required lock for every later disease

Before hidden target expression is opened, hash and freeze the geometry registration, target support mask, 16-anchor object, hidden-gene list, registered-IDW prediction, MAP prediction, internal operator assignments, guard/fallback assignments, predicted target depth, source model, and configuration. A failed hash stops that target. No outcome-based exclusion or post-outcome retuning is permitted.

## Immutable downstream rule

PSC, alcohol-associated hepatitis, HCC, or any other later liver disease must use `FROZEN_CONFIG.json` and the source model whose SHA256 is recorded here. Any scientifically necessary deviation must be labeled a new protocol and cannot be pooled with or represented as an exact application of this frozen multi-disease validation.

