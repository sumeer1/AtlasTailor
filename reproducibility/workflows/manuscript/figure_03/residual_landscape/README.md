
# Geometry-corrected residual landscape audit



## Purpose



This module tests whether molecular residuals remaining after geometry-only

registration represent structured specimen-specific biological variation rather

than counting noise, sequencing depth, registration instability or arbitrary

model output.



The biological quantity is calculated from measured target expression:



\[

R^{T}_{g,i}=Y^{T}_{g,i}-B^{T}_{g,i},

\]



where:



- \(Y^{T}_{g,i}\) is measured target expression;

- \(B^{T}_{g,i}\) is the frozen registered-IDW reference prediction;

- \(R^{T}_{g,i}\) is the measured geometry-corrected residual.



Measured residuals are used for biological analysis. HyperSpatial-MAP predictions

are used only to test whether this biological signal can be reconstructed from the

16-gene target panel.



## Frozen input



The audit reads the finalized cross-stage run:



`results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/`



The six frozen reciprocal evaluations are:



- 50% epiboly: E1 → E2 and E2 → E1

- 75% epiboly: E1 → E2 and E2 → E1

- 6-somite: E1 → E2 and E2 → E1



No frozen registrations, anchors, predictions, guards, splits or figures are

modified.



## Reproduction and immutability checks



- 72/72 recomputed frozen residual-spatial-Pearson summary values matched exactly.

- Absolute reproduction difference: 0.0.

- 108/108 frozen prediction files matched their recorded SHA-256 hashes.

- No files in the frozen result directories were modified.

- All generated outputs were written to:



`results_v10/hyperspatial_map_residual_landscape_20260805_1/`



## Main findings



1. **Residual variance exceeds counting-noise expectations**



   Across the evaluated genes and targets, 80.2% of measured residual variance

   exceeded the corresponding Poisson counting-noise floor.



2. **Residuals are spatially organized**



   Residual Moran's I exceeded spatial-permutation expectations, indicating that

   residual expression was organized into spatial fields rather than independent

   cell-level noise.



3. **Residuals align with developmental compartments**



   Between-annotation-class residual variance was approximately 161-fold greater

   than under permuted annotation labels.



   The annotations included germ-layer, cell-type, tissue and cluster labels and

   were not used by the frozen HyperSpatial-MAP protocol.



4. **HyperSpatial-MAP reconstructs compartment-level residual structure**



   Measured and predicted compartment-level residual profiles achieved a median

   Pearson correlation of 0.772.



5. **Depth and registration instability do not explain the result**



   - Adjustment for sequencing depth changed residual recovery by only -0.014.

   - Recovery increased by only 0.048 from the least to the most

     registration-stable quintile.



6. **Residual organization becomes fine-scale at the 6-somite stage**



   At 6-somite, residual organization was stronger at tissue scale

   (\(\eta^2=0.112\)) than at broad germ-layer scale

   (\(\eta^2=0.013\)), indicating sub-germ-layer molecular variation.



## Important qualifications



- Recovery is strongly related to molecular similarity to the anchor panel

  (correlation 0.769).

- Genes in the lowest anchor-similarity quartile still achieved median residual

  recovery of 0.397, showing that recovery is not explained only by direct

  anchor-like behaviour.

- Anchor-only prediction is more effective for some axial genes.

- Reciprocal E1 → E2 and E2 → E1 evaluations use the same embryo pair in reverse

  and are therefore described as **consistency under reciprocal atlas

  construction**, not independent biological replication.

- This audit does not establish developmental canalization.

- Canalization, amplification or persistence must be tested separately using

  measured early and later residuals, lineage-aware analyses and appropriate nulls.



## Module structure



- `config.py`: six-target configuration and write-protection guards

- `io_frozen.py`: read-only frozen-input loaders

- `residual_fields.py`: measured and predicted residual reconstruction

- `structure.py`: variance, Poisson floor and spatial-structure analyses

- `annotation.py`: annotation-associated residual organization

- `confounds.py`: depth, registration and anchor-similarity controls

- `report.py`: machine-readable and human-readable reporting

- `run_audit.py`: audit entry point



## Tests



The module is covered by:



`tests/test_residual_landscape.py`



Twenty tests pass, including frozen-output reproduction, immutability and metric

validation checks.



## Reproduction





```bash

bash results_v10/hyperspatial_map_residual_landscape_20260805_1/REPRODUCE.sh

```



Use the environment and package versions recorded in the audit output directory.



## Supported interpretation



The audit supports the conclusion that:



> Geometry-corrected molecular residuals contain spatially structured,

> developmental-compartment-associated variation that is not explained by

> counting noise, sequencing depth or registration instability, and this

> variation can be partially reconstructed from a sparse molecular panel.



It does not yet support claims about population-level variability, independent

biological replication or developmental canalization.

