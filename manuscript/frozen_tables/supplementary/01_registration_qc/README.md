# Supplementary Figure S1 — geometry-only registration QC

Status: **PASS**. The finalized figure contains only displays supported by authoritative frozen geometry inputs and scalar registration outputs. No registration, reconstruction, fitting, tuning, prediction, or expression analysis was performed.

## Authoritative inputs and SHA-256

| Input | SHA-256 | Use |
|---|---|---|
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/09_figures_supplementary/source_data/Supplementary_Figure_3_registration_qc.tsv` | `6f703ee39f13de2873ece535aa84878e0df23cabbc25c0468739b4b08f7263c9` | Frozen input/final Chamfer, reflection selection, and final geometry loss for 18 directional seed runs |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/09_figures_supplementary/source_data/Supplementary_Figure_4_coordinate_qc.tsv` | `706957bbc534d471c5742a8b2b7d6bb7ce5813ba0a0250be99559c1a2dae6b4b` | Frozen coordinate keys, cell counts, canonical scales, and coordinate extents |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/01_data_audit/POST_RUN_COORDINATE_AUDIT.md` | `ce8af29c47845091c949bb2896bae57d102780e7375c9692205f08be05c97a10` | Coordinate-choice audit |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/00_protocol_lock/PROTOCOL_LOCK.json` | `e579bf1f7e55f8223eab3513168cdcdbbff6213e2191c871371516495544bba5` | Frozen six-direction protocol |
| `run_hyperspatial_map_stage1b_20260803.py` | `317c5d642541def2e3384ec914c20067b6630dd535aecd4be880bd1a2a550bb6` | Frozen runner provenance |
| `run_wemerfish_temporal_holdout.py` | `1d14d143af98f9196e121be98094b993b4d6ca5c8bde8a61ed447c9d71ee7635` | Frozen coordinate loading and canonicalization used for panel A |

Authoritative coordinate-file hashes:

| Stage | Embryo | Frozen coordinate choice | SHA-256 |
|---|---|---|---|
| 50% epiboly | E1 | `global_sphere` | `8c579f6d1278ec12e17665b6645c892890b29d8e2500cb4a3eecafcafc620884` |
| 50% epiboly | E2 | `global_sphere` | `e0a88d080cc5c5750c94ff7e2dfb313f257f1a4beaeae03c76d9a466f7f7c2d6` |
| 75% epiboly | E1 | `global_sphere` | `bf0fb07556f700948999b818de83d156f95dc8ecdd7aaa52bdaca0c7b2392434` |
| 75% epiboly | E2 | `global_sphere` | `ab33956942d325df380bd51d21f62c132f3a36d0e5a95f0f3cf3b7fed15a7daa` |
| 6-somite | E1 | raw `spatial` | `a0ea9d01a7e8a59bfda3876b9a3d0f0aa345d50670c718e79fbb7821cf1ea14b` |
| 6-somite | E2 | raw `spatial` | `49249578ca4b09c3b3ffd5ea7abf7d2646c4d14229ef0a017c1cec8f0a757632` |

## Panel definitions

- **A — canonicalized source/target geometry before registration:** frozen input coordinates, median-centered and scaled by the frozen 99th-percentile radial canonicalization, overlaid for all six directions. The display uses axes 1–2 and all cells. No expression was accessed.
- **B — frozen input → final registered Chamfer QC:** exact frozen input and final registered boundary Chamfer values for seeds 42–44, with all six directions kept separate. The logarithmic axis is a display choice only.
- **C — registration stability across seeds 42–44:** **C1** shows exact frozen final registered Chamfer; **C2** shows exact frozen final geometry loss. Reflection selection is `none` in 18/18 frozen runs.

Panels A–C use the primary frozen coordinate choices: `global_sphere` for 50% and 75% epiboly and raw `spatial` for 6-somite. No alternative-coordinate result is shown, and no post-hoc sensitivity was computed.

## Units and interpretation

The biological unit is the embryo. Each directional embryo-pair registration is a geometry-QC unit. Seed is an algorithmic repeat: the seed analysis is sensitivity/QC and is not biological replication. Cells are observations within embryos, not biological replicates.

Across the six frozen directions, final registered Chamfer is lower than input Chamfer. Seeds 42–44 yield closely grouped final registered Chamfer values and final geometry losses, and no reflection is selected. These are geometry-only registration-QC observations.

## Limitations

Intermediate transformed coordinates and deformation fields were not serialized; therefore, they were not reconstructed post hoc. The figure does not show similarity-aligned maps, final registered point clouds, deformation magnitudes, Jacobian/folding diagnostics, or alternative-coordinate results. It does not support claims about local deformation smoothness, invertibility, or absence of folding.
