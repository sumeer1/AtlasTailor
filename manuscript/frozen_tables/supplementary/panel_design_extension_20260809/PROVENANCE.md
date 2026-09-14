# Provenance — post-hoc experimental-design extension

This analysis is post hoc, source-only, and did not affect manuscript primary results.

## Exact frozen inputs read

| Path | SHA-256 |
|---|---|
| `data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E1.h5ad` | `8c579f6d1278ec12e17665b6645c892890b29d8e2500cb4a3eecafcafc620884` |
| `data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E2.h5ad` | `e0a88d080cc5c5750c94ff7e2dfb313f257f1a4beaeae03c76d9a466f7f7c2d6` |
| `data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_B_75p_E1.h5ad` | `bf0fb07556f700948999b818de83d156f95dc8ecdd7aaa52bdaca0c7b2392434` |
| `data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_B_75p_E2.h5ad` | `ab33956942d325df380bd51d21f62c132f3a36d0e5a95f0f3cf3b7fed15a7daa` |
| `data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_C_6s_E1_rescaled_z.h5ad` | `a0ea9d01a7e8a59bfda3876b9a3d0f0aa345d50670c718e79fbb7821cf1ea14b` |
| `data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_C_6s_E2_rescaled_z.h5ad` | `49249578ca4b09c3b3ffd5ea7abf7d2646c4d14229ef0a017c1cec8f0a757632` |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv` | `7d02422b8c89f00dacb010e2fb66922f27b4ce07c3c7cd8c804a7880fc47244f` |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E1_to_E2/prediction_manifest.json` | `cc72962f0d0c566caf89cc2d102bab70a4c1a2724e2b320bd458f7ee4e461c0f` |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E2_to_E1/prediction_manifest.json` | `a1613b056f052f21a19d36d9168b28499d3c7c2fbe6734578f961eb3560ecc07` |
| `results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2/prediction_manifest.json` | `52b1c9bd3021655715dccd9b94e630d68b406ee4d40222e7cca52ec811d97658` |
| `results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1/prediction_manifest.json` | `0fff2f37c5fbbe33c2421f7292dc3ab4d65c28de61be083e38b34a33286244c4` |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E1_to_E2/prediction_manifest.json` | `34cdf2dee1d871e9b67d2edfd075c85d26399b2a287755e6443250b26e4e0bfb` |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E2_to_E1/prediction_manifest.json` | `03108cf23b75c8c513bd30de1f1083e09f447284b3aa3ab2e31d303191737dbd` |
| `run_hyperspatial_map_stage1_20260803.py` | `6ece4008d579821799f2eea2ba0abd447b03ea470b90c96090ad6f3f5be49d12` |
| `run_hyperspatial_map_stage1b_20260803.py` | `317c5d642541def2e3384ec914c20067b6630dd535aecd4be880bd1a2a550bb6` |
| `run_wemerfish_temporal_holdout.py` | `1d14d143af98f9196e121be98094b993b4d6ca5c8bde8a61ed447c9d71ee7635` |
| `results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/00_protocol_lock/PROTOCOL_LOCK.json` | `e579bf1f7e55f8223eab3513168cdcdbbff6213e2191c871371516495544bba5` |

## New scripts created

- `analysis/panel_design_extension/run_panel_design_extension.py` — SHA-256 `b91319959a6fed3c002b316fb75618dd045bc49928f916e398798df9141f8af1`

## Isolation confirmation

- No frozen prediction, model, registration, configuration, evaluation table, manuscript, or main/supplement figure was written or regenerated.
- All frozen inputs were read-only and their pre/post SHA-256 values are identical.
- New files are confined to `analysis/panel_design_extension/` and `results_v10/panel_design_extension_20260809/`.
- The new budget-16 panels are experimental-design recommendations only and do not replace the manuscript's frozen panels.
