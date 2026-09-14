# Frozen liver configuration

`FROZEN_PROTOCOL.md` and `FROZEN_CONFIG.json` define the configuration frozen after the MASLD analysis and applied without retuning to PSC and alcohol-associated hepatitis (AH).

The generic files `01_ANCHORS.tsv`, `02_PREDICTION_LOCK_MANIFEST.tsv`, and `03_SHA256_PRE_UNLOCK.txt` are the MASLD records. PSC- and AH-specific applied configurations and prediction-lock records carry explicit `PSC_` and `AH_` prefixes. These files are evidence records, not editable defaults.
