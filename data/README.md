# Data access

Only deterministic synthetic fixtures are bundled under `examples/`. Manuscript-scale spatial expression matrices, human molecular data, and prediction arrays are excluded from GitHub.

`DATASETS.tsv` records the authoritative public study or repository for each manuscript experiment. Dataset licenses remain with their providers. Users must review current provider terms before downloading or redistributing data.

## Verification

1. Download from the official repository or accession.
2. Preserve the provider archive unchanged.
3. Verify the recorded checksum where one is available.
4. Extract into an external data root, not the Git repository.
5. Run scientific workflows into a new output root.

The MASLD `Visium.zip` archive, for example, was verified at exactly 336,420,009 bytes with MD5 `2962d8463da108f9fb9ef951d9204dbd`; the archive itself is not redistributed here.

Some workflows depend on publication-linked processed data whose upstream construction is not fully reproducible from raw reads. Such limitations remain documented in the original audits and must not be obscured.

