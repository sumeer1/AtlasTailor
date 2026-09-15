# Preprocessing boundary

HyperSpatial-MAP expects a reference and sparse target in the documented AnnData contract; see [data formats](../docs/data_formats.md). Reusable checks belong in the package, while dataset-specific conversion and QC remain part of the manuscript reproducibility record.

The general sequence is:

1. preserve the provider archive and verify its checksum;
2. assign explicit specimen, patient, section, and platform identifiers;
3. retain raw counts in an AnnData layer where available;
4. apply the documented expression transformation without target-outcome tuning;
5. place spatial coordinates in `obsm["spatial"]`;
6. intersect source and target gene identifiers deterministically;
7. select anchors from source information only;
8. export the target-anchor view separately from withheld target expression.

Provider-specific scripts are not rewritten as generic software because doing so would obscure the code actually executed. Their byte-preserved snapshots, configurations, accessions, and run order are in the companion reproducibility repository.
