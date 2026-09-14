# Public-release checklist

## Technical checks

- [ ] `python scripts/verify_release.py` passes.
- [ ] `pytest` passes in clean Python 3.10 and Python 3.11 environments.
- [ ] All tutorial notebooks execute from top to bottom.
- [ ] `mkdocs build --strict` passes.
- [ ] Wheel and source distribution build successfully.
- [ ] Clean wheel installation imports `hyperspatial` outside the source tree.
- [ ] No credentials, browser profiles, raw human data, or restricted data are committed.
- [ ] Historical absolute paths occur only inside immutable executed scripts/lock records and are documented as provenance, not portable configuration.
- [ ] Large files and externally hosted arrays are represented by accessions and checksums.

## Scientific checks

- [ ] Every manuscript panel maps to a source panel and frozen result table.
- [ ] Executed workflow copies match the recorded source SHA256 values.
- [ ] Prediction-lock and leakage-audit records are included where applicable.
- [ ] Missing exact renderers are disclosed and are not silently reconstructed.
- [ ] Biological replication units are stated for every experiment.
- [ ] IDW residual correlation is never encoded as zero.
- [ ] Manual Inkscape assemblies are absent from this computational release.

## Author-controlled release fields

- [x] Repository owner and URL (`sumeer1/HyperSpatial-MAP`; private during preparation).
- [ ] Complete author names and ORCIDs in `CITATION.cff` and `.zenodo.json`.
- [ ] Final manuscript title and bibliographic citation.
- [ ] Software DOI and data-archive DOI.
- [ ] Copyright-holder approval for BSD-3-Clause.
- [ ] Version number and immutable manuscript tag.
- [ ] Nature Biotechnology code-availability and data-availability statements cross-checked against the accepted manuscript.

Do not create a public archival release until all author-controlled fields are resolved.
