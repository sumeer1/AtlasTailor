# Manuscript reproducibility

The files under `workflows/` are read-only snapshots of scripts executed in the research tree. They are copied without scientific edits and hashed in `provenance/EXECUTED_SCRIPT_PROVENANCE.tsv`.

`ORIGINAL_RUN_ORIGINS.tsv` records the immutable project directories from which the bundled workflow snapshots and generated source assets were copied. Historical absolute paths inside lock manifests are intentionally retained as provenance; portable public execution should use the package examples and repository-relative configuration files.

## Start here

```bash
conda env create -f environment-manuscript.yml
conda activate hyperspatial-map-manuscript
python -m pip install -e .
python scripts/verify_release.py
python scripts/list_manuscript_workflows.py
```

Read `RUN_ORDER.md` before attempting a manuscript-scale rerun.

## Fast path

The fast path does not retrain HyperSpatial-MAP. It verifies exact checksums and lets readers inspect the frozen tables and source-panel SVGs:

```bash
python scripts/verify_release.py
jupyter lab notebooks/06_manuscript_evidence_walkthrough.ipynb
```

## Full path

Full reproduction requires provider data that are intentionally excluded from GitHub. Download them according to `data/DATASETS.tsv`, verify provider checksums, and recreate the expected research-root layout in an external workspace. Run workflows only into new output directories.

The preserved scripts include historical research-relative paths. They are provenance snapshots, not silently rewritten portable substitutes. This avoids changing executed scientific code. `WORKFLOW_INDEX.tsv` records the entry point, required data, output class, and whether an exact final panel renderer was serialized.

## Information-boundary requirement

For masked-target experiments, target non-anchor expression must remain inaccessible through fitting, anchor selection, registration, calibration, guard selection, and prediction serialization. Evaluation follows successful hash verification of the prediction lock.

## Missing renderer policy

Some historical panels have exact frozen SVGs and source tables but no identifiable final plotting script. These are labelled `renderer_not_serialized`. The release verifies their artifact hashes and does not invent a replacement renderer.
