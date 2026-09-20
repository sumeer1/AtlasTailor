# Tutorials

## End-to-end biological tutorials

These notebooks run the reusable AtlasTailor API from source-only panel design through prediction locking and post-lock evaluation.

| Tutorial | Biological data | External requirement |
|---|---|---|
| [Human DLPFC Visium](https://github.com/sumeer1/AtlasTailor/blob/main/notebooks/00_run_atlastailor_dlpfc.ipynb) | Sections 151507 → 151508 | Official spatialLIBD object and 10x matrices |
| [Zebrafish weMERFISH](https://github.com/sumeer1/AtlasTailor/blob/main/notebooks/01_run_atlastailor_zebrafish.ipynb) | 50%-epiboly E1 → E2 | Two Dryad H5AD files |
| [Bring your own data](https://github.com/sumeer1/AtlasTailor/blob/main/notebooks/04_run_atlastailor_on_your_data.ipynb) | User-supplied reference and target | Prepared AnnData files |

Provider-scale matrices are not committed to GitHub. The notebooks identify authoritative records, verify frozen hashes where available and write new timestamped output directories. Continuous integration syntax-checks these notebooks; full execution is opt-in because it requires external data and can be computationally substantial.

## Published-results walkthroughs

The `10`–`12` notebooks inspect compact frozen evidence for the zebrafish, cross-platform and liver-disease results. They do not fit AtlasTailor and are clearly separated from the execution tutorials. All three execute in continuous integration.

## Exact manuscript workflows

The tutorials exercise the reusable interface. They are not substitutes for the exact manuscript computations. Executed workflow snapshots, prediction-lock manifests, complete frozen tables and generated source panels live in [AtlasTailor-reproducibility](https://github.com/sumeer1/AtlasTailor-reproducibility).
