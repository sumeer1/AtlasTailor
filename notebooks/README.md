# Notebooks

## End-to-end real-data tutorials

| Notebook | Purpose | External input |
|---|---|---|
| `00_run_atlastailor_dlpfc.ipynb` | Prepare two public human DLPFC Visium sections, design 16 source-only measurements, adapt, lock and evaluate | spatialLIBD / Figshare 10.6084/m9.figshare.13623902.v1 |
| `01_run_atlastailor_zebrafish.ipynb` | Run one reciprocal 3D zebrafish weMERFISH reconstruction from source-only panel design through post-lock evaluation | Dryad 10.5061/dryad.j0zpc86v9 |
| `04_run_atlastailor_on_your_data.ipynb` | Apply the same prediction boundary to user-supplied AnnData files | user supplied |

The provider-scale matrices remain outside Git under `data/external/`. Continuous integration syntax-checks these notebooks; full execution is opt-in because it requires external downloads and can be computationally substantial.

## Published-results walkthroughs

The `10`–`12` notebooks read compact frozen evidence bundled under `examples/manuscript_evidence/`. They document reported results but do not fit AtlasTailor. These notebooks are executed in continuous integration.

The exact manuscript-scale workflows and prediction locks are maintained in [AtlasTailor-reproducibility](https://github.com/sumeer1/AtlasTailor-reproducibility).
