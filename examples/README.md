# Published real-data examples

This directory contains compact, frozen evidence from the biological datasets reported in the manuscript. The files are byte-identical copies of the corresponding tables in the companion reproducibility repository.

Executable biological tutorials are kept separately under [`notebooks/`](../notebooks/):

- `00_run_atlastailor_dlpfc.ipynb` — public human DLPFC Visium data;
- `01_run_atlastailor_zebrafish.ipynb` — public 3D zebrafish weMERFISH data;
- `04_run_atlastailor_on_your_data.ipynb` — the same information boundary for user-supplied AnnData files.

The `10`–`12` notebooks are published-results walkthroughs and read the frozen evidence below; they are not presented as model-execution tutorials.

| Directory | Analysis | Full workflow |
|---|---|---|
| `manuscript_evidence/figure_02/` | Reciprocal 3D zebrafish reconstruction and external-method comparisons | `reproducibility/workflows/manuscript/figure_02/` |
| `manuscript_evidence/figure_05/` | Visium HD → Xenium/CosMx cross-platform reconstruction | `reproducibility/workflows/manuscript/figure_05/` |
| `manuscript_evidence/figure_06/` | MASLD, PSC and alcohol-associated hepatitis validation | `reproducibility/workflows/manuscript/figure_06/` |

The workflow paths refer to [AtlasTailor-reproducibility](https://github.com/sumeer1/AtlasTailor-reproducibility). Provider-scale matrices and prediction arrays are not duplicated here; obtain them through [data/DATASETS.tsv](../data/DATASETS.tsv).

Small simulated files needed only for automated software tests are isolated under `tests/fixtures/` and are not presented as scientific examples.
