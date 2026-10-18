# AtlasTailor

**Sparse measurements tailor a deeply measured spatial atlas to an individual specimen.**

[![CI](https://github.com/sumeer1/AtlasTailor/actions/workflows/ci.yml/badge.svg)](https://github.com/sumeer1/AtlasTailor/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)

AtlasTailor reconstructs an individual target’s spatial molecular state by combining a deeply measured reference atlas with target geometry and a small, prespecified panel of target genes:

```text
registered atlas prior + sparse target adaptation = individualized molecular reconstruction
```

## Framework overview

[![AtlasTailor framework and validation design](docs/assets/Figure_1_preview.png)](docs/assets/Figure_1.pdf)

## Installation

```bash
git clone https://github.com/sumeer1/AtlasTailor.git
cd AtlasTailor
conda env create -f environment.yml
conda activate atlas-tailor
python -m pip install -e .
```

Python 3.10–3.11 is supported. The primary command-line executable is `atlastailor`; the Python import namespace and legacy `hyperspatial` command remain available for backward compatibility.

## Minimal Python workflow

This example designs a 16-gene panel from a reference atlas, reconstructs a target specimen and optionally evaluates the reconstruction when complete target measurements are available.

```python
from hyperspatial import adapt, design_panel, validate
from hyperspatial.io import read_reference, read_target, read_truth

# Deeply measured reference atlas
reference = read_reference("reference.h5ad", coordinate_key="spatial")

# Select the measurement panel using the reference only
panel = design_panel(reference, budget=16, out="runs/panel")

# Load target geometry and the selected measurements
target = read_target(
    "target.h5ad",
    measured_genes=panel.genes,
    coordinate_key="spatial",
)

# Geometry-only registration, registered atlas transfer and sparse adaptation
result = adapt(
    reference,
    target,
    measured_genes=panel.genes,
    out="runs/adaptation",
)

# Optional evaluation with complete target measurements
truth, _ = read_truth("target.h5ad", result.genes, coordinate_key="spatial")
metrics = validate(result, truth, out="runs/adaptation/gene_metrics.tsv")
```

The [documentation](https://atlastailor.readthedocs.io) describes the pipeline, input formats and API.

## Run AtlasTailor on real data

The end-to-end tutorials execute AtlasTailor on public biological datasets and cover data preparation, 16-gene panel design, spatial registration, reconstruction and evaluation.

| Tutorial | Dataset | Public record | Notebook |
|---|---|---|---|
| 3D reciprocal reconstruction | Zebrafish 50%-epiboly weMERFISH, E1 → E2 | [Dryad](https://doi.org/10.5061/dryad.j0zpc86v9) | [Run the zebrafish tutorial](notebooks/01_run_atlastailor_zebrafish.ipynb) |
| 2D serial-section reconstruction | Human DLPFC Visium, 151507 → 151508 | [spatialLIBD / Figshare](https://doi.org/10.6084/m9.figshare.13623902.v1) | [Run the DLPFC tutorial](notebooks/00_run_atlastailor_dlpfc.ipynb) |
| User dataset template | Prepared reference and target AnnData files | — | [Run AtlasTailor on your data](notebooks/04_run_atlastailor_on_your_data.ipynb) |

The public datasets are not redistributed in this repository. Each tutorial identifies the authoritative record, verifies published file checksums where available and writes results to a user-selected run directory.

## Command-line workflow

AtlasTailor accepts reference and target AnnData objects with spatial coordinates in `obsm["spatial"]`. The target can contain the selected measurement panel or a complete matrix; the adaptation workflow reads the selected panel.

```bash
atlastailor design \
  --reference reference.h5ad \
  --budget 16 \
  --out panel_run

atlastailor adapt \
  --reference reference.h5ad \
  --target target_anchors_only.h5ad \
  --panel panel_run/recommended_panel.tsv \
  --config config.yaml \
  --out adaptation_run
```

See the [quickstart](docs/quickstart.md), [input contract](docs/data_formats.md), [preprocessing boundary](preprocessing/README.md), [adaptation workflow](docs/adapt.md), and [CLI reference](docs/cli.md).

## Repository structure

```text
src/hyperspatial/      installable implementation and CLI
notebooks/             end-to-end real-data tutorials and worked examples
examples/              compact example tables used by notebooks
data/                  dataset accessions and external-data policy
preprocessing/         input preparation utilities
tests/                 software tests
docs/                  methods, tutorials and API reference
provenance/            software-release manifest and validation report
```

## Citation and licence

Citation metadata are provided in [CITATION.cff](CITATION.cff). AtlasTailor is distributed under the [BSD 3-Clause License](LICENSE). External datasets retain their original licences and access terms.
