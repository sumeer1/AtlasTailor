# AtlasTailor

**Sparse measurements tailor a deeply measured spatial atlas to an individual specimen.**

[![CI](https://github.com/sumeer1/AtlasTailor/actions/workflows/ci.yml/badge.svg)](https://github.com/sumeer1/AtlasTailor/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)

AtlasTailor reconstructs an individual target’s spatial molecular state by combining a deeply measured reference atlas with target geometry and a small, prespecified panel of target genes:

```text
registered atlas prior + sparse target adaptation = individualized molecular reconstruction
```

Unlike conventional within-sample imputation, the reference and target may be different specimens, developmental stages, tissue sections, disease states, or spatial technologies. Geometry-only registration establishes an atlas prior; measured target genes inform an atlas-relative correction; and a source-validation guard retains registered 12-nearest-neighbour inverse-distance weighting when adaptation is unsupported.

## Framework overview

[![AtlasTailor framework and validation design](docs/assets/Figure_1_preview.png)](docs/assets/Figure_1.pdf)

[Open the publication-quality vector PDF](docs/assets/Figure_1.pdf)

## Installation

```bash
git clone https://github.com/sumeer1/AtlasTailor.git
cd AtlasTailor
conda env create -f environment.yml
conda activate atlas-tailor
python -m pip install -e .
```

Python 3.10–3.11 is supported. The import namespace and command-line executable remain `hyperspatial` for compatibility with the frozen analyses.

## Published real-data examples

The public notebooks below use frozen results from the biological datasets reported in the manuscript; they do not use simulated biological examples.

| Analysis | Biological system | Public record | Notebook |
|---|---|---|---|
| Reciprocal atlas adaptation | Zebrafish embryogenesis, 3D weMERFISH | [Dryad](https://doi.org/10.5061/dryad.j0zpc86v9) | [Figure 2 benchmark](notebooks/01_figure2_benchmark.ipynb) |
| Cross-platform atlas transfer | Human COAD and ovarian cancer; Visium HD → Xenium/CosMx | [SPATCH](https://spatch.pku-genomics.org) | [Cross-platform benchmark](notebooks/02_cross_platform_benchmark.ipynb) |
| Multi-disease atlas adaptation | Human MASLD, PSC and alcohol-associated hepatitis | [Dataset records](data/DATASETS.tsv) | [Liver disease validation](notebooks/03_liver_disease_validation.ipynb) |

The exact executed workflows, prediction locks, frozen configurations, full evidence tables, source-panel SVGs, and checksums are maintained in [AtlasTailor-reproducibility](https://github.com/sumeer1/AtlasTailor-reproducibility). See [examples/README.md](examples/README.md) for the mapping between datasets, notebooks, and executable workflows.

## Running AtlasTailor on a new dataset

AtlasTailor accepts reference and target AnnData objects with spatial coordinates in `obsm["spatial"]`. Target input supplied to prediction must contain only the declared measured panel.

```bash
hyperspatial design \
  --reference reference.h5ad \
  --budget 16 \
  --out panel_run

hyperspatial adapt \
  --reference reference.h5ad \
  --target target_anchors_only.h5ad \
  --panel panel_run/recommended_panel.tsv \
  --config config.yaml \
  --out adaptation_run
```

See the [input contract](docs/data_formats.md), [preprocessing boundary](preprocessing/README.md), [adaptation workflow](docs/adapt.md), and [CLI reference](docs/cli.md).

## Repository structure

```text
src/hyperspatial/      installable implementation and CLI
notebooks/             real manuscript-data walkthroughs
examples/              frozen real-data evidence used by notebooks
data/                  dataset accessions and external-data policy
preprocessing/         input preparation and information boundaries
tests/                 software tests and internal CI fixtures
docs/                  methods, API and scientific guardrails
provenance/            software-release manifest and validation report
```

## Reproducibility

This repository is deliberately focused on reusable software. Manuscript-scale workflows and evidence live in the companion reproducibility repository so that users do not need to clone the full research record to install AtlasTailor.

- [Software/reproducibility boundary](docs/repository_scope.md)
- [Dataset catalogue](data/DATASETS.tsv)
- [Figure provenance](docs/figure_provenance.md)
- [Release validation](docs/validation.md)

## Scientific safeguards

- Target molecular expression is excluded from geometry registration.
- Anchor selection uses source information only.
- Target non-anchor expression remains withheld until predictions are serialized and locked.
- Predictor selection, calibration, and fallback use source validation rather than target outcomes.
- Registered IDW predicts no atlas-relative correction; its residual correlation is undefined, not zero.
- Spots and genes are evaluation units, not independent biological replicates.

See [scientific guardrails](docs/scientific_guardrails.md) and [statistical units](docs/statistical_units.md).

## Citation and licence

Citation metadata are provided in [CITATION.cff](CITATION.cff). AtlasTailor is distributed under the [BSD 3-Clause License](LICENSE). External datasets retain their original licences and access terms.
