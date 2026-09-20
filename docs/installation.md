# Installation

AtlasTailor supports Python 3.10 and 3.11 on Linux and macOS. CPU execution is the default. A CUDA-enabled PyTorch installation may be used for non-rigid registration, but GPU execution is not required for the package interface or tutorials.

## Local CPU installation

```bash
git clone https://github.com/sumeer1/AtlasTailor.git
cd AtlasTailor
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
hyperspatial --version
```

Use `pip install -e '.[notebooks]'` for tutorials, `.[docs]` for documentation and `.[test]` for development. PyTorch controls GPU support; CPU execution is the default.

## Conda

```bash
conda env create -f environment.yml
conda activate atlas-tailor
```

The non-rigid registration stage is the most computationally intensive component. Public manuscript workflows document the compute and data requirements for each real dataset; compact files under `tests/fixtures/` are used only for automated software validation.

## Build the documentation locally

```bash
python -m pip install -e '.[docs]'
mkdocs serve
```

The versioned documentation configuration is in `.readthedocs.yaml`; a release check can be run with `mkdocs build --strict`.
