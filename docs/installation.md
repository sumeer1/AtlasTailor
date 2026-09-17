# Installation

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

The non-rigid registration stage is the most computationally intensive component. The bundled examples use pre-registered synthetic coordinates and complete quickly on CPU.
