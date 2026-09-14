# HyperSpatial-MAP

**Sparse measurements adapt a deeply measured spatial atlas toward the molecular state of an individual specimen.**

HyperSpatial-MAP is an atlas-adaptation framework for spatial omics. It separates the molecular field expected from registered anatomy from a source-supported, target-specific correction:

 ## Manuscript Figure 1

  [![HyperSpatial-MAP Figure 1](manuscript/source_panels/figure_01/Figure_1_HyperSpatial_MAP_Nature.svg)](manuscript/final_display_figures/Figure_1.pdf)

  [View publication-quality Figure 1 PDF](manuscript/final_display_figures/Figure_1.pdf)


## Repository scope

This repository combines two layers:

1. a reusable Python package with deterministic 2D and 3D examples; and
2. a manuscript reproducibility record containing the executed workflows, frozen configurations, compact source tables, checksums, and independently generated source-panel SVGs.

The final manuscript layouts were assembled manually in Inkscape and are intentionally **not included**. No combined manuscript Figure 1–6 SVG/PDF/PNG is presented as programmatically generated. See [manuscript provenance](manuscript/README.md).

## Scientific workflow

```text
deep spatial reference
        +
target geometry
        +
sparse measured target genes
        |
geometry-only registration
        |
registered 12-NN IDW atlas prior
        +
source-trained target-specific residual prediction
        |
source-validation guard / IDW fallback
        |
adapted target molecular state
```

## Installation

Python 3.10–3.11 is supported. A fresh environment is recommended.

```bash
git clone https://github.com/sumeer1/HyperSpatial-MAP.git
cd HyperSpatial-MAP
conda env create -f environment.yml
conda activate hyperspatial-map
python -m pip install -e .
```

For manuscript-scale workflows:

```bash
conda env create -f environment-manuscript.yml
conda activate hyperspatial-map-manuscript
python -m pip install -e .
```

Repository ownership, author metadata, and archival DOI remain explicit pre-release placeholders; see [release checklist](RELEASE_CHECKLIST.md).

The current local release checks are summarized in [`VALIDATION.md`](VALIDATION.md).

## Five-minute example

```python
import hyperspatial as hs
from hyperspatial.io import read_reference, read_target

reference = read_reference("examples/minimal_2d/reference.h5ad")
panel = hs.design_panel(reference, budget=8)
target = read_target("examples/minimal_2d/target.h5ad", panel.genes)

result = hs.adapt(
    reference=reference,
    target=target,
    measured_genes=panel.genes,
    config=hs.AdaptConfig(registration="pre_registered", seeds=(42,)),
)
result.export("adaptation_run")
```

The bundled fixtures are deterministic synthetic examples for software testing, not manuscript evidence.

## Tutorials

- [Quickstart](notebooks/00_quickstart.ipynb)
- [Source-only panel design](notebooks/01_design_sparse_panel.ipynb)
- [3D specimen adaptation](notebooks/02_adapt_3d_specimen.ipynb)
- [Atlas-relative residual exploration](notebooks/03_explore_target_specific_residuals.ipynb)
- [Tracking-conditioned forecasting](notebooks/04_forecast_with_MAPT.ipynb)
- [2D Visium adaptation](notebooks/05_adapt_2d_visium.ipynb)
- [Manuscript evidence walkthrough](notebooks/06_manuscript_evidence_walkthrough.ipynb)

## Manuscript reproducibility

| Component | Location |
|---|---|
| Final panel-to-source map | [`manuscript/FIGURE_PROVENANCE.tsv`](manuscript/FIGURE_PROVENANCE.tsv) |
| Supplementary provenance | [`manuscript/SUPPLEMENTARY_PROVENANCE.tsv`](manuscript/SUPPLEMENTARY_PROVENANCE.tsv) |
| Byte-identical source-panel provenance | [`manuscript/SOURCE_PANEL_PROVENANCE.tsv`](manuscript/SOURCE_PANEL_PROVENANCE.tsv) |
| Frozen compact tables | [`manuscript/frozen_tables/`](manuscript/frozen_tables/) |
| Generated source panels | [`manuscript/source_panels/`](manuscript/source_panels/) |
| Executed workflow snapshots | [`reproducibility/workflows/`](reproducibility/workflows/) |
| Frozen configurations and locks | [`reproducibility/configs/`](reproducibility/configs/) |
| Executed-script provenance | [`reproducibility/provenance/EXECUTED_SCRIPT_PROVENANCE.tsv`](reproducibility/provenance/EXECUTED_SCRIPT_PROVENANCE.tsv) |
| Data access and checksums | [`data/`](data/) |
| Integrity verification | `python scripts/verify_release.py` |

Two reproduction levels are supported:

- **Fast verification:** validate copied scripts, tables, configurations, and source panels against `SHA256SUMS.tsv`.
- **Full scientific reproduction:** obtain provider data, reproduce each frozen workflow in a new directory, lock predictions before opening withheld targets, and compare against the archived manifests. Full runs are compute- and storage-intensive.

See [reproducibility guide](reproducibility/README.md) and [workflow order](reproducibility/RUN_ORDER.md).

## Tests and quality checks

```bash
pytest
python scripts/execute_notebooks.py
python scripts/verify_release.py
ruff check src tests scripts
```

Useful development commands are also available through `make`:

```bash
make install
make test
make notebooks
make verify
make docs
```

## Scientific guardrails

- Target molecular expression is excluded from geometry registration.
- Only declared target anchors are available during prediction.
- Target non-anchor expression is withheld until predictions are serialized and locked.
- Predictor selection, calibration, and fallback use source validation rather than target outcomes.
- Registered IDW predicts no atlas-relative correction; its residual correlation is undefined, not zero.
- Genes and spots are technical evaluation units and do not create biological replication.
- Atlas-relative residuals are not automatically biological, causal, or clinically predictive.

See [scientific guardrails](docs/scientific_guardrails.md) and [statistical units](docs/statistical_units.md).

## Data availability

Manuscript-scale provider data and prediction arrays are not redistributed through GitHub. Accession-level instructions, licenses, checksums, and the distinction between bundled and external artifacts are documented in [data/README.md](data/README.md) and [data/DATASETS.tsv](data/DATASETS.tsv).

## Citation and license

Software citation metadata are provided in [`CITATION.cff`](CITATION.cff). Author, manuscript, and DOI fields must be finalized before public release. Code is provided under the BSD 3-Clause License; external datasets retain their original licenses and terms.
