#!/usr/bin/env python3
"""Build the external-data AtlasTailor tutorials tracked in the repository."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write(name: str, title: str, intro: str, cells: list, data_records: list[str]):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "atlastailor": {
            "category": "end_to_end_real_data_tutorial",
            "requires_external_data": True,
            "data_records": data_records,
        },
    }
    notebook["cells"] = [markdown(f"# {title}\n\n{intro}"), *cells]
    nbf.write(notebook, NOTEBOOKS / name)


COMMON_SETUP = r'''
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hyperspatial as hs
from hyperspatial.io import read_reference, read_target, read_truth
from hyperspatial.metrics import evaluate_matrices

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = Path.cwd().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def new_output(label: str) -> Path:
    root = Path(os.environ.get("ATLASTAILOR_RUNS", ROOT / "tutorial_runs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / f"{label}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output
'''


write(
    "00_run_atlastailor_dlpfc.ipynb",
    "Run AtlasTailor end to end on human DLPFC Visium data",
    """This tutorial uses two public, same-donor DLPFC Visium sections from the
spatialLIBD study: section 151507 is the deeply measured reference and section
151508 is the target. It covers official data acquisition, source-only feature
preparation, 16-gene panel design, geometry-only registration, prediction
locking and post-lock evaluation.

This is a **software tutorial**, not a reanalysis of the frozen Figure 5
benchmark. The exact manuscript workflow, including its complete evaluation
universe and prediction locks, remains in the companion reproducibility
repository.""",
    [
        markdown("""
## 1. Obtain the authoritative public inputs

The expression matrices are the official 10x filtered matrices published by
the Lieber Institute. Coordinates are exported from the official spatialLIBD
object. Install the R dependency once with:

```r
if (!requireNamespace("BiocManager")) install.packages("BiocManager")
BiocManager::install("spatialLIBD")
```

Then run `Rscript preprocessing/export_spatiallibd_coordinates.R
data/external/dlpfc/spot_coordinates.tsv`. The two small 10x matrices can be
downloaded from the public URLs declared below. No target expression is used
for panel design, registration or model selection.
"""),
        code(COMMON_SETUP + r'''
import urllib.request

DATA = Path(os.environ.get("ATLASTAILOR_DLPFC_DATA", ROOT / "data" / "external" / "dlpfc"))
DATA.mkdir(parents=True, exist_ok=True)

FILES = {
    "151507_filtered_feature_bc_matrix.h5": {
        "url": "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151507_filtered_feature_bc_matrix.h5",
        "sha256": "686521ab02a68dfdd810c3cde26daea2257d016c8047859877ffb8c7ac5bdd01",
    },
    "151508_filtered_feature_bc_matrix.h5": {
        "url": "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/151508_filtered_feature_bc_matrix.h5",
        "sha256": "48ab74b340c79f19070e433b8b7db67932a6e889d421a691f70293587fff0621",
    },
}

for name, record in FILES.items():
    path = DATA / name
    if not path.exists():
        print(f"Downloading {name} from the official spatialLIBD host")
        urllib.request.urlretrieve(record["url"], path)
    observed = sha256(path)
    if observed != record["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {name}: {observed}")

coordinates_path = DATA / "spot_coordinates.tsv"
if not coordinates_path.exists():
    raise FileNotFoundError(
        "Create the coordinate table with: Rscript "
        "preprocessing/export_spatiallibd_coordinates.R "
        "data/external/dlpfc/spot_coordinates.tsv"
    )

pd.DataFrame(
    [{"file": name, "sha256": sha256(DATA / name)} for name in FILES]
)
'''),
        markdown("""
## 2. Prepare a source-defined tutorial gene universe

The complete 33,538-gene analysis is intentionally retained in the
reproducibility repository. To keep this interactive tutorial practical, we
select up to 2,000 variable genes using section 151507 only, restricted to
genes available on section 151508. Target gene names are assay metadata; no
target molecular values are used for selection. The prepared target matrix is
sealed on disk and is subsequently opened in backed mode so that prediction
materializes only the 16 selected columns.
"""),
        code(r'''
import anndata as ad
import scanpy as sc

coordinates = pd.read_csv(coordinates_path, sep="\t", dtype={"section": str, "barcode": str})


def read_section(section: str) -> ad.AnnData:
    matrix = sc.read_10x_h5(DATA / f"{section}_filtered_feature_bc_matrix.h5", gex_only=True)
    if "gene_ids" in matrix.var:
        matrix.var["gene_symbol"] = matrix.var_names.astype(str)
        matrix.var_names = matrix.var["gene_ids"].astype(str)
    if not matrix.var_names.is_unique:
        raise ValueError("Gene identifiers must be unique")
    meta = coordinates.loc[coordinates.section == section].copy().set_index("barcode")
    missing = matrix.obs_names.difference(meta.index)
    if len(missing):
        raise ValueError(f"{section}: {len(missing)} matrix barcodes lack coordinates")
    meta = meta.loc[matrix.obs_names]
    matrix.obsm["spatial"] = meta[["x", "y"]].to_numpy(np.float32)
    matrix.obs["section"] = section
    return matrix


source_full = read_section("151507")
target_full = read_section("151508")
shared = source_full.var_names.intersection(target_full.var_names)

source_for_selection = source_full[:, shared].copy()
sc.pp.normalize_total(source_for_selection, target_sum=1e4)
sc.pp.log1p(source_for_selection)
sc.pp.highly_variable_genes(source_for_selection, flavor="seurat", n_top_genes=min(2000, len(shared)))
tutorial_genes = source_for_selection.var_names[source_for_selection.var.highly_variable].tolist()

prepared = new_output("dlpfc_prepared")
source_path = prepared / "151507_reference.h5ad"
target_path = prepared / "151508_target_full_truth.h5ad"
source_full[:, tutorial_genes].copy().write_h5ad(source_path, compression="gzip")
target_full[:, tutorial_genes].copy().write_h5ad(target_path, compression="gzip")

{
    "source_spots": source_full.n_obs,
    "target_spots": target_full.n_obs,
    "shared_genes": len(shared),
    "tutorial_genes": len(tutorial_genes),
    "source_sha256": sha256(source_path),
    "target_sha256": sha256(target_path),
}
'''),
        markdown("""
## 3. Design exactly 16 measurements from the reference

`design_panel` accepts only the reference. The target object is not passed to
this step.
"""),
        code(r'''
run_root = new_output("dlpfc_atlastailor")
reference = read_reference(source_path)
design = hs.design_panel(
    reference,
    config=hs.DesignConfig(budget=16, budgets=(8, 16, 32)),
    out=run_root / "panel_design",
)
assert len(design.genes) == 16
design.ordered_panel[["position", "gene"]]
'''),
        markdown("""
## 4. Adapt and lock before opening target truth

`read_target` opens the target in backed mode and materializes only the 16
panel columns. Geometry registration receives coordinates only. Writing the
adaptation run completes its manifest and hashes the prediction artifacts.
"""),
        code(r'''
target_anchors = read_target(target_path, design.genes)
assert target_anchors.measured_expression.shape[1] == 16
assert target_anchors.metadata["target_nonanchor_materialized"] is False

result = hs.adapt(
    reference,
    target_anchors,
    design.genes,
    out=run_root / "adaptation",
    config=hs.AdaptConfig(registration="geometry", seeds=(42, 43, 44)),
)
assert result.manifest.status == "completed"
assert result.manifest.data["target_nonanchor_access_before_prediction"] is False
{
    "run": str(result.manifest.run_dir),
    "genes": len(result.genes),
    "source_supported_corrections": int(result.support.supported.sum()),
    "manifest_status": result.manifest.status,
}
'''),
        markdown("""
## 5. Post-lock evaluation

Only now is the full target matrix opened. Metrics use the same AtlasTailor
definitions; the output is a tutorial result and should not be substituted for
the frozen manuscript table.
"""),
        code(r'''
truth, truth_coordinates = read_truth(target_path, result.genes)
np.testing.assert_allclose(truth_coordinates, result.coordinates)
map_metrics = hs.validate(result, truth, out=run_root / "map_metrics.tsv")
idw_metrics = evaluate_matrices(
    truth, result.registered_atlas, result.registered_atlas, result.coordinates
)
metrics = pd.DataFrame({
    "gene": result.genes,
    "IDW_spatial_Pearson": idw_metrics.spatial_pearson,
    "MAP_spatial_Pearson": map_metrics.spatial_pearson,
    "IDW_log1p_RMSE": idw_metrics.log1p_rmse,
    "MAP_log1p_RMSE": map_metrics.log1p_rmse,
    "MAP_residual_Pearson": map_metrics.residual_pearson,
})
metrics.to_csv(run_root / "post_lock_comparison.tsv", sep="\t", index=False)
hidden = ~metrics.gene.isin(design.genes)

summary = pd.Series({
    "hidden_genes": int(hidden.sum()),
    "IDW_median_spatial_Pearson": metrics.loc[hidden, "IDW_spatial_Pearson"].median(),
    "MAP_median_spatial_Pearson": metrics.loc[hidden, "MAP_spatial_Pearson"].median(),
    "IDW_log1p_RMSE": np.sqrt(np.mean((truth[:, hidden] - result.registered_atlas[:, hidden]) ** 2)),
    "MAP_log1p_RMSE": np.sqrt(np.mean((truth[:, hidden] - result.adapted[:, hidden]) ** 2)),
})
summary
'''),
        code(r'''
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].scatter(metrics.loc[hidden, "IDW_spatial_Pearson"], metrics.loc[hidden, "MAP_spatial_Pearson"], s=9, alpha=0.45)
limits = [np.nanmin(metrics.loc[hidden, ["IDW_spatial_Pearson", "MAP_spatial_Pearson"]].to_numpy()),
          np.nanmax(metrics.loc[hidden, ["IDW_spatial_Pearson", "MAP_spatial_Pearson"]].to_numpy())]
axes[0].plot(limits, limits, color="0.35", lw=1)
axes[0].set(xlabel="Registered IDW spatial Pearson", ylabel="AtlasTailor spatial Pearson")
axes[1].scatter(metrics.loc[hidden, "IDW_log1p_RMSE"], metrics.loc[hidden, "MAP_log1p_RMSE"], s=9, alpha=0.45)
limits = [np.nanmin(metrics.loc[hidden, ["IDW_log1p_RMSE", "MAP_log1p_RMSE"]].to_numpy()),
          np.nanmax(metrics.loc[hidden, ["IDW_log1p_RMSE", "MAP_log1p_RMSE"]].to_numpy())]
axes[1].plot(limits, limits, color="0.35", lw=1)
axes[1].set(xlabel="Registered IDW log1p RMSE", ylabel="AtlasTailor log1p RMSE")
fig.tight_layout()
'''),
    ],
    data_records=["https://doi.org/10.6084/m9.figshare.13623902.v1"],
)


write(
    "01_run_atlastailor_zebrafish.ipynb",
    "Run AtlasTailor end to end on 3D zebrafish weMERFISH data",
    """This tutorial runs one real reciprocal reconstruction direction at 50%
epiboly: embryo E1 is the deeply measured reference and embryo E2 is the
target. Both H5AD files are public on Dryad. AtlasTailor selects exactly 16
genes from E1, registers geometry without target expression, freezes the E2
prediction and opens E2 non-anchor expression only for evaluation.

The notebook executes the reusable AtlasTailor API. The six-direction frozen
manuscript analysis and its original prediction locks remain in the companion
reproducibility repository.""",
    [
        markdown("""
## 1. Download and verify the public data

Download these two files from
[Dryad 10.5061/dryad.j0zpc86v9](https://doi.org/10.5061/dryad.j0zpc86v9)
and place them in `data/external/wemerfish/`:

- `weMERFISH_measured_A_50p_E1.h5ad`
- `weMERFISH_measured_A_50p_E2.h5ad`

Dryad may require an interactive public-download session for large files. This
notebook does not automate browser verification. The next cell verifies the
exact frozen SHA-256 hashes before any analysis.
"""),
        code(COMMON_SETUP + r'''
DATA = Path(os.environ.get("ATLASTAILOR_WEMERFISH_DATA", ROOT / "data" / "external" / "wemerfish"))
SOURCE = DATA / "weMERFISH_measured_A_50p_E1.h5ad"
TARGET = DATA / "weMERFISH_measured_A_50p_E2.h5ad"
EXPECTED = {
    SOURCE.name: "8c579f6d1278ec12e17665b6645c892890b29d8e2500cb4a3eecafcafc620884",
    TARGET.name: "e0a88d080cc5c5750c94ff7e2dfb313f257f1a4beaeae03c76d9a466f7f7c2d6",
}

for path in (SOURCE, TARGET):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download it from Dryad DOI 10.5061/dryad.j0zpc86v9."
        )
    observed = sha256(path)
    if observed != EXPECTED[path.name]:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: {observed}")

pd.DataFrame([{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
              for path in (SOURCE, TARGET)])
'''),
        markdown("""
## 2. Source-only 16-gene panel design

Only E1 is passed to the panel-design function. The 3D coordinate field is
read from `obsm['global_sphere']`, matching the public deposit.
"""),
        code(r'''
run_root = new_output("wemerfish_50p_E1_to_E2")
reference = read_reference(SOURCE, coordinate_key="global_sphere")
design = hs.design_panel(
    reference,
    config=hs.DesignConfig(budget=16, budgets=(8, 16, 32)),
    out=run_root / "panel_design",
)
assert len(design.genes) == 16
design.ordered_panel[["position", "gene"]]
'''),
        markdown("""
## 3. Anchor-only target loading, adaptation and prediction lock

Although the public E2 file contains its full transcriptome, `read_target`
opens it in backed mode and materializes only the declared 16 columns. E2
non-anchor expression is therefore unavailable to registration, fitting,
operator selection, calibration and guard decisions.
"""),
        code(r'''
target_anchors = read_target(TARGET, design.genes, coordinate_key="global_sphere")
assert target_anchors.measured_expression.shape[1] == 16
assert target_anchors.metadata["target_nonanchor_materialized"] is False

result = hs.adapt(
    reference,
    target_anchors,
    design.genes,
    out=run_root / "adaptation",
    config=hs.AdaptConfig(registration="geometry", seeds=(42, 43, 44)),
)
assert result.manifest.status == "completed"
assert result.manifest.data["target_nonanchor_access_before_prediction"] is False
{
    "run": str(result.manifest.run_dir),
    "predicted_genes": len(result.genes),
    "source_supported_corrections": int(result.support.supported.sum()),
    "manifest_status": result.manifest.status,
}
'''),
        markdown("""
## 4. Post-lock evaluation

The target truth is opened only after the completed prediction manifest has
been asserted. This evaluation is a fresh software tutorial run; manuscript
numbers should be taken from the frozen results walkthrough and companion
reproducibility repository.
"""),
        code(r'''
truth, truth_coordinates = read_truth(TARGET, result.genes, coordinate_key="global_sphere")
np.testing.assert_allclose(truth_coordinates, result.coordinates)
map_metrics = hs.validate(result, truth, out=run_root / "map_metrics.tsv")
idw_metrics = evaluate_matrices(
    truth, result.registered_atlas, result.registered_atlas, result.coordinates
)
metrics = pd.DataFrame({
    "gene": result.genes,
    "IDW_spatial_Pearson": idw_metrics.spatial_pearson,
    "MAP_spatial_Pearson": map_metrics.spatial_pearson,
    "IDW_log1p_RMSE": idw_metrics.log1p_rmse,
    "MAP_log1p_RMSE": map_metrics.log1p_rmse,
    "MAP_residual_Pearson": map_metrics.residual_pearson,
})
metrics.to_csv(run_root / "post_lock_comparison.tsv", sep="\t", index=False)
hidden = ~metrics.gene.isin(design.genes)
summary = pd.Series({
    "hidden_genes": int(hidden.sum()),
    "IDW_median_spatial_Pearson": metrics.loc[hidden, "IDW_spatial_Pearson"].median(),
    "MAP_median_spatial_Pearson": metrics.loc[hidden, "MAP_spatial_Pearson"].median(),
    "IDW_log1p_RMSE": np.sqrt(np.mean((truth[:, hidden] - result.registered_atlas[:, hidden]) ** 2)),
    "MAP_log1p_RMSE": np.sqrt(np.mean((truth[:, hidden] - result.adapted[:, hidden]) ** 2)),
})
summary
'''),
        code(r'''
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].scatter(metrics.loc[hidden, "IDW_spatial_Pearson"],
                metrics.loc[hidden, "MAP_spatial_Pearson"], s=10, alpha=0.5)
limits = [np.nanmin(metrics.loc[hidden, ["IDW_spatial_Pearson", "MAP_spatial_Pearson"]].to_numpy()),
          np.nanmax(metrics.loc[hidden, ["IDW_spatial_Pearson", "MAP_spatial_Pearson"]].to_numpy())]
axes[0].plot(limits, limits, color="0.35", lw=1)
axes[0].set(xlabel="Registered IDW spatial Pearson", ylabel="AtlasTailor spatial Pearson")
axes[1].scatter(metrics.loc[hidden, "IDW_log1p_RMSE"], metrics.loc[hidden, "MAP_log1p_RMSE"],
                s=10, alpha=0.5)
limits = [np.nanmin(metrics.loc[hidden, ["IDW_log1p_RMSE", "MAP_log1p_RMSE"]].to_numpy()),
          np.nanmax(metrics.loc[hidden, ["IDW_log1p_RMSE", "MAP_log1p_RMSE"]].to_numpy())]
axes[1].plot(limits, limits, color="0.35", lw=1)
axes[1].set(xlabel="Registered IDW log1p RMSE", ylabel="AtlasTailor log1p RMSE")
fig.tight_layout()
'''),
    ],
    data_records=["https://doi.org/10.5061/dryad.j0zpc86v9"],
)


write(
    "04_run_atlastailor_on_your_data.ipynb",
    "Run AtlasTailor on your own reference and target",
    """A concise, executable template for AnnData inputs. The reference contains
the deeply measured source atlas. The target contributes spatial coordinates
and the measured panel at prediction time; full target expression is optional
and must be opened only after prediction locking when retrospective validation
is intended.""",
    [
        markdown("""
## Input contract

Both files require unique gene identifiers and finite 2D or 3D coordinates in
`obsm['spatial']` (or an explicitly supplied coordinate key). Counts belong in
`X` or a named layer. Panel design is source-only. At prediction time,
`read_target` uses backed access and materializes only the selected columns.

Set `ATLASTAILOR_REFERENCE` and `ATLASTAILOR_TARGET` before running. For a
prospective experiment, the target file may contain only the measured genes.
For retrospective masking, it may contain the full matrix because the loader
enforces column-restricted access.
"""),
        code(COMMON_SETUP + r'''
REFERENCE_PATH = Path(os.environ["ATLASTAILOR_REFERENCE"])
TARGET_PATH = Path(os.environ["ATLASTAILOR_TARGET"])
COORDINATE_KEY = os.environ.get("ATLASTAILOR_COORDINATES", "spatial")

for path in (REFERENCE_PATH, TARGET_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

{"reference": str(REFERENCE_PATH), "target": str(TARGET_PATH),
 "reference_sha256": sha256(REFERENCE_PATH), "target_sha256": sha256(TARGET_PATH)}
'''),
        code(r'''
run_root = new_output("atlastailor_user_data")
reference = read_reference(REFERENCE_PATH, coordinate_key=COORDINATE_KEY)
design = hs.design_panel(
    reference,
    config=hs.DesignConfig(budget=16, budgets=(8, 16, 32)),
    out=run_root / "panel_design",
)
assert len(design.genes) == 16
design.ordered_panel[["position", "gene"]]
'''),
        code(r'''
target_anchors = read_target(TARGET_PATH, design.genes, coordinate_key=COORDINATE_KEY)
assert target_anchors.measured_expression.shape[1] == 16
assert target_anchors.metadata["target_nonanchor_materialized"] is False

result = hs.adapt(
    reference,
    target_anchors,
    design.genes,
    out=run_root / "adaptation",
    config=hs.AdaptConfig(registration="geometry", seeds=(42, 43, 44)),
)
assert result.manifest.status == "completed"
assert result.manifest.data["target_nonanchor_access_before_prediction"] is False
hs.inspect_run(result.manifest.run_dir)
'''),
        markdown("""
## Optional retrospective validation

Skip this section for prospective applications without target ground truth. If
the target file contains the full transcriptome, this cell opens it only after
the prediction manifest has completed.
"""),
        code(r'''
VALIDATE = os.environ.get("ATLASTAILOR_VALIDATE", "0") == "1"
if VALIDATE:
    truth, truth_coordinates = read_truth(TARGET_PATH, result.genes, coordinate_key=COORDINATE_KEY)
    np.testing.assert_allclose(truth_coordinates, result.coordinates)
    metrics = hs.validate(result, truth, out=run_root / "post_lock_metrics.tsv")
    display(metrics.head())
else:
    print("Prediction is locked. Set ATLASTAILOR_VALIDATE=1 only when retrospective truth is available.")
'''),
    ],
    data_records=[],
)

print("Wrote three external-data AtlasTailor tutorial notebooks.")
