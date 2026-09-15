from __future__ import annotations
from pathlib import Path
import numpy as np
from .models import ReferenceAtlas, TargetSpecimen

def _dense(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)

def _coordinates(adata, key: str | None = None) -> tuple[np.ndarray, str]:
    choices = [key] if key else ["global_sphere", "spatial", "X_spatial"]
    for candidate in choices:
        if candidate and candidate in adata.obsm:
            x=np.asarray(adata.obsm[candidate],dtype=np.float32)
            if x.ndim!=2 or x.shape[1] not in (2,3): raise ValueError("coordinates must have 2 or 3 columns")
            if not np.isfinite(x).all(): raise ValueError("coordinates contain non-finite values")
            return x,candidate
    raise ValueError(f"no coordinate matrix found; tried {choices}")

def _matrix(adata, layer: str | None):
    if layer:
        if layer not in adata.layers: raise ValueError(f"missing layer {layer}")
        return _dense(adata.layers[layer])
    if "spliced" in adata.layers and "unspliced" in adata.layers:
        return _dense(adata.layers["spliced"])+_dense(adata.layers["unspliced"])
    return _dense(adata.X)

def read_reference(path: str | Path, coordinate_key: str | None=None,
                   layer: str | None=None, already_log: bool=False) -> ReferenceAtlas:
    import anndata as ad
    path=Path(path); a=ad.read_h5ad(path); coords,key=_coordinates(a,coordinate_key)
    raw=_matrix(a,layer); genes=np.asarray(a.var_names.astype(str))
    if len(np.unique(genes))!=len(genes): raise ValueError("gene identifiers must be unique")
    if already_log: expression=raw
    else:
        depth=np.maximum(raw.sum(1),1); target=float(np.median(depth)); expression=np.log1p(raw*(target/depth)[:,None]).astype(np.float32)
    return ReferenceAtlas(expression,coords,genes,np.asarray(a.obs_names.astype(str)),path,raw,
                          {"coordinate_key":key,"expression_layer":layer or "X/spliced+unspliced","already_log":already_log})

def read_target(path: str | Path, measured_genes: list[str], coordinate_key: str | None=None,
                layer: str | None=None, already_log: bool=False) -> TargetSpecimen:
    """Create a prediction-time target containing only coordinates and panel columns."""
    import anndata as ad
    path=Path(path); a=ad.read_h5ad(path,backed="r"); coords,key=_coordinates(a,coordinate_key)
    names=np.asarray(a.var_names.astype(str)); lookup={g:i for i,g in enumerate(names)}
    missing=[g for g in measured_genes if g not in lookup]
    if missing: raise ValueError(f"target lacks measured genes: {missing[:8]}")
    idx=np.asarray([lookup[g] for g in measured_genes])
    order=np.argsort(idx); reverse=np.argsort(order)
    # Only declared columns are materialized; the backed object is then closed.
    source=a.layers[layer] if layer else a.X
    values=_dense(source[:,idx[order]])[:,reverse]; obs=np.asarray(a.obs_names.astype(str)); a.file.close()
    if not already_log:
        # Target-wide depth is forbidden. Depth correction is learned later from anchor sums.
        values=np.asarray(values,np.float32)
    return TargetSpecimen(coords,values,np.asarray(measured_genes),obs,path,
                          {"coordinate_key":key,"expression_layer":layer or "X","already_log":already_log,
                           "target_nonanchor_materialized":False})

def read_truth(path: str | Path, genes: np.ndarray, coordinate_key: str|None=None,
               layer: str|None=None, depth_target: float|None=None) -> tuple[np.ndarray,np.ndarray]:
    import anndata as ad
    a=ad.read_h5ad(path); coords,_=_coordinates(a,coordinate_key); names=np.asarray(a.var_names.astype(str)); lookup={g:i for i,g in enumerate(names)}
    missing=[g for g in genes if g not in lookup]
    if missing: raise ValueError(f"truth lacks genes: {missing[:8]}")
    raw=_matrix(a,layer)[:,[lookup[g] for g in genes]]; depth=np.maximum(_matrix(a,layer).sum(1),1)
    target=float(np.median(depth)) if depth_target is None else depth_target
    return np.log1p(raw*(target/depth)[:,None]).astype(np.float32),coords
