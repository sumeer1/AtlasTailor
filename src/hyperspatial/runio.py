from pathlib import Path
import numpy as np
import pandas as pd
from .models import AdaptationResult, RunManifest

def load_adaptation(path: str|Path):
    import anndata as ad
    p=Path(path); adapted=ad.read_h5ad(p/'adapted.h5ad');base=ad.read_h5ad(p/'registered_atlas.h5ad');res=ad.read_h5ad(p/'predicted_residual.h5ad')
    support=pd.read_csv(p/'gene_support.tsv',sep='\t'); panel=pd.read_csv(p/'panel.tsv',sep='\t').gene.tolist(); mapped=np.load(p/'registered_coordinates.npy')
    from .workflows import inspect_run
    data=inspect_run(p);m=RunManifest(data['run_id'],data['workflow'],data['status'],p,data)
    return AdaptationResult(np.asarray(adapted.var_names),np.asarray(adapted.obsm['spatial']),np.asarray(adapted.X),np.asarray(base.X),np.asarray(res.X),support,mapped,panel,m)
