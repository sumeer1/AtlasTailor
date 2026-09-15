import numpy as np
import anndata as ad
from hyperspatial.io import read_target

def test_target_loader_materializes_only_panel(tmp_path):
    a=ad.AnnData(X=np.arange(120,dtype='f4').reshape(12,10));a.var_names=[f'g{i}' for i in range(10)];a.obsm['spatial']=np.random.default_rng(1).normal(size=(12,2));p=tmp_path/'target.h5ad';a.write_h5ad(p)
    t=read_target(p,['g2','g7'])
    assert t.measured_expression.shape==(12,2)
    assert t.metadata['target_nonanchor_materialized'] is False
    assert t.measured_genes.tolist()==['g2','g7']

def test_target_loader_preserves_unsorted_panel_order(tmp_path):
    a=ad.AnnData(X=np.arange(60,dtype='f4').reshape(6,10));a.var_names=[f'g{i}' for i in range(10)];a.obsm['spatial']=np.ones((6,2));p=tmp_path/'target.h5ad';a.write_h5ad(p)
    t=read_target(p,['g8','g1','g5'])
    assert np.array_equal(t.measured_expression,np.asarray(a.X)[:,[8,1,5]])

def test_malformed_coordinate_dimension(tmp_path):
    a=ad.AnnData(X=np.ones((5,3)));a.var_names=['a','b','c'];a.obsm['spatial']=np.ones((5,4));p=tmp_path/'bad.h5ad';a.write_h5ad(p)
    import pytest
    with pytest.raises(ValueError):read_target(p,['a'])
