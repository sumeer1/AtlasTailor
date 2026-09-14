import numpy as np
from hyperspatial.registration import register_geometry

def test_geometry_registration_accepts_2d_and_3d_without_expression():
    rng=np.random.default_rng(31)
    for d in (2,3):
        source=rng.normal(0,.2,size=(32,d)).astype('f4')
        target=(source*1.03+.02).astype('f4')
        mapped,qc=register_geometry(source,target,seed=42,epochs=1)
        assert mapped.shape==target.shape
        assert qc['mode']=='geometry' and qc['seed']==42
        assert np.isfinite(mapped).all()
