import numpy as np
from hyperspatial.core import registered_prior,residual_features
from run_hyperspatial_map_stage1_20260803 import idw,features

def test_wrappers_equal_frozen_core():
    rng=np.random.default_rng(4);s=rng.normal(size=(40,3)).astype('f4');q=rng.normal(size=(17,3)).astype('f4');v=np.abs(rng.normal(size=(40,8))).astype('f4')
    assert np.allclose(registered_prior(s,v,q),np.log1p(idw(s,v,q,12)))
    r=rng.normal(size=(17,4)).astype('f4')
    assert np.allclose(residual_features(r,q),features(r,q,True))
