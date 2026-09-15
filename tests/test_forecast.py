import numpy as np
from hyperspatial import forecast
from hyperspatial.models import AdaptationResult
import pandas as pd

def test_lineage_forecast(tmp_path):
    early=np.arange(12,dtype='f4').reshape(4,3);r=AdaptationResult(np.asarray(['a','b','c']),np.zeros((4,2)),early,early,np.zeros_like(early),pd.DataFrame({'supported':[False]*3}),np.zeros((4,2)),[])
    p=tmp_path/'tracking.npz';np.savez(p,descendant_ancestor=np.asarray([1,1,2,3]),early_ancestors=np.asarray([1,2,3,4]),late_coordinates=np.zeros((4,2)),temporal_residual_prediction=np.ones((4,3),dtype='f4'),supported=np.asarray([1,0,1],bool))
    f=forecast(r,p);assert np.allclose(f.temporal_residual[:,1],0);assert np.allclose(f.forecast[:,1],f.transported_early[:,1])
