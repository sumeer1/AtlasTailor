import json
import os
from pathlib import Path
import numpy as np
import pytest
from hyperspatial.io import read_reference
from hyperspatial.core import select_panel

ROOT=Path(os.environ.get('HYPERSPATIAL_RESEARCH_ROOT', Path(__file__).resolve().parents[1]))
@pytest.mark.regression
def test_source_selector_reproduces_frozen_50p_panel():
    source=ROOT/'data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E1.h5ad';manifest=ROOT/'results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E1_to_E2/prediction_manifest.json'
    if not source.exists():pytest.skip('frozen source absent')
    ref=read_reference(source);idx,_=select_panel(ref.expression,ref.genes,16);expected=json.loads(manifest.read_text())['anchor_genes'];assert ref.genes[idx].tolist()==expected

@pytest.mark.regression
def test_public_map_core_reproduces_frozen_50p_prediction():
    from hyperspatial.config import AdaptConfig
    from hyperspatial.workflows import _source_fit
    from hyperspatial.core import (apply_centered,candidate_fusion,canonicalize,
        depth_predictor,predicted_depth,residual_features)
    from run_hyperspatial_map_stage1_20260803 import target_anchor_counts
    import anndata as ad
    source=ROOT/'data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E1.h5ad'
    target=ROOT/'data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E2.h5ad'
    manifest=ROOT/'results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E1_to_E2/prediction_manifest.json'
    if not source.exists():pytest.skip('frozen source absent')
    m=json.loads(manifest.read_text());ref=read_reference(source)
    fit=_source_fit(ref,m['anchor_genes'],AdaptConfig(seeds=(42,),registration='pre_registered'))
    paths=m['methods']['42']['predictions'];base=np.load(ROOT/paths['registered_idw_k12']['path'])
    raw=target_anchor_counts(target,fit['anchors']);weight=depth_predictor(ref.counts,fit['anchors'])
    depth=predicted_depth(raw,weight);source_depth=float(np.median(np.maximum(ref.counts.sum(1),1)))
    anchor_log=np.log1p(raw*(source_depth/depth)[:,None]).astype('f4');anchor_residual=anchor_log-base[:,fit['anchors']]
    target_coordinates=canonicalize(ad.read_h5ad(target).obsm['global_sphere'])[0]
    gm,lm,am=fit['models'];gain=fit['chosen']['gain']
    global_log=base+gain*apply_centered(gm,anchor_residual)
    local_log=base+gain*apply_centered(lm,residual_features(anchor_residual,target_coordinates,(12,32)))
    anchor_only=apply_centered(am,anchor_log);candidates,names=candidate_fusion(base,global_log,local_log,anchor_only)
    assert names==fit['names'];prediction=candidates[fit['selection'],:,np.arange(len(fit['genes']))].T
    prediction[:,~fit['predictable']]=base[:,~fit['predictable']];prediction=np.maximum(prediction,0)
    frozen=np.load(ROOT/paths['hyperspatial_map_calibrated_fusion']['path'])
    assert np.allclose(prediction,frozen,atol=1e-5,rtol=1e-5)

@pytest.mark.regression
def test_frozen_map_t_reconstruction_equation():
    path=ROOT/'results_v10/hyperspatial_map_t_4d_frozen_20260804_112617/03_fold_A/predictions_pre_unblinding/decomposition.npz'
    if not path.exists():pytest.skip('frozen MAP-T decomposition absent')
    z=np.load(path);assert np.array_equal(np.maximum(z['early_map_lineage']+z['predicted_residual'],0),z['final_prediction'])
