import numpy as np
from hyperspatial import adapt,AdaptConfig,ReferenceAtlas,TargetSpecimen

def synthetic(d):
    rng=np.random.default_rng(9+d);n=80;g=24;c=rng.normal(size=(n,d)).astype('f4');raw=rng.poisson(2,size=(n,g)).astype('f4');raw[rng.random(raw.shape)<.22]=0;depth=np.maximum(raw.sum(1),1);log=np.log1p(raw*(np.median(depth)/depth)[:,None]).astype('f4')
    ref=ReferenceAtlas(log,c,np.asarray([f'g{i}' for i in range(g)]),counts=raw)
    panel=[f'g{i}' for i in range(4)];tar=TargetSpecimen(c.copy(),log[:,:4].copy(),np.asarray(panel),metadata={'already_log':True,'target_nonanchor_materialized':False})
    return ref,tar,panel

def test_adapt_2d_and_3d_no_target_truth():
    for d in (2,3):
        ref,tar,panel=synthetic(d);r=adapt(ref,tar,panel,config=AdaptConfig(seeds=(42,),registration='pre_registered'))
        assert r.adapted.shape==(80,24);assert np.allclose(r.adapted-r.registered_atlas,r.predicted_residual);assert set(r.support.selected_predictor)<= {'IDW fallback','Global residual','Local residual','Anchor-only','Calibrated mixture'}

def test_deterministic_pre_registered():
    ref,tar,panel=synthetic(2);cfg=AdaptConfig(seeds=(42,),registration='pre_registered');a=adapt(ref,tar,panel,config=cfg);b=adapt(ref,tar,panel,config=cfg);assert np.array_equal(a.adapted,b.adapted)

def test_exported_adapt_run_is_complete_and_hashed(tmp_path):
    import json
    ref,tar,panel=synthetic(2);out=tmp_path/'run';r=adapt(ref,tar,panel,out=out,config=AdaptConfig(seeds=(42,),registration='pre_registered'))
    required={'adapted.h5ad','registered_atlas.h5ad','predicted_residual.h5ad','gene_support.tsv','predictor_assignments.tsv','panel.tsv','registration.json','run_config.yaml','manifest.json','input_hashes.tsv','environment.txt','reproduce.sh','report.html'}
    assert required<=set(x.name for x in out.iterdir());m=json.loads((out/'manifest.json').read_text());assert m['status']=='completed';assert all(len(x['sha256'])==64 for x in m['artifacts'])
