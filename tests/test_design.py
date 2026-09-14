import numpy as np
from hyperspatial import ReferenceAtlas, design_panel
from hyperspatial.config import DesignConfig

def test_design_is_source_only_nested_and_exports(tmp_path):
    rng=np.random.default_rng(18);n,g=96,38
    coordinates=rng.normal(size=(n,2)).astype('f4')
    counts=rng.poisson(2.5,size=(n,g)).astype('f4')
    depth=np.maximum(counts.sum(1),1);log=np.log1p(counts*(np.median(depth)/depth)[:,None]).astype('f4')
    ref=ReferenceAtlas(log,coordinates,np.asarray([f'g{i}' for i in range(g)]),counts=counts)
    result=design_panel(ref,config=DesignConfig(budget=3,budgets=(2,3)),out=tmp_path/'design')
    assert len(result.genes)==3
    assert set(result.budget_curve.budget)=={2,3}
    assert {'recommended_panel.tsv','marker_utility.tsv','conditional_utility.tsv','budget_curve.tsv','panel_design.h5ad','report.html','manifest.json'}<=set(p.name for p in (tmp_path/'design').iterdir())
    assert result.manifest.data['selector_metadata']['target_information_used'] is False
