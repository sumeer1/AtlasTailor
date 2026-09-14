from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import export_adaptation, export_forecast, export_panel, finish, new_run, write_report
from .config import AdaptConfig, DesignConfig, ForecastConfig
from .core import (apply_centered, candidate_fusion, canonicalize, depth_predictor,
    fit_centered_ridge_rank, orthants, pearson_columns, predicted_depth,
    predictor_label, pseudo_block_prediction, registered_prior, residual_features,
    select_panel, source_gate)
from .models import AdaptationResult, ForecastResult, PanelDesign, ReferenceAtlas, TargetSpecimen
from .panel import analyze_design
from .registration import register_geometry

def design_panel(reference: ReferenceAtlas, budget: int=16, out: str|Path|None=None,
                 config: DesignConfig|None=None, **kwargs) -> PanelDesign:
    """Design a source-only experimental panel; target data are not accepted."""
    config=config or DesignConfig(budget=budget,**kwargs)
    panel,utility,redundancy,curve,coverage,meta=analyze_design(reference,config)
    result=PanelDesign(panel.gene.tolist(),panel,utility,redundancy,curve,coverage)
    if out:
        inputs=[reference.source_path] if reference.source_path else []
        m=new_run('design',Path(out),config.to_dict(),inputs); result.manifest=m
        files=export_panel(result,m.run_dir)
        finish(m,files,{"selected_genes":result.genes,"source_only":True,"post_hoc_experimental_design":True,"selector_metadata":meta})
    return result

def _source_fit(reference: ReferenceAtlas, panel: list[str], config: AdaptConfig):
    genes=reference.genes.astype(str); lookup={g:i for i,g in enumerate(genes)}
    missing=[g for g in panel if g not in lookup]
    if missing: raise ValueError(f"reference lacks panel genes: {missing}")
    anchors=np.asarray([lookup[g] for g in panel]); source_log=np.asarray(reference.expression,np.float32)
    values=np.expm1(source_log).astype(np.float32); coords=canonicalize(reference.coordinates)[0]
    labels=orthants(coords); pseudo=np.log1p(pseudo_block_prediction(values,coords,labels,config.k_atlas)).astype(np.float32)
    residual=source_log-pseudo; raw=residual[:,anchors]; local=residual_features(raw,coords,config.residual_neighbors)
    max_label=int(labels.max()); validation=np.isin(labels,[0,max_label]); train=~validation
    records=[];best=None
    for ridge in config.ridge_grid:
        gm=fit_centered_ridge_rank(raw[train],residual[train],ridge,config.rank)
        lm=fit_centered_ridge_rank(local[train],residual[train],ridge,config.rank)
        am=fit_centered_ridge_rank(source_log[train][:,anchors],source_log[train],ridge,config.rank)
        for gain in config.gain_grid:
            gl=pseudo[validation]+gain*apply_centered(gm,raw[validation]); ll=pseudo[validation]+gain*apply_centered(lm,local[validation]); al=apply_centered(am,source_log[validation][:,anchors])
            candidates,names=candidate_fusion(pseudo[validation],gl,ll,al)
            selection,predictable,metadata=source_gate(candidates,source_log[validation],pseudo[validation],config.guard_mse_tolerance)
            selected=candidates[selection,:,np.arange(len(genes))].T
            corr=float(np.nanmedian(pearson_columns(source_log[validation],selected)));rmse=float(np.sqrt(np.mean((source_log[validation]-selected)**2)))
            rec={"ridge":ridge,"gain":gain,"median_spatial_pearson":corr,"log1p_rmse":rmse,"predictable_genes":int(predictable.sum())};records.append(rec)
            key=(corr,-rmse,int(predictable.sum()))
            if best is None or key>best[0]: best=(key,rec,names,selection,predictable,metadata)
    chosen,names,selection,predictable,metadata=best[1:];ridge=chosen['ridge']
    models=(fit_centered_ridge_rank(raw,residual,ridge,config.rank),fit_centered_ridge_rank(local,residual,ridge,config.rank),fit_centered_ridge_rank(source_log[:,anchors],source_log,ridge,config.rank))
    predictable[anchors]=False
    return {"genes":genes,"anchors":anchors,"source_log":source_log,"source_values":values,"coords":coords,
            "models":models,"chosen":chosen,"names":names,"selection":selection,"predictable":predictable,"gate":metadata,"grid":records}

def adapt(reference: ReferenceAtlas, target: TargetSpecimen, measured_genes: list[str]|None=None,
          out: str|Path|None=None, config: AdaptConfig|None=None) -> AdaptationResult:
    """Adapt a reference while target non-anchor expression remains unavailable."""
    config=config or AdaptConfig(); panel=list(measured_genes or target.measured_genes.astype(str))
    if set(panel)!=set(target.measured_genes.astype(str)): raise ValueError("target object must contain exactly the declared measured panel")
    if target.metadata.get('target_nonanchor_materialized',False): raise ValueError("target non-anchor expression is forbidden")
    fit=_source_fit(reference,panel,config); genes=fit['genes'];anchors=fit['anchors']
    target_order={g:i for i,g in enumerate(target.measured_genes.astype(str))}; anchor_raw=target.measured_expression[:,[target_order[g] for g in panel]]
    if target.metadata.get('already_log'):
        anchor_log=np.asarray(anchor_raw,np.float32)
    else:
        source_counts=reference.counts if reference.counts is not None else fit['source_values']
        weight=depth_predictor(np.asarray(source_counts,np.float32),anchors); depth=predicted_depth(anchor_raw,weight); source_depth=float(np.median(np.maximum(np.asarray(source_counts).sum(1),1)))
        anchor_log=np.log1p(anchor_raw*(source_depth/depth)[:,None]).astype(np.float32)
    target_coords=canonicalize(target.coordinates)[0];primary=int(config.seeds[0]);mapped=None;geometry_repeats=[]
    for seed in config.seeds:
        current,geometry=register_geometry(fit['coords'],target_coords,int(seed),config.registration_epochs,config.device,config.registration=='pre_registered')
        geometry_repeats.append(geometry)
        if int(seed)==primary:mapped=current
    if mapped is None:raise RuntimeError('primary registration seed did not run')
    base=registered_prior(mapped,fit['source_values'],target_coords,config.k_atlas); anchor_residual=anchor_log-base[:,anchors]
    gm,lm,am=fit['models']; gain=fit['chosen']['gain']; gl=base+gain*apply_centered(gm,anchor_residual);ll=base+gain*apply_centered(lm,residual_features(anchor_residual,target_coords,config.residual_neighbors));al=apply_centered(am,anchor_log)
    candidates,names=candidate_fusion(base,gl,ll,al)
    if names!=fit['names']: raise RuntimeError("fusion candidate order changed")
    adapted=candidates[fit['selection'],:,np.arange(len(genes))].T;adapted[:,~fit['predictable']]=base[:,~fit['predictable']];adapted=np.maximum(adapted,0)
    selected_names=np.asarray(names)[fit['selection']]; selected_names[~fit['predictable']]='registered_idw'
    support=pd.DataFrame({"gene":genes,"is_anchor":np.isin(np.arange(len(genes)),anchors),"supported":fit['predictable'],"selected_candidate":selected_names,"selected_predictor":[predictor_label(x) for x in selected_names],
        "source_cv_pearson":fit['gate']['selected_validation_pearson'],"source_cv_idw_pearson":fit['gate']['base_validation_pearson'],"source_cv_mse":fit['gate']['selected_validation_mse'],"source_cv_idw_mse":fit['gate']['base_validation_mse'],
        "source_enrichment_threshold":np.quantile(fit['source_values'],.8,axis=0)})
    result=AdaptationResult(genes,target_coords,adapted,base,adapted-base,support,mapped,panel)
    if out:
        inputs=[p for p in (reference.source_path,target.source_path) if p]
        m=new_run('adapt',Path(out),config.to_dict()|{"panel":panel},inputs);result.manifest=m
        files=export_adaptation(result,m.run_dir);(m.run_dir/'registration.json').write_text(json.dumps({'primary_seed':primary,'algorithmic_repeats':geometry_repeats},indent=2)+"\n");files.append(m.run_dir/'registration.json');write_report(result,m.run_dir/'report.html');files.append(m.run_dir/'report.html')
        finish(m,files,{"selected_genes":panel,"selected_source_configuration":fit['chosen'],"source_selection_grid":fit['grid'],"target_nonanchor_access_before_prediction":False,"primary_algorithmic_seed":primary})
    return result

def explore(result: AdaptationResult, gene: str|None=None, out: str|Path|None=None):
    from .metrics import moran
    table=result.support.copy(); table['predicted_residual_moran']=[moran(result.predicted_residual[:,j],result.coordinates) for j in range(len(result.genes))]
    table['mean_absolute_predicted_residual']=np.mean(np.abs(result.predicted_residual),axis=0)
    table=table.sort_values(['supported','predicted_residual_moran'],ascending=[False,False])
    if out: table.to_csv(out,sep='\t',index=False)
    return result.gene(gene) if gene else table

def forecast(early_state: AdaptationResult, tracking_map: str|Path, out: str|Path|None=None,
             config: ForecastConfig|None=None) -> ForecastResult:
    """Apply a compatible, source-trained temporal residual bundle.

    The NPZ must contain descendant_ancestor, early_ancestors, late_coordinates and
    temporal_residual_prediction. It is a serialized source-trained MAP-T output;
    no target-late expression is accepted.
    """
    from hyperspatial_map_t_core_20260804 import lineage_transport
    config=config or ForecastConfig(); bundle=np.load(tracking_map,allow_pickle=False)
    transported,transport_indices=lineage_transport(early_state.adapted,bundle['descendant_ancestor'],bundle['early_ancestors'])
    residual=np.asarray(bundle['temporal_residual_prediction'],np.float32)
    if residual.shape!=transported.shape: raise ValueError("temporal residual shape mismatch")
    final=np.maximum(transported+residual,0); supported=bundle['supported'].astype(bool) if 'supported' in bundle else np.ones(len(early_state.genes),bool)
    residual[:,~supported]=0;final[:,~supported]=transported[:,~supported]
    support=pd.DataFrame({'gene':early_state.genes,'supported':supported,'selected_predictor':np.where(supported,'Trajectory-aware MAP-T','Transported-state fallback')})
    summary=pd.DataFrame({'late_index':range(len(final)),'ancestor_id':bundle['descendant_ancestor'],'early_index':transport_indices})
    result=ForecastResult(early_state.genes,np.asarray(bundle['late_coordinates']),transported,residual,final,support,summary)
    if out:
        inputs=[Path(tracking_map)]+([early_state.manifest.run_dir/'manifest.json'] if early_state.manifest else [])
        m=new_run('forecast',Path(out),config.to_dict(),inputs);result.manifest=m;files=export_forecast(result,m.run_dir);finish(m,files,{"target_late_expression_access_before_prediction":False,"description":"tracking-map-conditioned forecasting"})
    return result

def validate(result: AdaptationResult, truth: np.ndarray, out: str|Path|None=None):
    from .metrics import evaluate_matrices
    if result.manifest and result.manifest.status!='completed': raise RuntimeError("predictions must be frozen before truth is opened")
    if truth.shape!=result.adapted.shape: raise ValueError("truth shape mismatch")
    thresholds=result.support.source_enrichment_threshold.to_numpy() if 'source_enrichment_threshold' in result.support else None
    d=evaluate_matrices(truth,result.adapted,result.registered_atlas,result.coordinates,thresholds);d.insert(0,'gene',result.genes)
    if out: d.to_csv(out,sep='\t',index=False)
    return d

def inspect_run(path: str|Path):
    path=Path(path);data=json.loads((path/'manifest.json').read_text())
    if data.get('run_id')!=path.name:data['scientific_run_id']=data.get('run_id');data['run_id']=path.name
    return data
