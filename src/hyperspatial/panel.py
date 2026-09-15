"""Source-only experimental panel design using the frozen extension machinery."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import DesignConfig
from .core import canonicalize, orthants, pseudo_block_prediction
from run_hyperspatial_map_stage1b_20260803 import AXIAL, EXPANDED_EXCLUDE
from analysis.panel_design_extension.run_panel_design_extension import (
    add_raw, cv_panel, raw_stats, stagewise_outer_metrics, stagewise_select,
    standardized_stats, subtract_raw,
)

def _candidate_set(source_log, genes, config):
    prevalence=np.mean(source_log>0,axis=0);variance=np.var(source_log,axis=0)
    allowed=set(config.candidates) if config.candidates else None
    user_exclude={x.lower() for x in config.exclusions}
    candidates=np.asarray([i for i,g in enumerate(genes)
        if config.prevalence_min<=prevalence[i]<=config.prevalence_max
        and variance[i]>config.variance_min
        and g.lower() not in EXPANDED_EXCLUDE and g.lower() not in user_exclude
        and not g.lower().startswith(('mt-','rpl','rps'))
        and (allowed is None or g in allowed)],np.int64)
    rng=np.random.default_rng(config.seed)
    sample=np.sort(rng.choice(len(source_log),min(config.max_source_points,len(source_log)),replace=False))
    z=source_log[sample].astype(np.float64);z=(z-z.mean(0))/np.maximum(z.std(0),1e-5)
    corr=np.abs((z[:,candidates].T@z)/max(len(z)-1,1))
    axial=[int(np.flatnonzero(genes==g)[0]) for g in AXIAL if np.any(genes==g)]
    if axial:
        keep=np.max(corr[:,axial],axis=1)<0.40;candidates,corr=candidates[keep],corr[keep]
    return candidates,corr,sample

def _fold_states(residual, candidates, labels):
    folds=np.unique(labels);x=residual[:,candidates]
    parts={int(f):raw_stats(x,residual,labels==f) for f in folds};total=add_raw(parts.values())
    states=[standardized_stats(subtract_raw(total,[parts[int(f)]]),parts[int(f)]) for f in folds]
    return folds,parts,total,states

def analyze_design(reference, config: DesignConfig):
    """Run strict nested source spatial CV without accepting any target object."""
    genes=np.asarray(reference.genes).astype(str);source_log=np.asarray(reference.expression,np.float32)
    coordinates=canonicalize(reference.coordinates)[0];labels=orthants(coordinates)
    values=np.expm1(source_log).astype(np.float32)
    residual=source_log-np.log1p(pseudo_block_prediction(values,coordinates,labels,12)).astype(np.float32)
    candidates,corr_abs,sample=_candidate_set(source_log,genes,config)
    budgets=sorted(set([int(b) for b in config.budgets]+[int(config.budget)]));max_budget=max(budgets)
    if len(candidates)<max_budget: raise ValueError(f"only {len(candidates)} eligible source genes remain for maximum budget {max_budget}")
    folds,parts,total,states=_fold_states(residual,candidates,labels)

    # The same stagewise conditional marginal-CV selector and sufficient-statistic
    # evaluator as the frozen panel-design extension. Generic references use a
    # prespecified member of the frozen grid (ridge=1, gain=1), never target tuning.
    selected,records=stagewise_select(states,candidates,genes,corr_abs,config.ridge,max_budget)
    selected=np.asarray(selected,np.int64);gene_indices=candidates[selected]

    outer=[]
    for outer_fold in folds:
        inner=[]
        for inner_fold in folds:
            if inner_fold==outer_fold:continue
            train=subtract_raw(total,[parts[int(outer_fold)],parts[int(inner_fold)]])
            inner.append(standardized_stats(train,parts[int(inner_fold)]))
        order,_=stagewise_select(inner,candidates,genes,corr_abs,config.ridge,max_budget)
        held=standardized_stats(subtract_raw(total,[parts[int(outer_fold)]]),parts[int(outer_fold)])
        perf=stagewise_outer_metrics(held,order,tuple(budgets),config.ridge,config.gain)
        for budget in budgets:
            pearson,mse=perf[budget];outer.append({'budget':budget,'outer_fold':int(outer_fold),
                'median_residual_pearson':float(np.nanmedian(pearson)),
                'mean_residual_mse':float(np.nanmean(mse)),'test_points':int(held['n'])})
    outer=pd.DataFrame(outer)
    curve=outer.groupby('budget',as_index=False).agg(
        expected_source_cv_residual_pearson=('median_residual_pearson','median'),
        source_cv_residual_mse=('mean_residual_mse','mean'),spatial_folds=('outer_fold','count'),
        held_out_points=('test_points','sum'))

    full_pos=selected[:config.budget].tolist();full_corr,full_mse=cv_panel(full_pos,states,config.ridge,config.gain)
    anchors=gene_indices[:config.budget];eval_mask=np.ones(len(genes),bool);eval_mask[anchors]=False
    loo={}
    utility=[]
    for i,a in enumerate(anchors):
        keep=[p for j,p in enumerate(full_pos) if j!=i]
        corr,mse=cv_panel(keep,states,config.ridge,config.gain) if keep else (np.full(len(genes),np.nan),np.full(len(genes),np.nan))
        loo[i]=(corr,mse);delta=full_corr-corr;valid=eval_mask&np.isfinite(delta)
        order=np.flatnonzero(valid);order=order[np.argsort(delta[order])[::-1]]
        utility.append({'position':i+1,'gene':genes[a],'loao_utility':float(np.nanmedian(delta[valid])),
            'source_cv_mse_improvement':float(np.nanmedian((mse-full_mse)[valid])),
            'fraction_genes_helped':float(np.mean(delta[valid]>0)),'genes_evaluated':int(valid.sum()),
            'top_supported_genes':','.join(genes[order[:10]].tolist())})
    redundancy=[]
    for ai,a in enumerate(anchors):
        for bi,b in enumerate(anchors):
            if ai==bi:continue
            keep=[p for j,p in enumerate(full_pos) if j not in (ai,bi)]
            pair_corr,pair_mse=cv_panel(keep,states,config.ridge,config.gain) if keep else (np.full(len(genes),np.nan),np.full(len(genes),np.nan))
            u=loo[bi][0]-pair_corr;valid=eval_mask&np.isfinite(u)
            redundancy.append({'anchor':genes[a],'conditioning_anchor':genes[b],
                'conditional_utility':float(np.nanmedian(u[valid])),
                'conditional_mse_improvement':float(np.nanmedian((pair_mse-loo[bi][1])[valid])),
                'fraction_genes_helped_conditionally':float(np.mean(u[valid]>0)),
                'interpretation':'lower conditional utility indicates greater substitutability/redundancy'})
    panel=pd.DataFrame(records[:config.budget]).rename(columns={'rank':'position'})
    panel=panel[['position','gene','marginal_residual_pearson_gain','expected_source_cv_residual_pearson',
                 'coverage_mean_max_squared_correlation','redundancy_mean_pairwise_squared_correlation',
                 'top_supported_genes']]
    coverage=np.max(corr_abs[selected[:config.budget]]**2,axis=0)
    covered=pd.DataFrame({'gene':genes,'max_squared_correlation_to_panel':coverage}).sort_values('max_squared_correlation_to_panel')
    meta={'candidate_count':int(len(candidates)),'selector_sample_points':int(len(sample)),
          'spatial_folds':int(len(folds)),'outer_cv':'strict nested spatial folds',
          'ridge':config.ridge,'gain':config.gain,
          'selector':'frozen stagewise conditional marginal source-CV recoverability',
          'target_information_used':False}
    return panel,pd.DataFrame(utility),pd.DataFrame(redundancy),curve,covered,meta
