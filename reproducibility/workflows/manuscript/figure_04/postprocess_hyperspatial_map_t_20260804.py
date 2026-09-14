#!/usr/bin/env python3
"""Analysis-only completion of frozen HyperSpatial-MAP-T artifacts."""
from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/mplconfig_map_t")
import h5py, matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from hyperspatial_map_t_core_20260804 import canonicalize, pearson_columns, pooled_pearson, read_counts, read_gene_names, read_geometry, sha256

ROOT=Path("results_v10/hyperspatial_map_t_4d_frozen_20260804_112617")
INPUTS=Path("results_v10/hyperspatial_map_t_4d_frozen_20260804_105810/00_data_audit/inputs")
FOLDS={"A":("E1","E2","03_fold_A"),"B":("E2","E1","04_fold_B")}

def write_tsv(path,rows):
    rows=list(rows)
    with path.open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def verify_hashes():
    rows=[]
    for fold,(_,_,directory) in FOLDS.items():
        manifest=ROOT/directory/'PRE_UNBLINDING_PREDICTION_HASHES.tsv'
        sentinel=json.load(open(ROOT/directory/'UNBLINDING_SENTINEL.json'))
        manifest_ok=sha256(manifest)==sentinel['prediction_hash_manifest_sha256']
        table=pd.read_csv(manifest,sep='\t')
        for row in table.itertuples():
            path=Path(row.path); current=sha256(path)
            rows.append({'fold':fold,'method':row.method,'path':row.path,'frozen_sha256':row.sha256,
                         'current_sha256':current,'unchanged':current==row.sha256,'manifest_matches_sentinel':manifest_ok})
    write_tsv(ROOT/'14_logs/POST_UNBLINDING_PREDICTION_INTEGRITY.tsv',rows)
    if not all(r['unchanged'] and r['manifest_matches_sentinel'] for r in rows): raise RuntimeError('Prediction integrity failure')

def main():
    verify_hashes(); genes=read_gene_names(INPUTS/'matched_A_50p_E1_sphere_logFC.h5ad')
    stratified=[]; per_cell=[]; gene_sets={}
    panels=[]
    for fold,(source,target,directory) in FOLDS.items():
        fd=ROOT/directory; config=json.load(open(fd/'FROZEN_FOLD_CONFIG.json')); anchors=set(config['anchor_indices'])
        eval_genes=np.asarray([i for i in range(len(genes)) if i not in anchors])
        pred_dir=fd/'predictions_pre_unblinding'
        preds={name:np.load(pred_dir/f'{name}.npy',mmap_mode='r') for name in (
            'B1_lineage_registered_idw','B2_lineage_early_map','B4_anchor_only_temporal','B5_trajectory_free_map_t','B6_trajectory_aware_map_t')}
        depth=config['early_map']['depth_target']; truth=np.log1p(read_counts(INPUTS/f'matched_B_75p_{target}_sphere_logFC.h5ad')*(depth/np.maximum(read_counts(INPUTS/f'matched_B_75p_{target}_sphere_logFC.h5ad').sum(1),1))[:,None])
        xyz=canonicalize(read_geometry(INPUTS/f'matched_B_75p_{target}_sphere_logFC.h5ad'))[0]
        lineage=np.asarray(preds['B2_lineage_early_map']); full=np.asarray(preds['B6_trajectory_aware_map_t']); base=np.asarray(preds['B1_lineage_registered_idw'])
        err=np.sqrt(np.mean((truth[:,eval_genes]-full[:,eval_genes])**2,axis=1))
        base_err=np.sqrt(np.mean((truth[:,eval_genes]-base[:,eval_genes])**2,axis=1))
        for i,(e,b) in enumerate(zip(err,base_err)): per_cell.append({'fold':fold,'target_embryo':target,'evaluation_row':i,'map_t_rmse':float(e),'idw_rmse':float(b),'map_t_improvement':float(b-e)})
        # Source-only stage-emergent and spatial sets.
        early=read_counts(INPUTS/f'matched_A_50p_{source}_sphere_logFC.h5ad'); late=read_counts(INPUTS/f'matched_B_75p_{source}_sphere_logFC.h5ad')
        early_mean=np.log1p(early*(depth/np.maximum(early.sum(1),1))[:,None]).mean(0)
        late_log=np.log1p(late*(depth/np.maximum(late.sum(1),1))[:,None]); change=late_log.mean(0)-early_mean
        emergent=np.argsort(np.abs(change))[::-1][:50]
        var=np.var(late_log,axis=0); svg=np.argsort(var)[::-1][:100]
        gene_sets[fold]={'source_embryo':source,'stage_emergent_genes':genes[emergent].tolist(),'source_variable_genes':genes[svg].tolist(),'anchors':config['anchors']}
        for group,idx in (('all_non_anchor',eval_genes),('source_stage_emergent',np.asarray([i for i in emergent if i not in anchors])),('source_variable',np.asarray([i for i in svg if i not in anchors]))):
            rt=truth[:,idx]-lineage[:,idx]
            for method,pred in preds.items():
                stratified.append({'fold':fold,'stratum_type':'source_gene_group','stratum':group,'method':method,
                                   'cells':len(truth),'genes':len(idx),'rho_future':pooled_pearson(rt,np.asarray(pred)[:,idx]-lineage[:,idx])})
        # Spatial strata derived without expression.
        radius=np.linalg.norm(xyz,axis=1); density=1/np.maximum(cKDTree(xyz).query(xyz,k=13)[0][:,-1],1e-5)**3
        for label,values in (('radial_position',radius),('local_density',density)):
            bins=np.digitize(values,np.quantile(values,[.25,.5,.75]))
            for q in range(4):
                m=bins==q; rt=truth[m][:,eval_genes]-lineage[m][:,eval_genes]
                for method,pred in preds.items(): stratified.append({'fold':fold,'stratum_type':label,'stratum':f'Q{q+1}','method':method,'cells':int(m.sum()),'genes':len(eval_genes),'rho_future':pooled_pearson(rt,np.asarray(pred)[m][:,eval_genes]-lineage[m][:,eval_genes])})
        panels.append((fold,target,truth,preds,xyz,config,emergent))
    write_tsv(ROOT/'12_source_data/PER_CELL_METRICS.tsv',per_cell)
    write_tsv(ROOT/'12_source_data/STRATIFIED_ENDPOINTS.tsv',stratified)
    with (ROOT/'12_source_data/SOURCE_DEFINED_GENE_SETS.json').open('x') as h: json.dump(gene_sets,h,indent=2)
    # Panels H-N. Display genes are predeclared biology or source-emergent, never target-ranked.
    fig=plt.figure(figsize=(13.2,11),layout='constrained'); gs=fig.add_gridspec(3,4)
    axes=[]
    for c,(fold,target,truth,preds,xyz,config,emergent) in enumerate(panels):
        ax=fig.add_subplot(gs[0,c]); axes.append(ax); sample=np.linspace(0,len(xyz)-1,min(2500,len(xyz))).astype(int)
        ax.scatter(xyz[sample,0],xyz[sample,1],s=2,c=xyz[sample,2],cmap='viridis',alpha=.65);ax.set_title(f'Fold {fold}: target 75% geometry');ax.axis('off');ax.set_aspect('equal')
    ax=fig.add_subplot(gs[0,2:]); axes.append(ax)
    d=pd.read_csv(ROOT/'12_source_data/DOMAIN_METRICS_ALL_FOLDS.tsv',sep='\t'); q=d.groupby(['fold','method'])[['dice','boundary_f1','centroid_error']].median().reset_index()
    for fold,marker in [('A','o'),('B','s')]:
        x=q[(q.fold==fold)&(q.method=='B1_lineage_registered_idw')].iloc[0]; y=q[(q.fold==fold)&(q.method=='B6_trajectory_aware_map_t')].iloc[0]
        ax.scatter([x.dice,y.dice],[x.centroid_error,y.centroid_error],marker=marker,s=[50,80],c=['#E69F00','#D62728']);ax.plot([x.dice,y.dice],[x.centroid_error,y.centroid_error],color='#888888',lw=1)
    ax.set(xlabel='Median domain Dice →',ylabel='Centroid error ↓',title='Localization beyond transported atlas');ax.spines[['top','right']].set_visible(False)
    display=['tbxta','noto','shha']; gene_indices=[int(np.flatnonzero(genes==g)[0]) for g in display]
    for row,gidx in enumerate(gene_indices,1):
        fold,target,truth,preds,xyz,config,emergent=panels[(row-1)%2]
        for col,(name,title) in enumerate((('B1_lineage_registered_idw','IDW'),('B6_trajectory_aware_map_t','MAP-T'))):
            ax=fig.add_subplot(gs[row if row<3 else 2,col]); axes.append(ax)
            vmin,vmax=np.quantile(truth[:,gidx],[.02,.98]); sc=ax.scatter(xyz[:,0],xyz[:,1],c=np.asarray(preds[name])[:,gidx],s=3,cmap='magma',vmin=vmin,vmax=vmax,rasterized=True)
            ax.set_title(f'{genes[gidx]} · {title} · Fold {fold}');ax.axis('off');ax.set_aspect('equal')
        if row==2: break
    ax=fig.add_subplot(gs[1,2]); axes.append(ax)
    allg=pd.read_csv(ROOT/'12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv',sep='\t');
    for fold,color in [('A','#0072B2'),('B','#D55E00')]:
        a=allg[(allg.fold==fold)&(allg.method=='B5_trajectory_free_map_t')].future_residual_pearson.values;b=allg[(allg.fold==fold)&(allg.method=='B6_trajectory_aware_map_t')].future_residual_pearson.values
        ax.hist(b-a,bins=35,histtype='step',density=True,color=color,label=f'Fold {fold}')
    ax.axvline(0,color='#777',ls=':');ax.set(xlabel='Trajectory-aware − trajectory-free\nper-gene residual Pearson',ylabel='Density',title='Trajectory contribution');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[1,3]); axes.append(ax)
    guards=pd.concat([pd.read_csv(ROOT/f'02_source_validation/fold_{f}_temporal_guards.tsv',sep='\t') for f in 'AB']); guards=guards[guards.model=='B6']
    counts=guards.groupby(['fold','guard_pass']).size().unstack(fill_value=0); bottoms=np.zeros(2)
    for state,color in [(True,'#D62728'),(False,'#BDBDBD')]: vals=[counts.loc[f,state] for f in 'AB'];ax.bar(['Fold A','Fold B'],vals,bottom=bottoms,color=color,label='corrected' if state else 'fallback');bottoms+=vals
    ax.set(title='Source-only temporal guards',ylabel='Genes');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[2,2]); axes.append(ax)
    s=pd.DataFrame(stratified); s=s[(s.stratum_type=='source_gene_group')&(s.method.isin(['B1_lineage_registered_idw','B6_trajectory_aware_map_t']))]
    for method,color,label in [('B1_lineage_registered_idw','#E69F00','IDW'),('B6_trajectory_aware_map_t','#D62728','MAP-T')]:
        v=s[s.method==method].groupby('stratum').rho_future.mean(); ax.plot(range(len(v)),v.values,'-o',color=color,label=label)
    ax.set_xticks(range(len(v)),v.index,rotation=20);ax.set(ylabel='Future-residual Pearson',title='Source-defined gene groups');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[2,3]); axes.append(ax)
    u=pd.concat([pd.read_csv(ROOT/f'0{3 if f=="A" else 4}_fold_{f}/UNCERTAINTY_CALIBRATION.tsv',sep='\t') for f in 'AB'])
    for fold,color in [('A','#0072B2'),('B','#D55E00')]:
        z=u[(u.fold==fold)&(~u.is_anchor)];ax.hexbin(z.source_cv_sigma,z.target_mae,gridsize=25,mincnt=1,cmap='Greys',alpha=.45);ax.scatter([],[],color=color,label=f'Fold {fold}')
    ax.set(xlabel='Source-CV uncertainty',ylabel='Target MAE',title='Uncertainty–error relation');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    for label,ax in zip('hijklmn',axes): ax.text(-.1,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
    for ext in ('pdf','svg','png'): fig.savefig(ROOT/f'10_figures_main/Figure_MAP_T_biology_and_limits.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)
    # Additional all-gene/calibration supplementary overview.
    fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    g=pd.read_csv(ROOT/'12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv',sep='\t')
    for fold,color in [('A','#0072B2'),('B','#D55E00')]:
        for method,ls in [('B1_lineage_registered_idw','--'),('B6_trajectory_aware_map_t','-')]:
            v=np.sort(g[(g.fold==fold)&(g.method==method)].future_residual_pearson.dropna());axs[0,0].plot(v,np.linspace(0,1,len(v)),color=color,ls=ls,label=f'{fold} {"MAP-T" if method.startswith("B6") else "IDW"}')
    axs[0,0].set(xlabel='Per-gene future-residual Pearson',ylabel='ECDF');axs[0,0].legend(frameon=False,fontsize=8)
    for fold,color in [('A','#0072B2'),('B','#D55E00')]:
        z=u[(u.fold==fold)&(~u.is_anchor)];axs[0,1].scatter(z.source_cv_sigma,z.target_95pct_coverage,s=5,alpha=.25,color=color,label=fold)
    axs[0,1].axhline(.95,color='#777',ls=':');axs[0,1].set(xlabel='Source-CV sigma',ylabel='95% interval coverage');axs[0,1].legend(frameon=False)
    cells=pd.DataFrame(per_cell)
    for fold,color in [('A','#0072B2'),('B','#D55E00')]: axs[1,0].hist(cells[cells.fold==fold].map_t_improvement,bins=50,histtype='step',density=True,color=color,label=fold)
    axs[1,0].axvline(0,color='#777',ls=':');axs[1,0].set(xlabel='Per-cell RMSE improvement over IDW',ylabel='Density');axs[1,0].legend(frameon=False)
    st=pd.DataFrame(stratified);st=st[(st.stratum_type=='local_density')&(st.method=='B6_trajectory_aware_map_t')]
    for fold,color in [('A','#0072B2'),('B','#D55E00')]:
        z=st[st.fold==fold];axs[1,1].plot(z.stratum,z.rho_future,'-o',color=color,label=fold)
    axs[1,1].set(xlabel='Local-density quartile',ylabel='Future-residual Pearson');axs[1,1].legend(frameon=False)
    for ax in axs.flat: ax.spines[['top','right']].set_visible(False)
    for ext in ('pdf','svg','png'):fig.savefig(ROOT/f'11_figures_supplement/Supplementary_MAP_T_all_gene_and_calibration.{ext}',dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)
    print('postprocessing complete')

if __name__=='__main__':main()
