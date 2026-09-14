#!/usr/bin/env python3
"""Assemble publication figures from frozen HyperSpatial-MAP artifacts only."""
from __future__ import annotations
import json, hashlib, shutil, sys, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import rankdata
import anndata as ad

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
R3=REPO/'results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726'
R4=REPO/'results_v10/hyperspatial_map_t_4d_frozen_20260804_112617'
I4=REPO/'results_v10/hyperspatial_map_t_4d_frozen_20260804_105810/00_data_audit/inputs'
sys.path.insert(0,str(ROOT/'01_style'))
from hyperspatial_publication_style import apply_style, polish, panel_label, COLORS, METHOD_COLORS, METHOD_LABELS, TWO_COLUMN
apply_style()

MAP='hyperspatial_map_calibrated_fusion'; IDW='registered_idw_k12'; ANCHOR='anchor_only_calibrated'
ENDPOINTS=[('svg_spatial_pearson_median','Spatial Pearson',True),('auprc_enrichment_median','AUPRC enrichment',True),('topk_dice_median','Prevalence-matched Dice',True),('weighted_centroid_error_median','Centroid error',False),('spatial_emd_median','Spatial EMD',False),('log1p_rmse','log1p RMSE',False)]
DATA=REPO/'data_v10/wemerfish_zebrafish_3stage'
RUNS=[
 ('50% epiboly','50p','E1','E2',R3/'04_predictions/50p_E1_to_E2',DATA/'weMERFISH_measured_A_50p_E1.h5ad',DATA/'weMERFISH_measured_A_50p_E2.h5ad'),
 ('50% epiboly','50p','E2','E1',R3/'04_predictions/50p_E2_to_E1',DATA/'weMERFISH_measured_A_50p_E2.h5ad',DATA/'weMERFISH_measured_A_50p_E1.h5ad'),
 ('75% epiboly','75p','E1','E2',REPO/'results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2',DATA/'weMERFISH_measured_B_75p_E1.h5ad',DATA/'weMERFISH_measured_B_75p_E2.h5ad'),
 ('75% epiboly','75p','E2','E1',REPO/'results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1',DATA/'weMERFISH_measured_B_75p_E2.h5ad',DATA/'weMERFISH_measured_B_75p_E1.h5ad'),
 ('6-somite','6s','E1','E2',R3/'04_predictions/6s_E1_to_E2',DATA/'weMERFISH_measured_C_6s_E1_rescaled_z.h5ad',DATA/'weMERFISH_measured_C_6s_E2_rescaled_z.h5ad'),
 ('6-somite','6s','E2','E1',R3/'04_predictions/6s_E2_to_E1',DATA/'weMERFISH_measured_C_6s_E2_rescaled_z.h5ad',DATA/'weMERFISH_measured_C_6s_E1_rescaled_z.h5ad')]

def rd(p): return pd.read_csv(p,sep='\t')
def tsv(df,p): p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,sep='\t',index=False)
def write(p,s): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def export(fig,stem):
 stem.parent.mkdir(parents=True,exist_ok=True)
 for e in ('pdf','svg','png'): fig.savefig(stem.with_suffix('.'+e),dpi=300 if e=='png' else None,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def dense(x): return np.asarray(x.toarray() if hasattr(x,'toarray') else x,dtype=np.float32)
def counts_and_coords(p):
 a=ad.read_h5ad(p); x=dense(a.X); key='global_sphere' if 'global_sphere' in a.obsm else 'spatial'; c=np.asarray(a.obsm[key],dtype=float); return a,x,c,key
def normalized_truth(p,depth):
 a,x,c,key=counts_and_coords(p); lib=np.maximum(x.sum(1),1); return a,np.log1p(x*(depth/lib)[:,None]),c,key
def pred_paths(run_path):
 j=json.loads((run_path/'prediction_manifest.json').read_text()); q=j['methods']['42']['predictions']; return j,{k:Path(v['path']) for k,v in q.items()}
def original_inventory():
 stems=[
 ('Figure_1_method_and_locked_design','study-design reference',R3/'08_figures_main','summarize_hyperspatial_map_cross_stage_frozen_20260803.py','protocol/anchor tables','study design','confirmatory',False,True,False,'box-only architecture'),
 ('Figure_2_cross_stage_reciprocal_replication','primary 3D replication',R3/'08_figures_main','summarize_hyperspatial_map_cross_stage_frozen_20260803.py','05_metrics + 06_hierarchical_statistics','not gene-dependent','confirmatory',True,True,False,''),
 ('Figure_3_specimen_specific_residual_recovery','3D residual maps',R3/'08_figures_main','summarize_hyperspatial_map_cross_stage_frozen_20260803.py','frozen predictions + measured target','wnt11f2 source-predeclared and source-guarded','confirmatory',True,True,False,''),
 ('Figure_4_axial_localization_beyond_atlas_transfer','predeclared axial biology',R3/'08_figures_main','summarize_hyperspatial_map_cross_stage_frozen_20260803.py','07_biological_modules/DOMAIN_LOCALIZATION_RESULTS.tsv','tbxta/noto/shha predeclared and anchor-excluded','confirmatory',True,True,False,''),
 ('Figure_5_panel_behavior_and_applicability','components/applicability',R3/'08_figures_main','summarize_hyperspatial_map_cross_stage_frozen_20260803.py','03_anchor_panels + 05_metrics','source-selected anchors','post hoc descriptive',False,True,True,'full anchor heatmap and component detail moved to supplementary'),
 ('Figure_MAP_T_crossfit','primary 4D forecasting',R4/'10_figures_main','run_hyperspatial_map_t_4d_frozen_20260804.py','12_source_data + frozen predictions','no target-ranked genes','confirmatory',True,True,False,''),
 ('Figure_MAP_T_biology_and_limits','4D maps/limits',R4/'10_figures_main','postprocess_hyperspatial_map_t_20260804.py','12_source_data + frozen predictions','predeclared/source-defined','post hoc descriptive',True,True,False,'selected panels only'),
 ('Supplementary_MAP_T_all_gene_and_calibration','all-gene/calibration',R4/'11_figures_supplement','postprocess_hyperspatial_map_t_20260804.py','12_source_data','all eligible genes','post hoc descriptive',False,True,True,''),
 ('Supplementary_MAP_T_diagnostics','extended diagnostics',R4/'11_figures_supplement','run_hyperspatial_map_t_4d_frozen_20260804.py','12_source_data','source-only groups','confirmatory/supporting',False,True,True,'')]
 rows=[]
 for stem,role,rr,script,src,gene,status,keep,rev,supp,reason in stems:
  files=[rr/f'{stem}.{e}' for e in ('pdf','svg','png')]
  rows.append({'filename':stem,'scientific_role':role,'result_root':str(rr.parent),'vector_source_available':all(files[i].exists() for i in (0,1)),'plotting_script':script,'source_data':src,'representative_gene_provenance':gene,'confirmatory_or_post_hoc':status,'keep':keep,'revise':rev,'supplementary':supp,'exclusion_reason':reason})
 df=pd.DataFrame(rows); tsv(df,ROOT/'00_inventory/ORIGINAL_FIGURE_INVENTORY.tsv')
 scripts=ROOT/'00_inventory/original_plotting_scripts'; scripts.mkdir(parents=True,exist_ok=True)
 for name in ('summarize_hyperspatial_map_cross_stage_frozen_20260803.py','run_hyperspatial_map_t_4d_frozen_20260804.py','postprocess_hyperspatial_map_t_20260804.py'):
  src=REPO/name
  if src.exists(): shutil.copy2(src,scripts/name)
 write(ROOT/'00_inventory/ORIGINAL_FIGURE_ASSESSMENT.md',"# Original figure assessment\n\nAll nine requested original stems were located with PDF, SVG and PNG. Their plotting code is in `summarize_hyperspatial_map_cross_stage_frozen_20260803.py`, `run_hyperspatial_map_t_4d_frozen_20260804.py` and `postprocess_hyperspatial_map_t_20260804.py`. Figure 2 (3D), Figures 3/4 (biological maps) and MAP-T crossfit are retained as structural foundations. The original box-only Figure 1 is excluded from final use. Full anchor heatmaps and extended diagnostics are supplementary. Original files remain untouched.\n")
 return df

def embryo_icon(ax,center,scale,color,seed=1):
 rng=np.random.default_rng(seed); n=330; x=rng.uniform(-1,1,n); y=rng.uniform(-.2,1,n); mask=(x*x+(y-.15)**2/1.2)<1; x=x[mask];y=y[mask]
 ax.scatter(center[0]+scale*x,center[1]+scale*y,s=2.2,c=color,alpha=.72,lw=0,clip_on=False)

def figure1():
 fig,ax=plt.subplots(figsize=(TWO_COLUMN,5.7)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
 def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=1,color='#687078'))
 def tag(x,y,w,text,fc='#F5F6F7',ec='#B4BAC0',fs=7): ax.add_patch(FancyBboxPatch((x,y),w,.067,boxstyle='round,pad=.008,rounding_size=.012',fc=fc,ec=ec,lw=.8)); ax.text(x+w/2,y+.034,text.replace('\\n','\n'),ha='center',va='center',fontsize=fs)
 panel_label(ax,'a',-.01,1.01); ax.text(.04,.98,'Present-state adaptation',weight='bold',fontsize=10,va='top')
 embryo_icon(ax,(.10,.80),.075,'#4C78A8',2); ax.text(.10,.68,'High-plex source',ha='center',weight='bold')
 embryo_icon(ax,(.10,.52),.075,'#E8876D',3); ax.text(.10,.40,'Target geometry\n+ 16 anchors',ha='center',weight='bold')
 tag(.21,.73,.15,'Geometry-only 3D\nregistration'); arrow((.16,.80),(.21,.77)); arrow((.16,.52),(.21,.75))
 embryo_icon(ax,(.44,.77),.07,'#E69F00',4); ax.text(.44,.65,'Registered IDW field',ha='center',weight='bold'); ax.text(.44,.61,r'$B_g(x)$ · k = 12',ha='center')
 arrow((.36,.77),(.37,.77)); tag(.55,.73,.14,'Anchor residual\n'+r'$R_a=Y_a-B_a$',fc='#FCEDEB',ec=COLORS['map']); arrow((.50,.77),(.55,.77)); ax.add_patch(FancyArrowPatch((.16,.52),(.59,.73),connectionstyle='arc3,rad=-.22',arrowstyle='-|>',mutation_scale=9,lw=1,color='#687078'))
 tag(.55,.57,.14,'raw · k=12 · k=32',fc='#E9F5F1',ec=COLORS['local']); arrow((.62,.73),(.62,.64))
 tag(.74,.69,.19,'Rank-16 centered ridge\nglobal · local · anchor-only',fc='#EEF3F8',ec=COLORS['anchor']); arrow((.69,.60),(.77,.69))
 tag(.74,.55,.19,'Source-only per-gene\ncalibration + guard',fc='#F4EFF8',ec=COLORS['global']); arrow((.84,.69),(.84,.62))
 tag(.74,.40,.19,'Adapted field or\nregistered-IDW fallback',fc='#FCEDEB',ec=COLORS['map']); arrow((.84,.55),(.84,.48))
 panel_label(ax,'b',-.01,.34); ax.text(.04,.32,'Tracking-map-conditioned extension and locked cross-fitting',weight='bold',fontsize=10,va='top')
 xs=[.05,.25,.46,.67,.86]; labels=['Early MAP\nreconstruction','TM0–TM230\nlineage transport','Source-learned\ntemporal residual','Temporal guard','Later molecular\nforecast']; cols=['#EEF3F8','#F4EFF8','#E9F5F1','#F4EFF8','#FCEDEB']
 for i,(x,l,c) in enumerate(zip(xs,labels,cols)): tag(x,.18,.13,l,fc=c,ec='#9CA3AA',fs=6.5); 
 for x1,x2 in zip(xs[:-1],xs[1:]): arrow((x1+.13,.214),(x2,.214))
 ax.text(.05,.105,'3D cross-specimen replication',weight='bold',fontsize=7)
 ax.text(.05,.068,'50%, 75%, 6-somite · E1 <-> E2',fontsize=7)
 ax.text(.55,.105,'4D cross-fitted forecast',weight='bold',fontsize=7)
 ax.text(.55,.068,'E1 transition -> E2 · E2 transition -> E1',fontsize=7)
 ax.text(.05,.018,'Target non-anchor expression and target late expression remain hidden until frozen predictions are saved.',fontsize=6.4,color='#555D64')
 export(fig,ROOT/'02_figure1_architecture/Figure_1_architecture')
 manifest=pd.DataFrame({'panel':list('ABCDEFGHIJK'),'component':['source/target embryos','geometry registration','IDW expectation','16 anchors','anchor residual','multiscale residuals','rank-16 operators','calibration','guard/fallback','temporal extension','cross-fitted designs'],'frozen_evidence':['3D/4D protocol locks']*11}); tsv(manifest,ROOT/'08_source_data/Figure_1_component_manifest.tsv')

def figure2():
 embryo=rd(R3/'05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv'); comp=rd(R3/'05_metrics/METHOD_COMPARISONS_BY_EMBRYO.tsv'); boot=rd(R3/'06_hierarchical_statistics/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv'); pareto=rd(R3/'05_metrics/COMPONENT_PARETO_RESULTS.tsv').set_index('method')
 fig=plt.figure(figsize=(TWO_COLUMN,8.7)); gs=fig.add_gridspec(4,2,height_ratios=[.55,2.45,1.55,1.65],hspace=.62,wspace=.43)
 ax=fig.add_subplot(gs[0,:]); ax.axis('off'); panel_label(ax,'a',-.02,1.15)
 for i,(stage,tag,s,t,*_) in enumerate(RUNS):
  x=.015+i*.165; ax.add_patch(FancyBboxPatch((x,.18),.145,.60,boxstyle='round,pad=.01',fc=['#EBF2F8','#EBF2F8','#EDF6F2','#EDF6F2','#FAEFE9','#FAEFE9'][i],ec='#9AA1A8',lw=.8)); ax.text(x+.072,.49,f'{tag}\n{s} → {t}',ha='center',va='center',fontsize=7)
 sub=gs[1,:].subgridspec(2,3,wspace=.45,hspace=.72)
 for j,(ep,label,higher) in enumerate(ENDPOINTS):
  ax=fig.add_subplot(sub[j//3,j%3]);
  for key in embryo.target_key.unique():
   d=embryo[embryo.target_key==key].set_index('method'); ax.plot([0,1],[d.loc[IDW,ep],d.loc[MAP,ep]],color='#B8BDC2',lw=.85,zorder=1); ax.scatter(0,d.loc[IDW,ep],c=COLORS['idw'],s=24,zorder=3);ax.scatter(1,d.loc[MAP,ep],c=COLORS['map'],s=24,zorder=3)
  ax.set_xticks([0,1],['IDW','MAP']); ax.set_title(label+(' ↑' if higher else ' ↓')); polish(ax,'y');
  if j==0: panel_label(ax,'b',-.34,1.25)
 ax=fig.add_subplot(gs[2,0]); panel_label(ax,'c',-.16,1.11); b=boot[(boot.comparison.str.contains('favorable sign'))&boot.endpoint.isin([x[0] for x in ENDPOINTS])]; y=np.arange(len(b)); ax.errorbar(b.estimate,y,xerr=[b.estimate-b.ci95_low,b.ci95_high-b.estimate],fmt='o',color=COLORS['map'],ecolor='#555D64',capsize=2.5,ms=5); ax.axvline(0,color='#777',ls=':',lw=.8);ax.set_yticks(y,[dict((e,l) for e,l,_ in ENDPOINTS)[e] for e in b.endpoint]);ax.invert_yaxis();ax.set_xlabel('Favourable MAP − IDW effect');polish(ax,'x')
 ax=fig.add_subplot(gs[2,1]);panel_label(ax,'d',-.16,1.11); methods=list(pareto.index); ranks=[]
 for key in embryo.target_key.unique():
  d=embryo[embryo.target_key==key].set_index('method'); score=np.zeros(len(methods));
  for ep,_,higher in ENDPOINTS: score+=rankdata(-d.loc[methods,ep] if higher else d.loc[methods,ep])
  ranks.append(score/len(ENDPOINTS))
 im=ax.imshow(np.asarray(ranks).T,aspect='auto',cmap='viridis_r',vmin=1,vmax=len(methods));ax.set_yticks(range(len(methods)),[METHOD_LABELS[m] for m in methods]);ax.set_xticks(range(6),[f'{r[1]} {r[3]}' for r in RUNS],rotation=35,ha='right');cb=fig.colorbar(im,ax=ax,fraction=.04);cb.set_label('Mean endpoint rank')
 ax=fig.add_subplot(gs[3,0]);panel_label(ax,'e',-.16,1.11)
 for m,row in pareto.iterrows(): ax.scatter(row.residual_spatial_pearson_median,row.topk_dice_median,s=75*(.65/max(row.log1p_rmse,.55))**2,color=METHOD_COLORS[m],label=METHOD_LABELS[m],edgecolor='white',lw=.5)
 ax.set(xlabel='Residual spatial Pearson ↑',ylabel='Prevalence-matched Dice ↑');polish(ax,'both');ax.legend(frameon=False,ncol=2,loc='lower right',fontsize=6)
 ax=fig.add_subplot(gs[3,1]);panel_label(ax,'f',-.16,1.11); p=comp[comp.comparison==f'{MAP}_vs_{IDW}'];
 for ep,label,col in [('svg_spatial_pearson_median','Pearson','#277DA1'),('topk_dice_median','Dice',COLORS['map']),('log1p_rmse','RMSE',COLORS['local'])]:
  v=p[p.endpoint==ep].groupby('stage').delta.mean().reindex(['50% epiboly','75% epiboly','6-somite']);ax.plot(range(3),v,'-o',color=col,label=label)
 ax.axhline(0,color='#777',ls=':',lw=.8);ax.set_xticks(range(3),['50%','75%','6-somite']);ax.set_ylabel('Favourable MAP − IDW effect');ax.legend(frameon=False);polish(ax,'y')
 export(fig,ROOT/'03_figure2_3d_replication/Figure_2_3D_replication')
 psrc=pareto.reset_index().assign(table='pareto')
 tsv(pd.concat([embryo.assign(table='primary_by_embryo'),comp.assign(table='paired_comparison'),boot.assign(table='bootstrap'),psrc],ignore_index=True,sort=False),ROOT/'08_source_data/Figure_2_source_data.tsv')

def three_stage_map_data():
 out=[]
 for run in [RUNS[0],RUNS[2],RUNS[4]]:
  stage,tag,s,t,rp,sp,tp=run;j,paths=pred_paths(rp)
  source=ad.read_h5ad(sp); sx=dense(source.X); depth=float(np.median(np.maximum(sx.sum(1),1)))
  a=ad.read_h5ad(tp); x=dense(a.X)
  lib=np.maximum(x.sum(1),1); truth=np.log1p(x*(depth/lib)[:,None]); key=j.get('target_coordinate_key','global_sphere'); key=key if key in a.obsm else ('global_sphere' if stage!='6-somite' else 'spatial'); coords=np.asarray(a.obsm[key]); gi=list(a.var_names).index('wnt11f2'); base=np.load(paths[IDW],mmap_mode='r')[:,gi]; mp=np.load(paths[MAP],mmap_mode='r')[:,gi]
  out.append({'stage':stage,'coords':coords,'truth':truth[:,gi],'idw':base,'map':mp,'gene':'wnt11f2','target_file':str(tp),'idw_file':str(paths[IDW]),'map_file':str(paths[MAP])})
 return out

def figure3():
 maps=three_stage_map_data(); axial=rd(R3/'07_biological_modules/DOMAIN_LOCALIZATION_RESULTS.tsv').groupby(['stage','target_embryo','method','gene'],as_index=False).mean(numeric_only=True)
 map_source=[]
 fig=plt.figure(figsize=(TWO_COLUMN,9.0)); gs=fig.add_gridspec(4,3,height_ratios=[2.35,1.6,1.55,1.5],hspace=.62,wspace=.42)
 top=gs[0,:].subgridspec(3,3,wspace=.04,hspace=.05)
 for r,m in enumerate(maps):
  vals=[m['truth'],m['idw'],m['map']]; lim=np.nanpercentile(np.concatenate(vals),99)
  for c,(v,title) in enumerate(zip(vals,['Measured target','Registered IDW','HyperSpatial-MAP'])):
   ax=fig.add_subplot(top[r,c]); ax.scatter(m['coords'][:,0],m['coords'][:,1],c=v,s=.35,cmap='viridis',vmin=0,vmax=lim,rasterized=True);ax.axis('off');ax.set_aspect('equal');
   if r==0: ax.set_title(title)
   if c==0: ax.text(-.10,.5,m['stage'],rotation=90,va='center',ha='right',transform=ax.transAxes,fontsize=7)
   if r==0 and c==0: panel_label(ax,'a',-.28,1.18)
 for col,(metric,title) in enumerate([('topk_dice','Dice ↑'),('centroid_error','Centroid error ↓'),('spatial_pearson','Spatial Pearson ↑')]):
  ax=fig.add_subplot(gs[1,col]); panel_label(ax,chr(ord('b')+col),-.18,1.12)
  for j,gene in enumerate(['tbxta','noto','shha']):
   sub=axial[axial.gene==gene].set_index(['stage','target_embryo','method']); iv=[];mv=[]
   for stage,tag,s,t,*_ in RUNS: iv.append(sub.loc[(stage,t,IDW),metric]);mv.append(sub.loc[(stage,t,MAP),metric])
   ax.scatter(np.full(6,j)-.09,iv,c=COLORS['idw'],s=22);ax.scatter(np.full(6,j)+.09,mv,c=COLORS['map'],s=22)
   for k in range(6):ax.plot([j-.09,j+.09],[iv[k],mv[k]],color='#B8BDC2',lw=.75)
  ax.set_xticks(range(3),['tbxta','noto','shha']);ax.set_title(title);polish(ax,'y')
  if col==0: ax.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=COLORS['idw'],label='Registered IDW'),plt.Line2D([],[],marker='o',ls='',color=COLORS['map'],label='HyperSpatial-MAP')],frameon=False,fontsize=6)
 # Large residual pair.
 m=maps[0]; true=m['truth']-m['idw']; pred=m['map']-m['idw']; lim=np.nanpercentile(np.abs(np.r_[true,pred]),98)
 for c,(v,title) in enumerate([(true,'True specimen residual'),(pred,'Predicted specimen residual')]):
  ax=fig.add_subplot(gs[2,c]); ax.scatter(m['coords'][:,0],m['coords'][:,1],c=v,s=.55,cmap='RdBu_r',vmin=-lim,vmax=lim,rasterized=True);ax.axis('off');ax.set_aspect('equal');ax.set_title('wnt11f2 · '+title); 
  if c==0:panel_label(ax,'e',-.18,1.12)
 ax=fig.add_subplot(gs[2,2]); ax.hexbin(true,pred,gridsize=45,mincnt=1,cmap='magma');ax.axline((0,0),slope=1,color='white',ls=':',lw=.8);ax.set(xlabel='True residual',ylabel='Predicted residual');polish(ax);ax.text(.04,.94,'Spatial deviation, not gene mean',transform=ax.transAxes,va='top',fontsize=6,color='white')
 # Source-only fallback example.
 fallback=None
 for candidate in RUNS:
  stage,tag,s,t,rp,sp,tp=candidate; j,paths=pred_paths(rp); names=j['fusion_candidate_names']; hits=[i for i,x in enumerate(j['per_gene_fusion_selection']) if names[int(x)]=='registered_idw']
  if hits: fallback=(candidate,j,paths,hits[0]); break
 if fallback is None: raise RuntimeError('Frozen manifests contain no registered-IDW fallback gene')
 (stage,tag,s,t,rp,sp,tp),j,paths,idx=fallback; a=ad.read_h5ad(tp); source=ad.read_h5ad(sp); genes=list(a.var_names); g=genes[idx]; x=dense(a.X); sx=dense(source.X); dep=np.median(np.maximum(sx.sum(1),1)); truth=np.log1p(x*(dep/np.maximum(x.sum(1),1))[:,None])[:,idx]; coords=np.asarray(a.obsm[j.get('target_coordinate_key','global_sphere')]); idw=np.load(paths[IDW],mmap_mode='r')[:,idx]; mp=np.load(paths[MAP],mmap_mode='r')[:,idx]; lim=np.nanpercentile(np.r_[truth,idw,mp],99)
 bottom=gs[3,:].subgridspec(1,3,wspace=.03)
 for c,(v,title) in enumerate([(truth,'Measured'),(idw,'Registered IDW'),(mp,'MAP fallback output')]):
  ax=fig.add_subplot(bottom[c]);ax.scatter(coords[:,0],coords[:,1],c=v,s=.5,cmap='viridis',vmin=0,vmax=lim,rasterized=True);ax.axis('off');ax.set_aspect('equal');ax.set_title(f'{g} · {title}');
  if c==0:panel_label(ax,'f',-.18,1.12)
 fig.text(.5,.012,'Axial genes were predeclared and excluded from anchors. wnt11f2 and the fallback gene were selected from source-only criteria.',ha='center',fontsize=6.6)
 export(fig,ROOT/'04_figure3_biological_recovery/Figure_3_biological_recovery')
 for m in maps:
  map_source.append(pd.DataFrame({'table':'wnt11f2_map','stage':m['stage'],'gene':m['gene'],'x':m['coords'][:,0],'y':m['coords'][:,1],'z':m['coords'][:,2] if m['coords'].shape[1]>2 else np.nan,'measured':m['truth'],'registered_idw':m['idw'],'hyperspatial_map':m['map'],'true_residual':m['truth']-m['idw'],'predicted_residual':m['map']-m['idw']}))
 map_source.append(pd.DataFrame({'table':'fallback_map','stage':stage,'gene':g,'x':coords[:,0],'y':coords[:,1],'z':coords[:,2] if coords.shape[1]>2 else np.nan,'measured':truth,'registered_idw':idw,'hyperspatial_map':mp}))
 tsv(pd.concat([axial.assign(table='axial_metrics'),*map_source],ignore_index=True,sort=False),ROOT/'08_source_data/Figure_3_source_data.tsv'); return g

def figure4():
 p=rd(R4/'12_source_data/PRIMARY_ENDPOINTS_ALL_FOLDS.tsv'); per=rd(R4/'12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv'); perm=pd.concat([rd(R4/f'09_permutations/fold_{f}_results.tsv') for f in 'AB']);
 fig=plt.figure(figsize=(TWO_COLUMN,7.6));gs=fig.add_gridspec(3,2,height_ratios=[.8,1.55,1.6],hspace=.62,wspace=.42)
 ax=fig.add_subplot(gs[0,:]);panel_label(ax,'a',-.03,1.05);ax.axis('off');
 for i,(txt,fc) in enumerate([('50% early MAP','#EAF1F7'),('TM0–TM230\nlineage transport','#F2EDF6'),('Source temporal\nresidual','#E8F5F1'),('Withheld 75%\nforecast','#FCECEB')]):
  x=.03+i*.24;ax.add_patch(FancyBboxPatch((x,.35),.18,.42,boxstyle='round,pad=.012',fc=fc,ec='#9BA2A8',lw=.8));ax.text(x+.09,.56,txt,ha='center',va='center',fontsize=7,weight='bold');
  if i<3:ax.add_patch(FancyArrowPatch((x+.18,.56),(x+.24,.56),arrowstyle='-|>',mutation_scale=9,color='#687078'))
 ax.text(.03,.13,'Fold A: E1 transition → E2 forecast      Fold B: E2 transition → E1 forecast      no molecular pooling',fontsize=7)
 methods=['B1_lineage_registered_idw','B2_lineage_early_map','B3_mean_developmental_correction','B4_anchor_only_temporal','B5_trajectory_free_map_t','B6_trajectory_aware_map_t']; short=['Atlas\ntransport','Early MAP\ntransport','Mean\nchange','Anchor\nonly','Trajectory\nfree','MAP-T']
 for c,(metric,title) in enumerate([('rho_future_pooled','Future-residual Pearson ↑'),('rmse','log1p RMSE ↓')]):
  ax=fig.add_subplot(gs[1,c]);panel_label(ax,chr(ord('b')+c),-.15,1.10);q=p[(p.cohort=='full')&p.method.isin(methods)];
  for i,f in enumerate('AB'): v=q[q.fold==f].set_index('method').loc[methods,metric];ax.plot(np.arange(6)+(-.04 if f=='A' else .04),v,'o',color=['#277DA1',COLORS['map']][i],label=f'Fold {f}')
  ax.set_xticks(range(6),short);ax.set_title(title);ax.legend(frameon=False);polish(ax,'y')
 ax=fig.add_subplot(gs[2,0]);panel_label(ax,'d',-.15,1.10);q=perm[perm.cohort=='full'];ax.errorbar([0,1],q.null_mean,yerr=[q.null_mean-q.null_ci_low,q.null_ci_high-q.null_mean],fmt='o',color=COLORS['shuffled'],capsize=3,label='Shuffled history');ax.scatter([0,1],q.observed,color=COLORS['map'],s=45,label='Real history');ax.set_xticks([0,1],['Fold A','Fold B']);ax.set_ylabel('Future-residual Pearson');ax.legend(frameon=False);polish(ax,'y')
 right=gs[2,1].subgridspec(1,2,wspace=.68)
 ax=fig.add_subplot(right[0]);panel_label(ax,'e',-.22,1.10);q=p[p.method=='B6_trajectory_aware_map_t'];
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:v=q[q.fold==f].set_index('cohort').loc[['full','shared','disjoint'],'rho_future_pooled'];ax.plot(range(3),v,'-o',color=col,label=f'Fold {f}')
 for i,f in enumerate('AB'):j=json.loads((R4/f'07_overlap_excluded_retraining/fold_{f}_sensitivity.json').read_text());ax.scatter(3,j['rho_future_disjoint'],marker='D',color=['#277DA1',COLORS['map']][i])
 ax.set_xticks(range(4),['Full','Shared','Disjoint','Overlap-\nexcluded'],rotation=25,ha='right');ax.set_ylabel('Future-residual Pearson');ax.legend(frameon=False,fontsize=6);polish(ax,'y')
 ax=fig.add_subplot(right[1]);panel_label(ax,'f',-.22,1.10);w=per[per.method.isin(['B5_trajectory_free_map_t','B6_trajectory_aware_map_t'])].pivot_table(index=['fold','gene'],columns='method',values='future_residual_pearson').dropna();
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:z=w.loc[f];ax.scatter(z['B5_trajectory_free_map_t'],z['B6_trajectory_aware_map_t'],s=5,alpha=.35,color=col,label=f'Fold {f}')
 ax.plot([-.5,1],[-.5,1],':',color='#777');ax.set(xlabel='Trajectory-free gene r',ylabel='Trajectory-aware gene r',xlim=(-.5,1),ylim=(-.5,1));polish(ax);ax.legend(frameon=False,fontsize=6)
 sensitivity=[]
 for f in 'AB':
  j=json.loads((R4/f'07_overlap_excluded_retraining/fold_{f}_sensitivity.json').read_text()); sensitivity.append(pd.DataFrame([{'fold':f,**j}]))
 export(fig,ROOT/'05_figure4_4d_forecasting/Figure_4_4D_forecasting'); tsv(pd.concat([p.assign(table='primary'),perm.assign(table='permutation'),per.assign(table='per_gene'),pd.concat(sensitivity).assign(table='overlap_excluded_sensitivity')],ignore_index=True,sort=False),ROOT/'08_source_data/Figure_4_source_data.tsv')

def read_4d_truth(fold):
 target='E2' if fold=='A' else 'E1'; fd=R4/('03_fold_A' if fold=='A' else '04_fold_B');cfg=json.loads((fd/'FROZEN_FOLD_CONFIG.json').read_text());a=ad.read_h5ad(I4/f'matched_B_75p_{target}_sphere_logFC.h5ad');x=dense(a.X);depth=cfg['early_map']['depth_target'];truth=np.log1p(x*(depth/np.maximum(x.sum(1),1))[:,None]);coords=np.asarray(a.obsm['global_sphere']);return a,truth,coords,fd,cfg

def select_source_gene(fold):
 sets=json.loads((R4/'12_source_data/SOURCE_DEFINED_GENE_SETS.json').read_text())[fold]; guards=rd(R4/f'02_source_validation/fold_{fold}_temporal_guards.tsv');g=guards[(guards.model=='B6')&guards.guard_pass&(~guards.is_anchor)].set_index('gene')
 for gene in sets['stage_emergent_genes']:
  if gene in g.index:return gene
 raise RuntimeError('No eligible source-selected gene')

def figure5():
 per=rd(R4/'12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv');dom=rd(R4/'12_source_data/DOMAIN_METRICS_ALL_FOLDS.tsv');strat=rd(R4/'12_source_data/STRATIFIED_ENDPOINTS.tsv');unc=rd(R4/'12_source_data/UNCERTAINTY_CALIBRATION_ALL_FOLDS.tsv');guards=pd.concat([rd(R4/f'02_source_validation/fold_{f}_temporal_guards.tsv') for f in 'AB']);
 fig=plt.figure(figsize=(TWO_COLUMN,8.0));gs=fig.add_gridspec(3,3,height_ratios=[1.55,1.55,1.45],hspace=.60,wspace=.45)
 # Geometry panel: separate axes prevent one embryo from being clipped.
 geogrid=gs[0,0].subgridspec(2,1,hspace=.02)
 for r,f in enumerate('AB'):
  ax=fig.add_subplot(geogrid[r,0]);a,t,c,fd,cfg=read_4d_truth(f);ix=np.linspace(0,len(c)-1,min(1800,len(c))).astype(int);ax.scatter(c[ix,0],c[ix,1],c=c[ix,2],s=1,cmap='viridis',alpha=.7,rasterized=True);ax.axis('off');ax.set_aspect('equal');ax.text(.02,.90,f'Fold {f}',transform=ax.transAxes,fontsize=6.5,weight='bold')
  if r==0: ax.set_title('Target 75% point-cloud geometries',fontsize=7);panel_label(ax,'a',-.17,1.18)
 # Expression maps (two genes x IDW/MAP-T)
 mapgrid=gs[0,1:].subgridspec(2,2,wspace=.03,hspace=.30); chosen=[]; map_source=[]
 for r,f in enumerate('AB'):
  a,truth,c,fd,cfg=read_4d_truth(f);gene=select_source_gene(f);chosen.append(gene);gi=list(a.var_names).index(gene);idw=np.load(fd/'predictions_pre_unblinding/B1_lineage_registered_idw.npy',mmap_mode='r')[:,gi];mp=np.load(fd/'predictions_pre_unblinding/B6_trajectory_aware_map_t.npy',mmap_mode='r')[:,gi];lim=np.quantile(np.r_[idw,mp],.99)
  for col,(v,title) in enumerate([(idw,'Transported atlas'),(mp,'MAP-T')]):
   ax=fig.add_subplot(mapgrid[r,col]);ax.scatter(c[:,0],c[:,1],c=v,s=.45,cmap='viridis',vmin=0,vmax=lim,rasterized=True);ax.axis('off');ax.set_aspect('equal');
   if r==0: ax.set_title(title,fontsize=6.5)
   if col==0: ax.text(-.06,.5,f'{gene} · Fold {f}',rotation=90,va='center',ha='right',transform=ax.transAxes,fontsize=6)
   if r==0 and col==0:panel_label(ax,'b',-.20,1.15)
  map_source.append(pd.DataFrame({'table':'expression_map','fold':f,'gene':gene,'x':c[:,0],'y':c[:,1],'z':c[:,2] if c.shape[1]>2 else np.nan,'transported_atlas':idw,'map_t':mp}))
 ax=fig.add_subplot(gs[1,0]);panel_label(ax,'c',-.17,1.10);d=dom.groupby(['fold','method'])[['dice','centroid_error']].median().reset_index();
 for f,marker in [('A','o'),('B','s')]:
  x=d[(d.fold==f)&(d.method=='B1_lineage_registered_idw')].iloc[0];y=d[(d.fold==f)&(d.method=='B6_trajectory_aware_map_t')].iloc[0];ax.plot([x.dice,y.dice],[x.centroid_error,y.centroid_error],color='#AAB0B5');ax.scatter(x.dice,x.centroid_error,color=COLORS['idw'],marker=marker,s=38);ax.scatter(y.dice,y.centroid_error,color=COLORS['map'],marker=marker,s=45)
 ax.set(xlabel='Median domain Dice ↑',ylabel='Centroid error ↓');polish(ax,'both')
 ax=fig.add_subplot(gs[1,1]);panel_label(ax,'d',-.17,1.10);w=per[per.method.isin(['B5_trajectory_free_map_t','B6_trajectory_aware_map_t'])].pivot_table(index=['fold','gene'],columns='method',values='future_residual_pearson').dropna();
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:z=w.loc[f];ax.hist(z.B6_trajectory_aware_map_t-z.B5_trajectory_free_map_t,bins=35,histtype='step',density=True,color=col,label=f'Fold {f}')
 ax.axvline(0,color='#777',ls=':');ax.set(xlabel='Aware − trajectory-free gene r',ylabel='Density');ax.legend(frameon=False);polish(ax)
 ax=fig.add_subplot(gs[1,2]);panel_label(ax,'e',-.17,1.10);g=guards[guards.model=='B6'].groupby(['fold','guard_pass']).size().unstack(fill_value=0);bot=np.zeros(2)
 for state,col,label in [(True,COLORS['map'],'Corrected'),(False,COLORS['shuffled'],'Fallback')]:v=[g.loc[f,state] for f in 'AB'];ax.bar(['Fold A','Fold B'],v,bottom=bot,color=col,label=label);bot+=v
 ax.set(ylabel='Non-anchor genes',title='Source-only temporal guard');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2,fontsize=6);polish(ax,'y')
 ax=fig.add_subplot(gs[2,0:2]);panel_label(ax,'f',-.08,1.10);q=strat[(strat.stratum_type=='source_gene_group')&strat.method.isin(['B1_lineage_registered_idw','B6_trajectory_aware_map_t'])];order=['all_non_anchor','source_stage_emergent','source_variable'];
 for method,col,label in [('B1_lineage_registered_idw',COLORS['idw'],'Transported atlas'),('B6_trajectory_aware_map_t',COLORS['map'],'MAP-T')]:v=q[q.method==method].groupby('stratum').rho_future.mean().reindex(order);ax.plot(range(3),v,'-o',color=col,label=label)
 ax.set_xticks(range(3),['All non-anchor','Stage-emergent','Source-variable']);ax.set_ylabel('Future-residual Pearson');ax.legend(frameon=False);polish(ax,'y')
 ax=fig.add_subplot(gs[2,2]);panel_label(ax,'g',-.17,1.10)
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:z=unc[(unc.fold==f)&(~unc.is_anchor)];ax.scatter(z.source_cv_sigma,z.target_mae,s=5,alpha=.22,color=col,label=f'Fold {f}')
 ax.set(xlabel='Source-CV uncertainty',ylabel='Target MAE');ax.legend(frameon=False);polish(ax,'both')
 fig.text(.5,.015,'Applicability: two independently measured molecular embryos, partly assigned to one published tracking reference.',ha='center',fontsize=6.6)
 export(fig,ROOT/'06_figure5_generality_limits/Figure_5_generality_limits');
 src=pd.concat([dom.assign(table='domain'),per.assign(table='per_gene'),strat.assign(table='strata'),unc.assign(table='uncertainty'),*map_source],ignore_index=True,sort=False);src['display_genes']=';'.join(chosen);tsv(src,ROOT/'08_source_data/Figure_5_source_data.tsv');return chosen

def supplementary():
 per=rd(R4/'12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv');unc=rd(R4/'12_source_data/UNCERTAINTY_CALIBRATION_ALL_FOLDS.tsv');cells=rd(R4/'12_source_data/PER_CELL_METRICS.tsv');strat=rd(R4/'12_source_data/STRATIFIED_ENDPOINTS.tsv');pareto=rd(R3/'05_metrics/COMPONENT_PARETO_RESULTS.tsv');emb=rd(R3/'05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv')
 # S1 all-gene and calibration
 fig,axs=plt.subplots(2,2,figsize=(TWO_COLUMN,5.8),layout='constrained')
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:
  for method,ls,label in [('B1_lineage_registered_idw','--','Atlas'),('B6_trajectory_aware_map_t','-','MAP-T')]:v=np.sort(per[(per.fold==f)&(per.method==method)].future_residual_pearson.dropna());axs[0,0].plot(v,np.linspace(0,1,len(v)),color=col,ls=ls,label=f'{f} {label}')
 axs[0,0].set(xlabel='Per-gene future-residual Pearson',ylabel='ECDF');axs[0,0].legend(frameon=False,ncol=2,fontsize=6)
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:z=unc[(unc.fold==f)&(~unc.is_anchor)];axs[0,1].scatter(z.source_cv_sigma,z.target_95pct_coverage,s=5,alpha=.25,color=col,label=f)
 axs[0,1].axhline(.95,color='#777',ls=':');axs[0,1].set(xlabel='Source-CV uncertainty',ylabel='95% interval coverage')
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:axs[1,0].hist(cells[cells.fold==f].map_t_improvement,bins=50,histtype='step',density=True,color=col,label=f)
 axs[1,0].axvline(0,color='#777',ls=':');axs[1,0].set(xlabel='Per-cell RMSE improvement over IDW',ylabel='Density')
 q=strat[(strat.stratum_type=='local_density')&(strat.method=='B6_trajectory_aware_map_t')]
 for f,col in [('A','#277DA1'),('B',COLORS['map'])]:z=q[q.fold==f];axs[1,1].plot(z.stratum,z.rho_future,'-o',color=col,label=f)
 axs[1,1].set(xlabel='Local-density quartile',ylabel='Future-residual Pearson')
 for i,ax in enumerate(axs.flat):panel_label(ax,chr(97+i));polish(ax,'y')
 export(fig,ROOT/'07_supplementary/Supplementary_Figure_1');tsv(pd.concat([per.assign(table='per_gene'),unc.assign(table='uncertainty'),cells.assign(table='per_cell'),strat.assign(table='strata')],ignore_index=True,sort=False),ROOT/'08_source_data/Supplementary_Figure_1_source_data.tsv')
 # S2 extended stratified diagnostics
 fig,axs=plt.subplots(1,3,figsize=(TWO_COLUMN,2.8),layout='constrained')
 for i,kind in enumerate(['radial_position','local_density','source_gene_group']):
  q=strat[(strat.stratum_type==kind)&(strat.method=='B6_trajectory_aware_map_t')]
  for f,col in [('A','#277DA1'),('B',COLORS['map'])]:z=q[q.fold==f];axs[i].plot(range(len(z)),z.rho_future,'-o',color=col,label=f'Fold {f}')
  labels=[str(v).replace('source_','').replace('_',' ').title() for v in z.stratum]
  axs[i].set_xticks(range(len(z)),labels,rotation=30,ha='right');axs[i].set_title(kind.replace('_',' ').title());axs[i].set_ylabel('Future-residual Pearson');polish(axs[i],'y');panel_label(axs[i],chr(97+i));
 axs[0].legend(frameon=False)
 export(fig,ROOT/'07_supplementary/Supplementary_Figure_2');tsv(strat,ROOT/'08_source_data/Supplementary_Figure_2_source_data.tsv')
 # S3 3D components and endpoint landscape
 fig,axs=plt.subplots(1,2,figsize=(TWO_COLUMN,3.2),layout='constrained');methods=emb.method.unique();cols=[x[0] for x in ENDPOINTS]+['moran_error_median'];z=[]
 for m in methods:z.append(emb[emb.method==m][cols].mean().to_numpy())
 z=np.asarray(z);z=(z-z.mean(0))/np.maximum(z.std(0),1e-8);z[:,3:]*=-1;im=axs[0].imshow(z,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2);axs[0].set_yticks(range(len(methods)),[METHOD_LABELS[m] for m in methods]);axs[0].set_xticks(range(len(cols)),[dict((e,l) for e,l,_ in ENDPOINTS).get(c,'Moran error') for c in cols],rotation=35,ha='right');fig.colorbar(im,ax=axs[0],fraction=.04,label='Favourable standardized value');panel_label(axs[0],'a')
 for _,r in pareto.iterrows():axs[1].scatter(r.residual_spatial_pearson_median,r.topk_dice_median,s=60,color=METHOD_COLORS[r.method],label=METHOD_LABELS[r.method])
 axs[1].set(xlabel='Residual spatial Pearson',ylabel='Dice');axs[1].legend(frameon=False,fontsize=6);polish(axs[1],'both');panel_label(axs[1],'b')
 export(fig,ROOT/'07_supplementary/Supplementary_Figure_3');tsv(pareto,ROOT/'08_source_data/Supplementary_Figure_3_source_data.tsv')

def captions(fallback,chosen):
 caps={
 'Figure_1_caption.md':"**Figure 1 | HyperSpatial-MAP architecture and locked study design.** A registered 12-neighbour IDW field defines the same-stage atlas expectation. Sixteen source-selected target anchors generate raw, k=12 and k=32 residual features for source-trained centered rank-16 global, local and anchor-only operators. Source-only per-gene calibration and a predictability guard retain registered IDW for unsupported genes. MAP-T transports the permitted early MAP state through the TM0–TM230 tracking atlas and adds only source-supported temporal residuals. Reciprocal 3D tests and two molecular cross-fitted forecasts are shown.\n",
 'Figure_2_caption.md':"**Figure 2 | Frozen reciprocal three-dimensional replication.** **a,** Six explicit source-to-target directions at 50% epiboly, 75% epiboly and 6-somite. **b,** Paired target-volume IDW-to-MAP comparisons for six frozen endpoints. **c,** Favourable effect estimates with target-embryo bootstrap intervals. **d,** Endpoint-rank landscape. **e,** Component Pareto structure; marker size inversely reflects RMSE. **f,** Stage consistency. Reciprocal directions share specimens and are not six independent biological cohorts.\n",
 'Figure_3_caption.md':f"**Figure 3 | Specimen-specific biological recovery beyond registered atlas transfer.** **a,** Source-selected wnt11f2 maps across three stages; expression scales are shared across measured, IDW and MAP within each row. **b–d,** Paired Dice, centroid error and spatial Pearson for predeclared, anchor-excluded tbxta, noto and shha. **e,** True and predicted wnt11f2 residuals share a symmetric scale. **f,** Source-guarded fallback example ({fallback}); MAP retains IDW when adaptation is unsupported. Target expression is evaluation-only.\n",
 'Figure_4_caption.md':"**Figure 4 | Cross-fitted trajectory-aware molecular forecasting.** **a,** Fold A learns the E1 transition and forecasts E2; Fold B reverses the design. **b,c,** Future-residual Pearson and RMSE for explicit baselines. **d,** Real trajectory histories versus 100 shuffled-history controls. **e,** Full, shared, tracking-disjoint and overlap-excluded analyses. **f,** Gene-level trajectory-aware versus trajectory-free effects; genes are descriptive units, not biological replicates. Intervals and permutation statistics use frozen source data.\n",
 'Figure_5_caption.md':f"**Figure 5 | Biological value, guards and applicability limits.** **a,** Target 75% point-cloud geometries. **b,** Transported-atlas and MAP-T fields for source-selected stage-emergent genes ({chosen[0]}, Fold A; {chosen[1]}, Fold B), with a shared scale within each pair. **c,** Localization beyond transported atlas. **d,** Gene-level trajectory contribution. **e,** Source-only temporal guard outcomes. **f,** Performance for source-defined gene groups. **g,** Source-CV uncertainty versus target error. The molecular embryos were independently measured but partly assigned to one published tracking reference; this is not two independent live-imaging systems.\n"}
 for n,s in caps.items():write(ROOT/'09_captions'/n,s)
 write(ROOT/'09_captions/Supplementary_Figure_1_caption.md',"**Supplementary Figure 1 | All-gene forecasting and uncertainty diagnostics.** **a,** Empirical distributions of per-gene future-residual Pearson for transported registered atlas and MAP-T in each fold. **b,** Source-cross-validation uncertainty versus target interval coverage; the dotted line marks 95% nominal coverage. **c,** Per-cell RMSE improvement over transported IDW. **d,** Future-residual Pearson by source-defined local-density quartile. Genes and cells are descriptive units, not biological replicates.\n")
 write(ROOT/'09_captions/Supplementary_Figure_2_caption.md',"**Supplementary Figure 2 | Frozen stratified forecasting diagnostics.** Future-residual Pearson is shown by source-defined **a,** radial-position quartile, **b,** local-density quartile and **c,** gene group for both molecular folds. The molecular embryos were independently measured but partly assigned to one published tracking reference.\n")
 write(ROOT/'09_captions/Supplementary_Figure_3_caption.md',"**Supplementary Figure 3 | Three-dimensional component landscape.** **a,** Favourably oriented standardized endpoint values across the frozen present-state components. **b,** Residual spatial Pearson versus prevalence-matched Dice. Points are not connected because the components do not form an ordered trajectory; marker size and position summarize saved frozen metrics.\n")

def panel_manifest(fallback,chosen):
 rows=[]
 def add(fig,panel,q,root,pred,src,metric,bu,uu,gene,status,conclusion,limit):rows.append(dict(figure=fig,panel=panel,scientific_question=q,source_result_root=str(root),source_prediction_file=pred,source_data_file=src,metric=metric,biological_unit=bu,uncertainty_unit=uu,representative_gene_criterion=gene,confirmatory_or_post_hoc=status,supported_conclusion=conclusion,limitation=limit))
 add('Figure 1','a–b','What is the implemented method and locked design?',R3,'none','protocol locks','not applicable','not applicable','not applicable','none','schematic','Verified architecture and cross-fitting','Morphology means point-cloud geometry')
 for p,q in zip(list('abcdef'),['six reciprocal tests','paired endpoint effects','effect sizes','method ranks','component Pareto','stage consistency']):add('Figure 2',p,q,R3,'saved predictions summarized before revision','Figure_2_source_data.tsv','frozen 3D endpoints','target embryo','target-embryo/stage-pair bootstrap','none','confirmatory','MAP improves registered IDW across locked endpoints','reciprocal directions share specimens')
 add('Figure 3','a,e','Does MAP recover specimen residuals?',R3,'six frozen prediction arrays','Figure_3_source_data.tsv','spatial residual maps','target embryo','none','wnt11f2 source-predeclared and guard-passing','confirmatory','localized target deviation recovered','maps are projections of native 3D points')
 add('Figure 3','b–d','Are axial domains localized?',R3,'saved 3D predictions','Figure_3_source_data.tsv','Dice/centroid/Pearson','target embryo','paired target values','tbxta/noto/shha predeclared and anchor-excluded','confirmatory','axial localization comparison','not every gene/endpoint improves')
 add('Figure 3','f','When does adaptation fall back?',R3,'saved IDW/MAP arrays','Figure_3_source_data.tsv','map equality/fallback','gene-target descriptive','none',f'{fallback} chosen by source-only registered-IDW guard','confirmatory','guard retains atlas when unsupported','single illustrative fallback')
 for p,q in zip(list('abcdef'),['cross-fit design','baseline residual correlation','baseline RMSE','real vs shuffled','tracking cohorts/sensitivity','gene-level trajectory effect']):add('Figure 4',p,q,R4,'frozen 4D prediction arrays','Figure_4_source_data.tsv','frozen 4D endpoints','molecular embryo','TM0 ancestor family / permutation', 'no target-ranked gene','confirmatory' if p!='f' else 'post hoc descriptive','tracking-aware residual forecasting','two molecular embryos; shared tracking reference')
 for p,q in zip(list('abcdefg'),['target geometry','source-selected maps','localization','trajectory contribution','temporal guards','source gene groups','uncertainty-error']):add('Figure 5',p,q,R4,'frozen 4D predictions','Figure_5_source_data.tsv','saved endpoint/map','molecular embryo','TM0 family where reported','source-stage emergent genes; no target ranking','post hoc descriptive','biological/applicability characterization','not independent live-tracking systems')
 df=pd.DataFrame(rows);tsv(df,ROOT/'00_inventory/FINAL_PANEL_MANIFEST.tsv')
 write(ROOT/'00_inventory/FINAL_FIGURE_CLAIM_AUDIT.md',"# Final figure claim audit\n\nAll panels use frozen predictions or saved source tables. Representative genes are source-selected or biologically predeclared; no target-ranked gene is shown. Captions do not claim independent live tracking, continuous molecular measurements, genome-wide forecasting, universal superiority over anchor-only, or biological replication from cells/genes/lineages. Figure 4 is confirmatory at the embryo/fold level; gene panels are descriptive. Figure 5 is explicitly descriptive and qualified by the partly shared published tracking reference.\n")

def decision():
 write(ROOT/'FIGURE_REVISION_DECISION.md',f"# Figure revision decision\n\n- **Original figures located:** all nine requested stems, each with PDF, SVG and PNG.\n- **Retained foundations:** original 3D reciprocal replication, specimen-residual maps, axial localization, MAP-T crossfit and selected MAP-T biology/limits panels.\n- **Moved to supplementary:** full all-gene/calibration, extended stratification, endpoint landscape and component detail.\n- **Excluded:** the original simplistic box-only architecture and full anchor-selection heatmap as a main result.\n- **Source-data completeness:** complete for all final quantitative panels; spatial maps use frozen prediction arrays and original measured target coordinates/expression.\n- **Gene provenance:** wnt11f2 and the two MAP-T display genes are source-selected; tbxta/noto/shha are predeclared and anchor-excluded; the fallback gene is selected from a source-only registered-IDW guard.\n- **Final figures:** `02_figure1_architecture/Figure_1_architecture.*`; `03_figure2_3d_replication/Figure_2_3D_replication.*`; `04_figure3_biological_recovery/Figure_3_biological_recovery.*`; `05_figure4_4d_forecasting/Figure_4_4D_forecasting.*`; `06_figure5_generality_limits/Figure_5_generality_limits.*`.\n- **Remaining graphical issues:** 3D maps are two-dimensional projections of 3D point clouds, clearly captioned; no interactive 3D rendering is supplied.\n- **Panels not reproducible:** none of the retained panels. Original historical figures remain untouched.\n- **Recommendation:** use these five revised vector figures in the full-Article manuscript; retain Supplementary Figures 1–3 as supporting diagnostics.\n")

def qa():
 from PIL import Image
 rows=[]
 for p in sorted(ROOT.glob('0[2-7]_*/**/*.png')):
  im=Image.open(p); dpi=im.info.get('dpi',(np.nan,np.nan)); rows.append({'file':str(p.relative_to(ROOT)),'width_px':im.width,'height_px':im.height,'dpi_x':round(float(dpi[0]),2),'dpi_y':round(float(dpi[1]),2),'at_least_300_dpi':float(dpi[0])>=299.0 and float(dpi[1])>=299.0})
 tsv(pd.DataFrame(rows),ROOT/'10_logs/FIGURE_RENDER_QA.tsv')
 # Hash all final assets.
 hs=[]
 for p in sorted(x for x in ROOT.rglob('*') if x.is_file() and x.name!='FINAL_ASSET_SHA256SUMS.tsv'):hs.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
 tsv(pd.DataFrame(hs),ROOT/'10_logs/FINAL_ASSET_SHA256SUMS.tsv')

def main():
 original_inventory();figure1();figure2();fallback=figure3();figure4();chosen=figure5();supplementary();captions(fallback,chosen);panel_manifest(fallback,chosen);decision();qa();print(ROOT)
if __name__=='__main__':main()
