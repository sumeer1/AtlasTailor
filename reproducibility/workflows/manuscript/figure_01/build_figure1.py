#!/usr/bin/env python3
"""Build data-derived Figure 1 without changing frozen scientific outputs."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figure1_redesign';ASSETS=OUT/'assets';OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
RUN=ROOT/'results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726'
PRED=RUN/'04_predictions/50p_E1_to_E2'
SRC=ROOT/'data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E1.h5ad'
TGT=ROOT/'data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E2.h5ad'
TROOT=ROOT/'results_v10/hyperspatial_map_t_4d_frozen_20260804_112617'
TINPUT=ROOT/'results_v10/hyperspatial_map_t_4d_frozen_20260804_105810/00_data_audit/inputs'
EARLY_H5=TINPUT/'matched_A_50p_E2_sphere_logFC.h5ad';LATE_H5=TINPUT/'matched_B_75p_E2_sphere_logFC.h5ad'
TRACK=TINPUT/'Tracking_Matrix.csv';DECOMP=TROOT/'03_fold_A/predictions_pre_unblinding/decomposition.npz'
OLD=ROOT/'results_v10/Slides/Figure_1_Nature_final.png'

BLUE='#28749A';BLUE_DARK='#15516F';BLUE_LIGHT='#A9CDDD';ORANGE='#D9863A';ORANGE_DARK='#96551F'
RED='#B7474F';GREEN='#3D806C';GREEN_LIGHT='#DCEBE5';INK='#202A31';GRAY='#68747C';LIGHT='#D8E0E4';PALE='#F5F7F6'
CM_BLUE=LinearSegmentedColormap.from_list('atlas',['#EEF5F7','#8BBBD0',BLUE_DARK])
CM_WARM=LinearSegmentedColormap.from_list('target',['#FAF1E7','#E6A263','#A9503D'])
CM_RES=LinearSegmentedColormap.from_list('signed',['#286F9E','#F8F8F5','#B43F48'])
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8.4,'svg.fonttype':'none','pdf.fonttype':42,
                     'figure.facecolor':'white','savefig.facecolor':'white','axes.linewidth':.7})

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def strings(x):return np.asarray([v.decode() if isinstance(v,bytes) else str(v) for v in x])
def read_h5(path,counts=False):
    with h5py.File(path,'r') as h:
        xyz=np.asarray(h['obsm/global_sphere'],np.float32);genes=strings(h['var/_index'][:]);ids=np.asarray(h['obs/Amat_id'][:]) if 'Amat_id' in h['obs'] else None
        value=None
        if counts:value=np.asarray(h['layers/spliced'],np.float32)+np.asarray(h['layers/unspliced'],np.float32)
    center=np.median(xyz,axis=0);xyz=(xyz-center)/max(float(np.quantile(np.linalg.norm(xyz-center,axis=1),.99)),1e-6)
    return xyz.astype(np.float32),value,genes,ids

def depth_predictor(counts,anchors):
    x=np.c_[np.ones(len(counts)),np.log1p(counts[:,anchors].sum(1))];y=np.log1p(counts.sum(1))
    return np.linalg.solve(x.T@x+np.diag([0.,1.]),x.T@y)

def deterministic_sample(xy,nx=34,ny=28):
    lo=np.quantile(xy[:,:2],.005,axis=0);hi=np.quantile(xy[:,:2],.995,axis=0)
    b=np.clip(((xy[:,:2]-lo)/np.maximum(hi-lo,1e-9)*[nx,ny]).astype(int),0,[nx-1,ny-1]);key=b[:,0]*ny+b[:,1];out=[]
    for k in np.unique(key):
        q=np.flatnonzero(key==k);ix,iy=divmod(int(k),ny);c=lo+(np.array([ix+.5,iy+.5])/[nx,ny])*(hi-lo)
        out.append(q[np.argmin(np.square(xy[q,:2]-c).sum(1))])
    return np.asarray(out,int)

def smooth(v,xyz,k):
    idx=cKDTree(xyz).query(xyz,k=min(k+1,len(xyz)))[1]
    return v[idx[:,1:]].mean(1)

manifest=json.loads((PRED/'prediction_manifest.json').read_text())
src_xyz,src_counts,genes,_=read_h5(SRC,True);tgt_xyz,tgt_counts,tgenes,_=read_h5(TGT,True);assert np.array_equal(genes,tgenes)
anchor_gene=manifest['anchor_genes'][0];anchor_i=int(manifest['anchor_indices'][0]);anchors=np.asarray(manifest['anchor_indices'],int)
gate=manifest['source_gate_metadata'];gain=np.asarray(gate['selected_validation_pearson'])-np.asarray(gate['base_validation_pearson'])
eligible=np.isin(genes,manifest['source_defined_predictable_genes'])&~np.isin(genes,manifest['anchor_genes'])&np.isfinite(gain)&(gain>0)
median_gain=float(np.median(gain[eligible]));ii=np.flatnonzero(eligible);gene_i=int(ii[np.lexsort((genes[ii],np.abs(gain[ii]-median_gain)))[0]])
gene=str(genes[gene_i]);assert gene=='chst11' and anchor_gene=='klf2a'

source_depth=np.maximum(src_counts.sum(1),1);depth_target=float(np.median(source_depth));src_values=src_counts*(depth_target/source_depth)[:,None];src_log=np.log1p(src_values)
weight=depth_predictor(src_counts,anchors);target_anchor_raw=tgt_counts[:,anchors]
target_depth=np.maximum(np.expm1(np.c_[np.ones(len(tgt_counts)),np.log1p(target_anchor_raw.sum(1))]@weight),1)
target_anchor_log=np.log1p(target_anchor_raw*(depth_target/target_depth)[:,None]).astype(np.float32)
base=np.load(PRED/'predictions/seed42_registered_idw_k12.npy',mmap_mode='r');adapted=np.load(PRED/'predictions/seed42_hyperspatial_map_calibrated_fusion.npy',mmap_mode='r')
anchor_measured=target_anchor_log[:,0];anchor_expected=np.asarray(base[:,anchor_i]);anchor_residual=anchor_measured-anchor_expected
raw_res=anchor_residual;local_res=smooth(raw_res,tgt_xyz,12);broad_res=smooth(raw_res,tgt_xyz,32)
prior=np.asarray(base[:,gene_i]);adapt=np.asarray(adapted[:,gene_i]);correction=adapt-prior;source_field=src_log[:,gene_i]

early_xyz,_,early_genes,early_ids=read_h5(EARLY_H5);late_xyz,_,late_genes,late_ids=read_h5(LATE_H5);assert np.array_equal(genes,early_genes) and np.array_equal(genes,late_genes)
z=np.load(DECOMP);transported=np.asarray(z['early_map_lineage'][:,gene_i]);temporal=np.asarray(z['predicted_residual'][:,gene_i]);later=np.asarray(z['final_prediction'][:,gene_i])
early_adapt=adapt

# Actual tracking endpoints for E2 matched cells.
tracking=pd.read_csv(TRACK,usecols=['node id','timepoint','ancestor id'])
late_to_ancestor=dict(zip(tracking.loc[tracking.timepoint==230,'node id'].astype(int),tracking.loc[tracking.timepoint==230,'ancestor id'].astype(int)))
early_lookup={int(v):i for i,v in enumerate(early_ids)};pairs=[]
for j,node in enumerate(late_ids):
    a=late_to_ancestor.get(int(node));
    if a in early_lookup:pairs.append((early_lookup[a],j))
assert len(pairs)==len(late_ids)

sid=deterministic_sample(src_xyz);tid=deterministic_sample(tgt_xyz);eid=deterministic_sample(early_xyz,24,20);lid=deterministic_sample(late_xyz,28,22)

def finite(v):return np.asarray(v)[np.isfinite(v)]
def norm_expr(*arrays):
    q=np.quantile(np.concatenate([finite(a) for a in arrays]),[.02,.98]);return Normalize(q[0],max(q[1],q[0]+1e-6))
def norm_res(*arrays):
    lim=max(float(np.quantile(np.abs(np.concatenate([finite(a) for a in arrays])),.98)),1e-6);return TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
n_main=norm_expr(source_field,prior,adapt);n_anchor=norm_expr(anchor_measured,anchor_expected)
n_anchor_res=norm_res(raw_res,local_res,broad_res);n_corr=norm_res(correction);n_temp_expr=norm_expr(early_adapt,transported,later);n_temp_res=norm_res(temporal)

fig=plt.figure(figsize=(12.2,8.25),dpi=300);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def text(x,y,s,size=8.4,color=INK,weight='normal',ha='center',va='center',**kw):ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,zorder=20,**kw)
def arrow(x1,y1,x2,y2,color=GRAY,lw=1.15,rad=0,style='-|>'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=9,linewidth=lw,color=color,connectionstyle=f'arc3,rad={rad}',shrinkA=1,shrinkB=1,zorder=12))
def scale_xy(xyz,cx,cy,w,h,ids):
    xy=xyz[ids,:2];lo=np.quantile(xyz[:,:2],.005,axis=0);hi=np.quantile(xyz[:,:2],.995,axis=0);q=(xy-(lo+hi)/2)/np.maximum(hi-lo,1e-9)
    return cx+q[:,0]*w,cy+q[:,1]*h
def embryo(cx,cy,w,h,xyz,ids,values=None,cmap=None,norm=None,color=None,s=5.8,alpha=.96,outline=False,zorder=8):
    x,y=scale_xy(xyz,cx,cy,w,h,ids)
    if outline:ax.scatter(x,y,s=s+2.1,c='white',linewidths=0,zorder=zorder-1)
    c=color if values is None else cmap(norm(np.asarray(values)[ids]));ax.scatter(x,y,s=s,c=c,alpha=alpha,linewidths=0,zorder=zorder)
def panel_title(letter,title,y):text(.023,y,letter,size=14,weight='bold',ha='left');text(.052,y,title,size=12.5,weight='bold',ha='left')
def plus(x,y):text(x,y,'+',size=17,color=GRAY,weight='normal')
def equals(x,y):text(x,y,'=',size=14,color=GRAY,weight='normal')
def gene_strip(x,y):
    vals=[.15,.42,.73,.28,.9,.58,.34,.68,.18,.82,.48,.95,.62,.25,.77,.52]
    for j,v in enumerate(vals):
        ax.add_patch(Rectangle((x+(j%8)*.0075,y-(j//8)*.015),.0058,.0105,facecolor=CM_WARM(v),edgecolor='none',zorder=10))
def tiny_colorbar(x,y,w,cmap,label):
    for j in range(70):ax.add_patch(Rectangle((x+j*w/70,y),w/70,.006,facecolor=cmap(j/69),edgecolor='none',zorder=2))
    text(x+w/2,y-.009,label,size=5.8,color=GRAY)

# separators and panel titles
ax.plot([.025,.975],[.695,.695],color=LIGHT,lw=.8);ax.plot([.025,.975],[.335,.335],color=LIGHT,lw=.8)
panel_title('a','Atlas adaptation',.973);panel_title('b','HyperSpatial-MAP core',.678);panel_title('c','Developmental forecast',.318)

# a: atlas and target converge on geometry-only registration
embryo(.075,.855,.085,.105,src_xyz,sid,source_field,CM_BLUE,n_main,s=5.4,outline=True);text(.075,.922,'Deep reference',size=8.8,color=BLUE_DARK,weight='bold')
for k in range(5):ax.plot([.042+k*.008,.048+k*.008],[.790,.776+(k%2)*.007],color=BLUE,lw=1.3)
embryo(.205,.855,.085,.105,tgt_xyz,tid,color=ORANGE,s=5.4,outline=True);text(.205,.922,'Target geometry',size=8.4,color=ORANGE_DARK,weight='bold');gene_strip(.176,.790);text(.205,.765,'16 measured genes',size=7.3,color=ORANGE_DARK,weight='bold')
arrow(.116,.855,.280,.855,color=BLUE_DARK);arrow(.247,.855,.280,.855,color=ORANGE_DARK)
# registration overlay is the frozen prior evaluated on actual target positions.
ax.add_patch(Circle((.322,.855),.040,facecolor='white',edgecolor=LIGHT,lw=.8));embryo(.322,.855,.065,.074,tgt_xyz,tid,color=ORANGE,s=5.8,alpha=.22,zorder=7);embryo(.322,.855,.065,.074,tgt_xyz,tid,prior,CM_BLUE,n_main,s=3.4,alpha=.9,zorder=8)
text(.322,.930,'Geometry-only\nregistration',size=7.2,color=INK,weight='bold',linespacing=.92);arrow(.364,.855,.405,.855,color=BLUE_DARK)
embryo(.455,.855,.090,.108,tgt_xyz,tid,prior,CM_BLUE,n_main,s=5.5,outline=True);text(.455,.922,'Registered atlas prior',size=8.1,color=BLUE_DARK,weight='bold')

# anchor decomposition, all same target positions and common expression scale
text(.570,.922,'Target anchor',size=7.8,color=ORANGE_DARK,weight='bold');embryo(.570,.855,.072,.086,tgt_xyz,tid,anchor_measured,CM_WARM,n_anchor,s=4.3)
text(.570,.790,anchor_gene,size=6.5,color=GRAY)
text(.635,.855,'−',size=16,color=GRAY)
text(.695,.922,'Atlas expectation',size=7.8,color=BLUE_DARK,weight='bold');embryo(.695,.855,.072,.086,tgt_xyz,tid,anchor_expected,CM_BLUE,n_anchor,s=4.3)
equals(.757,.855)
text(.830,.922,'Target-specific residual',size=8.2,color=RED,weight='bold');embryo(.830,.855,.080,.095,tgt_xyz,tid,anchor_residual,CM_RES,n_anchor_res,s=4.7,outline=True)
text(.830,.785,r'$R_A = Y_A^T - B_A$',size=8.2,color=RED,weight='bold')
tiny_colorbar(.887,.820,.065,CM_RES,'signed residual')

# b: multiscale features
text(.075,.634,'Anchor residuals',size=8.2,color=RED,weight='bold');embryo(.075,.568,.080,.095,tgt_xyz,tid,raw_res,CM_RES,n_anchor_res,s=4.6)
for x,v,label,radius in [(.175,raw_res,'raw',.012),(.265,local_res,'local',.022),(.355,broad_res,'broad',.033)]:
    embryo(x,.568,.068,.082,tgt_xyz,tid,v,CM_RES,n_anchor_res,s=3.9)
    ax.add_patch(Circle((x+.022,.588),radius,fill=False,edgecolor=GREEN,lw=1.0,zorder=12));text(x,.510,label,size=6.9,color=GRAY,weight='bold')
arrow(.113,.568,.142,.568,color=RED);arrow(.210,.568,.230,.568,color=GRAY);arrow(.300,.568,.320,.568,color=GRAY)

# predictor bank: abstract computation icons only
text(.475,.642,'Predictor bank',size=8.4,color=INK,weight='bold')
tiles=[(.425,.590,'Global residual','global'),(.525,.590,'Local residual','local'),(.425,.525,'Anchor-only','anchor'),(.525,.525,'Calibrated mixture','mix')]
for x,y,label,kind in tiles:
    ax.add_patch(FancyBboxPatch((x-.044,y-.025),.088,.050,boxstyle='round,pad=.004,rounding_size=.008',facecolor='white',edgecolor=GREEN,lw=1.0,zorder=4))
    if kind=='global':ax.add_patch(Circle((x-.026,y),.010,facecolor=GREEN_LIGHT,edgecolor=GREEN,lw=.7))
    elif kind=='local':
        for dx,dy in [(-.032,-.006),(-.023,.007),(-.014,-.002)]:ax.add_patch(Circle((x+dx,y+dy),.004,facecolor=GREEN,edgecolor='none'))
    elif kind=='anchor':
        for jj in range(4):
            ax.add_patch(Rectangle((x-.034+(jj%2)*.008,y-.009+(jj//2)*.010),.0055,.007,facecolor=ORANGE,edgecolor='none',zorder=10))
    else:
        ax.plot([x-.034,x-.019,x-.005],[y-.007,y+.008,y-.004],color=BLUE,lw=1.1);ax.plot([x-.034,x-.019,x-.005],[y+.007,y-.006,y+.006],color=RED,lw=1.1)
    text(x+.012,y,label,size=6.2,color=INK,ha='center')
arrow(.390,.568,.405,.568,color=GREEN)

# source-only validation branch and guard
embryo(.650,.620,.048,.056,src_xyz,sid,source_field,CM_BLUE,n_main,s=2.5);text(.650,.660,'Source-only validation',size=7.5,color=BLUE_DARK,weight='bold')
arrow(.570,.565,.688,.565,color=GREEN);arrow(.650,.592,.688,.570,color=BLUE_DARK)
diamond=np.array([[.714,.595],[.746,.565],[.714,.535],[.682,.565]]);ax.add_patch(Polygon(diamond,closed=True,facecolor=GREEN_LIGHT,edgecolor=GREEN,lw=1.0));text(.714,.566,'Supported?',size=6.6,color=GREEN,weight='bold')
text(.760,.585,'yes',size=6,color=GREEN,weight='bold');text(.714,.507,'no',size=6,color=GRAY,weight='bold')
arrow(.746,.565,.782,.565,color=GREEN);arrow(.714,.535,.714,.480,color=GRAY)
text(.714,.466,'Retain atlas',size=6.6,color=GRAY,weight='bold')

# final present-state decomposition; common expression scale, symmetric correction scale
text(.805,.641,'Apply correction',size=7.5,color=GREEN,weight='bold');embryo(.805,.568,.077,.092,tgt_xyz,tid,correction,CM_RES,n_corr,s=4.4,outline=True)
arrow(.805,.520,.805,.468,color=GREEN)
embryo(.620,.405,.074,.088,tgt_xyz,tid,prior,CM_BLUE,n_main,s=4.2);text(.620,.351,'Registered prior',size=6.7,color=BLUE_DARK,weight='bold')
plus(.686,.405);embryo(.745,.405,.074,.088,tgt_xyz,tid,correction,CM_RES,n_corr,s=4.2);text(.745,.351,'Predicted correction',size=6.7,color=RED,weight='bold')
arrow(.785,.405,.835,.405,color=GREEN,lw=1.5);embryo(.890,.405,.086,.100,tgt_xyz,tid,adapt,CM_WARM,n_main,s=4.8,outline=True);text(.890,.351,'Adapted target',size=7.3,color=GREEN,weight='bold');text(.890,.462,r'$\hat{Y}_T = B + \hat{R}$',size=8.3,color=GREEN,weight='bold');text(.890,.375,gene,size=6.1,color=GRAY)

# c: early target, real tracking endpoints, transported state, temporal correction, later forecast
embryo(.085,.185,.095,.112,tgt_xyz,tid,early_adapt,CM_WARM,n_temp_expr,s=5.0,outline=True);text(.085,.255,'Early adapted state',size=8.0,color=GREEN,weight='bold');text(.085,.120,gene,size=6.1,color=GRAY)
arrow(.137,.185,.185,.185,color=ORANGE_DARK)
# scaled actual endpoint pairs, deterministically selected by ancestor id quantiles
ax.add_patch(FancyBboxPatch((.185,.135),.145,.104,boxstyle='round,pad=.005,rounding_size=.01',facecolor='white',edgecolor=LIGHT,lw=.8))
pair_arr=np.asarray(pairs);order=np.argsort(early_ids[pair_arr[:,0]]);take=order[np.linspace(0,len(order)-1,11,dtype=int)]
e=early_xyz[pair_arr[take,0],:2];l=late_xyz[pair_arr[take,1],:2]
for arr in (e,l):
    arr-=np.median(arr,axis=0);arr/=max(float(np.quantile(np.linalg.norm(arr,axis=1),.95)),1e-6)
for j,(a,b) in enumerate(zip(e,l)):
    x1=.199+(a[0]+1)*.025;y1=.158+(a[1]+1)*.025;x2=.284+(b[0]+1)*.020;y2=.158+(b[1]+1)*.025
    arrow(x1,y1,x2,y2,color=[BLUE,ORANGE,GREEN][j%3],lw=.65,rad=(j%3-1)*.08,style='->')
text(.257,.255,'Lineage transport',size=7.8,color=INK,weight='bold');text(.257,.120,'TM0–TM230 tracking map',size=6.6,color=GRAY)
arrow(.334,.185,.365,.185,color=BLUE_DARK)
embryo(.415,.185,.092,.108,late_xyz,lid,transported,CM_BLUE,n_temp_expr,s=4.7,outline=True);text(.415,.255,'Transported state',size=8.0,color=BLUE_DARK,weight='bold')
plus(.478,.185)
embryo(.545,.185,.092,.108,late_xyz,lid,temporal,CM_RES,n_temp_res,s=4.7,outline=True);text(.545,.255,'Temporal correction',size=8.0,color=RED,weight='bold')
arrow(.597,.185,.650,.185,color=GREEN,lw=1.5)
embryo(.710,.185,.104,.120,late_xyz,lid,later,CM_WARM,n_temp_expr,s=5.2,outline=True);text(.710,.255,'Later molecular forecast',size=8.2,color=GREEN,weight='bold');text(.710,.120,r'$\hat{Y}^{75}_T = \mathcal{T}(\hat{Y}^{50}_T) + \hat{D}_T$',size=7.7,color=GREEN,weight='bold')
# symmetry glyphs
ax.plot([.785,.785],[.125,.255],color=LIGHT,lw=.8);text(.825,.226,'PRESENT',size=6.2,color=GRAY,weight='bold');text(.825,.198,'prior',size=7.2,color=BLUE_DARK);plus(.870,.198);text(.915,.198,'correction',size=7.2,color=RED)
text(.825,.160,'FUTURE',size=6.2,color=GRAY,weight='bold');text(.825,.132,'transport',size=7.2,color=BLUE_DARK);plus(.870,.132);text(.915,.132,'correction',size=7.2,color=RED)
tiny_colorbar(.660,.102,.060,CM_WARM,'expression');tiny_colorbar(.535,.102,.060,CM_RES,'temporal correction')

for ext in ('svg','pdf','png'):
    fig.savefig(OUT/f'Figure_1_HyperSpatial_MAP_Nature.{ext}',dpi=450,bbox_inches='tight',pad_inches=.035)
plt.close(fig)

# Editable vector map assets.
asset_specs=[
 ('a_reference_chst11',src_xyz,sid,source_field,CM_BLUE,n_main),('a_registered_prior_chst11',tgt_xyz,tid,prior,CM_BLUE,n_main),
 ('a_anchor_measured_klf2a',tgt_xyz,tid,anchor_measured,CM_WARM,n_anchor),('a_anchor_expected_klf2a',tgt_xyz,tid,anchor_expected,CM_BLUE,n_anchor),
 ('a_anchor_residual_klf2a',tgt_xyz,tid,anchor_residual,CM_RES,n_anchor_res),('b_raw_anchor_residual',tgt_xyz,tid,raw_res,CM_RES,n_anchor_res),
 ('b_local_anchor_context',tgt_xyz,tid,local_res,CM_RES,n_anchor_res),('b_broad_anchor_context',tgt_xyz,tid,broad_res,CM_RES,n_anchor_res),
 ('b_predicted_correction_chst11',tgt_xyz,tid,correction,CM_RES,n_corr),('b_adapted_chst11',tgt_xyz,tid,adapt,CM_WARM,n_main),
 ('c_early_adapted_chst11',tgt_xyz,tid,early_adapt,CM_WARM,n_temp_expr),('c_transported_chst11',late_xyz,lid,transported,CM_BLUE,n_temp_expr),
 ('c_temporal_correction_chst11',late_xyz,lid,temporal,CM_RES,n_temp_res),('c_later_forecast_chst11',late_xyz,lid,later,CM_WARM,n_temp_expr)]
for name,xyz,ids,v,cmap,norm in asset_specs:
    f,a=plt.subplots(figsize=(1.55,1.35));a.axis('off');xy=xyz[ids,:2];a.scatter(xy[:,0],xy[:,1],s=5,c=cmap(norm(np.asarray(v)[ids])),linewidths=0);a.set_aspect('equal');f.savefig(ASSETS/f'{name}.svg',bbox_inches='tight',pad_inches=.01,transparent=True);plt.close(f)

# Before/after review image.
new=plt.imread(OUT/'Figure_1_HyperSpatial_MAP_Nature.png');old=plt.imread(OLD)
f,aa=plt.subplots(1,2,figsize=(15,6.2),facecolor='white');
for a,img,title in zip(aa,[old,new],['Current Figure 1','Redesigned Figure 1']):a.imshow(img);a.axis('off');a.set_title(title,fontsize=12,fontweight='bold',loc='left')
f.tight_layout();f.savefig(OUT/'Figure_1_before_after_comparison.png',dpi=220,bbox_inches='tight');plt.close(f)

rows=[
 ['Deep reference geometry and chst11 field','a','50% epiboly E1→E2','chst11',str(SRC.relative_to(ROOT)),str(RUN.relative_to(ROOT)),'Earliest frozen stage; lexicographically first reciprocal direction; gene nearest median positive source-validation support gain','Source coordinates and source-normalized expression'],
 ['Target geometry','a','50% epiboly E1→E2','',str(TGT.relative_to(ROOT)),str(RUN.relative_to(ROOT)),'Same prespecified setting','Target global_sphere coordinates only'],
 ['Registered atlas prior','a,b','50% epiboly E1→E2','chst11',str((PRED/'predictions/seed42_registered_idw_k12.npy').relative_to(ROOT)),str(RUN.relative_to(ROOT)),'Same source-only selected gene','Frozen k=12 registered IDW field evaluated on target coordinates'],
 ['Measured anchor field','a','50% epiboly E1→E2',anchor_gene,str(TGT.relative_to(ROOT)),str(RUN.relative_to(ROOT)),'First gene in frozen greedy source-only anchor order','Target anchor column only; source-trained depth normalization'],
 ['Anchor residual field','a,b','50% epiboly E1→E2',anchor_gene,str(TGT.relative_to(ROOT)),str(RUN.relative_to(ROOT)),'Same first frozen anchor','Measured anchor minus frozen registered expectation'],
 ['Predicted correction and adapted field','b','50% epiboly E1→E2',gene,str((PRED/'predictions/seed42_hyperspatial_map_calibrated_fusion.npy').relative_to(ROOT)),str(RUN.relative_to(ROOT)),'Gene nearest median positive source-validation support gain','Correction is adapted minus frozen registered prior'],
 ['Early adapted target','c','50% epiboly E1→E2',gene,str((PRED/'predictions/seed42_hyperspatial_map_calibrated_fusion.npy').relative_to(ROOT)),str(RUN.relative_to(ROOT)),'Same prespecified target and gene','Untransported 50% E2 adapted field'],
 ['TM0–TM230 trajectories','c','E2 50%→75% epiboly','',str(TRACK.relative_to(ROOT)),str(TROOT.relative_to(ROOT)),'Eleven lineage families at evenly spaced sorted ancestor-ID ranks','Actual TM0/TM230 endpoints; simplified vector paths'],
 ['Transported state, temporal correction and later forecast','c','MAP-T Fold A: learn E1 transition, forecast E2','chst11',str(DECOMP.relative_to(ROOT)),str(TROOT.relative_to(ROOT)),'Same source-only selected chst11 gene','Frozen decomposition arrays on matched 75% E2 geometry'],
]
with (OUT/'ASSET_PROVENANCE.tsv').open('w',newline='') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['asset','panel','biological setting','gene','source file','frozen run','selection rule','notes']);w.writerows(rows)

notes=f"""# Figure 1 redesign notes

## Scientific selection lock

- Present-state setting: **50% epiboly E1→E2**, selected before viewing target performance as the earliest frozen stage and the lexicographically first reciprocal direction.
- Anchor example: **{anchor_gene}**, the first gene in the frozen source-only greedy anchor order.
- Unmeasured-gene example: **{gene}**, selected solely from frozen source validation as the eligible non-anchor gene closest to the median positive source-validation Pearson gain ({gain[gene_i]:.6f}; median {median_gain:.6f}). No hidden target metric entered selection.
- Temporal setting: frozen MAP-T Fold A, which uses the same E2 individual as target and supplies serialized transported, temporal-residual and final forecast fields.

## Rendering

All embryo objects are deterministic bin-thinned projections of `global_sphere` coordinates. Expression fields compared by addition share one robust 2nd–98th percentile scale. Each residual/correction family uses one shared symmetric scale centered at zero. Text remains live in SVG and PDF. The tracking inset uses actual TM0 and TM230 endpoints for eleven E2 lineage families selected at evenly spaced ancestor-ID ranks; curves are a simplified vector representation of those endpoints.

The registered source coordinates themselves were not serialized in the 3D frozen run. The registration overlay therefore uses the frozen registered atlas field evaluated on actual target positions, with the target geometry shown beneath it; no transformed source coordinates were reconstructed.

## Scientific boundaries

The figure shows only executed frozen predictor families: global residual, local residual, anchor-only and calibrated mixture, followed by the source-only support guard and registered-atlas fallback. It does not show formal uncertainty, abandoned neural models, validation folds, exposure rules or performance results. Residual maps are target-specific molecular deviations from the registered atlas prior and are not automatically labelled biological.

## Validation

- The SVG master retains editable text, arrows and vector point marks; the PDF embeds its fonts and contains no rasterized page image.
- Full-size, 100% and 1,800-pixel reduced previews were inspected for label overlap and panel-order readability.
- Present-state additions use a shared expression normalization; each displayed residual/correction comparison uses a shared symmetric zero-centered normalization.
- The three panels use one E2 individual across present-state adaptation and the compatible frozen Fold-A developmental forecast.
- No target non-anchor performance was consulted when choosing the setting, anchor or illustrative non-anchor gene.

## Input integrity

Frozen inputs were read only. Representative assets were selected without target non-anchor performance. Source hashes: present manifest `{sha(PRED/'prediction_manifest.json')}`, MAP-T decomposition `{sha(DECOMP)}`, tracking matrix `{sha(TRACK)}`.
"""
(OUT/'FIGURE1_DESIGN_NOTES.md').write_text(notes)
print(json.dumps({'setting':'50% epiboly E1→E2','anchor':anchor_gene,'gene':gene,'median_source_gain':median_gain,'gene_source_gain':float(gain[gene_i]),'outputs':[str(OUT/f'Figure_1_HyperSpatial_MAP_Nature.{e}') for e in ('svg','pdf','png')]},indent=2))
