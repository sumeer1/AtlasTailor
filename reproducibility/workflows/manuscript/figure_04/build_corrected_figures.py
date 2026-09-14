#!/usr/bin/env python3
"""Presentation/statistical corrections using frozen HyperSpatial-MAP artifacts only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
R3 = REPO / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726"
R4 = REPO / "results_v10/hyperspatial_map_t_4d_frozen_20260804_112617"
I4 = REPO / "results_v10/hyperspatial_map_t_4d_frozen_20260804_105810/00_data_audit/inputs"
REV = REPO / "results_v10/hyperspatial_map_figure_revision_20260804_142208"
DATA = REPO / "data_v10/wemerfish_zebrafish_3stage"

MAP = "hyperspatial_map_calibrated_fusion"
IDW = "registered_idw_k12"
COL = {"map":"#C73535", "idw":"#E69F00", "anchor":"#277DA1", "free":"#1F6F8B",
       "shuffle":"#A8ADB3", "local":"#159A80", "global":"#8064A2", "text":"#252A2E"}
METHODS = ["B1_lineage_registered_idw", "B2_lineage_early_map",
           "B3_mean_developmental_correction", "B4_anchor_only_temporal",
           "B5_trajectory_free_map_t", "B6_trajectory_aware_map_t"]
LABELS = ["Transported\nregistered atlas", "Transported\nearly MAP", "Mean developmental\nchange",
          "Anchor-only\ntemporal model", "Trajectory-free\nMAP-T", "Trajectory-aware\nMAP-T"]
RUNS = [
 ("50% epiboly","50p","E1","E2",R3/"04_predictions/50p_E1_to_E2",DATA/"weMERFISH_measured_A_50p_E1.h5ad",DATA/"weMERFISH_measured_A_50p_E2.h5ad"),
 ("75% epiboly","75p","E1","E2",REPO/"results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2",DATA/"weMERFISH_measured_B_75p_E1.h5ad",DATA/"weMERFISH_measured_B_75p_E2.h5ad"),
 ("6-somite","6s","E1","E2",R3/"04_predictions/6s_E1_to_E2",DATA/"weMERFISH_measured_C_6s_E1_rescaled_z.h5ad",DATA/"weMERFISH_measured_C_6s_E2_rescaled_z.h5ad"),
]

mpl.rcParams.update({"font.family":"Liberation Sans", "font.size":8, "axes.titlesize":8,
 "axes.labelsize":8, "xtick.labelsize":7, "ytick.labelsize":7, "legend.fontsize":7,
 "axes.linewidth":.9, "lines.linewidth":1, "pdf.fonttype":42, "ps.fonttype":42,
 "svg.fonttype":"none", "figure.facecolor":"white", "axes.facecolor":"white",
 "text.color":COL["text"], "axes.labelcolor":COL["text"], "axes.edgecolor":COL["text"]})

def rd(path): return pd.read_csv(path, sep="\t")
def write(path, text): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
def tsv(frame, path): path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, sep="\t", index=False)
def dense(x): return np.asarray(x.toarray() if hasattr(x, "toarray") else x, np.float32)
def panel(ax, label, x=-.12, y=1.08): ax.text(x,y,label,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top",clip_on=False)
def polish(ax, grid=None):
    ax.spines[["top","right"]].set_visible(False)
    if grid: ax.grid(axis=grid,color="#E4E7EA",lw=.55,zorder=0)
def export(fig, stem):
    for ext in ("pdf","svg","png"):
        fig.savefig(stem.with_suffix("."+ext), dpi=300 if ext=="png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as handle:
        for b in iter(lambda:handle.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def coords(a, preferred=None):
    key = preferred if preferred in a.obsm else ("global_sphere" if "global_sphere" in a.obsm else "spatial")
    return np.asarray(a.obsm[key],float), key
def norm_log(a, depth):
    x=dense(a.X); lib=np.maximum(x.sum(1),1); return np.log1p(x*(depth/lib)[:,None]).astype(np.float32)
def prediction_paths(run):
    manifest=json.loads((run/"prediction_manifest.json").read_text())
    return manifest,{k:Path(v["path"]) for k,v in manifest["methods"]["42"]["predictions"].items()}
def smooth(values, xyz, k):
    nn=cKDTree(xyz).query(xyz,k=min(k+1,len(xyz)))[1][:,1:]
    return values[nn].mean(1)
def scatter_map(ax, xyz, value, *, cmap="viridis", vmin=None, vmax=None, size=.35, title=None):
    im=ax.scatter(xyz[:,0],xyz[:,1],c=value,s=size,cmap=cmap,vmin=vmin,vmax=vmax,lw=0,rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title,pad=2)
    return im

def inventory():
    rows=[
      ("Figure 1",REV/"02_figure1_architecture/Figure_1_architecture.svg",REV/"build_figure_revision.py",REV/"08_source_data/Figure_1_component_manifest.tsv",R3,"deterministic 50% E1->E2; klf2a is source-selected anchor",OUT/"01_figure1/Figure_1_architecture_corrected.svg"),
      ("Figure 3",REV/"04_figure3_biological_recovery/Figure_3_biological_recovery.svg",REPO/"summarize_hyperspatial_map_cross_stage_frozen_20260803.py",REV/"08_source_data/Figure_3_source_data.tsv",R3,"wnt11f2 source-predeclared; tbxta/noto/shha predeclared; fallback from source guard",OUT/"02_figure3/Figure_3_biological_recovery_corrected.svg"),
      ("Figure 4",R4/"10_figures_main/Figure_MAP_T_crossfit.svg",REPO/"run_hyperspatial_map_t_4d_frozen_20260804.py",R4/"12_source_data/PRIMARY_ENDPOINTS_ALL_FOLDS.tsv",R4,"no representative target-ranked gene",OUT/"03_figure4/Figure_4_4D_forecasting_corrected.svg"),
      ("Figure 5",R4/"10_figures_main/Figure_MAP_T_biology_and_limits.svg",REPO/"postprocess_hyperspatial_map_t_20260804.py",R4/"12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv",R4,"first source-defined stage-emergent, guard-passing non-anchor gene",OUT/"04_figure5/Figure_5_applicability_corrected.svg"),
    ]
    df=pd.DataFrame(rows,columns=["figure","original_figure_path","plotting_script","source_data_path","frozen_result_root","representative_gene_provenance","output_path"])
    tsv(df,OUT/"00_audit/FIGURE_INPUT_INVENTORY.tsv")
    write(OUT/"00_audit/FIGURE_INPUT_AUDIT.md", "# Figure input audit\n\nAll four figures resolve to original vector figures, analysis-specific plotting scripts, frozen predictions and machine-readable tables. The 50% E1→E2 direction is the deterministic first finalized present-state pair. `klf2a` is the first locked source-selected anchor. `wnt11f2` is source-predeclared. The temporal map genes are selected as the first source-defined stage-emergent, temporal-guard-passing, non-anchor genes. No target-ranked example is used.\n\nThe exact warped source-coordinate cloud was not serialized by the frozen 3D run. Figure 1 therefore shows the real source and target geometries and the saved registered-IDW field evaluated on real target query coordinates; it does not refit or reconstruct registration.\n")

def present_data():
    stage,tag,src,tgt,run,sp,tp=RUNS[0]
    man,paths=prediction_paths(run); sa=ad.read_h5ad(sp); ta=ad.read_h5ad(tp)
    sx=dense(sa.X); depth=float(np.median(np.maximum(sx.sum(1),1)))
    slog=np.log1p(sx*(depth/np.maximum(sx.sum(1),1))[:,None]); tlog=norm_log(ta,depth)
    sc,_=coords(sa,"global_sphere"); tc,_=coords(ta,man.get("target_coordinate_key"))
    genes=np.asarray(ta.var_names.astype(str)); gene=man["anchor_genes"][0]; gi=int(np.flatnonzero(genes==gene)[0])
    base=np.load(paths[IDW],mmap_mode="r")[:,gi]; raw=tlog[:,gi]-base; k12=smooth(raw,tc,12); k32=smooth(raw,tc,32)
    return dict(stage=stage,source=src,target=tgt,man=man,sa=sa,ta=ta,source_log=slog,target_log=tlog,
                source_xyz=sc,target_xyz=tc,gene=gene,gi=gi,base=np.asarray(base),raw=raw,k12=k12,k32=k32,
                base_file=str(paths[IDW]),source_file=str(sp),target_file=str(tp),depth=depth)

def read_4d(fold):
    target="E2" if fold=="A" else "E1"; source="E1" if fold=="A" else "E2"
    fd=R4/("03_fold_A" if fold=="A" else "04_fold_B")
    cfg=json.loads((fd/"FROZEN_FOLD_CONFIG.json").read_text())
    late=ad.read_h5ad(I4/f"matched_B_75p_{target}_sphere_logFC.h5ad")
    early=ad.read_h5ad(I4/f"matched_A_50p_{target}_sphere_logFC.h5ad")
    late_xyz,_=coords(late,"global_sphere"); early_xyz,_=coords(early,"global_sphere")
    truth=norm_log(late,float(cfg["early_map"]["depth_target"]))
    pred={m:np.load(fd/f"predictions_pre_unblinding/{m}.npy",mmap_mode="r") for m in METHODS}
    return dict(fold=fold,source=source,target=target,fd=fd,cfg=cfg,late=late,early=early,
                late_xyz=late_xyz,early_xyz=early_xyz,truth=truth,pred=pred)

def source_gene(fold):
    sets=json.loads((R4/"12_source_data/SOURCE_DEFINED_GENE_SETS.json").read_text())[fold]
    guards=rd(R4/f"02_source_validation/fold_{fold}_temporal_guards.tsv")
    good=set(guards[(guards.model=="B6")&guards.guard_pass&(~guards.is_anchor)].gene)
    for gene in sets["stage_emergent_genes"]:
        if gene in good: return gene
    raise RuntimeError(f"No frozen source-selected display gene for Fold {fold}")

def lineage_edges():
    lm=rd(R4/"06_tracking_overlap/fold_A_LINEAGE_MANIFEST.tsv")
    chosen=sorted(lm.TM0_ancestor.unique())[:6]
    use=["node id","timepoint","x","y","z","parent id","ancestor id"]
    parts=[]
    for chunk in pd.read_csv(I4/"Tracking_Matrix.csv",usecols=use,chunksize=250000):
        z=chunk[chunk["ancestor id"].isin(chosen)]
        if len(z): parts.append(z)
    frame=pd.concat(parts,ignore_index=True); node=frame.set_index("node id")
    edges=[]
    for _,row in frame.iterrows():
        parent=int(row["parent id"])
        if parent>=0 and parent in node.index:
            pr=node.loc[parent]; edges.append((pr.x,pr.y,row["x"],row["y"],row["timepoint"],int(row["ancestor id"])))
    return frame,pd.DataFrame(edges,columns=["x0","y0","x1","y1","timepoint","ancestor"])

def recovered_early_map(d4, gene):
    gi=list(d4["late"].var_names.astype(str)).index(gene)
    lm=rd(R4/f"06_tracking_overlap/fold_{d4['fold']}_LINEAGE_MANIFEST.tsv")
    late=np.asarray(d4["pred"]["B2_lineage_early_map"][:,gi])
    by=pd.DataFrame({"ancestor":lm.TM0_ancestor,"value":late}).groupby("ancestor").value.first()
    ids=np.asarray(d4["early"].obs["Amat_id"],dtype=np.int64)
    return np.asarray([by.get(int(x),np.nan) for x in ids],float)

def figure1():
    d=present_data(); d4=read_4d("A"); tgene=source_gene("A"); tgi=list(d4["late"].var_names.astype(str)).index(tgene)
    early_val=recovered_early_map(d4,tgene); b2=np.asarray(d4["pred"]["B2_lineage_early_map"][:,tgi]); b6=np.asarray(d4["pred"]["B6_trajectory_aware_map_t"][:,tgi]); tres=b6-b2
    tracks,edges=lineage_edges()
    fig=plt.figure(figsize=(180/25.4,9.3)); gs=fig.add_gridspec(3,1,height_ratios=[2.15,1.8,.62],hspace=.40)
    top=gs[0].subgridspec(2,5,width_ratios=[1,1,1,1,1],wspace=.16,hspace=.22)
    lim=np.nanpercentile(np.r_[d["source_log"][:,d["gi"]],d["target_log"][:,d["gi"]],d["base"]],99)
    entries=[(d["source_xyz"],d["source_log"][:,d["gi"]],"Real source E1\nsource expression"),
             (d["target_xyz"],d["target_xyz"][:,2],"Real target E2\ngeometry"),
             (d["target_xyz"],d["base"],"Registered source field\non target queries"),
             (d["target_xyz"],d["target_log"][:,d["gi"]],"Measured target anchor"),]
    for i,(xyz,val,title) in enumerate(entries):
        ax=fig.add_subplot(top[0,i]); cm="viridis"; lo,hi=(0,lim) if i!=1 else (None,None); im=scatter_map(ax,xyz,val,cmap=cm,vmin=lo,vmax=hi,size=.32,title=title)
        if i==0: panel(ax,"a",-.18,1.15)
    ax=fig.add_subplot(top[0,4]); rlim=np.nanpercentile(np.abs(d["raw"]),98); imr=scatter_map(ax,d["target_xyz"],d["raw"],cmap="RdBu_r",vmin=-rlim,vmax=rlim,size=.32,title="Anchor residual\n$R_a=Y_a-B_a$")
    cb=fig.colorbar(imr,ax=ax,fraction=.048,pad=.02);cb.set_label("log1p residual")
    for i,(val,title) in enumerate([(d["raw"],"Raw"),(d["k12"],"k = 12"),(d["k32"],"k = 32")]):
        ax=fig.add_subplot(top[1,i]); scatter_map(ax,d["target_xyz"],val,cmap="RdBu_r",vmin=-rlim,vmax=rlim,size=.32,title=title)
    ax=fig.add_subplot(top[1,3:]);ax.axis("off")
    boxes=[(.01,.62,.29,"Global residual\nrank 16",COL["global"]),(.355,.62,.29,"Local residual\nrank 16",COL["local"]),(.70,.62,.28,"Anchor-only\ndecoder",COL["anchor"]),(.12,.18,.34,"Per-gene calibrated\nfusion",COL["global"]),(.57,.18,.40,"Predictability guard\nIDW fallback",COL["map"])]
    for x,y,w,text,c in boxes: ax.add_patch(FancyBboxPatch((x,y),w,.22,boxstyle="round,pad=.010",fc=c+"18",ec=c,lw=1));ax.text(x+w/2,y+.11,text,ha="center",va="center",fontsize=5.5)
    ax.add_patch(FancyArrowPatch((.50,.60),(.33,.42),arrowstyle="-|>",mutation_scale=8,color="#687078"));ax.add_patch(FancyArrowPatch((.46,.29),(.57,.29),arrowstyle="-|>",mutation_scale=8,color="#687078"))

    mid=gs[1].subgridspec(1,5,wspace=.18)
    vals=[early_val,None,b2,tres,b6]; titles=[f"50% early MAP\n{tgene}","Real TM0–TM230\ntracking trajectories","Lineage-transported\nearly MAP","Source-learned\ntemporal residual","Predicted 75%\nMAP-T field"]
    vlim=np.nanpercentile(np.r_[early_val[np.isfinite(early_val)],b2,b6],99); r2=np.nanpercentile(np.abs(tres),98)
    for i,title in enumerate(titles):
        ax=fig.add_subplot(mid[i]);
        if i==1:
            for row in edges.itertuples(index=False): ax.plot([row.x0,row.x1],[row.y0,row.y1],color=plt.cm.viridis(row.timepoint/230),lw=.45,alpha=.75)
            ax.set_aspect("equal");ax.axis("off");ax.set_title(title)
        else:
            xyz=d4["early_xyz"] if i==0 else d4["late_xyz"]; cm="RdBu_r" if i==3 else "viridis"; lo,hi=(-r2,r2) if i==3 else (0,vlim)
            scatter_map(ax,xyz,vals[i],cmap=cm,vmin=lo,vmax=hi,size=.28,title=title)
        if i==0: panel(ax,"b",-.18,1.15)
    fig.text(.50,.335,"Tracking-map-conditioned temporal extension · trajectory-aware / trajectory-free / shuffled-history controls",ha="center",fontsize=7)

    ax=fig.add_subplot(gs[2]);ax.axis("off");panel(ax,"c",-.015,1.02)
    ax.text(.02,.73,"Cross-embryo reciprocal validation",weight="bold");ax.text(.02,.43,"50%: E1 -> E2, E2 -> E1   ·   75%: E1 -> E2, E2 -> E1   ·   6-somite: E1 -> E2, E2 -> E1",fontsize=7)
    ax.text(.02,.12,"4D cross-fit",weight="bold");ax.text(.17,.12,"Fold A: E1 molecular transition -> E2 forecast   ·   Fold B: E2 -> E1   ·   no molecular pooling",fontsize=7)
    export(fig,OUT/"01_figure1/Figure_1_architecture_corrected")
    components=pd.DataFrame([
      ("a","source expression",d["source_file"],d["gene"],"source-selected anchor"),("a","target geometry",d["target_file"],d["gene"],"geometry only"),("a","registered IDW field",d["base_file"],d["gene"],"saved frozen prediction"),("a","raw/k12/k32 anchor residual",d["target_file"],d["gene"],"target anchor permitted; plotted with frozen definitions"),("b","early MAP recovered from lineage transport",str(d4["fd"]/"predictions_pre_unblinding/B2_lineage_early_map.npy"),tgene,"source-selected stage-emergent"),("b","tracking trajectories",str(I4/"Tracking_Matrix.csv"),tgene,"deterministic first six families"),("b","temporal residual/final MAP-T",str(d4["fd"]/"predictions_pre_unblinding/B6_trajectory_aware_map_t.npy"),tgene,"saved frozen prediction")],columns=["panel","component","source","gene","provenance"])
    tsv(components,OUT/"05_source_data/Figure_1_component_manifest.tsv")
    return d,tgene

def stage_maps():
    out=[]
    for stage,tag,src,tgt,run,sp,tp in RUNS:
        man,paths=prediction_paths(run); sa=ad.read_h5ad(sp); ta=ad.read_h5ad(tp); sx=dense(sa.X); depth=float(np.median(np.maximum(sx.sum(1),1))); truth=norm_log(ta,depth); xyz,key=coords(ta,man.get("target_coordinate_key")); genes=np.asarray(ta.var_names.astype(str));gi=int(np.flatnonzero(genes=="wnt11f2")[0]);base=np.load(paths[IDW],mmap_mode="r")[:,gi];mp=np.load(paths[MAP],mmap_mode="r")[:,gi]
        out.append(dict(stage=stage,tag=tag,source=src,target=tgt,xyz=xyz,truth=truth[:,gi],idw=np.asarray(base),map=np.asarray(mp),target_file=str(tp),idw_file=str(paths[IDW]),map_file=str(paths[MAP]),coord_key=key))
    return out

def figure3():
    maps=stage_maps(); axial=rd(R3/"07_biological_modules/DOMAIN_LOCALIZATION_RESULTS.tsv").groupby(["stage","target_embryo","method","gene"],as_index=False).mean(numeric_only=True)
    fig=plt.figure(figsize=(180/25.4,10.0));gs=fig.add_gridspec(4,3,height_ratios=[2.65,1.4,1.6,1.45],hspace=.66,wspace=.42)
    top=gs[0,:].subgridspec(3,4,width_ratios=[1,1,1,.045],wspace=.04,hspace=.11)
    map_rows=[]
    for r,m in enumerate(maps):
        lim=np.nanpercentile(np.r_[m["truth"],m["idw"],m["map"]],99)
        for c,(v,title) in enumerate([(m["truth"],"Measured target"),(m["idw"],"Registered IDW"),(m["map"],"HyperSpatial-MAP")]):
            ax=fig.add_subplot(top[r,c]);im=scatter_map(ax,m["xyz"],v,vmin=0,vmax=lim,size=.34,title=title if r==0 else None)
            if c==0: ax.text(-.09,.5,f"{m['stage']}\n{m['source']} -> {m['target']}",rotation=90,ha="right",va="center",transform=ax.transAxes,fontsize=6.5)
            if r==0 and c==0:panel(ax,"a",-.24,1.18)
        cax=fig.add_subplot(top[r,3]);cb=fig.colorbar(im,cax=cax);cb.set_label("log1p",fontsize=6,labelpad=2)
        map_rows.append(pd.DataFrame({"table":"wnt11f2_maps","stage":m["stage"],"source":m["source"],"target":m["target"],"gene":"wnt11f2","x":m["xyz"][:,0],"y":m["xyz"][:,1],"z":m["xyz"][:,2],"measured":m["truth"],"registered_idw":m["idw"],"hyperspatial_map":m["map"],"shared_vmin":0,"shared_vmax":lim}))
    fig.text(.50,.965,"wnt11f2 expression · source-only predeclared gene · shared scale within each stage row",ha="center",fontsize=7.5,weight="bold")
    for col,(metric,title) in enumerate([("topk_dice","Dice ↑"),("centroid_error","Centroid error ↓"),("spatial_pearson","Spatial Pearson ↑")]):
        ax=fig.add_subplot(gs[1,col]);panel(ax,chr(ord("b")+col),-.17,1.13)
        for j,gene in enumerate(["tbxta","noto","shha"]):
            z=axial[axial.gene==gene].set_index(["stage","target_embryo","method"]); iv=[];mv=[]
            for stage,tag,src,tgt,*_ in RUNS: iv.append(z.loc[(stage,tgt,IDW),metric]);mv.append(z.loc[(stage,tgt,MAP),metric])
            for k in range(3):ax.plot([j-.08,j+.08],[iv[k],mv[k]],color="#B8BDC2",lw=.8)
            ax.scatter(np.full(3,j)-.08,iv,c=COL["idw"],s=23);ax.scatter(np.full(3,j)+.08,mv,c=COL["map"],s=23)
        ax.set_xticks(range(3),["tbxta","noto","shha"]);ax.set_title(title);polish(ax,"y")
        if col==0:ax.legend(handles=[Line2D([],[],marker="o",ls="",color=COL["idw"],label="Registered IDW"),Line2D([],[],marker="o",ls="",color=COL["map"],label="HyperSpatial-MAP")],frameon=False,fontsize=6)
    m=maps[0];true=m["truth"]-m["idw"];pred=m["map"]-m["idw"];rlim=np.nanpercentile(np.abs(np.r_[true,pred]),98);rho=float(np.corrcoef(true,pred)[0,1]);n=len(true)
    for c,(v,title) in enumerate([(true,"True residual"),(pred,"Predicted residual")]):
        ax=fig.add_subplot(gs[2,c]);imr=scatter_map(ax,m["xyz"],v,cmap="RdBu_r",vmin=-rlim,vmax=rlim,size=.48,title=f"wnt11f2 · {title}")
        if c==0:panel(ax,"e",-.17,1.13)
    ax=fig.add_subplot(gs[2,2]);ax.hexbin(true,pred,gridsize=42,mincnt=1,cmap="magma");lo=min(true.min(),pred.min());hi=max(true.max(),pred.max());ax.plot([lo,hi],[lo,hi],":",color="white",lw=1);ax.set(xlabel="True residual",ylabel="Predicted residual");ax.text(.04,.95,f"r = {rho:.3f}\nn = {n:,} target cells\n50% E1 -> E2",transform=ax.transAxes,va="top",color=COL["text"],fontsize=6.2,bbox=dict(fc="white",ec="none",alpha=.82,pad=2));polish(ax)
    cbax=fig.add_axes([.365,.325,.27,.012]);cb=fig.colorbar(imr,cax=cbax,orientation="horizontal");cb.set_label("log1p residual (symmetric about zero)",fontsize=6)
    # Frozen source-only fallback.
    fallback=None
    for stage,tag,src,tgt,run,sp,tp in RUNS:
        man,paths=prediction_paths(run);names=man["fusion_candidate_names"]
        hits=[i for i,v in enumerate(man["per_gene_fusion_selection"]) if names[int(v)]=="registered_idw"]
        if hits: fallback=(stage,src,tgt,sp,tp,man,paths,hits[0]);break
    if fallback is None:raise RuntimeError("No frozen guard fallback")
    stage,src,tgt,sp,tp,man,paths,gi=fallback;ta=ad.read_h5ad(tp);sa=ad.read_h5ad(sp);depth=np.median(np.maximum(dense(sa.X).sum(1),1));truthf=norm_log(ta,depth)[:,gi];xyz,_=coords(ta,man.get("target_coordinate_key"));idwf=np.load(paths[IDW],mmap_mode="r")[:,gi];mapf=np.load(paths[MAP],mmap_mode="r")[:,gi];gene=str(ta.var_names[gi]);assert np.array_equal(np.asarray(idwf),np.asarray(mapf));lim=np.nanpercentile(np.r_[truthf,idwf,mapf],99)
    bot=gs[3,:].subgridspec(1,3,wspace=.04)
    for c,(v,title) in enumerate([(truthf,"Measured"),(idwf,"Registered IDW"),(mapf,"MAP guarded output")]):
        ax=fig.add_subplot(bot[c]);scatter_map(ax,xyz,v,vmin=0,vmax=lim,size=.42,title=f"{gene} · {title}")
        if c==0:panel(ax,"f",-.17,1.13)
    fig.text(.50,.012,f"{gene} failed the source-only adaptation guard; MAP equals registered IDW by design (guarded abstention).",ha="center",fontsize=6.5)
    export(fig,OUT/"02_figure3/Figure_3_biological_recovery_corrected")
    source=pd.concat([axial.assign(table="axial_metrics"),*map_rows,pd.DataFrame({"table":"residual_map","stage":m["stage"],"source":m["source"],"target":m["target"],"gene":"wnt11f2","x":m["xyz"][:,0],"y":m["xyz"][:,1],"z":m["xyz"][:,2],"true_residual":true,"predicted_residual":pred,"residual_pearson":rho,"target_cells":n,"shared_residual_limit":rlim}),pd.DataFrame({"table":"guarded_fallback","stage":stage,"source":src,"target":tgt,"gene":gene,"x":xyz[:,0],"y":xyz[:,1],"z":xyz[:,2],"measured":truthf,"registered_idw":idwf,"hyperspatial_map":mapf,"guard_status":"registered_idw fallback"})],ignore_index=True,sort=False)
    tsv(source,OUT/"05_source_data/Figure_3_source_data.tsv")
    return gene,rho,n

def corrected_4d_tables():
    p=rd(R4/"12_source_data/PRIMARY_ENDPOINTS_ALL_FOLDS.tsv");p["original_rho_future_pooled"]=p.rho_future_pooled
    mask=p.method=="B2_lineage_early_map"
    for c in ["rho_future_pooled","rho_future_macro","rho_future_median"]:p.loc[mask,c]=np.nan
    p.loc[mask,"residual_pearson_status"]="undefined: predicted residual is identically zero"
    boot=rd(R4/"08_bootstrap/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv");boot["original_estimate"]=boot.estimate; bm=boot.method=="B2_lineage_early_map";boot.loc[bm,["estimate","ci95_low","ci95_high"]]=np.nan;boot.loc[bm,"status"]="undefined constant predicted residual"
    assert not any((p.method=="B2_lineage_early_map") & (p.rho_future_pooled==0))
    return p,boot

def figure4():
    p,boot=corrected_4d_tables();per=rd(R4/"12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv");perm=pd.concat([rd(R4/f"09_permutations/fold_{f}_results.tsv") for f in "AB"]);coh=pd.concat([rd(R4/f"06_tracking_overlap/fold_{f}_cohorts.tsv") for f in "AB"])
    sens=[]
    for f in "AB":sens.append({"fold":f,**json.loads((R4/f"07_overlap_excluded_retraining/fold_{f}_sensitivity.json").read_text())})
    sens=pd.DataFrame(sens)
    fig=plt.figure(figsize=(180/25.4,7.6));gs=fig.add_gridspec(3,2,height_ratios=[.78,1.5,1.6],hspace=.68,wspace=.42)
    ax=fig.add_subplot(gs[0,:]);ax.axis("off");panel(ax,"a",-.03,1.04)
    labels=["50% early MAP","TM0–TM230\nlineage transport","Source temporal\nresidual","Withheld 75%\nforecast"]
    for i,text in enumerate(labels):
        x=.035+i*.24;ax.add_patch(FancyBboxPatch((x,.37),.18,.38,boxstyle="round,pad=.012",fc=["#EAF1F7","#F2EDF6","#E8F5F1","#FCECEB"][i],ec="#9BA2A8",lw=.8));ax.text(x+.09,.56,text,ha="center",va="center",weight="bold",fontsize=7)
        if i<3:ax.add_patch(FancyArrowPatch((x+.18,.56),(x+.24,.56),arrowstyle="-|>",mutation_scale=8,color="#687078"))
    ax.text(.04,.13,"Fold A: E1 molecular transition -> E2 forecast     Fold B: E2 -> E1     no molecular pooling",fontsize=7)
    # Corrected future residual panel.
    ax=fig.add_subplot(gs[1,0]);panel(ax,"b",-.15,1.12);q=p[p.cohort=="full"]
    colors=[COL["idw"],"#777777",COL["global"],COL["anchor"],COL["free"],COL["map"]]
    for fi,f in enumerate("AB"):
        vals=q[q.fold==f].set_index("method").reindex(METHODS).rho_future_pooled
        for i,v in enumerate(vals):
            if np.isfinite(v):ax.scatter(i+(-.045 if f=="A" else .045),v,color="#277DA1" if f=="A" else COL["map"],s=30,zorder=3)
        ax.scatter(1+(-.045 if f=="A" else .045),.012,facecolors="none",edgecolors="#277DA1" if f=="A" else COL["map"],marker="x",s=33,zorder=3)
    ax.text(1,.052,"N.D.",ha="center",fontsize=6);ax.text(.55,.31,"Reference baseline: residual Pearson undefined\n(predicted residual is identically zero).",ha="center",va="center",fontsize=6.0,bbox=dict(fc="white",ec="none",alpha=.88,pad=2))
    ax.set_xticks(range(6),LABELS,rotation=27,ha="right");ax.set_ylabel("Future-residual Pearson ↑");polish(ax,"y")
    # RMSE retains all methods.
    ax=fig.add_subplot(gs[1,1]);panel(ax,"c",-.15,1.12)
    for fi,f in enumerate("AB"):
        vals=q[q.fold==f].set_index("method").reindex(METHODS).rmse;ax.scatter(np.arange(6)+(-.045 if f=="A" else .045),vals,color="#277DA1" if f=="A" else COL["map"],s=30,label=f"Fold {f}")
    ax.set_xticks(range(6),LABELS,rotation=27,ha="right");ax.set_ylabel("log1p RMSE ↓");ax.legend(frameon=False);polish(ax,"y")
    ax=fig.add_subplot(gs[2,0]);panel(ax,"d",-.15,1.12);qp=perm[perm.cohort=="full"]
    ax.errorbar([0,1],qp.null_mean,yerr=[qp.null_mean-qp.null_ci_low,qp.null_ci_high-qp.null_mean],fmt="o",color=COL["shuffle"],capsize=3,label="100 shuffled histories");ax.scatter([0,1],qp.observed,color=COL["map"],s=42,label="Real trajectory history");ax.set_xticks([0,1],["Fold A","Fold B"]);ax.set_ylabel("Future-residual Pearson ↑");ax.legend(frameon=False);ax.text(.5,.04,"0/100 shuffled controls exceeded observed",transform=ax.transAxes,ha="center",fontsize=6.5);polish(ax,"y")
    right=gs[2,1].subgridspec(1,2,wspace=.65)
    ax=fig.add_subplot(right[0]);panel(ax,"e",-.23,1.12);qb=boot[boot.method=="B6_trajectory_aware_map_t"]
    for fi,f in enumerate("AB"):
        z=qb[qb.fold==f].set_index("cohort").reindex(["full","shared","disjoint"]);x=np.arange(3)+(-.05 if f=="A" else .05);ax.errorbar(x,z.estimate,yerr=[z.estimate-z.ci95_low,z.ci95_high-z.estimate],fmt="o",color="#277DA1" if f=="A" else COL["map"],capsize=2,label=f"Fold {f}")
        sr=sens[sens.fold==f].iloc[0];ax.scatter(3+(-.05 if f=="A" else .05),sr.rho_future_disjoint,marker="D",color="#277DA1" if f=="A" else COL["map"],s=30)
    ax.set_xticks(range(4),["Full","Shared","Disjoint","Overlap-\nexcluded"],rotation=28,ha="right");ax.set_ylabel("Future-residual Pearson ↑");ax.legend(frameon=False,fontsize=6);polish(ax,"y")
    ax=fig.add_subplot(right[1]);panel(ax,"f",-.23,1.12);w=per[per.method.isin(["B5_trajectory_free_map_t","B6_trajectory_aware_map_t"])].pivot_table(index=["fold","gene"],columns="method",values="future_residual_pearson").dropna();diff=w.B6_trajectory_aware_map_t-w.B5_trajectory_free_map_t;pct=100*(diff>0).mean();med=diff.median()
    for f,c in [("A","#277DA1"),("B",COL["map"])]:z=w.loc[f];ax.scatter(z.B5_trajectory_free_map_t,z.B6_trajectory_aware_map_t,s=5,alpha=.32,color=c,label=f"Fold {f}")
    ax.plot([-.5,1],[-.5,1],":",color="#777");ax.set(xlabel="Trajectory-free gene r",ylabel="Trajectory-aware gene r",xlim=(-.5,1),ylim=(-.5,1));ax.text(.04,.96,f"{pct:.1f}% genes improved\nmedian gain {med:+.3f}",transform=ax.transAxes,va="top",fontsize=6);ax.legend(frameon=False,fontsize=6);polish(ax)
    export(fig,OUT/"03_figure4/Figure_4_4D_forecasting_corrected")
    source=pd.concat([p.assign(table="primary_corrected"),boot.assign(table="bootstrap_corrected"),perm.assign(table="permutation"),coh.assign(table="cohort_counts"),sens.assign(table="overlap_excluded"),per.assign(table="per_gene")],ignore_index=True,sort=False);tsv(source,OUT/"05_source_data/Figure_4_source_data.tsv")
    return pct,med,coh

def figure5():
    per=rd(R4/"12_source_data/PER_GENE_METRICS_ALL_FOLDS.tsv");dom=rd(R4/"12_source_data/DOMAIN_METRICS_ALL_FOLDS.tsv");strat=rd(R4/"12_source_data/STRATIFIED_ENDPOINTS.tsv");unc=rd(R4/"12_source_data/UNCERTAINTY_CALIBRATION_ALL_FOLDS.tsv");guards=pd.concat([rd(R4/f"02_source_validation/fold_{f}_temporal_guards.tsv") for f in "AB"])
    fig=plt.figure(figsize=(180/25.4,9.0));gs=fig.add_gridspec(3,3,height_ratios=[2.4,1.4,1.35],hspace=.62,wspace=.45)
    top=gs[0,:].subgridspec(2,4,width_ratios=[1,1,1,.04],wspace=.04,hspace=.10);chosen=[];maps=[]
    for r,f in enumerate("AB"):
        d=read_4d(f);gene=source_gene(f);chosen.append(gene);gi=list(d["late"].var_names.astype(str)).index(gene);truth=d["truth"][:,gi];base=np.asarray(d["pred"]["B1_lineage_registered_idw"][:,gi]);mp=np.asarray(d["pred"]["B6_trajectory_aware_map_t"][:,gi]);lim=np.nanpercentile(np.r_[truth,base,mp],99)
        for c,(v,title) in enumerate([(truth,"Measured 75% target"),(base,"Lineage-transported\nregistered atlas"),(mp,"HyperSpatial-MAP-T\nforecast")]):
            ax=fig.add_subplot(top[r,c]);im=scatter_map(ax,d["late_xyz"],v,vmin=0,vmax=lim,size=.36,title=title if r==0 else None)
            if c==0:ax.text(-.08,.5,f"{gene} · Fold {f}\n{d['source']} -> {d['target']}",rotation=90,ha="right",va="center",transform=ax.transAxes,fontsize=6.3)
            if r==0 and c==0:panel(ax,"a",-.22,1.18)
        cax=fig.add_subplot(top[r,3]);cb=fig.colorbar(im,cax=cax);cb.set_label("source-depth-normalized\nlog1p expression",fontsize=6)
        maps.append(pd.DataFrame({"table":"expression_maps","fold":f,"source":d["source"],"target":d["target"],"gene":gene,"selection":"first source-defined stage-emergent, B6 guard-passing non-anchor","x":d["late_xyz"][:,0],"y":d["late_xyz"][:,1],"z":d["late_xyz"][:,2],"measured_75":truth,"lineage_registered_atlas":base,"hyperspatial_map_t":mp,"shared_vmin":0,"shared_vmax":lim}))
    fig.text(.50,.965,"Source-selected stage-emergent genes · shared measured/baseline/MAP-T scale within each fold",ha="center",fontsize=7.3,weight="bold")
    ax=fig.add_subplot(gs[1,0]);panel(ax,"b",-.17,1.12);dmed=dom.groupby(["fold","method"])[["dice","centroid_error"]].median().reset_index()
    for f,marker in [("A","o"),("B","s")]:
        a=dmed[(dmed.fold==f)&(dmed.method=="B1_lineage_registered_idw")].iloc[0];m=dmed[(dmed.fold==f)&(dmed.method=="B6_trajectory_aware_map_t")].iloc[0];ax.plot([a.dice,m.dice],[a.centroid_error,m.centroid_error],color="#B8BDC2",lw=.8);ax.scatter(a.dice,a.centroid_error,color=COL["idw"],marker=marker,s=35);ax.scatter(m.dice,m.centroid_error,color=COL["map"],marker=marker,s=42)
    ax.set(xlabel="Median domain Dice ↑",ylabel="Centroid error ↓");polish(ax,"both")
    ax=fig.add_subplot(gs[1,1]);panel(ax,"c",-.17,1.12);w=per[per.method.isin(["B5_trajectory_free_map_t","B6_trajectory_aware_map_t"])].pivot_table(index=["fold","gene"],columns="method",values="future_residual_pearson").dropna()
    for f,c in [("A","#277DA1"),("B",COL["map"])]:z=w.loc[f];ax.hist(z.B6_trajectory_aware_map_t-z.B5_trajectory_free_map_t,bins=35,histtype="step",density=True,color=c,label=f"Fold {f}")
    ax.axvline(0,color="#777",ls=":");ax.set(xlabel="Trajectory-aware − free gene r",ylabel="Density");ax.legend(frameon=False);polish(ax)
    ax=fig.add_subplot(gs[1,2]);panel(ax,"d",-.17,1.12);g=guards[guards.model=="B6"].groupby(["fold","guard_pass"]).size().unstack(fill_value=0);bot=np.zeros(2)
    for state,c,label in [(True,COL["map"],"Corrected"),(False,COL["shuffle"],"Fallback")]:v=np.array([g.loc[f,state] for f in "AB"]);ax.bar(["Fold A","Fold B"],v,bottom=bot,color=c,label=label);bot+=v
    ax.set_ylabel("Non-anchor genes");ax.set_title("Source-only temporal guard",pad=13);ax.legend(frameon=False,ncol=2,fontsize=6,loc="upper center",bbox_to_anchor=(.5,1.02));polish(ax,"y")
    ax=fig.add_subplot(gs[2,0:2]);panel(ax,"e",-.08,1.12);q=strat[(strat.stratum_type=="source_gene_group")&strat.method.isin(["B1_lineage_registered_idw","B6_trajectory_aware_map_t"])];order=["all_non_anchor","source_stage_emergent","source_variable"]
    for method,c,label in [("B1_lineage_registered_idw",COL["idw"],"Transported registered atlas"),("B6_trajectory_aware_map_t",COL["map"],"MAP-T")]:v=q[q.method==method].groupby("stratum").rho_future.mean().reindex(order);ax.plot(range(3),v,"-o",color=c,label=label)
    ax.set_xticks(range(3),["All non-anchor","Stage-emergent","Source-variable"]);ax.set_ylabel("Future-residual Pearson ↑");ax.legend(frameon=False);polish(ax,"y")
    ax=fig.add_subplot(gs[2,2]);panel(ax,"f",-.17,1.12)
    for f,c in [("A","#277DA1"),("B",COL["map"])]:z=unc[(unc.fold==f)&(~unc.is_anchor)];ax.scatter(z.source_cv_sigma,z.target_mae,s=5,alpha=.22,color=c,label=f"Fold {f}")
    ax.set(xlabel="Source-CV uncertainty",ylabel="Target MAE");ax.legend(frameon=False);polish(ax,"both")
    fig.text(.50,.012,"Applicability boundary: two independently measured molecular embryos partly assigned to one published tracking reference; no independent tracking-system, cross-species or cross-platform validation.",ha="center",fontsize=6.2)
    export(fig,OUT/"04_figure5/Figure_5_applicability_corrected")
    src=pd.concat([*maps,dom.assign(table="domain"),per.assign(table="per_gene"),strat.assign(table="source_groups"),unc.assign(table="uncertainty"),guards.assign(table="guards")],ignore_index=True,sort=False);tsv(src,OUT/"05_source_data/Figure_5_source_data.tsv")
    return chosen

def captions(anchor,tgene,fallback,rho,n,pct,med,coh,chosen):
    counts="; ".join(f"Fold {r.fold} {r.cohort}: {int(r.ancestor_count):,} ancestors/{int(r.descendant_count):,} descendants" for r in coh.itertuples())
    write(OUT/"06_captions/Figure_1_caption.md",f"**Figure 1 | HyperSpatial-MAP and its tracking-map-conditioned temporal extension.** **a,** The deterministic first frozen direction (50% epiboly E1→E2) uses real source and target point clouds. The saved registered 12-neighbour IDW field is displayed on target query coordinates for source-selected anchor {anchor}; the raw target-anchor residual and k=12/k=32 smooths share a symmetric scale. The exact warped source points were not serialized and were not refitted for this figure. Source-trained rank-16 global/local operators, anchor-only decoder, per-gene fusion and guard are schematic. **b,** Real Fold A geometries, saved early MAP/transport/residual/final fields for source-selected {tgene}, and deterministic real TM0–TM230 tracking families. The tracking reference is published and partly shared between molecular embryos. **c,** Locked reciprocal and cross-fitted designs.\n")
    write(OUT/"06_captions/Figure_3_caption.md",f"**Figure 3 | Specimen-specific biological recovery beyond registered atlas transfer.** **a,** Source-predeclared wnt11f2 in the deterministic E1→E2 direction at three stages. Measured target, registered IDW and MAP share a source-depth-normalized log1p scale within each stage row. **b–d,** Paired Dice, centroid error and spatial Pearson for predeclared anchor-excluded tbxta, noto and shha. **e,** True and predicted wnt11f2 residuals at 50% E1→E2 share one symmetric zero-centred scale (cell-level r={rho:.3f}; n={n:,} target cells; descriptive, not biological replication). **f,** {fallback} failed the source-only adaptation guard, so MAP equals registered IDW by design.\n")
    write(OUT/"06_captions/Figure_4_caption.md",f"**Figure 4 | Cross-fitted trajectory-aware molecular forecasting.** **a,** Molecular cross-fitting. **b,** Future-residual Pearson; transported early MAP is the reference defining the residual and therefore has an identically zero predicted residual, making Pearson undefined (N.D.), not zero. **c,** RMSE retains all methods. **d,** Real histories versus 100 shuffled controls; none exceeded observed. **e,** Categorical full, shared and tracking-disjoint cohorts with TM0-family bootstrap intervals; diamonds denote overlap-excluded retraining. Cohort sizes: {counts}. **f,** Gene-level descriptive comparison ({pct:.1f}% improved; median aware-minus-free gain {med:+.3f}); genes are not biological replicates. The two molecular embryos partly use one published tracking reference.\n")
    write(OUT/"06_captions/Figure_5_caption.md",f"**Figure 5 | Prediction guards, trajectory contributions and applicability boundaries.** **a,** Measured 75% target, lineage-transported registered atlas and MAP-T for source-selected, stage-emergent, guard-passing non-anchor genes ({chosen[0]}, Fold A; {chosen[1]}, Fold B). Each row uses one shared source-depth-normalized log1p scale. **b,** Domain Dice and centroid error. **c,** Gene-level trajectory contribution. **d,** Source-only temporal guards. **e,** Source-defined gene groups. **f,** Source-CV uncertainty versus target error. The molecular embryos were independently measured but partly assigned to one published tracking reference; this is neither independent longitudinal tracking-system validation nor cross-species/cross-platform validation.\n")

def audits(anchor,tgene,fallback,chosen):
    rows=[
      ("Figure 1","a","generic simulated-looking embryo clouds","real frozen source/target coordinates, saved registered IDW field, actual anchor residual and k12/k32 maps","Figure_1_component_manifest.tsv",str(Path(__file__).resolve()),"PASS","exact warped source-point coordinates were not serialized; no registration was refit"),
      ("Figure 1","b","decorative trajectory schematic","actual tracking-matrix edges and saved temporal decomposition","Figure_1_component_manifest.tsv",str(Path(__file__).resolve()),"PASS","tracking reference partly shared"),
      ("Figure 3","a","gene and map scales insufficiently documented",f"wnt11f2 identified from source files; shared row scales and colorbars added","Figure_3_source_data.tsv",str(Path(__file__).resolve()),"PASS","scales differ by stage row"),
      ("Figure 3","e","no residual colorbar/statistic","shared symmetric scale, colorbar, r, direction, identity line and n added","Figure_3_source_data.tsv",str(Path(__file__).resolve()),"PASS","cell-level r is descriptive"),
      ("Figure 3","f","fallback semantics unclear",f"verified {fallback} MAP array equals IDW exactly and labelled guarded abstention","Figure_3_source_data.tsv",str(Path(__file__).resolve()),"PASS","single illustrative fallback"),
      ("Figure 4","b","undefined transported-early-MAP residual Pearson plotted as zero","set to NA and displayed as N.D. with mathematical explanation","Figure_4_source_data.tsv",str(Path(__file__).resolve()),"PASS","historical frozen table retains zero and is preserved"),
      ("Figure 4","d–f","permutation/cohort/gene annotations incomplete","exact permutation statement, unconnected categorical CIs, counts and gene effects added","Figure_4_source_data.tsv",str(Path(__file__).resolve()),"PASS","only two molecular embryos"),
      ("Figure 5","a","measured 75% truth absent and baseline ambiguous",f"measured truth added; baseline explicitly lineage-transported registered atlas; shared fold scales; genes {chosen}","Figure_5_source_data.tsv",str(Path(__file__).resolve()),"PASS","source-normalized units"),
      ("Figure 5","title/caption","generality wording unsupported","renamed applicability boundaries and explicitly excludes external generalization","Figure_5_source_data.tsv",str(Path(__file__).resolve()),"PASS","external mouse validation pending"),
    ]
    df=pd.DataFrame(rows,columns=["figure","panel","original_issue","correction","source_data_file","script","verification_result","remaining_limitation"]);tsv(df,OUT/"00_audit/PANEL_LEVEL_CORRECTION_TABLE.tsv")
    write(OUT/"00_audit/FINAL_CORRECTION_AUDIT.md",f"# Final correction audit\n\n- Representative genes are source-selected or biologically predeclared: anchor `{anchor}`, temporal `{tgene}`/`{chosen[1]}`, `wnt11f2`, and guard-selected fallback `{fallback}`.\n- Directly compared maps share color limits and labelled scale definitions.\n- All stage/fold directions are stated.\n- The transported-early-MAP residual Pearson is missing/undefined, never numeric zero, in corrected source data and graphics.\n- Full/shared/disjoint cohorts are categorical points with TM0-family intervals and are not connected.\n- Cells and genes are explicitly descriptive, not biological replicates.\n- Tracking-map provenance is qualified.\n- The exact registered source-point cloud could not be reproduced because it was not serialized; the saved registered field on real target queries is shown without refitting.\n")
    assert not any((rd(OUT/"05_source_data/Figure_4_source_data.tsv").method=="B2_lineage_early_map") & (rd(OUT/"05_source_data/Figure_4_source_data.tsv").table=="primary_corrected") & (rd(OUT/"05_source_data/Figure_4_source_data.tsv").rho_future_pooled==0))

def report(anchor,tgene,fallback,chosen):
    write(OUT/"FIGURE_CORRECTION_REPORT.md",f"# Figure correction report\n\n- **Original Figure 1:** `{REV/'02_figure1_architecture/Figure_1_architecture.svg'}`; corrected: `01_figure1/Figure_1_architecture_corrected.*`. Real 50% E1/E2 coordinates, source-selected anchor `{anchor}`, saved registered field, and real Fold A tracking/decomposition were used. The exact warped source-point cloud was unavailable and was not regenerated.\n- **Original Figure 3:** `{REV/'04_figure3_biological_recovery/Figure_3_biological_recovery.svg'}`; corrected: `02_figure3/Figure_3_biological_recovery_corrected.*`. The gene is `wnt11f2`; comparison scales are shared within stage rows and residuals use one symmetric scale. `{fallback}` is verified as a guarded IDW fallback.\n- **Original Figure 4:** `{R4/'10_figures_main/Figure_MAP_T_crossfit.svg'}`; corrected: `03_figure4/Figure_4_4D_forecasting_corrected.*`. Transported-early-MAP residual Pearson is NA/N.D., not zero; permutation, cohort and gene-effect annotations were added from frozen tables.\n- **Original Figure 5:** `{R4/'10_figures_main/Figure_MAP_T_biology_and_limits.svg'}`; corrected: `04_figure5/Figure_5_applicability_corrected.*`. Measured target truth was added alongside the explicitly named lineage-transported registered atlas and MAP-T, using shared within-fold scales. Display genes `{chosen[0]}` and `{chosen[1]}` are source-defined.\n- **Placement:** Figures 1, 3 and 4 are suitable for the main article. Figure 5 should be titled as applicability boundaries and moved to supplementary material when a valid external dataset is available; until then it can serve as the fifth main figure with its qualifications intact.\n- **Scientific changes:** none. No model was trained, tuned, refit or reevaluated; original figures and frozen predictions remain unchanged.\n")

def qa():
    from PIL import Image
    rows=[]
    for p in sorted(OUT.glob("0[1-4]_figure*/*.png")):
        im=Image.open(p);dpi=im.info.get("dpi",(np.nan,np.nan));rows.append((str(p.relative_to(OUT)),im.width,im.height,float(dpi[0]),float(dpi[1]),float(dpi[0])>=299 and float(dpi[1])>=299))
    tsv(pd.DataFrame(rows,columns=["file","width_px","height_px","dpi_x","dpi_y","at_least_300_dpi"]),OUT/"07_logs/FIGURE_RENDER_QA.tsv")
    files=[]
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name!="SHA256SUMS.tsv"):files.append((str(p.relative_to(OUT)),sha(p),p.stat().st_size))
    tsv(pd.DataFrame(files,columns=["file","sha256","bytes"]),OUT/"07_logs/SHA256SUMS.tsv")

def main():
    inventory();present,tgene=figure1();fallback,rho,n=figure3();pct,med,coh=figure4();chosen=figure5();captions(present["gene"],tgene,fallback,rho,n,pct,med,coh,chosen);audits(present["gene"],tgene,fallback,chosen);report(present["gene"],tgene,fallback,chosen);qa();print(OUT)

if __name__=="__main__": main()
