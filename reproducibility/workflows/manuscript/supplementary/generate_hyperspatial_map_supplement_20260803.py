#!/usr/bin/env python3
"""Generate frozen-output-only HyperSpatial-MAP supplementary figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import summarize_hyperspatial_map_cross_stage_frozen_20260803 as base
from run_wemerfish_temporal_holdout import canonicalize, coordinates_only, measured_counts, normalize_counts


ROOT = base.ROOT
OUT = ROOT / "09_figures_supplementary/revision_20260803_172006"
SOURCE = OUT / "source_data"
MAP, IDW, ANCHOR = base.MAP, base.IDW, base.ANCHOR
METHODS = [ANCHOR, "atlas_plus_global_anchor_residual", "atlas_plus_local_anchor_residual", MAP, IDW, "registered_nearest_neighbor"]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def save(fig, name):
    for suffix,kwargs in (("pdf",{}),("svg",{}),("png",{"dpi":600})):
        fig.savefig(OUT/f"{name}.{suffix}",bbox_inches="tight",**kwargs)
    plt.close(fig)


def label(ax, letter): ax.text(-.13,1.07,letter,fontweight="bold",fontsize=10,transform=ax.transAxes)


def ecdf(values):
    values=np.sort(np.asarray(values)[np.isfinite(values)])
    return values,np.arange(1,len(values)+1)/max(len(values),1)


def figure_gene_distributions(gene):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2),constrained_layout=True)
    for ax,endpoint,title in [(axes[0,0],"spatial_pearson","Source-selected SVG spatial Pearson"),(axes[0,1],"residual_spatial_pearson","Source-predictable-gene residual Pearson")]:
        for method in (IDW,ANCHOR,MAP):
            x,y=ecdf(gene[(gene.endpoint==endpoint)&(gene.method==method)].value)
            ax.plot(x,y,lw=1.4,color=base.COLORS[method],label=base.METHOD_LABEL[method])
        ax.axvline(0,color="#777",ls=":",lw=.8); ax.set_xlabel(title); ax.set_ylabel("Cumulative fraction"); ax.legend(frameon=False,fontsize=7)
    label(axes[0,0],"a"); label(axes[0,1],"b")
    wide=gene[gene.endpoint=="spatial_pearson"].pivot_table(index=["target_key","gene"],columns="method",values="value").dropna()
    delta=(wide[MAP]-wide[IDW]).rename("delta").reset_index().merge(pd.DataFrame([{"target_key":f"{r['tag']}_{r['source']}_to_{r['target']}","stage":r['stage'],"target":r['target']} for r in base.RUNS]),on="target_key")
    order=[f"{r['tag']} {r['target']}" for r in base.RUNS]
    for i,r in enumerate(base.RUNS):
        vals=delta[delta.target_key==f"{r['tag']}_{r['source']}_to_{r['target']}"].delta
        parts=axes[1,0].violinplot(vals,positions=[i],showextrema=False,widths=.75); [b.set_facecolor("#56B4E9") or b.set_alpha(.65) for b in parts["bodies"]]
        axes[1,0].scatter(i,np.median(vals),c="#222",s=12,zorder=3)
    axes[1,0].axhline(0,color="#777",ls=":",lw=.8); axes[1,0].set_xticks(range(6),order,rotation=35,ha="right",fontsize=7); axes[1,0].set_ylabel("MAP − IDW per-gene Pearson"); label(axes[1,0],"c")
    fraction=(wide[MAP]>wide[IDW]).groupby(level=0).mean().reindex([f"{r['tag']}_{r['source']}_to_{r['target']}" for r in base.RUNS])
    axes[1,1].scatter(range(6),fraction,c="#56B4E9",s=35); axes[1,1].axhline(.5,color="#777",ls=":",lw=.8); axes[1,1].set_ylim(0,1); axes[1,1].set_xticks(range(6),order,rotation=35,ha="right",fontsize=7); axes[1,1].set_ylabel("Fraction of SVGs improved"); label(axes[1,1],"d")
    gene.to_csv(SOURCE/"Supplementary_Figure_1_per_gene_metrics.tsv",sep="\t",index=False); delta.to_csv(SOURCE/"Supplementary_Figure_1_per_gene_deltas.tsv",sep="\t",index=False)
    save(fig,"Supplementary_Figure_1_all_gene_metric_distributions")


def figure_all_targets(embryo,residual):
    cols=[e[0] for e in base.ENDPOINTS]+["residual_spatial_pearson_median","moran_error_median"]
    titles=[e[1] for e in base.ENDPOINTS]+["Residual Pearson","Moran error"]
    merged=embryo.merge(residual[["target_key","method","residual_spatial_pearson_median"]],on=["target_key","method"])
    fig,axes=plt.subplots(2,4,figsize=(7.2,4.5),constrained_layout=True)
    for ax,col,title in zip(axes.ravel(),cols,titles):
        for mi,m in enumerate(METHODS):
            d=merged[merged.method==m].set_index("target_key").reindex([f"{r['tag']}_{r['source']}_to_{r['target']}" for r in base.RUNS])
            ax.plot(range(6),d[col],marker="o",ms=2.8,lw=.8,color=base.COLORS[m],label=base.METHOD_LABEL[m])
        ax.set_title(title+(" ↓" if col in {"weighted_centroid_error_median","spatial_emd_median","log1p_rmse","moran_error_median"} else " ↑"),fontsize=7.5); ax.set_xticks(range(6),[f"{r['tag']} {r['target']}" for r in base.RUNS],rotation=40,ha="right",fontsize=5.5); ax.tick_params(axis="y",labelsize=6)
    label(axes[0,0],"a"); fig.legend(handles=[plt.Line2D([],[],color=base.COLORS[m],marker="o",ms=3,label=base.METHOD_LABEL[m]) for m in METHODS],fontsize=5.5,frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,-.035))
    merged.to_csv(SOURCE/"Supplementary_Figure_2_all_targets_methods.tsv",sep="\t",index=False)
    save(fig,"Supplementary_Figure_2_all_embryos_directions_endpoints")


def registration_table():
    rows=[]
    for run in base.RUNS:
        key=f"{run['tag']}_{run['source']}_to_{run['target']}"; manifest=json.loads((run["path"]/"prediction_manifest.json").read_text())
        for seed in (42,43,44):
            g=manifest["methods"][str(seed)]["geometry"]
            rows.append({"target_key":key,"stage":run["stage"],"target":run["target"],"seed":seed,"raw_chamfer":g["raw_chamfer"],"registered_chamfer":g["registered_chamfer"],"relative_reduction":(g["raw_chamfer"]-g["registered_chamfer"])/g["raw_chamfer"],"reflection":g["reflection"],"registration_runtime_seconds":g["runtime_seconds"],"final_loss":g["final_loss"]})
    return pd.DataFrame(rows)


def figure_registration(reg):
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),constrained_layout=True)
    for i,row in reg.iterrows(): axes[0].plot([0,1],[row.raw_chamfer,row.registered_chamfer],color="#BBBBBB",lw=.6); axes[0].scatter(0,row.raw_chamfer,c="#999",s=10); axes[0].scatter(1,row.registered_chamfer,c="#009E73",s=10)
    axes[0].set_xticks([0,1],["Raw","Registered"]); axes[0].set_ylabel("Boundary Chamfer (canonical units)"); label(axes[0],"a")
    for i,key in enumerate(reg.target_key.unique()):
        d=reg[reg.target_key==key]; axes[1].scatter(np.full(len(d),i),d.relative_reduction,c="#009E73",s=18)
    axes[1].axhline(0,color="#777",ls=":",lw=.8); axes[1].set_xticks(range(6),[f"{r['tag']} {r['target']}" for r in base.RUNS],rotation=40,ha="right",fontsize=6); axes[1].set_ylabel("Relative Chamfer reduction"); label(axes[1],"b")
    refl=reg.pivot_table(index="target_key",columns="seed",values="reflection",aggfunc="first").reindex([f"{r['tag']}_{r['source']}_to_{r['target']}" for r in base.RUNS])
    axes[2].imshow((refl!="none").astype(int),aspect="auto",cmap=mpl.colors.ListedColormap(["#F2F2F2","#CC79A7"]),vmin=0,vmax=1); axes[2].set_yticks(range(6),[f"{r['tag']} {r['target']}" for r in base.RUNS],fontsize=6); axes[2].set_xticks(range(3),refl.columns); axes[2].set_xlabel("Seed"); axes[2].set_title("Non-identity reflection selected",fontsize=8); axes[2].text(1,2.5,"None selected",ha="center",va="center",fontsize=8,color="#555"); label(axes[2],"c")
    reg.to_csv(SOURCE/"Supplementary_Figure_3_registration_qc.tsv",sep="\t",index=False); save(fig,"Supplementary_Figure_3_registration_QC")


def figure_coordinates():
    fig,axes=plt.subplots(6,2,figsize=(7.2,8.0),constrained_layout=True); rows=[]; rng=np.random.default_rng(20260803)
    volumes=[]
    for stage,tag,embryo,path in [("50% epiboly","50p","E1",base.DATA/"weMERFISH_measured_A_50p_E1.h5ad"),("50% epiboly","50p","E2",base.DATA/"weMERFISH_measured_A_50p_E2.h5ad"),("75% epiboly","75p","E1",base.DATA/"weMERFISH_measured_B_75p_E1.h5ad"),("75% epiboly","75p","E2",base.DATA/"weMERFISH_measured_B_75p_E2.h5ad"),("6-somite","6s","E1",base.DATA/"weMERFISH_measured_C_6s_E1_rescaled_z.h5ad"),("6-somite","6s","E2",base.DATA/"weMERFISH_measured_C_6s_E2_rescaled_z.h5ad")]: volumes.append((stage,tag,embryo,path))
    for i,(stage,tag,embryo,path) in enumerate(volumes):
        coords,_,ids,key=coordinates_only(path); canon,meta=canonicalize(coords); take=np.sort(rng.choice(len(canon),min(2500,len(canon)),replace=False)); c=canon[take]
        axes[i,0].scatter(c[:,0],c[:,1],c=c[:,2],s=.4,cmap="viridis",rasterized=True); axes[i,1].scatter(c[:,0],c[:,2],c=c[:,1],s=.4,cmap="viridis",rasterized=True)
        for ax in axes[i]: ax.set_aspect("equal"); ax.tick_params(labelsize=5)
        axes[i,0].set_ylabel(f"{tag} {embryo}\naxis 2",fontsize=6); axes[i,1].set_ylabel("axis 3",fontsize=6)
        rows.append({"stage":stage,"embryo":embryo,"coordinate_key":key,"spots":len(coords),"canonical_scale":meta["scale"],"axis1_min":canon[:,0].min(),"axis1_max":canon[:,0].max(),"axis2_min":canon[:,1].min(),"axis2_max":canon[:,1].max(),"axis3_min":canon[:,2].min(),"axis3_max":canon[:,2].max(),"spatial_rescaled_z_available":tag=="6s","spatial_rescaled_z_used":False if tag=="6s" else "not_applicable"})
    axes[0,0].set_title("Canonical axes 1–2; color=axis 3",fontsize=8); axes[0,1].set_title("Canonical axes 1–3; color=axis 2",fontsize=8); label(axes[0,0],"a"); axes[0,1].text(-.28,1.14,"b",fontweight="bold",fontsize=10,transform=axes[0,1].transAxes); fig.text(.5,-.01,"Coordinate axes are supplied geometric coordinates, not verified anatomical orientations. Six-somite uses raw spatial, not spatial_rescaled_z.",ha="center",fontsize=7)
    pd.DataFrame(rows).to_csv(SOURCE/"Supplementary_Figure_4_coordinate_qc.tsv",sep="\t",index=False); save(fig,"Supplementary_Figure_4_coordinate_orientation_QC")


def anchor_qc(anchor):
    rows=[]
    for run in base.RUNS:
        key=f"{run['tag']}_{run['source']}_to_{run['target']}"; manifest=json.loads((run["path"]/"prediction_manifest.json").read_text()); counts,_,genes=measured_counts(run["source_file"]); depth=np.maximum(counts.sum(1),1); target=float(np.median(depth)); log=np.log1p(normalize_counts(counts,target))
        for rank,(g,i) in enumerate(zip(manifest["anchor_genes"],manifest["anchor_indices"]),1): rows.append({"target_key":key,"stage":run["stage"],"source":run["source"],"rank":rank,"gene":g,"prevalence":np.mean(counts[:,i]>0),"mean_log1p":np.mean(log[:,i]),"variance_log1p":np.var(log[:,i]),"selection_coverage":manifest["anchor_metadata"]["mean_squared_gene_coverage"]})
    return pd.DataFrame(rows)


def figure_anchors(anchor_qc_df):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2),constrained_layout=True); genes=anchor_qc_df.gene.value_counts().head(30).index; keys=anchor_qc_df.target_key.unique(); mat=np.zeros((len(genes),len(keys)))
    for i,g in enumerate(genes):
        for j,k in enumerate(keys): mat[i,j]=int(((anchor_qc_df.gene==g)&(anchor_qc_df.target_key==k)).any())
    axes[0,0].imshow(mat,aspect="auto",cmap=mpl.colors.ListedColormap(["#F2F2F2","#0072B2"])); axes[0,0].set_yticks(range(len(genes)),genes,fontsize=5); axes[0,0].set_xticks(range(6),[f"{r['tag']} {r['source']}" for r in base.RUNS],rotation=40,ha="right",fontsize=6); axes[0,0].set_title("30 most recurrent anchors",fontsize=8); label(axes[0,0],"a")
    for i,k in enumerate(keys):
        d=anchor_qc_df[anchor_qc_df.target_key==k]; axes[0,1].scatter(np.full(len(d),i),d.prevalence,c="#56B4E9",s=10,alpha=.7); axes[1,0].scatter(np.full(len(d),i),d.mean_log1p,c="#009E73",s=10,alpha=.7)
    for ax,ylabel in [(axes[0,1],"Source detection prevalence"),(axes[1,0],"Source mean log1p expression")]: ax.set_xticks(range(6),[f"{r['tag']} {r['source']}" for r in base.RUNS],rotation=40,ha="right",fontsize=6); ax.set_ylabel(ylabel)
    label(axes[0,1],"b"); label(axes[1,0],"c")
    cov=anchor_qc_df.groupby("target_key").selection_coverage.first().reindex(keys); axes[1,1].scatter(range(6),cov,c="#CC79A7",s=28); axes[1,1].set_xticks(range(6),[f"{r['tag']} {r['source']}" for r in base.RUNS],rotation=40,ha="right",fontsize=6); axes[1,1].set_ylabel("Source-only squared-gene coverage"); label(axes[1,1],"d")
    anchor_qc_df.to_csv(SOURCE/"Supplementary_Figure_5_anchor_expression_qc.tsv",sep="\t",index=False); save(fig,"Supplementary_Figure_5_anchor_panel_expression_QC")


def figure_bootstrap(boot,comparison):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),constrained_layout=True); ordinary=boot[boot.comparison.str.contains("favorable sign")]; cluster=boot[boot.comparison.str.contains("pair-cluster")]; labels=[x[1] for x in base.ENDPOINTS]; y=np.arange(6)
    for d,offset,color,name in [(ordinary,-.12,"#C83E4D","Embryo bootstrap"),(cluster,.12,"#0072B2","Stage-pair sensitivity")]:
        axes[0].errorbar(d.estimate,y+offset,xerr=[d.estimate-d.ci95_low,d.ci95_high-d.estimate],fmt="o",ms=4,capsize=2,color=color,label=name)
    axes[0].axvline(0,color="#777",ls=":",lw=.8); axes[0].set_yticks(y,labels,fontsize=7); axes[0].invert_yaxis(); axes[0].set_xlabel("Favorable MAP − IDW effect"); axes[0].legend(frameon=False,fontsize=6); label(axes[0],"a")
    primary=comparison[comparison.comparison==f"{MAP}_vs_{IDW}"]; loo=[]
    for endpoint,title,_ in base.ENDPOINTS:
        d=primary[primary.endpoint==endpoint]
        for omitted in d.target_key:
            loo.append({"endpoint":endpoint,"endpoint_label":title,"omitted_target":omitted,"estimate":d[d.target_key!=omitted].delta.mean()})
    loo=pd.DataFrame(loo)
    for i,(endpoint,title,_) in enumerate(base.ENDPOINTS):
        d=loo[loo.endpoint==endpoint]; axes[1].scatter(d.estimate,np.full(len(d),i),c="#666",s=15); axes[1].plot([d.estimate.min(),d.estimate.max()],[i,i],color="#AAA",lw=.8)
    axes[1].axvline(0,color="#777",ls=":",lw=.8); axes[1].set_yticks(range(6),labels,fontsize=7); axes[1].invert_yaxis(); axes[1].set_xlabel("Leave-one-target-out favorable effect"); label(axes[1],"b")
    boot.to_csv(SOURCE/"Supplementary_Figure_6_bootstrap_intervals.tsv",sep="\t",index=False); loo.to_csv(SOURCE/"Supplementary_Figure_6_leave_one_target_out.tsv",sep="\t",index=False); save(fig,"Supplementary_Figure_6_bootstrap_and_pair_dependence_diagnostics")


def figure_components(embryo,residual,pareto):
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.2),constrained_layout=True); score=[]
    for m in METHODS:
        row=[]
        for endpoint,_,higher in base.ENDPOINTS:
            vals=embryo.groupby("method")[endpoint].mean().reindex(METHODS); ranks=rankdata(-vals if higher else vals); row.append(ranks[METHODS.index(m)])
        score.append(row)
    im=axes[0].imshow(score,aspect="auto",cmap="viridis_r",vmin=1,vmax=6); axes[0].set_yticks(range(6),[base.METHOD_LABEL[m] for m in METHODS],fontsize=6); axes[0].set_xticks(range(6),[x[1] for x in base.ENDPOINTS],rotation=40,ha="right",fontsize=6); fig.colorbar(im,ax=axes[0],fraction=.05,label="Rank"); label(axes[0],"a")
    for m,row in pareto.iterrows(): axes[1].scatter(row.residual_spatial_pearson_median,row.topk_dice_median,s=70,color=base.COLORS[m],label=base.METHOD_LABEL[m],alpha=.9)
    axes[1].set_xlabel("Residual Pearson"); axes[1].set_ylabel("Dice"); axes[1].legend(frameon=False,fontsize=5.5); label(axes[1],"b")
    ranks=np.asarray(score); wins=(ranks==ranks.min(0)).sum(1); axes[2].scatter(wins,range(6),c=[base.COLORS[m] for m in METHODS],s=35); axes[2].set_yticks(range(6),[base.METHOD_LABEL[m] for m in METHODS],fontsize=6); axes[2].set_xlabel("Pooled endpoint wins/ties"); axes[2].set_xlim(-.2,6.2); label(axes[2],"c")
    pd.DataFrame(score,index=METHODS,columns=[x[0] for x in base.ENDPOINTS]).to_csv(SOURCE/"Supplementary_Figure_7_component_ranks.tsv",sep="\t"); pareto.reset_index().to_csv(SOURCE/"Supplementary_Figure_7_component_pareto.tsv",sep="\t",index=False); save(fig,"Supplementary_Figure_7_component_ablation_and_Pareto")


def runtime_table():
    rows=[]
    for run in base.RUNS:
        d=json.loads((run["path"]/"STAGE_DECISION.json").read_text()); target_counts,_,_=measured_counts(run["target_file"])
        rows.append({"target_key":f"{run['tag']}_{run['source']}_to_{run['target']}","stage":run["stage"],"target":run["target"],"target_spots":len(target_counts),"total_runtime_seconds":d["runtime_seconds"],"peak_cpu_rss_gb":d["peak_cpu_rss_kb"]/1024**2,"seeds":3,"methods":6,"serialized_prediction_arrays":18})
    return pd.DataFrame(rows)


def figure_runtime(runtime):
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),constrained_layout=True)
    offsets=[(4,6),(4,6),(4,8),(4,-11),(4,6),(4,-11)]
    for (i,r),off in zip(runtime.iterrows(),offsets): axes[0].scatter(r.target_spots,r.total_runtime_seconds/60,c="#0072B2",s=30); axes[0].annotate(r.target_key,(r.target_spots,r.total_runtime_seconds/60),fontsize=5,xytext=off,textcoords="offset points")
    axes[0].set_xlabel("Target spots"); axes[0].set_ylabel("Complete locked run time (min)"); label(axes[0],"a")
    for (i,r),off in zip(runtime.iterrows(),offsets): axes[1].scatter(r.target_spots,r.peak_cpu_rss_gb,c="#009E73",s=30); axes[1].annotate(r.target_key,(r.target_spots,r.peak_cpu_rss_gb),fontsize=5,xytext=off,textcoords="offset points")
    axes[1].set_xlabel("Target spots"); axes[1].set_ylabel("Peak CPU RSS (GB)"); label(axes[1],"b")
    runtime.to_csv(SOURCE/"Supplementary_Figure_8_runtime_memory.tsv",sep="\t",index=False); save(fig,"Supplementary_Figure_8_runtime_and_memory")


def figure_axial_gene(gene,number):
    fig,axes=plt.subplots(6,4,figsize=(7.2,8.0),constrained_layout=True)
    for row,run in enumerate(base.RUNS):
        manifest=json.loads((run["path"]/"prediction_manifest.json").read_text()); source_counts,_,genes=measured_counts(run["source_file"]); depth=float(np.median(np.maximum(source_counts.sum(1),1))); target_counts,coords,_=measured_counts(run["target_file"]); coords=canonicalize(coords)[0]; truth=np.log1p(normalize_counts(target_counts,depth)); gi=int(np.flatnonzero(genes==gene)[0]); paths=manifest["methods"]["42"]["predictions"]
        values=[truth[:,gi],np.load(paths[IDW]["path"],mmap_mode="r")[:,gi],np.load(paths[ANCHOR]["path"],mmap_mode="r")[:,gi],np.load(paths[MAP]["path"],mmap_mode="r")[:,gi]]; vmax=np.quantile(np.concatenate(values),.99)
        for col,(v,title) in enumerate(zip(values,["Measured","Registered IDW","Anchor-only","HyperSpatial-MAP"])):
            axes[row,col].scatter(coords[:,0],coords[:,1],c=v,s=.35,cmap="viridis",vmin=0,vmax=vmax,rasterized=True); axes[row,col].set_aspect("equal"); axes[row,col].axis("off");
            if row==0: axes[row,col].set_title(title,fontsize=7)
        axes[row,0].text(-.1,.5,f"{run['tag']} {run['source']}→{run['target']}",rotation=90,va="center",ha="right",transform=axes[row,0].transAxes,fontsize=6)
    axes[0,0].text(-.2,1.15,"a",fontweight="bold",transform=axes[0,0].transAxes); fig.text(.5,-.005,f"{gene}: shared expression scale within each target row; two-dimensional projection of supplied 3D coordinates.",ha="center",fontsize=7)
    save(fig,f"Supplementary_Figure_{number}_{gene}_all_targets_domain_maps")


def write_caption_manifest(files):
    captions="""# Supplementary figure captions

**Supplementary Figure 1 | Per-gene performance distributions.** ECDFs and paired gene-level improvements for source-selected SVGs and source-predictable non-anchor genes across six target volumes. Genes are technical inner units; target embryos are biological units.

**Supplementary Figure 2 | All embryos, directions and locked endpoints.** Absolute embryo-level values for all six methods, six target volumes and the six primary endpoints, residual Pearson and secondary Moran error.

**Supplementary Figure 3 | Geometry-only registration QC.** Raw and registered Chamfer distances, relative reduction and reflection selection across three frozen seeds. Coordinates alone were used.

**Supplementary Figure 4 | Coordinate and orientation QC.** Canonical coordinate projections for all six measured volumes. Axes are supplied geometry, not verified anatomical orientation. Six-somite used raw `spatial`, not `spatial_rescaled_z`.

**Supplementary Figure 5 | Source-selected anchor-panel QC.** Panel overlap, source detection prevalence, normalized mean expression and source-only coverage. Biological programme labels were not inferred post hoc.

**Supplementary Figure 6 | Bootstrap and reciprocal-pair diagnostics.** Embryo-first confidence intervals, stage-pair cluster sensitivity and leave-one-target-out effects.

**Supplementary Figure 7 | Component specialization.** Pooled endpoint ranks, residual-localization Pareto structure and endpoint wins. No component is presented as universally optimal.

**Supplementary Figure 8 | Runtime and memory.** Complete three-seed, six-method locked-run wall time and peak process RSS. These are pipeline measurements, not isolated inference benchmarks.

**Supplementary Figures 9–11 | Predeclared axial genes across all targets.** Measured, registered IDW, calibrated anchor-only and HyperSpatial-MAP maps for `tbxta`, `noto` and `shha`. All were excluded from anchor selection. Scales are shared within each target row; views are two-dimensional projections.
"""
    (OUT/"SUPPLEMENTARY_FIGURE_CAPTIONS.md").write_text(captions)
    manifest=[]
    for p in sorted(OUT.iterdir()):
        if p.name == "SUPPLEMENTARY_FIGURE_MANIFEST.json":
            continue
        if p.is_file(): manifest.append({"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size})
    for p in sorted(SOURCE.iterdir()): manifest.append({"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size})
    (OUT/"SUPPLEMENTARY_FIGURE_MANIFEST.json").write_text(json.dumps({"status":"frozen_outputs_only","figures":11,"unsupported_not_generated":["predictive uncertainty calibration","permutation or matched-random panels","unfrozen broad biological modules"],"artifacts":manifest},indent=2)+"\n")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    SOURCE.mkdir()
    mpl.rcParams.update({"font.family":"Liberation Sans","font.size":8,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    embryo=pd.read_csv(ROOT/"05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv",sep="\t"); residual=pd.read_csv(ROOT/"05_metrics/RESIDUAL_ENDPOINTS_BY_EMBRYO.tsv",sep="\t"); gene=pd.read_csv(ROOT/"05_metrics/PER_GENE_SPATIAL_AND_RESIDUAL_METRICS.tsv",sep="\t"); anchor=pd.read_csv(ROOT/"03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv",sep="\t"); boot=pd.read_csv(ROOT/"06_hierarchical_statistics/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv",sep="\t"); comparison=pd.read_csv(ROOT/"05_metrics/METHOD_COMPARISONS_BY_EMBRYO.tsv",sep="\t"); pareto=pd.read_csv(ROOT/"05_metrics/COMPONENT_PARETO_RESULTS.tsv",sep="\t").set_index("method")
    figure_gene_distributions(gene); figure_all_targets(embryo,residual); reg=registration_table(); figure_registration(reg); figure_coordinates(); aq=anchor_qc(anchor); figure_anchors(aq); figure_bootstrap(boot,comparison); figure_components(embryo,residual,pareto); runtime=runtime_table(); figure_runtime(runtime); figure_axial_gene("tbxta",9); figure_axial_gene("noto",10); figure_axial_gene("shha",11); write_caption_manifest(11)
    print(json.dumps({"output":str(OUT),"figures":11}))


if __name__=="__main__": main()
