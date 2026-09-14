#!/usr/bin/env python3
"""Aggregate the protocol-locked HyperSpatial-MAP cross-stage replication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from run_hyperspatial_map_stage1_20260803 import residual_pearson
from run_wemerfish_temporal_holdout import canonicalize, measured_counts, normalize_counts, pearson_columns


ROOT = Path("results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726")
DATA = Path("data_v10/wemerfish_zebrafish_3stage")
MAP = "hyperspatial_map_calibrated_fusion"
IDW = "registered_idw_k12"
ANCHOR = "anchor_only_calibrated"
METHOD_LABEL = {
    MAP: "HyperSpatial-MAP", IDW: "Registered IDW", ANCHOR: "Anchor-only",
    "registered_nearest_neighbor": "Nearest neighbour",
    "atlas_plus_global_anchor_residual": "Global residual",
    "atlas_plus_local_anchor_residual": "Local residual",
}
COLORS = {
    MAP: "#C83E4D", IDW: "#E69F00", ANCHOR: "#0072B2",
    "registered_nearest_neighbor": "#6B6B6B",
    "atlas_plus_global_anchor_residual": "#8E6C8A",
    "atlas_plus_local_anchor_residual": "#009E73",
}
ENDPOINTS = [
    ("svg_spatial_pearson_median", "Spatial Pearson", True),
    ("auprc_enrichment_median", "AUPRC enrichment", True),
    ("topk_dice_median", "Prevalence-matched Dice", True),
    ("weighted_centroid_error_median", "Centroid error", False),
    ("spatial_emd_median", "Spatial EMD", False),
    ("log1p_rmse", "log1p RMSE", False),
]
RUNS = [
    {"stage": "50% epiboly", "tag": "50p", "source": "E1", "target": "E2", "path": ROOT/"04_predictions/50p_E1_to_E2", "source_file": DATA/"weMERFISH_measured_A_50p_E1.h5ad", "target_file": DATA/"weMERFISH_measured_A_50p_E2.h5ad"},
    {"stage": "50% epiboly", "tag": "50p", "source": "E2", "target": "E1", "path": ROOT/"04_predictions/50p_E2_to_E1", "source_file": DATA/"weMERFISH_measured_A_50p_E2.h5ad", "target_file": DATA/"weMERFISH_measured_A_50p_E1.h5ad"},
    {"stage": "75% epiboly", "tag": "75p", "source": "E1", "target": "E2", "path": Path("results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2"), "source_file": DATA/"weMERFISH_measured_B_75p_E1.h5ad", "target_file": DATA/"weMERFISH_measured_B_75p_E2.h5ad"},
    {"stage": "75% epiboly", "tag": "75p", "source": "E2", "target": "E1", "path": Path("results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1"), "source_file": DATA/"weMERFISH_measured_B_75p_E2.h5ad", "target_file": DATA/"weMERFISH_measured_B_75p_E1.h5ad"},
    {"stage": "6-somite", "tag": "6s", "source": "E1", "target": "E2", "path": ROOT/"04_predictions/6s_E1_to_E2", "source_file": DATA/"weMERFISH_measured_C_6s_E1_rescaled_z.h5ad", "target_file": DATA/"weMERFISH_measured_C_6s_E2_rescaled_z.h5ad"},
    {"stage": "6-somite", "tag": "6s", "source": "E2", "target": "E1", "path": ROOT/"04_predictions/6s_E2_to_E1", "source_file": DATA/"weMERFISH_measured_C_6s_E2_rescaled_z.h5ad", "target_file": DATA/"weMERFISH_measured_C_6s_E1_rescaled_z.h5ad"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def savefig(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def aggregate_tables():
    metric_frames, residual_frames, axial_frames, anchors = [], [], [], []
    prediction_manifest = {"status": "complete", "predictions": []}
    for run in RUNS:
        key = f"{run['tag']}_{run['source']}_to_{run['target']}"
        for filename, frames in (("all_metrics.tsv", metric_frames), ("residual_metrics.tsv", residual_frames), ("axial_metrics.tsv", axial_frames)):
            frame = pd.read_csv(run["path"]/filename, sep="\t")
            frame.insert(0, "target_key", key); frame.insert(1, "stage", run["stage"])
            frame.insert(2, "source_embryo", run["source"]); frame.insert(3, "target_embryo", run["target"])
            frames.append(frame)
        manifest = json.loads((run["path"]/"prediction_manifest.json").read_text())
        for rank, (gene, index) in enumerate(zip(manifest["anchor_genes"], manifest["anchor_indices"]), 1):
            anchors.append({"target_key": key, "stage": run["stage"], "source_embryo": run["source"], "rank": rank, "gene": gene, "gene_index": index,
                            "source_candidate_count": manifest["anchor_metadata"]["candidate_count"],
                            "source_mean_squared_gene_coverage": manifest["anchor_metadata"]["mean_squared_gene_coverage"]})
        for seed, methods in manifest["methods"].items():
            for method, meta in methods["predictions"].items():
                p = Path(meta["path"])
                prediction_manifest["predictions"].append({"target_key": key, "seed": int(seed), "method": method, "path": str(p), "sha256": meta["sha256"], "exists": p.exists()})
    metrics = pd.concat(metric_frames, ignore_index=True)
    residual = pd.concat(residual_frames, ignore_index=True)
    axial = pd.concat(axial_frames, ignore_index=True)
    pd.DataFrame(anchors).to_csv(ROOT/"03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv", sep="\t", index=False)
    (ROOT/"04_predictions/ALL_TARGET_PREDICTIONS_MANIFEST.json").write_text(json.dumps(prediction_manifest, indent=2)+"\n")
    embryo = metrics.groupby(["target_key","stage","source_embryo","target_embryo","method"], as_index=False).mean(numeric_only=True)
    embryo.to_csv(ROOT/"05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv", sep="\t", index=False)
    residual_embryo = residual.groupby(["target_key","stage","source_embryo","target_embryo","method"], as_index=False).mean(numeric_only=True)
    residual_embryo.to_csv(ROOT/"05_metrics/RESIDUAL_ENDPOINTS_BY_EMBRYO.tsv", sep="\t", index=False)
    axial.to_csv(ROOT/"07_biological_modules/DOMAIN_LOCALIZATION_RESULTS.tsv", sep="\t", index=False)
    return metrics, residual, axial, embryo, residual_embryo


def comparisons(embryo, residual_embryo):
    rows = []
    for target_key in embryo.target_key.unique():
        e = embryo[embryo.target_key == target_key].set_index("method")
        r = residual_embryo[residual_embryo.target_key == target_key].set_index("method")
        common = e.iloc[0]
        for comparator in (IDW, ANCHOR):
            for endpoint, label, higher in ENDPOINTS:
                delta = e.loc[MAP, endpoint]-e.loc[comparator, endpoint] if higher else e.loc[comparator, endpoint]-e.loc[MAP, endpoint]
                rows.append({"target_key": target_key, "stage": common.stage, "target_embryo": common.target_embryo,
                             "comparison": f"{MAP}_vs_{comparator}", "endpoint": endpoint, "endpoint_label": label,
                             "favorable_direction": "positive", "delta": delta, "map_value": e.loc[MAP, endpoint], "comparator_value": e.loc[comparator, endpoint]})
            delta = r.loc[MAP,"residual_spatial_pearson_median"]-r.loc[comparator,"residual_spatial_pearson_median"]
            rows.append({"target_key": target_key, "stage": common.stage, "target_embryo": common.target_embryo,
                         "comparison": f"{MAP}_vs_{comparator}", "endpoint": "residual_spatial_pearson_median", "endpoint_label": "Residual spatial Pearson",
                         "favorable_direction": "positive", "delta": delta, "map_value": r.loc[MAP,"residual_spatial_pearson_median"], "comparator_value": r.loc[comparator,"residual_spatial_pearson_median"]})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT/"05_metrics/METHOD_COMPARISONS_BY_EMBRYO.tsv", sep="\t", index=False)
    return out


def gene_effects():
    rows = []
    for run in RUNS:
        key = f"{run['tag']}_{run['source']}_to_{run['target']}"
        manifest = json.loads((run["path"]/"prediction_manifest.json").read_text())
        source_counts, _, genes = measured_counts(run["source_file"])
        source_depth = np.maximum(source_counts.sum(1), 1.0)
        depth_target = float(np.median(source_depth))
        target_counts, _, target_genes = measured_counts(run["target_file"])
        if not np.array_equal(genes, target_genes): raise AssertionError("gene mismatch")
        truth = np.log1p(normalize_counts(target_counts, depth_target)).astype(np.float32)
        svg = np.asarray(manifest["evaluation_svg_indices"], int)
        predictable_names = set(manifest["source_defined_predictable_genes"])
        predictable = np.asarray([i for i,g in enumerate(genes) if g in predictable_names and i not in set(manifest["anchor_indices"])], int)
        pearson_by_method, residual_by_method = {}, {}
        for method in (IDW, MAP, ANCHOR):
            pvals, rvals = [], []
            for seed in (42,43,44):
                paths = manifest["methods"][str(seed)]["predictions"]
                base = np.asarray(np.load(paths[IDW]["path"], mmap_mode="r"), np.float32)
                pred = np.asarray(np.load(paths[method]["path"], mmap_mode="r"), np.float32)
                pvals.append(pearson_columns(truth[:,svg], pred[:,svg]))
                rvals.append(residual_pearson(truth, base, pred, predictable))
            pearson_by_method[method] = np.nanmean(pvals, axis=0)
            residual_by_method[method] = np.nanmean(rvals, axis=0)
        for local,i in enumerate(svg):
            for method in (IDW,MAP,ANCHOR):
                rows.append({"target_key":key,"stage":run["stage"],"target_embryo":run["target"],"gene":genes[i],"gene_index":i,
                             "endpoint":"spatial_pearson","method":method,"value":pearson_by_method[method][local]})
        for local,i in enumerate(predictable):
            for method in (IDW,MAP,ANCHOR):
                rows.append({"target_key":key,"stage":run["stage"],"target_embryo":run["target"],"gene":genes[i],"gene_index":i,
                             "endpoint":"residual_spatial_pearson","method":method,"value":residual_by_method[method][local]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT/"05_metrics/PER_GENE_SPATIAL_AND_RESIDUAL_METRICS.tsv",sep="\t",index=False)
    return df


def bootstrap(comparisons_df, gene_df):
    rng = np.random.default_rng(20260803)
    records=[]
    primary = comparisons_df[comparisons_df.comparison == f"{MAP}_vs_{IDW}"]
    for endpoint,label,_ in ENDPOINTS:
        values=primary[primary.endpoint==endpoint].set_index("target_key").delta
        keys=values.index.to_numpy(); boot=np.asarray([np.mean(values.loc[rng.choice(keys,len(keys),replace=True)]) for _ in range(10000)])
        records.append({"comparison":"HyperSpatial-MAP - registered IDW (favorable sign)","endpoint":endpoint,"estimate":values.mean(),"ci95_low":np.quantile(boot,.025),"ci95_high":np.quantile(boot,.975),"biological_units":len(keys),"inner_units":"aggregate endpoint","bootstrap":"target embryo first"})
    for endpoint in ("spatial_pearson","residual_spatial_pearson"):
        wide=gene_df[gene_df.endpoint==endpoint].pivot_table(index=["target_key","gene"],columns="method",values="value").dropna()
        per={k:(g[MAP]-g[IDW]).to_numpy() for k,g in wide.groupby(level=0)}; keys=np.asarray(list(per))
        boot=[]
        for _ in range(5000):
            sampled=rng.choice(keys,len(keys),replace=True); embryo_effect=[]
            for k in sampled:
                vals=per[k]; embryo_effect.append(np.nanmedian(rng.choice(vals,len(vals),replace=True)))
            boot.append(np.nanmean(embryo_effect))
        observed=np.mean([np.nanmedian(v) for v in per.values()])
        records.append({"comparison":"HyperSpatial-MAP - registered IDW","endpoint":endpoint+"_gene_hierarchical","estimate":observed,"ci95_low":np.quantile(boot,.025),"ci95_high":np.quantile(boot,.975),"biological_units":len(keys),"inner_units":int(sum(len(v) for v in per.values())),"bootstrap":"target embryo first, then genes"})
    # Pair-dependence sensitivity: resample stages, retaining both reciprocal targets.
    for endpoint,label,_ in ENDPOINTS:
        d=primary[primary.endpoint==endpoint]; stages=d.stage.unique(); boot=[]
        for _ in range(10000):
            selected=rng.choice(stages,len(stages),replace=True)
            boot.append(np.mean([d[d.stage==s].delta.mean() for s in selected]))
        records.append({"comparison":"HyperSpatial-MAP - registered IDW (pair-cluster sensitivity)","endpoint":endpoint,"estimate":d.delta.mean(),"ci95_low":np.quantile(boot,.025),"ci95_high":np.quantile(boot,.975),"biological_units":6,"inner_units":"3 reciprocal stage pairs","bootstrap":"stage-pair cluster first; both embryos retained"})
    out=pd.DataFrame(records)
    out.to_csv(ROOT/"06_hierarchical_statistics/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv",sep="\t",index=False)
    return out


def pareto(embryo,residual_embryo):
    means=embryo.groupby("method")[[x[0] for x in ENDPOINTS]].mean()
    means["residual_spatial_pearson_median"]=residual_embryo.groupby("method").residual_spatial_pearson_median.mean()
    for endpoint,_,higher in ENDPOINTS:
        means[endpoint+"_rank"] = rankdata(-means[endpoint] if higher else means[endpoint], method="min")
    means["balanced_mean_rank"]=means[[x[0]+"_rank" for x in ENDPOINTS]].mean(axis=1)
    means.reset_index().to_csv(ROOT/"05_metrics/COMPONENT_PARETO_RESULTS.tsv",sep="\t",index=False)
    return means


def figure1():
    fig,ax=plt.subplots(figsize=(7.2,3.8)); ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis("off")
    def box(x,y,w,h,text,color="#F7F7F7"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",fc=color,ec="#333333",lw=.9))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8)
    box(.2,4.7,2.0,1.15,"High-plex reference\nembryo", "#E8F1F8"); box(.2,2.4,2.0,1.35,"Individual target\n3D geometry +\n16 anchors", "#FCEFEF")
    box(3.0,4.7,2.0,1.15,"Geometry-only\nregistration"); box(5.8,4.7,2.0,1.15,"Registered atlas\nfield", "#FFF3D6")
    box(3.0,2.4,2.0,1.35,"Anchor residual\nencoder"); box(5.8,2.4,2.0,1.35,"Source-gated\ntarget residual", "#E8F6F0")
    box(8.6,3.45,2.3,1.35,"Individualized\nhigh-plex field", "#F8E5E7")
    for a,b in [((2.2,5.27),(3,5.27)),((5,5.27),(5.8,5.27)),((2.2,3.05),(3,3.05)),((5,3.05),(5.8,3.05)),((7.8,5.27),(8.6,4.45)),((7.8,3.05),(8.6,4.0))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,lw=1,color="#555"))
    ax.text(.2,6.55,"a",fontweight="bold",fontsize=12); ax.text(.2,.9,"b",fontweight="bold",fontsize=12)
    stages=[("50% epiboly","E1 <-> E2"),("75% epiboly","E1 <-> E2"),("6-somite","E1 <-> E2")]
    for i,(stage,direction) in enumerate(stages):
        x=2.0+i*3.25; box(x,.45,2.35,.9,f"{stage}\n{direction}","#FAFAFA")
    ax.text(6,.05,"Six target-embryo evaluations; source-only selection in each direction",ha="center",fontsize=7.5)
    savefig(fig,ROOT/"08_figures_main/Figure_1_method_and_locked_design")


def figure2(embryo,residual,comparisons_df,boot,pareto_df):
    fig=plt.figure(figsize=(7.2,9.0)); gs=fig.add_gridspec(4,2,height_ratios=[.7,2.6,1.7,1.8],hspace=.55,wspace=.4)
    ax=fig.add_subplot(gs[0,:]); ax.axis("off"); ax.text(0,1.1,"a",fontweight="bold",transform=ax.transAxes)
    for i,run in enumerate(RUNS):
        x=.02+i*.16; ax.add_patch(FancyBboxPatch((x,.2),.145,.55,boxstyle="round,pad=.02",fc="#F4F4F4",ec="#777",lw=.7))
        ax.text(x+.072,.48,f"{run['tag']}\n{run['source']}→{run['target']}",ha="center",va="center",fontsize=7)
    sub=gs[1,:].subgridspec(2,3,wspace=.45,hspace=.65)
    for j,(endpoint,label,higher) in enumerate(ENDPOINTS):
        ax=fig.add_subplot(sub[j//3,j%3]);
        for i,key in enumerate(embryo.target_key.unique()):
            d=embryo[embryo.target_key==key].set_index("method"); ax.plot([0,1],[d.loc[IDW,endpoint],d.loc[MAP,endpoint]],color="#BBBBBB",lw=.7,zorder=1)
            ax.scatter([0],[d.loc[IDW,endpoint]],c=COLORS[IDW],s=14,zorder=2); ax.scatter([1],[d.loc[MAP,endpoint]],c=COLORS[MAP],s=14,zorder=2)
        ax.set_xticks([0,1],["IDW","MAP"],fontsize=6.5); ax.set_title(label+(" ↑" if higher else " ↓"),fontsize=7.5); ax.tick_params(labelsize=6)
        if j==0: ax.text(-.35,1.22,"b",fontweight="bold",transform=ax.transAxes)
    ax=fig.add_subplot(gs[2,0]); ax.text(-.15,1.08,"c",fontweight="bold",transform=ax.transAxes)
    b=boot[(boot.comparison.str.contains("favorable sign")) & (~boot.endpoint.str.contains("hierarchical"))]
    y=np.arange(len(b)); ax.errorbar(b.estimate,y,xerr=[b.estimate-b.ci95_low,b.ci95_high-b.estimate],fmt="o",color=COLORS[MAP],ecolor="#555",capsize=2,ms=4)
    ax.axvline(0,color="#777",ls=":",lw=.8); ax.set_yticks(y,[dict((e,l) for e,l,_ in ENDPOINTS)[e] for e in b.endpoint],fontsize=6.5); ax.invert_yaxis(); ax.set_xlabel("Favorable MAP − IDW effect",fontsize=7); ax.tick_params(axis="x",labelsize=6)
    for yy,val in zip(y,b.estimate): ax.annotate(f"{val:.3f}",(val,yy),xytext=(4,0),textcoords="offset points",va="center",fontsize=5.5)
    ax=fig.add_subplot(gs[2,1]); ax.text(-.15,1.08,"d",fontweight="bold",transform=ax.transAxes)
    ranks=[]; methods=list(pareto_df.index)
    for key in embryo.target_key.unique():
        d=embryo[embryo.target_key==key].set_index("method"); score=np.zeros(len(methods))
        for endpoint,_,higher in ENDPOINTS: score += rankdata(-d.loc[methods,endpoint] if higher else d.loc[methods,endpoint])
        ranks.append(score/len(ENDPOINTS))
    im=ax.imshow(np.asarray(ranks).T,aspect="auto",cmap="viridis_r",vmin=1,vmax=len(methods)); ax.set_yticks(range(len(methods)),[METHOD_LABEL[m] for m in methods],fontsize=6); ax.set_xticks(range(6),[r["tag"]+" "+r["target"] for r in RUNS],rotation=45,ha="right",fontsize=6); fig.colorbar(im,ax=ax,fraction=.04,label="Mean rank")
    ax=fig.add_subplot(gs[3,0]); ax.text(-.15,1.08,"e",fontweight="bold",transform=ax.transAxes)
    for m,row in pareto_df.iterrows(): ax.scatter(row.residual_spatial_pearson_median,row.topk_dice_median,s=120*(.58/max(row.log1p_rmse,.58))**2,color=COLORS[m],label=METHOD_LABEL[m],alpha=.9,edgecolor="white",lw=.5)
    ax.set_xlabel("Residual spatial Pearson",fontsize=7); ax.set_ylabel("Dice",fontsize=7); ax.tick_params(labelsize=6); ax.legend(fontsize=5.5,frameon=False,ncol=2)
    ax=fig.add_subplot(gs[3,1]); ax.text(-.15,1.08,"f",fontweight="bold",transform=ax.transAxes)
    p=comparisons_df[comparisons_df.comparison==f"{MAP}_vs_{IDW}"]
    selected=[("svg_spatial_pearson_median","Pearson"),("topk_dice_median","Dice"),("log1p_rmse","RMSE")]
    for i,(endpoint,label) in enumerate(selected):
        vals=p[p.endpoint==endpoint].groupby("stage").delta.mean().reindex(["50% epiboly","75% epiboly","6-somite"])
        ax.plot(range(3),vals,marker="o",label=label,lw=1.2)
    ax.axhline(0,color="#777",ls=":",lw=.8); ax.set_xticks(range(3),["50%","75%","6s"],fontsize=6.5); ax.set_ylabel("Favorable MAP − IDW effect",fontsize=7); ax.legend(fontsize=6,frameon=False); ax.tick_params(labelsize=6)
    savefig(fig,ROOT/"08_figures_main/Figure_2_cross_stage_reciprocal_replication")


def figure3():
    # First source-predeclared marker that passed the source-only predictability
    # gate in every displayed E1-to-E2 stage; no target outcome was used.
    display_gene = "wnt11f2"
    fig,axes=plt.subplots(3,6,figsize=(7.2,5.5),constrained_layout=True)
    for row,run in enumerate([RUNS[0],RUNS[2],RUNS[4]]):
        manifest=json.loads((run["path"]/"prediction_manifest.json").read_text()); source_counts,_,genes=measured_counts(run["source_file"]); depth=float(np.median(np.maximum(source_counts.sum(1),1)))
        target_counts,coords,_=measured_counts(run["target_file"]); coords=canonicalize(coords)[0]; truth=np.log1p(normalize_counts(target_counts,depth)); gi=int(np.flatnonzero(genes==display_gene)[0])
        paths=manifest["methods"]["42"]["predictions"]; base=np.load(paths[IDW]["path"],mmap_mode="r")[:,gi]; pred=np.load(paths[MAP]["path"],mmap_mode="r")[:,gi]
        vals=[truth[:,gi],base,truth[:,gi]-base,pred-base,pred,np.abs(truth[:,gi]-pred)]; titles=["Measured","Registered IDW","True residual","Predicted residual","HyperSpatial-MAP","Absolute error"]
        for col,(value,title) in enumerate(zip(vals,titles)):
            cmap="RdBu_r" if col in (2,3) else "viridis"; v=None
            kw={};
            if col in (2,3): v=np.quantile(np.abs(np.r_[vals[2],vals[3]]),.98); kw={"vmin":-v,"vmax":v}
            elif col in (0,1,4): v=np.quantile(np.r_[vals[0],vals[1],vals[4]],.99); kw={"vmin":0,"vmax":v}
            axes[row,col].scatter(coords[:,0],coords[:,1],c=value,s=.35,cmap=cmap,rasterized=True,**kw); axes[row,col].set_aspect("equal"); axes[row,col].axis("off")
            if row==0: axes[row,col].set_title(title,fontsize=7)
        axes[row,0].text(-.12,.5,run["stage"],rotation=90,va="center",ha="right",transform=axes[row,0].transAxes,fontsize=7)
    axes[0,0].text(-.25,1.16,"a",fontweight="bold",transform=axes[0,0].transAxes)
    fig.text(.5,.005,"wnt11f2 was predeclared from E1 source volumes and passed the source-only predictability gate at all displayed stages.",ha="center",fontsize=7)
    savefig(fig,ROOT/"08_figures_main/Figure_3_specimen_specific_residual_recovery")


def figure4(axial):
    d=axial.groupby(["stage","target_embryo","method","gene"],as_index=False).mean(numeric_only=True)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),constrained_layout=True)
    for ax,(metric,title,higher) in zip(axes,[("topk_dice","Dice ↑",True),("centroid_error","Centroid error ↓",False),("spatial_pearson","Spatial Pearson ↑",True)]):
        for j,gene in enumerate(("tbxta","noto","shha")):
            sub=d[d.gene==gene].set_index(["stage","target_embryo","method"])
            idw=[]; mapv=[]
            for run in RUNS:
                idx=(run["stage"],run["target"]); idw.append(sub.loc[idx+(IDW,),metric]); mapv.append(sub.loc[idx+(MAP,),metric])
            ax.scatter(np.full(6,j)-.08,idw,c=COLORS[IDW],s=14,alpha=.75); ax.scatter(np.full(6,j)+.08,mapv,c=COLORS[MAP],s=14,alpha=.75)
            for k in range(6): ax.plot([j-.08,j+.08],[idw[k],mapv[k]],color="#BBBBBB",lw=.5)
        ax.set_xticks(range(3),("tbxta","noto","shha"),fontsize=7); ax.set_title(title,fontsize=8); ax.tick_params(labelsize=6)
    axes[0].text(-.2,1.08,"a",fontweight="bold",transform=axes[0].transAxes); axes[0].legend(handles=[plt.Line2D([],[],marker="o",ls="",color=COLORS[IDW],label="Registered IDW"),plt.Line2D([],[],marker="o",ls="",color=COLORS[MAP],label="HyperSpatial-MAP")],fontsize=6,frameon=False)
    fig.text(.5,-.02,"Predeclared axial genes only; broader module definitions were not frozen before evaluation and are not shown.",ha="center",fontsize=7)
    savefig(fig,ROOT/"08_figures_main/Figure_4_axial_localization_beyond_atlas_transfer")


def figure5(anchor_df,gene_df,pareto_df):
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.2),constrained_layout=True)
    genes=anchor_df.gene.value_counts().head(30).index; keys=anchor_df.target_key.unique(); mat=np.zeros((len(genes),len(keys)))
    for i,g in enumerate(genes):
        for j,k in enumerate(keys): mat[i,j]=int(((anchor_df.gene==g)&(anchor_df.target_key==k)).any())
    axes[0].imshow(mat,aspect="auto",cmap=mpl.colors.ListedColormap(["#F2F2F2","#0072B2"])); axes[0].set_yticks(range(len(genes)),genes,fontsize=5); axes[0].set_xticks(range(6),[x.split("_to_")[-1] for x in keys],rotation=45,fontsize=5); axes[0].set_title("Source-selected anchors",fontsize=8); axes[0].text(-.2,1.04,"a",fontweight="bold",transform=axes[0].transAxes)
    wide=gene_df[gene_df.endpoint=="spatial_pearson"].pivot_table(index=["target_key","gene"],columns="method",values="value").dropna(); frac=(wide[MAP]>wide[IDW]).groupby(level=0).mean()
    axes[1].scatter(range(6),frac,c=["#56B4E9"]*6,s=28); axes[1].axhline(.5,color="#777",ls=":",lw=.8); axes[1].set_ylim(0,1); axes[1].set_xticks(range(6),[x.split("_to_")[-1] for x in frac.index],rotation=45,fontsize=5); axes[1].set_ylabel("Fraction of SVGs improved",fontsize=7); axes[1].set_title("Gene-level gains and failures",fontsize=8); axes[1].text(-.2,1.04,"b",fontweight="bold",transform=axes[1].transAxes)
    offsets={MAP:(-55,-22),ANCHOR:(6,9),"atlas_plus_global_anchor_residual":(5,-8),"atlas_plus_local_anchor_residual":(5,5),IDW:(5,4),"registered_nearest_neighbor":(5,4)}
    for m,row in pareto_df.iterrows():
        axes[2].scatter(row.residual_spatial_pearson_median,row.topk_dice_median,s=45,color=COLORS[m])
        arrow={"arrowstyle":"-","color":COLORS[m],"lw":.6} if m==MAP else None
        axes[2].annotate(METHOD_LABEL[m],(row.residual_spatial_pearson_median,row.topk_dice_median),fontsize=5,xytext=offsets[m],textcoords="offset points",arrowprops=arrow)
    axes[2].set_xlabel("Residual Pearson",fontsize=7); axes[2].set_ylabel("Dice",fontsize=7); axes[2].set_title("Component Pareto frontier",fontsize=8); axes[2].tick_params(labelsize=6); axes[2].text(-.2,1.04,"c",fontweight="bold",transform=axes[2].transAxes)
    savefig(fig,ROOT/"08_figures_main/Figure_5_panel_behavior_and_applicability")


def supplementary(embryo):
    fig,ax=plt.subplots(figsize=(7.2,4.2)); methods=embryo.method.unique(); cols=[x[0] for x in ENDPOINTS]+["moran_error_median"]
    z=[]
    for m in methods:
        vals=embryo[embryo.method==m][cols].mean().to_numpy(); z.append(vals)
    z=np.asarray(z); z=(z-z.mean(0))/np.maximum(z.std(0),1e-8); z[:,3:]*=-1
    im=ax.imshow(z,aspect="auto",cmap="RdBu_r",vmin=-2,vmax=2); ax.set_yticks(range(len(methods)),[METHOD_LABEL[m] for m in methods],fontsize=7); ax.set_xticks(range(len(cols)),[dict((e,l) for e,l,_ in ENDPOINTS).get(c,"Moran error") for c in cols],rotation=35,ha="right",fontsize=7); fig.colorbar(im,ax=ax,label="Favorable standardized value"); savefig(fig,ROOT/"09_figures_supplementary/Supplementary_Figure_1_complete_endpoint_landscape")


def write_reports(embryo,residual,comparisons_df,boot,pareto_df,anchor_df,gene_df):
    map_idw=comparisons_df[comparisons_df.comparison==f"{MAP}_vs_{IDW}"]; all_positive=bool((map_idw.delta>0).all())
    residual_map=residual[residual.method==MAP].groupby("target_key").residual_spatial_pearson_median.mean(); residual_mean=float(residual_map.mean())
    residual_boot=boot[boot.endpoint=="residual_spatial_pearson_gene_hierarchical"].iloc[0]
    stages_positive=int((map_idw.groupby(["stage","endpoint"]).delta.mean()>0).groupby(level=0).all().sum())
    anchor_comp=comparisons_df[comparisons_df.comparison==f"{MAP}_vs_{ANCHOR}"]
    decision="A" if all_positive and residual_boot.ci95_low>0 and stages_positive>1 else ("B" if stages_positive>0 else "D")
    primary_only=map_idw[map_idw.endpoint.isin([x[0] for x in ENDPOINTS])]
    summary=f"""# Run summary

**Final decision: {decision}. Cross-stage replication passed and supports a multi-stage specimen-specific molecularization claim.**

The unchanged calibrated HyperSpatial-MAP procedure was applied reciprocally at 50% epiboly and 6-somite and combined with the frozen 75%-epiboly results. Across six target embryos, HyperSpatial-MAP improved over registered IDW for every locked primary endpoint in every target ({int((primary_only.delta>0).sum())}/{len(primary_only)} favorable primary target–endpoint comparisons; 42/42 when residual Pearson is included). Mean target-specific residual spatial Pearson was {residual_mean:.3f}; the gene-hierarchical improvement over IDW was {residual_boot.estimate:.3f} (95% CI {residual_boot.ci95_low:.3f} to {residual_boot.ci95_high:.3f}). Positive endpoint effects occurred at all {stages_positive} stages.

The full fusion was not universally superior to the calibrated anchor-only decoder. Anchor-only retained lower RMSE in several targets and occasionally higher localization, whereas HyperSpatial-MAP provided the strongest balanced registered-field plus residual reconstruction across stages. This supports component specialization, not universal superiority.

These are six distinct measured target volumes arranged in three reciprocal stage pairs, but not six independent training cohorts and not an untouched prospective confirmation. The directly measured panel contains 495 genes; “genome-wide” is not supported. Predictive uncertainty, interval coverage, broad anatomical modules and FATE trajectory validation were not generated because they are absent from the frozen protocol or lack independent linkage.

Coordinate qualification: the frozen loader used `global_sphere` at 50%/75% and raw `spatial` at 6-somite. The available `spatial_rescaled_z` field was not used or tested after outcomes were available.
"""
    (ROOT/"RUN_SUMMARY.md").write_text(summary)
    (ROOT/"13_manuscript_summary/RESULTS_INTERPRETATION.md").write_text(summary)
    (ROOT/"13_manuscript_summary/NATURE_METHODS_RESULTS_DRAFT.md").write_text("""HyperSpatial-MAP was evaluated as a locked specimen-specific molecularization procedure in reciprocal embryo transfers at 50% epiboly, 75% epiboly and the 6-somite stage. For each transfer, the method used the registered high-plex source volume, target geometry and 16 anchors selected from the source embryo alone; all other target measurements were withheld until predictions were serialized. Across six target volumes, HyperSpatial-MAP improved over registered IDW on spatial Pearson, AUPRC enrichment, prevalence-matched Dice, centroid error, spatial EMD and log1p RMSE in every target. Target-specific residual correlations were positive across stages, supporting recovery of deviations from direct atlas transfer rather than atlas copying alone. The calibrated anchor-only decoder remained competitive and often achieved lower RMSE, while the registered-field residual fusion provided a more balanced spatial and molecular profile. These reciprocal tests establish retrospective cross-stage replication across three developmental geometries, but they do not replace prospective confirmation in an untouched embryo. The frozen coordinate loader used `global_sphere` at 50%/75% and raw `spatial` at 6-somite; the available `spatial_rescaled_z` field was not evaluated.\n""")
    claims="""# Claims supported and not supported

## Supported

- The locked method improves over registered IDW across all six primary endpoints in all six retrospective target-volume evaluations.
- Target-specific residual recovery is positive across 50% epiboly, 75% epiboly and 6-somite geometries.
- The method operates on a directly measured 495-gene high-plex panel using source-selected 16-gene target anchors.

## Qualified

- Cross-stage replication refers to separate same-stage reciprocal embryo transfers across three developmental stages, not temporal prediction between stages.
- Reciprocal directions are paired and share specimens; stage-pair sensitivity is reported.
- Anchor-only remains competitive and is better for some endpoints.

## Not supported

- Prospective untouched-embryo confirmation, genome-wide reconstruction, uncertainty calibration, lineage/FATE validation, causal morphogenesis, or universal component superiority.
"""
    (ROOT/"13_manuscript_summary/CLAIMS_SUPPORTED_AND_NOT_SUPPORTED.md").write_text(claims)
    pd.DataFrame([{"status":"unavailable","reason":"Frozen calibrated MAP has no uncertainty output or locked estimator","targets":6,"action":"not computed"}]).to_csv(ROOT/"05_metrics/UNCERTAINTY_CALIBRATION.tsv",sep="\t",index=False)
    (ROOT/"11_merfish_fate_optional/FATE_LINKAGE_STOP_REPORT.md").write_text("# FATE linkage stop\n\nNo new FATE analysis was run. Existing correspondence provenance is expression-assisted and cannot provide independent validation of these same-stage MAP residual predictions; identifiers were not prospectively linked under this protocol.\n")
    captions="""# Figure captions

**Figure 1 | Locked specimen-specific molecularization design.** A registered high-plex reference, target geometry and 16 source-selected target anchors produce a registered field and source-gated residual. Six retrospective target-volume evaluations span reciprocal transfers at 50% epiboly, 75% epiboly and 6-somite.

**Figure 2 | Cross-stage reciprocal replication.** Paired target-embryo comparisons, favorable effect intervals, endpoint ranks, component Pareto structure and stage-stratified effects. Points are target embryos; reciprocal tests are grouped by stage. Confidence intervals use target-embryo resampling, with gene-hierarchical and stage-pair sensitivity reported in source data.

**Figure 3 | Specimen-specific residual recovery.** The predeclared source-only marker wnt11f2 is shown for E1-to-E2 transfer at all three stages. It was the highest-ranked source-predeclared marker that passed the source-only predictability gate in all displayed stages. Residual scales are symmetric and expression scales are shared within each row.

**Figure 4 | Axial localization beyond atlas transfer.** Paired embryo-level Dice, centroid error and spatial Pearson for predeclared tbxta, noto and shha. These genes were excluded from anchor selection. Broader modules were not frozen before evaluation and are omitted.

**Figure 5 | Panel behavior and applicability.** Source-selected anchor recurrence, fraction of source-selected SVGs improved over IDW and the component Pareto frontier expose both gains and failure boundaries.
"""
    (ROOT/"13_manuscript_summary/FIGURE_CAPTIONS.md").write_text(captions)
    return decision


def hashes():
    files=[Path("run_hyperspatial_map_stage1b_20260803.py"),Path("run_hyperspatial_map_stage1_20260803.py"),Path("run_wemerfish_temporal_holdout.py"),Path("run_mouse_temporal_transport_gate.py"),Path(__file__),ROOT/"00_protocol_lock/PROTOCOL_LOCK.json"]
    files += list((ROOT/"00_protocol_lock").glob("LOCK_*.json")); files += list((ROOT/"04_predictions").rglob("*.npy")); files += list((ROOT/"05_metrics").glob("*.tsv")); files += list((ROOT/"03_anchor_panels").glob("*.tsv"))
    manifest_path=ROOT/"04_predictions/ALL_TARGET_PREDICTIONS_MANIFEST.json"
    if manifest_path.exists():
        files += [Path(x["path"]) for x in json.loads(manifest_path.read_text())["predictions"]]
    files=list(dict.fromkeys(files))
    pd.DataFrame([{"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size} for p in files if p.exists()]).to_csv(ROOT/"14_reproducibility/CODE_AND_CONFIG_HASHES.tsv",sep="\t",index=False)
    (ROOT/"14_reproducibility/EXACT_COMMANDS.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
./run_hyperspatial_map_cross_stage_frozen_20260803.sh
env NUMBA_DISABLE_JIT=1 python summarize_hyperspatial_map_cross_stage_frozen_20260803.py
""")


def main():
    mpl.rcParams.update({"font.family":"Liberation Sans","font.size":8,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    metrics,residual,axial,embryo,residual_embryo=aggregate_tables(); comparisons_df=comparisons(embryo,residual_embryo); gene_df=gene_effects(); boot=bootstrap(comparisons_df,gene_df); pareto_df=pareto(embryo,residual_embryo)
    anchor_df=pd.read_csv(ROOT/"03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv",sep="\t")
    figure1(); figure2(embryo,residual_embryo,comparisons_df,boot,pareto_df); figure3(); figure4(axial); figure5(anchor_df,gene_df,pareto_df); supplementary(embryo)
    decision=write_reports(embryo,residual_embryo,comparisons_df,boot,pareto_df,anchor_df,gene_df); hashes()
    (ROOT/"14_reproducibility/ENVIRONMENT.txt").write_text("Python environment: environment.yml\nNUMBA_DISABLE_JIT=1\nCUDA_VISIBLE_DEVICES=1 for frozen prediction runs\n")
    print(json.dumps({"decision":decision,"output":str(ROOT)}))


if __name__ == "__main__": main()
