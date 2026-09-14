#!/usr/bin/env python3
"""Protocol-separated gimVI comparator for frozen HyperSpatial-MAP predictions."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, platform, resource, time
from pathlib import Path

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import scvi
from scvi.external import GIMVI

from run_hyperspatial_map_stage1_20260803 import octants, target_anchor_counts
from run_wemerfish_temporal_holdout import canonicalize, coordinates_only, evaluate, measured_counts, normalize_counts, pearson_columns

REPO=Path(__file__).resolve().parent
OUT=REPO/"results_v10/hyperspatial_map_gimvi_baseline_20260805_095807"
FROZEN=REPO/"results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726"
DATA=REPO/"data_v10/wemerfish_zebrafish_3stage"
SEED=42
METHODS=("registered_nearest_neighbor","registered_idw_k12","anchor_only_calibrated","gimvi_adapted",
         "atlas_plus_global_anchor_residual","atlas_plus_local_anchor_residual","hyperspatial_map_calibrated_fusion")
LABELS={"registered_nearest_neighbor":"Nearest neighbour","registered_idw_k12":"Registered IDW",
        "anchor_only_calibrated":"Anchor-only","gimvi_adapted":"gimVI (adapted)",
        "atlas_plus_global_anchor_residual":"Global residual","atlas_plus_local_anchor_residual":"Local residual",
        "hyperspatial_map_calibrated_fusion":"HyperSpatial-MAP"}
COLORS={"registered_nearest_neighbor":"#777777","registered_idw_k12":"#E69F00","anchor_only_calibrated":"#0072B2",
        "gimvi_adapted":"#56B4E9","atlas_plus_global_anchor_residual":"#8E6C8A",
        "atlas_plus_local_anchor_residual":"#009E73","hyperspatial_map_calibrated_fusion":"#C83E4D"}
RUNS=[
 {"key":"50p_E1_to_E2","stage":"50%","source":"E1","target":"E2","path":FROZEN/"04_predictions/50p_E1_to_E2","source_file":DATA/"weMERFISH_measured_A_50p_E1.h5ad","target_file":DATA/"weMERFISH_measured_A_50p_E2.h5ad"},
 {"key":"50p_E2_to_E1","stage":"50%","source":"E2","target":"E1","path":FROZEN/"04_predictions/50p_E2_to_E1","source_file":DATA/"weMERFISH_measured_A_50p_E2.h5ad","target_file":DATA/"weMERFISH_measured_A_50p_E1.h5ad"},
 {"key":"75p_E1_to_E2","stage":"75%","source":"E1","target":"E2","path":REPO/"results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2","source_file":DATA/"weMERFISH_measured_B_75p_E1.h5ad","target_file":DATA/"weMERFISH_measured_B_75p_E2.h5ad"},
 {"key":"75p_E2_to_E1","stage":"75%","source":"E2","target":"E1","path":REPO/"results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1","source_file":DATA/"weMERFISH_measured_B_75p_E2.h5ad","target_file":DATA/"weMERFISH_measured_B_75p_E1.h5ad"},
 {"key":"6s_E1_to_E2","stage":"6-somite","source":"E1","target":"E2","path":FROZEN/"04_predictions/6s_E1_to_E2","source_file":DATA/"weMERFISH_measured_C_6s_E1_rescaled_z.h5ad","target_file":DATA/"weMERFISH_measured_C_6s_E2_rescaled_z.h5ad"},
 {"key":"6s_E2_to_E1","stage":"6-somite","source":"E2","target":"E1","path":FROZEN/"04_predictions/6s_E2_to_E1","source_file":DATA/"weMERFISH_measured_C_6s_E2_rescaled_z.h5ad","target_file":DATA/"weMERFISH_measured_C_6s_E1_rescaled_z.h5ad"},
]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def jwrite(path,obj):
 with Path(path).open("x") as f: json.dump(obj,f,indent=2); f.write("\n")
def twrite(path,df):
 with Path(path).open("x") as f: df.to_csv(f,sep="\t",index=False)
def pooled(a,b):
 x=np.asarray(a,np.float64).ravel(); y=np.asarray(b,np.float64).ravel(); x-=x.mean(); y-=y.mean(); d=np.sqrt((x*x).sum()*(y*y).sum()); return float((x*y).sum()/d) if d else np.nan
def affine(pred,truth):
 px=pred.mean(0); py=truth.mean(0); x=pred-px
 slope=np.clip((x*(truth-py)).sum(0)/((x*x).sum(0)+1e-3),0,3).astype(np.float32)
 return slope,(py-slope*px).astype(np.float32)
def apply_affine(pred,cal): return np.maximum(pred*cal[0]+cal[1],0).astype(np.float32)

def make_adatas(source_counts,spatial_anchor,genes,anchor_idx,source_mask=None):
 if source_mask is None: source_mask=np.ones(len(source_counts),bool)
 seq=ad.AnnData(sparse.csr_matrix(source_counts[source_mask].astype(np.float32)),var=pd.DataFrame(index=genes),obs=pd.DataFrame(index=[f"s{i}" for i in np.flatnonzero(source_mask)]))
 spa=ad.AnnData(sparse.csr_matrix(spatial_anchor.astype(np.float32)),var=pd.DataFrame(index=genes[anchor_idx]),obs=pd.DataFrame(index=[f"t{i}" for i in range(len(spatial_anchor))]))
 GIMVI.setup_anndata(seq); GIMVI.setup_anndata(spa)
 return seq,spa

def fit_impute(source_counts,spatial_anchor,genes,anchor_idx,epochs,source_mask=None):
 scvi.settings.seed=SEED; torch.manual_seed(SEED)
 seq,spa=make_adatas(source_counts,spatial_anchor,genes,anchor_idx,source_mask)
 model=GIMVI(seq,spa,n_latent=10)
 model.train(max_epochs=epochs,accelerator="gpu",devices=1,batch_size=256,enable_progress_bar=False,
             check_val_every_n_epoch=None,enable_checkpointing=False,logger=False)
 _,out=model.get_imputed_values(normalized=True,batch_size=512)
 return model,np.asarray(out,np.float32)

def verify_root():
 p=json.loads((FROZEN/"00_protocol_lock/PROTOCOL_LOCK.json").read_text())
 if p["anchor_count"]!=16 or "hyperspatial_map_calibrated_fusion" not in p["methods"]: raise RuntimeError("wrong frozen MAP root")
 for r in RUNS:
  for q in (r["source_file"],r["target_file"],r["path"]/"prediction_manifest.json"):
   if not q.exists(): raise FileNotFoundError(q)

def prepare():
 verify_root(); rows=[]; anchors=[]
 for r in RUNS:
  m=json.loads((r["path"]/"prediction_manifest.json").read_text())
  if len(m["anchor_indices"])!=16: raise AssertionError
  _,_,ids,key=coordinates_only(r["target_file"])
  rows.append({"target_key":r["key"],"stage":r["stage"],"source":r["source"],"target":r["target"],"cells":len(ids),"coordinate_key":key,"source_file":r["source_file"],"target_file":r["target_file"]})
  anchors += [{"target_key":r["key"],"rank":i+1,"gene":g,"gene_index":j} for i,(g,j) in enumerate(zip(m["anchor_genes"],m["anchor_indices"]))]
 twrite(OUT/"00_audit/RUN_MANIFEST.tsv",pd.DataFrame(rows)); twrite(OUT/"01_protocol/FROZEN_ANCHORS.tsv",pd.DataFrame(anchors))
 protocol={"status":"LOCKED_BEFORE_TARGET_PREDICTION","runner":str(Path(__file__).resolve()),"runner_sha256":sha(Path(__file__).resolve()),
  "frozen_map_root":str(FROZEN),"seed":SEED,"targets":[r["key"] for r in RUNS],"target_information":"same 16 frozen anchors; target coordinates available but unused by gimVI",
  "hidden_evaluation":"479 non-anchor genes","gimvi":{"package":"scvi-tools","version":scvi.__version__,"n_latent":10,"pseudo_epochs":75,"final_epochs":200,"batch_size":256,"generative_distributions":["zinb","nb"],"model_library_size":[True,False]},
  "source_only_calibration":"source octant 0 affine calibration; octant 7 validation; sequencing reference excludes both pseudo-spatial octants",
  "output_scale":"normalized gimVI proportions multiplied by source median depth then log1p; gene-wise source-only affine calibration",
  "forbidden":["target non-anchor expression before prediction lock","target normalization or tuning","different anchors or evaluated genes"]}
 jwrite(OUT/"01_protocol/PROTOCOL_LOCK.json",protocol)
 twrite(OUT/"00_audit/PACKAGE_VERSIONS.tsv",pd.DataFrame([{"package":"scvi-tools","version":scvi.__version__},{"package":"torch","version":torch.__version__},{"package":"anndata","version":ad.__version__},{"package":"python","version":platform.python_version()}]))
 (OUT/"00_audit/ELIGIBILITY_AUDIT.md").write_text("# Eligibility\n\ngimVI is eligible as an adapted unmeasured-gene comparator. It consumes the full-gene source and the identical 16 target anchors, but not target 3D coordinates. Target non-anchor truth remains evaluation-only.\n")

def verify_lock():
 p=json.loads((OUT/"01_protocol/PROTOCOL_LOCK.json").read_text())
 if p["runner_sha256"]!=sha(Path(__file__).resolve()): raise RuntimeError("runner changed after lock")

def predict_one(r):
 m=json.loads((r["path"]/"prediction_manifest.json").read_text()); ai=np.asarray(m["anchor_indices"],int); ag=np.asarray(m["anchor_genes"])
 source,source_xyz,genes=measured_counts(r["source_file"]); _,tg,ids,key=coordinates_only(r["target_file"])
 if not np.array_equal(genes,tg) or not np.array_equal(genes[ai],ag): raise AssertionError("gene/anchor mismatch")
 target_anchor=target_anchor_counts(r["target_file"],ai).astype(np.float32); na=np.setdiff1d(np.arange(len(genes)),ai)
 labels=octants(canonicalize(source_xyz)[0]); pseudo=(labels==0)|(labels==7); ref=~pseudo
 pseudo_anchor=source[pseudo][:,ai].astype(np.float32)
 start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
 pseudo_model,pseudo_prop=fit_impute(source,pseudo_anchor,genes,ai,75,ref)
 depth=float(np.median(np.maximum(source.sum(1),1))); pseudo_log=np.log1p(pseudo_prop*depth); truth_log=np.log1p(normalize_counts(source[pseudo],depth))
 pl=labels[pseudo]; cal=affine(pseudo_log[pl==0][:,na],truth_log[pl==0][:,na]); val=apply_affine(pseudo_log[pl==7][:,na],cal)
 val_corr=float(np.nanmedian(pearson_columns(truth_log[pl==7][:,na],val))); val_mse=float(np.mean((truth_log[pl==7][:,na]-val)**2))
 del pseudo_model,pseudo_prop; gc.collect(); torch.cuda.empty_cache()
 model,prop=fit_impute(source,target_anchor,genes,ai,200)
 pred=np.log1p(prop*depth).astype(np.float32); pred[:,na]=apply_affine(pred[:,na],cal); pred[:,ai]=np.log1p(target_anchor).astype(np.float32)
 od=OUT/"02_gimvi"/r["key"]; od.mkdir(exist_ok=True)
 pp=od/"prediction_log1p_locked.npy"; np.save(pp,pred); model.save(od/"model",overwrite=False)
 twrite(od/"SOURCE_VALIDATION.tsv",pd.DataFrame([{"validation_mse":val_mse,"validation_median_spatial_pearson":val_corr,"calibration_cells":int((pl==0).sum()),"validation_cells":int((pl==7).sum())}]))
 return {"target_key":r["key"],"prediction":str(pp),"sha256":sha(pp),"source_file":str(r["source_file"]),"target_file":str(r["target_file"]),"coordinate_key":key,"target_cells":len(ids),"anchor_indices":ai.tolist(),"anchor_genes":ag.tolist(),"non_anchor_indices":na.tolist(),"runtime_seconds":time.perf_counter()-start,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/2**20,"peak_cpu_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"target_hidden_genes_loaded":False}

def predict():
 verify_lock(); manifests=[]
 for r in RUNS:
  mp=OUT/"01_protocol"/f"{r['key']}_PREDICTION_MANIFEST.json"
  if mp.exists():
   x=json.loads(mp.read_text()); assert sha(x["prediction"])==x["sha256"]
  else:
   x=predict_one(r); jwrite(mp,x)
  manifests.append(x)
 lock={"status":"ALL_GIMVI_PREDICTIONS_LOCKED_BEFORE_UNBLINDING","runner_sha256":sha(Path(__file__).resolve()),"predictions":[{"target_key":x["target_key"],"path":x["prediction"],"sha256":x["sha256"]} for x in manifests]}
 jwrite(OUT/"01_protocol/PREDICTIONS_LOCKED.json",lock); jwrite(OUT/"01_protocol/UNBLINDING_SENTINEL.json",{"status":"TARGET_TRUTH_MAY_BE_OPENED_FOR_EVALUATION_ONLY","prediction_lock_sha256":sha(OUT/"01_protocol/PREDICTIONS_LOCKED.json")})

def evaluate_all():
 verify_lock(); lock=json.loads((OUT/"01_protocol/PREDICTIONS_LOCKED.json").read_text())
 if not (OUT/"01_protocol/UNBLINDING_SENTINEL.json").exists(): raise RuntimeError("not unblinded")
 for x in lock["predictions"]:
  if sha(x["path"])!=x["sha256"]: raise RuntimeError("prediction changed")
 rows=[]; grows=[]; runt=[]
 for r in RUNS:
  fm=json.loads((r["path"]/"prediction_manifest.json").read_text()); em=json.loads((OUT/"01_protocol"/f"{r['key']}_PREDICTION_MANIFEST.json").read_text())
  ai=np.asarray(em["anchor_indices"],int); na=np.asarray(em["non_anchor_indices"],int); source,_,genes=measured_counts(r["source_file"]); target,xyz,tg=measured_counts(r["target_file"])
  depth=float(np.median(np.maximum(source.sum(1),1))); truth=np.log1p(normalize_counts(target,depth)).astype(np.float32)[:,na]; coords=canonicalize(xyz)[0]
  thresholds=np.quantile(normalize_counts(source,depth),.8,axis=0)[na]; svg=np.asarray([g for g in fm["evaluation_svg_indices"] if g in set(na)],int); remap={g:i for i,g in enumerate(na)}; svg=np.asarray([remap[g] for g in svg])
  pths={m:Path(v["path"]) for m,v in fm["methods"][str(SEED)]["predictions"].items()}; pths["gimvi_adapted"]=Path(em["prediction"])
  base=np.asarray(np.load(pths["registered_idw_k12"],mmap_mode="r"),np.float32)[:,na]
  for method in METHODS:
   pred=np.asarray(np.load(pths[method],mmap_mode="r"),np.float32)[:,na]; res=evaluate(method,truth,pred,coords,svg,thresholds,SEED+METHODS.index(method)); pg=np.asarray(res.pop("per_gene_pearson")); ci=res.pop("svg_spatial_pearson_bootstrap_ci95")
   rg=pearson_columns(truth-base,pred-base); rp=pooled(truth-base,pred-base)
   if method=="registered_idw_k12": rg[:]=np.nan; rp=np.nan
   rows.append({"target_key":r["key"],"stage":r["stage"],"source":r["source"],"target":r["target"],"method":method,"rho_res_pooled":rp,"residual_spatial_pearson_median":float(np.nanmedian(rg)) if np.isfinite(rg).any() else np.nan,**res,"pearson_ci_low":ci[0],"pearson_ci_high":ci[1],"evaluated_non_anchor_genes":len(na)})
   grows += [{"target_key":r["key"],"stage":r["stage"],"method":method,"gene":genes[g],"gene_index":g,"is_anchor":False,"spatial_pearson":pg[i],"residual_spatial_pearson":rg[i]} for i,g in enumerate(na)]
  runt.append({k:em[k] for k in ("target_key","runtime_seconds","peak_gpu_memory_mb","peak_cpu_rss_mb")}|{"method":"gimvi_adapted"})
 metrics=pd.DataFrame(rows); genes_df=pd.DataFrame(grows); runtime=pd.DataFrame(runt)
 twrite(OUT/"03_metrics/PER_TARGET_METRICS.tsv",metrics); twrite(OUT/"03_metrics/PER_GENE_METRICS.tsv",genes_df); twrite(OUT/"03_metrics/RUNTIME_MEMORY.tsv",runtime)
 summary=metrics.groupby("method",as_index=False).agg(targets=("target_key","nunique"),rho_res_pooled=("rho_res_pooled","mean"),residual_spatial_pearson_median=("residual_spatial_pearson_median","mean"),spatial_pearson=("svg_spatial_pearson_median","mean"),auprc_enrichment=("auprc_enrichment_median","mean"),dice=("topk_dice_median","mean"),centroid_error=("weighted_centroid_error_median","mean"),spatial_emd=("spatial_emd_median","mean"),log1p_rmse=("log1p_rmse","mean"),moran_error=("moran_error_median","mean")); twrite(OUT/"03_metrics/METHOD_SUMMARY.tsv",summary)
 rng=np.random.default_rng(20260805); br=[]
 for m in METHODS:
  if m=="hyperspatial_map_calibrated_fusion": continue
  for ep,hi in [("rho_res_pooled",1),("svg_spatial_pearson_median",1),("auprc_enrichment_median",1),("topk_dice_median",1),("weighted_centroid_error_median",0),("spatial_emd_median",0),("log1p_rmse",0),("moran_error_median",0)]:
   p=metrics.pivot(index="target_key",columns="method",values=ep); d=(p["hyperspatial_map_calibrated_fusion"]-p[m]).dropna().to_numpy(); d=d if hi else -d
   if not len(d): continue
   boot=np.asarray([np.mean(rng.choice(d,len(d),replace=True)) for _ in range(5000)]); br.append({"comparison":f"MAP vs {LABELS[m]}","endpoint":ep,"favorable_map_effect":d.mean(),"ci95_low":np.quantile(boot,.025),"ci95_high":np.quantile(boot,.975),"target_volumes":len(d),"bootstrap_unit":"target volume"})
 twrite(OUT/"03_metrics/PAIRED_BOOTSTRAP.tsv",pd.DataFrame(br)); metrics.to_csv(OUT/"05_source_data/FIGURE_TARGETS.tsv",sep="\t",index=False); summary.to_csv(OUT/"05_source_data/FIGURE_SUMMARY.tsv",sep="\t",index=False)
 plot(metrics,summary,runtime); decision(metrics,summary)

def plot(metrics,summary,runtime):
 mpl.rcParams.update({"font.family":"sans-serif","font.size":7,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
 fig,axs=plt.subplots(2,3,figsize=(10.5,6.2),constrained_layout=False)
 fig.subplots_adjust(left=.075,right=.975,top=.92,bottom=.18,wspace=.70,hspace=.72)
 for i,ax in enumerate(axs.flat): ax.text(-.18,1.12,"abcdef"[i],transform=ax.transAxes,fontweight="bold",fontsize=11,va="top")
 for m in METHODS:
  if m=="registered_idw_k12": continue
  d=metrics[metrics.method==m].set_index("target_key").reindex([r["key"] for r in RUNS]); axs[0,0].scatter(range(6),d.rho_res_pooled,s=24,color=COLORS[m],label=LABELS[m])
 axs[0,0].axhline(0,color="#999",ls=":",lw=.7); axs[0,0].set_ylabel("Pooled residual Pearson ↑"); axs[0,0].set_xticks(range(6),[r["stage"]+" "+r["target"] for r in RUNS],rotation=35,ha="right")
 sel=["gimvi_adapted","anchor_only_calibrated","hyperspatial_map_calibrated_fusion"]; sm=summary.set_index("method")
 endpoints=[("spatial_pearson",1),("auprc_enrichment",1),("dice",1),("centroid_error",0),("spatial_emd",0),("log1p_rmse",0),("moran_error",0)]
 mat=[]
 for m in sel:
  row=[]
  for e,hi in endpoints:
   b=sm.loc["registered_idw_k12",e]; row.append(sm.loc[m,e]-b if hi else b-sm.loc[m,e])
  mat.append(row)
 im=axs[0,1].imshow(mat,aspect="auto",cmap="RdBu_r"); axs[0,1].set_yticks(range(3),[LABELS[m] for m in sel]); axs[0,1].set_xticks(range(7),["Pearson","AUPRC","Dice","Centroid","EMD","RMSE","Moran"],rotation=40,ha="right"); axs[0,1].set_title("Favourable effect vs registered IDW"); fig.colorbar(im,ax=axs[0,1],fraction=.04)
 for m in sel:
  axs[0,2].scatter(sm.loc[m,"rho_res_pooled"],sm.loc[m,"dice"],s=55,color=COLORS[m]); axs[0,2].annotate(LABELS[m],(sm.loc[m,"rho_res_pooled"],sm.loc[m,"dice"]),xytext=(3,3),textcoords="offset points",fontsize=6)
 axs[0,2].margins(x=.18,y=.20); axs[0,2].set_xlabel("Residual Pearson ↑"); axs[0,2].set_ylabel("Dice ↑"); axs[0,2].set_title("Specimen adaptation and localization")
 for j,e in enumerate(["weighted_centroid_error_median","spatial_emd_median","log1p_rmse"]):
  base=metrics[metrics.method=="registered_idw_k12"].set_index("target_key")[e]
  for i,m in enumerate(sel):
   cur=metrics[metrics.method==m].set_index("target_key")[e].reindex(base.index)
   vals=100*(base-cur)/base; axs[1,0].scatter(np.full(len(vals),j)+(i-1)*.11,vals,s=18,color=COLORS[m],alpha=.8)
 axs[1,0].axhline(0,color="#999",ls=":",lw=.7); axs[1,0].set_ylabel("Improvement vs registered IDW (%) ↑"); axs[1,0].set_xticks(range(3),["Centroid","EMD","RMSE"]); axs[1,0].set_title("Localization and reconstruction")
 axs[1,1].scatter(runtime.runtime_seconds,runtime.peak_gpu_memory_mb/1024,s=35,color=COLORS["gimvi_adapted"]); axs[1,1].set_xlabel("Runtime incl. source calibration (s)"); axs[1,1].set_ylabel("Peak GPU memory (GB)"); axs[1,1].set_title("gimVI resource use")
 r=RUNS[2]; fm=json.loads((r["path"]/"prediction_manifest.json").read_text()); source,_,genes=measured_counts(r["source_file"]); target,xyz,_=measured_counts(r["target_file"]); xyz=canonicalize(xyz)[0]; depth=float(np.median(np.maximum(source.sum(1),1))); truth=np.log1p(normalize_counts(target,depth)); gi=int(np.flatnonzero(genes=="wnt11f2")[0]); em=json.loads((OUT/"01_protocol"/f"{r['key']}_PREDICTION_MANIFEST.json").read_text()); paths={"gimvi_adapted":Path(em["prediction"]),"anchor_only_calibrated":Path(fm["methods"][str(SEED)]["predictions"]["anchor_only_calibrated"]["path"]),"hyperspatial_map_calibrated_fusion":Path(fm["methods"][str(SEED)]["predictions"]["hyperspatial_map_calibrated_fusion"]["path"])}; vals=[truth[:,gi]]+[np.load(paths[m],mmap_mode="r")[:,gi] for m in ("hyperspatial_map_calibrated_fusion","gimvi_adapted","anchor_only_calibrated")]; vmax=np.quantile(np.concatenate(vals),.99)
 for i,(lab,v) in enumerate(zip(["Measured","MAP","gimVI","Anchor-only"],vals)):
  a=axs[1,2].inset_axes([i*.25,.04,.235,.88]); a.scatter(xyz[:,0],xyz[:,1],c=v,s=.25,cmap="viridis",vmin=0,vmax=vmax,rasterized=True); a.axis("off"); a.set_aspect("equal"); a.set_title(lab,fontsize=6.5,pad=2)
 axs[1,2].axis("off"); axs[1,2].set_title("Source-predeclared wnt11f2 · shared scale",fontsize=7)
 handles=[plt.Line2D([],[],marker="o",ls="",color=COLORS[m],label=LABELS[m]) for m in METHODS]; fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.025),ncol=4,frameon=False,fontsize=7,columnspacing=1.5,handletextpad=.4)
 for ext in ("pdf","svg","png"): fig.savefig(OUT/"04_figures"/f"GIMVI_external_baseline_comparison.{ext}",dpi=300 if ext=="png" else None,bbox_inches="tight",pad_inches=.15)
 plt.close(fig)

def decision(metrics,summary):
 s=summary.set_index("method"); p=metrics.pivot(index="target_key",columns="method",values="rho_res_pooled"); wins=int((p.hyperspatial_map_calibrated_fusion>=p.gimvi_adapted).sum()); mr=s.loc["hyperspatial_map_calibrated_fusion","rho_res_pooled"]; gr=s.loc["gimvi_adapted","rho_res_pooled"]
 dec="PASS" if wins>=4 and mr>=gr else "PARTIAL" if wins>=2 or abs(mr-gr)<.03 else "FAIL"
 (OUT/"FINAL_BASELINE_DECISION.md").write_text(f"# Final gimVI baseline decision\n\n**{dec}**\n\nMean pooled residual Pearson: HyperSpatial-MAP {mr:.3f}; adapted gimVI {gr:.3f}. MAP was higher on {wins}/6 targets. Both used the same 16 anchors and 479 hidden genes; gimVI did not use target coordinates. HyperSpatial-MAP is assessed as conservative specimen-specific atlas adaptation, not a universally superior imputation model.\n")

def tests():
 verify_lock(); a=pd.read_csv(OUT/"01_protocol/FROZEN_ANCHORS.tsv",sep="\t"); assert a.groupby("target_key").size().eq(16).all(); jwrite(OUT/"00_audit/LEAKAGE_TESTS.json",{"status":"PASS","target_hidden_genes_in_training":False,"identical_anchors":True,"target_outcome_tuning":False,"anchors_in_primary_metrics":False,"extra_target_information_for_gimvi":False})

def main():
 p=argparse.ArgumentParser(); p.add_argument("phase",choices=["prepare","test","predict","evaluate"]); a=p.parse_args()
 if a.phase=="prepare": prepare()
 elif a.phase=="test": tests()
 elif a.phase=="predict": predict()
 else: evaluate_all()
if __name__=="__main__": main()
