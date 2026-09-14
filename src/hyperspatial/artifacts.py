from __future__ import annotations
import hashlib, json, os, platform, shlex, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def new_run(workflow: str, out: Path, config: dict, inputs: list[Path]):
    from .models import RunManifest
    out=out.resolve(); out.mkdir(parents=True,exist_ok=False)
    run_id=f"{workflow}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    hashes=[]
    for p in inputs:
        if p and p.exists(): hashes.append({"path":str(p.resolve()),"sha256":sha256(p),"bytes":p.stat().st_size})
    pd.DataFrame(hashes,columns=['path','sha256','bytes']).to_csv(out/'input_hashes.tsv',sep='\t',index=False)
    (out/'run_config.yaml').write_text(yaml.safe_dump(config,sort_keys=True))
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit=None
    data={"schema_version":"1.0","run_id":run_id,"workflow":workflow,"status":"running",
          "created_utc":datetime.now(timezone.utc).isoformat(),"config":config,"inputs":hashes,
          "scientific_core":"frozen HyperSpatial-MAP/MAP-T adapters","tool_version":"0.1.0","git_commit":commit}
    (out/'manifest.json').write_text(json.dumps(data,indent=2)+"\n")
    (out/'environment.txt').write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\npandas={pd.__version__}\n")
    (out/'run.log').write_text(f"{data['created_utc']} created {workflow} run {run_id}\n")
    return RunManifest(run_id,workflow,"running",out,data)

def finish(manifest, artifacts: list[Path], extra: dict|None=None):
    manifest.status="completed"; manifest.data["status"]="completed"
    with (manifest.run_dir/'run.log').open('a') as log: log.write(f"{datetime.now(timezone.utc).isoformat()} completed\n")
    artifacts=list(artifacts)+[manifest.run_dir/'run.log']
    manifest.data["artifacts"]=[{"path":p.name,"sha256":sha256(p),"bytes":p.stat().st_size} for p in artifacts if p.exists()]
    if extra: manifest.data.update(extra)
    (manifest.run_dir/'manifest.json').write_text(json.dumps(manifest.data,indent=2)+"\n")
    paths=[x['path'] for x in manifest.data.get('inputs',[])]
    if manifest.workflow=='design' and len(paths)>=1: cmd=f"hyperspatial design --reference {shlex.quote(paths[0])} --config run_config.yaml --out reproduced_run"
    elif manifest.workflow=='adapt' and len(paths)>=2: cmd=f"hyperspatial adapt --reference {shlex.quote(paths[0])} --target {shlex.quote(paths[1])} --panel panel.tsv --config run_config.yaml --out reproduced_run"
    elif manifest.workflow=='forecast' and len(paths)>1: cmd=f"hyperspatial forecast --early {shlex.quote(str(Path(paths[1]).parent))} --tracking {shlex.quote(paths[0])} --config run_config.yaml --out reproduced_run"
    else: cmd=f"# inspect run_config.yaml and manifest.json to reproduce {manifest.workflow}"
    (manifest.run_dir/'reproduce.sh').write_text(f"#!/bin/sh\nset -eu\n{cmd}\n")
    os.chmod(manifest.run_dir/'reproduce.sh',0o755)
    return manifest

def _adata(values, result, layer_name):
    import anndata as ad
    a=ad.AnnData(X=np.asarray(values,np.float32)); a.var_names=np.asarray(result.genes).astype(str); a.obsm['spatial']=result.coordinates
    a.uns['hyperspatial']={"layer":layer_name,"run_id":result.manifest.run_id if result.manifest else None}
    return a

def export_panel(result,out):
    out.mkdir(parents=True,exist_ok=True)
    files=[]
    for name,df in [('recommended_panel.tsv',result.ordered_panel),('marker_utility.tsv',result.marker_utility),('conditional_utility.tsv',result.conditional_utility),('budget_curve.tsv',result.budget_curve),('coverage.tsv',result.coverage)]:
        p=out/name;df.to_csv(p,sep='\t',index=False);files.append(p)
    (out/'recommended_panel.json').write_text(json.dumps(result.genes,indent=2)+"\n");files.append(out/'recommended_panel.json')
    import anndata as ad
    coverage=result.coverage.set_index('gene')
    a=ad.AnnData(X=np.asarray([coverage.max_squared_correlation_to_panel],np.float32))
    a.var_names=coverage.index.astype(str);a.obs_names=['source_only_panel_design']
    a.var['selected']=a.var_names.isin(result.genes)
    a.var['panel_position']=[result.genes.index(g)+1 if g in result.genes else 0 for g in a.var_names]
    a.uns['hyperspatial']={'workflow':'design','source_only':True,'post_hoc_experimental_design':True}
    p=out/'panel_design.h5ad';a.write_h5ad(p);files.append(p)
    p=out/'report.html';p.write_text("<!doctype html><html><head><meta charset='utf-8'><title>Source-only panel design</title></head><body><h1>Source-only experimental panel design</h1><p>This panel is source/stage specific and is not a universally optimal gene set.</p><h2>Recommended genes</h2>"+result.ordered_panel.to_html(index=False)+"<h2>Expected source-CV recoverability</h2>"+result.budget_curve.to_html(index=False)+"</body></html>");files.append(p)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.2));fig.patch.set_facecolor('white')
    axes[0].plot(result.budget_curve.budget,result.budget_curve.expected_source_cv_residual_pearson,'o-',color='#226a88');axes[0].set(xlabel='Measurement budget',ylabel='Expected source-CV residual Pearson',title='Recoverability by budget')
    u=result.marker_utility.sort_values('loao_utility');axes[1].barh(u.gene,u.loao_utility,color='#367b66');axes[1].set(xlabel='LOAO utility, Δr',title='Panel utility')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        p=out/f'panel_design_summary.{ext}';fig.savefig(p,dpi=300,bbox_inches='tight');files.append(p)
    plt.close(fig)
    return files

def export_adaptation(result,out):
    out.mkdir(parents=True,exist_ok=True); files=[]
    for name,val in [('adapted.h5ad',result.adapted),('registered_atlas.h5ad',result.registered_atlas),('predicted_residual.h5ad',result.predicted_residual)]:
        p=out/name;_adata(val,result,name).write_h5ad(p);files.append(p)
    for name in ('gene_support.tsv','predictor_assignments.tsv'):
        p=out/name;result.support.to_csv(p,sep='\t',index=False);files.append(p)
    p=out/'panel.tsv';pd.DataFrame({'position':range(1,len(result.panel)+1),'gene':result.panel}).to_csv(p,sep='\t',index=False);files.append(p)
    p=out/'registered_coordinates.npy';np.save(p,result.registered_source_coordinates);files.append(p)
    import matplotlib.pyplot as plt
    counts=result.support.selected_predictor.value_counts();fig,axes=plt.subplots(1,2,figsize=(8,3.2));fig.patch.set_facecolor('white')
    axes[0].bar(counts.index,counts.values,color='#70538d');axes[0].tick_params(axis='x',rotation=35);axes[0].set(ylabel='Genes',title='Source-only predictor assignments')
    axes[1].hist(np.mean(np.abs(result.predicted_residual),axis=0),bins=24,color='#367b66');axes[1].set(xlabel='Mean absolute predicted residual',ylabel='Genes',title='Correction magnitude')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        p=out/f'adaptation_summary.{ext}';fig.savefig(p,dpi=300,bbox_inches='tight');files.append(p)
    plt.close(fig)
    return files

def export_forecast(result,out):
    import anndata as ad
    out.mkdir(parents=True,exist_ok=True);files=[]
    for name,val in [('transported_early.h5ad',result.transported_early),('temporal_residual.h5ad',result.temporal_residual),('forecast.h5ad',result.forecast)]:
        a=ad.AnnData(X=np.asarray(val,np.float32));a.var_names=result.genes.astype(str);a.obsm['spatial']=result.coordinates;p=out/name;a.write_h5ad(p);files.append(p)
    p=out/'trajectory_summary.tsv';result.trajectory_summary.to_csv(p,sep='\t',index=False);files.append(p)
    import matplotlib.pyplot as plt
    counts=result.support.selected_predictor.value_counts();fig,axes=plt.subplots(1,2,figsize=(8,3.2));fig.patch.set_facecolor('white')
    axes[0].bar(counts.index,counts.values,color='#70538d');axes[0].tick_params(axis='x',rotation=25);axes[0].set(ylabel='Genes',title='Temporal guard assignments')
    axes[1].hist(np.mean(np.abs(result.temporal_residual),axis=0),bins=24,color='#367b66');axes[1].set(xlabel='Mean absolute temporal residual',ylabel='Genes',title='Temporal correction magnitude')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        p=out/f'forecast_summary.{ext}';fig.savefig(p,dpi=300,bbox_inches='tight');files.append(p)
    plt.close(fig)
    return files

def write_report(result,path):
    title="HyperSpatial-MAP run report"; supported=int(result.support.supported.sum())
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body><h1>{title}</h1><p>{len(result.genes)} genes; {supported} source-supported corrections.</p><p>Source support is a predictability guard, not uncertainty.</p></body></html>")
    return path
