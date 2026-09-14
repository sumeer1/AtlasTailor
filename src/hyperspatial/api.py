from __future__ import annotations
import json,os,traceback,uuid
from importlib.resources import files
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .config import AdaptConfig,DesignConfig
from .io import read_reference,read_target,read_truth
from .runio import load_adaptation
from .workflows import adapt,design_panel,explore,forecast,inspect_run,validate

ROOT=Path(os.environ.get('HYPERSPATIAL_RUNS_ROOT','hyperspatial_runs')).resolve();ROOT.mkdir(parents=True,exist_ok=True)
try: WEB=Path(str(files('web')))
except ModuleNotFoundError: WEB=Path('__web_client_not_installed__')
pool=ThreadPoolExecutor(max_workers=2);jobs={};guard=Lock()
app=FastAPI(title='HyperSpatial-MAP API',version='1.0.0',description='Source-validated spatial atlas adaptation')

class DesignRequest(BaseModel):
    reference: str; budget: int=Field(16,ge=1,le=256); coordinate_key: Optional[str]=None; layer: Optional[str]=None; expression_is_log:bool=False; exclusions:list[str]=Field(default_factory=list); candidates:list[str]=Field(default_factory=list)
class AdaptRequest(BaseModel):
    reference:str;target:str;measured_genes:list[str];coordinate_key:Optional[str]=None;layer:Optional[str]=None;reference_expression_is_log:bool=False;target_measurements_are_log:bool=False;registration:str='geometry'
class ForecastRequest(BaseModel): early_run:str;tracking_map:str
class ValidateRequest(BaseModel): run_id:str;truth:str;layer:Optional[str]=None

def safe_run(run_id):
    p=(ROOT/run_id).resolve()
    if ROOT not in p.parents: raise HTTPException(400,'invalid run id')
    return p
def submit(kind,fn):
    jid=f'{kind}-{uuid.uuid4().hex[:10]}';jobs[jid]={'run_id':jid,'workflow':kind,'status':'queued','progress':0,'logs':[]}
    def go():
        try:
            jobs[jid]|={'status':'running','progress':5};fn(jid,lambda p,m='':jobs[jid].update(progress=p,logs=jobs[jid]['logs']+[m] if m else jobs[jid]['logs']));jobs[jid]|={'status':'completed','progress':100}
        except Exception as e: jobs[jid]|={'status':'failed','progress':100,'error':str(e),'logs':jobs[jid]['logs']+[traceback.format_exc()]}
    pool.submit(go);return jobs[jid]

@app.post('/api/v1/design',status_code=202)
def post_design(q:DesignRequest):
    def task(j,progress): progress(10,'Reading source AnnData');ref=read_reference(q.reference,q.coordinate_key,q.layer,q.expression_is_log);progress(30,'Running source-only design');design_panel(ref,config=DesignConfig(budget=q.budget,exclusions=tuple(q.exclusions),candidates=tuple(q.candidates)),out=safe_run(j))
    return submit('design',task)
@app.post('/api/v1/adapt',status_code=202)
def post_adapt(q:AdaptRequest):
    def task(j,progress): progress(10,'Reading reference and restricted target panel');ref=read_reference(q.reference,q.coordinate_key,q.layer,q.reference_expression_is_log);tar=read_target(q.target,q.measured_genes,q.coordinate_key,q.layer,q.target_measurements_are_log);progress(25,'Fitting source-only operators');adapt(ref,tar,q.measured_genes,safe_run(j),AdaptConfig(registration=q.registration))
    return submit('adapt',task)
@app.post('/api/v1/forecast',status_code=202)
def post_forecast(q:ForecastRequest):
    return submit('forecast',lambda j,p:forecast(load_adaptation(safe_run(q.early_run)),q.tracking_map,safe_run(j)))
@app.post('/api/v1/validate',status_code=202)
def post_validate(q:ValidateRequest):
    def task(j,progress):
        r=load_adaptation(safe_run(q.run_id));truth,_=read_truth(q.truth,r.genes,layer=q.layer);out=safe_run(q.run_id)/'metrics.tsv';validate(r,truth,out);jobs[j]['artifact']=str(out)
    return submit('validate',task)
@app.get('/api/v1/runs')
def get_runs():
    disk=[]
    for p in ROOT.iterdir():
        if (p/'manifest.json').exists():
            try: disk.append(inspect_run(p))
            except Exception: pass
    active=[x for x in jobs.values() if x['status']!='completed' or not (safe_run(x['run_id'])/'manifest.json').exists()]
    return sorted(active+disk,key=lambda x:x.get('created_utc',''),reverse=True)
@app.get('/api/v1/runs/{run_id}')
def get_run(run_id:str):
    if run_id in jobs and (jobs[run_id]['status']!='completed' or jobs[run_id]['workflow']=='validate'): return jobs[run_id]
    p=safe_run(run_id)
    if not (p/'manifest.json').exists(): raise HTTPException(404,'run not found')
    return inspect_run(p)
@app.get('/api/v1/runs/{run_id}/manifest')
def get_manifest(run_id:str): return get_run(run_id)
@app.get('/api/v1/runs/{run_id}/artifacts')
def get_artifacts(run_id:str):
    p=safe_run(run_id);return [{'name':x.name,'bytes':x.stat().st_size,'url':f'/api/v1/runs/{run_id}/artifacts/{x.name}'} for x in p.iterdir() if x.is_file()]
@app.get('/api/v1/runs/{run_id}/artifacts/{name}')
def artifact(run_id:str,name:str):
    p=(safe_run(run_id)/name).resolve()
    if p.parent!=safe_run(run_id) or not p.exists():raise HTTPException(404,'artifact not found')
    return FileResponse(p)
@app.get('/api/v1/runs/{run_id}/genes')
def genes(run_id:str): return explore(load_adaptation(safe_run(run_id))).replace({np.nan:None}).to_dict('records')
@app.get('/api/v1/runs/{run_id}/design')
def design_result(run_id:str):
    p=safe_run(run_id)
    needed={'panel':'recommended_panel.tsv','utility':'marker_utility.tsv','redundancy':'conditional_utility.tsv','curve':'budget_curve.tsv','coverage':'coverage.tsv'}
    if not all((p/name).exists() for name in needed.values()):raise HTTPException(404,'design artifacts not found')
    return {key:pd.read_csv(p/name,sep='\t').replace({np.nan:None}).to_dict('records') for key,name in needed.items()}
@app.get('/api/v1/runs/{run_id}/genes/{gene}')
def gene(run_id:str,gene:str):
    try:d=load_adaptation(safe_run(run_id)).gene(gene).to_dict()
    except KeyError:raise HTTPException(404,'gene not found')
    # Browser payload is deterministically capped for large point clouds.
    n=len(d['adapted']);idx=np.linspace(0,n-1,min(n,5000),dtype=int);r=load_adaptation(safe_run(run_id))
    return {k:(v[idx].tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()}|{'coordinates':r.coordinates[idx].tolist()}
@app.get('/api/v1/health')
def health():return {'status':'ok','version':'1.0.0','runs_root':str(ROOT)}
@app.post('/api/v1/uploads/{filename}')
async def upload(filename:str,request:Request):
    """Stream one local-lab input into the configured runs root."""
    name=Path(filename).name
    if name!=filename or Path(name).suffix.lower() not in {'.h5ad','.h5','.hdf5','.npz','.npy','.tsv','.csv','.json'}:
        raise HTTPException(400,'unsupported or unsafe filename')
    directory=ROOT/'_uploads';directory.mkdir(exist_ok=True)
    path=directory/f'{uuid.uuid4().hex[:12]}-{name}';size=0
    with path.open('xb') as handle:
        async for chunk in request.stream():
            size+=len(chunk)
            if size>50*(1<<30):
                handle.close();path.unlink(missing_ok=True);raise HTTPException(413,'upload exceeds 50 GiB local limit')
            handle.write(chunk)
    return {'path':str(path),'bytes':size}
@app.get('/api/v1/demos')
def demos():
    path=Path(str(files('demos')/'catalog.json'))
    return json.loads(path.read_text()) if path.exists() else []

if WEB.exists():
    app.mount('/assets',StaticFiles(directory=WEB/'assets'),name='assets')
    @app.get('/')
    def index():return FileResponse(WEB/'index.html')
