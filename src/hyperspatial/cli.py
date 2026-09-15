from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
from .config import AdaptConfig, DesignConfig, ForecastConfig
from .io import read_reference, read_target, read_truth
from .runio import load_adaptation
from .workflows import adapt, design_panel, explore, forecast, inspect_run, validate

def _yaml(path): return yaml.safe_load(Path(path).read_text()) if path else {}
def _section(raw,name):
    if name in raw:return dict(raw[name])
    if any(key in raw for key in ('design','adapt','forecast')):return {}
    return dict(raw)
def _panel(path):
    path=Path(path)
    if path.suffix.lower() in ('.tsv','.csv'):
        import pandas as pd
        table=pd.read_csv(path,sep='\t' if path.suffix.lower()=='.tsv' else ',')
        if 'gene' not in table.columns: raise ValueError("panel table must contain a gene column")
        return table['gene'].astype(str).tolist()
    return [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.lower().startswith('gene')]

def parser():
    p=argparse.ArgumentParser(prog='hyperspatial',description='HyperSpatial-MAP lab toolkit');p.add_argument('--version',action='version',version='0.1.0');s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('design');q.add_argument('--reference',required=True);q.add_argument('--budget',type=int);q.add_argument('--coordinates');q.add_argument('--layer');q.add_argument('--expression-is-log',action='store_true');q.add_argument('--out',required=True);q.add_argument('--config')
    q=s.add_parser('adapt');q.add_argument('--reference',required=True);q.add_argument('--target',required=True);q.add_argument('--panel',required=True);q.add_argument('--coordinates');q.add_argument('--layer');q.add_argument('--reference-expression-is-log',action='store_true');q.add_argument('--target-measurements-are-log',action='store_true');q.add_argument('--out',required=True);q.add_argument('--config')
    q=s.add_parser('explore');q.add_argument('--run',required=True);q.add_argument('--gene');q.add_argument('--out')
    q=s.add_parser('report');q.add_argument('--run',required=True);q.add_argument('--gene');q.add_argument('--out',required=True)
    q=s.add_parser('forecast');q.add_argument('--early',required=True);q.add_argument('--tracking',required=True);q.add_argument('--out',required=True);q.add_argument('--config')
    q=s.add_parser('validate');q.add_argument('--run',required=True);q.add_argument('--truth',required=True);q.add_argument('--layer');q.add_argument('--out',required=True)
    q=s.add_parser('inspect-run');q.add_argument('run')
    q=s.add_parser('serve');q.add_argument('--host',default='127.0.0.1');q.add_argument('--port',type=int,default=8000);q.add_argument('--runs-root',default='hyperspatial_runs')
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=='design':
        raw=_yaml(a.config);values=_section(raw,'design');values['budget']=a.budget if a.budget is not None else values.get('budget',16);cfg=DesignConfig(**values);r=design_panel(read_reference(a.reference,a.coordinates,a.layer,a.expression_is_log),config=cfg,out=a.out);print(json.dumps({'run_id':r.manifest.run_id,'genes':r.genes}))
    elif a.command=='adapt':
        genes=_panel(a.panel);raw=_yaml(a.config);values=_section(raw,'adapt');values.pop('panel',None);cfg=AdaptConfig(**values);ref=read_reference(a.reference,a.coordinates,a.layer,a.reference_expression_is_log);tar=read_target(a.target,genes,a.coordinates,a.layer,a.target_measurements_are_log);r=adapt(ref,tar,genes,a.out,cfg);print(json.dumps({'run_id':r.manifest.run_id,'supported':int(r.support.supported.sum())}))
    elif a.command in ('explore','report'):
        x=explore(load_adaptation(a.run),a.gene,a.out)
        if hasattr(x,'to_json'): print(x.to_json(orient='records'))
        else:
            value=x.to_dict() if hasattr(x,'to_dict') else x
            print(json.dumps({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in value.items()}))
    elif a.command=='forecast':
        raw=_yaml(a.config);r=forecast(load_adaptation(a.early),a.tracking,a.out,ForecastConfig(**_section(raw,'forecast')));print(json.dumps({'run_id':r.manifest.run_id}))
    elif a.command=='validate':
        r=load_adaptation(a.run);truth,_=read_truth(a.truth,r.genes,layer=a.layer);d=validate(r,truth,a.out);print(json.dumps({'genes':len(d)}))
    elif a.command=='inspect-run': print(json.dumps(inspect_run(a.run),indent=2))
    elif a.command=='serve':
        import os,uvicorn;os.environ['HYPERSPATIAL_RUNS_ROOT']=a.runs_root;uvicorn.run('hyperspatial.api:app',host=a.host,port=a.port,reload=False)

if __name__=='__main__': main()
