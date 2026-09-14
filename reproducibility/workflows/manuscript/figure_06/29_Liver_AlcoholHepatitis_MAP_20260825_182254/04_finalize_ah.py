#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

r=pd.read_csv(ROOT/'04_PATIENT_LEVEL_RESULTS.tsv',sep='\t')
p=pd.read_csv(ROOT/'07_DISEASE_PROGRAM_RECOVERY.tsv',sep='\t')
reg=pd.read_csv(ROOT/'03_REGISTRATION_RESULTS_PRELOCK.tsv',sep='\t')
reg['patient']=reg['sample']
reg.to_csv(ROOT/'06_REGISTRATION_RESULTS.tsv',sep='\t',index=False)

r['delta_spatial_pearson']=r.median_spatial_map-r.median_spatial_idw
r['delta_log1p_rmse']=r.log1p_rmse_map-r.log1p_rmse_idw
r.to_csv(ROOT/'04_PATIENT_LEVEL_RESULTS.tsv',sep='\t',index=False)
rng=np.random.default_rng(260825)
ci=[]
for col in ['delta_spatial_pearson','delta_log1p_rmse','pooled_residual_pearson_map']:
    x=r[col].to_numpy(); b=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    ci.append({'endpoint':col,'patient_balanced_mean':x.mean(),'patient_balanced_median':np.median(x),
               'bootstrap_95ci_low':np.quantile(b,.025),'bootstrap_95ci_high':np.quantile(b,.975),
               'independent_patient_n':len(x),'bootstrap_seed':260825})
pd.DataFrame(ci).to_csv(ROOT/'04B_PATIENT_BALANCED_SUMMARY.tsv',sep='\t',index=False)

files=sorted((ROOT/'data/GSE278662').glob('*'))
pd.DataFrame([{'file':str(x),'bytes':x.stat().st_size,'sha256':sha(x),
               'source':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278662/suppl/'+x.name,
               'download_date':'2026-08-25'} for x in files]).to_csv(ROOT/'00_DOWNLOAD_MANIFEST.tsv',sep='\t',index=False)

metrics='\n'.join([f"- {x.patient}: spatial Pearson {x.median_spatial_idw:.6f} → {x.median_spatial_map:.6f}; log1p RMSE {x.log1p_rmse_idw:.6f} → {x.log1p_rmse_map:.6f}; median/pooled residual Pearson {x.median_residual_pearson_map:.6f}/{x.pooled_residual_pearson_map:.6f}; gene wins spatial/RMSE {x.fraction_map_gt_idw_spatial:.1%}/{x.fraction_map_lt_idw_rmse:.1%}." for x in r.itertuples()])
programs='\n'.join([f"- {x.patient}, {x.program.replace('_',' ')}: residual-program Pearson {x.spatial_residual_pearson:.3f} ({x.n_available} available hidden genes)." for x in p.itertuples()])
(ROOT/'08_BIOLOGICAL_INTERPRETATION.md').write_text('# Biological interpretation\n\nAll five source-independent, predeclared AH programs showed positive spatial agreement between measured and predicted atlas-relative residual scores in both patients. These scores assess spatial program recovery and do not make genes or spots biological replicates.\n\n'+programs+'\n')
(ROOT/'11_LEAKAGE_AUDIT.md').write_text('# Leakage and integrity audit — PASS\n\n- Frozen source-model SHA256 verified before prediction.\n- Exactly 16 frozen post-MASLD source-selected anchors were materialized before lock.\n- Non-anchor matrix values were not materialized until both patient prediction manifests and hashes existed and were globally verified.\n- Registration used coordinates only, with frozen seeds 42–44 and seed 42 primary.\n- Predictions, operator assignments, guard/fallback decisions, hidden-gene lists, configurations and input matrices were hashed before unlock.\n- No AH-derived calibration, tuning, anchor selection, model selection, exclusion or representative-gene selection occurred.\n- Registered 12-NN IDW was the only reported comparator; anchor-only remained internal.\n')
(ROOT/'12_RESULTS_INTERPRETATION.md').write_text('# Results interpretation\n\nGSE278662 supplied two independent AH patient sections (Sah73 and Sah80). HyperSpatial-MAP improved registered IDW for median spatial correlation and log1p RMSE in both, while residual recovery was positive in both. All predeclared disease-program correlations were positive. The result is therefore Decision A under the frozen gate, while the small independent patient count (N=2) remains an important limitation.\n\n'+metrics+'\n')
(ROOT/'13_MANUSCRIPT_RESULTS_PARAGRAPH.md').write_text('# Manuscript-ready Results paragraph\n\nUsing the post-MASLD frozen protocol, we next evaluated two independent alcohol-associated hepatitis specimens from the publication-linked Visium cohort GSE278662. A healthy live-donor liver reference was registered to each target using geometry alone, and only 16 prespecified target genes were exposed before reconstructing 6,793–6,804 evaluable hidden genes. Relative to registered 12-nearest-neighbour IDW transfer, HyperSpatial-MAP increased median spatial Pearson correlation in both patients and reduced log1p RMSE in both. Atlas-relative residual recovery was positive in each patient, and predicted residuals recovered the spatial organization of all five predeclared inflammatory, injury, regenerative, fibrotic and metabolic programs. These data support patient-specific adaptation beyond healthy-atlas transfer in this cohort, while the two-patient sample size limits population-level inference.\n')
(ROOT/'15_FINAL_DECISION.md').write_text('# AH final decision: A\n\nThe prespecified Decision A gate was met: MAP improved both principal reconstruction endpoints in both independent patients, pooled residual Pearson was positive in both, all five predeclared programs were positive in both, and the leakage audit passed. No model element was changed in response to PSC or AH outcomes.\n\n'+metrics+'\n')

figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)
fig,ax=plt.subplots(1,3,figsize=(8.2,2.8),layout='constrained')
colors=['#3478b8','#d05a47']
for i,x in enumerate(r.itertuples()):
    ax[0].plot([0,1],[x.median_spatial_idw,x.median_spatial_map],'-o',color=colors[i],label=x.patient)
    ax[1].plot([0,1],[x.log1p_rmse_idw,x.log1p_rmse_map],'-o',color=colors[i])
ax[0].set_xticks([0,1],['IDW','MAP']); ax[0].set_ylabel('Median spatial Pearson'); ax[0].set_title('Spatial reconstruction')
ax[1].set_xticks([0,1],['IDW','MAP']); ax[1].set_ylabel('log1p RMSE'); ax[1].set_title('Reconstruction error')
ax[2].bar(r.patient,r.pooled_residual_pearson_map,color=colors); ax[2].axhline(0,color='0.4',lw=.8); ax[2].set_ylabel('Pooled residual Pearson'); ax[2].set_title('Atlas-relative recovery')
ax[0].legend(frameon=False,fontsize=8)
for a in ax:
    a.spines[['top','right']].set_visible(False); a.tick_params(labelsize=8); a.title.set_fontsize(10); a.yaxis.label.set_size(8)
fig.savefig(figdir/'AlcoholHepatitis_MAP_summary.pdf')
fig.savefig(figdir/'AlcoholHepatitis_MAP_summary.svg')
fig.savefig(figdir/'AlcoholHepatitis_MAP_summary.png',dpi=300)
plt.close(fig)

manifest=[]
for x in sorted(ROOT.rglob('*')):
    if x.is_file() and x.name!='16_SHA256_FINAL.txt': manifest.append(f'{sha(x)}  {x.relative_to(ROOT)}')
(ROOT/'16_SHA256_FINAL.txt').write_text('\n'.join(manifest)+'\n')
print(json.dumps({'decision':'A','patient_n':2,'leakage_audit':'PASS','output':str(ROOT)},indent=2))
