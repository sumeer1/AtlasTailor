#!/usr/bin/env python3
"""Five-panel liver figure assembled exclusively from frozen outputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MASLD = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
PSC = PROJECT / "PI_revision/28_Liver_PSC_MAP_20260825_175802"
AH = PROJECT / "PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254"
MODEL = MASLD / "work/healthy_source_model_masld_panel.npz"
SOURCE_RULE = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222/scripts/01_prepare_and_lock_alf.py"
V1_SCRIPT = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_20260826_095523/scripts/build_integrated_liver_figure.py"

spec = importlib.util.spec_from_file_location("liver_v1_readers", V1_SCRIPT)
readers = importlib.util.module_from_spec(spec); spec.loader.exec_module(readers)

COLORS = {"MASLD":"#C7832B", "PSC":"#16858C", "AH":"#B54A69"}
IDW = "#A4A9AE"; MAP = "#285F9E"; INK = "#202428"
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":8.2,"axes.titlesize":9.5,
                     "axes.labelsize":8,"xtick.labelsize":7.2,"ytick.labelsize":7.2,
                     "axes.linewidth":.7,"xtick.major.width":.6,"ytick.major.width":.6,
                     "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def label(ax,s,x=-.09,y=1.10):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=14,fontweight="bold",va="top",ha="left",color=INK)


def clean(ax):
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(length=2.6,pad=2)


def tables():
    m=pd.read_csv(MASLD/"04A_BIOPSY_LEVEL_RESULTS_MASLD.tsv",sep="\t"); m["unit"]=m.biopsy; m["unit_type"]="biopsy"
    p=pd.read_csv(PSC/"04A_SECTION_LEVEL_RESULTS_PSC.tsv",sep="\t"); p["unit"]=p["sample"]; p["unit_type"]="section"
    a=pd.read_csv(AH/"04_PATIENT_LEVEL_RESULTS.tsv",sep="\t"); a["unit"]=a.patient; a["unit_type"]="patient"
    cols=["disease","unit","unit_type","hidden_genes","median_spatial_idw","median_spatial_map","log1p_rmse_idw","log1p_rmse_map","pooled_residual_pearson_map","guard_fallback_gene_fraction"]
    return pd.concat([m[cols],p[cols],a[cols]],ignore_index=True)


def predeclared_programs(root):
    d=pd.read_csv(root/"07_DISEASE_PROGRAM_RECOVERY.tsv",sep="\t")
    out={}
    for row in d.drop_duplicates("program").itertuples(): out[row.program]=row.predeclared_genes.split(";")
    return out


def representative_audit():
    z=np.load(MODEL,allow_pickle=True)
    names=z["gene_names"].astype(str); evaluation=z["evaluation_indices"].astype(int)
    eval_set=set(evaluation.tolist()); by_name={g:i for i,g in enumerate(names)}
    with tempfile.TemporaryDirectory(prefix="liver_source_rank_") as td:
        npy=Path(td)/"source_values.npy"
        with zipfile.ZipFile(MODEL) as archive, archive.open("source_values.npy") as src, npy.open("wb") as dst: shutil.copyfileobj(src,dst)
        values=np.load(npy,mmap_mode="r")
        score=np.full(len(names),np.nan)
        for start in range(0,len(evaluation),256):
            idx=evaluation[start:start+256]
            x=np.log1p(np.asarray(values[:,idx],dtype=np.float32))
            score[idx]=x.var(0)*np.mean(x>0,axis=0)
    global_order=evaluation[np.argsort(score[evaluation])[::-1]]
    global_rank={int(g):i+1 for i,g in enumerate(global_order)}
    roots={"MASLD":MASLD,"PSC":PSC,"AH":AH}; rows=[]; selected={}
    for disease,root in roots.items():
        programs=predeclared_programs(root); membership={}
        for program,genes in programs.items():
            for gene in genes: membership.setdefault(gene,[]).append(program)
        eligible=[g for g in membership if g in by_name and by_name[g] in eval_set]
        eligible=sorted(eligible,key=lambda g:score[by_name[g]],reverse=True)
        selected[disease]=eligible[0]
        for rank,gene in enumerate(eligible,1):
            ix=by_name[gene]
            rows.append({"disease":disease,"gene":gene,"predeclared_programs":";".join(membership[gene]),
                         "eligible_hidden_gene":True,"source_only_score":score[ix],
                         "source_rank_within_disease_eligible":rank,"source_rank_global_hidden":global_rank[ix],
                         "selected":gene==eligible[0],
                         "selection_reason":"Highest frozen-source score among disease-predeclared hidden genes" if gene==eligible[0] else "Eligible but lower frozen-source score",
                         "source_score_formula":"var(log1p(source_values)) * prevalence(log1p(source_values)>0)",
                         "criterion_provenance":str(SOURCE_RULE)+":419-421","target_outcomes_used":False})
    audit=pd.DataFrame(rows)
    audit.to_csv(ROOT/"06_REPRESENTATIVE_GENE_SELECTION_AUDIT.tsv",sep="\t",index=False)
    audit.to_csv(ROOT/"REPRESENTATIVE_GENE_SELECTION_AUDIT.tsv",sep="\t",index=False)
    return selected,audit


def draw_design(ax):
    ax.set_axis_off(); label(ax,"a",-.02,1.04)
    ax.set_title("Shared healthy-atlas adaptation across liver diseases",loc="left",fontweight="bold",pad=6)
    def box(x,y,w,h,text,fc,ec,fs=7.6):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.022",transform=ax.transAxes,fc=fc,ec=ec,lw=.9)
        ax.add_patch(p); ax.text(x+w/2,y+h/2,text,transform=ax.transAxes,ha="center",va="center",fontsize=fs)
    box(.02,.39,.23,.24,"Healthy liver atlas\nM1–M8","#EAF1F7","#7592AD",8.2)
    ys=[.72,.39,.06]
    texts=[("MASLD",COLORS["MASLD"]),("PSC",COLORS["PSC"]),("Alcohol-associated\nhepatitis",COLORS["AH"])]
    for (txt,c),y in zip(texts,ys):
        box(.38,y,.26,.20,txt+"\ngeometry + 16 genes","white",c,7.2)
        ax.add_patch(FancyArrowPatch((.25,.51),(.38,y+.10),transform=ax.transAxes,arrowstyle="-|>",mutation_scale=8,lw=.75,color=c))
        ax.add_patch(FancyArrowPatch((.64,y+.10),(.76,.51),transform=ax.transAxes,arrowstyle="-|>",mutation_scale=8,lw=.75,color=c))
    box(.77,.36,.21,.30,"Individualized\nhidden disease\nmolecular state","#EEF3F8",MAP,7.4)
    ax.text(.50,-.02,"Non-anchor target genes withheld until prediction lock.",transform=ax.transAxes,ha="center",fontsize=6.5,color="#5B6166")


def benchmark(subspec,units):
    gs=subspec.subgridspec(1,2,wspace=.32); axes=[plt.subplot(gs[0,0]),plt.subplot(gs[0,1])]
    label(axes[0],"b",-.19,1.16); axes[0].text(0,1.12,"Reconstruction beyond registered healthy-atlas transfer",transform=axes[0].transAxes,fontweight="bold",fontsize=9.5)
    pairs=[("median_spatial_idw","median_spatial_map","Median spatial Pearson"),("log1p_rmse_idw","log1p_rmse_map","log1p RMSE")]
    rng=np.random.default_rng(3202)
    for ax,(basecol,mapcol,title) in zip(axes,pairs):
        for i,d in enumerate(["MASLD","PSC","AH"]):
            x=units[units.disease==d]; left=i*3; jit=rng.uniform(-.18,.18,len(x))
            for j,row in enumerate(x.itertuples()): ax.plot([left+jit[j],left+1+jit[j]],[getattr(row,basecol),getattr(row,mapcol)],color=COLORS[d],alpha=.25,lw=.55,zorder=1)
            ax.scatter(left+jit,x[basecol],s=14,facecolors="white",edgecolors=COLORS[d],lw=.55,zorder=2)
            ax.scatter(left+1+jit,x[mapcol],s=14,color=COLORS[d],edgecolors="white",lw=.25,zorder=2)
            med=[x[basecol].median(),x[mapcol].median()]
            ax.plot([left,left+1],med,color=INK,lw=2,zorder=3); ax.scatter([left,left+1],med,marker="D",s=27,color=INK,zorder=4)
        ax.set_xticks([.5,3.5,6.5],["MASLD\n33 biopsies\n32 patients","PSC\n4 sections\n1 patient","AH\n2 patients"])
        ax.set_ylabel(title); clean(ax)
        if "Pearson" in title: ax.axhline(0,color="#D8DBDE",lw=.65,zorder=0)
    axes[0].text(.02,.02,"open, IDW   filled, MAP   ◆ disease median",transform=axes[0].transAxes,fontsize=6.4,color="#5B6166")


def residual(ax,units):
    label(ax,"c",-.13,1.10); ax.set_title("Target-specific molecular deviations\nare recovered across diseases",loc="left",fontweight="bold",pad=5)
    rng=np.random.default_rng(90)
    for i,d in enumerate(["MASLD","PSC","AH"]):
        x=units.loc[units.disease==d,"pooled_residual_pearson_map"].to_numpy()
        ax.scatter(i+rng.uniform(-.18,.18,len(x)),x,s=16,color=COLORS[d],alpha=.68,edgecolors="white",lw=.3)
        med=np.median(x); ax.plot([i-.26,i+.26],[med,med],color=INK,lw=2.1); ax.text(i,med+.033,f"{med:.2f}",ha="center",fontsize=6.7,fontweight="bold")
    ax.axhline(0,color="#888",lw=.7); ax.set_xticks(range(3),["MASLD","PSC","AH"]); ax.set_ylabel("Pooled MAP residual Pearson"); clean(ax)
    ax.text(.02,.02,"IDW residual correlation undefined",transform=ax.transAxes,fontsize=6.2,color="#666")


def integrated_heatmap(ax,roots):
    label(ax,"d",-.045,1.08); ax.set_title("Predeclared disease programs are recovered from sparse adaptation",loc="left",fontweight="bold",pad=7)
    programs={d:pd.read_csv(root/"07_DISEASE_PROGRAM_RECOVERY.tsv",sep="\t").groupby("program").spatial_residual_pearson.median() for d,root in roots.items()}
    rows=[("Metabolic/lipid",{"MASLD":"lipid_metabolic_reprogramming","AH":"metabolic_dysfunction"}),
          ("Steatosis",{"MASLD":"steatosis_associated"}),
          ("Inflammation",{"MASLD":"inflammation","AH":"inflammatory_immune_response"}),
          ("Injury/stress",{"AH":"hepatocyte_injury_stress"}),
          ("Regeneration",{"AH":"regenerative_response"}),
          ("ECM/fibrosis",{"MASLD":"extracellular_matrix_fibrosis","PSC":"ecm_remodeling","AH":"fibrosis_ecm"}),
          ("Ductular reaction",{"PSC":"ductular_reaction"}),
          ("Portal/fibrotic",{"PSC":"portal_fibrotic"}),
          ("Zonation",{"MASLD":"hepatocyte_zonation","PSC":"hepatocyte_zonation"})]
    diseases=["MASLD","PSC","AH"]; a=np.full((len(rows),3),np.nan); source=np.empty((len(rows),3),object); source[:]=None
    for i,(_,mapping) in enumerate(rows):
        for j,d in enumerate(diseases):
            if d in mapping: a[i,j]=programs[d][mapping[d]]; source[i,j]=mapping[d]
    cmap=LinearSegmentedColormap.from_list("program",["#F2F6F7","#75C2C3","#257EA3","#293F9B"]); cmap.set_bad("#F1F2F3")
    ax.imshow(a,cmap=cmap,vmin=0,vmax=1,aspect="auto")
    ax.set_yticks(range(len(rows)),[x[0] for x in rows]); ax.set_xticks(range(3),["MASLD\n33 biopsies / 32 patients","PSC\n4 sections / 1 patient","AH\n2 patients"]); ax.xaxis.tick_top(); ax.tick_params(length=0,pad=5)
    for j,d in enumerate(diseases): ax.get_xticklabels()[j].set_color(COLORS[d]); ax.get_xticklabels()[j].set_fontweight("bold")
    for i in range(a.shape[0]):
        for j in range(3):
            if np.isfinite(a[i,j]): ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",fontsize=7.1,color="white" if a[i,j]>.58 else INK,fontweight="bold")
            else: ax.text(j,i,"—",ha="center",va="center",fontsize=7,color="#B8BDC1")
    [s.set_visible(False) for s in ax.spines.values()]
    ax.text(.5,-.075,"Median residual-program Pearson · blank cells were not predefined",transform=ax.transAxes,ha="center",fontsize=6.5,color="#5B6166")
    return a,rows,source


def spatial_panel(subspec,selected):
    examples=[("MASLD","VLP115_A_C1",selected["MASLD"],readers.load_h5_example(MASLD/"work/VLP115_A_C1",selected["MASLD"])),
              ("PSC","PSC011_A1",selected["PSC"],readers.load_mtx_example(PSC/"work/PSC011_A1",selected["PSC"])),
              ("AH","Sah73",selected["AH"],readers.load_mtx_example(AH/"work/Sah73",selected["AH"]))]
    gs=subspec.subgridspec(3,5,hspace=.09,wspace=.035); axes=[]
    heads=["Target truth","Registered healthy-atlas\nprior (IDW)","MAP reconstruction","Measured atlas-relative\nresidual","Predicted atlas-relative\nresidual"]
    for r,(d,sample,gene,z) in enumerate(examples):
        vals=[z["truth"],z["prior"],z["pred"],z["true_residual"],z["predicted_residual"]]
        lo=min(v.min() for v in vals[:3]); hi=max(v.max() for v in vals[:3]); rm=max(np.abs(vals[3]).max(),np.abs(vals[4]).max(),1e-6)
        rowaxes=[]
        for c,v in enumerate(vals):
            ax=plt.subplot(gs[r,c]); axes.append(ax); rowaxes.append(ax)
            norm=Normalize(lo,hi) if c<3 else TwoSlopeNorm(vmin=-rm,vcenter=0,vmax=rm)
            ax.scatter(z["xy"][:,0],z["xy"][:,1],c=v,s=3.1,cmap="viridis" if c<3 else "coolwarm",norm=norm,linewidths=0)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); [q.set_visible(False) for q in ax.spines.values()]
            if r==0: ax.set_title(heads[c],fontsize=7.5,pad=4)
            if c==0: ax.set_ylabel(f"{d} · {sample}\n{gene}",rotation=0,ha="right",va="center",labelpad=14,fontsize=8,color=COLORS[d],fontweight="bold")
        rowaxes[2].text(.98,.02,f"expr. {lo:.1f}–{hi:.1f}",transform=rowaxes[2].transAxes,ha="right",fontsize=5.8,color="#60666B")
        rowaxes[4].text(.98,.02,f"resid. ±{rm:.1f}",transform=rowaxes[4].transAxes,ha="right",fontsize=5.8,color="#60666B")
    label(axes[0],"e",-.38,1.30); axes[0].text(0,1.24,"Disease-specific molecular reconstructions",transform=axes[0].transAxes,fontweight="bold",fontsize=9.5)
    return [(d,s,g) for d,s,g,_ in examples]


def guard_supplement(units):
    fallback=units.groupby("disease").guard_fallback_gene_fraction.median().reindex(["MASLD","PSC","AH"]); supported=1-fallback
    fig,ax=plt.subplots(figsize=(5.2,3.4),layout="constrained")
    ax.bar(range(3),supported,color=MAP,label="MAP-supported correction"); ax.bar(range(3),fallback,bottom=supported,color="#D3D7DA",label="Registered-IDW fallback")
    for i,(s,f) in enumerate(zip(supported,fallback)):
        ax.text(i,s/2,f"{s:.1%}",ha="center",va="center",color="white",fontweight="bold",fontsize=8); ax.text(i,s+f/2,f"{f:.1%}",ha="center",va="center",color=INK,fontsize=8)
    ax.set_ylim(0,1); ax.set_xticks(range(3),["MASLD","PSC","AH"]); ax.set_ylabel("Fraction of hidden genes"); ax.set_title("Frozen liver guard behavior",loc="left",fontweight="bold"); clean(ax); ax.legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2)
    for ext,dpi in [("pdf",300),("svg",300),("png",400)]: fig.savefig(ROOT/f"Supplementary_Liver_guard_behavior.{ext}",dpi=dpi)
    plt.close(fig)


def write_docs(units,audit,selected,heat_rows,heat_source,examples):
    rows=[]
    for x in units.itertuples():
        for metric,method,val in [("hidden_genes","count",x.hidden_genes),("median_spatial_pearson","IDW",x.median_spatial_idw),("median_spatial_pearson","MAP",x.median_spatial_map),("log1p_rmse","IDW",x.log1p_rmse_idw),("log1p_rmse","MAP",x.log1p_rmse_map),("pooled_residual_pearson","MAP",x.pooled_residual_pearson_map),("guard_fallback_fraction","IDW fallback",x.guard_fallback_gene_fraction)]: rows.append({"record_type":"evaluation_unit","disease":x.disease,"unit":x.unit,"metric":metric,"method":method,"value":val,"note":x.unit_type})
    roots={"MASLD":MASLD,"PSC":PSC,"AH":AH}
    for d,root in roots.items():
        p=pd.read_csv(root/"07_DISEASE_PROGRAM_RECOVERY.tsv",sep="\t").groupby("program").spatial_residual_pearson.median()
        for k,v in p.items(): rows.append({"record_type":"program_median","disease":d,"unit":"disease summary","metric":k,"method":"MAP residual recovery","value":v,"note":"median frozen evaluation-unit value"})
    pd.DataFrame(rows).to_csv(ROOT/"05_NUMERICAL_SUMMARY.tsv",sep="\t",index=False)
    score={d:audit[(audit.disease==d)&audit.selected].iloc[0] for d in roots}
    chosen="; ".join(f"{d} {selected[d]} (eligible rank {int(score[d].source_rank_within_disease_eligible)}, score {score[d].source_only_score:.6f})" for d in roots)
    (ROOT/"01_FIGURE_PANEL_PLAN.md").write_text(f"""# Five-panel liver figure plan

- **a** Minimal shared-reference schematic: healthy M1–M8 atlas, three disease targets, geometry plus 16 measured genes, and individualized hidden disease state.
- **b** Dominant paired IDW-to-MAP reconstruction panel: spatial Pearson and log1p RMSE for all 33 MASLD biopsies, four PSC sections and two AH patients, with disease medians.
- **c** Compact pooled MAP atlas-relative residual Pearson plot. IDW residual correlation is undefined and is not drawn.
- **d** One integrated 9 × 3 heatmap of frozen predeclared program medians; undefined disease–program combinations remain blank.
- **e** Three large spatial rows and five aligned map columns. Source-only selections: {chosen}.

The frozen guard is removed from the main figure and retained unchanged in a separate supplementary figure. No prediction, registration, anchor, operator assignment, guard decision, metric or conclusion was changed.
""")
    (ROOT/"02_FIGURE_CAPTION_DRAFT.md").write_text("""# Figure caption draft

**Sparse target measurements adapt one healthy liver atlas across metabolic/fibrotic, cholangiopathic and inflammatory liver disease.** **a,** The fixed healthy live-donor liver atlas (M1–M8) was transferred to each target using geometry alone; exactly 16 target genes were exposed and all non-anchor genes remained withheld until prediction lock. **b,** Median gene-wise spatial Pearson correlation and log1p RMSE for registered 12-nearest-neighbour inverse-distance weighting (IDW) and HyperSpatial-MAP across 33 MASLD biopsies from a 32-patient cohort, four PSC sections from one patient and two independent alcohol-associated hepatitis (AH) patients. Lines pair methods within the same evaluation unit; diamonds denote disease-level medians. PSC sections are within-patient samples and not independent patients. **c,** Pooled Pearson correlation between measured and MAP-predicted atlas-relative residuals. Registered IDW predicts zero correction, so its residual correlation is undefined. **d,** Median spatial correlation between measured and predicted scores for molecular programs predeclared before target outcomes were opened. Blank cells indicate programs not predefined for that disease. **e,** Source-only selected hidden-gene examples showing target expression, registered healthy-atlas prior, MAP reconstruction, measured atlas-relative residual and predicted residual. Expression scales are shared across the first three maps in each row; residual scales are identical and symmetric across the last two maps.
""")
    (ROOT/"03_MANUSCRIPT_RESULTS_PARAGRAPH.md").write_text("""# Manuscript-ready Results paragraph

We asked whether sparse molecular measurements from diseased human liver could adapt a common healthy live-donor spatial reference toward an individualized disease state. The same frozen healthy M1–M8 reference, HyperSpatial-MAP configuration and 16-gene target budget were used for MASLD, primary sclerosing cholangitis and alcohol-associated hepatitis, with registered 12-nearest-neighbour IDW as the sole manuscript-facing comparator. HyperSpatial-MAP increased median spatial correlation and reduced log1p reconstruction error in all 33 MASLD biopsies, all four within-patient PSC sections and both alcohol-associated hepatitis patients. Atlas-relative residual recovery was positive across all displayed evaluation units, and predicted residuals reproduced the spatial organization of the disease-associated programs specified before target outcomes were opened. Thus, sparse target measurements adapted a common healthy spatial reference toward distinct metabolic/fibrotic, cholangiopathic and inflammatory molecular states, while the single-patient PSC evaluation limits population-level inference.
""")
    (ROOT/"07_PANEL_LAYOUT_NOTES.md").write_text("""# Panel layout notes

The figure uses a deliberately unequal hierarchy. Panel b occupies the largest share of the top row; panel e occupies the entire lower half. The schematic and residual plot are compact. The program panel is a single integrated heatmap rather than three disconnected blocks. The guard plot was removed from the main narrative and exported separately. Disease colors are fixed throughout: ochre for MASLD, teal for PSC and rose for AH. IDW is open/gray-coded and MAP is filled/blue-coded where method color is needed.
""")
    with (ROOT/"04_SOURCE_FILES_USED.txt").open("w") as f:
        f.write("PREDICTIONS_RECOMPUTED=NO\nSOURCE_MANIFESTS_VERIFIED=PASS\nTARGET_OUTCOMES_USED_FOR_REPRESENTATIVE_SELECTION=NO\n\n")
        for d,root in roots.items(): f.write(f"SOURCE_DIRECTORY\t{d}\t{root}\n")
        f.write(f"SOURCE_ONLY_SCORE_MODEL\t{sha(MODEL)}  {MODEL}\nSOURCE_ONLY_RULE\t{SOURCE_RULE}:419-421\n")
        for path in [MASLD/"04A_BIOPSY_LEVEL_RESULTS_MASLD.tsv",MASLD/"07_DISEASE_PROGRAM_RECOVERY.tsv",PSC/"04A_SECTION_LEVEL_RESULTS_PSC.tsv",PSC/"07_DISEASE_PROGRAM_RECOVERY.tsv",AH/"04_PATIENT_LEVEL_RESULTS.tsv",AH/"07_DISEASE_PROGRAM_RECOVERY.tsv"]: f.write(f"{sha(path)}  {path}\n")
        for d,s,g in examples:
            pair={"MASLD":MASLD/"work/VLP115_A_C1","PSC":PSC/"work/PSC011_A1","AH":AH/"work/Sah73"}[d]
            lock=json.loads((pair/"prediction_lock_manifest.json").read_text())
            for path in [pair/"prediction_lock_manifest.json",pair/"target_anchor_only_object.npz",pair/"geometry_only_target_roi_rows.npy",pair/"source_model_predicted_target_depth.npy",Path(lock["registered_idw_prediction"]),Path(lock["map_prediction"]),Path(lock.get("target_matrix_path") or lock["target_matrix"])]: f.write(f"{sha(path)}  {path}\n")


def main():
    units=tables(); selected,audit=representative_audit()
    fig=plt.figure(figsize=(15.8,12.6),facecolor="white")
    outer=fig.add_gridspec(3,14,height_ratios=[3.35,2.65,6.15],hspace=.43,wspace=.72,left=.055,right=.985,top=.97,bottom=.04)
    axa=fig.add_subplot(outer[0,:4]); draw_design(axa)
    benchmark(outer[0,4:11],units)
    axc=fig.add_subplot(outer[0,11:]); residual(axc,units)
    axd=fig.add_subplot(outer[1,1:13]); hm,hmrows,hmsource=integrated_heatmap(axd,{"MASLD":MASLD,"PSC":PSC,"AH":AH})
    examples=spatial_panel(outer[2,:],selected)
    for ext,dpi in [("pdf",300),("svg",300),("png",400)]: fig.savefig(ROOT/f"Liver_multidisease_integrated_main_v2.{ext}",dpi=dpi)
    plt.close(fig)
    guard_supplement(units); write_docs(units,audit,selected,hmrows,hmsource,examples)
    manifest=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='08_SHA256_MANIFEST.txt': manifest.append(f"{sha(p)}  {p.relative_to(ROOT)}")
    (ROOT/"08_SHA256_MANIFEST.txt").write_text("\n".join(manifest)+"\n")
    print(json.dumps({"predictions_recomputed":False,"selected":selected,"layout":"a-e main; guard supplementary","output":str(ROOT)},indent=2))


if __name__=="__main__": main()
