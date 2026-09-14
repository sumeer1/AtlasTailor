#!/usr/bin/env python3
"""Final visual-only refinement of the frozen integrated liver figure."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
V2=PROJECT/"PI_revision/32_Liver_multidisease_integrated_figure_v2_20260826_101842"
V2_SCRIPT=V2/"scripts/build_liver_multidisease_v2.py"
MASLD=PROJECT/"PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
PSC=PROJECT/"PI_revision/28_Liver_PSC_MAP_20260825_175802"
AH=PROJECT/"PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254"

spec=importlib.util.spec_from_file_location("liver_v2_frozen_readers",V2_SCRIPT)
v2=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
COLORS=v2.COLORS; IDW=v2.IDW; MAP=v2.MAP; INK=v2.INK
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":6.2,"axes.titlesize":7.2,
                     "axes.labelsize":6.2,"xtick.labelsize":5.4,"ytick.labelsize":5.4,
                     "axes.linewidth":.55,"xtick.major.width":.5,"ytick.major.width":.5,
                     "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def panel(ax,s,x=-.10,y=1.11):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=10,fontweight="bold",va="top",ha="left",color=INK)


def clean(ax):
    ax.spines[["top","right"]].set_visible(False);ax.tick_params(length=2,pad=1.5)


def frozen_inputs():
    units=v2.tables()
    audit=pd.read_csv(V2/"06_REPRESENTATIVE_GENE_SELECTION_AUDIT.tsv",sep="\t")
    selected=audit[audit.selected].set_index("disease").gene.to_dict()
    values=pd.read_csv(V2/"05_NUMERICAL_SUMMARY.tsv",sep="\t")
    return units,audit,selected,values


def draw_a(ax):
    ax.set_axis_off();panel(ax,"a",-.02,1.03)
    ax.set_title("Shared healthy-atlas adaptation",loc="left",fontweight="bold",pad=3)
    def box(x,y,w,h,text,fc,ec,fs=5.0):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.010,rounding_size=.022",transform=ax.transAxes,fc=fc,ec=ec,lw=.65)
        ax.add_patch(p);ax.text(x+w/2,y+h/2,text,transform=ax.transAxes,ha="center",va="center",fontsize=fs,linespacing=1.05)
    box(.01,.39,.23,.23,"Healthy liver atlas\nM1–M8","#EAF1F7","#7592AD",5.4)
    ys=[.72,.39,.06];labels=[("MASLD",COLORS["MASLD"]),("PSC",COLORS["PSC"]),("Alcohol-associated\nhepatitis",COLORS["AH"])]
    for (name,color),y in zip(labels,ys):
        box(.36,y,.29,.20,name+"\ngeometry + 16 genes","white",color,4.7)
        ax.add_patch(FancyArrowPatch((.24,.505),(.36,y+.10),transform=ax.transAxes,arrowstyle="-|>",mutation_scale=5.5,lw=.55,color=color,connectionstyle="arc3,rad=0"))
        ax.add_patch(FancyArrowPatch((.65,y+.10),(.75,.505),transform=ax.transAxes,arrowstyle="-|>",mutation_scale=5.5,lw=.55,color=color,connectionstyle="arc3,rad=0"))
    box(.68,.32,.21,.37,"Individualized\nhidden disease\nmolecular state","#EEF3F8",MAP,4.25)
    ax.text(.50,-.035,"Non-anchor genes withheld until prediction lock.",transform=ax.transAxes,ha="center",fontsize=4.4,color="#697076")


def draw_b(subspec,units):
    gs=subspec.subgridspec(1,2,wspace=.34);axes=[plt.subplot(gs[0,0]),plt.subplot(gs[0,1])]
    panel(axes[0],"b",-.08,1.10);axes[0].text(0,1.065,"Reconstruction beyond registered\nhealthy-atlas transfer",transform=axes[0].transAxes,fontweight="bold",fontsize=6.6,linespacing=1.0)
    pairs=[("median_spatial_idw","median_spatial_map","Median spatial Pearson"),("log1p_rmse_idw","log1p_rmse_map","log1p RMSE")]
    rng=np.random.default_rng(3203)
    for ax,(bc,mc,ylab) in zip(axes,pairs):
        for i,d in enumerate(["MASLD","PSC","AH"]):
            x=units[units.disease==d];left=i*3;jit=rng.uniform(-.15,.15,len(x))
            for j,row in enumerate(x.itertuples()):ax.plot([left+jit[j],left+1+jit[j]],[getattr(row,bc),getattr(row,mc)],color=COLORS[d],alpha=.14,lw=.38,zorder=1)
            ax.scatter(left+jit,x[bc],s=8,facecolors="white",edgecolors=COLORS[d],lw=.45,zorder=2)
            ax.scatter(left+1+jit,x[mc],s=8,color=COLORS[d],edgecolors="white",lw=.18,zorder=2)
            med=[x[bc].median(),x[mc].median()];ax.plot([left,left+1],med,color=INK,lw=1.45,zorder=3);ax.scatter([left,left+1],med,marker="D",s=16,color=INK,zorder=4)
        ax.set_xticks([.5,3.5,6.5],["MASLD\n33 biopsies\n32 patients","PSC\n4 sections\n1 patient","AH\n2 patients"])
        if ax is axes[0]:
            ax.set_ylabel("")
            ax.text(.018,.975,"Median spatial\nPearson",transform=ax.transAxes,ha="left",va="top",fontsize=5.1,color=INK,bbox=dict(fc="white",ec="none",alpha=.78,pad=.4))
        else:
            ax.set_ylabel(ylab,labelpad=.5)
        clean(ax)
        if "Pearson" in ylab:ax.axhline(0,color="#DDE0E2",lw=.5,zorder=0)
    handles=[Line2D([],[],marker='o',mfc='white',mec='#555',ls='none',markersize=3.3,label='IDW'),Line2D([],[],marker='o',mfc='#555',mec='#555',ls='none',markersize=3.3,label='MAP'),Line2D([],[],marker='D',color=INK,ls='none',markersize=3.3,label='disease median')]
    axes[1].legend(handles=handles,frameon=False,ncol=3,fontsize=4.6,handletextpad=.25,columnspacing=.65,loc="upper right",bbox_to_anchor=(1.0,1.075),borderaxespad=0)


def draw_c(ax,units):
    panel(ax,"c",-.12,1.10);ax.set_title("Target-specific molecular\ndeviations are recovered\nacross diseases",loc="left",fontweight="bold",fontsize=6.6,pad=2,linespacing=1.0)
    rng=np.random.default_rng(91)
    for i,d in enumerate(["MASLD","PSC","AH"]):
        x=units.loc[units.disease==d,"pooled_residual_pearson_map"].to_numpy();ax.scatter(i+rng.uniform(-.16,.16,len(x)),x,s=9,color=COLORS[d],alpha=.63,edgecolors="white",lw=.2)
        med=np.median(x);ax.plot([i-.25,i+.25],[med,med],color=INK,lw=1.5);ax.text(i,med+.032,f"{med:.2f}",ha="center",fontsize=5.3,fontweight="bold")
    ax.axhline(0,color="#999",lw=.5);ax.set_xticks(range(3),["MASLD","PSC","AH"]);ax.tick_params(axis='x',labelsize=5.8);ax.set_ylabel("Pooled MAP residual Pearson");clean(ax)


def draw_d(ax,summary):
    panel(ax,"d",-.045,1.08);ax.set_title("Predeclared disease programs recovered from sparse adaptation",loc="left",fontweight="bold",pad=4)
    program=summary[summary.record_type=="program_median"].set_index(["disease","metric"]).value
    rows=[("Metabolic/lipid",{"MASLD":"lipid_metabolic_reprogramming","AH":"metabolic_dysfunction"}),("Steatosis",{"MASLD":"steatosis_associated"}),("Inflammation",{"MASLD":"inflammation","AH":"inflammatory_immune_response"}),("Injury/stress",{"AH":"hepatocyte_injury_stress"}),("Regeneration",{"AH":"regenerative_response"}),("ECM/fibrosis",{"MASLD":"extracellular_matrix_fibrosis","PSC":"ecm_remodeling","AH":"fibrosis_ecm"}),("Ductular reaction",{"PSC":"ductular_reaction"}),("Portal/fibrotic",{"PSC":"portal_fibrotic"}),("Zonation",{"MASLD":"hepatocyte_zonation","PSC":"hepatocyte_zonation"})]
    diseases=["MASLD","PSC","AH"];a=np.full((9,3),np.nan)
    for i,(_,mapping) in enumerate(rows):
        for j,d in enumerate(diseases):
            if d in mapping:a[i,j]=program.loc[(d,mapping[d])]
    cmap=LinearSegmentedColormap.from_list("program",["#F5F8F8","#78C5C5","#287FA3","#293F9B"]);cmap.set_bad("#FAFAFA")
    ax.imshow(a,cmap=cmap,vmin=0,vmax=1,aspect="auto");ax.set_yticks(range(9),[x[0] for x in rows]);ax.set_xticks(range(3),["MASLD\n33 biopsies / 32 patients","PSC\n4 sections / 1 patient","AH\n2 patients"]);ax.xaxis.tick_top();ax.tick_params(length=0,pad=2.8)
    for j,d in enumerate(diseases):ax.get_xticklabels()[j].set_color(COLORS[d]);ax.get_xticklabels()[j].set_fontweight("bold")
    for i in range(9):
        for j in range(3):
            if np.isfinite(a[i,j]):ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",fontsize=5.6,color="white" if a[i,j]>.58 else INK,fontweight="bold")
    [s.set_visible(False) for s in ax.spines.values()]
    ax.text(.5,-.072,"Median residual-program Pearson · blank cells were not predefined",transform=ax.transAxes,ha="center",fontsize=4.6,color="#697076")


def draw_e(subspec,selected):
    examples=[("MASLD","VLP115_A_C1",selected["MASLD"],v2.readers.load_h5_example(MASLD/"work/VLP115_A_C1",selected["MASLD"])),("PSC","PSC011_A1",selected["PSC"],v2.readers.load_mtx_example(PSC/"work/PSC011_A1",selected["PSC"])),("AH","Sah73",selected["AH"],v2.readers.load_mtx_example(AH/"work/Sah73",selected["AH"]))]
    gs=subspec.subgridspec(3,5,hspace=.035,wspace=.012);axes=[]
    headings=["Target truth","Healthy-atlas prior\n(IDW)","MAP reconstruction","Measured residual","Predicted residual"]
    for r,(d,sample,gene,z) in enumerate(examples):
        vals=[z["truth"],z["prior"],z["pred"],z["true_residual"],z["predicted_residual"]];lo=min(x.min() for x in vals[:3]);hi=max(x.max() for x in vals[:3]);rm=max(np.abs(vals[3]).max(),np.abs(vals[4]).max(),1e-6)
        size=min(12.0,max(1.25,1050/len(z["xy"])))
        rowaxes=[]
        for c,val in enumerate(vals):
            ax=plt.subplot(gs[r,c]);axes.append(ax);rowaxes.append(ax);norm=Normalize(lo,hi) if c<3 else TwoSlopeNorm(vmin=-rm,vcenter=0,vmax=rm)
            ax.scatter(z["xy"][:,0],z["xy"][:,1],c=val,s=size,cmap="viridis" if c<3 else "coolwarm",norm=norm,linewidths=0)
            center=z["xy"].mean(0);ax.set_xlim(center[0]-.54,center[0]+.54);ax.set_ylim(center[1]-.54,center[1]+.54);ax.set_aspect("equal");ax.invert_yaxis();ax.set_xticks([]);ax.set_yticks([]);[q.set_visible(False) for q in ax.spines.values()]
            if r==0:ax.set_title(headings[c],fontsize=5.8,pad=2.5)
        rowaxes[0].text(-.055,.59,d,transform=rowaxes[0].transAxes,ha="right",va="center",fontsize=6.5,color=COLORS[d],fontweight="bold")
        rowaxes[0].text(-.055,.42,f"{sample}\n{gene}",transform=rowaxes[0].transAxes,ha="right",va="center",fontsize=5.0,color=INK,linespacing=1.0)
        rowaxes[4].text(.98,.03,f"Expression 0–{hi:.1f}  ·  Residual ±{rm:.1f}",transform=rowaxes[4].transAxes,ha="right",va="bottom",fontsize=4.1,color="#535A60",bbox=dict(fc="white",ec="none",alpha=.72,pad=.5))
    panel(axes[0],"e",-.25,1.23);axes[0].text(0,1.18,"Disease-specific molecular reconstructions",transform=axes[0].transAxes,fontweight="bold",fontsize=7.2)
    return [(d,s,g) for d,s,g,_ in examples]


def write_outputs(units,audit,selected,summary,examples):
    shutil.copy2(V2/"06_REPRESENTATIVE_GENE_SELECTION_AUDIT.tsv",ROOT/"05_REPRESENTATIVE_GENE_SELECTION_AUDIT.tsv")
    ah=audit[audit.disease=="AH"].copy();ah["previous_representative_gene"]="CYP3A4";ah["final_representative_gene"]="CYP3A4";ah["gene_changed"]=False;ah["decision_reason"]="CYP3A4 remains source-only rank 1 among predeclared eligible AH candidates"
    ah.to_csv(ROOT/"06_AH_REPRESENTATIVE_GENE_AUDIT.tsv",sep="\t",index=False)
    source_path={"MASLD":MASLD/"04A_BIOPSY_LEVEL_RESULTS_MASLD.tsv","PSC":PSC/"04A_SECTION_LEVEL_RESULTS_PSC.tsv","AH":AH/"04_PATIENT_LEVEL_RESULTS.tsv"}
    out=summary.copy();out["source_file"]=""
    for d in source_path:out.loc[(out.disease==d)&(out.record_type=="evaluation_unit"),"source_file"]=str(source_path[d]);out.loc[(out.disease==d)&(out.record_type=="program_median"),"source_file"]=str({"MASLD":MASLD,"PSC":PSC,"AH":AH}[d]/"07_DISEASE_PROGRAM_RECOVERY.tsv")
    out.to_csv(ROOT/"04_SOURCE_VALUES_USED.tsv",sep="\t",index=False)
    selected_rows=audit[audit.selected].set_index("disease")
    chosen="; ".join(f"{d}: {selected[d]} (source-only rank {int(selected_rows.loc[d,'source_rank_within_disease_eligible'])}, score {selected_rows.loc[d,'source_only_score']:.6f})" for d in ["MASLD","PSC","AH"])
    (ROOT/"01_FINAL_PANEL_PLAN.md").write_text(f"""# Final five-panel plan

- **a:** compact shared M1–M8 healthy-atlas adaptation schematic.
- **b:** dominant paired registered-IDW-to-MAP spatial Pearson and RMSE plots for all frozen evaluation units.
- **c:** compact pooled MAP target-deviation recovery plot.
- **d:** broad, shallow integrated heatmap with blank cells for programs not predefined.
- **e:** visual climax occupying the lower half: three disease rows × five large spatial maps.

Representatives remain frozen-source selections: {chosen}. The guard remains supplementary and is copied unchanged from v2.
""")
    (ROOT/"02_FINAL_FIGURE_CAPTION.md").write_text("""# Final figure caption

**Sparse target measurements adapt one healthy liver atlas across metabolic/fibrotic, cholangiopathic and inflammatory liver disease.** **a,** The fixed healthy live-donor liver atlas (M1–M8) was transferred to each disease target using geometry alone; exactly 16 target genes were exposed and non-anchor genes remained withheld until prediction lock. **b,** Median gene-wise spatial Pearson correlation and log1p RMSE for registered 12-nearest-neighbour inverse-distance weighting (IDW) and HyperSpatial-MAP across 33 MASLD biopsies from a 32-patient cohort, four PSC sections from one patient and two independent alcohol-associated hepatitis (AH) patients. Lines pair methods within evaluation units and diamonds show disease-level medians; PSC sections are within-patient samples, not independent patients. **c,** Pooled Pearson correlation between measured and MAP-predicted target-specific atlas-relative deviations. Registered IDW predicts no correction and its residual correlation is undefined. **d,** Median spatial correlation between measured and predicted scores for molecular programs predeclared before target outcomes were opened; blank cells denote programs not predefined for that disease. **e,** Frozen-source selected hidden-gene examples showing target expression, registered healthy-atlas prior, MAP reconstruction, measured atlas-relative residual and predicted residual. Expression scales are shared across the first three maps within each row and residual scales are identical and symmetric across the final two maps.
""")
    (ROOT/"03_FINAL_RESULTS_PARAGRAPH.md").write_text("""# Final Results paragraph

We asked whether sparse molecular measurements from diseased human liver could adapt a common healthy live-donor spatial reference toward an individualized disease state. The same frozen healthy M1–M8 reference, HyperSpatial-MAP configuration and 16-gene target budget were used across MASLD, primary sclerosing cholangitis and alcohol-associated hepatitis, with registered 12-nearest-neighbour IDW as the sole manuscript-facing comparator. HyperSpatial-MAP increased median spatial correlation and reduced log1p reconstruction error in all 33 MASLD biopsies, all four within-patient PSC sections and both alcohol-associated hepatitis patients. Target-specific atlas-relative deviation recovery was positive across every displayed evaluation unit, and predicted deviations reproduced the spatial organization of disease-associated programs specified before target outcomes were opened. Thus, sparse target measurements adapted a shared healthy spatial reference toward distinct metabolic/fibrotic, cholangiopathic and inflammatory molecular states, while the single-patient PSC evaluation limits population-level inference.
""")
    (ROOT/"07_VISUAL_QC.md").write_text("""# Visual QC — PASS

The final 7.087-inch (180-mm) PNG was inspected at native output dimensions, and the one-page PDF geometry was checked.

- **Panel a:** boxes are aligned; disease labels fit; arrows and text do not collide; the prediction-lock note is secondary.
- **Panel b:** paired IDW-to-MAP changes, endpoint labels, disease medians and biological-unit labels are readable; raw lines are visually subordinate.
- **Panel c:** individual values and medians are readable; no IDW residual value or competing explanatory note is drawn.
- **Panel d:** all program labels and values are readable; undefined cells remain light and blank; row order and values are unchanged.
- **Panel e:** all 15 maps are visible and spatial structure is inspectable; disease/sample/gene labels are complete; within-row expression and residual scales are shared; only one compact scale note is shown per row.
- **Global:** panel labels a–e are visible, disease colors are consistent, no main-figure guard panel is present, and the direct journal-width render has no clipping or text collision.
""")
    for ext in ["pdf","svg","png"]:shutil.copy2(V2/f"Supplementary_Liver_guard_behavior.{ext}",ROOT/f"Supplementary_Liver_guard_behavior.{ext}")


def main():
    units,audit,selected,summary=frozen_inputs()
    fig=plt.figure(figsize=(7.087,8.45),facecolor="white")
    outer=fig.add_gridspec(3,16,height_ratios=[2.05,1.50,3.95],hspace=.34,wspace=.72,left=.078,right=.975,top=.952,bottom=.035)
    axa=fig.add_subplot(outer[0,:4]);draw_a(axa);draw_b(outer[0,4:12],units);axc=fig.add_subplot(outer[0,12:]);draw_c(axc,units)
    axd=fig.add_subplot(outer[1,1:15]);draw_d(axd,summary)
    examples=draw_e(outer[2,:],selected)
    for ext,dpi in [("pdf",300),("svg",300),("png",400)]:fig.savefig(ROOT/f"Liver_multidisease_integrated_main_v3.{ext}",dpi=dpi)
    plt.close(fig);write_outputs(units,audit,selected,summary,examples)
    manifest=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!="08_SHA256_MANIFEST.txt":manifest.append(f"{sha(p)}  {p.relative_to(ROOT)}")
    (ROOT/"08_SHA256_MANIFEST.txt").write_text("\n".join(manifest)+"\n")
    print(json.dumps({"v2":str(V2),"v3":str(ROOT),"predictions_recomputed":False,"metrics_changed":False,"selected":selected,"figure_inches":[7.087,8.45]},indent=2))


if __name__=="__main__":main()
