#!/usr/bin/env python3
"""Evaluate locked CONCERT and MAP outputs on one identical stroke matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
MAP_ROOT = REPO / "PI_revision/17_CONCERT_real_stroke_MAP_20260820_100844"
GEOM_ROOT = REPO / "PI_revision/16_CONCERT_stroke_geometry_20260820_095543"
ORIGINAL = REPO / "PI_revision/15_CONCERT_3D_stroke_audit_20260820_092123/official_sources/Mouse_brain_stroke_all_data.h5"
LOCK = ROOT / "LOCKED_CONCERT_OUTPUT"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))


# Exact metric/preprocessing definitions used by the frozen stroke evaluator.
# Kept local to avoid importing unrelated model/Scanpy modules during this
# read-only post-lock evaluation.
def normalize_counts(counts: np.ndarray, depth_target: float) -> np.ndarray:
    depth = counts.sum(1)
    return counts * (depth_target / np.maximum(depth, 1.0))[:, None]


def pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc, yc = x - x.mean(0), y - y.mean(0)
    return np.sum(xc * yc, axis=0) / np.sqrt(
        np.maximum(np.sum(xc**2, axis=0) * np.sum(yc**2, axis=0), 1e-16)
    )


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, float).ravel(); y = np.asarray(b, float).ravel()
    use = np.isfinite(x) & np.isfinite(y)
    if use.sum() < 3 or np.std(x[use]) <= 1e-12 or np.std(y[use]) <= 1e-12:
        return np.nan
    return float(pearsonr(x[use], y[use]).statistic)


def moran_columns(values: np.ndarray, neighbors: np.ndarray, chunk: int = 32) -> np.ndarray:
    out = np.empty(values.shape[1], float)
    for start in range(0, values.shape[1], chunk):
        x = np.asarray(values[:, start:start + chunk], float)
        z = x - x.mean(0, keepdims=True)
        den = np.square(z).sum(0)
        num = (z * z[neighbors].mean(1)).sum(0)
        out[start:start + chunk] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-12)
    return out


def load_inputs():
    source = np.load(MAP_ROOT / "LOCKED_PREDICTIONS/SOURCE_MODEL.npz")
    genes = source["genes"].astype(str)
    hidden = source["hidden_indices"].astype(int)
    stroke = ad.read_h5ad(GEOM_ROOT / "STROKE_3D.h5ad")
    counts = stroke.X.toarray() if hasattr(stroke.X, "toarray") else np.asarray(stroke.X)
    counts = np.asarray(counts, np.float32)
    obs_names = stroke.obs_names.astype(str).to_numpy()
    if not np.array_equal(stroke.var_names.astype(str).to_numpy(), genes):
        raise RuntimeError("MAP and stroke gene orders differ")
    with h5py.File(ORIGINAL, "r") as h:
        batch = h["Batch"][:].astype(str)
        barcodes = h["Barcode"][:].astype(str)
        labels = h["Celltype_coarse"][:].astype(str)
    pt = np.char.startswith(batch, "PT")
    if not np.array_equal(barcodes[pt], obs_names):
        raise RuntimeError("CONCERT PT and MAP target spot orders differ")
    pred_c = np.load(LOCK / "CONCERT_PT_normalized_counts.npy")
    if pred_c.shape != counts.shape:
        raise RuntimeError(f"CONCERT prediction shape {pred_c.shape} != {counts.shape}")
    lock = json.loads((LOCK / "CONCERT_PREDICTION_LOCK.json").read_text())
    depth = float(json.loads((MAP_ROOT / "LOCKED_PREDICTIONS/SOURCE_AND_ATLAS_FREEZE.json").read_text())["depth_target"])
    # The official decoder emits the normalized mean before multiplication by
    # the per-spot size factor. Convert its training-median scale to the fixed
    # MAP source-median depth using one predetermined global factor.
    pred_c_map_scale = pred_c * (depth / float(lock["training_median_depth"]))
    concert_log = np.log1p(np.maximum(pred_c_map_scale, 0)).astype(np.float32)
    truth_log = np.log1p(normalize_counts(counts, depth)).astype(np.float32)
    regions_raw = labels[pt]
    regions = np.where(regions_raw == "ICA", "ICA", np.where(regions_raw == "PIA", "PIA", "Other"))
    coords = np.asarray(stroke.obsm["spatial_3d"], np.float32)
    return source, genes, hidden, obs_names, truth_log, concert_log, regions, coords, lock


def write_mask(obs_names, genes, hidden):
    with (ROOT / "SHARED_EVALUATION_MASK.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "axis_index", "identifier", "included", "note"])
        for i, b in enumerate(obs_names):
            w.writerow(["spot", i, b, True, "all stroke/PT target spots"])
        hidden_set = set(hidden.tolist())
        for i, g in enumerate(genes):
            w.writerow(["gene", i, g, i in hidden_set, "source-frozen eligible non-anchor" if i in hidden_set else "not in shared hidden-gene mask"])


def evaluate():
    source, genes, hidden, obs_names, truth, concert, regions, coords, concert_lock = load_inputs()
    write_mask(obs_names, genes, hidden)
    methods = ["HyperSpatial-MAP", "CONCERT"]
    rows = []
    gene_seed_rows = []
    region_rows = []
    spatial_rows = []
    cached = {}
    k = min(13, len(coords))
    neighbors = cKDTree(coords).query(coords, k=k)[1][:, 1:]

    for seed in (42, 43, 44):
        map_pred = np.load(MAP_ROOT / f"LOCKED_PREDICTIONS/seed{seed}_hyperspatial_map_calibrated_fusion.npy")
        base = np.load(MAP_ROOT / f"LOCKED_PREDICTIONS/seed{seed}_registered_idw_k12.npy")
        for method, pred in [(methods[0], map_pred), (methods[1], concert)]:
            y = truth[:, hidden]; ph = pred[:, hidden]; b = base[:, hidden]
            true_res = y - b; pred_res = ph - b
            spatial = pearson_columns(y, ph)
            residual = pearson_columns(true_res, pred_res)
            rmse = np.sqrt(np.mean(np.square(y - ph), axis=0))
            rows.append({
                "seed": seed, "method": method, "n_spots": len(obs_names), "n_hidden_genes": len(hidden),
                "median_spatial_pearson": np.nanmedian(spatial),
                "pooled_expression_pearson": safe_corr(y, ph),
                "pooled_log1p_rmse": np.sqrt(np.mean(np.square(y - ph))),
                "median_gene_log1p_rmse": np.nanmedian(rmse),
                "median_gene_residual_pearson": np.nanmedian(residual),
                "pooled_residual_pearson": safe_corr(true_res, pred_res),
                "residual_log1p_rmse": np.sqrt(np.mean(np.square(true_res - pred_res))),
            })
            for j, gi in enumerate(hidden):
                gene_seed_rows.append({"seed": seed, "gene": genes[gi], "gene_index": gi, "method": method,
                                       "spatial_pearson": spatial[j], "residual_pearson": residual[j], "log1p_rmse": rmse[j]})
            for region in ("ICA", "PIA", "Other"):
                use = regions == region
                rsp = pearson_columns(y[use], ph[use])
                rres = pearson_columns(true_res[use], pred_res[use])
                region_rows.append({
                    "seed": seed, "region": region, "method": method, "n_spots": int(use.sum()),
                    "median_spatial_pearson": np.nanmedian(rsp),
                    "pooled_expression_pearson": safe_corr(y[use], ph[use]),
                    "pooled_residual_pearson": safe_corr(true_res[use], pred_res[use]),
                    "median_gene_residual_pearson": np.nanmedian(rres),
                    "pooled_log1p_rmse": np.sqrt(np.mean(np.square(y[use] - ph[use]))),
                })
            true_m = moran_columns(true_res, neighbors)
            pred_m = moran_columns(pred_res, neighbors)
            spatial_rows.append({
                "seed": seed, "method": method, "n_genes": len(hidden),
                "median_true_residual_moran_I": np.nanmedian(true_m),
                "median_predicted_residual_moran_I": np.nanmedian(pred_m),
                "true_predicted_gene_moran_correlation": safe_corr(true_m, pred_m),
                "median_moran_attenuation_predicted_minus_true": np.nanmedian(pred_m - true_m),
                "pooled_true_predicted_residual_profile_correlation": safe_corr(true_res, pred_res),
            })
            cached[(seed, method)] = pred

    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby("method", as_index=False).agg(
        n_spots=("n_spots", "first"), n_hidden_genes=("n_hidden_genes", "first"),
        median_spatial_pearson=("median_spatial_pearson", "mean"),
        pooled_expression_pearson=("pooled_expression_pearson", "mean"),
        pooled_log1p_rmse=("pooled_log1p_rmse", "mean"),
        median_gene_log1p_rmse=("median_gene_log1p_rmse", "mean"),
        median_gene_residual_pearson=("median_gene_residual_pearson", "mean"),
        pooled_residual_pearson=("pooled_residual_pearson", "mean"),
        residual_log1p_rmse=("residual_log1p_rmse", "mean"),
    )
    summary.to_csv(ROOT / "RESULTS_SUMMARY.csv", index=False)

    gs = pd.DataFrame(gene_seed_rows)
    gene = gs.groupby(["gene", "gene_index", "method"], as_index=False).agg(
        spatial_pearson=("spatial_pearson", "mean"), residual_pearson=("residual_pearson", "mean"),
        log1p_rmse=("log1p_rmse", "mean"), spatial_pearson_sd=("spatial_pearson", "std"),
        residual_pearson_sd=("residual_pearson", "std"), log1p_rmse_sd=("log1p_rmse", "std"))
    wide = gene.pivot(index=["gene", "gene_index"], columns="method", values=["spatial_pearson", "residual_pearson", "log1p_rmse"])
    out = pd.DataFrame(index=wide.index).reset_index()
    for metric in ("spatial_pearson", "residual_pearson", "log1p_rmse"):
        out[f"MAP_{metric}"] = wide[(metric, "HyperSpatial-MAP")].to_numpy()
        out[f"CONCERT_{metric}"] = wide[(metric, "CONCERT")].to_numpy()
    out["MAP_minus_CONCERT_spatial_pearson"] = out.MAP_spatial_pearson - out.CONCERT_spatial_pearson
    out["MAP_minus_CONCERT_residual_pearson"] = out.MAP_residual_pearson - out.CONCERT_residual_pearson
    out["favorable_MAP_minus_CONCERT_RMSE"] = out.CONCERT_log1p_rmse - out.MAP_log1p_rmse
    out.to_csv(ROOT / "GENE_LEVEL_COMPARISON.csv", index=False)

    region = pd.DataFrame(region_rows).groupby(["region", "method"], as_index=False).agg(
        n_spots=("n_spots", "first"), median_spatial_pearson=("median_spatial_pearson", "mean"),
        pooled_expression_pearson=("pooled_expression_pearson", "mean"),
        pooled_residual_pearson=("pooled_residual_pearson", "mean"),
        median_gene_residual_pearson=("median_gene_residual_pearson", "mean"),
        pooled_log1p_rmse=("pooled_log1p_rmse", "mean"))
    region.to_csv(ROOT / "REGION_COMPARISON.csv", index=False)
    spatial = pd.DataFrame(spatial_rows)
    spatial.to_csv(ROOT / "SPATIAL_STRUCTURE.csv", index=False)

    info = pd.DataFrame([
        {"Method": "HyperSpatial-MAP", "Reference expression": "complete sham, 2,003 genes", "Target geometry": "label-free reconstructed xyz, 10,583 spots", "16 measured target genes": "yes; exactly 16", "Perturbation identity": "no", "Lesion/context information": "no before lock; ICA/PIA only for evaluation", "Other target information": "public gene identifiers; hidden 1,283 genes sealed"},
        {"Method": "CONCERT", "Reference expression": "complete sham, 2,003 genes", "Target geometry": "deposited pathology-aligned 3D_pos_PT plus 3D_pos_sham; PT/sham scaled separately", "16 measured target genes": "not used as a sparse panel", "Perturbation identity": "yes; PT/sham batch and ICA-vs-other attribute", "Lesion/context information": "yes; Celltype_coarse ICA labels during training", "Other target information": "complete PT expression for all 2,003 genes, per-spot size factors, learned per-spot basal embeddings; 5% internal validation"},
    ])
    info.to_csv(ROOT / "INFORMATION_BUDGET.csv", index=False)

    sm = summary.set_index("method")
    frac_sp = float((out.MAP_spatial_pearson > out.CONCERT_spatial_pearson).mean())
    frac_rmse = float((out.MAP_log1p_rmse < out.CONCERT_log1p_rmse).mean())
    raw_concert = np.load(LOCK / "CONCERT_PT_normalized_counts.npy")
    ceiling = raw_concert >= 1e6
    clean_hidden = ~np.any(ceiling[:, hidden], axis=0)
    pd.DataFrame([{
        "scope": "complete_PT_output", "n_entries": raw_concert.size,
        "n_ceiling": int(ceiling.sum()), "fraction_ceiling": float(ceiling.mean()),
        "n_spots_any_ceiling": int(np.any(ceiling, axis=1).sum()),
        "n_genes_any_ceiling": int(np.any(ceiling, axis=0).sum()),
        "median_library_sum": float(np.median(raw_concert.sum(1))),
        "p99_library_sum": float(np.percentile(raw_concert.sum(1), 99)),
        "max_library_sum": float(np.max(raw_concert.sum(1))),
    }]).to_csv(ROOT / "CONCERT_OUTPUT_QC.csv", index=False)
    decision = "A" if (sm.loc["HyperSpatial-MAP", "median_spatial_pearson"] > sm.loc["CONCERT", "median_spatial_pearson"] and sm.loc["HyperSpatial-MAP", "pooled_log1p_rmse"] < sm.loc["CONCERT", "pooled_log1p_rmse"]) else "B" if ((sm.loc["HyperSpatial-MAP", "median_spatial_pearson"] > sm.loc["CONCERT", "median_spatial_pearson"]) != (sm.loc["HyperSpatial-MAP", "pooled_log1p_rmse"] < sm.loc["CONCERT", "pooled_log1p_rmse"])) else "C"

    meta = {
        "decision": decision, "fraction_genes_MAP_wins_spatial_pearson": frac_sp,
        "fraction_genes_MAP_wins_RMSE": frac_rmse,
        "concert_global_scale_factor_to_MAP_depth": 4060.0 / float(concert_lock["training_median_depth"]),
        "representative_gene": "Ppp1r1b",
        "representative_rule": "previously frozen source-only top source-validation Pearson among active-correction hidden genes",
        "concert_ceiling_fraction": float(ceiling.mean()),
        "n_clean_hidden_genes": int(clean_hidden.sum()),
        "clean_fraction_MAP_wins_spatial": float(np.mean(out.loc[clean_hidden, "MAP_spatial_pearson"].to_numpy() > out.loc[clean_hidden, "CONCERT_spatial_pearson"].to_numpy())),
        "clean_fraction_MAP_wins_RMSE": float(np.mean(out.loc[clean_hidden, "MAP_log1p_rmse"].to_numpy() < out.loc[clean_hidden, "CONCERT_log1p_rmse"].to_numpy())),
        "clean_MAP_median_spatial": float(np.nanmedian(out.loc[clean_hidden, "MAP_spatial_pearson"])),
        "clean_CONCERT_median_spatial": float(np.nanmedian(out.loc[clean_hidden, "CONCERT_spatial_pearson"])),
        "clean_MAP_median_RMSE": float(np.nanmedian(out.loc[clean_hidden, "MAP_log1p_rmse"])),
        "clean_CONCERT_median_RMSE": float(np.nanmedian(out.loc[clean_hidden, "CONCERT_log1p_rmse"])),
    }
    (ROOT / "DECISION_METRICS.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_documents(summary, region, spatial, meta, concert_lock)
    make_figure(truth, concert, cached[(42, "HyperSpatial-MAP")], np.load(MAP_ROOT / "LOCKED_PREDICTIONS/seed42_registered_idw_k12.npy"), genes, hidden, coords, regions, out, region, info)
    write_manifest()


def write_documents(summary, region, spatial, meta, lock):
    s = summary.set_index("method"); m=s.loc["HyperSpatial-MAP"]; c=s.loc["CONCERT"]
    decision_text = {"A": "MAP higher on both prespecified central endpoints", "B": "mixed metric-specific strengths", "C": "CONCERT higher on both prespecified central endpoints"}[meta["decision"]]
    (ROOT / "CONCERT_REPRODUCTION.md").write_text(f"""# CONCERT reproduction\n\nThe official CONCERT repository at commit `8babda00d09a44b5812bac250434cdd5d057bb18` was used without editing its model. The released 3D architecture (`concert_batch_3D_stroke.py`) and `config_3D.yaml` settings were used: 3D spatial GP dimension 2; Gaussian dimension 8; encoder 256/64; decoder 64/256; x/y/z range 60; kernel scale 60 with batch-specific scaling; AdamW learning rate 1e-4; weight decay 1e-5; batch size 512; 95% training and 5% validation; maximum 3,000 epochs and patience 200. The repository supplied no stroke checkpoint. RNG seed 0 was fixed only for reproducibility because the official runner is otherwise unseeded.\n\nTraining used the complete 19,777-spot by 2,003-gene H5, both sham and PT expression, deposited `3D_pos_PT`/`3D_pos_sham`, the PT-versus-sham batch indicator, and the `Celltype_coarse == ICA` perturbation attribute. The locked output was produced with the official `batching_denoise_counts` API using 25 latent samples. This is reconstruction of observations used by the model, not hidden-gene forecasting.\n\nCONCERT emits its normalized decoder mean. It was multiplied by the fixed ratio 4060/{lock['training_median_depth']:.0f} = {4060/lock['training_median_depth']:.9f} to place it on MAP's sham-median depth scale, then transformed with log1p. This deterministic conversion was specified from preprocessing scales and did not optimize against target truth.\n\nTraining time was {lock['training_seconds']/60:.1f} min. Model SHA256: `{lock['model_sha256']}`. Prediction SHA256: `{lock['prediction_sha256']}`.\n\n## Environment and output-range QC\n\nThe released environment specifies PyTorch 2.1.0, Scanpy 1.10.1, scikit-learn 1.4.0, SciPy 1.12.0, pandas 2.2.0 and NumPy 1.23.5. The available GPU-compatible environment used PyTorch 2.7.0+cu126, Scanpy 1.10.4, scikit-learn 1.4.2, SciPy 1.15.1, pandas 2.2.3 and NumPy 1.26.4. This is therefore successful execution of the official code/model/configuration, but not a bitwise recreation in the exact package environment; no released stroke checkpoint was available.\n\nThe official `MeanAct` ceiling of 1,000,000 was reached in {100*meta['concert_ceiling_fraction']:.3f}% of PT output entries. No post-hoc clipping or target-calibrated correction was applied. In the target-truth-independent subset of {meta['n_clean_hidden_genes']} hidden genes with no ceiling value, MAP still won spatial Pearson for {100*meta['clean_fraction_MAP_wins_spatial']:.1f}% and RMSE for {100*meta['clean_fraction_MAP_wins_RMSE']:.1f}%; median spatial Pearson was {meta['clean_MAP_median_spatial']:.4f} versus {meta['clean_CONCERT_median_spatial']:.4f}, and median gene RMSE was {meta['clean_MAP_median_RMSE']:.4f} versus {meta['clean_CONCERT_median_RMSE']:.4f}.\n""")
    (ROOT / "FAIRNESS_AUDIT.md").write_text("""# Fairness and information-use audit\n\nThis is an **output-matched reconstruction benchmark under method-specific information settings**, not a matched-input comparison. Both outputs were evaluated at the identical 10,583 PT spots and 1,283 frozen MAP hidden genes.\n\nBefore its prediction lock, HyperSpatial-MAP used complete sham expression, the label-free frozen target geometry, public gene identifiers and exactly 16 source-selected stroke genes. It did not use PT non-anchor expression, ICA/PIA labels, lesion identity, the pathology-informed deposited alignment or CONCERT's lesion footprint.\n\nCONCERT training used complete PT expression for all genes, the deposited sham/PT coordinates, PT/sham condition identity, ICA-derived perturbation labels, per-spot depth factors and a learned per-observation basal embedding. The 5% validation subset controlled early stopping. Consequently, the 1,283 evaluation genes were not hidden from CONCERT training. ICA/PIA/Other were used by MAP only after lock, but ICA was a CONCERT training attribute.\n\nNo MAP or CONCERT hyperparameter was selected using the head-to-head results. The shared atlas prior B_T was the existing seed-specific registered sham expectation from the frozen MAP run and was not recomputed for CONCERT.\n\nThe CONCERT decoder ceiling and package-version difference are documented in `CONCERT_REPRODUCTION.md`. The primary comparison retains the complete locked output; a ceiling-free-gene sensitivity was used only to test dependence on numerical saturation, not to replace prespecified metrics.\n""")
    safe = "HyperSpatial-MAP achieved higher reconstruction accuracy on the same perturbed target under its sparse-target-measurement setting." if meta["decision"] == "A" else ("HyperSpatial-MAP and CONCERT showed metric-specific strengths on the same perturbed target under different method-specific information settings." if meta["decision"] == "B" else "CONCERT achieved higher reconstruction accuracy on the same perturbed target, although it used complete target expression and explicit condition information during training.")
    (ROOT / "MANUSCRIPT_SAFE_INTERPRETATION.md").write_text(f"# Manuscript-safe interpretation\n\n{safe}\n\nThis output-matched comparison does not establish universal superiority and must be accompanied by the information-budget table. CONCERT's evaluated genes were present during training, whereas HyperSpatial-MAP received only 16 measured target genes.\n")
    (ROOT / "FINAL_DECISION.md").write_text(f"""# Final decision: {meta['decision']} — {decision_text}\n\n| Metric | HyperSpatial-MAP | CONCERT |\n|---|---:|---:|\n| Median spatial Pearson | {m.median_spatial_pearson:.6f} | {c.median_spatial_pearson:.6f} |\n| Pooled log1p RMSE | {m.pooled_log1p_rmse:.6f} | {c.pooled_log1p_rmse:.6f} |\n| Median gene-wise residual Pearson | {m.median_gene_residual_pearson:.6f} | {c.median_gene_residual_pearson:.6f} |\n| Pooled residual Pearson | {m.pooled_residual_pearson:.6f} | {c.pooled_residual_pearson:.6f} |\n\nMAP won spatial Pearson for {100*meta['fraction_genes_MAP_wins_spatial_pearson']:.2f}% of genes and RMSE for {100*meta['fraction_genes_MAP_wins_RMSE']:.2f}%.\n\nThis is an output-matched reconstruction benchmark under method-specific information settings. CONCERT had complete PT expression and ICA condition labels during training; MAP had 16 PT genes and no lesion identity.\n\nThe official CONCERT decoder reached its built-in 10^6 ceiling in {100*meta['concert_ceiling_fraction']:.3f}% of output entries. The primary analysis retained these values. In the target-truth-independent sensitivity restricted to {meta['n_clean_hidden_genes']} hidden genes without ceiling values, MAP still won spatial Pearson for {100*meta['clean_fraction_MAP_wins_spatial']:.1f}% and RMSE for {100*meta['clean_fraction_MAP_wins_RMSE']:.1f}%, so Decision A was not driven solely by saturation.\n\n**Manuscript-safe statement:** {safe}\n""")


def make_figure(truth, concert, map_pred, base, genes, hidden, coords, regions, gene, region, info):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(12.4, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.78, 1.22], width_ratios=[1.05, 1, 1, 1.05])
    axa = fig.add_subplot(gs[0, 0:2]); axb=fig.add_subplot(gs[0,2]); axc=fig.add_subplot(gs[0,3]); axd=fig.add_subplot(gs[1,0]); axf=fig.add_subplot(gs[1,3])
    # a information budget
    axa.axis('off'); axa.set_title("a  Method-specific information", loc='left', fontweight='bold', pad=8)
    axa.text(.35,.82,'HyperSpatial-MAP',ha='center',fontweight='bold',color='#167D70',transform=axa.transAxes)
    axa.text(.76,.82,'CONCERT',ha='center',fontweight='bold',color='#B77900',transform=axa.transAxes)
    rows=[('Molecular','16 target genes','Complete PT expression'),('Geometry','Label-free target xyz','Deposited aligned xyz'),('Condition','No lesion identity','PT/sham + ICA identity'),('Evaluation genes','Sealed before prediction','Used during training')]
    for i,(lab,left,right) in enumerate(rows):
        y=.65-i*.16
        axa.text(.01,y,lab,ha='left',va='center',fontweight='bold',color='.25',transform=axa.transAxes)
        axa.text(.35,y,left,ha='center',va='center',transform=axa.transAxes)
        axa.text(.76,y,right,ha='center',va='center',transform=axa.transAxes)
        axa.plot([.15,.96],[y-.075,y-.075],color='.88',lw=.6,transform=axa.transAxes,clip_on=False)
    # b/c paired per gene scatter
    lim=[np.nanpercentile(np.r_[gene.MAP_spatial_pearson,gene.CONCERT_spatial_pearson],[1,99])[0],np.nanpercentile(np.r_[gene.MAP_spatial_pearson,gene.CONCERT_spatial_pearson],[1,99])[1]]
    axb.scatter(gene.CONCERT_spatial_pearson,gene.MAP_spatial_pearson,s=5,alpha=.25,color='#2A9D8F',rasterized=True);axb.plot(lim,lim,color='.5',lw=.8);axb.set(xlabel='CONCERT',ylabel='HyperSpatial-MAP');axb.set_title('b  Spatial Pearson',loc='left',fontweight='bold');axb.set_xlim(lim);axb.set_ylim(lim)
    limr=[0,max(np.nanpercentile(gene[['MAP_log1p_rmse','CONCERT_log1p_rmse']].to_numpy(),99),.1)]
    axc.scatter(gene.CONCERT_log1p_rmse,gene.MAP_log1p_rmse,s=5,alpha=.25,color='#E76F51',rasterized=True);axc.plot(limr,limr,color='.5',lw=.8);axc.set(xlabel='CONCERT',ylabel='HyperSpatial-MAP');axc.set_title('c  log1p RMSE',loc='left',fontweight='bold');axc.set_xlim(limr);axc.set_ylim(limr)
    # d residual paired distributions
    vals=[gene.MAP_residual_pearson.dropna(),gene.CONCERT_residual_pearson.dropna()]
    vp=axd.violinplot(vals,showmedians=True,showextrema=False)
    for body,color in zip(vp['bodies'],['#2A9D8F','#E9C46A']): body.set_facecolor(color);body.set_alpha(.8)
    axd.axhline(0,color='.6',lw=.7);axd.set_xticks([1,2],['MAP','CONCERT']);axd.set(ylabel='Gene-wise residual Pearson');axd.set_title('d  Atlas-relative residual recovery',loc='left',fontweight='bold')
    # e representative maps, source-predeclared Ppp1r1b
    gi=int(np.where(genes=='Ppp1r1b')[0][0]); true_res=truth[:,gi]-base[:,gi]; map_res=map_pred[:,gi]-base[:,gi]; c_res=concert[:,gi]-base[:,gi]
    maps=[truth[:,gi],map_pred[:,gi],concert[:,gi],true_res,map_res,c_res];titles=['Measured','MAP','CONCERT','True residual','MAP residual','CONCERT residual']
    sub=gs[1,1:3].subgridspec(2,3,wspace=.02,hspace=.12)
    vmax=np.nanpercentile(np.r_[maps[0],maps[1],maps[2]],99); rmax=np.nanpercentile(np.abs(np.r_[maps[3],maps[4],maps[5]]),99)
    for i,(v,t) in enumerate(zip(maps,titles)):
        a=fig.add_subplot(sub[i//3,i%3],projection='3d'); a.scatter(coords[:,0],coords[:,1],coords[:,2],c=v,s=.3,cmap='viridis' if i<3 else 'coolwarm',vmin=0 if i<3 else -rmax,vmax=vmax if i<3 else rmax,rasterized=True);a.set_title(('e  Ppp1r1b (source-predeclared)\n' if i==0 else '')+t,fontsize=7,pad=0,loc='left' if i==0 else 'center',fontweight='bold' if i==0 else 'normal');a.set_axis_off();a.view_init(25,-68)
    # f regional
    piv=region.pivot(index='region',columns='method',values='pooled_residual_pearson').loc[['ICA','PIA','Other']];x=np.arange(3);w=.36
    axf.bar(x-w/2,piv['HyperSpatial-MAP'],w,label='MAP',color='#2A9D8F');axf.bar(x+w/2,piv['CONCERT'],w,label='CONCERT',color='#E9C46A');axf.set_xticks(x,['ICA','PIA','Other']);axf.set(ylabel='Pooled residual Pearson');axf.set_title('f  Regional recovery',loc='left',fontweight='bold');axf.legend(frameon=False,fontsize=7)
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'MAP_vs_CONCERT_stroke.{ext}',dpi=400 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)


def write_manifest():
    paths=[MAP_ROOT/"LOCKED_PREDICTIONS/PREDICTION_LOCK.json",MAP_ROOT/"LOCKED_PREDICTIONS/SOURCE_MODEL.npz",GEOM_ROOT/"STROKE_3D.h5ad",GEOM_ROOT/"SHAM_3D.h5ad",ORIGINAL,LOCK/"CONCERT_PREDICTION_LOCK.json",LOCK/"CONCERT_PT_normalized_counts.npy"]
    (ROOT/"INPUT_AND_OUTPUT_HASHES.txt").write_text(''.join(f'{digest(p)}  {p}\n' for p in paths))


if __name__ == '__main__':
    evaluate()
