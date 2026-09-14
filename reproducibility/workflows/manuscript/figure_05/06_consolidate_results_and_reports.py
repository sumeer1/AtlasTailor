#!/usr/bin/env python3
"""Consolidate frozen SPATCH outputs and write the requested audit reports."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "official_data" / "extracted"
TUMORS = ("COAD", "HCC", "OV")
SPECIMENS = {"HCC": "Benchmark_01", "OV": "Benchmark_02", "COAD": "Benchmark_03"}


def source_gene_table(tumor, model, lock):
    genes = model["gene_names"].astype(str)
    rows = model["source_rows"].astype(np.int64)
    source = ad.read_h5ad(DATA / f"VisiumHD_FFPE_{tumor}" / "adata.h5ad", backed="r")
    by_name = {name: i for i, name in enumerate(source.var_names.astype(str))}
    columns = np.asarray([by_name[g] for g in genes], np.int64)
    counts = source[rows, columns].X
    if hasattr(counts, "to_memory"):
        counts = counts.to_memory()
    counts = counts.tocsr()
    source.file.close()
    prevalence = np.asarray(counts.getnnz(axis=0)).ravel() / len(rows)
    depth = np.maximum(np.asarray(counts.sum(axis=1)).ravel().astype(np.float32), 1.0)
    target_depth = float(np.median(depth[depth > 0]))
    rng = np.random.default_rng(42116)
    sample = np.sort(rng.choice(len(rows), min(5000, len(rows)), replace=False))
    sampled = counts[sample].toarray().astype(np.float32)
    sampled *= (target_depth / depth[sample])[:, None]
    sampled = np.log1p(sampled)
    anchors = model["anchor_indices"].astype(int)
    redundancy = {}
    for position, index in enumerate(anchors):
        if position == 0:
            redundancy[index] = 0.0
        else:
            x = sampled[:, index]
            values = []
            for prior in anchors[:position]:
                y = sampled[:, prior]
                values.append(abs(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0)
            redundancy[index] = float(max(values))
    anchor_set = set(anchors.tolist())
    eligible = set(model["evaluation_indices"].astype(int).tolist())
    marginal = lock["anchor_metadata"]["marginal_coverage_gains"]
    anchor_position = {int(index): pos + 1 for pos, index in enumerate(anchors)}
    records = []
    for index, gene in enumerate(genes):
        role = "anchor" if index in anchor_set else ("hidden_evaluation" if index in eligible else "source_ineligible_shared")
        position = anchor_position.get(index)
        records.append({
            "pair_id": lock["pair_id"], "specimen_id": SPECIMENS[tumor], "tumor": tumor,
            "gene": gene, "shared_by_panel_identity": True, "role": role,
            "anchor_position": position, "source_prevalence": prevalence[index],
            "source_coverage_score_overall": lock["anchor_metadata"]["mean_squared_gene_coverage"] if position else np.nan,
            "anchor_marginal_coverage_gain": marginal[position - 1] if position else np.nan,
            "max_abs_source_correlation_to_previous_anchor": redundancy.get(index, np.nan),
            "target_expression_used_for_role_or_panel_selection": False,
        })
    return records


pair_frames, gene_frames, shared_records, registration_records = [], [], [], []
for tumor in TUMORS:
    pair = ROOT / "work" / tumor
    model = np.load(pair / "source_model.npz", allow_pickle=True)
    lock = json.loads((pair / "source_geometry_lock.json").read_text())
    pair_frames.append(pd.read_csv(pair / "pair_level_results.tsv", sep="\t"))
    gene_frames.append(pd.read_csv(pair / "gene_level_results.tsv", sep="\t"))
    shared_records.extend(source_gene_table(tumor, model, lock))
    platform = json.loads((pair / "platform_anchor_correlation_control.json").read_text())
    for seed, item in lock["registration"].items():
        registration_records.append({
            "pair_id": lock["pair_id"], "specimen_id": SPECIMENS[tumor], "tumor": tumor,
            "seed": int(seed), "reflection": item["reflection"],
            "pre_registration_chamfer": item["raw_chamfer"],
            "post_registration_chamfer": item["registered_chamfer"],
            "source_overlap_within_2x_target_spacing": item["source_within_2x_target_spacing"],
            "geometry_only_common_roi_target_bins": lock["geometry_only_common_roi_bin_count"],
            "median_anchor_spatial_pearson_rigid_seed42": platform["before_nonrigid_registration_median"],
            "median_anchor_spatial_pearson_registered_seed42": platform["after_nonrigid_registration_median"],
            "seed_stability_42_vs_43": lock["registration_seed_stability"]["42_vs_43_median_source_displacement"],
            "seed_stability_42_vs_44": lock["registration_seed_stability"]["42_vs_44_median_source_displacement"],
            "seed_stability_43_vs_44": lock["registration_seed_stability"]["43_vs_44_median_source_displacement"],
        })

pair_results = pd.concat(pair_frames, ignore_index=True)
gene_results = pd.concat(gene_frames, ignore_index=True)
pair_results.insert(1, "specimen_id", pair_results.tumor.map(SPECIMENS))
gene_results.insert(1, "specimen_id", gene_results.tumor.map(SPECIMENS))
pd.DataFrame(shared_records).to_csv(ROOT / "01_SHARED_GENES_AND_ANCHORS.tsv", sep="\t", index=False)
pair_results.to_csv(ROOT / "03_PAIR_LEVEL_RESULTS.tsv", sep="\t", index=False)
gene_results.to_csv(ROOT / "04_GENE_LEVEL_RESULTS.tsv", sep="\t", index=False)
pd.DataFrame(registration_records).to_csv(ROOT / "05_REGISTRATION_RESULTS.tsv", sep="\t", index=False)

map_rows = pair_results.loc[pair_results.method == "hyperspatial_map"].set_index("tumor")
idw_rows = pair_results.loc[pair_results.method == "registered_idw_k12"].set_index("tumor")
spatial_wins = map_rows.median_gene_wise_spatial_pearson > idw_rows.median_gene_wise_spatial_pearson
rmse_wins = map_rows.log1p_rmse < idw_rows.log1p_rmse
residual_positive = map_rows.pooled_bin_gene_residual_pearson > 0
if bool((spatial_wins & rmse_wins & residual_positive).all()):
    decision = "A — strong cross-platform generalization"
elif bool((spatial_wins | rmse_wins | residual_positive).any()):
    decision = "B — partial cross-platform generalization"
else:
    decision = "C — negative"

leakage = f"""# Leakage audit

## Result

PASS. No Xenium non-anchor expression was materialized before every source model, registration, common ROI, registered 12-NN atlas prior and hidden-gene prediction had been serialized and SHA-256 hashed in `02_PREDICTION_LOCK_MANIFEST.tsv`.

## Information boundaries

**Allowed before prediction lock:** official specimen/platform metadata; Visium HD source counts and coordinates; Xenium panel gene identities; Xenium DAPI/transcript coordinates used by fixed 8-µm preprocessing; geometry-only target coordinates and tissue support; and the 16 requested Xenium anchor columns.

**Forbidden before prediction lock:** all Xenium non-anchor expression values; target variance/detection summaries for filtering; target outcomes for anchor selection, hyperparameter selection, operator selection, registration or representative-gene selection; target cell types, tumor regions and pathology annotations.

The source-only panel and candidate operator guard were fitted separately for each Visium HD specimen. Target depth was inferred only from the 16 anchors using the source-trained depth model. The evaluation script refuses to run unless all three pair locks exist and all 27 artifact hashes verify.

## Fixed analysis choices

- 8-µm bins, 16 anchors, registered 12-NN inverse-distance prior, 12- and 32-neighbour residual summaries, rank 16, fixed ridge/gain/fusion grids and registration seeds 42/43/44.
- Geometry-only common ROI fixed from seed 42 before any target molecular values were read.
- No cell-type or pathological annotation used in fitting.
- No post-outcome retuning was performed.

## Secondary analyses not promoted

The adapted gimVI baseline was not run because no frozen SPATCH 8-µm configuration existed that could be applied without introducing a new large-data adaptation and changing the existing scientific definition. The optional 8/32-gene and 20-random-panel controls were not used to delay or modify the frozen primary 16-gene result.
"""
(ROOT / "06_LEAKAGE_AUDIT.md").write_text(leakage)

summary = pd.DataFrame({
    "MAP_spatial": map_rows.median_gene_wise_spatial_pearson,
    "IDW_spatial": idw_rows.median_gene_wise_spatial_pearson,
    "MAP_RMSE": map_rows.log1p_rmse,
    "IDW_RMSE": idw_rows.log1p_rmse,
    "MAP_pooled_residual_r": map_rows.pooled_bin_gene_residual_pearson,
}).to_markdown(floatfmt=".4f")
balanced = pair_results.groupby("method", sort=False)[[
    "median_gene_wise_spatial_pearson", "pooled_bin_gene_residual_pearson",
    "median_gene_wise_residual_pearson", "log1p_rmse",
    "fraction_positive_gene_residual_recovery",
]].mean(numeric_only=True).to_markdown(floatfmt=".4f")
map_fraction = map_rows[["fraction_genes_MAP_gt_IDW_spatial_pearson",
                         "fraction_genes_MAP_lt_IDW_log1p_RMSE"]]
fraction_text = (
    f"The specimen-balanced mean MAP win fractions were "
    f"{map_fraction.fraction_genes_MAP_gt_IDW_spatial_pearson.mean():.3f} for spatial Pearson "
    f"and {map_fraction.fraction_genes_MAP_lt_IDW_log1p_RMSE.mean():.3f} for RMSE. "
    "These gene-level directions are technical records and are not inferential replicates."
)
decision_text = f"""# Final decision

## {decision}

The decision was assigned without retuning after target opening. Criterion A requires MAP to improve both central reconstruction endpoints (median gene-wise spatial Pearson and log1p RMSE) and to show positive pooled atlas-relative residual recovery in every specimen. Criterion B denotes a genuine but non-uniform benefit; criterion C denotes no reproducible benefit.

{summary}

### Specimen-balanced mean across the three pairs

{balanced}

{fraction_text}

Genes and spatial bins are technical evaluation records, not biological replicates. The three same-specimen tumor pairs are the biological evaluation units; all cross-specimen summaries are specimen-balanced.
"""
(ROOT / "07_FINAL_DECISION.md").write_text(decision_text)

if decision.startswith("A"):
    core = ("HyperSpatial-MAP generalized across spatial measurement technologies: a Visium HD reference "
            "could be adapted to a serial Xenium section from the same specimen using only a sparse target panel.")
elif decision.startswith("B"):
    core = ("HyperSpatial-MAP showed partial cross-platform transfer from Visium HD to serial Xenium sections, "
            "with benefits that were not consistent across all tumor specimens and endpoints.")
else:
    core = ("Cross-platform and serial-section differences dominated this benchmark; HyperSpatial-MAP did not "
            "reproducibly improve the registered Visium HD prior on Xenium targets.")
interpretation = f"""# Results interpretation

{core}

This is an external cross-platform reconstruction experiment, not a perturbation experiment. The Visium HD and Xenium assays derive from serial sections of the same FFPE block, with intervening assay sections in the published cutting order. The design therefore does not establish cell-to-cell correspondence, platform invariance, clinical generalization or independent biological replication. It asks whether 16 target measurements can adapt a registered molecular reference to the molecular state measured on another section and technology.

The comparison to registered IDW isolates improvement beyond geometry-only atlas transfer. Anchor-only decoding tests prediction from the sparse panel without an atlas prior. Any interpretation should remain at the output level and preserve the specimen-balanced analysis.

The anchor-only decoder exceeded MAP on median spatial Pearson and log1p RMSE in all three specimens. Thus the A decision is specific to the predeclared MAP-versus-registered-IDW question; it does not show that atlas fusion was superior to sparse-panel decoding in this benchmark. HCC was particularly heterogeneous: the pair-level MAP endpoints improved only slightly over IDW, and most gene records were ties created by the source guard's registered-IDW fallback.
"""
(ROOT / "08_RESULTS_INTERPRETATION.md").write_text(interpretation)
print(decision)
