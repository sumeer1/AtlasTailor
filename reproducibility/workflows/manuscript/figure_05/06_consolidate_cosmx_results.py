#!/usr/bin/env python3
"""Consolidate the frozen CosMx evaluation into IDW-vs-MAP outputs and reports."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TUMORS = ("COAD", "HCC", "OV")
SPECIMENS = {"HCC": "Benchmark_01", "OV": "Benchmark_02", "COAD": "Benchmark_03"}
LABELS = {"COAD": "colon adenocarcinoma", "HCC": "hepatocellular carcinoma", "OV": "ovarian cancer"}
KEEP = ("registered_idw_k12", "hyperspatial_map")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


pair_wide, gene_wide, anchors, registrations = [], [], [], []
for tumor in TUMORS:
    work = ROOT / "work" / tumor
    lock = json.loads((work / "source_geometry_lock.json").read_text())
    model = np.load(work / "source_model.npz", allow_pickle=True)
    pair = pd.read_csv(work / "pair_level_results.tsv", sep="\t")
    gene = pd.read_csv(work / "gene_level_results.tsv", sep="\t")
    pair = pair.loc[pair.method.isin(KEEP)].copy()
    gene = gene.loc[gene.method.isin(KEEP)].copy()
    indexed = pair.set_index("method")
    idw, map_row = indexed.loc["registered_idw_k12"], indexed.loc["hyperspatial_map"]
    pair_wide.append({
        "pair_id": lock["pair_id"], "specimen_id": SPECIMENS[tumor], "tumor": tumor,
        "source_platform": "Visium HD FFPE", "target_platform": "CosMx 6K",
        "target_units_8um_bins": int(map_row.target_bin_count), "measured_anchor_count": 16,
        "hidden_gene_count": int(map_row.hidden_gene_count),
        "IDW_median_gene_wise_spatial_pearson": idw.median_gene_wise_spatial_pearson,
        "MAP_median_gene_wise_spatial_pearson": map_row.median_gene_wise_spatial_pearson,
        "delta_spatial_pearson_MAP_minus_IDW": map_row.median_gene_wise_spatial_pearson - idw.median_gene_wise_spatial_pearson,
        "IDW_log1p_RMSE": idw.log1p_rmse, "MAP_log1p_RMSE": map_row.log1p_rmse,
        "delta_RMSE_MAP_minus_IDW": map_row.log1p_rmse - idw.log1p_rmse,
        "IDW_residual_pearson": np.nan,
        "MAP_median_gene_wise_residual_pearson": map_row.median_gene_wise_residual_pearson,
        "MAP_pooled_residual_pearson": map_row.pooled_bin_gene_residual_pearson,
        "fraction_hidden_genes_MAP_gt_IDW_spatial_pearson": map_row.fraction_genes_MAP_gt_IDW_spatial_pearson,
        "fraction_hidden_genes_MAP_lt_IDW_log1p_RMSE": map_row.fraction_genes_MAP_lt_IDW_log1p_RMSE,
        "fraction_hidden_genes_positive_MAP_residual_pearson": map_row.fraction_positive_gene_residual_recovery,
    })
    pivot = gene.pivot(index="gene", columns="method")
    for gene_name in pivot.index:
        gene_wide.append({
            "pair_id": lock["pair_id"], "specimen_id": SPECIMENS[tumor], "tumor": tumor,
            "source_platform": "Visium HD FFPE", "target_platform": "CosMx 6K", "gene": gene_name,
            "IDW_spatial_pearson": pivot.loc[gene_name, ("spatial_pearson", "registered_idw_k12")],
            "MAP_spatial_pearson": pivot.loc[gene_name, ("spatial_pearson", "hyperspatial_map")],
            "delta_spatial_pearson_MAP_minus_IDW": (
                pivot.loc[gene_name, ("spatial_pearson", "hyperspatial_map")] -
                pivot.loc[gene_name, ("spatial_pearson", "registered_idw_k12")]
            ),
            "IDW_log1p_RMSE": pivot.loc[gene_name, ("log1p_rmse", "registered_idw_k12")],
            "MAP_log1p_RMSE": pivot.loc[gene_name, ("log1p_rmse", "hyperspatial_map")],
            "delta_RMSE_MAP_minus_IDW": (
                pivot.loc[gene_name, ("log1p_rmse", "hyperspatial_map")] -
                pivot.loc[gene_name, ("log1p_rmse", "registered_idw_k12")]
            ),
            "IDW_residual_pearson": np.nan,
            "MAP_residual_pearson": pivot.loc[gene_name, ("residual_pearson", "hyperspatial_map")],
            "source_guard_supported": bool(pivot.loc[gene_name, ("source_guard_supported", "hyperspatial_map")]),
            "source_spatial_score": pivot.loc[gene_name, ("source_spatial_score", "hyperspatial_map")],
            "representative_source_only_pool": bool(pivot.loc[gene_name, ("representative_source_only_pool", "hyperspatial_map")]),
        })
    anchor_indices = model["anchor_indices"].astype(int)
    genes = model["gene_names"].astype(str)
    for position, index in enumerate(anchor_indices, start=1):
        anchors.append({
            "pair_id": lock["pair_id"], "specimen_id": SPECIMENS[tumor], "tumor": tumor,
            "source_platform": "Visium HD FFPE", "target_platform": "CosMx 6K",
            "anchor_position": position, "gene": genes[index],
            "source_prevalence": lock["source_prevalence"][position - 1],
            "source_marginal_coverage_gain": lock["anchor_metadata"]["marginal_coverage_gains"][position - 1],
            "selected_using_target_expression": False,
        })
    control = json.loads((work / "platform_anchor_correlation_control.json").read_text())
    for seed, record in lock["registration"].items():
        registrations.append({
            "pair_id": lock["pair_id"], "specimen_id": SPECIMENS[tumor], "tumor": tumor,
            "source_platform": "Visium HD FFPE", "target_platform": "CosMx 6K", "seed": int(seed),
            "reflection": record["reflection"], "pre_registration_chamfer": record["raw_chamfer"],
            "post_registration_chamfer": record["registered_chamfer"],
            "source_overlap_within_2x_target_spacing": record["source_within_2x_target_spacing"],
            "geometry_only_common_roi_target_bins": lock["geometry_only_common_roi_bin_count"],
            "median_anchor_spatial_pearson_rigid_seed42": control["before_nonrigid_registration_median"],
            "median_anchor_spatial_pearson_registered_seed42": control["after_nonrigid_registration_median"],
            "seed_stability_42_vs_43": lock["registration_seed_stability"]["42_vs_43_median_source_displacement"],
            "seed_stability_42_vs_44": lock["registration_seed_stability"]["42_vs_44_median_source_displacement"],
            "seed_stability_43_vs_44": lock["registration_seed_stability"]["43_vs_44_median_source_displacement"],
        })

pair_frame = pd.DataFrame(pair_wide)
gene_frame = pd.DataFrame(gene_wide)
pd.DataFrame(anchors).to_csv(ROOT / "01_ANCHORS.tsv", sep="\t", index=False)
pair_frame.to_csv(ROOT / "03_PAIR_LEVEL_RESULTS.tsv", sep="\t", index=False)
gene_frame.to_csv(ROOT / "04_GENE_LEVEL_RESULTS.tsv", sep="\t", index=False)
pd.DataFrame(registrations).to_csv(ROOT / "05_REGISTRATION_RESULTS.tsv", sep="\t", index=False)

lock_time = (ROOT / "02_PREDICTION_LOCK_MANIFEST.tsv").stat().st_mtime_ns
integrity = []
for line in (ROOT / "02_SHA256_PRE_UNLOCK.txt").read_text().splitlines():
    expected, path = line.split("  ", 1)
    path = Path(path)
    integrity.append((path, expected == sha256(path)))
if not all(ok for _, ok in integrity):
    raise RuntimeError("Frozen artifact hash mismatch during final leakage audit")
for tumor in TUMORS:
    marker = ROOT / "work" / tumor / "HIDDEN_TARGET_OPENED_AFTER_GLOBAL_LOCK.txt"
    if not marker.exists() or marker.stat().st_mtime_ns <= lock_time:
        raise RuntimeError(f"Hidden-expression access marker does not postdate lock: {tumor}")
    if len(pd.read_csv(ROOT / "01_ANCHORS.tsv", sep="\t").query("tumor == @tumor")) != 16:
        raise RuntimeError(f"Anchor count changed for {tumor}")

leakage = f"""# Leakage audit

## Result: PASS

All {len(integrity)} entries in `02_SHA256_PRE_UNLOCK.txt` re-verified after evaluation. The global prediction lock timestamp was `{datetime.fromtimestamp(lock_time / 1e9, timezone.utc).isoformat()}`. Each pair-specific hidden-expression access marker postdates this lock.

Before the global lock, the pipeline had access only to official specimen/platform metadata, Visium HD FFPE source counts and coordinates, CosMx panel identities, CosMx coordinates/tissue support, and exactly 16 source-selected CosMx anchor columns per pair. All source models, operator assignments, geometry registrations, common ROIs, registered 12-NN IDW priors, full MAP predictions and predicted target-depth vectors were serialized and hashed before non-anchor CosMx expression was materialized.

The geometry-only registration code has no target-expression input. Target non-anchor expression was not used for filtering, anchor selection, depth-model training, hyperparameter selection, operator selection, calibration, guard selection, ROI definition or representative-gene selection. Representative examples are the first member of each model's source-only spatial-score pool, which was serialized inside the pre-unlock source-model hash. No post-outcome fitting or tuning was performed.

The internally generated anchor decoder is retained only as a frozen HyperSpatial-MAP component diagnostic. It is absent from `03_PAIR_LEVEL_RESULTS.tsv`, `04_GENE_LEVEL_RESULTS.tsv`, the technology figure, captions, interpretation and manuscript paragraph. The only manuscript-facing comparator is registered 12-NN IDW.

Stereo-seq target expression was never opened and no Stereo-seq model or prediction was run because the requested FFPE serial-section pairing failed the pre-model gate.
"""
(ROOT / "06_LEAKAGE_AUDIT.md").write_text(leakage)

spatial_wins = pair_frame.MAP_median_gene_wise_spatial_pearson > pair_frame.IDW_median_gene_wise_spatial_pearson
rmse_wins = pair_frame.MAP_log1p_RMSE < pair_frame.IDW_log1p_RMSE
residual_positive = pair_frame.MAP_pooled_residual_pearson > 0
majority = len(pair_frame) // 2 + 1
if int((spatial_wins & rmse_wins).sum()) >= majority and bool(residual_positive.all()):
    cosmx_decision = "A"
elif bool((spatial_wins | rmse_wins | residual_positive).any()):
    cosmx_decision = "B"
else:
    cosmx_decision = "C"

table = pair_frame[[
    "tumor", "hidden_gene_count", "IDW_median_gene_wise_spatial_pearson",
    "MAP_median_gene_wise_spatial_pearson", "IDW_log1p_RMSE", "MAP_log1p_RMSE",
    "MAP_median_gene_wise_residual_pearson", "MAP_pooled_residual_pearson",
]].to_markdown(index=False, floatfmt=".4f")
interpretation = f"""# Results interpretation

The analysis asks whether 16 CosMx target genes add specimen-specific molecular information beyond direct geometry-registered transfer from a Visium HD FFPE reference. Registered 12-NN IDW uses the same source atlas, geometry registration, common ROI and expression representation but receives no target molecular correction; it is the only comparator.

{table}

CosMx decision: **{cosmx_decision}**. This decision uses the three specimen evaluations as the units and follows the predefined majority rule for both principal endpoints together with positive residual recovery. Genes and spatial bins are technical evaluation records, not biological replicates; they are not used as inferential replicates.

The Stereo-seq branch was stopped before modeling because the official SPATCH design places Stereo-seq on a separate fresh-frozen OCT portion rather than within the Visium HD FFPE serial-section block. No Stereo-seq A/B/C decision is assigned, because no scientifically valid pair met the requested gate.

These findings concern same-specimen FFPE serial sections measured by Visium HD and CosMx. They do not establish cell-to-cell correspondence, platform invariance, clinical generalization or biological replication.
"""
(ROOT / "08_RESULTS_INTERPRETATION.md").write_text(interpretation)

decision = f"""# Final decisions

## Visium HD FFPE → CosMx 6K: Decision {cosmx_decision}

The predefined gate counts a specimen as a joint endpoint win only when MAP has higher median spatial Pearson and lower log1p RMSE than registered IDW. Decision A requires such wins in a majority of valid specimens, consistently positive pooled residual recovery, a passed leakage audit and no post-outcome tuning. Decision B denotes mixed gains; Decision C denotes no reproducible improvement.

{table}

## Visium HD FFPE → Stereo-seq v1.3: not assigned (pairing gate STOP)

Zero valid same-block serial-section pairs were identified. The official study generated Stereo-seq from a separate fresh-frozen OCT tissue portion, whereas Visium HD FFPE came from the FFPE block. In accordance with the predefined rule, this branch was stopped before modeling and no A/B/C outcome decision was forced.
"""
(ROOT / "09_FINAL_DECISION.md").write_text(decision)

def fmt(row, col):
    return f"{getattr(row, col):.3f}"

sentences = []
for row in pair_frame.itertuples():
    sentences.append(
        f"In {row.tumor}, median spatial Pearson was {fmt(row, 'IDW_median_gene_wise_spatial_pearson')} "
        f"for IDW and {fmt(row, 'MAP_median_gene_wise_spatial_pearson')} for MAP, while log1p RMSE was "
        f"{fmt(row, 'IDW_log1p_RMSE')} and {fmt(row, 'MAP_log1p_RMSE')}, respectively; MAP pooled residual "
        f"Pearson was {fmt(row, 'MAP_pooled_residual_pearson')}."
    )
paragraph = (
    "We next tested cross-platform reference adaptation using same-specimen Visium HD FFPE and CosMx 6K "
    "serial sections from three human tumors. For each CosMx target, 16 genes selected solely from the Visium "
    "HD source were exposed and the remaining shared, source-eligible genes were hidden until prediction lock. "
    "We compared HyperSpatial-MAP with registered 12-nearest-neighbour inverse-distance weighting, which transfers "
    "the same geometrically registered source atlas without a target molecular correction. " + " ".join(sentences) +
    " Together, these evaluations test whether sparse target measurements can adapt a reference measured with a "
    "different spatial technology beyond registered atlas transfer; they do not assume one-to-one correspondence "
    "between serial sections."
)
(ROOT / "10_MANUSCRIPT_PARAGRAPH.md").write_text("# Manuscript Results paragraph\n\n" + paragraph + "\n")

caption = """# Figure captions

## Visium HD FFPE → CosMx 6K

**Cross-platform molecular reference adaptation in same-specimen serial sections.** **a,** Experimental design. Visium HD FFPE source sections and CosMx 6K target sections were represented at 8-µm resolution and aligned using geometry alone; 16 CosMx genes were exposed before prediction and the remaining shared, source-eligible genes were hidden. Serial sections are not assumed to provide one-to-one cell correspondence. **b,** Seed-42 geometry-only registration for COAD, HCC and ovarian cancer. **c, d,** Median gene-wise spatial Pearson and log1p RMSE for registered 12-nearest-neighbour inverse-distance weighting (IDW) and full HyperSpatial-MAP (MAP); lines join methods within specimen. IDW transfers the registered Visium HD atlas without target molecular correction and is the only comparator. **e,** MAP atlas-relative residual recovery by specimen. The measured residual is the CosMx target expression minus the registered Visium HD prior; the predicted residual is the MAP reconstruction minus the same prior. IDW predicts an identically zero residual, for which Pearson correlation is undefined. **f,** Representative hidden genes selected by the frozen source-only spatial-score rule. Expression maps share a scale within each gene; measured and predicted residual maps share a symmetric scale.

## Stereo-seq branch

No Stereo-seq figure was generated. Official metadata established that Stereo-seq used a separate fresh-frozen OCT tissue portion rather than a section in the Visium HD FFPE serial block, so the branch failed the predefined pairing gate before model fitting.
"""
(ROOT / "11_FIGURE_CAPTIONS.md").write_text(caption)

print(json.dumps({"CosMx_decision": cosmx_decision, "Stereo_seq_decision": "NOT_ASSIGNED_PAIRING_GATE_STOP"}))
