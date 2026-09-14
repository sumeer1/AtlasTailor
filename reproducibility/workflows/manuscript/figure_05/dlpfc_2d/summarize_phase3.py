#!/usr/bin/env python3
"""Summarize frozen DLPFC predictions after Phase 3 evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAP = "hyperspatial_map_calibrated_fusion"
IDW = "registered_idw_k12"


def mean_finite(values):
    values = np.asarray(values, float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    components = pd.read_csv(root / "07_tables" / "all_component_metrics.tsv", sep="\t")
    genes = pd.read_csv(root / "07_tables" / "all_gene_map_vs_idw.tsv", sep="\t")
    lock = json.loads((root / "02_frozen_protocol" / "PHASE2_PROTOCOL_LOCK.json").read_text())
    direction_rows = []
    anchor_frames = []
    guard_rows = []
    predictor_rows = []

    for direction in lock["directions"]:
        comp = components[components.direction == direction]
        gene = genes[genes.direction == direction]
        map_comp = comp[comp.method == MAP]
        idw_comp = comp[comp.method == IDW]
        per_gene = gene.groupby(["gene_id", "gene_name", "guard_supported"], as_index=False).agg({
            "map_spatial_pearson": "mean",
            "idw_spatial_pearson": "mean",
            "map_minus_idw_spatial_pearson": "mean",
            "map_residual_spatial_pearson": "mean",
            "map_log1p_rmse": "mean",
            "idw_log1p_rmse": "mean",
            "idw_minus_map_log1p_rmse": "mean",
        })
        finite_spatial = np.isfinite(per_gene.map_spatial_pearson) & np.isfinite(per_gene.idw_spatial_pearson)
        improved_spatial = finite_spatial & (
            per_gene.map_spatial_pearson > per_gene.idw_spatial_pearson
        )
        finite_rmse = np.isfinite(per_gene.map_log1p_rmse) & np.isfinite(per_gene.idw_log1p_rmse)
        improved_rmse = finite_rmse & (per_gene.map_log1p_rmse < per_gene.idw_log1p_rmse)
        fit_dir = root / "03_pairs" / direction / "source_fit"
        request = json.loads((fit_dir / "anchor_request.json").read_text())
        anchor = pd.read_csv(fit_dir / "anchors.tsv", sep="\t")
        anchor["source_section"] = request["source"]
        anchor["target_section"] = request["target"]
        anchor["donor"] = request["donor"]
        anchor["adjacent_pair"] = request["adjacent_pair"]
        anchor_frames.append(anchor)
        eligible = int(request["eligible_nonanchor_gene_count"])
        supported = int(request["source_defined_predictable_nonanchor_count"])
        model = np.load(fit_dir / "source_model.npz", allow_pickle=False)
        eval_indices = model["evaluation_indices"].astype(np.int64)
        support_mask = model["predictable"].astype(bool)[eval_indices]
        fusion_selection = model["fusion_selection"].astype(np.int64)[eval_indices]
        fusion_names = request["fusion_candidate_names"]
        assignments = []
        for is_supported, selection in zip(support_mask, fusion_selection):
            if not is_supported:
                assignments.append("IDW fallback")
                continue
            name = fusion_names[int(selection)]
            if name.endswith("w1.00"):
                assignments.append("anchor-only")
            elif name.endswith("w0.00") and name.startswith("global"):
                assignments.append("global residual")
            elif name.endswith("w0.00") and name.startswith("local"):
                assignments.append("local residual")
            else:
                assignments.append("calibrated mixture")
        for predictor, count in pd.Series(assignments).value_counts().items():
            predictor_rows.append({
                "direction": direction, "donor": request["donor"],
                "adjacent_pair": request["adjacent_pair"],
                "selected_predictor": predictor, "gene_count": int(count),
                "fraction_of_eligible_nonanchors": float(count / eligible),
            })
        guard_rows.append({
            "direction": direction, "donor": request["donor"],
            "adjacent_pair": request["adjacent_pair"],
            "eligible_nonanchor_genes": eligible,
            "guard_supported_genes": supported,
            "idw_fallback_genes": eligible - supported,
            "guard_supported_fraction": supported / eligible,
            "idw_fallback_fraction": (eligible - supported) / eligible,
        })
        direction_rows.append({
            "direction": direction, "source_section": request["source"],
            "target_section": request["target"], "donor": request["donor"],
            "adjacent_pair": request["adjacent_pair"],
            "eligible_nonanchor_genes": eligible,
            "guard_supported_genes": supported,
            "map_residual_spatial_pearson": mean_finite(map_comp.residual_spatial_pearson_median),
            "idw_residual_spatial_pearson": np.nan,
            "map_all_gene_spatial_pearson": mean_finite(map_comp.all_gene_spatial_pearson_median),
            "idw_all_gene_spatial_pearson": mean_finite(idw_comp.all_gene_spatial_pearson_median),
            "map_minus_idw_spatial_pearson": (
                mean_finite(map_comp.all_gene_spatial_pearson_median)
                - mean_finite(idw_comp.all_gene_spatial_pearson_median)
            ),
            "paired_gene_median_map_minus_idw_spatial_pearson": float(
                np.nanmedian(per_gene.map_minus_idw_spatial_pearson)
            ),
            "map_log1p_rmse": mean_finite(map_comp.log1p_rmse),
            "idw_log1p_rmse": mean_finite(idw_comp.log1p_rmse),
            "idw_minus_map_log1p_rmse": (
                mean_finite(idw_comp.log1p_rmse) - mean_finite(map_comp.log1p_rmse)
            ),
            "paired_gene_median_idw_minus_map_log1p_rmse": float(
                np.nanmedian(per_gene.idw_minus_map_log1p_rmse)
            ),
            "map_minus_idw_auprc_enrichment": mean_finite(map_comp.auprc_enrichment_median) - mean_finite(idw_comp.auprc_enrichment_median),
            "map_minus_idw_dice": mean_finite(map_comp.prevalence_matched_dice_median) - mean_finite(idw_comp.prevalence_matched_dice_median),
            "idw_minus_map_centroid_error": mean_finite(idw_comp.centroid_error_median) - mean_finite(map_comp.centroid_error_median),
            "idw_minus_map_spatial_emd_2d": mean_finite(idw_comp.spatial_emd_2d_median) - mean_finite(map_comp.spatial_emd_2d_median),
            "spatial_pearson_improved_genes": int(improved_spatial.sum()),
            "spatial_pearson_comparable_genes": int(finite_spatial.sum()),
            "spatial_pearson_improved_fraction": float(improved_spatial.sum() / max(finite_spatial.sum(), 1)),
            "guard_supported_spatial_pearson_improved_genes": int(
                (improved_spatial & per_gene.guard_supported).sum()
            ),
            "guard_supported_spatial_pearson_improved_fraction": float(
                (improved_spatial & per_gene.guard_supported).sum() / max(supported, 1)
            ),
            "rmse_improved_genes": int(improved_rmse.sum()),
            "rmse_comparable_genes": int(finite_rmse.sum()),
            "rmse_improved_fraction": float(improved_rmse.sum() / max(finite_rmse.sum(), 1)),
        })

    directions = pd.DataFrame(direction_rows)
    guards = pd.DataFrame(guard_rows)
    pairs = directions.groupby(["donor", "adjacent_pair"], as_index=False).agg({
        "map_residual_spatial_pearson": "mean",
        "map_minus_idw_spatial_pearson": "mean",
        "paired_gene_median_map_minus_idw_spatial_pearson": "mean",
        "idw_minus_map_log1p_rmse": "mean",
        "paired_gene_median_idw_minus_map_log1p_rmse": "mean",
        "map_minus_idw_auprc_enrichment": "mean",
        "map_minus_idw_dice": "mean",
        "idw_minus_map_centroid_error": "mean",
        "idw_minus_map_spatial_emd_2d": "mean",
        "spatial_pearson_improved_fraction": "mean",
        "rmse_improved_fraction": "mean",
    })
    pairs["directions"] = pairs.adjacent_pair.map(
        directions.groupby("adjacent_pair").direction.apply(lambda x: ";".join(x))
    )
    donors = pairs.groupby("donor", as_index=False).agg({
        "map_residual_spatial_pearson": "mean",
        "map_minus_idw_spatial_pearson": "mean",
        "paired_gene_median_map_minus_idw_spatial_pearson": "mean",
        "idw_minus_map_log1p_rmse": "mean",
        "paired_gene_median_idw_minus_map_log1p_rmse": "mean",
        "map_minus_idw_auprc_enrichment": "mean",
        "map_minus_idw_dice": "mean",
        "idw_minus_map_centroid_error": "mean",
        "idw_minus_map_spatial_emd_2d": "mean",
        "spatial_pearson_improved_fraction": "mean",
        "rmse_improved_fraction": "mean",
    })
    tables = root / "07_tables"
    directions.to_csv(tables / "direction_summary.tsv", sep="\t", index=False)
    pairs.to_csv(tables / "reciprocal_pair_summary.tsv", sep="\t", index=False)
    donors.to_csv(tables / "donor_summary.tsv", sep="\t", index=False)
    guards.to_csv(tables / "guard_fallback_summary.tsv", sep="\t", index=False)
    pd.DataFrame(predictor_rows).to_csv(
        tables / "selected_predictor_summary.tsv", sep="\t", index=False
    )
    pd.concat(anchor_frames, ignore_index=True).to_csv(
        tables / "source_selected_anchors.tsv", sep="\t", index=False
    )

    pair_positive = int((pairs.map_minus_idw_spatial_pearson > 0).sum())
    pair_residual_positive = int((pairs.map_residual_spatial_pearson > 0).sum())
    donor_positive = int((donors.map_minus_idw_spatial_pearson > 0).sum())
    residual_positive = int((directions.map_residual_spatial_pearson > 0).sum())
    decision = "PASS" if pair_positive >= 5 and pair_residual_positive == 6 and donor_positive == 3 else (
        "PARTIAL" if pair_positive >= 3 and donor_positive >= 2 else "FAIL"
    )
    strongest = directions.loc[directions.map_minus_idw_spatial_pearson.idxmax()]
    weakest = directions.loc[directions.map_minus_idw_spatial_pearson.idxmin()]
    overall = {
        "decision": decision,
        "primary_interpretation": "2D external generalization of the atlas-adaptation framework",
        "positive_reciprocal_pairs": pair_positive,
        "total_reciprocal_pairs": 6,
        "positive_residual_recovery_pairs": pair_residual_positive,
        "positive_donors": donor_positive,
        "total_donors": 3,
        "directions_with_positive_map_residual_recovery": residual_positive,
        "total_directions": 12,
        "mean_pair_map_minus_idw_spatial_pearson": float(pairs.map_minus_idw_spatial_pearson.mean()),
        "mean_pair_map_residual_spatial_pearson": float(pairs.map_residual_spatial_pearson.mean()),
        "total_gene_direction_records": int(directions.eligible_nonanchor_genes.sum()),
        "spatial_pearson_improved_gene_direction_records": int(directions.spatial_pearson_improved_genes.sum()),
        "spatial_pearson_comparable_gene_direction_records": int(directions.spatial_pearson_comparable_genes.sum()),
        "spatial_pearson_improved_fraction": float(
            directions.spatial_pearson_improved_genes.sum() /
            max(directions.spatial_pearson_comparable_genes.sum(), 1)
        ),
        "guard_supported_spatial_pearson_improved_records": int(
            directions.guard_supported_spatial_pearson_improved_genes.sum()
        ),
        "guard_supported_spatial_pearson_improved_fraction": float(
            directions.guard_supported_spatial_pearson_improved_genes.sum()
            / max(directions.guard_supported_genes.sum(), 1)
        ),
        "rmse_improved_gene_direction_records": int(directions.rmse_improved_genes.sum()),
        "rmse_comparable_gene_direction_records": int(directions.rmse_comparable_genes.sum()),
        "rmse_improved_fraction": float(
            directions.rmse_improved_genes.sum() /
            max(directions.rmse_comparable_genes.sum(), 1)
        ),
        "mean_guard_supported_fraction": float(guards.guard_supported_fraction.mean()),
        "mean_idw_fallback_fraction": float(guards.idw_fallback_fraction.mean()),
        "strongest_direction_by_spatial_effect": strongest.direction,
        "strongest_direction_effect": float(strongest.map_minus_idw_spatial_pearson),
        "weakest_direction_by_spatial_effect": weakest.direction,
        "weakest_direction_effect": float(weakest.map_minus_idw_spatial_pearson),
        "substantive_deviation_from_frozen_map": False,
    }
    (tables / "overall_summary.json").write_text(json.dumps(overall, indent=2) + "\n")

    colors = {"Br5292": "#3B82B8", "Br5595": "#D9893D", "Br8100": "#6C8E55"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    ax = axes[0, 0]
    x = np.arange(len(directions))
    ax.bar(x, directions.map_minus_idw_spatial_pearson, color=[colors[d] for d in directions.donor])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(directions.direction.str.replace("_to_", "→"), rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("MAP − IDW spatial Pearson")
    ax.set_title("a  Directional target performance")

    ax = axes[0, 1]
    xp = np.arange(len(pairs))
    ax.bar(xp, pairs.map_minus_idw_spatial_pearson, color=[colors[d] for d in pairs.donor])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xp)
    ax.set_xticklabels(pairs.adjacent_pair, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Reciprocal mean effect")
    ax.set_title("b  Six adjacent-section pairs")

    ax = axes[1, 0]
    ax.scatter(
        directions.map_residual_spatial_pearson,
        directions.map_minus_idw_spatial_pearson,
        c=[colors[d] for d in directions.donor], s=55, edgecolor="white", linewidth=0.6,
    )
    for row in directions.itertuples():
        ax.annotate(row.direction.replace("_to_", "→"),
                    (row.map_residual_spatial_pearson, row.map_minus_idw_spatial_pearson),
                    xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("MAP target-specific residual Pearson")
    ax.set_ylabel("MAP − IDW spatial Pearson")
    ax.set_title("c  Residual recovery and atlas-transfer effect")

    ax = axes[1, 1]
    ax.bar(x - 0.18, directions.spatial_pearson_improved_fraction, width=0.36,
           color="#4C78A8", label="Spatial Pearson")
    ax.bar(x + 0.18, directions.rmse_improved_fraction, width=0.36,
           color="#72B7B2", label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(directions.direction.str.replace("_to_", "→"), rotation=55, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of eligible non-anchor genes improved")
    ax.set_title("d  Gene-level effects")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Frozen HyperSpatial-MAP principles in external 2D human DLPFC", fontsize=14)
    fig.text(0.5, 0.01,
             "Six adjacent-section pairs; directions are technical evaluations, pairs are paired units, donors are biological sources.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    figures = root / "06_figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(figures / f"DLPFC_2D_external_generalization.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(overall))


if __name__ == "__main__":
    main()
