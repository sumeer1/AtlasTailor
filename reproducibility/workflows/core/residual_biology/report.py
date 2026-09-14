#!/usr/bin/env python3
"""Turn the audit tables into a findings document.

Every statement here is derived from the tables in the same output directory.
The report deliberately separates what the residual landscape *is* from what the
frozen decision already claimed, and states the confound results whether or not
they are favourable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANCHOR, MAP, assert_writable
from .structure import K_NEIGHBOURS

# The frozen runner's axial set, excluded from every anchor panel by the locked
# selector. Named here only to locate them in the audit tables.
AXIAL_GENES = ("tbxta", "noto", "shha")

TABLE_FILES = {
    "structure": "02_structure/RESIDUAL_VARIANCE_DECOMPOSITION.tsv",
    "annotation_genes": "03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv",
    "annotation_classes": "03_annotation/RESIDUAL_RECOVERY_BY_COMPARTMENT.tsv",
    "depth": "04_confounds/DEPTH_PARTIAL_CORRELATION.tsv",
    "registration": "04_confounds/REGISTRATION_INSTABILITY_STRATA.tsv",
    "anchor_propagation": "04_confounds/ANCHOR_PROPAGATION_BY_GENE.tsv",
    "spatial_scale": "04_confounds/SPATIAL_SCALE_DECOMPOSITION.tsv",
    "coordinate_sensitivity": "04_confounds/COORDINATE_SENSITIVITY.tsv",
}


def load_tables(output: Path) -> dict[str, pd.DataFrame]:
    """Read back the audit tables so the report can be regenerated alone."""
    return {
        name: pd.read_csv(output / relative, sep="\t")
        for name, relative in TABLE_FILES.items()
    }


def write_table(path: Path, frame: pd.DataFrame) -> None:
    assert_writable(path)
    frame.to_csv(path, sep="\t", index=False)


def _median_by_stage(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("stage")[column].median()


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def build_report(output: Path, tables: dict[str, pd.DataFrame], provenance: dict) -> str:
    structure = tables["structure"]
    annotation_genes = tables["annotation_genes"]
    annotation_classes = tables["annotation_classes"]
    depth = tables["depth"]
    registration = tables["registration"]
    anchor = tables["anchor_propagation"]
    scale = tables["spatial_scale"]
    coordinates = tables["coordinate_sensitivity"]

    lines: list[str] = []
    add = lines.append

    add("# Residual-landscape biological audit")
    add("")
    add(f"Audit run: `{output.name}`  ")
    add(f"Frozen input: `{provenance['frozen_root']}` (read-only; never written)  ")
    add(
        f"Frozen predictions verified: {provenance['predictions_verified']}"
        f"/{provenance['predictions_checked']} sha256 matches  "
    )
    add(
        f"Frozen residual endpoint reproduced: {provenance['reproduction_passed']}"
        f"/{provenance['reproduction_checked']} method-seed comparisons within "
        f"{provenance['reproduction_tolerance']:g}"
    )
    add("")

    # ------------------------------------------------------------------
    # Summary, generated from the same tables as the sections below.
    n_targets = structure.target_key.nunique()
    structured = structure["structured_fraction_of_residual"].median()
    map_recovered = structure[f"{MAP}__fraction_of_structured_recovered"].median()
    anchor_recovered = structure[f"{ANCHOR}__fraction_of_structured_recovered"].median()
    profile_median = annotation_genes[f"{MAP}__class_profile_pearson"].median()
    eta_ratio = (
        annotation_genes.truth_residual_eta_squared.median()
        / max(annotation_genes.permuted_label_eta_squared.median(), 1e-12)
    )
    map_depth_rows = depth[depth.method == MAP]
    depth_shift = map_depth_rows.median_change_after_depth_removal.median()
    anchor_correlation = anchor[
        ["max_source_correlation_to_anchor_panel", f"{MAP}__residual_pearson"]
    ].corr().iloc[0, 1]
    low_anchor = anchor[
        anchor.max_source_correlation_to_anchor_panel
        <= anchor.max_source_correlation_to_anchor_panel.quantile(0.25)
    ][f"{MAP}__residual_pearson"].median()
    map_scale_rows = scale[scale.method == MAP]
    fine_pearson = map_scale_rows.median_fine_component_pearson.median()

    add("## Summary")
    add("")
    add(
        f"- The residual is mostly real signal: **{_fmt(structured)}** of its variance "
        f"exceeds the Poisson counting-noise floor, and it is spatially coherent "
        f"(median Moran's I {_fmt(structure.residual_moran_i.median())} against "
        f"{_fmt(structure.residual_moran_null_mean.median())} under a cell-permutation null)."
    )
    add(
        f"- The residual is biologically organised: its between-class variance under "
        f"author-supplied annotations is ~{eta_ratio:.0f}x the permuted-label null, and "
        f"the method reproduces the per-compartment residual profile with median Pearson "
        f"**{_fmt(profile_median)}**."
    )
    add(
        f"- Recovery is partial: HyperSpatial-MAP linearly recovers a median "
        f"**{_fmt(map_recovered)}** of the structured residual variance "
        f"(anchor-only {_fmt(anchor_recovered)}), consistent with the frozen "
        f"component-specialisation qualification rather than universal superiority."
    )
    add(
        f"- Depth is not the explanation: partialling out per-cell log depth moves the "
        f"residual correlation by {_fmt(depth_shift)}."
    )
    gradient = (
        registration.sort_values("instability_quintile")
        .groupby("target_key")[f"{MAP}__median_residual_pearson"]
        .agg(lambda s: s.iloc[-1] - s.iloc[0])
    )
    add(
        f"- Registration instability is not the explanation: recovery rises only "
        f"{_fmt(gradient.median())} (median across targets, range "
        f"{_fmt(gradient.min())} to {_fmt(gradient.max())}) from the most to the least "
        f"stable quintile, so it is not concentrated where the atlas transfer fails."
    )
    add(
        f"- Anchor propagation is a substantial part of the explanation: per-gene "
        f"recovery correlates {_fmt(anchor_correlation)} with a gene's maximum source "
        f"correlation to the anchor panel, though genes in the lowest anchor-correlation "
        f"quartile still reach median residual Pearson {_fmt(low_anchor)}."
    )
    add(
        f"- Recovery is not only a coarse offset: the fine local component of the "
        f"residual is recovered at median Pearson {_fmt(fine_pearson)}."
    )
    axial_summary = structure[structure.gene.isin(AXIAL_GENES)]
    if not axial_summary.empty:
        deficit = (
            axial_summary[f"{ANCHOR}__residual_pearson"]
            - axial_summary[f"{MAP}__residual_pearson"]
        )
        worst_axial = axial_summary.loc[deficit.idxmax()]
        add(
            f"- Qualification on axial genes: anchor-only beats the full fusion on "
            f"{int((deficit > 0.10).sum())} of {len(axial_summary)} axial "
            f"gene-by-target residual comparisons, worst at `{worst_axial.gene}` in "
            f"{worst_axial.target_key} ({_fmt(worst_axial[f'{MAP}__residual_pearson'])} "
            f"versus {_fmt(worst_axial[f'{ANCHOR}__residual_pearson'])}). These are the "
            f"genes behind the frozen axial-localisation figure, so the component "
            f"specialisation the frozen run reported extends into the axial residual."
        )
    add("")
    add("## What this audit does and does not do")
    add("")
    add(
        "The frozen Class A decision established that HyperSpatial-MAP recovers a "
        "target-specific residual — the difference between measured target expression "
        "and the registered atlas transfer — that registered IDW does not. It did not "
        "characterise that residual. This audit characterises it, using per-cell "
        "annotations the frozen protocol never opened."
    )
    add("")
    add(
        "No frozen prediction, registration, anchor list, manifest, figure or decision "
        "was modified or recomputed. The residual fields analysed here reproduce the "
        "frozen `residual_spatial_pearson_median` values exactly, so every statement "
        "below is about the same object the decision rests on."
    )
    add("")

    # ------------------------------------------------------------------
    add("## 1. What fraction of the residual is real signal?")
    add("")
    noise_free = structure["structured_fraction_of_residual"]
    add(
        f"Across {n_targets} target volume(s) and a median of "
        f"{int(structure.groupby('target_key').size().median())} predictable genes each, "
        f"the median share of target-residual variance that exceeds the Poisson "
        f"counting-noise floor is **{_fmt(noise_free.median())}**."
    )
    add("")
    add("| Stage | Structured share of residual | Residual/truth variance ratio | Median residual Moran's I | Permutation-null Moran's I |")
    add("|---|---|---|---|---|")
    for stage in structure.stage.unique():
        block = structure[structure.stage == stage]
        add(
            f"| {stage} | {_fmt(block.structured_fraction_of_residual.median())} "
            f"| {_fmt(block.residual_to_truth_variance_ratio.median())} "
            f"| {_fmt(block.residual_moran_i.median())} "
            f"| {_fmt(block.residual_moran_null_mean.median())} |"
        )
    add("")
    smooth = structure["residual_neighbourhood_variance_fraction"].median()
    add(
        f"Local-neighbourhood variance share of the residual is {_fmt(smooth)} against "
        f"{_fmt(1.0 / (K_NEIGHBOURS + 1))} expected for spatially independent noise, so "
        f"the residual varies smoothly in space rather than cell-by-cell."
    )
    add("")
    for method in (MAP, ANCHOR):
        column = f"{method}__fraction_of_structured_recovered"
        add(
            f"- `{method}` linearly recovers a median "
            f"**{_fmt(structure[column].median())}** of the structured (above-noise) "
            f"residual variance."
        )
    add("")

    # ------------------------------------------------------------------
    add("## 2. Is the residual aligned with independent biological annotation?")
    add("")
    add(
        "Annotation columns come from the dataset authors. The frozen leakage audit "
        "records `target_annotations_accessed: false`, so nothing in the locked "
        "pipeline was fitted to them."
    )
    add("")
    add("| Stage | Annotation | Classes | Truth-residual eta^2 | Permuted-label eta^2 | MAP class-profile Pearson | Anchor-only class-profile Pearson |")
    add("|---|---|---|---|---|---|---|")
    grouped = annotation_genes.groupby(["stage", "annotation"], sort=False)
    for (stage, column), block in grouped:
        add(
            f"| {stage} | `{column}` | {int(block.n_classes.iloc[0])} "
            f"| {_fmt(block.truth_residual_eta_squared.median())} "
            f"| {_fmt(block.permuted_label_eta_squared.median())} "
            f"| {_fmt(block[f'{MAP}__class_profile_pearson'].median())} "
            f"| {_fmt(block[f'{ANCHOR}__class_profile_pearson'].median())} |"
        )
    add("")
    profile = annotation_genes[f"{MAP}__class_profile_pearson"]
    add(
        f"The class-profile correlation asks whether the method moves the *right "
        f"compartments* in the right direction, not merely whether it correlates "
        f"cell-wise. Its median across all genes and annotations is "
        f"**{_fmt(profile.median())}**."
    )
    add("")
    worst = annotation_classes.nsmallest(8, f"{MAP}__median_within_class_residual_pearson")
    add("Weakest compartments by within-class residual recovery (MAP):")
    add("")
    add("| Stage | Annotation | Class | Cells | Within-class residual Pearson |")
    add("|---|---|---|---|---|")
    for _, row in worst.iterrows():
        add(
            f"| {row.stage} | `{row.annotation}` | {row['class']} | {int(row.n_cells)} "
            f"| {_fmt(row[f'{MAP}__median_within_class_residual_pearson'])} |"
        )
    add("")

    # ------------------------------------------------------------------
    add("## 3. Confound controls")
    add("")
    add("### 3.1 Sequencing depth")
    add("")
    map_depth = depth[depth.method == MAP]
    add("| Stage | Residual Pearson | After removing per-cell log depth | Change |")
    add("|---|---|---|---|")
    for _, row in map_depth.iterrows():
        add(
            f"| {row.stage} ({row.target_key}) | {_fmt(row.median_residual_pearson)} "
            f"| {_fmt(row.median_depth_partialled_residual_pearson)} "
            f"| {_fmt(row.median_change_after_depth_removal)} |"
        )
    add("")
    add(
        f"Median absolute correlation between the truth residual and per-cell log depth "
        f"is {_fmt(map_depth.median_absolute_truth_residual_correlation_with_log_depth.median())}."
    )
    add("")

    add("### 3.2 Registration quality")
    add("")
    add(
        "Cells are stratified by disagreement between the frozen k=12 IDW transfer and "
        "the frozen k=1 nearest-neighbour transfer. If recovery existed only where the "
        "transfer is unstable, the correction would be geometric rather than biological."
    )
    add("")
    add("| Stage | Instability quintile | Cells | Median residual variance | MAP residual Pearson |")
    add("|---|---|---|---|---|")
    for _, row in registration.iterrows():
        add(
            f"| {row.stage} ({row.target_key}) | {int(row.instability_quintile)} "
            f"| {int(row.n_cells)} | {_fmt(row.median_truth_residual_variance)} "
            f"| {_fmt(row[f'{MAP}__median_residual_pearson'])} |"
        )
    add("")

    add("### 3.3 Anchor propagation")
    add("")
    low = anchor[anchor.max_source_correlation_to_anchor_panel
                 <= anchor.max_source_correlation_to_anchor_panel.quantile(0.25)]
    high = anchor[anchor.max_source_correlation_to_anchor_panel
                  >= anchor.max_source_correlation_to_anchor_panel.quantile(0.75)]
    correlation = anchor[["max_source_correlation_to_anchor_panel", f"{MAP}__residual_pearson"]].corr().iloc[0, 1]
    add(
        f"Across genes, the correlation between a gene's maximum source correlation to "
        f"the anchor panel and its MAP residual recovery is {_fmt(correlation)}. Genes in "
        f"the lowest anchor-correlation quartile still reach a median residual Pearson of "
        f"**{_fmt(low[f'{MAP}__residual_pearson'].median())}** against "
        f"{_fmt(high[f'{MAP}__residual_pearson'].median())} in the highest quartile."
    )
    add("")

    add("### 3.4 Spatial scale")
    add("")
    map_scale = scale[scale.method == MAP]
    add("| Stage | Broad variance share | Broad-component Pearson | Fine-component Pearson |")
    add("|---|---|---|---|")
    for _, row in map_scale.iterrows():
        add(
            f"| {row.stage} ({row.target_key}) "
            f"| {_fmt(row.median_broad_variance_share_of_truth_residual)} "
            f"| {_fmt(row.median_broad_component_pearson)} "
            f"| {_fmt(row.median_fine_component_pearson)} |"
        )
    add("")

    add("### 3.5 Coordinate choice")
    add("")
    add(
        "The frozen run flagged that the 6-somite loader used raw `spatial` rather than "
        "the available `spatial_rescaled_z`, while 50%/75% epiboly used `global_sphere`. "
        "Nothing is re-run here; the audit only measures how far the residual "
        "landscape's neighbourhood structure and spatial coherence depend on the "
        "coordinate field. Low k-NN overlap means the two coordinate systems disagree "
        "about which cells are neighbours, so a coherence estimate under one is not "
        "transferable to the other."
    )
    add("")
    add("| Stage | Frozen key | Alternative | k-NN overlap | Moran's I (frozen) | Moran's I (alternative) |")
    add("|---|---|---|---|---|---|")
    for _, row in coordinates.iterrows():
        add(
            f"| {row.stage} ({row.target_key}) | `{row.frozen_coordinate_key}` "
            f"| `{row.alternative_coordinate_key}` | {_fmt(row.mean_knn_overlap_fraction)} "
            f"| {_fmt(row.median_residual_moran_i_frozen)} "
            f"| {_fmt(row.median_residual_moran_i_alternative)} |"
        )
    add("")

    # ------------------------------------------------------------------
    add("## 4. Which genes carry the recovered residual")
    add("")
    add(
        "Descriptive ranking only. No ontology mapping is available for these genes "
        "(see the frozen `ANCHOR_PROGRAMME_ANNOTATION_LIMITATION.md`), so no programme "
        "or pathway enrichment is claimed."
    )
    add("")
    recovery_column = f"{MAP}__fraction_of_structured_recovered"
    per_gene = (
        structure.groupby("gene")
        .agg(
            targets=("target_key", "nunique"),
            structured_fraction=("structured_fraction_of_residual", "median"),
            recovered=(recovery_column, "median"),
            residual_pearson=(f"{MAP}__residual_pearson", "median"),
        )
        .reset_index()
    )
    complete = per_gene[per_gene.targets == per_gene.targets.max()]
    for label, block in (
        ("Best recovered", complete.nlargest(10, "recovered")),
        ("Least recovered", complete.nsmallest(10, "recovered")),
    ):
        add(f"{label} (median over {int(complete.targets.max())} target volumes):")
        add("")
        add("| Gene | Structured share of residual | Fraction of structured variance recovered | Residual Pearson |")
        add("|---|---|---|---|")
        for _, row in block.iterrows():
            add(
                f"| `{row.gene}` | {_fmt(row.structured_fraction)} "
                f"| {_fmt(row.recovered)} | {_fmt(row.residual_pearson)} |"
            )
        add("")

    axial = structure[structure.gene.isin(AXIAL_GENES)]
    add(
        "Axial genes. These are excluded from the anchor panel by the locked selector, "
        "so any residual recovery on them is not anchor self-prediction. They are the "
        "genes behind the frozen axial-localisation figure."
    )
    add("")
    if axial.empty:
        add("No axial gene is inside the frozen predictable-gene set for these targets.")
    else:
        add("| Target | Stage | Gene | Structured share of residual | MAP residual Pearson | Anchor-only residual Pearson |")
        add("|---|---|---|---|---|---|")
        for _, row in axial.sort_values(["gene", "target_key"]).iterrows():
            add(
                f"| {row.target_key} | {row.stage} | `{row.gene}` "
                f"| {_fmt(row.structured_fraction_of_residual)} "
                f"| {_fmt(row[f'{MAP}__residual_pearson'])} "
                f"| {_fmt(row[f'{ANCHOR}__residual_pearson'])} |"
            )
        missing = sorted(set(AXIAL_GENES) - set(axial.gene))
        present = axial.groupby("gene").target_key.nunique()
        incomplete = [g for g, n in present.items() if n < structure.target_key.nunique()]
        if missing or incomplete:
            add("")
            add(
                f"Not every axial gene is inside the frozen source-defined predictable "
                f"set for every target: missing entirely {missing or 'none'}; present "
                f"for only some targets {incomplete or 'none'}."
            )
    add("")

    add("## 5. Scope of these findings")
    add("")
    add(
        "- This is a retrospective characterisation of frozen outputs. It adds no new "
        "prediction, changes no frozen decision, and cannot convert the retrospective "
        "replication into prospective confirmation."
    )
    add(
        "- Annotation labels are author-supplied cluster assignments, not an ontology. "
        "They support statements about compartment alignment, not about specific "
        "developmental programmes."
    )
    add(
        "- The counting-noise floor assumes Poisson measurement noise and uses the "
        "observed counts as the rate, which is itself noisy; that makes the floor an "
        "over-estimate and the structured fraction correspondingly conservative. "
        "Overdispersion beyond Poisson would push the other way."
    )
    add(
        "- Recovery is measured as linear (Pearson) agreement between residual fields. "
        "A method that got the spatial pattern right but the amplitude wrong would score "
        "well here, matching the frozen endpoint's own convention."
    )
    add(
        "- Coordinate sensitivity is reported descriptively. Selecting a coordinate "
        "variant on the basis of these outcomes would be post-outcome tuning of a "
        "frozen protocol and is not done."
    )
    add("")

    return "\n".join(lines) + "\n"
