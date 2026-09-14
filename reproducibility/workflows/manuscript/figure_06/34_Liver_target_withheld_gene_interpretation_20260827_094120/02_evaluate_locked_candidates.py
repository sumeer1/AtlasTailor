#!/usr/bin/env python3
"""Stage 2: evaluate only the already SHA256-locked candidate genes."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MASLD = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
PSC = PROJECT / "PI_revision/28_Liver_PSC_MAP_20260825_175802"
AH = PROJECT / "PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254"
EXPECTED_LOCK_SHA256 = "4d23e9a3c3079ef8cc78d0fe0d8200bdedc80da161a0f0f6ecb7cc4c60dcc0f1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lock_path = ROOT / "CANDIDATE_GENE_LOCK.tsv"
    observed = sha256(lock_path)
    if observed != EXPECTED_LOCK_SHA256:
        raise RuntimeError(f"Candidate lock mismatch: {observed}")
    lock = pd.read_csv(lock_path, sep="\t")

    sources = {
        "MASLD": (MASLD / "05_GENE_LEVEL_RESULTS_MASLD.tsv", "biopsy", "33 biopsies / 32-patient cohort"),
        "PSC": (PSC / "05_GENE_LEVEL_RESULTS.tsv", "sample", "4 sections / 1 patient"),
        "AH": (AH / "05_GENE_LEVEL_RESULTS.tsv", "patient", "2 independent patients"),
    }

    per_unit_parts = []
    for disease, (path, unit_col, _) in sources.items():
        wanted = lock.loc[lock["disease"] == disease, "gene"].tolist()
        table = pd.read_csv(path, sep="\t")
        table = table[table["gene"].isin(wanted)].copy()
        observed_genes = set(table["gene"])
        if observed_genes != set(wanted):
            raise RuntimeError(f"{disease}: missing locked genes {set(wanted) - observed_genes}")
        table["disease"] = disease
        table["biological_unit"] = table[unit_col]
        table["delta_spatial_Pearson"] = table["map_spatial_pearson"] - table["idw_spatial_pearson"]
        table["delta_RMSE"] = table["map_log1p_rmse"] - table["idw_log1p_rmse"]
        table = table.merge(
            lock[["disease", "gene", "predeclared_program", "biological_reason"]],
            on=["disease", "gene"], how="left", validate="many_to_one"
        )
        table = table.rename(columns={
            "predeclared_program": "program",
            "biological_reason": "biological_role",
            "idw_spatial_pearson": "IDW_spatial_Pearson",
            "map_spatial_pearson": "MAP_spatial_Pearson",
            "idw_log1p_rmse": "IDW_RMSE",
            "map_log1p_rmse": "MAP_RMSE",
            "map_residual_pearson": "residual_Pearson",
        })
        keep = [
            "disease", "biological_unit", "gene", "program", "biological_role",
            "IDW_spatial_Pearson", "MAP_spatial_Pearson", "delta_spatial_Pearson",
            "IDW_RMSE", "MAP_RMSE", "delta_RMSE", "residual_Pearson",
        ]
        per_unit_parts.append(table[keep])

    per_unit = pd.concat(per_unit_parts, ignore_index=True)
    per_unit.to_csv(ROOT / "TARGET_WITHHELD_GENE_RESULTS_PER_UNIT.tsv", sep="\t", index=False)

    summary_rows = []
    example_rows = []
    for row in lock.itertuples(index=False):
        x = per_unit[(per_unit["disease"] == row.disease) & (per_unit["gene"] == row.gene)].copy()
        n = len(x)
        spatial_wins = int((x["delta_spatial_Pearson"] > 0).sum())
        rmse_wins = int((x["delta_RMSE"] < 0).sum())
        residual_positive = int((x["residual_Pearson"].fillna(-np.inf) > 0).sum())
        med = x[[
            "IDW_spatial_Pearson", "MAP_spatial_Pearson", "delta_spatial_Pearson",
            "IDW_RMSE", "MAP_RMSE", "delta_RMSE", "residual_Pearson"
        ]].median(numeric_only=True)
        # The summary-table deltas follow the requested literal definition:
        # aggregate MAP minus aggregate IDW. Per-unit paired deltas and their
        # distribution remain available in the per-unit table.
        med["delta_spatial_Pearson"] = med["MAP_spatial_Pearson"] - med["IDW_spatial_Pearson"]
        med["delta_RMSE"] = med["MAP_RMSE"] - med["IDW_RMSE"]
        unit_label = sources[row.disease][2]
        consistency = (
            f"spatial MAP>IDW {spatial_wins}/{n}; RMSE MAP<IDW {rmse_wins}/{n}; "
            f"positive residual r {residual_positive}/{n}"
        )
        notes = (
            "Median across biopsy evaluations; cohort has 32 patients and one repeated biopsy; "
            "the deposited mapping does not permit exact patient-balanced collapse."
            if row.disease == "MASLD" else
            "Four sections from one patient; section-level consistency is not patient replication."
            if row.disease == "PSC" else
            "Two independent patients reported separately and summarized with an equal-weight median."
        )
        summary_rows.append({
            "disease": row.disease,
            "gene": row.gene,
            "program": row.predeclared_program,
            "biological_role": row.biological_reason,
            "number_of_biological_units": unit_label,
            **{k: med[k] for k in med.index},
            "consistency_across_units": consistency,
            "notes": notes,
        })

        all_medians_favorable = (
            med["delta_spatial_Pearson"] > 0
            and med["delta_RMSE"] < 0
            and med["residual_Pearson"] > 0
        )
        strict_majority = n // 2 + 1
        majority_consistent = spatial_wins >= strict_majority and rmse_wins >= strict_majority and residual_positive >= strict_majority
        appropriate = bool(all_medians_favorable and majority_consistent)
        if appropriate:
            recovery = (
                f"median spatial r {med['IDW_spatial_Pearson']:.3f}→{med['MAP_spatial_Pearson']:.3f}; "
                f"RMSE {med['IDW_RMSE']:.3f}→{med['MAP_RMSE']:.3f}; "
                f"median residual r={med['residual_Pearson']:.3f}"
            )
            reason = "All three median endpoints favorable with majority consistency across available units."
        else:
            recovery = (
                f"mixed/limited: spatial MAP>IDW {spatial_wins}/{n}; "
                f"RMSE MAP<IDW {rmse_wins}/{n}; positive residual r {residual_positive}/{n}; "
                f"median residual r={med['residual_Pearson']:.3f}"
            )
            reason = "Retained transparently because one or more predefined recovery criteria were not met."
        example_rows.append({
            "disease": row.disease,
            "gene": row.gene,
            "program": row.predeclared_program,
            "why_biologically_relevant": row.biological_reason,
            "target_withheld_yes_no": "yes",
            "MAP_recovery_summary": recovery,
            "appropriate_for_main_text_yes_no": "yes" if appropriate else "no",
            "reason": reason,
        })

    summary = pd.DataFrame(summary_rows)
    examples = pd.DataFrame(example_rows)
    summary.to_csv(ROOT / "TARGET_WITHHELD_GENE_RESULTS.tsv", sep="\t", index=False)
    examples.to_csv(ROOT / "MANUSCRIPT_GENE_EXAMPLES.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
