#!/usr/bin/env python3
"""Stage 1: lock biologically selected genes without opening target outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
MASLD = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222_MASLD"
PSC = PROJECT / "PI_revision/28_Liver_PSC_MAP_20260825_175802"
AH = PROJECT / "PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254"
SOURCE_RUN = PROJECT / "PI_revision/26_HumanLiver_multidisease_MAP_20260825_110222"
MODEL = MASLD / "work/healthy_source_model_masld_panel.npz"
SOURCE_H5 = SOURCE_RUN / "data/healthy_live_donor/extracted/h5"
FROZEN_AUDIT = PROJECT / "PI_revision/32_Liver_multidisease_integrated_figure_v2_20260826_101842/REPRESENTATIVE_GENE_SELECTION_AUDIT.tsv"

# These choices were specified before loading any target metric or target
# expression. They span distinct predeclared biology and follow frozen
# healthy-source ranking within each biological program.
LOCKED = {
    "MASLD": [
        ("TAT", "hepatocyte_zonation", "periportal amino-acid metabolism and hepatocyte zonation"),
        ("SCD", "lipid_metabolic_reprogramming;steatosis_associated", "fatty-acid desaturation and steatosis-associated lipid synthesis"),
        ("DCN", "extracellular_matrix_fibrosis", "stromal extracellular-matrix organization and fibrotic remodeling"),
        ("NFKBIA", "inflammation", "NF-kappaB feedback and inflammatory signaling"),
    ],
    "PSC": [
        ("TAT", "hepatocyte_zonation", "periportal hepatocyte identity and zonation"),
        ("DCN", "portal_fibrotic", "portal stromal matrix and fibrotic remodeling"),
        ("KRT8", "ductular_reaction", "epithelial intermediate filament represented in ductular reaction"),
        ("TIMP1", "ecm_remodeling", "matrix metalloproteinase inhibition and extracellular-matrix turnover"),
    ],
    "AH": [
        ("CYP3A4", "metabolic_dysfunction", "zonated hepatocyte xenobiotic metabolism altered in liver injury"),
        ("DCN", "fibrosis_ecm", "stromal extracellular-matrix organization"),
        ("TUBB", "regenerative_response", "microtubule component represented in regenerative/proliferative response"),
        ("KRT8", "hepatocyte_injury_stress", "hepatocyte epithelial cytoskeleton and injury-associated remodeling"),
        ("NFKBIA", "inflammatory_immune_response", "NF-kappaB feedback and inflammatory signaling"),
    ],
}

PROGRAM_SOURCES = {
    "MASLD": MASLD / "scripts/02_verify_unlock_evaluate_masld.py",
    "PSC": PSC / "PSC_PROGRAMS.json",
    "AH": AH / "AH_PROGRAMS.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def source_prevalence(gene_ids: list[str]) -> np.ndarray:
    detected = np.zeros(len(gene_ids), dtype=np.int64)
    total = 0
    for donor in [f"M{i}" for i in range(1, 9)]:
        path = SOURCE_H5 / f"{donor}_filtered_feature_bc_matrix.h5"
        with h5py.File(path, "r") as handle:
            group = handle["matrix"]
            matrix = csc_matrix(
                (group["data"][:], group["indices"][:], group["indptr"][:]),
                shape=tuple(int(x) for x in group["shape"][:]),
            ).T.tocsr()
            ids = decode(group["features/id"][:])
        lookup = {gene_id: i for i, gene_id in enumerate(ids)}
        columns = np.asarray([lookup[x] for x in gene_ids], dtype=np.int64)
        detected += np.asarray((matrix[:, columns] > 0).sum(axis=0)).ravel()
        total += matrix.shape[0]
    return detected / total


def main() -> None:
    # Source-only rank table: produced previously without target outcomes.
    audit = pd.read_csv(FROZEN_AUDIT, sep="\t")
    if audit["target_outcomes_used"].astype(str).str.lower().ne("false").any():
        raise RuntimeError("Frozen source-only audit is not outcome-independent")

    model = np.load(MODEL, allow_pickle=True)
    names = model["gene_names"].astype(str)
    ids = model["gene_ids"].astype(str)
    evaluation = set(names[model["evaluation_indices"].astype(int)])
    name_to_id = dict(zip(names, ids))

    anchor_sets = {}
    hidden_sets = {}
    manifest_paths = {
        "MASLD": MASLD / "work/VLP115_A_C1/prediction_lock_manifest.json",
        "PSC": PSC / "work/PSC011_A1/prediction_lock_manifest.json",
        "AH": AH / "work/Sah73/prediction_lock_manifest.json",
    }
    for disease, path in manifest_paths.items():
        payload = json.loads(path.read_text())
        anchor_sets[disease] = set(payload["anchors"])
        hidden_sets[disease] = set(payload["hidden_gene_names"])
        if len(anchor_sets[disease]) != 16:
            raise RuntimeError(f"{disease}: anchor count is not 16")

    all_genes = []
    for disease in LOCKED:
        all_genes.extend(gene for gene, _, _ in LOCKED[disease])
    unique_genes = list(dict.fromkeys(all_genes))
    prevalence = dict(zip(unique_genes, source_prevalence([name_to_id[g] for g in unique_genes])))

    rows = []
    for disease, candidates in LOCKED.items():
        disease_audit = audit[audit["disease"] == disease].set_index("gene")
        for selection_rank, (gene, program, reason) in enumerate(candidates, 1):
            if gene not in disease_audit.index:
                raise RuntimeError(f"{disease}/{gene}: absent from frozen source-only eligible audit")
            record = disease_audit.loc[gene]
            if gene in anchor_sets[disease]:
                raise RuntimeError(f"{disease}/{gene}: candidate is an anchor")
            if gene not in hidden_sets[disease] or gene not in evaluation:
                raise RuntimeError(f"{disease}/{gene}: candidate is not in frozen hidden set")
            rows.append({
                "disease": disease,
                "gene": gene,
                "predeclared_program": program,
                "anchor_yes_no": "no",
                "hidden_target_yes_no": "yes",
                "source_detectability_or_prevalence": f"{prevalence[gene]:.6f}",
                "biological_reason": reason,
                "selection_rank": selection_rank,
                "selection_basis": (
                    f"predeclared program membership; frozen healthy-source score "
                    f"{float(record['source_only_score']):.6f}, disease-eligible source rank "
                    f"{int(record['source_rank_within_disease_eligible'])}; cross-program coverage; "
                    "no target prediction metric"
                ),
            })

    lock = pd.DataFrame(rows)
    lock_path = ROOT / "CANDIDATE_GENE_LOCK.tsv"
    lock.to_csv(lock_path, sep="\t", index=False)
    digest = sha256(lock_path)
    (ROOT / "CANDIDATE_GENE_LOCK.sha256").write_text(
        f"{digest}  CANDIDATE_GENE_LOCK.tsv\n"
    )

    anchors_text = []
    for disease in ["MASLD", "PSC", "AH"]:
        anchors_text.append(f"- {disease}: " + ", ".join(sorted(anchor_sets[disease])))
    candidate_text = []
    for disease in ["MASLD", "PSC", "AH"]:
        names_text = ", ".join(f"{g} ({p})" for g, p, _ in LOCKED[disease])
        candidate_text.append(f"- {disease}: {names_text}")
    md = f"""# Candidate selection and lock

## Chronology and information boundary

This file records Stage 1 of the target-withheld disease-gene interpretation. Candidate selection was completed and SHA256-locked **before any target MAP prediction metric, target truth map, IDW map, MAP map or target residual map for these candidates was loaded by the analysis workflow**. The lock digest is `{digest}`.

Candidates were drawn only from the exact disease programs predeclared in the canonical frozen analyses. Eligibility required absence from the 16 anchors, membership in the frozen hidden-gene list, measurement on the target feature axis and adequate detectability in the M1--M8 healthy source. Ranking used the pre-existing source-only score (`variance(log1p(source expression)) × prevalence`) and biological coverage across distinct predeclared programs. No target prediction outcome entered selection.

## Frozen anchors

{chr(10).join(anchors_text)}

## Locked candidates

{chr(10).join(candidate_text)}

## Program-definition provenance

- MASLD: `{PROGRAM_SOURCES['MASLD']}` (`PROGRAMS`, defined before target-outcome evaluation).
- PSC: `{PROGRAM_SOURCES['PSC']}`.
- AH: `{PROGRAM_SOURCES['AH']}`.

`source_detectability_or_prevalence` is the fraction of spots with non-zero raw counts across the eight frozen healthy live-donor Visium sections. The prior source-only audit is `{FROZEN_AUDIT}` and explicitly records `target_outcomes_used = False`.
"""
    (ROOT / "CANDIDATE_SELECTION.md").write_text(md)
    print(f"LOCKED {len(lock)} candidates: {digest}")


if __name__ == "__main__":
    main()
