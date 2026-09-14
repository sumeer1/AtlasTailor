#!/usr/bin/env python3
"""Run the residual-landscape biological audit over the six frozen targets.

Usage::

    python -m analysis.residual_landscape.run_audit
    python -m analysis.residual_landscape.run_audit --targets 6s_E1_to_E2

Outputs land in ``results_v10/hyperspatial_map_residual_landscape_<timestamp>``.
The frozen trees are opened read-only and verified by sha256 before use.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import annotation, confounds, structure
from .config import (
    FROZEN_ROOT,
    RUNS,
    SEEDS,
    assert_writable,
    new_output_dir,
)
from .io_frozen import load_target, verify_frozen_predictions
from .report import build_report, load_tables, write_table
from .residual_fields import (
    REPRODUCTION_TOLERANCE,
    build_fields,
    reproduction_check,
    seed_mean_truth,
)

# The seed used for the single-seed confound stratification. Registration is
# re-fit per seed, so the instability field is seed-specific; the residual
# fields it is compared against are seed-averaged, which is conservative.
REFERENCE_SEED = SEEDS[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="*", default=None,
        help="Target keys to audit (default: all six).",
    )
    parser.add_argument(
        "--timestamp", default=None,
        help="Override the output directory timestamp.",
    )
    parser.add_argument(
        "--skip-hash-verification", action="store_true",
        help="Skip sha256 verification of frozen predictions (2.4 GB of reads).",
    )
    parser.add_argument(
        "--report-only", type=Path, default=None,
        help="Rebuild the report from an existing audit directory's tables.",
    )
    return parser.parse_args(argv)


def regenerate_report(output: Path) -> int:
    """Rewrite the findings document from tables already on disk."""
    assert_writable(output)
    provenance = json.loads((output / "00_provenance/AUDIT_PROVENANCE.json").read_text())
    report = build_report(output, load_tables(output), provenance)
    (output / "05_report/RESIDUAL_LANDSCAPE_AUDIT.md").write_text(report)
    print(f"[audit] report -> {output / '05_report/RESIDUAL_LANDSCAPE_AUDIT.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report_only is not None:
        return regenerate_report(args.report_only)
    runs = [r for r in RUNS if args.targets is None or r.key in args.targets]
    if not runs:
        raise SystemExit(f"No frozen target matches {args.targets}")

    output = new_output_dir(args.timestamp)
    started = time.perf_counter()
    print(f"[audit] output -> {output}")

    verification: list[dict] = []
    if not args.skip_hash_verification:
        for run in runs:
            print(f"[audit] verifying frozen predictions for {run.key}")
            verification.extend(verify_frozen_predictions(run))
        failures = [r for r in verification if not r["matches"]]
        if failures:
            raise SystemExit(
                f"Frozen predictions changed since the run was sealed: {failures[:3]}"
            )

    reproduction: list[dict] = []
    structure_rows: list[dict] = []
    annotation_gene_rows: list[dict] = []
    annotation_class_rows: list[dict] = []
    depth_rows: list[dict] = []
    registration_rows: list[dict] = []
    anchor_rows: list[dict] = []
    scale_rows: list[dict] = []
    coordinate_rows: list[dict] = []

    for run in runs:
        print(f"[audit] loading {run.key}")
        data = load_target(run)

        for seed in SEEDS:
            reproduction.extend(reproduction_check(data, build_fields(data, seed)))

        truth_residual, predicted_residual = seed_mean_truth(data)
        print(
            f"[audit]   {data.n_cells} cells, {truth_residual.shape[1]} predictable genes, "
            f"coordinate key {data.coordinate_key}"
        )

        structure_rows.extend(structure.structure_table(data, truth_residual, predicted_residual))
        genes_rows, class_rows = annotation.annotation_table(
            data, truth_residual, predicted_residual
        )
        annotation_gene_rows.extend(genes_rows)
        annotation_class_rows.extend(class_rows)
        depth_rows.extend(confounds.depth_confound(data, truth_residual, predicted_residual))
        registration_rows.extend(
            confounds.registration_confound(data, truth_residual, predicted_residual, REFERENCE_SEED)
        )
        anchor_rows.extend(confounds.anchor_propagation(data, truth_residual, predicted_residual))
        scale_rows.extend(confounds.spatial_scale(data, truth_residual, predicted_residual))
        coordinate_rows.extend(confounds.coordinate_sensitivity(data, truth_residual))
        del data, truth_residual, predicted_residual

    reproduction_frame = pd.DataFrame(reproduction)
    if not bool(reproduction_frame.reproduced.all()):
        failed = reproduction_frame[~reproduction_frame.reproduced]
        write_table(output / "00_provenance/RESIDUAL_REPRODUCTION_CHECK.tsv", reproduction_frame)
        raise SystemExit(
            "Recomputed residual endpoint does not match the frozen record; "
            f"the audit would describe a different object.\n{failed.to_string()}"
        )

    tables = {
        "structure": pd.DataFrame(structure_rows),
        "annotation_genes": pd.DataFrame(annotation_gene_rows),
        "annotation_classes": pd.DataFrame(annotation_class_rows),
        "depth": pd.DataFrame(depth_rows),
        "registration": pd.DataFrame(registration_rows),
        "anchor_propagation": pd.DataFrame(anchor_rows),
        "spatial_scale": pd.DataFrame(scale_rows),
        "coordinate_sensitivity": pd.DataFrame(coordinate_rows),
    }

    write_table(output / "00_provenance/RESIDUAL_REPRODUCTION_CHECK.tsv", reproduction_frame)
    if verification:
        write_table(output / "00_provenance/FROZEN_PREDICTION_HASHES.tsv", pd.DataFrame(verification))
    write_table(output / "02_structure/RESIDUAL_VARIANCE_DECOMPOSITION.tsv", tables["structure"])
    write_table(output / "03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv", tables["annotation_genes"])
    write_table(output / "03_annotation/RESIDUAL_RECOVERY_BY_COMPARTMENT.tsv", tables["annotation_classes"])
    write_table(output / "04_confounds/DEPTH_PARTIAL_CORRELATION.tsv", tables["depth"])
    write_table(output / "04_confounds/REGISTRATION_INSTABILITY_STRATA.tsv", tables["registration"])
    write_table(output / "04_confounds/ANCHOR_PROPAGATION_BY_GENE.tsv", tables["anchor_propagation"])
    write_table(output / "04_confounds/SPATIAL_SCALE_DECOMPOSITION.tsv", tables["spatial_scale"])
    write_table(output / "04_confounds/COORDINATE_SENSITIVITY.tsv", tables["coordinate_sensitivity"])

    provenance = {
        "audit": "residual landscape biological audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_root": str(FROZEN_ROOT),
        "frozen_root_modified": False,
        "targets": [r.key for r in runs],
        "seeds": list(SEEDS),
        "reference_seed_for_confound_strata": REFERENCE_SEED,
        "predictions_checked": len(verification),
        "predictions_verified": int(sum(r["matches"] for r in verification)),
        "reproduction_checked": len(reproduction_frame),
        "reproduction_passed": int(reproduction_frame.reproduced.sum()),
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "max_absolute_reproduction_difference": float(reproduction_frame.absolute_difference.max()),
        "annotation_columns_used": sorted(tables["annotation_genes"].annotation.unique().tolist()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "frozen_artifacts_modified": [],
        "scope": (
            "retrospective characterisation of frozen residual fields; no frozen "
            "prediction, registration, anchor list, manifest, figure or decision was "
            "modified or recomputed"
        ),
    }
    assert_writable(output / "00_provenance/AUDIT_PROVENANCE.json")
    (output / "00_provenance/AUDIT_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    report = build_report(output, tables, provenance)
    assert_writable(output / "05_report/RESIDUAL_LANDSCAPE_AUDIT.md")
    (output / "05_report/RESIDUAL_LANDSCAPE_AUDIT.md").write_text(report)

    print(f"[audit] complete in {provenance['runtime_seconds']}s")
    print(f"[audit] report -> {output / '05_report/RESIDUAL_LANDSCAPE_AUDIT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
