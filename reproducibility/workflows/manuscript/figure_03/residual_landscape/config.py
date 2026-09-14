#!/usr/bin/env python3
"""Frozen input locations, the six-target run table, and write-scope guards.

The frozen cross-stage directory is treated as immutable read-only input. No
function in this package may open a path inside it for writing, and the audit
refuses to start if the requested output directory resolves inside any frozen
tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Immutable read-only inputs.
FROZEN_ROOT = PROJECT_ROOT / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726"
FROZEN_75P_FORWARD = PROJECT_ROOT / "results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2"
FROZEN_75P_REVERSE = PROJECT_ROOT / "results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1"
DATA_ROOT = PROJECT_ROOT / "data_v10/wemerfish_zebrafish_3stage"

# Every tree the audit may read but must never write into.
PROTECTED_TREES = (
    FROZEN_ROOT,
    FROZEN_75P_FORWARD.parent,
    FROZEN_75P_REVERSE.parent,
    DATA_ROOT,
)

OUTPUT_PREFIX = "hyperspatial_map_residual_landscape"
OUTPUT_PARENT = PROJECT_ROOT / "results_v10"

# Frozen method names, unchanged from the locked protocol.
MAP = "hyperspatial_map_calibrated_fusion"
IDW = "registered_idw_k12"
NEAREST = "registered_nearest_neighbor"
ANCHOR = "anchor_only_calibrated"
GLOBAL_RESIDUAL = "atlas_plus_global_anchor_residual"
LOCAL_RESIDUAL = "atlas_plus_local_anchor_residual"

# Methods whose residual fields the audit characterises. IDW is the base, so its
# own residual is identically zero and it is never audited as a predictor.
AUDITED_METHODS = (MAP, ANCHOR, GLOBAL_RESIDUAL, LOCAL_RESIDUAL)

SEEDS = (42, 43, 44)

# Annotation columns supplied by the dataset authors. None of these were read by
# the frozen runner (LEAKAGE_AUDIT: target_annotations_accessed = false), so they
# are independent of every choice the method made.
ANNOTATION_COLUMNS = ("germlayer", "type", "tissue", "clusters")


@dataclass(frozen=True)
class TargetRun:
    """One frozen source-to-target transfer."""

    key: str
    stage: str
    tag: str
    source: str
    target: str
    prediction_dir: Path
    source_file: Path
    target_file: Path

    @property
    def manifest_path(self) -> Path:
        return self.prediction_dir / "prediction_manifest.json"


RUNS: tuple[TargetRun, ...] = (
    TargetRun(
        key="50p_E1_to_E2", stage="50% epiboly", tag="50p", source="E1", target="E2",
        prediction_dir=FROZEN_ROOT / "04_predictions/50p_E1_to_E2",
        source_file=DATA_ROOT / "weMERFISH_measured_A_50p_E1.h5ad",
        target_file=DATA_ROOT / "weMERFISH_measured_A_50p_E2.h5ad",
    ),
    TargetRun(
        key="50p_E2_to_E1", stage="50% epiboly", tag="50p", source="E2", target="E1",
        prediction_dir=FROZEN_ROOT / "04_predictions/50p_E2_to_E1",
        source_file=DATA_ROOT / "weMERFISH_measured_A_50p_E2.h5ad",
        target_file=DATA_ROOT / "weMERFISH_measured_A_50p_E1.h5ad",
    ),
    TargetRun(
        key="75p_E1_to_E2", stage="75% epiboly", tag="75p", source="E1", target="E2",
        prediction_dir=FROZEN_75P_FORWARD,
        source_file=DATA_ROOT / "weMERFISH_measured_B_75p_E1.h5ad",
        target_file=DATA_ROOT / "weMERFISH_measured_B_75p_E2.h5ad",
    ),
    TargetRun(
        key="75p_E2_to_E1", stage="75% epiboly", tag="75p", source="E2", target="E1",
        prediction_dir=FROZEN_75P_REVERSE,
        source_file=DATA_ROOT / "weMERFISH_measured_B_75p_E2.h5ad",
        target_file=DATA_ROOT / "weMERFISH_measured_B_75p_E1.h5ad",
    ),
    TargetRun(
        key="6s_E1_to_E2", stage="6-somite", tag="6s", source="E1", target="E2",
        prediction_dir=FROZEN_ROOT / "04_predictions/6s_E1_to_E2",
        source_file=DATA_ROOT / "weMERFISH_measured_C_6s_E1_rescaled_z.h5ad",
        target_file=DATA_ROOT / "weMERFISH_measured_C_6s_E2_rescaled_z.h5ad",
    ),
    TargetRun(
        key="6s_E2_to_E1", stage="6-somite", tag="6s", source="E2", target="E1",
        prediction_dir=FROZEN_ROOT / "04_predictions/6s_E2_to_E1",
        source_file=DATA_ROOT / "weMERFISH_measured_C_6s_E2_rescaled_z.h5ad",
        target_file=DATA_ROOT / "weMERFISH_measured_C_6s_E1_rescaled_z.h5ad",
    ),
)


def is_protected(path: Path) -> bool:
    """True when ``path`` lies inside a frozen tree that must not be written."""
    resolved = path.resolve()
    for tree in PROTECTED_TREES:
        try:
            resolved.relative_to(tree.resolve())
        except ValueError:
            continue
        return True
    return False


def assert_writable(path: Path) -> Path:
    """Raise unless ``path`` is outside every frozen tree."""
    if is_protected(path):
        raise PermissionError(
            f"Refusing to write inside a frozen read-only tree: {path}"
        )
    return path


def new_output_dir(timestamp: str | None = None) -> Path:
    """Create and return a fresh timestamped audit output directory."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_PARENT / f"{OUTPUT_PREFIX}_{stamp}"
    assert_writable(path)
    if path.exists():
        raise RuntimeError(f"Output directory already exists: {path}")
    # Residual fields themselves are not serialised: they are hundreds of
    # megabytes and are rebuilt deterministically from the frozen predictions.
    for section in ("00_provenance", "02_structure", "03_annotation",
                    "04_confounds", "05_report"):
        (path / section).mkdir(parents=True)
    return path
