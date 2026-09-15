"""Thin adapters over executed HyperSpatial-MAP operations."""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import numpy as np
from scipy.spatial import cKDTree
from run_hyperspatial_map_stage1_20260803 import idw, depth_predictor, predicted_depth
from run_hyperspatial_map_stage1b_20260803 import (apply_centered, candidate_fusion,
    fit_centered_ridge_rank, select_non_axial_anchors, source_gate)
from analysis.hyperspatial_map_2d_dlpfc.dlpfc_common import canonicalize, pearson_columns

def orthants(coordinates: np.ndarray) -> np.ndarray:
    med=np.median(coordinates,axis=0); labels=np.zeros(len(coordinates),np.int8)
    for d in range(coordinates.shape[1]): labels += (2**d)*(coordinates[:,d]>med[d])
    return labels

def pseudo_block_prediction(values, coordinates, labels, k=12):
    out=np.empty_like(values)
    for label in np.unique(labels):
        q=np.flatnonzero(labels==label); ref=np.flatnonzero(labels!=label)
        if not len(q) or len(ref)<k: raise ValueError(f"spatial fold {label} has insufficient reference points")
        out[q]=idw(coordinates[ref],values[ref],coordinates[q],k)
    return out

def smooth(values, coordinates, k):
    idx=cKDTree(coordinates).query(coordinates,k=min(k+1,len(coordinates)))[1]
    if idx.ndim==1: return values.copy()
    return values[idx[:,1:]].mean(1).astype(np.float32)

def residual_features(residual, coordinates, neighbors=(12,32)):
    return np.concatenate([residual]+[smooth(residual,coordinates,k) for k in neighbors],axis=1).astype(np.float32)

def registered_prior(source_coordinates, source_values, target_coordinates, k=12):
    return np.log1p(idw(source_coordinates,source_values,target_coordinates,k)).astype(np.float32)

def select_panel(source_log, genes, budget):
    return select_non_axial_anchors(source_log,np.asarray(genes),budget)

def predictor_label(name: str) -> str:
    if name=="registered_idw": return "IDW fallback"
    if name.startswith("global_anchor_w0.00"): return "Global residual"
    if name.startswith("local_anchor_w0.00"): return "Local residual"
    if name.endswith("w1.00"): return "Anchor-only"
    return "Calibrated mixture"
