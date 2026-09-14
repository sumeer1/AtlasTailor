#!/usr/bin/env python3
"""Post-hoc, source-only experimental panel-design extension.

This runner is intentionally isolated. It reads frozen source embryos, panel
manifests and selector definitions, but writes only to the dedicated extension
output directory. It never imports or invokes a registration/model runner.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "analysis/panel_design_extension"
OUT = ROOT / "results_v10/panel_design_extension_20260809"
AUDIT = OUT / "00_audit"
os.environ.setdefault("MPLCONFIGDIR", str(AUDIT / "mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree


LABEL = "post-hoc experimental-design extension"
K_IDW = 12
RANK = 16
FOLDS = tuple(range(8))
BUDGETS = (8, 16, 32)
EXPANDED_EXCLUDE = {
    "admp", "bambia", "foxa", "foxa2", "foxa3", "gata6", "gbx1", "gli1",
    "lhx1a", "msgn1", "noto", "otx1", "shha", "sox19a", "sp5l", "tbx1",
    "tbxta", "tbxtb", "zic2b",
}
AXIAL = ("tbxta", "noto", "shha")


@dataclass(frozen=True)
class Setting:
    key: str
    stage: str
    source: str
    direction: str
    embryo: Path
    manifest: Path


SETTINGS = (
    Setting("50p_E1_to_E2", "50% epiboly", "E1", "E1→E2",
            ROOT / "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E1.h5ad",
            ROOT / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E1_to_E2/prediction_manifest.json"),
    Setting("50p_E2_to_E1", "50% epiboly", "E2", "E2→E1",
            ROOT / "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E2.h5ad",
            ROOT / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E2_to_E1/prediction_manifest.json"),
    Setting("75p_E1_to_E2", "75% epiboly", "E1", "E1→E2",
            ROOT / "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_B_75p_E1.h5ad",
            ROOT / "results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2/prediction_manifest.json"),
    Setting("75p_E2_to_E1", "75% epiboly", "E2", "E2→E1",
            ROOT / "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_B_75p_E2.h5ad",
            ROOT / "results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1/prediction_manifest.json"),
    Setting("6s_E1_to_E2", "6-somite", "E1", "E1→E2",
            ROOT / "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_C_6s_E1_rescaled_z.h5ad",
            ROOT / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E1_to_E2/prediction_manifest.json"),
    Setting("6s_E2_to_E1", "6-somite", "E2", "E2→E1",
            ROOT / "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_C_6s_E2_rescaled_z.h5ad",
            ROOT / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E2_to_E1/prediction_manifest.json"),
)

PANEL_TABLE = ROOT / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv"
PROTOCOL = ROOT / "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/00_protocol_lock/PROTOCOL_LOCK.json"
SOURCE_SCRIPTS = (
    ROOT / "run_hyperspatial_map_stage1_20260803.py",
    ROOT / "run_hyperspatial_map_stage1b_20260803.py",
    ROOT / "run_wemerfish_temporal_holdout.py",
)

EXPECTED_HASHES = {
    "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E1.h5ad": "8c579f6d1278ec12e17665b6645c892890b29d8e2500cb4a3eecafcafc620884",
    "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_A_50p_E2.h5ad": "e0a88d080cc5c5750c94ff7e2dfb313f257f1a4beaeae03c76d9a466f7f7c2d6",
    "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_B_75p_E1.h5ad": "bf0fb07556f700948999b818de83d156f95dc8ecdd7aaa52bdaca0c7b2392434",
    "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_B_75p_E2.h5ad": "ab33956942d325df380bd51d21f62c132f3a36d0e5a95f0f3cf3b7fed15a7daa",
    "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_C_6s_E1_rescaled_z.h5ad": "a0ea9d01a7e8a59bfda3876b9a3d0f0aa345d50670c718e79fbb7821cf1ea14b",
    "data_v10/wemerfish_zebrafish_3stage/weMERFISH_measured_C_6s_E2_rescaled_z.h5ad": "49249578ca4b09c3b3ffd5ea7abf7d2646c4d14229ef0a017c1cec8f0a757632",
    "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/03_anchor_panels/SOURCE_ONLY_ANCHOR_PANELS.tsv": "7d02422b8c89f00dacb010e2fb66922f27b4ce07c3c7cd8c804a7880fc47244f",
    "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E1_to_E2/prediction_manifest.json": "cc72962f0d0c566caf89cc2d102bab70a4c1a2724e2b320bd458f7ee4e461c0f",
    "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/50p_E2_to_E1/prediction_manifest.json": "a1613b056f052f21a19d36d9168b28499d3c7c2fbe6734578f961eb3560ecc07",
    "results_v10/hyperspatial_map_calibrated_20260803_142652/E1_to_E2/prediction_manifest.json": "52b1c9bd3021655715dccd9b94e630d68b406ee4d40222e7cca52ec811d97658",
    "results_v10/hyperspatial_map_reciprocal_frozen_20260803_144112/E2_to_E1/prediction_manifest.json": "0fff2f37c5fbbe33c2421f7292dc3ab4d65c28de61be083e38b34a33286244c4",
    "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E1_to_E2/prediction_manifest.json": "34cdf2dee1d871e9b67d2edfd075c85d26399b2a287755e6443250b26e4e0bfb",
    "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/04_predictions/6s_E2_to_E1/prediction_manifest.json": "03108cf23b75c8c513bd30de1f1083e09f447284b3aa3ab2e31d303191737dbd",
    "run_hyperspatial_map_stage1_20260803.py": "6ece4008d579821799f2eea2ba0abd447b03ea470b90c96090ad6f3f5be49d12",
    "run_hyperspatial_map_stage1b_20260803.py": "317c5d642541def2e3384ec914c20067b6630dd535aecd4be880bd1a2a550bb6",
    "run_wemerfish_temporal_holdout.py": "1d14d143af98f9196e121be98094b993b4d6ca5c8bde8a61ed447c9d71ee7635",
    "results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726/00_protocol_lock/PROTOCOL_LOCK.json": "e579bf1f7e55f8223eab3513168cdcdbbff6213e2191c871371516495544bba5",
}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_status() -> list[str]:
    text = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True
    )
    return text.splitlines()


def outside_allowed(lines: list[str]) -> list[str]:
    allowed = ("analysis/panel_design_extension/", "results_v10/panel_design_extension_20260809/")
    result = []
    for line in lines:
        path = line[3:] if len(line) >= 4 else line
        if not path.startswith(allowed):
            result.append(line)
    return result


def audit_inputs(phase: str) -> pd.DataFrame:
    rows = []
    for name, expected in EXPECTED_HASHES.items():
        path = ROOT / name
        actual = sha256(path)
        rows.append({"path": name, "sha256_before" if phase == "before" else "sha256_after": actual,
                     "expected_sha256": expected, "matches_expected": actual == expected})
        if actual != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {name}")
    return pd.DataFrame(rows)


def read_dense(handle: h5py.File, key: str) -> np.ndarray:
    node = handle[key]
    if isinstance(node, h5py.Dataset):
        return np.asarray(node, np.float32)
    # AnnData sparse CSR encoding fallback.
    from scipy.sparse import csr_matrix
    shape = tuple(node.attrs["shape"])
    return csr_matrix((node["data"][:], node["indices"][:], node["indptr"][:]), shape=shape).toarray().astype(np.float32)


def load_source(path: Path):
    with h5py.File(path, "r") as h:
        spliced = read_dense(h, "layers/spliced")
        unspliced = read_dense(h, "layers/unspliced")
        counts = spliced + unspliced
        key = "global_sphere" if "global_sphere" in h["obsm"] else "spatial"
        coordinates = np.asarray(h[f"obsm/{key}"], np.float32)
        genes = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in h["var/_index"][:]])
    if not np.allclose(counts, np.rint(counts)):
        raise AssertionError(f"Non-integer counts: {path}")
    center = np.median(coordinates, axis=0)
    coordinates = coordinates - center
    scale = max(float(np.quantile(np.linalg.norm(coordinates, axis=1), 0.99)), 1e-6)
    coordinates = (coordinates / scale).astype(np.float32)
    depth = np.maximum(counts.sum(1), 1.0)
    depth_target = float(np.median(depth))
    values = (counts * (depth_target / depth)[:, None]).astype(np.float32)
    log_values = np.log1p(values).astype(np.float32)
    return values, log_values, coordinates, genes, key, depth_target


def octants(coordinates: np.ndarray) -> np.ndarray:
    med = np.median(coordinates, axis=0)
    return ((coordinates[:, 0] > med[0]).astype(np.int8)
            + 2 * (coordinates[:, 1] > med[1]).astype(np.int8)
            + 4 * (coordinates[:, 2] > med[2]).astype(np.int8))


def idw(source_coordinates, source_values, query_coordinates, k=K_IDW, chunk=1024):
    tree = cKDTree(source_coordinates)
    out = np.empty((len(query_coordinates), source_values.shape[1]), np.float32)
    for begin in range(0, len(query_coordinates), chunk):
        q = query_coordinates[begin:begin + chunk]
        distance, index = tree.query(q, k=min(k, len(source_coordinates)))
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        weight = np.maximum(distance, 1e-5) ** -2
        weight /= np.maximum(weight.sum(1, keepdims=True), 1e-12)
        out[begin:begin + len(q)] = np.einsum("bk,bkg->bg", weight, source_values[index], optimize=True)
    return out


def pseudo_residual(values, log_values, coordinates, labels):
    base = np.empty_like(values)
    for fold in FOLDS:
        val = labels == fold
        train = ~val
        if val.sum() == 0 or train.sum() < K_IDW:
            raise RuntimeError(f"Invalid octant {fold}")
        base[val] = idw(coordinates[train], values[train], coordinates[val])
    return (log_values - np.log1p(base)).astype(np.float32)


def candidate_set(log_values, genes, expected_count):
    prevalence = np.mean(log_values > 0, axis=0)
    variance = np.var(log_values, axis=0)
    candidates = np.asarray([
        i for i, gene in enumerate(genes)
        if 0.05 <= prevalence[i] <= 0.98 and variance[i] > 1e-5
        and gene.lower() not in EXPANDED_EXCLUDE
        and not gene.lower().startswith(("mt-", "rpl", "rps"))
    ], np.int64)
    rng = np.random.default_rng(42116)
    sample = np.sort(rng.choice(len(log_values), min(5000, len(log_values)), replace=False))
    z = log_values[sample].astype(np.float64)
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-5)
    corr = np.abs((z[:, candidates].T @ z) / max(len(z) - 1, 1))
    axial = [int(np.flatnonzero(genes == g)[0]) for g in AXIAL if np.any(genes == g)]
    if axial:
        keep = np.max(corr[:, axial], axis=1) < 0.40
        candidates, corr = candidates[keep], corr[keep]
    if len(candidates) != expected_count:
        raise AssertionError(f"Candidate count {len(candidates)} != frozen {expected_count}")
    return candidates, corr.astype(np.float64), sample


@dataclass
class RawStats:
    n: int
    sx: np.ndarray
    sy: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray


def raw_stats(x, y, mask) -> RawStats:
    xx = x[mask].astype(np.float64, copy=False)
    yy = y[mask].astype(np.float64, copy=False)
    return RawStats(
        int(mask.sum()), xx.sum(0), yy.sum(0), xx.T @ xx, xx.T @ yy,
        np.einsum("ng,ng->g", yy, yy),
    )


def add_raw(parts) -> RawStats:
    parts = list(parts)
    return RawStats(sum(p.n for p in parts), sum((p.sx for p in parts), np.zeros_like(parts[0].sx)),
                    sum((p.sy for p in parts), np.zeros_like(parts[0].sy)),
                    sum((p.xx for p in parts), np.zeros_like(parts[0].xx)),
                    sum((p.xy for p in parts), np.zeros_like(parts[0].xy)),
                    sum((p.yy for p in parts), np.zeros_like(parts[0].yy)))


def subtract_raw(total: RawStats, removed) -> RawStats:
    removed = add_raw(removed)
    return RawStats(total.n - removed.n, total.sx - removed.sx, total.sy - removed.sy,
                    total.xx - removed.xx, total.xy - removed.xy, total.yy - removed.yy)


def standardized_stats(train: RawStats, val: RawStats):
    mu = train.sx / train.n
    var = np.diag(train.xx) / train.n - mu ** 2
    scale = np.sqrt(np.maximum(var, 1e-10))
    denom = np.outer(scale, scale)
    A = (train.xx - np.outer(train.sx, train.sx) / train.n) / denom
    B = (train.xy - np.outer(mu, train.sy)) / scale[:, None]
    V = (val.xx - np.outer(mu, val.sx) - np.outer(val.sx, mu)
         + val.n * np.outer(mu, mu)) / denom
    D = (val.xy - np.outer(mu, val.sy)) / scale[:, None]
    sz = (val.sx - val.n * mu) / scale
    return {"A": A, "B": B, "V": V, "D": D, "sz": sz,
            "ymean": train.sy / train.n, "sy": val.sy, "yy": val.yy, "n": val.n,
            "n_train": train.n}


def pearson_from_sums(n, sy, yy, sp, pp, yp):
    cov = yp - sy * sp / n
    vy = yy - sy ** 2 / n
    vp = pp - sp ** 2 / n
    den = np.sqrt(np.maximum(vy * vp, 0))
    out = np.full(np.broadcast_shapes(np.shape(cov), np.shape(den)), np.nan, dtype=np.float64)
    np.divide(cov, den, out=out, where=den > 1e-12)
    return out


def low_rank_weight(A, B, ridge, rank=RANK):
    w = np.linalg.solve(A + ridge * np.eye(len(A)), B)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    keep = min(rank, len(s))
    return (u[:, :keep] * s[:keep]) @ vt[:keep]


def cv_panel(panel_pos, fold_states, ridge, gain):
    panel_pos = np.asarray(panel_pos, np.int64)
    n_total = 0
    sy = yy = sp = pp = yp = None
    for st in fold_states:
        A = st["A"][np.ix_(panel_pos, panel_pos)]
        B = st["B"][panel_pos]
        V = st["V"][np.ix_(panel_pos, panel_pos)]
        D = st["D"][panel_pos]
        sz = st["sz"][panel_pos]
        w = low_rank_weight(A, B, ridge)
        zw_sum = sz @ w
        p_sum = gain * (st["n"] * st["ymean"] + zw_sum)
        p_sq = gain ** 2 * (st["n"] * st["ymean"] ** 2
                              + 2 * st["ymean"] * zw_sum
                              + np.sum(w * (V @ w), axis=0))
        y_p = gain * (st["ymean"] * st["sy"] + np.sum(D * w, axis=0))
        if sy is None:
            sy, yy, sp, pp, yp = st["sy"].copy(), st["yy"].copy(), p_sum, p_sq, y_p
        else:
            sy += st["sy"]; yy += st["yy"]; sp += p_sum; pp += p_sq; yp += y_p
        n_total += st["n"]
    corr = pearson_from_sums(n_total, sy, yy, sp, pp, yp)
    mse = np.maximum((yy + pp - 2 * yp) / n_total, 0)
    return corr, mse


def init_stage(st):
    ymean = st["ymean"]
    return {
        **st,
        "train_e": st["B"].copy(),
        "xvp": np.outer(st["sz"], ymean),
        "sp": st["n"] * ymean.copy(),
        "pp": st["n"] * ymean ** 2,
        "yp": ymean * st["sy"],
    }


def pooled_current(states):
    n = sum(s["n"] for s in states)
    sy = sum((s["sy"] for s in states), np.zeros_like(states[0]["sy"]))
    yy = sum((s["yy"] for s in states), np.zeros_like(states[0]["yy"]))
    sp = sum((s["sp"] for s in states), np.zeros_like(states[0]["sp"]))
    pp = sum((s["pp"] for s in states), np.zeros_like(states[0]["pp"]))
    yp = sum((s["yp"] for s in states), np.zeros_like(states[0]["yp"]))
    return pearson_from_sums(n, sy, yy, sp, pp, yp), (n, sy, yy, sp, pp, yp)


def stagewise_select(base_states, candidate_gene_indices, genes, corr_abs, ridge, max_budget=32):
    states = [init_stage(s) for s in base_states]
    c, g = states[0]["B"].shape
    selected = []
    records = []
    current_corr, _ = pooled_current(states)
    current_score = float(np.nanmedian(current_corr))
    for rank in range(1, max_budget + 1):
        n = 0
        sy = yy = sp = pp = yp = None
        betas = []
        for st in states:
            denom = np.diag(st["A"]) + ridge
            beta = st["train_e"] / denom[:, None]
            betas.append(beta)
            csp = st["sp"][None, :] + st["sz"][:, None] * beta
            cpp = st["pp"][None, :] + 2 * beta * st["xvp"] + beta ** 2 * np.diag(st["V"])[:, None]
            cyp = st["yp"][None, :] + beta * st["D"]
            if sy is None:
                sy, yy, sp, pp, yp = st["sy"].copy(), st["yy"].copy(), csp, cpp, cyp
            else:
                sy += st["sy"]; yy += st["yy"]; sp += csp; pp += cpp; yp += cyp
            n += st["n"]
        candidate_corr = pearson_from_sums(n, sy[None, :], yy[None, :], sp, pp, yp)
        score_matrix = candidate_corr.copy()
        if selected:
            score_matrix[:, candidate_gene_indices[selected]] = np.nan
        score_matrix[np.arange(c), candidate_gene_indices] = np.nan
        scores = np.nanmedian(score_matrix, axis=1)
        scores[selected] = -np.inf
        chosen = int(np.nanargmax(scores))
        utility = candidate_corr[chosen] - current_corr
        eligible_gene = np.ones(g, bool)
        eligible_gene[candidate_gene_indices[selected + [chosen]]] = False
        helped = eligible_gene & np.isfinite(utility) & (utility > 0)
        top = np.flatnonzero(helped)
        top = top[np.argsort(utility[top])[::-1]][:10]
        selected.append(chosen)
        for st, beta_all in zip(states, betas):
            beta = beta_all[chosen]
            old_xvp = st["xvp"][chosen].copy()
            st["sp"] += st["sz"][chosen] * beta
            st["pp"] += 2 * beta * old_xvp + beta ** 2 * st["V"][chosen, chosen]
            st["yp"] += beta * st["D"][chosen]
            st["xvp"] += st["V"][:, chosen, None] * beta[None, :]
            st["train_e"] -= st["A"][:, chosen, None] * beta[None, :]
        new_corr, _ = pooled_current(states)
        new_score = float(np.nanmedian(new_corr[eligible_gene]))
        sel_corr = corr_abs[selected]
        coverage = float(np.mean(np.max(sel_corr ** 2, axis=0)))
        if len(selected) == 1:
            redundancy = 0.0
            max_corr = 0.0
        else:
            pair = sel_corr[:, candidate_gene_indices[selected]]
            tri = pair[np.triu_indices(len(selected), 1)]
            redundancy = float(np.mean(tri ** 2))
            max_corr = float(np.max(np.abs(pair[:-1, -1])))
        records.append({
            "rank": rank, "candidate_position": chosen,
            "gene_index": int(candidate_gene_indices[chosen]), "gene": str(genes[candidate_gene_indices[chosen]]),
            "marginal_residual_pearson_gain": new_score - current_score,
            "expected_source_cv_residual_pearson": new_score,
            "coverage_mean_max_squared_correlation": coverage,
            "redundancy_mean_pairwise_squared_correlation": redundancy,
            "max_abs_correlation_to_previous": max_corr,
            "genes_helped_count": int(helped.sum()),
            "genes_evaluated_count": int(eligible_gene.sum()),
            "top_supported_genes": ",".join(genes[top].tolist()),
        })
        current_corr, current_score = new_corr, new_score
    return selected, records


def stagewise_outer_metrics(state, selected, budgets, ridge, gain):
    st = init_stage(state)
    result = {}
    for rank, chosen in enumerate(selected, 1):
        beta = st["train_e"][chosen] / (st["A"][chosen, chosen] + ridge)
        old_xvp = st["xvp"][chosen].copy()
        st["sp"] += st["sz"][chosen] * beta
        st["pp"] += 2 * beta * old_xvp + beta ** 2 * st["V"][chosen, chosen]
        st["yp"] += beta * st["D"][chosen]
        st["xvp"] += st["V"][:, chosen, None] * beta[None, :]
        st["train_e"] -= st["A"][:, chosen, None] * beta[None, :]
        if rank in budgets:
            corr = pearson_from_sums(st["n"], st["sy"], st["yy"], gain * st["sp"], gain ** 2 * st["pp"], gain * st["yp"])
            mse = np.maximum((st["yy"] + gain ** 2 * st["pp"] - 2 * gain * st["yp"]) / st["n"], 0)
            result[rank] = (corr, mse)
    return result


def setting_prefix(setting, manifest):
    return {"extension": LABEL, "setting": setting.key, "stage": setting.stage,
            "source_embryo": setting.source, "direction": setting.direction,
            "frozen_ridge": manifest["selected_source_configuration"]["ridge"],
            "frozen_gain": manifest["selected_source_configuration"]["gain"]}


def analyze_setting(setting: Setting, panels: pd.DataFrame):
    print(f"[{setting.key}] load source", flush=True)
    manifest = json.loads(setting.manifest.read_text())
    values, log_values, coordinates, genes, coord_key, depth_target = load_source(setting.embryo)
    frozen_panel = panels.loc[panels.target_key == setting.key].sort_values("rank")
    anchors = frozen_panel.gene_index.to_numpy(np.int64)
    if frozen_panel.gene.tolist() != manifest["anchor_genes"]:
        raise AssertionError(f"Panel mismatch: {setting.key}")
    labels = octants(coordinates)
    print(f"[{setting.key}] exact octant pseudo-holdouts", flush=True)
    residual = pseudo_residual(values, log_values, coordinates, labels)
    del values
    candidates, corr_abs, selector_sample = candidate_set(
        log_values, genes, int(manifest["anchor_metadata"]["candidate_count"])
    )
    candidate_lookup = {int(g): i for i, g in enumerate(candidates)}
    if any(int(a) not in candidate_lookup for a in anchors):
        raise AssertionError("Frozen anchor absent from reconstructed frozen candidate set")
    anchor_pos = [candidate_lookup[int(a)] for a in anchors]
    x = residual[:, candidates]
    print(f"[{setting.key}] sufficient statistics ({len(candidates)} candidates)", flush=True)
    fold_raw = {f: raw_stats(x, residual, labels == f) for f in FOLDS}
    total = add_raw(fold_raw.values())
    fold_states = [standardized_stats(subtract_raw(total, [fold_raw[f]]), fold_raw[f]) for f in FOLDS]
    ridge = float(manifest["selected_source_configuration"]["ridge"])
    gain = float(manifest["selected_source_configuration"]["gain"])
    prefix = setting_prefix(setting, manifest)

    # A. Frozen-panel leave-one-anchor-out utility.
    print(f"[{setting.key}] leave-one-anchor-out utility", flush=True)
    full_corr, full_mse = cv_panel(anchor_pos, fold_states, ridge, gain)
    loo = {}
    for i in range(16):
        keep = [p for j, p in enumerate(anchor_pos) if j != i]
        loo[i] = cv_panel(keep, fold_states, ridge, gain)
    eval_mask = np.ones(len(genes), bool)
    eval_mask[anchors] = False
    utility_rows, gene_rows = [], []
    u_medians = {}
    for i, anchor in enumerate(anchors):
        u = full_corr - loo[i][0]
        mse_gain = loo[i][1] - full_mse
        valid = eval_mask & np.isfinite(u)
        order = np.flatnonzero(valid)
        order = order[np.argsort(u[order])[::-1]]
        u_medians[i] = float(np.nanmedian(u[valid]))
        utility_rows.append({
            **prefix, "anchor_rank": i + 1, "anchor_gene": genes[anchor], "anchor_gene_index": int(anchor),
            "utility_median_residual_pearson": u_medians[i],
            "mse_improvement_median": float(np.nanmedian(mse_gain[valid])),
            "fraction_genes_helped": float(np.mean(u[valid] > 0)),
            "genes_evaluated": int(valid.sum()),
            "top_supported_genes": ",".join(genes[order[:10]].tolist()),
        })
        rank_map = {int(g): rank for rank, g in enumerate(order, 1)}
        for gene in np.flatnonzero(eval_mask):
            gene_rows.append({
                **prefix, "anchor_rank": i + 1, "anchor_gene": genes[anchor],
                "anchor_gene_index": int(anchor), "supported_gene": genes[gene],
                "supported_gene_index": int(gene),
                "full_panel_residual_pearson": full_corr[gene],
                "leave_one_anchor_out_residual_pearson": loo[i][0][gene],
                "utility_residual_pearson": u[gene],
                "full_panel_residual_mse": full_mse[gene],
                "leave_one_anchor_out_residual_mse": loo[i][1][gene],
                "mse_improvement": mse_gain[gene],
                "helped": bool(np.isfinite(u[gene]) and u[gene] > 0),
                "support_rank": rank_map.get(int(gene), ""),
            })

    # B. Ordered conditional utility U_{a|b}.
    print(f"[{setting.key}] conditional redundancy", flush=True)
    pair_perf = {}
    for i in range(16):
        for j in range(i + 1, 16):
            keep = [p for k, p in enumerate(anchor_pos) if k not in (i, j)]
            pair_perf[(i, j)] = cv_panel(keep, fold_states, ridge, gain)
    redundancy_rows = []
    for a in range(16):
        for b in range(16):
            if a == b:
                continue
            pair = pair_perf[tuple(sorted((a, b)))]
            u_cond = loo[b][0] - pair[0]
            mse_cond = pair[1] - loo[b][1]
            valid = eval_mask & np.isfinite(u_cond)
            redundancy_rows.append({
                **prefix, "anchor_a": genes[anchors[a]], "anchor_b_conditioned": genes[anchors[b]],
                "anchor_a_rank": a + 1, "anchor_b_rank": b + 1,
                "conditional_utility_median_residual_pearson": float(np.nanmedian(u_cond[valid])),
                "conditional_mse_improvement_median": float(np.nanmedian(mse_cond[valid])),
                "fraction_genes_helped_conditionally": float(np.mean(u_cond[valid] > 0)),
                "genes_evaluated": int(valid.sum()),
                "interpretation": "lower conditional utility indicates greater substitution/redundancy",
            })

    # C. Strict nested source CV for budgets.
    print(f"[{setting.key}] strict nested outer CV", flush=True)
    outer_rows = []
    outer_orders = {}
    for outer in FOLDS:
        inner_states = []
        for inner in FOLDS:
            if inner == outer:
                continue
            train = subtract_raw(total, [fold_raw[outer], fold_raw[inner]])
            inner_states.append(standardized_stats(train, fold_raw[inner]))
        selected, _ = stagewise_select(inner_states, candidates, genes, corr_abs, ridge, max(BUDGETS))
        outer_orders[outer] = selected
        outer_state = standardized_stats(subtract_raw(total, [fold_raw[outer]]), fold_raw[outer])
        performance = stagewise_outer_metrics(outer_state, selected, BUDGETS, ridge, gain)
        for budget in BUDGETS:
            corr, mse = performance[budget]
            panel = selected[:budget]
            panel_corr = corr_abs[panel]
            coverage = float(np.mean(np.max(panel_corr ** 2, axis=0)))
            pair = panel_corr[:, candidates[panel]]
            tri = pair[np.triu_indices(len(panel), 1)]
            outer_rows.append({
                **prefix, "budget": budget, "record_type": "outer_fold", "outer_fold": outer,
                "n_train_cells": int(outer_state["n_train"]), "n_test_cells": int(outer_state["n"]),
                "panel_genes": ",".join(genes[candidates[panel]].tolist()),
                "median_residual_pearson": float(np.nanmedian(corr)),
                "mean_residual_mse": float(np.nanmean(mse)),
                "coverage_mean_max_squared_correlation": coverage,
                "redundancy_mean_pairwise_squared_correlation": float(np.mean(tri ** 2)),
                "evaluation_unit": "source cells in one held-out spatial octant; genes are technical units",
            })

    print(f"[{setting.key}] final all-source recommendations", flush=True)
    final_selected, final_records = stagewise_select(fold_states, candidates, genes, corr_abs, ridge, max(BUDGETS))
    design = []
    for rec in final_records:
        design.append({**prefix, **{k: v for k, v in rec.items() if k != "candidate_position"},
                       "coordinate_key": coord_key, "source_cells": len(log_values),
                       "selector_sample_cells": len(selector_sample),
                       "status": LABEL,
                       "not_primary_panel": True})
    for budget in BUDGETS:
        rows_b = [r for r in outer_rows if r["budget"] == budget]
        info = final_records[budget - 1]
        outer_rows.append({
            **prefix, "budget": budget, "record_type": "outer_summary", "outer_fold": "ALL",
            "n_train_cells": "", "n_test_cells": "",
            "panel_genes": ",".join(genes[candidates[final_selected[:budget]]].tolist()),
            "median_residual_pearson": float(np.median([r["median_residual_pearson"] for r in rows_b])),
            "mean_residual_mse": float(np.mean([r["mean_residual_mse"] for r in rows_b])),
            "coverage_mean_max_squared_correlation": info["coverage_mean_max_squared_correlation"],
            "redundancy_mean_pairwise_squared_correlation": info["redundancy_mean_pairwise_squared_correlation"],
            "evaluation_unit": "median/mean across eight held-out source spatial octants",
        })

    metadata = {**prefix, "coordinate_key": coord_key, "depth_target": depth_target,
                "source_cells": len(log_values), "genes": len(genes), "candidate_count": len(candidates)}
    return utility_rows, gene_rows, redundancy_rows, design, outer_rows, metadata


def write_figure(utility, redundancy, design, budget):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    colors = {"50% epiboly": "#3C78A8", "75% epiboly": "#D4863A", "6-somite": "#488A75"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    for i, (key, sub) in enumerate(utility.groupby("setting", sort=False)):
        vals = sub.utility_median_residual_pearson.to_numpy()
        ax.scatter(np.full(len(vals), i), vals, color=colors[sub.stage.iloc[0]], s=25, alpha=.85)
        ax.plot([i-.18, i+.18], [np.median(vals)]*2, color="#20262C", lw=1.2)
    ax.axhline(0, color="#AAB2B8", lw=.8)
    ax.set_xticks(range(6), utility.setting.drop_duplicates(), rotation=35, ha="right")
    ax.set_ylabel("LOAO utility, median Δ residual Pearson")
    ax.set_title("a  Frozen 16-anchor utility", loc="left", fontweight="bold")

    ax = axes[0, 1]
    vals = redundancy.conditional_utility_median_residual_pearson.to_numpy()
    ax.hist(vals[np.isfinite(vals)], bins=30, color="#6F5A8A", alpha=.82)
    ax.axvline(0, color="#20262C", lw=.9)
    ax.set_xlabel("Conditional utility $U_{a|b}$")
    ax.set_ylabel("Ordered anchor pairs")
    ax.set_title("b  Conditional redundancy", loc="left", fontweight="bold")

    ax = axes[1, 0]
    summary = budget[budget.record_type == "outer_summary"]
    for key, sub in summary.groupby("setting", sort=False):
        ax.plot(sub.budget, sub.median_residual_pearson, marker="o", lw=1.5,
                color=colors[sub.stage.iloc[0]], label=key)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Experimental gene budget")
    ax.set_ylabel("Nested source-CV residual Pearson")
    ax.set_title("c  Unbiased outer-spatial-fold performance", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6, ncol=2)

    ax = axes[1, 1]
    final = design[design["rank"].isin(BUDGETS)]
    for key, sub in final.groupby("setting", sort=False):
        ax.plot(sub["rank"], sub.coverage_mean_max_squared_correlation, marker="o", lw=1.5,
                color=colors[sub.stage.iloc[0]])
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Experimental gene budget")
    ax.set_ylabel("Source-only squared-gene coverage")
    ax.set_title("d  Coverage of recommended panels", loc="left", fontweight="bold")

    fig.suptitle("Post-hoc experimental-design extension — source-only panel utility",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "panel_design_summary.pdf")
    fig.savefig(OUT / "panel_design_summary.png", dpi=300)
    plt.close(fig)


def write_reports(utility, redundancy, design, budget, metadata, input_hashes):
    best = utility.sort_values(["setting", "utility_median_residual_pearson"], ascending=[True, False]).groupby("setting").head(1)
    panels16 = design[design["rank"] <= 16].groupby("setting").gene.apply(lambda x: ", ".join(x)).to_dict()
    findings = "\n".join(
        f"- `{r.setting}`: highest frozen-panel LOAO utility `{r.anchor_gene}` "
        f"(median Δ residual Pearson {r.utility_median_residual_pearson:.4f}; {r.fraction_genes_helped:.1%} of evaluated genes helped)."
        for r in best.itertuples()
    )
    panel_lines = "\n".join(f"- `{k}` budget 16: {v}." for k, v in panels16.items())
    readme = f"""# Source-only anchor utility and experimental panel design

**Status:** {LABEL}. This output is isolated from all frozen HyperSpatial-MAP and HyperSpatial-MAP-T analyses.

## Questions answered

1. `anchor_utility.tsv` ranks each frozen anchor by leave-one-anchor-out source-CV utility.
2. `anchor_gene_utility.tsv` identifies the genes supported by each anchor without using coefficient magnitude.
3. `anchor_redundancy.tsv` reports ordered conditional utility $U_{{a|b}}$; lower values indicate stronger substitution/redundancy.
4. `panel_design_8.tsv`, `panel_design_16.tsv`, and `panel_design_32.tsv` give new source-only experimental recommendations.

## Main descriptive findings

{findings}

## Budget-16 experimental recommendations

{panel_lines}

These budget-16 recommendations are experimental-design results only. They do not replace, revise, or reinterpret the frozen 16-gene panels used by the manuscript analyses. No claim is made that 16 genes is universally optimal.

## Methods

Each embryo is analyzed only in its source role. Counts are normalized exactly as in the frozen source runner. Coordinates use the frozen loader choice (`global_sphere` for 50%/75%; raw `spatial` for 6-somite), median centering, and 99th-percentile radial scaling. Eight source spatial pseudo-holdouts are the frozen coordinate octants. Registered 12-neighbour IDW is reproduced only within a source embryo to define pseudo-holdout residuals; no cross-embryo registration is run.

For a frozen panel $A$, `PerfCV_g(A)` is eight-fold source spatial-CV residual Pearson from a centered ridge predictor with the setting's already-frozen ridge, gain, and rank cap. Leave-one-anchor-out utility is `PerfCV_g(A) - PerfCV_g(A\\{{a}})`. The MSE column is oriented so positive means that retaining the anchor lowers MSE. Evaluation genes exclude the complete frozen 16-anchor panel.

Conditional utility is `PerfCV(A\\{{b}}) - PerfCV(A\\{{a,b}})`. Low or negative conditional utility means anchor `a` adds little once `b` is absent and is therefore more substitutable/redundant; it is not coefficient importance.

Panel design uses strict nested source CV. Each outer octant is untouched during panel selection. The other seven octants provide inner spatial folds. Greedy additions maximize the conditional marginal inner-CV residual Pearson of a stagewise centered-ridge residual decoder. Because every candidate is evaluated after conditioning on the already-selected panel, redundant candidates receive little marginal utility. Frozen eligibility/exclusion rules are retained. Coverage is the frozen source criterion `mean_g max_a |r_source(g,a)|²`; redundancy is mean pairwise squared source correlation. After unbiased outer-fold evaluation, the final recommendation is selected using all source cells with the same eight spatial folds. No target expression is accessed.

The embryo is the biological unit. Cells, genes, gene-by-anchor records, and spatial folds are technical evaluation units. Reciprocal source settings are not independent biological replicates.

## Files

- `anchor_utility.tsv`: anchor-level LOAO summaries.
- `anchor_gene_utility.tsv`: per-anchor/per-gene utilities.
- `anchor_redundancy.tsv`: ordered conditional-utility matrix in long form.
- `panel_design_{{8,16,32}}.tsv`: nested source-only recommended panels.
- `panel_budget_performance.tsv`: outer-fold and aggregate source-CV performance.
- `panel_design_summary.pdf/png`: descriptive extension summary.
- `INPUT_HASHES.tsv`, `PROVENANCE.md`, and `00_audit/`: isolation and integrity records.
"""
    (OUT / "README.md").write_text(readme)

    scripts = [CODE_DIR / "run_panel_design_extension.py"]
    provenance = [
        "# Provenance — post-hoc experimental-design extension", "",
        "This analysis is post hoc, source-only, and did not affect manuscript primary results.", "",
        "## Exact frozen inputs read", "",
        "| Path | SHA-256 |", "|---|---|",
    ]
    for r in input_hashes.itertuples():
        provenance.append(f"| `{r.path}` | `{r.sha256_before}` |")
    provenance += ["", "## New scripts created", ""]
    for path in scripts:
        provenance.append(f"- `{rel(path)}` — SHA-256 `{sha256(path)}`")
    provenance += [
        "", "## Isolation confirmation", "",
        "- No frozen prediction, model, registration, configuration, evaluation table, manuscript, or main/supplement figure was written or regenerated.",
        "- All frozen inputs were read-only and their pre/post SHA-256 values are identical.",
        "- New files are confined to `analysis/panel_design_extension/` and `results_v10/panel_design_extension_20260809/`.",
        "- The new budget-16 panels are experimental-design recommendations only and do not replace the manuscript's frozen panels.",
    ]
    (OUT / "PROVENANCE.md").write_text("\n".join(provenance) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    before_status = git_status()
    (AUDIT / "GIT_STATUS_BEFORE.txt").write_text("\n".join(before_status) + "\n")
    before_outside = outside_allowed(before_status)
    hashes_before = audit_inputs("before")
    panels = pd.read_csv(PANEL_TABLE, sep="\t")

    all_utility, all_gene, all_redundancy, all_design, all_budget, metadata = [], [], [], [], [], []
    for setting in SETTINGS:
        u, g, r, d, b, m = analyze_setting(setting, panels)
        all_utility.extend(u); all_gene.extend(g); all_redundancy.extend(r)
        all_design.extend(d); all_budget.extend(b); metadata.append(m)

    utility = pd.DataFrame(all_utility)
    gene = pd.DataFrame(all_gene)
    redundancy = pd.DataFrame(all_redundancy)
    design = pd.DataFrame(all_design)
    budget = pd.DataFrame(all_budget)
    utility.to_csv(OUT / "anchor_utility.tsv", sep="\t", index=False)
    gene.to_csv(OUT / "anchor_gene_utility.tsv", sep="\t", index=False)
    redundancy.to_csv(OUT / "anchor_redundancy.tsv", sep="\t", index=False)
    for b in BUDGETS:
        design[design["rank"] <= b].assign(budget=b).to_csv(OUT / f"panel_design_{b}.tsv", sep="\t", index=False)
    budget.to_csv(OUT / "panel_budget_performance.tsv", sep="\t", index=False)
    pd.DataFrame(metadata).to_csv(AUDIT / "SOURCE_SETTINGS.tsv", sep="\t", index=False)
    write_figure(utility, redundancy, design, budget)

    hashes_after = audit_inputs("after")
    hashes = hashes_before.merge(hashes_after[["path", "sha256_after"]], on="path")
    hashes["unchanged"] = hashes.sha256_before == hashes.sha256_after
    hashes.to_csv(OUT / "INPUT_HASHES.tsv", sep="\t", index=False)
    write_reports(utility, redundancy, design, budget, metadata, hashes)

    after_status = git_status()
    (AUDIT / "GIT_STATUS_AFTER.txt").write_text("\n".join(after_status) + "\n")
    diff_names = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True)
    (AUDIT / "GIT_DIFF_NAME_ONLY_AFTER.txt").write_text(diff_names)
    after_outside = outside_allowed(after_status)
    same_outside = before_outside == after_outside
    hashes_ok = bool(hashes.unchanged.all())
    safety = f"""# Final safety check

- Frozen input hashes unchanged: **{hashes_ok}**
- Repository status outside the two allowed extension paths unchanged: **{same_outside}**
- Allowed code path: `analysis/panel_design_extension/`
- Allowed output path: `results_v10/panel_design_extension_20260809/`
- No registration, primary MAP/MAP-T model, prediction, manuscript, or figure runner was invoked.
"""
    (AUDIT / "SAFETY_CHECK.md").write_text(safety)
    if not hashes_ok or not same_outside:
        raise RuntimeError("Isolation safety check failed")
    print("COMPLETE: isolated source-only panel-design extension", flush=True)


if __name__ == "__main__":
    main()
