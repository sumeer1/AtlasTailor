#!/usr/bin/env python3
"""Source-calibrated NB null for frozen atlas-relative residual fields."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.residual_landscape.annotation import _design, eta_squared, usable_annotations
from analysis.residual_landscape.config import IDW, RUNS, SEEDS
from analysis.residual_landscape.io_frozen import (
    _canonicalize, _measured, _normalize, load_prediction, load_target,
)
from analysis.residual_landscape.residual_fields import seed_mean_truth
from analysis.residual_landscape.structure import neighbour_index


ROOT = PROJECT_ROOT
OUT = Path(__file__).resolve().parents[1]
FROZEN_AUDIT = ROOT / "results_v10/hyperspatial_map_residual_landscape_20260805_135217"
N_NEIGHBOURS = 12
BASE_SEED = 20260818
EPS = 1e-8
METHOD_MAP = "HyperSpatial-MAP registration"
BLUE = "#2474A6"
ORANGE = "#D98324"
STAGE_ORDER = ["50% epiboly", "75% epiboly", "6-somite"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sparse_neighbour_matrix(coords: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
    neighbours = neighbour_index(coords, N_NEIGHBOURS)
    n, k = neighbours.shape
    rows = np.repeat(np.arange(n), k)
    cols = neighbours.ravel()
    weights = np.full(len(rows), 1.0 / k, np.float32)
    return neighbours, csr_matrix((weights, (rows, cols)), shape=(n, n))


def fit_source_dispersion(run, predictable_indices: np.ndarray) -> pd.DataFrame:
    counts, coords_raw, genes, _, _ = _measured(run.source_file)
    counts = counts.astype(np.float32, copy=False)
    depth = counts.sum(1).astype(np.float32)
    depth_target = float(np.median(np.maximum(depth, 1.0)))
    normalized = _normalize(counts, depth_target).astype(np.float32)
    _, wmat = sparse_neighbour_matrix(_canonicalize(coords_raw))
    local_normalized = np.asarray(wmat @ normalized, np.float32)
    mu_raw = local_normalized * (depth / depth_target)[:, None]
    denom = np.sum(mu_raw.astype(np.float64) ** 2, axis=0)
    numer = np.sum((counts - mu_raw).astype(np.float64) ** 2 - mu_raw, axis=0)
    alpha_raw = np.maximum(numer / np.maximum(denom, 1e-16), 0.0)
    mean_normalized = normalized.mean(0).astype(np.float64)
    detected = (counts > 0).sum(0).astype(np.int64)

    order = np.argsort(mean_normalized, kind="mergesort")
    log_alpha_sorted = np.log(alpha_raw[order] + EPS)
    trend_sorted = pd.Series(log_alpha_sorted).rolling(51, center=True, min_periods=11).median()
    trend_sorted = trend_sorted.interpolate(limit_direction="both").to_numpy()
    if np.isnan(trend_sorted).any():
        trend_sorted[np.isnan(trend_sorted)] = np.nanmedian(log_alpha_sorted)
    trend_log = np.empty_like(trend_sorted)
    trend_log[order] = trend_sorted
    weight = detected / (detected + 100.0)
    weight = np.where(detected >= 20, weight, 0.0)
    shrunk_log = weight * np.log(alpha_raw + EPS) + (1.0 - weight) * trend_log
    alpha = np.maximum(np.exp(shrunk_log) - EPS, EPS)

    frame = pd.DataFrame({
        "target_key": run.key,
        "stage": run.stage,
        "source_embryo": run.source,
        "gene": genes,
        "gene_index": np.arange(len(genes)),
        "source_mean_normalized_count": mean_normalized,
        "source_detected_cells": detected,
        "source_cells": len(counts),
        "raw_mom_alpha": alpha_raw,
        "trend_alpha": np.maximum(np.exp(trend_log) - EPS, EPS),
        "gene_specific_weight": weight,
        "final_shrunk_alpha": alpha,
        "eligible_non_anchor": np.isin(np.arange(len(genes)), predictable_indices),
    })
    return frame


def moran_columns(values: np.ndarray, wmat: csr_matrix) -> np.ndarray:
    centered = values - values.mean(0, keepdims=True)
    denominator = np.sum(centered.astype(np.float64) ** 2, axis=0)
    lag = wmat @ centered
    numerator = np.sum(centered.astype(np.float64) * lag, axis=0)
    return numerator / np.maximum(denominator, 1e-16)


def empirical_summary(null: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    mean = null.mean(0)
    median = np.median(null, axis=0)
    sd = np.maximum(null.std(0, ddof=1), 1e-12)
    q95 = np.quantile(null, .95, axis=0)
    percentile = np.mean(null < observed[None, :], axis=0)
    p_upper = (1 + np.sum(null >= observed[None, :], axis=0)) / (null.shape[0] + 1)
    return {"mean": mean, "median": median, "sd": sd, "q95": q95,
            "percentile": percentile, "p_upper": p_upper,
            "z": (observed - mean) / sd}


def nb_draw(rng: np.random.Generator, mu_raw: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    shape = 1.0 / np.maximum(alpha, EPS)
    probability = shape[None, :] / (shape[None, :] + mu_raw)
    return rng.negative_binomial(shape[None, :], probability).astype(np.float32)


def simulate_target(run, run_index: int, n_sim: int, frozen_structure: pd.DataFrame,
                    frozen_annotation: pd.DataFrame):
    data = load_target(run)
    truth_residual, _ = seed_mean_truth(data)
    idx = data.predictable_indices
    genes = data.genes[idx]
    if len(genes) != len(frozen_structure):
        raise AssertionError(f"{run.key}: eligibility mismatch")

    dispersion_all = fit_source_dispersion(run, idx)
    disp = dispersion_all.set_index("gene_index").loc[idx]
    alpha = disp.final_shrunk_alpha.to_numpy(np.float64)

    base = np.mean([load_prediction(run, seed, IDW, data.manifest)[:, idx] for seed in SEEDS], axis=0)
    base = np.asarray(base, np.float32)
    factor = data.depth_target / np.maximum(data.target_depth, 1.0)
    mu_normalized = np.maximum(np.expm1(base), 0.0)
    mu_raw = np.asarray(mu_normalized / factor[:, None], np.float64)
    _, wmat = sparse_neighbour_matrix(data.coordinates)

    observed_var = truth_residual.var(0).astype(np.float64)
    observed_moran = moran_columns(truth_residual, wmat)
    if not np.allclose(observed_var, frozen_structure.residual_variance, rtol=2e-5, atol=2e-6):
        raise AssertionError(f"{run.key}: observed residual variance does not reproduce frozen table")
    if not np.allclose(observed_moran, frozen_structure.residual_moran_i, rtol=2e-5, atol=2e-6):
        raise AssertionError(f"{run.key}: observed Moran I does not reproduce frozen table")

    annotation_defs = {}
    observed_eta = {}
    for column, labels in usable_annotations(data).items():
        names, codes, _ = _design(labels)
        annotation_defs[column] = (codes, len(names))
        observed_eta[column] = eta_squared(truth_residual, codes, len(names))
        frozen_block = frozen_annotation[frozen_annotation.annotation == column].set_index("gene").loc[genes]
        if not np.allclose(observed_eta[column], frozen_block.truth_residual_eta_squared,
                           rtol=2e-5, atol=2e-6):
            raise AssertionError(f"{run.key}/{column}: eta-squared does not reproduce frozen table")

    var_null = np.empty((n_sim, len(genes)), np.float32)
    moran_null = np.empty((n_sim, len(genes)), np.float32)
    eta_null = {name: np.empty((n_sim, len(genes)), np.float32) for name in annotation_defs}
    rng_seed = BASE_SEED + 10000 * run_index
    rng = np.random.default_rng(rng_seed)
    started = time.perf_counter()
    for b in range(n_sim):
        counts = nb_draw(rng, mu_raw, alpha)
        simulated_log = np.log1p(counts * factor[:, None]).astype(np.float32)
        residual = simulated_log - base
        var_null[b] = residual.var(0)
        moran_null[b] = moran_columns(residual, wmat)
        for name, (codes, n_classes) in annotation_defs.items():
            eta_null[name][b] = eta_squared(residual, codes, n_classes)
        if (b + 1) % 50 == 0:
            elapsed = time.perf_counter() - started
            print(f"[{run.key}] {b+1}/{n_sim} null fields ({elapsed:.1f}s)", flush=True)

    vs = empirical_summary(var_null, observed_var)
    ms = empirical_summary(moran_null, observed_moran)
    variance_rows = []
    moran_rows = []
    for j, gene in enumerate(genes):
        variance_rows.append({
            "target_key": run.key, "stage": run.stage, "target_embryo": run.target,
            "gene": gene, "gene_index": int(idx[j]), "n_null": n_sim, "rng_seed": rng_seed,
            "observed_residual_variance": observed_var[j],
            "poisson_null_variance": float(frozen_structure.iloc[j].counting_noise_variance),
            "nb_null_mean": vs["mean"][j], "nb_null_median": vs["median"][j],
            "nb_null_sd": vs["sd"][j], "nb_null_q95": vs["q95"][j],
            "observed_to_nb_median_ratio": observed_var[j] / max(vs["median"][j], 1e-16),
            "observed_minus_nb_median": observed_var[j] - vs["median"][j],
            "fraction_observed_variance_above_nb_median": max(observed_var[j] - vs["median"][j], 0) / max(observed_var[j], 1e-16),
            "null_percentile": vs["percentile"][j], "empirical_p_upper": vs["p_upper"][j],
            "observed_exceeds_nb_q95": bool(observed_var[j] > vs["q95"][j]),
            "source_alpha": alpha[j],
        })
        moran_rows.append({
            "target_key": run.key, "stage": run.stage, "target_embryo": run.target,
            "gene": gene, "gene_index": int(idx[j]), "n_null": n_sim, "rng_seed": rng_seed,
            "observed_moran_i": observed_moran[j], "nb_null_mean": ms["mean"][j],
            "nb_null_median": ms["median"][j], "nb_null_sd": ms["sd"][j],
            "nb_null_q95": ms["q95"][j], "moran_z": ms["z"][j],
            "null_percentile": ms["percentile"][j], "empirical_p_upper": ms["p_upper"][j],
            "observed_exceeds_nb_q95": bool(observed_moran[j] > ms["q95"][j]),
        })

    eta_rows = []
    for name, null in eta_null.items():
        summary = empirical_summary(null, observed_eta[name])
        for j, gene in enumerate(genes):
            eta_rows.append({
                "target_key": run.key, "stage": run.stage, "target_embryo": run.target,
                "annotation": name, "gene": gene, "gene_index": int(idx[j]),
                "n_null": n_sim, "rng_seed": rng_seed,
                "observed_eta_squared": observed_eta[name][j],
                "nb_null_mean": summary["mean"][j], "nb_null_median": summary["median"][j],
                "nb_null_sd": summary["sd"][j], "nb_null_q95": summary["q95"][j],
                "observed_to_nb_median_ratio": observed_eta[name][j] / max(summary["median"][j], 1e-16),
                "eta_z": summary["z"][j], "null_percentile": summary["percentile"][j],
                "empirical_p_upper": summary["p_upper"][j],
                "observed_exceeds_nb_q95": bool(observed_eta[name][j] > summary["q95"][j]),
            })
    return dispersion_all, variance_rows, moran_rows, eta_rows


def target_stage_summary(var: pd.DataFrame, moran: pd.DataFrame, eta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, keys in [("target", [[k] for k in [r.key for r in RUNS]]),
                        ("stage", [[r.key for r in RUNS if r.stage == s] for s in STAGE_ORDER])]:
        for group_keys in keys:
            v = var[var.target_key.isin(group_keys)]
            m = moran[moran.target_key.isin(group_keys)]
            e = eta[eta.target_key.isin(group_keys)]
            rows.append({
                "summary_level": level,
                "group": group_keys[0] if level == "target" else v.stage.iloc[0],
                "target_keys": ";".join(group_keys),
                "n_gene_target_records": len(v),
                "median_observed_to_nb_variance_ratio": v.observed_to_nb_median_ratio.median(),
                "fraction_variance_above_nb_q95": v.observed_exceeds_nb_q95.mean(),
                "median_fraction_variance_above_nb_median": v.fraction_observed_variance_above_nb_median.median(),
                "median_observed_moran_i": m.observed_moran_i.median(),
                "median_nb_null_moran_i": m.nb_null_median.median(),
                "median_moran_z": m.moran_z.median(),
                "fraction_moran_above_nb_q95": m.observed_exceeds_nb_q95.mean(),
                "n_gene_annotation_target_records": len(e),
                "median_observed_to_nb_eta2_ratio": e.observed_to_nb_median_ratio.median(),
                "fraction_eta2_above_nb_q95": e.observed_exceeds_nb_q95.mean(),
            })
    return pd.DataFrame(rows)


def figures(var: pd.DataFrame, moran: pd.DataFrame, eta: pd.DataFrame, summary: pd.DataFrame) -> None:
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    # R1
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    for ax, null_col, label in [(axes[0], "poisson_null_variance", "Poisson reference"),
                                (axes[1], "nb_null_median", "Source-calibrated NB median")]:
        ax.scatter(var[null_col], var.observed_residual_variance, s=4, alpha=.28, color=BLUE, rasterized=True)
        lo = max(min(var[null_col].min(), var.observed_residual_variance.min()), 1e-5)
        hi = max(var[null_col].max(), var.observed_residual_variance.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="#777", lw=.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(label); ax.set_ylabel("Observed residual variance")
    fig.suptitle("R1 | Residual variance against counting-noise references", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "R1_residual_variance_poisson_NB")

    # R2
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    groups = [moran[moran.target_key == r.key] for r in RUNS]
    pos = np.arange(6)
    ax.boxplot([g.observed_moran_i for g in groups], positions=pos-.16, widths=.26,
               patch_artist=True, showfliers=False,
               boxprops={"facecolor": BLUE, "alpha": .65}, medianprops={"color": "white"})
    ax.boxplot([g.nb_null_median for g in groups], positions=pos+.16, widths=.26,
               patch_artist=True, showfliers=False,
               boxprops={"facecolor": ORANGE, "alpha": .65}, medianprops={"color": "white"})
    ax.set_xticks(pos, [r.key.replace("_to_", "→").replace("p_", "% ").replace("6s_", "6s ") for r in RUNS],
                  rotation=28, ha="right")
    ax.set_ylabel("Moran’s I")
    ax.legend([plt.Rectangle((0,0),1,1,color=BLUE,alpha=.65), plt.Rectangle((0,0),1,1,color=ORANGE,alpha=.65)],
              ["Observed residual", "NB-null median"], frameon=False, ncol=2)
    ax.set_title("R2 | Spatial residual structure relative to source-calibrated NB null", fontweight="bold")
    ax.grid(axis="y", color="#E5E7E9", lw=.5)
    fig.tight_layout(); save_figure(fig, "R2_moran_NB_null")

    # R3
    pivot = eta.groupby(["target_key", "annotation"], sort=False).observed_to_nb_median_ratio.median().unstack()
    pivot = pivot.reindex([r.key for r in RUNS])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    shown = np.log10(np.maximum(pivot.to_numpy(), 1e-3))
    im = ax.imshow(shown, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot)), pivot.index)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            if np.isfinite(pivot.iloc[i, j]):
                ax.text(j, i, f"{pivot.iloc[i,j]:.1f}×", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, pad=.02, label="log10 observed/NB-null median η²")
    ax.set_title("R3 | Annotation-associated residual structure versus NB null", fontweight="bold")
    fig.tight_layout(); save_figure(fig, "R3_annotation_eta2_NB_null")

    # R4
    t = summary[summary.summary_level == "target"].set_index("group").loc[[r.key for r in RUNS]]
    matrix = np.vstack([t.fraction_variance_above_nb_q95,
                        t.fraction_moran_above_nb_q95,
                        t.fraction_eta2_above_nb_q95]).T
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3), ["Variance > NB q95", "Moran I > NB q95", "η² > NB q95"])
    ax.set_yticks(range(6), t.index)
    for i in range(6):
        for j in range(3):
            ax.text(j, i, f"{100*matrix[i,j]:.1f}%", ha="center", va="center",
                    color="white" if matrix[i,j]>.55 else "black")
    fig.colorbar(im, ax=ax, pad=.02, label="Fraction of technical records")
    ax.set_title("R4 | Six-target summary of enrichment above overdispersed null", fontweight="bold")
    fig.tight_layout(); save_figure(fig, "R4_six_target_summary")


def save_figure(fig, name: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def input_paths() -> list[Path]:
    paths = [
        Path(__file__),
        ROOT / "analysis/residual_landscape/io_frozen.py",
        ROOT / "analysis/residual_landscape/structure.py",
        ROOT / "analysis/residual_landscape/annotation.py",
        FROZEN_AUDIT / "02_structure/RESIDUAL_VARIANCE_DECOMPOSITION.tsv",
        FROZEN_AUDIT / "03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv",
        ROOT / "PI_revision/02A_alignment_benchmark_20260816_182905/ALIGNMENT_RESULTS.csv",
    ]
    for run in RUNS:
        paths += [run.source_file, run.target_file, run.manifest_path]
        manifest = json.loads(run.manifest_path.read_text())
        for seed in SEEDS:
            recorded = Path(manifest["methods"][str(seed)]["predictions"][IDW]["path"])
            paths.append(recorded if recorded.is_absolute() else ROOT / recorded)
    return list(dict.fromkeys(paths))


def simulate_run_wrapper(run_index: int, n_sim: int):
    """Process-safe target wrapper; target-specific RNG makes scheduling irrelevant."""
    run = RUNS[run_index]
    structure = pd.read_csv(FROZEN_AUDIT / "02_structure/RESIDUAL_VARIANCE_DECOMPOSITION.tsv", sep="\t")
    annotation = pd.read_csv(FROZEN_AUDIT / "03_annotation/RESIDUAL_ANNOTATION_ALIGNMENT_BY_GENE.tsv", sep="\t")
    fs = structure[structure.target_key == run.key].reset_index(drop=True)
    fa = annotation[annotation.target_key == run.key].reset_index(drop=True)
    print(f"[null] {run.key}", flush=True)
    return run_index, simulate_target(run, run_index, n_sim, fs, fa)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=500)
    args = parser.parse_args()
    if args.n_sim < 500:
        raise ValueError("Protocol requires at least 500 null realizations")
    paths = input_paths()
    hashes_before = {p: sha256(p) for p in paths}
    completed = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(simulate_run_wrapper, i, args.n_sim) for i in range(len(RUNS))]
        for future in concurrent.futures.as_completed(futures):
            i, result = future.result()
            completed[i] = result
    dispersions, variance_rows, moran_rows, eta_rows = [], [], [], []
    for i in range(len(RUNS)):
        d, v, m, e = completed[i]
        dispersions.append(d); variance_rows += v; moran_rows += m; eta_rows += e

    dispersion = pd.concat(dispersions, ignore_index=True)
    variance = pd.DataFrame(variance_rows)
    moran = pd.DataFrame(moran_rows)
    eta = pd.DataFrame(eta_rows)
    summary = target_stage_summary(variance, moran, eta)
    dispersion.to_csv(OUT / "SOURCE_DISPERSION_ESTIMATES.csv", index=False)
    variance.to_csv(OUT / "RESIDUAL_VARIANCE_NB_NULL.csv", index=False)
    moran.to_csv(OUT / "MORAN_NB_NULL.csv", index=False)
    eta.to_csv(OUT / "ANNOTATION_ETA2_NB_NULL.csv", index=False)
    summary.to_csv(OUT / "TARGET_STAGE_SUMMARY.csv", index=False)
    figures(variance, moran, eta, summary)

    hashes_after = {p: sha256(p) for p in paths}
    audit = pd.DataFrame([{"path": str(p.relative_to(ROOT)), "sha256_before": hashes_before[p],
                           "sha256_after": hashes_after[p], "unchanged": hashes_before[p] == hashes_after[p]}
                          for p in paths])
    audit.to_csv(OUT / "INPUT_HASHES.tsv", sep="\t", index=False)
    if not audit.unchanged.all():
        raise RuntimeError("Frozen input changed during analysis")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
