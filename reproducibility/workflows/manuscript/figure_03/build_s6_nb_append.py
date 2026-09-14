#!/usr/bin/env python3
"""Append frozen NB-null robustness panels to the authoritative S6 render.

The authoritative S6a--h PNG is embedded unchanged as the upper block.  This
script reads only serialized NB-null result tables and never invokes scientific
analysis code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
S6 = ROOT / "results_v10/supplement_nature_methods_20260809/08_residual_biology_robustness"
NB = ROOT / "PI_revision/03_overdispersed_residual_null_20260818_112038"

TARGETS = [
    "50p_E1_to_E2", "50p_E2_to_E1", "75p_E1_to_E2",
    "75p_E2_to_E1", "6s_E1_to_E2", "6s_E2_to_E1",
]
LABELS = [
    "50% E1→E2", "50% E2→E1", "75% E1→E2",
    "75% E2→E1", "6-somite E1→E2", "6-somite E2→E1",
]
COLORS = ["#0072B2", "#56B4E9", "#009E73", "#66C2A5", "#D55E00", "#E69F00"]
ANN_ORDER = ["germlayer", "type", "tissue", "clusters"]
ANN_LABELS = ["Germ layer", "Cell type", "Tissue", "Cluster"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_and_verify() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    variance = pd.read_csv(NB / "RESIDUAL_VARIANCE_NB_NULL.csv")
    moran = pd.read_csv(NB / "MORAN_NB_NULL.csv")
    eta = pd.read_csv(NB / "ANNOTATION_ETA2_NB_NULL.csv")
    assert len(variance) == 2742 and len(moran) == 2742 and len(eta) == 8126
    assert variance.target_key.drop_duplicates().tolist() == TARGETS
    assert moran.target_key.drop_duplicates().tolist() == TARGETS
    assert set(variance.n_null) == {500} and set(moran.n_null) == {500} and set(eta.n_null) == {500}
    stats = {
        "variance_fraction_q95": float(variance.observed_exceeds_nb_q95.mean()),
        "variance_median_excess": float(variance.fraction_observed_variance_above_nb_median.median()),
        "moran_fraction_q95": float(moran.observed_exceeds_nb_q95.mean()),
        "eta_median_enrichment": float(eta.observed_to_nb_median_ratio.median()),
        "eta_fraction_q95": float(eta.observed_exceeds_nb_q95.mean()),
        "poisson_median_excess": 0.8016834294319853,
    }
    assert round(100 * stats["variance_fraction_q95"], 1) == 91.6
    assert round(100 * stats["variance_median_excess"], 1) == 26.0
    assert round(100 * stats["moran_fraction_q95"], 2) == 99.96
    assert round(stats["eta_median_enrichment"], 2) == 17.47
    assert round(100 * stats["eta_fraction_q95"], 1) == 95.0
    return variance, moran, eta, stats


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.titlesize": 10.0, "axes.titleweight": "bold",
        "axes.labelsize": 8.0, "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def add_panel_letter(ax, letter: str) -> None:
    ax.text(-0.14, 1.09, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left")


def target_violins(ax, values: list[np.ndarray], log: bool = False) -> None:
    parts = ax.violinplot(values, positions=np.arange(6), widths=.76,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.48)
    bp = ax.boxplot(values, positions=np.arange(6), widths=.24, showfliers=False,
                    patch_artist=True, zorder=3,
                    medianprops={"color": "#222222", "linewidth": 1.0},
                    whiskerprops={"color": "#555555", "linewidth": .65},
                    capprops={"color": "#555555", "linewidth": .65})
    for box, color in zip(bp["boxes"], COLORS):
        box.set_facecolor(color); box.set_alpha(.73); box.set_linewidth(.65)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(np.arange(6), LABELS, rotation=35, ha="right")
    ax.grid(axis="y", color="#E4E7E9", linewidth=.5, zorder=0)


def panel_i(ax, variance: pd.DataFrame, stats: dict[str, float]) -> None:
    values = []
    for key in TARGETS:
        d = variance[variance.target_key == key]
        values.append((d.observed_residual_variance / d.nb_null_q95).to_numpy())
    target_violins(ax, values, log=True)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=.9)
    ax.text(5.42, 1.04, "NB q95", fontsize=6.5, color="#555555", ha="right", va="bottom")
    ax.set_ylim(.28, 14)
    ax.set_ylabel("Observed residual variance / NB q95")
    ax.set_title("Residual variance under overdispersed null", loc="left", pad=7)
    ax.text(.01, .98, "91.6% > NB q95\nmedian variance above NB null = 26.0%",
            transform=ax.transAxes, va="top", fontsize=7.0)
    ax.text(.99, .02, "Poisson excess: 80.2%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.3, color="#666666")
    add_panel_letter(ax, "I")


def panel_j(ax, moran: pd.DataFrame, stats: dict[str, float]) -> None:
    values = []
    for key in TARGETS:
        d = moran[moran.target_key == key]
        values.append((d.observed_moran_i / d.nb_null_q95).to_numpy())
    target_violins(ax, values, log=True)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=.9)
    ax.text(5.42, 1.04, "NB q95", fontsize=6.5, color="#555555", ha="right", va="bottom")
    ax.set_ylim(.6, 160)
    ax.set_ylabel("Observed Moran’s I / NB q95")
    ax.set_title("Spatial organization under overdispersed null", loc="left", pad=7)
    ax.text(.01, .98, "99.96% > NB Moran’s-I q95\nn = 2,742 gene×target records",
            transform=ax.transAxes, va="top", fontsize=7.0)
    add_panel_letter(ax, "J")


def eta_summary(eta: pd.DataFrame) -> pd.DataFrame:
    return (eta.groupby(["target_key", "annotation"], sort=False)
            .agg(n_records=("gene", "size"),
                 median_enrichment=("observed_to_nb_median_ratio", "median"),
                 fraction_above_q95=("observed_exceeds_nb_q95", "mean"))
            .reset_index())


def panel_k(ax, eta: pd.DataFrame, stats: dict[str, float]) -> None:
    summary = eta_summary(eta)
    ratio = summary.pivot(index="target_key", columns="annotation", values="median_enrichment").reindex(index=TARGETS, columns=ANN_ORDER)
    frac = summary.pivot(index="target_key", columns="annotation", values="fraction_above_q95").reindex(index=TARGETS, columns=ANN_ORDER)
    shown = np.log10(ratio.to_numpy(float))
    cmap = mpl.cm.get_cmap("YlGnBu").copy(); cmap.set_bad("#EFEFEF")
    im = ax.imshow(shown, cmap=cmap, norm=Normalize(vmin=0, vmax=2), aspect="auto")
    ax.set_xticks(np.arange(4), ANN_LABELS)
    ax.set_yticks(np.arange(6), LABELS)
    for i in range(6):
        for j in range(4):
            if np.isfinite(ratio.iloc[i, j]):
                color = "white" if shown[i, j] > 1.35 else "#222222"
                ax.text(j, i, f"{ratio.iloc[i,j]:.1f}×\n{100*frac.iloc[i,j]:.1f}% >q95",
                        ha="center", va="center", fontsize=5.5, color=color, linespacing=.95)
            else:
                ax.text(j, i, "—", ha="center", va="center", color="#999999")
    cb = plt.colorbar(im, ax=ax, pad=.025, fraction=.046)
    cb.set_label("log10 observed / NB-null median η²", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.8)
    ax.set_title("Annotation association under NB null", loc="left", pad=7)
    ax.text(.01, 1.01, "median enrichment = 17.47×; 95.0% > NB q95",
            transform=ax.transAxes, va="bottom", fontsize=6.8)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    add_panel_letter(ax, "K")


def save_standalone(panel_func, name: str, *args) -> None:
    fig, ax = plt.subplots(figsize=(4.36, 3.18))
    panel_func(ax, *args)
    fig.subplots_adjust(left=.17, right=.96, top=.83, bottom=.27)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_panel_data(variance: pd.DataFrame, moran: pd.DataFrame, eta: pd.DataFrame,
                     stats: dict[str, float]) -> None:
    i = variance[["target_key", "stage", "gene", "n_null", "observed_residual_variance",
                  "nb_null_median", "nb_null_q95", "observed_exceeds_nb_q95",
                  "fraction_observed_variance_above_nb_median"]].copy()
    i.insert(0, "panel", "S6i")
    i["plotted_value"] = i.observed_residual_variance / i.nb_null_q95
    j = moran[["target_key", "stage", "gene", "n_null", "observed_moran_i",
               "nb_null_median", "nb_null_q95", "observed_exceeds_nb_q95"]].copy()
    j.insert(0, "panel", "S6j")
    j["plotted_value"] = j.observed_moran_i / j.nb_null_q95
    k = eta_summary(eta)
    k.insert(0, "panel", "S6k")
    k = k.rename(columns={"median_enrichment": "plotted_value"})
    combined = pd.concat([i, j, k], ignore_index=True, sort=False)
    combined.to_csv(OUT / "PANEL_DATA.csv", index=False)


def make_row(variance: pd.DataFrame, moran: pd.DataFrame, eta: pd.DataFrame,
             stats: dict[str, float]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.466, 3.55), gridspec_kw={"wspace": .36})
    panel_i(axes[0], variance, stats)
    panel_j(axes[1], moran, stats)
    panel_k(axes[2], eta, stats)
    fig.suptitle("Source-calibrated negative-binomial residual-null analyses",
                 y=.985, fontsize=11.5, fontweight="bold")
    fig.text(.5, .012,
             "The directional target embryo is the biological evaluation unit; gene×target and gene×annotation×target records are technical units, not independent biological replicates. 500 null realizations per target (3,000 total).",
             ha="center", fontsize=6.6, color="#555555")
    fig.subplots_adjust(left=.055, right=.985, bottom=.245, top=.80)
    row = OUT / "S6i-k_appended_row.png"
    fig.savefig(row, dpi=300, facecolor="white")
    fig.savefig(OUT / "S6i-k_appended_row.pdf", facecolor="white")
    fig.savefig(OUT / "S6i-k_appended_row.svg", facecolor="white")
    plt.close(fig)
    return row


def compose(row_path: Path) -> None:
    original = Image.open(S6 / "Supp_Fig_S6_residual_biology_robustness.png").convert("RGB")
    row = Image.open(row_path).convert("RGB")
    if row.width != original.width:
        new_h = round(row.height * original.width / row.width)
        row = row.resize((original.width, new_h), Image.Resampling.LANCZOS)
    separator = 10
    composite = Image.new("RGB", (original.width, original.height + separator + row.height), "white")
    composite.paste(original, (0, 0))
    composite.paste(row, (0, original.height + separator))
    png = OUT / "Supplementary_Figure_S6_residual_biology_robustness_with_NB_null.png"
    composite.save(png, dpi=(300, 300), optimize=True)

    # High-resolution review PDF. Original a--h pixels are embedded without
    # resampling; standalone new-panel PDFs remain vector-editable.
    fig = plt.figure(figsize=(composite.width / 300, composite.height / 300))
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(composite); ax.axis("off")
    fig.savefig(OUT / "Supplementary_Figure_S6_residual_biology_robustness_with_NB_null.pdf",
                dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    style()
    variance, moran, eta, stats = load_and_verify()
    write_panel_data(variance, moran, eta, stats)
    save_standalone(panel_i, "S6i_residual_variance_NB_null", variance, stats)
    save_standalone(panel_j, "S6j_moran_NB_null", moran, stats)
    save_standalone(panel_k, "S6k_annotation_eta2_NB_null", eta, stats)
    row = make_row(variance, moran, eta, stats)
    compose(row)
    print(stats)


if __name__ == "__main__":
    main()
