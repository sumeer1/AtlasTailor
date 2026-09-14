#!/usr/bin/env python3
"""Plot-only Main Figure 2 revision from locked MAP/gimVI/TransImpSpa outputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
AUTH_ROOT = REPO / "results_v10/hyperspatial_map_figure_revision_20260804_142208"
AUTH_SCRIPT = AUTH_ROOT / "build_figure_revision.py"
GIMVI = REPO / "results_v10/hyperspatial_map_gimvi_baseline_20260805_095807"
TRANSIMP = REPO / "PI_revision/02B_spatial_reconstruction_comparator_20260817_132013"
S3 = REPO / "results_v10/supplement_nature_methods_20260809/03_full_ablations"

spec = importlib.util.spec_from_file_location("authoritative_figure_revision", AUTH_SCRIPT)
AUTH = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(AUTH)

MAP = AUTH.MAP
IDW = AUTH.IDW
ORDER = ["50p_E1_to_E2", "50p_E2_to_E1", "75p_E1_to_E2", "75p_E2_to_E1",
         "6s_E1_to_E2", "6s_E2_to_E1"]
SHORT_LABELS = ["50% E1→E2", "50% E2→E1", "75% E1→E2", "75% E2→E1",
                "6s E1→E2", "6s E2→E1"]
METHOD_ORDER = [IDW, "gimvi_adapted", "transimpspa", MAP]
LABELS = {
    IDW: "Registered 12-NN IDW",
    "gimvi_adapted": "adapted gimVI",
    "transimpspa": "TransImpSpa",
    MAP: "HyperSpatial-MAP",
}
COLORS = {
    IDW: "#E69F00",
    "gimvi_adapted": "#56B4E9",
    "transimpspa": "#6A3D9A",
    MAP: "#C83E4D",
    "grid": "#E4E7EA",
    "text": "#252A2E",
}
METRICS = [
    ("spatial_pearson", "Spatial Pearson ↑", True),
    ("pooled_residual_pearson", "Pooled cell×gene\nresidual Pearson ↑", True),
    ("log1p_rmse", "log1p RMSE ↓", False),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": .9,
        "lines.linewidth": 1.0, "lines.markersize": 4.5,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.facecolor": "white", "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"], "axes.edgecolor": COLORS["text"],
    })


def polish(ax, grid_axis=None):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=.55, zorder=0)
    ax.tick_params(direction="out")


def panel_label(ax, label, x=-.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", clip_on=False)


def export(fig, stem: str, png: bool = True):
    exts = ("pdf", "svg", "png") if png else ("pdf", "svg")
    for ext in exts:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=.10, facecolor="white")
    plt.close(fig)


def input_files() -> list[Path]:
    files = [
        AUTH_SCRIPT,
        AUTH.R3 / "05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv",
        AUTH.R3 / "05_metrics/METHOD_COMPARISONS_BY_EMBRYO.tsv",
        AUTH.R3 / "06_hierarchical_statistics/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv",
        AUTH_ROOT / "03_figure2_3d_replication/Figure_2_3D_replication.pdf",
        AUTH_ROOT / "03_figure2_3d_replication/Figure_2_3D_replication.svg",
        AUTH_ROOT / "03_figure2_3d_replication/Figure_2_3D_replication.png",
        GIMVI / "03_metrics/PER_TARGET_METRICS.tsv",
        GIMVI / "03_metrics/PER_GENE_METRICS.tsv",
        GIMVI / "01_protocol/PREDICTIONS_LOCKED.json",
        TRANSIMP / "metrics/RESULTS_BY_TARGET_LONG.csv",
        TRANSIMP / "GENE_LEVEL_RESULTS.csv",
        TRANSIMP / "protocol/PREDICTIONS_LOCKED.json",
        S3 / "Supp_Fig_S3_full_ablations.pdf",
        S3 / "Supp_Fig_S3_full_ablations.svg",
        S3 / "Supp_Fig_S3_full_ablations.png",
    ]
    for run in AUTH.RUNS:
        run_path = Path(run[4])
        manifest = run_path / "prediction_manifest.json"
        files.append(manifest)
        frozen = json.loads(manifest.read_text())
        paths = frozen["methods"]["42"]["predictions"]
        files.extend([Path(paths[IDW]["path"]), Path(paths[MAP]["path"])])
    for key in ORDER:
        gm = GIMVI / "01_protocol" / f"{key}_PREDICTION_MANIFEST.json"
        tm = TRANSIMP / "protocol" / f"{key}_PREDICTION_MANIFEST.json"
        files.extend([gm, tm])
        files.extend([Path(json.loads(gm.read_text())["prediction"]),
                      Path(json.loads(tm.read_text())["prediction"])])
    unique = []
    for path in files:
        path = path.resolve()
        if path not in unique:
            unique.append(path)
    return unique


def load_external_values() -> pd.DataFrame:
    gimvi = pd.read_csv(GIMVI / "03_metrics/PER_TARGET_METRICS.tsv", sep="\t")
    trans = pd.read_csv(TRANSIMP / "metrics/RESULTS_BY_TARGET_LONG.csv")
    rows = []
    source_cols = {
        "spatial_pearson": "svg_spatial_pearson_median",
        "pooled_residual_pearson": "pooled_cell_gene_residual_pearson",
        "log1p_rmse": "log1p_rmse",
    }
    for key in ORDER:
        for method in METHOD_ORDER:
            for metric, _, higher in METRICS:
                if method == IDW and metric == "pooled_residual_pearson":
                    value = np.nan
                    source_file = TRANSIMP / "metrics/RESULTS_BY_TARGET_LONG.csv"
                    source_col = source_cols[metric]
                    note = "undefined: registered IDW predicts zero target-specific residual"
                elif method == "gimvi_adapted":
                    row = gimvi[(gimvi.target_key == key) & (gimvi.method == method)].iloc[0]
                    source_col = "rho_res_pooled" if metric == "pooled_residual_pearson" else source_cols[metric]
                    value = float(row[source_col])
                    source_file = GIMVI / "03_metrics/PER_TARGET_METRICS.tsv"
                    note = "frozen audited gimVI benchmark"
                else:
                    row = trans[(trans.target_key == key) & (trans.method == method)].iloc[0]
                    source_col = source_cols[metric]
                    value = float(row[source_col])
                    source_file = TRANSIMP / "metrics/RESULTS_BY_TARGET_LONG.csv"
                    note = "completed frozen TransImpSpa matched benchmark"
                rows.append({
                    "evaluation": key, "display_evaluation": SHORT_LABELS[ORDER.index(key)],
                    "method": method, "display_method": LABELS[method], "metric": metric,
                    "value": value, "higher_is_better": higher,
                    "source_file": str(source_file.resolve()), "source_column": source_col,
                    "notes": note,
                })
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "FIG2D_VALUES_AUDIT.csv", index=False)
    return audit


def verify_external_claims(values: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for metric, _, higher in METRICS:
        pivot = values[values.metric == metric].pivot(index="evaluation", columns="method", values="value").reindex(ORDER)
        if metric == "pooled_residual_pearson":
            counts["MAP_gt_TransImpSpa_pooled_residual"] = int((pivot[MAP] > pivot["transimpspa"]).sum())
            counts["MAP_gt_gimVI_pooled_residual"] = int((pivot[MAP] > pivot["gimvi_adapted"]).sum())
        elif higher:
            counts[f"MAP_gt_TransImpSpa_{metric}"] = int((pivot[MAP] > pivot["transimpspa"]).sum())
        else:
            counts[f"MAP_lt_TransImpSpa_{metric}"] = int((pivot[MAP] < pivot["transimpspa"]).sum())
    expected = {
        "MAP_gt_TransImpSpa_spatial_pearson": 6,
        "MAP_gt_TransImpSpa_pooled_residual": 6,
        "MAP_gt_gimVI_pooled_residual": 6,
        "MAP_lt_TransImpSpa_log1p_rmse": 6,
    }
    if any(counts.get(k) != v for k, v in expected.items()):
        raise AssertionError(f"External-benchmark direction check failed: {counts}")
    return counts


def draw_external_benchmark(container, values, label="d", standalone=False,
                            compact_roomy=False):
    if standalone:
        axes = container.subplots(1, 3)
        fig = container
        header = None
    else:
        # In the compact main figure, retain a shallow title/legend header so
        # the three quantitative axes use the available panel height.  Their
        # box aspect is fixed below: added room enlarges rather than stretches
        # the plots.
        if compact_roomy:
            nested = container.subgridspec(2, 3, height_ratios=[.34, 1],
                                           hspace=.10, wspace=.34)
        else:
            nested = container.subgridspec(2, 3, height_ratios=[.75, 1],
                                           hspace=.34, wspace=.42)
        header = plt.gcf().add_subplot(nested[0, :])
        header.axis("off")
        axes = np.asarray([plt.gcf().add_subplot(nested[1, j]) for j in range(3)])
        fig = axes[0].figure
    for j, (metric, title, _) in enumerate(METRICS):
        ax = axes[j]
        for method in METHOD_ORDER:
            if metric == "pooled_residual_pearson" and method == IDW:
                continue
            q = values[(values.metric == metric) & (values.method == method)].set_index("evaluation").reindex(ORDER)
            ax.plot(np.arange(6), q.value, "-o", color=COLORS[method], lw=1.0, ms=3.8,
                    markerfacecolor="white", markeredgewidth=.9, label=LABELS[method], zorder=3)
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xticks(np.arange(6), ["50 E2", "50 E1", "75 E2", "75 E1", "6s E2", "6s E1"],
                      rotation=36, ha="right", fontsize=5.7)
        if compact_roomy and not standalone:
            ax.set_box_aspect(.82)
        polish(ax, "y")
        if metric == "pooled_residual_pearson":
            ax.text(.02, .02, "IDW residual r undefined", transform=ax.transAxes,
                    fontsize=5.4, color="#626A70", va="bottom")
    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", markerfacecolor="white",
                          lw=1, label=LABELS[m]) for m in METHOD_ORDER]
    if standalone:
        fig.suptitle("External molecular-reconstruction benchmarks", fontsize=9, fontweight="bold", y=.99)
        fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.5, .92),
                   frameon=False, fontsize=7)
        fig.subplots_adjust(left=.07, right=.99, bottom=.22, top=.76, wspace=.38)
    else:
        panel_label(header, label, -.045, 1.12)
        header.text(.5, .96, "External molecular-reconstruction benchmarks",
                    ha="center", va="top", fontsize=8, fontweight="bold")
        header.legend(handles=handles, ncol=4, loc="lower center",
                      bbox_to_anchor=(.5, -.02 if compact_roomy else .00),
                      frameon=False, fontsize=6.1, columnspacing=1.0, handletextpad=.35)
    return axes


def read_wnt11f2_maps(run_index=2, gene="wnt11f2") -> dict:
    # The original audited gimVI plotting implementation hard-coded RUNS[2].
    # Optional indices expose the same frozen map construction for requested
    # stage-matched diagnostic panels; the default remains the original panel.
    run = AUTH.RUNS[run_index]
    stage, tag, source_id, target_id, run_path, source_file, target_file = run
    key = ORDER[run_index]
    expected_key = f"{tag}_{source_id}_to_{target_id}"
    if key != expected_key:
        raise AssertionError(f"Authoritative run/key mismatch: {expected_key} != {key}")
    frozen = json.loads((Path(run_path) / "prediction_manifest.json").read_text())
    source, _, genes = __import__("run_wemerfish_temporal_holdout").measured_counts(Path(source_file))
    target, xyz, target_genes = __import__("run_wemerfish_temporal_holdout").measured_counts(Path(target_file))
    if not np.array_equal(genes, target_genes):
        raise AssertionError("Source/target gene order mismatch")
    gi = int(np.flatnonzero(genes == gene)[0])
    if gi in set(frozen["anchor_indices"]):
        raise AssertionError(f"{gene} is an anchor, not a hidden gene")
    if gi not in set(frozen.get("non_anchor_indices", [i for i in range(len(genes)) if i not in frozen["anchor_indices"]])):
        raise AssertionError(f"{gene} is not in the frozen hidden-gene universe")
    helper = __import__("run_wemerfish_temporal_holdout")
    coords = helper.canonicalize(xyz)[0]
    depth = float(np.median(np.maximum(source.sum(1), 1)))
    truth = np.log1p(helper.normalize_counts(target, depth))[:, gi]
    pred_paths = frozen["methods"]["42"]["predictions"]
    gimvi_manifest = json.loads((GIMVI / f"01_protocol/{key}_PREDICTION_MANIFEST.json").read_text())
    trans_manifest = json.loads((TRANSIMP / f"protocol/{key}_PREDICTION_MANIFEST.json").read_text())
    paths = {
        IDW: Path(pred_paths[IDW]["path"]),
        "gimvi_adapted": Path(gimvi_manifest["prediction"]),
        "transimpspa": Path(trans_manifest["prediction"]),
        MAP: Path(pred_paths[MAP]["path"]),
    }
    maps = {"measured": truth}
    for method, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        maps[method] = np.asarray(np.load(path, mmap_mode="r")[:, gi], dtype=np.float32)
        if maps[method].shape != truth.shape:
            raise AssertionError(
                f"Invalid prediction shape for {method}: "
                f"prediction={maps[method].shape}, target={truth.shape}, key={key}"
            )
    # Match the exact shared-scale rule in the prior audited comparator panel f:
    # measured target + MAP + gimVI + TransImpSpa, pooled 99th percentile.
    # Registered IDW is displayed on this frozen common scale but does not alter it.
    scale_values = np.concatenate([
        maps["measured"], maps[MAP], maps["gimvi_adapted"], maps["transimpspa"]
    ])
    vmin, vmax = 0.0, float(np.nanquantile(scale_values, .99))
    gene_metrics = pd.read_csv(GIMVI / "03_metrics/PER_GENE_METRICS.tsv", sep="\t")
    trans_gene = pd.read_csv(TRANSIMP / "GENE_LEVEL_RESULTS.csv")
    spatial_r = {"measured": np.nan}
    for method in (IDW, "gimvi_adapted", MAP):
        row = gene_metrics[(gene_metrics.target_key == key) & (gene_metrics.method == method) &
                           (gene_metrics.gene == gene)].iloc[0]
        spatial_r[method] = float(row.spatial_pearson)
    row = trans_gene[(trans_gene.target_key == key) & (trans_gene.gene == gene)].iloc[0]
    spatial_r["transimpspa"] = float(row.spatial_pearson_transimpspa)
    return {
        "target_key": key, "stage": stage, "source": source_id, "target": target_id,
        "gene": gene, "gene_index": gi, "coords": coords, "maps": maps,
        "paths": paths, "source_file": Path(source_file), "target_file": Path(target_file),
        "vmin": vmin, "vmax": vmax, "spatial_r": spatial_r,
        "selection_evidence": (
            "hard-coded as RUNS[2] and wnt11f2 in original audited gimVI plot()"
            if run_index == 2 and gene == "wnt11f2"
            else "stage/direction fixed by requested cross-stage rendering; gene retained from original panel"
        ),
    }


def draw_wnt11f2(container, data, label="e", standalone=False, side_panel=False,
                 include_idw=False):
    if standalone:
        fig = container
        host = fig.add_axes([.02, .04, .96, .88])
    else:
        fig = container.get_gridspec().figure
        # The original audited 02C panel f occupied one ~2.15-inch GridSpec cell.
        # Preserve that physical rendering size: with four .235-width insets, each
        # molecular map is ~0.50 inches wide.  Expanding the maps made the same
        # frozen point cloud appear artificially smoother.
        if side_panel:
            host_width = 3.15 if include_idw else 2.55
            margins = [.02, host_width, .02]
        else:
            margins = [2.30, 2.15, 2.30]
        centered = container.subgridspec(1, 3, width_ratios=margins, wspace=0)
        host = fig.add_subplot(centered[0, 1])
    host.axis("off")
    # Exact visual content/order of the audited 02C panel f.
    if include_idw:
        names = ["Measured", "IDW", "gimVI", "TransImpSpa", "MAP"]
        keys = ["measured", IDW, "gimvi_adapted", "transimpspa", MAP]
    else:
        names = ["Measured", "MAP", "gimVI", "TransImpSpa"]
        keys = ["measured", MAP, "gimvi_adapted", "transimpspa"]
    step = 1.0 / len(names)
    inset_width = .94 * step
    axes = []
    for i, (name, key) in enumerate(zip(names, keys)):
        ax = host.inset_axes([i * step, .04, inset_width, .88]); axes.append(ax)
        marker_size = .35 if side_panel else .25
        ax.scatter(data["coords"][:, 0], data["coords"][:, 1], c=data["maps"][key],
                   s=marker_size, cmap="viridis", vmin=data["vmin"], vmax=data["vmax"],
                   rasterized=True)
        ax.axis("off"); ax.set_aspect("equal")
        ax.set_title(name, fontsize=6.5, pad=2)
    host.set_title(
        f"Source-predeclared {data['gene']} · {data['stage']} "
        f"{data['source']}→{data['target']} · shared scale",
        fontsize=7,
    )
    if not standalone:
        panel_label(host, label, -.10, 1.08)
    return axes


def revised_main_figure(values, wnt, stem="Figure_2_revised"):
    embryo = pd.read_csv(AUTH.R3 / "05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv", sep="\t")
    comp = pd.read_csv(AUTH.R3 / "05_metrics/METHOD_COMPARISONS_BY_EMBRYO.tsv", sep="\t")
    boot = pd.read_csv(AUTH.R3 / "06_hierarchical_statistics/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv", sep="\t")

    fig = plt.figure(figsize=(AUTH.TWO_COLUMN, 11.85))
    gs = fig.add_gridspec(6, 2, height_ratios=[.48, 2.28, 1.15, 2.00, 1.35, 1.10],
                          hspace=.82, wspace=.46)

    # a: authoritative reciprocal design logic.
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); panel_label(ax, "a", -.02, 1.15)
    fills = ["#EBF2F8", "#EBF2F8", "#EDF6F2", "#EDF6F2", "#FAEFE9", "#FAEFE9"]
    for i, ((_, tag, source, target, *_), fill) in enumerate(zip(AUTH.RUNS, fills)):
        x = .015 + i * .165
        ax.add_patch(FancyBboxPatch((x, .18), .145, .60, boxstyle="round,pad=.01",
                                    fc=fill, ec="#9AA1A8", lw=.8))
        ax.text(x + .072, .49, f"{tag}\n{source} → {target}", ha="center", va="center", fontsize=7)

    # b: exact authoritative six endpoint values.
    sub = gs[1, :].subgridspec(2, 3, wspace=.45, hspace=.72)
    for j, (endpoint, title, higher) in enumerate(AUTH.ENDPOINTS):
        ax = fig.add_subplot(sub[j // 3, j % 3])
        for key in embryo.target_key.unique():
            d = embryo[embryo.target_key == key].set_index("method")
            ax.plot([0, 1], [d.loc[IDW, endpoint], d.loc[MAP, endpoint]], color="#B8BDC2", lw=.85, zorder=1)
            ax.scatter(0, d.loc[IDW, endpoint], c=AUTH.COLORS["idw"], s=24, zorder=3)
            ax.scatter(1, d.loc[MAP, endpoint], c=AUTH.COLORS["map"], s=24, zorder=3)
        ax.set_xticks([0, 1], ["IDW", "MAP"]); ax.set_title(title + (" ↑" if higher else " ↓"))
        polish(ax, "y")
        if j == 0:
            panel_label(ax, "b", -.34, 1.25)

    # c: exact authoritative target-level favorable effects and intervals.
    ax = fig.add_subplot(gs[2, :]); panel_label(ax, "c", -.085, 1.11)
    b = boot[(boot.comparison.str.contains("favorable sign")) &
             boot.endpoint.isin([x[0] for x in AUTH.ENDPOINTS])]
    y = np.arange(len(b))
    ax.errorbar(b.estimate, y, xerr=[b.estimate - b.ci95_low, b.ci95_high - b.estimate],
                fmt="o", color=AUTH.COLORS["map"], ecolor="#555D64", capsize=2.5, ms=5)
    ax.axvline(0, color="#777", ls=":", lw=.8)
    label_map = {endpoint: title for endpoint, title, _ in AUTH.ENDPOINTS}
    ax.set_yticks(y, [label_map[e] for e in b.endpoint]); ax.invert_yaxis()
    ax.set_xlabel("Favourable MAP − IDW effect"); polish(ax, "x")

    # d: external/reference approaches only.
    draw_external_benchmark(gs[3, :], values, label="d")

    # e: fixed source-predeclared map comparison.
    draw_wnt11f2(gs[4, :], wnt, label="e")

    # f: authoritative stage-stratified MAP−IDW values.
    ax = fig.add_subplot(gs[5, :]); panel_label(ax, "f", -.035, 1.10)
    p = comp[comp.comparison == f"{MAP}_vs_{IDW}"]
    for endpoint, title, color in [
        ("svg_spatial_pearson_median", "Pearson", "#277DA1"),
        ("topk_dice_median", "Dice", AUTH.COLORS["map"]),
        ("log1p_rmse", "RMSE", AUTH.COLORS["local"]),
    ]:
        q = p[p.endpoint == endpoint].groupby("stage").delta.mean().reindex(
            ["50% epiboly", "75% epiboly", "6-somite"])
        ax.plot(range(3), q, "-o", color=color, label=title)
    ax.axhline(0, color="#777", ls=":", lw=.8)
    ax.set_xticks(range(3), ["50%", "75%", "6-somite"])
    ax.set_ylabel("Favourable MAP − IDW\neffect", labelpad=1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.02))
    polish(ax, "y")
    export(fig, stem, png=True)


def compact_main_figure(values, wnt, stem="Figure_2_revised_compact",
                        roomy_panel_d=False):
    """Old-Figure-2-like compact layout; numerical panel content is unchanged."""
    embryo = pd.read_csv(AUTH.R3 / "05_metrics/PRIMARY_ENDPOINTS_BY_EMBRYO.tsv", sep="\t")
    comp = pd.read_csv(AUTH.R3 / "05_metrics/METHOD_COMPARISONS_BY_EMBRYO.tsv", sep="\t")
    boot = pd.read_csv(AUTH.R3 / "06_hierarchical_statistics/HIERARCHICAL_BOOTSTRAP_RESULTS.tsv", sep="\t")

    fig = plt.figure(figsize=(AUTH.TWO_COLUMN, 7.95))
    gs = fig.add_gridspec(4, 1, height_ratios=[.38, 2.35, 1.58, 1.34],
                          hspace=.74, left=.085, right=.985, top=.975, bottom=.07)
    middle = gs[2, 0].subgridspec(
        1, 2,
        width_ratios=[2.25, 4.45] if roomy_panel_d else [2.55, 4.15],
        wspace=.38 if roomy_panel_d else .48,
    )
    bottom = gs[3, 0].subgridspec(1, 2, width_ratios=[3.15, 3.55], wspace=.42)

    # a — unchanged reciprocal evaluation design.
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); panel_label(ax, "a", -.02, 1.18)
    fills = ["#EBF2F8", "#EBF2F8", "#EDF6F2", "#EDF6F2", "#FAEFE9", "#FAEFE9"]
    for i, ((_, tag, source, target, *_), fill) in enumerate(zip(AUTH.RUNS, fills)):
        x = .015 + i * .165
        ax.add_patch(FancyBboxPatch((x, .18), .145, .60, boxstyle="round,pad=.01",
                                    fc=fill, ec="#9AA1A8", lw=.8))
        ax.text(x + .072, .49, f"{tag}\n{source} → {target}", ha="center", va="center", fontsize=7)

    # b — unchanged paired endpoint values, restored to the original compact 2x3 block.
    sub = gs[1, 0].subgridspec(2, 3, wspace=.45, hspace=.68)
    for j, (endpoint, title, higher) in enumerate(AUTH.ENDPOINTS):
        ax = fig.add_subplot(sub[j // 3, j % 3])
        for key in embryo.target_key.unique():
            d = embryo[embryo.target_key == key].set_index("method")
            ax.plot([0, 1], [d.loc[IDW, endpoint], d.loc[MAP, endpoint]],
                    color="#B8BDC2", lw=.85, zorder=1)
            ax.scatter(0, d.loc[IDW, endpoint], c=AUTH.COLORS["idw"], s=24, zorder=3)
            ax.scatter(1, d.loc[MAP, endpoint], c=AUTH.COLORS["map"], s=24, zorder=3)
        ax.set_xticks([0, 1], ["IDW", "MAP"]); ax.set_title(title + (" ↑" if higher else " ↓"))
        polish(ax, "y")
        if j == 0:
            panel_label(ax, "b", -.34, 1.24)

    # c — unchanged favorable MAP-IDW effects, paired with d as in the old main-figure grammar.
    ax = fig.add_subplot(middle[0, 0]); panel_label(ax, "c", -.22, 1.14)
    b = boot[(boot.comparison.str.contains("favorable sign")) &
             boot.endpoint.isin([x[0] for x in AUTH.ENDPOINTS])]
    y = np.arange(len(b))
    ax.errorbar(b.estimate, y, xerr=[b.estimate - b.ci95_low, b.ci95_high - b.estimate],
                fmt="o", color=AUTH.COLORS["map"], ecolor="#555D64", capsize=2.5, ms=4.5)
    ax.axvline(0, color="#777", ls=":", lw=.8)
    label_map = {endpoint: title for endpoint, title, _ in AUTH.ENDPOINTS}
    ax.set_yticks(y, [label_map[e] for e in b.endpoint], fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("Favourable MAP − IDW effect", fontsize=7); polish(ax, "x")

    # d — same frozen external benchmark, no internal anchor-only series.
    draw_external_benchmark(middle[0, 1], values, label="d",
                            compact_roomy=roomy_panel_d)

    # e — exact 02C panel-f rendering scale and point texture in a compact left cell.
    draw_wnt11f2(bottom[0, 0], wnt, label="e", side_panel=True, include_idw=True)

    # f — unchanged stage-stratified MAP-IDW content.
    ax = fig.add_subplot(bottom[0, 1]); panel_label(ax, "f", -.13, 1.10)
    p = comp[comp.comparison == f"{MAP}_vs_{IDW}"]
    for endpoint, title, color in [
        ("svg_spatial_pearson_median", "Pearson", "#277DA1"),
        ("topk_dice_median", "Dice", AUTH.COLORS["map"]),
        ("log1p_rmse", "RMSE", AUTH.COLORS["local"]),
    ]:
        q = p[p.endpoint == endpoint].groupby("stage").delta.mean().reindex(
            ["50% epiboly", "75% epiboly", "6-somite"])
        ax.plot(range(3), q, "-o", color=color, label=title)
    ax.axhline(0, color="#777", ls=":", lw=.8)
    ax.set_xticks(range(3), ["50%", "75%", "6-somite"])
    ax.set_ylabel("Favourable MAP − IDW effect", fontsize=7, labelpad=2)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.04), fontsize=6.5)
    polish(ax, "y")
    export(fig, stem, png=True)


def write_audits(wnt, claims, hashes_before, hashes_after):
    lines = ["# Figure 2e wnt11f2 audit", "",
             "- **Gene:** `wnt11f2`.",
             "- **Direction:** `75p_E1_to_E2` (75% epiboly E1→E2).",
             "- **Selection evidence:** the original audited gimVI `plot()` function fixes `r=RUNS[2]` and then fixes `genes==\"wnt11f2\"`; neither the TransImpSpa result nor target ranking enters this choice.",
             f"- **Gene index:** {wnt['gene_index']}.",
             "- **Eligibility:** verified present in the frozen non-anchor/hidden gene universe and absent from the 16 anchors.",
             "- **Predictions displayed:** locked HyperSpatial-MAP, adapted-gimVI and TransImpSpa arrays with target-cell ordering matching the measured target. Registered IDW is intentionally not displayed, matching the exact 02C panel-f content.",
             "- **Expression representation:** target counts normalized to source median depth and transformed by `log1p`, matching the audited gimVI representative panel; prediction arrays are the locked log1p-scale outputs.",
             f"- **Shared display limits:** vmin = {wnt['vmin']:.9g}; vmax = {wnt['vmax']:.9g}.",
             "- **Scale rule:** exact prior audited panel-f rule and float32 pathway: lower limit zero and pooled 99th-percentile upper limit calculated once from measured target, HyperSpatial-MAP, adapted gimVI and TransImpSpa. No method-specific scaling.",
             "- **Residual row:** omitted to preserve main-figure readability.", "",
             "## Prediction paths", ""]
    lines.append(f"- Measured target: `{wnt['target_file']}`")
    for method in METHOD_ORDER:
        lines.append(f"- {LABELS[method]}: `{wnt['paths'][method]}`")
    (OUT / "FIG2E_WNT11F2_AUDIT.md").write_text("\n".join(lines) + "\n")

    layout = """# Figure 2 layout changes

- Panels a–c were regenerated from the same authoritative source tables and retain their numerical content.
- Previous internal component panels d/e were removed from the main figure only.
- New panel d contains only registered 12-NN IDW, adapted gimVI, TransImpSpa and HyperSpatial-MAP.
- New panel e reproduces the exact 02C panel-f map content and order: Measured, MAP, gimVI and TransImpSpa under one shared expression scale, without per-map spatial-Pearson annotations.
- The former stage-stratified panel f remains panel f with unchanged numerical content.
- Anchor-only, global-residual and local-residual analyses remain exclusively in Supplementary Fig. S3, which was not modified.
- No model, registration, anchor panel, prediction, metric table or authoritative figure was overwritten.
"""
    (OUT / "FIGURE2_LAYOUT_CHANGES.md").write_text(layout)

    legend = """Figure 2 | Frozen reciprocal three-dimensional atlas adaptation and external molecular-reconstruction benchmarks.

a, Six prespecified reciprocal source-to-target evaluations at 50% epiboly, 75% epiboly and 6-somite. b, Paired target-volume comparison of registered 12-nearest-neighbour inverse-distance weighting (IDW) and HyperSpatial-MAP for six reconstruction endpoints. c, Favourably oriented HyperSpatial-MAP-minus-IDW effects; points and horizontal intervals retain the statistics in the authoritative figure. d, Comparison with external molecular-reconstruction approaches under matched sparse target measurements. Absolute spatial Pearson, pooled cell-by-gene residual Pearson and log1p RMSE are shown across the six reciprocal evaluations for registered 12-NN IDW, adapted gimVI, TransImpSpa and HyperSpatial-MAP, where applicable. TransImpSpa used the target three-dimensional spatial-neighbourhood graph, whereas gimVI did not use target coordinates. HyperSpatial-MAP achieved higher spatial Pearson and pooled residual Pearson and lower RMSE than TransImpSpa in all six target-wise evaluations, and higher pooled residual Pearson than adapted gimVI in all six. TransImpSpa nevertheless retained isolated advantages, including higher median gene-wise residual Pearson in two of six directions. Registered IDW is omitted from residual-correlation comparisons because it predicts no target-specific residual. e, Representative reconstruction of the source-predeclared hidden gene wnt11f2 in the fixed 75% E1→E2 evaluation. Measured target expression and predictions from HyperSpatial-MAP, adapted gimVI and TransImpSpa are displayed using the exact common scale and map order of the audited comparator panel. No per-map spatial-Pearson values are displayed. The gene and representative evaluation were specified independently of target benchmark performance. f, Stage-stratified favourable HyperSpatial-MAP-minus-registered-IDW effects for spatial Pearson, prevalence-matched Dice and log1p RMSE. Reciprocal directions share specimens and are not six independent biological replicates.
"""
    (OUT / "FIGURE2_REVISED_LEGEND.txt").write_text(legend)

    audit = ["# Hash audit", "", "All listed inputs were hashed before and after plotting.", "",
             "| input | SHA-256 before | SHA-256 after | unchanged |", "|---|---|---|---|"]
    for path in hashes_before:
        audit.append(f"| `{path}` | `{hashes_before[path]}` | `{hashes_after[path]}` | {hashes_before[path] == hashes_after[path]} |")
    audit += ["", "## Direction-wise external claims", ""]
    for name, value in claims.items():
        audit.append(f"- `{name}`: {value}/6 directions.")
    audit += ["", "Supplementary Fig. S3 hashes were unchanged. Frozen prediction and result files were not modified.", ""]
    (OUT / "HASH_AUDIT.md").write_text("\n".join(audit))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    files = input_files()
    hashes_before = {path: sha256(path) for path in files}
    values = load_external_values()
    claims = verify_external_claims(values)
    wnt = read_wnt11f2_maps()

    fig = plt.figure(figsize=(AUTH.TWO_COLUMN, 2.75))
    draw_external_benchmark(fig, values, standalone=True)
    export(fig, "Fig2d_external_benchmark", png=False)

    fig = plt.figure(figsize=(AUTH.TWO_COLUMN, 1.85))
    draw_wnt11f2(fig, wnt, standalone=True)
    export(fig, "Fig2e_wnt11f2_comparison", png=False)

    revised_main_figure(values, wnt)
    hashes_after = {path: sha256(path) for path in files}
    if hashes_before != hashes_after:
        changed = [str(path) for path in files if hashes_before[path] != hashes_after[path]]
        raise RuntimeError(f"Frozen input changed during plotting: {changed}")
    write_audits(wnt, claims, hashes_before, hashes_after)


if __name__ == "__main__":
    main()
