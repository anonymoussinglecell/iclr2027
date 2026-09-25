#!/usr/bin/env python
"""
Generates the paper figures from finished runs of main.py.

  heatmaps  cell x drug encoder heatmaps, one panel per model
  bars      average MCC per split, whiskers to min/max across configurations
  floor     full model vs single-modality ablation vs prevalence baseline
  slope     cell-embedding rank under the two MCC conventions
  umap      UMAP projections of the frozen cell embeddings

Usage: python figures.py [FIGURE ...] [--refit]   (default: all)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from common import (
    CELL_NICE, CELL_ORDER, DATASETS, DRUG_NICE, DRUG_ORDER, METRIC_LABELS, MODEL_NICE,
    MODEL_ORDER, OUT_DIR, PAIRZS_TAG, SPLIT_NICE, SPLIT_TAGS, TITLES, collect, save,
)


# =====================================================================
# Cell x drug encoder heatmaps
# =====================================================================
HEATMAP_METRICS = [("mccdrug", "drug", "MCC"), ("mcccellline", "cell_line", "MCC"),
                   ("aurocdrug", "drug", "AUROC"), ("auroccellline", "cell_line", "AUROC")]
LABEL_FONTSIZE = 26
ANNOT_FONTSIZE = 14
ANNOT_FONTSIZE_OVERRIDE = {("mccdrug", PAIRZS_TAG): 22}


def blue_cmap():
    base = plt.cm.Blues(np.linspace(0.15, 0.92, 256))
    cmap = LinearSegmentedColormap.from_list("Blues_trunc", base)
    cmap.set_bad("#c7c7c7")
    return cmap


def grid(runs, field):
    if runs.empty:
        return pd.DataFrame(np.nan, index=CELL_ORDER, columns=DRUG_ORDER)
    piv = runs.pivot_table(index="cell", columns="drug", values=field, aggfunc="first")
    return piv.reindex(index=CELL_ORDER, columns=DRUG_ORDER)


def heatmap_panel(ax, runs, cmap, vmin, vmax, title, annot_fontsize, show_ylabels):
    M, SE = grid(runs, "mean"), grid(runs, "se")
    vals = M.values.astype(float)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([DRUG_NICE[c] for c in M.columns], rotation=45, ha="right",
                       fontsize=LABEL_FONTSIZE)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([CELL_NICE[c] for c in M.index] if show_ylabels else [],
                       fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=20, fontweight="bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", color="#888", fontsize=16)
                continue
            r, g, b, _ = cmap(np.clip((v - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0))
            se = SE.values[i, j]
            txt = f"{v:.2f}" + (f"\n±{se:.2f}" if np.isfinite(se) else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=annot_fontsize,
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else "black")
    return im


def fig_heatmaps(args):
    """Datasets as rows; one color scale per figure, spanning both rows' min..max."""
    cmap = blue_cmap()
    for mtag, by, metric in HEATMAP_METRICS:
        runs = [collect(ds, by, metric) for ds in DATASETS]
        for tag in SPLIT_TAGS:
            rows = []
            for ds, df in zip(DATASETS, runs):
                sub = df[(df["split"] == ds.split_for(tag)) & df["model"].isin(MODEL_ORDER)]
                if sub.empty:
                    sys.exit(f"no {ds.split_for(tag)} runs with {metric} ({by}) in {ds.results_dir}")
                rows.append(sub)
            vals = pd.concat(rows)["mean"]
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax <= vmin:
                vmax = vmin + 1e-6

            fig, axes = plt.subplots(len(DATASETS), len(MODEL_ORDER),
                                     figsize=(8.4 * len(MODEL_ORDER), 2.8 + 6.5 * len(DATASETS)),
                                     squeeze=False, constrained_layout=True)
            fig.get_layout_engine().set(w_pad=0.05, h_pad=0.09, wspace=0.03, hspace=0.02)
            fontsize = ANNOT_FONTSIZE_OVERRIDE.get((mtag, tag), ANNOT_FONTSIZE)
            for r, sub in enumerate(rows):
                for c, model in enumerate(MODEL_ORDER):
                    im = heatmap_panel(axes[r, c], sub[sub["model"] == model], cmap, vmin, vmax,
                                       MODEL_NICE[model] if r == 0 else "", fontsize,
                                       show_ylabels=(c == 0))
                    if r < len(DATASETS) - 1:
                        axes[r, c].set_xticklabels([])
            cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=35)
            cbar.set_label(metric, fontsize=18)
            cbar.ax.tick_params(labelsize=14)
            save(fig, f"{OUT_DIR}/model_heatmaps_{mtag}_{tag}")


# =====================================================================
# Average MCC per split
# =====================================================================
BAR_W = 0.20


def fig_bars(args):
    """Four bars per split: two MCC conventions for each dataset. Hue marks the dataset,
    dark shade = within-drug, light = within cell line. Whiskers run from each bar's mean
    to the observed min and max across configurations."""
    metrics = [METRIC_LABELS["MCC_drug"], METRIC_LABELS["MCC_cellline"]]
    lookups = []
    for ds in DATASETS:
        if not os.path.exists(ds.avg_csv):
            sys.exit(f"missing {ds.avg_csv}")
        df = pd.read_csv(ds.avg_csv).dropna(subset=["mean"])
        if not (df["split"] == ds.pairzs).any():
            sys.exit(f"{ds.avg_csv} has no {ds.pairzs} rows")
        df["split"] = df["split"].replace({ds.pairzs: PAIRZS_TAG})
        lookups.append({(r["metric"], r["split"]): r for _, r in df.iterrows()})

    splits = [s for s in SPLIT_TAGS if any((m, s) in lk for lk in lookups for m in metrics)]
    x = np.arange(len(splits), dtype=float)
    n_bars = len(DATASETS) * len(metrics)

    fig, ax = plt.subplots(figsize=(13.6, 5.8))
    lo, hi = 0.0, 0.0
    for j, (ds, lookup) in enumerate(zip(DATASETS, lookups)):
        for k, metric in enumerate(metrics):
            offs = (j * len(metrics) + k - (n_bars - 1) / 2) * BAR_W
            for xi, split in zip(x, splits):
                row = lookup.get((metric, split))
                if row is None:
                    continue
                mean, vmin, vmax = float(row["mean"]), float(row["min"]), float(row["max"])
                lo, hi = min(lo, vmin), max(hi, vmax)
                ax.bar(xi + offs, mean, BAR_W, color=ds.bar_colors[k],
                       edgecolor="white", linewidth=0.8, zorder=2)
                ax.errorbar(xi + offs, mean, yerr=[[mean - vmin], [vmax - mean]],
                            fmt="none", ecolor="#1a1a1a", elinewidth=1.1,
                            capsize=3.0, capthick=1.1, zorder=4)

    ax.axhline(0.0, color="#666666", linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_NICE[s].replace(" (", "\n(") for s in splits], fontsize=10.5)
    ax.set_ylabel("MCC", fontsize=12)
    ax.set_title("Average MCC per split", fontsize=14, fontweight="bold", pad=12)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="both", labelsize=10, color="#888888")

    handles = [Patch(facecolor=ds.bar_colors[k], label=f"{ds.nice} — {m}")
               for ds in DATASETS for k, m in enumerate(metrics)]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)
    save(fig, f"{OUT_DIR}/avg_min_max_mcc_by_split_bars", bbox_inches="tight")


# =====================================================================
# Full model vs matched ablation, prevalence baseline as reference line
#
# A cell_only model's output cannot vary with drug identity, so scoring it within a drug
# (across cell lines) is the only non-degenerate direction; similarly, drug_only pairs with
# within-cell-line. The other pairing gives MCC = 0 by construction and is not plotted.
# =====================================================================
GROUPINGS = ["within_drug", "within_cell_line"]
MATCHED_ABLATION = {"within_drug": "cell_only", "within_cell_line": "drug_only"}
MATCHED_NULL = {"within_drug": "cellline_only", "within_cell_line": "drug_only"}
BASELINE_GROUPING = {"within_drug": "drug", "within_cell_line": "cell_line"}
GROUPING_YLABEL = {"within_drug": METRIC_LABELS["MCC_drug"],
                   "within_cell_line": METRIC_LABELS["MCC_cellline"]}
FLOOR_MODELS = ["lightgbm", "mlp", "xrfm"]
C_FULL = "#4C4C4C"
C_ABL = {"cell_only": "#C87A20", "drug_only": "#2A9D8F"}
C_NULL = "#111111"
DATASET_LABEL = {"tahoe": "Shared-label dataset", "zenodo": "Single-cell label dataset"}
DATASET_GAP = 0.7
FLOOR_FIGSIZE = (9.8, 4.0)


def paired(scores, grouping):
    """Every ablation run paired on (split, swept embedding, model) with the full runs,
    averaged over the embedding the ablation does not see."""
    s = scores[(scores["grouping"] == grouping) & (scores["metric"] == "MCC")]
    abl = s[s["variant"] == MATCHED_ABLATION[grouping]]
    if abl.empty:
        return pd.DataFrame(columns=["split", "model", "full", "ablation"])
    inert = str(abl["inert_embedding"].iloc[0])
    on = ["split", "drug_emb" if inert == "cell_emb" else "cell_emb", "model"]
    full = (s[s["variant"] == "full"].groupby(on, as_index=False)["value"].mean()
            .rename(columns={"value": "full"}))
    return abl[on + ["value"]].rename(columns={"value": "ablation"}).merge(full, on=on)


def null_value(nulls, grouping, split):
    """The informative null for this grouping, or None where it degenerated
    (fell back to the global prior, or constant by construction)."""
    r = nulls[(nulls["split"] == split) & (nulls["baseline"] == MATCHED_NULL[grouping]) &
              (nulls["grouping"] == BASELINE_GROUPING[grouping]) & (nulls["metric"] == "MCC")]
    if r.empty:
        return None
    r = r.iloc[0]
    if bool(r.get("fell_back", False)) or str(r.get("classification", "")) != "informative":
        return None
    return float(r["mean"])


def floor_panel(ax, p, nulls, grouping, splits, off):
    splits = [s for s in splits if s in set(p["split"])]
    w = 0.34
    rng = np.random.default_rng(0)
    for i, sp in enumerate(splits):
        sub = p[p["split"] == sp]
        for j, (col, colour) in enumerate((("full", C_FULL),
                                           ("ablation", C_ABL[MATCHED_ABLATION[grouping]]))):
            xc = off + i + (j - 0.5) * w
            v = sub[col].to_numpy(dtype=float)
            ax.bar(xc, v.mean(), w, color=colour, edgecolor="white",
                   linewidth=0.5, alpha=0.85, zorder=2)
            ax.scatter(xc + rng.uniform(-w * .28, w * .28, v.size), v,
                       s=11, c="white", edgecolor="#333333", linewidth=0.4,
                       zorder=4, alpha=0.9)
            if v.size:
                ax.scatter([xc], [np.nanmax(v)], marker="*", s=70, c="black", zorder=6,
                           linewidth=0)
        nv = null_value(nulls, grouping, sp)
        if nv is not None:
            ax.plot([off + i - 1.1 * w, off + i + 1.1 * w], [nv, nv], "--",
                    lw=1.8, color=C_NULL, zorder=5)
    ax.axhline(0.0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_ylabel(GROUPING_YLABEL[grouping], fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_datasets(fig, ax, spans):
    r = fig.canvas.get_renderer()
    drop = (ax.get_window_extent(r).y0 - ax.xaxis.get_tightbbox(r).y0) * 72 / fig.dpi + 4
    line_tr = offset_copy(ax.get_xaxis_transform(), fig=fig, y=-drop, units="points")
    text_tr = offset_copy(ax.get_xaxis_transform(), fig=fig, y=-drop - 3, units="points")
    for ds, (lo, hi) in zip(DATASETS, spans):
        ax.plot([lo - 0.4, hi + 0.4], [0, 0], color="black", lw=0.8,
                transform=line_tr, clip_on=False)
        ax.text((lo + hi) / 2, 0, DATASET_LABEL[ds.name], transform=text_tr,
                ha="center", va="top", fontsize=9.5)


def floor_handles():
    return [Patch(facecolor=C_FULL, label="Full multimodal (cell + drug)"),
            Patch(facecolor=C_ABL["cell_only"], label="Cell-only (drug info removed)"),
            Patch(facecolor=C_ABL["drug_only"], label="Drug-only (cell info removed)"),
            Line2D([], [], ls="--", lw=1.8, color=C_NULL,
                   label="Prevalence baseline: marginal rate lookup"),
            Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor="white",
                   markeredgecolor="#333333", label="one encoder configuration"),
            Line2D([], [], marker="*", ls="", markersize=9, color="black",
                   label="best configuration")]


def fig_floor(args):
    """Datasets side by side within each panel; one y range shared by all panels."""
    data = []
    for ds in DATASETS:
        scores = pd.read_csv(f"{ds.ablation_dir}/ablation_scores_long.csv")
        nulls = pd.read_csv(f"{ds.ablation_dir}/baselines_pairlevel_long.csv")
        data.append((ds, {g: paired(scores, g) for g in GROUPINGS}, nulls))

    for model in FLOOR_MODELS:
        def draw(ax, ds, pairs, nulls, grouping, off):
            p = pairs[grouping]
            floor_panel(ax, p[p["model"] == model], nulls, grouping, ds.splits, off)

        lims = []
        for ds, pairs, nulls in data:
            for grouping in GROUPINGS:
                tmp_fig, tmp_ax = plt.subplots()
                draw(tmp_ax, ds, pairs, nulls, grouping, 0.0)
                lims.append(tmp_ax.get_ylim())
                plt.close(tmp_fig)
        ylim = (min(lo for lo, _ in lims), max(hi for _, hi in lims))

        fig, axes = plt.subplots(1, 2, figsize=FLOOR_FIGSIZE)
        for ax, grouping in zip(axes, GROUPINGS):
            ticks, labels, spans, off = [], [], [], 0.0
            for ds, pairs, nulls in data:
                draw(ax, ds, pairs, nulls, grouping, off)
                spans.append((off, off + len(ds.splits) - 1))
                ticks += [off + i for i in range(len(ds.splits))]
                labels += [SPLIT_NICE[s] for s in ds.splits]
                off += len(ds.splits) + DATASET_GAP
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
            ax.set_xlim(ticks[0] - 0.7, ticks[-1] + 0.7)
            ax.set_ylim(ylim)
            label_datasets(fig, ax, spans)
        fig.legend(handles=floor_handles(), loc="upper center", ncol=3, frameon=False,
                   fontsize=9.5, bbox_to_anchor=(0.5, 1.0 + 0.42 / FLOOR_FIGSIZE[1]))
        fig.tight_layout()
        save(fig, f"{OUT_DIR}/floor_bars_mcc_{model}_side_by_side", bbox_inches="tight")


# =====================================================================
# Cell-embedding rank under the two MCC conventions
#
# Value shown is rank-based: the fraction of other finished configurations (all models,
# within a split) that an embedding's configurations outperform, not a difference in raw
# MCC. Error bars: SEM across the 18 configurations (6 drug embeddings x 3 models).
# =====================================================================
SLOPE_RC = {
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.solid_capstyle": "round",
}
FONT_KEYS = ("font.size", "axes.titlesize", "axes.labelsize",
             "xtick.labelsize", "ytick.labelsize", "legend.fontsize")
FONT_SCALE = 1.6
MARKERSIZE = 8.5
LINEWIDTH = 2.6
CELL_COLOR = {"pca": "#0072B2", "scvi": "#E69F00", "gene_jepa": "#009E73",
              "gformer_cancer": "#CC79A7", "scgpt_pan": "#D55E00"}   # Okabe-Ito
TIE_TOL = 5e-3


def ranked(df):
    df = df.copy()
    df["rank_pct"] = df.groupby("split")["mean"].rank(ascending=False, pct=True)
    return df


def beats_pct(df, split):
    """(1 - mean percentile rank, SEM) per cell embedding within one split."""
    rk = df[(df["split"] == split) & df["model"].isin(MODEL_ORDER)]
    g = rk.groupby("cell")["rank_pct"].agg(["mean", "std", "count"]).reindex(CELL_ORDER)
    if g.isna().any().any():
        sys.exit(f"incomplete {split} panel")
    return 1 - g["mean"], g["std"] / np.sqrt(g["count"])


def tie_offsets(x_s, y_s, step_pt):
    """Embeddings at the same spot are spread symmetrically about their true position."""
    groups = []
    for u in CELL_ORDER:
        for g in groups:
            if abs(x_s[u] - x_s[g[0]]) <= TIE_TOL and abs(y_s[u] - y_s[g[0]]) <= TIE_TOL:
                g.append(u)
                break
        else:
            groups.append([u])
    offsets = {u: 0.0 for u in CELL_ORDER}
    for g in groups:
        if len(g) > 1:
            for i, u in enumerate(g):
                offsets[u] = (i - (len(g) - 1) / 2.0) * step_pt
    return offsets


def slope_panel(ax, drug_s, cl_s, drug_err, cl_err, title):
    dodge = tie_offsets(drug_s, cl_s, MARKERSIZE + 2.0)
    for u in CELL_ORDER:
        off = dodge[u]
        tr = offset_copy(ax.transData, fig=ax.get_figure(), x=off, y=off * 0.5,
                         units="points") if off else ax.transData
        le, re_ = float(drug_err[u]), float(cl_err[u])
        if le or re_:
            ax.errorbar([0, 1], [drug_s[u], cl_s[u]], yerr=[le, re_], fmt="none",
                        ecolor=CELL_COLOR[u], elinewidth=1.1, capsize=2.8, capthick=1.1,
                        alpha=0.85, zorder=2, transform=tr)
        ax.plot([0, 1], [drug_s[u], cl_s[u]], color=CELL_COLOR[u], linewidth=LINEWIDTH,
                marker="o", markersize=MARKERSIZE, markeredgecolor="white",
                markeredgewidth=0.7, zorder=3, transform=tr)
    ax.axvline(0, color="#999999", linewidth=0.9, zorder=1)
    ax.axvline(1, color="#999999", linewidth=0.9, zorder=1)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, color="#b0b0b0", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Drug", "Cell line"], fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True, linewidth=0.4, color="#e3e3e3", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontweight="bold", pad=6)


def fig_slope(args):
    data = [(ds, ranked(collect(ds, "drug")), ranked(collect(ds, "cell_line")))
            for ds in DATASETS]
    rc = {**SLOPE_RC, **{k: SLOPE_RC[k] * FONT_SCALE for k in FONT_KEYS}}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(DATASETS), len(SPLIT_TAGS), figsize=(12.0, 5.6),
                                 squeeze=False, layout="constrained", sharex=True, sharey=True)
        for r, (ds, drug_df, cl_df) in enumerate(data):
            for c, split in enumerate(ds.splits):
                drug_s, drug_err = beats_pct(drug_df, split)
                cl_s, cl_err = beats_pct(cl_df, split)
                slope_panel(axes[r, c], drug_s, cl_s, drug_err, cl_err,
                            TITLES[c] if r == 0 else None)
        for ax in axes.ravel():
            if not any(t.get_visible() for t in ax.get_yticklabels()):
                ax.tick_params(axis="y", length=0)
        for i, ax in enumerate(axes.ravel()):
            ax.text(0.03, 0.98, chr(ord("A") + i), transform=ax.transAxes, ha="left", va="top",
                    fontweight="bold", fontsize=plt.rcParams["axes.titlesize"] + 1)
        fig.supylabel("Average fraction of other configurations outperformed",
                      fontsize=plt.rcParams["axes.labelsize"])
        fig.supxlabel("MCC computed within", fontsize=plt.rcParams["axes.labelsize"])
        handles = [Line2D([0], [0], color=CELL_COLOR[u], marker="o",
                          markersize=6 * FONT_SCALE, markeredgecolor="white",
                          markeredgewidth=0.7, linewidth=1.8, label=CELL_NICE[u])
                   for u in CELL_ORDER]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(CELL_ORDER),
                   frameon=False, fontsize=plt.rcParams["axes.titlesize"])
        save(fig, f"{OUT_DIR}/cell_slope_by_split_two_row")


# =====================================================================
# UMAP projections, datasets as rows, coloured by cell line
# =====================================================================
UMAP_ORDER = ["pca", "scvi", "gene_jepa", "gformer_cancer", "scgpt_pan"]
BARCODE_COL = "BARCODE_SUB_LIB_ID"


def load_population(ds):
    """One row per barcode of the population, with a cell-line name."""
    import anndata as ad
    pairs = pd.read_parquet(ds.pairs_parquet, columns=[BARCODE_COL, "cell_line"])
    pairs = pairs.drop_duplicates(BARCODE_COL)
    cvcl_to_name = (ad.read_h5ad(ds.h5ad, backed="r").obs
                      .drop_duplicates("cell_line")
                      .set_index("cell_line")["cell_name"].astype(str))
    names = pairs["cell_line"].map(cvcl_to_name)
    return pd.Series(names.values, index=pairs[BARCODE_COL].astype(str).values,
                     name="cell_name").dropna()


def embedding_xy(ds, name, pop, refit=False):
    """2D UMAP coordinates for one embedding, read from cache unless missing or --refit."""
    cache = f"{ds.umap_cache_dir}/{name}_umap_xy.parquet"
    if os.path.exists(cache) and not refit:
        xy = pd.read_parquet(cache)
        xy.index = xy.index.astype(str)
        return xy
    import umap
    df = pd.read_parquet(f"{ds.cell_emb_dir}/{name}_cell_embeddings.parquet")
    df.index = df.index.astype(str)
    keep = df.index.intersection(pop.index)
    X = df.loc[keep].to_numpy(dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                       random_state=0).fit_transform(X)
    xy = pd.DataFrame(coords, columns=["x", "y"], index=pd.Index(keep, name="barcode"))
    os.makedirs(ds.umap_cache_dir, exist_ok=True)
    xy.to_parquet(cache)
    return xy


def fig_umap(args):
    n_cols = len(UMAP_ORDER)
    width_ratios = [1.0] * n_cols + [0.62]
    fig_w = 20.0
    fig = plt.figure(figsize=(fig_w, 2 * (fig_w / sum(width_ratios)) + 0.55))
    gs = fig.add_gridspec(2, n_cols + 1, width_ratios=width_ratios, wspace=0.035, hspace=0.015)
    # tab20 has only 20 colours, so stack tab20/20b/20c for 60
    palette = [plt.get_cmap(m)(i) for m in ("tab20", "tab20b", "tab20c") for i in range(20)]

    for r, ds in enumerate(DATASETS):
        pop = load_population(ds)
        lines = sorted(pop.unique())
        colors = {c: palette[i % len(palette)] for i, c in enumerate(lines)}
        for col, name in enumerate(UMAP_ORDER):
            xy = embedding_xy(ds, name, pop, refit=args.refit)
            labels = pop.reindex(xy.index)
            ax = fig.add_subplot(gs[r, col])
            for line, color in colors.items():
                m = (labels == line).to_numpy()
                if m.any():
                    ax.scatter(xy["x"].to_numpy()[m], xy["y"].to_numpy()[m], s=0.5,
                               linewidths=0, color=color, alpha=0.6, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(CELL_NICE[name], fontsize=26, fontweight="bold", pad=8)

        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          color=colors[c], label=c) for c in lines]
        leg_ax = fig.add_subplot(gs[r, n_cols])
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="center left", frameon=False, fontsize=8,
                      ncol=2 if len(handles) > 22 else 1, markerscale=1.4,
                      labelspacing=0.35, columnspacing=1.0, handletextpad=0.3)
    save(fig, f"{OUT_DIR}/umap_two_row_composite", dpi=400, bbox_inches="tight")


# =====================================================================
FIGURES = {"heatmaps": fig_heatmaps, "bars": fig_bars, "floor": fig_floor,
           "slope": fig_slope, "umap": fig_umap}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", metavar="FIGURE",
                    help=f"any of: {', '.join(FIGURES)} (default: all)")
    ap.add_argument("--refit", action="store_true",
                    help="recompute the UMAP fits instead of reading the cache.")
    args = ap.parse_args()
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(sorted(unknown))}")
    for name in args.figures or FIGURES:
        FIGURES[name](args)


if __name__ == "__main__":
    main()
