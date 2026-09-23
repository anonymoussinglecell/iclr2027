#!/usr/bin/env python
"""
Generates Figure 5

Four bars per split group, two MCC conventions for each dataset. 
Hue marks the datasets; the two shades within a hue marks the MCC convention.

Whiskers run from each bar's mean to the observed min and max across
configurations. See figure caption for details.

One pair-zero-shot column: shared-label Tahoe dataset uses pairzs_withoutleak,
second dataset allows single cell leakage since it's minimal due to dataset
properties (see main text).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import datasets as DS   
import figure_style as FS
from report_common import METRIC_LABELS  

OUT_DIR = DS.OUT_DIR
OUT_STEM = f"{OUT_DIR}/avg_min_max_mcc_by_split_bars"

SPLIT_ORDER = DS.SPLIT_TAGS
SPLIT_NICE = dict(DS.SPLIT_NICE, pairzs=DS.PAIRZS_LABEL)

# one bar per (dataset, metric), in this order within each split group
METRIC_ROWS = [METRIC_LABELS["MCC_drug"], METRIC_LABELS["MCC_cellline"]]
# dark shade = within-drug, light = within cell line; hue marks the dataset
DATASETS = [{"nice": d.nice, "csv": d.avg_csv, "pairzs": d.pairzs,
             "colors": dict(zip(METRIC_ROWS, d.bar_colors))} for d in DS.DATASETS]

BAR_W = 0.20


def load(csv_path, pairzs):
    if not os.path.exists(csv_path):
        sys.exit(f"missing {csv_path} -- run that dataset's make_avg_min_max_mcc_table.py first")
    df = pd.read_csv(csv_path).dropna(subset=["mean"])
    if not (df["split"] == pairzs).any():
        sys.exit(f"{csv_path} has no {pairzs} rows")
    df["split"] = df["split"].replace({pairzs: DS.PAIRZS_TAG})
    return {(r["metric"], r["split"]): r for _, r in df.iterrows()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in DATASETS:
        ds["lookup"] = load(ds["csv"], ds["pairzs"])

    splits = [s for s in SPLIT_ORDER
              if any((m, s) in ds["lookup"] for ds in DATASETS for m in METRIC_ROWS)]
    x = np.arange(len(splits), dtype=float)
    n_bars = len(DATASETS) * len(METRIC_ROWS)

    fig, ax = plt.subplots(figsize=(13.6, 5.8))

    lo, hi = 0.0, 0.0
    for j, ds in enumerate(DATASETS):
        for k, metric in enumerate(METRIC_ROWS):
            slot = j * len(METRIC_ROWS) + k
            offs = (slot - (n_bars - 1) / 2) * BAR_W
            for xi, split in zip(x, splits):
                row = ds["lookup"].get((metric, split))
                if row is None:
                    continue
                mean, vmin, vmax = float(row["mean"]), float(row["min"]), float(row["max"])
                lo, hi = min(lo, vmin), max(hi, vmax)
                ax.bar(xi + offs, mean, BAR_W, color=ds["colors"][metric],
                       edgecolor="white", linewidth=0.8, zorder=2)
                # asymmetric whisker: mean -> observed min and observed max
                ax.errorbar(xi + offs, mean,
                            yerr=[[mean - vmin], [vmax - mean]],
                            fmt="none", ecolor="#1a1a1a", elinewidth=1.1,
                            capsize=3.0, capthick=1.1, zorder=4)

    ax.axhline(0.0, color="#666666", linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_NICE[s].replace(" (", "\n(") for s in splits], fontsize=10.5)
    ax.set_ylabel("MCC", fontsize=12)
    ax.set_title("Average MCC per split",
                 fontsize=14, fontweight="bold", pad=12)
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

    handles = [Patch(facecolor=ds["colors"][m], label=f"{ds['nice']} — {m}")
               for ds in DATASETS for m in METRIC_ROWS]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)

    FS.save(fig, OUT_STEM, bbox_inches="tight")
    print(f"wrote {OUT_STEM}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
