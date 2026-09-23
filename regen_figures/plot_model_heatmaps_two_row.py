#!/usr/bin/env python
"""(Re)generates Figure 2.
"""
import os
import sys

import matplotlib.pyplot as plt

import datasets as DS   
import figure_style as FS

import plot_model_heatmaps as M

M.ANNOT_DECIMALS = 2
M.SE_DECIMALS = 2
DEFAULT_ANNOT_FONTSIZE = M.ANNOT_FONTSIZE
ANNOT_FONTSIZE_OVERRIDE = {("mccdrug", "pairzs"): 22}

OUT_DIR = DS.OUT_DIR
ROWS = DS.DATASETS
SPLITS = DS.SPLIT_TAGS
METRICS = [
    {"tag": "mccdrug", "by": "drug", "metric": "MCC"},
    {"tag": "mcccellline", "by": "cell_line", "metric": "MCC"},
    {"tag": "aurocdrug", "by": "drug", "metric": "AUROC"},
    {"tag": "auroccellline", "by": "cell_line", "metric": "AUROC"},
]

PANEL_W = 8.4
ROW_H = 5 * 1.3
PAD_H = 2.8


def collect_row(ds, split, by, metric):
    with DS.applied([M], RESULTS_DIR=ds.results_dir, VIEW=ds.view, METRIC=metric):
        return {model: M.collect(split, model, by) for model in M.MODELS}


def row_values(tables):
    return [v for t in tables.values() if not t.empty for v in t["mean"]]


def figure(tables_by_row, vmin, vmax, cbar_label, out_base):
    cmap = M.blue_cmap()
    fig, axes = plt.subplots(len(ROWS), len(M.MODELS),
                             figsize=(PANEL_W * len(M.MODELS), PAD_H + ROW_H * len(ROWS)),
                             squeeze=False, constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.09, wspace=0.03, hspace=0.02)
    im = None
    for r, tables in enumerate(tables_by_row):
        for c, model in enumerate(M.MODELS):
            mat = M.matrix(tables[model], M.CELL_ORDER, M.DRUG_ORDER, "mean")
            se = M.matrix(tables[model], M.CELL_ORDER, M.DRUG_ORDER, "se")
            im = M.draw_panel(axes[r, c], mat, se, cmap, vmin, vmax,
                              M.MODEL_NICE[model] if r == 0 else "",
                              show_ylabels=(c == 0))
            if r < len(ROWS) - 1:
                axes[r, c].set_xticklabels([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=35)
    cbar.set_label(cbar_label, fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    FS.save(fig, out_base)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for metric in METRICS:
        for tag in SPLITS:
            tables_by_row, values = [], []
            for ds in ROWS:
                split = ds.split_for(tag)
                tables = collect_row(ds, split, metric["by"], metric["metric"])
                vals = row_values(tables)
                if not vals:
                    sys.exit(f"no {split} runs with {metric['metric']} "
                             f"({metric['by']}) in {ds.results_dir}")
                tables_by_row.append(tables)
                values.extend(vals)

            M.ANNOT_FONTSIZE = ANNOT_FONTSIZE_OVERRIDE.get(
                (metric["tag"], tag), DEFAULT_ANNOT_FONTSIZE)
            vmin, vmax, _lo, _hi = M.per_split_limits(values)
            out_base = f"{OUT_DIR}/model_heatmaps_v2_{metric['tag']}_{tag}"
            figure(tables_by_row, vmin, vmax, metric["metric"], out_base)
            print(f"wrote {out_base}.{{png,pdf,svg}}  "
                  f"(shared color scale {vmin:.3f}..{vmax:.3f}; "
                  + "; ".join(f"{ds.name} {ds.split_for(tag)} n={len(row_values(t))}"
                              for ds, t in zip(ROWS, tables_by_row)) + ")")


if __name__ == "__main__":
    main()
