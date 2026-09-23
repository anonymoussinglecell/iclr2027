#!/usr/bin/env python
"""(Re)generates Figure 6.
"""
import matplotlib.pyplot as plt
import pandas as pd

import datasets as DS   

import report_common as RC
import plot_ablations as P

OUT_DIR = DS.OUT_DIR
ROWS = DS.DATASETS
MODELS = ["lightgbm", "mlp", "xrfm"]

SPLIT_NICE = DS.SPLIT_NICE
BASELINE_LABEL = "Prevalence baseline: marginal rate lookup"
ABL_COLOR = {"cell_only": "#C87A20", "drug_only": "#2A9D8F"}
P.C_ABL = ABL_COLOR

ROW_FIGSIZE = (9.8, 4.6)
JOINT_FIGSIZE = (9.8, 7.8)
LEGEND_PAD_IN = 0.6


def handles():
    hs = P._floor_handles()
    for h in hs:
        if str(h.get_label()).startswith("Null baseline"):
            h.set_label(BASELINE_LABEL)
    return hs


def draw_row(d, ds, axes, model, show_title, show_xlabels):
    rows = []
    with DS.applied([P], SPLIT_ORDER=ds.splits, SPLIT_NICE=SPLIT_NICE):
        for c, grouping in enumerate(P.GROUPINGS):
            ax = axes[c]
            P._floor_panel(d, ax, model, grouping, "MCC", rows)
            for t in [t for t in ax.texts if "null" in t.get_text()]:
                t.remove()
            if not show_title:
                ax.set_title("")
            if not show_xlabels:
                ax.set_xticklabels([])
    return pd.DataFrame(rows).assign(dataset=ds.name)


def shared_ylim(data, model):
    fig, axes = plt.subplots(len(ROWS), 2, figsize=JOINT_FIGSIZE, squeeze=False)
    for r, (ds, d) in enumerate(zip(ROWS, data)):
        draw_row(d, ds, axes[r], model, show_title=False, show_xlabels=False)
    lims = [ax.get_ylim() for ax in axes.ravel()]
    plt.close(fig)
    return min(lo for lo, _ in lims), max(hi for _, hi in lims)


def plot(rows, data, figsize, stem, model, ylim):
    fig, axes = plt.subplots(len(rows), 2, figsize=figsize, squeeze=False)
    tables = []
    for r, (ds, d) in enumerate(zip(rows, data)):
        tables.append(draw_row(d, ds, axes[r], model, show_title=(r == 0),
                               show_xlabels=(r == len(rows) - 1)))
    for ax in axes.ravel():
        ax.set_ylim(ylim)
    fig.legend(handles=handles(), loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0 + LEGEND_PAD_IN / figsize[1]))
    P._finish(fig, OUT_DIR, stem)
    return pd.concat(tables, ignore_index=True)


def main():
    data = [P.Data(ds.ablation_dir) for ds in ROWS]
    for model in MODELS:
        ylim = shared_ylim(data, model)
        for ds, d in zip(ROWS, data):
            stem = f"floor_bars_mcc_{model}_{ds.name}"
            table = plot([ds], [d], ROW_FIGSIZE, stem, model, ylim)
            RC.save_csv(table, f"{OUT_DIR}/{stem}.csv")
        plot(ROWS, data, JOINT_FIGSIZE, f"floor_bars_mcc_{model}_two_row", model, ylim)


if __name__ == "__main__":
    main()
