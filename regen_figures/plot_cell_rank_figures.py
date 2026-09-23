#!/usr/bin/env python
"""Cell-embedding rank figures, datasets as rows.

Two views of the same numbers + the table behind them:
    --kind slope    cell_slope_by_split_{two_row,tahoe,zenodo}
    --kind scatter  cell_rank_scatter_by_split_{two_row,tahoe,zenodo}
    --kind both     (default) all six, collected once instead of twice
The CSV (cell_slope_by_split_two_row.csv) is always written.
"""
import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

import datasets as DS   

import config_ranking as CR
import make_convention_inversion_figures as F
import report_common as RC

OUT_DIR = DS.OUT_DIR
TITLES = DS.TITLES
ROWS = DS.DATASETS
OUT_CSV = f"{OUT_DIR}/cell_slope_by_split_two_row.csv"

FONT_SCALE = 1.6
MARKERSIZE = 8.5
LINEWIDTH = 2.6
POINT_SIZE = 130
SLOPE_JOINT_FIGSIZE = (12.0, 7.4)
SLOPE_JOINT_XLIM = (-0.4, 1.4)
SCATTER_JOINT_FIGSIZE = (14.0, 8.6)
ROW_FIGSIZE = (14.0, 5.2)

CONVENTIONS = ("within_drug", "within_cell_line")


def collect(ds):
    with DS.applied([CR], RESULTS_DIR=ds.results_dir, VIEW=ds.view,
                    SPLIT_ORDER=ds.splits):
        return (F.add_rank_pct(CR.collect_all("drug")),
                F.add_rank_pct(CR.collect_all("cell_line")))


def panel(df, split, where):
    table = F.beats_pct_table(df, "cell", [split])
    if table.empty:
        sys.exit(f"no {split} runs in {where}")
    s = table.xs(split)["beats_pct"].reindex(F.CELL_ORDER)
    err = F.beats_pct_sem(df, "cell", [split], kind="config").reindex(F.CELL_ORDER)
    if s.isna().any() or err.isna().any():
        sys.exit(f"incomplete {split} panel in {where}")
    return s, err


def _figure(rows, data, figsize, out_path, draw_panel, decorate, line_legend):
    with plt.rc_context(F._scaled_fonts(FONT_SCALE)):
        fig, axes = plt.subplots(len(rows), len(TITLES), figsize=figsize, squeeze=False,
                                 layout="constrained", sharex=True, sharey=True)
        for r, (ds, (drug_df, cellline_df)) in enumerate(zip(rows, data)):
            for c, split in enumerate(ds.splits):
                drug_s, drug_err = panel(drug_df, split, ds.results_dir)
                cellline_s, cellline_err = panel(cellline_df, split, ds.results_dir)
                draw_panel(axes[r, c], drug_s, cellline_s, drug_err, cellline_err,
                           TITLES[c] if r == 0 else None)
        used = list(axes.ravel())
        decorate(fig, used)
        F._panel_letters(used)
        F._add_shared_legend(fig, F.CELL_ORDER, F.CELL_COLOR, F.CELL_NICE,
                             marker_map=F.CELL_MARKER, line=line_legend,
                             markersize=6 * FONT_SCALE)
        F._save_figure(fig, out_path)


def _slope_figure(rows, data, figsize, out_path, xlim):
    def draw(ax, drug_s, cellline_s, drug_err, cellline_err, title):
        F.draw_slope_panel(ax, drug_s, cellline_s, F.CELL_ORDER,
                           F.CELL_COLOR, F.CELL_NICE, marker_map=F.CELL_MARKER,
                           title=title, show_labels=False, wrap_xticks=True,
                           drug_err=drug_err, cellline_err=cellline_err, xlim=xlim,
                           markersize=MARKERSIZE, linewidth=LINEWIDTH)

    def decorate(fig, used):
        F._hide_unlabelled_yticks(used)
        fig.supylabel(F.BEATS_PCT_YLABEL, fontsize=plt.rcParams["axes.labelsize"])

    _figure(rows, data, figsize, out_path, draw, decorate, line_legend=True)


def _scatter_figure(rows, data, figsize, out_path):
    def draw(ax, drug_s, cellline_s, drug_err, cellline_err, title):
        F.draw_scatter_panel(ax, drug_s, cellline_s, F.CELL_ORDER,
                             F.CELL_COLOR, F.CELL_NICE, marker_map=F.CELL_MARKER,
                             title=title, show_labels=False, point_size=POINT_SIZE,
                             drug_err=drug_err, cellline_err=cellline_err)

    def decorate(fig, used):
        fig.supxlabel(F.SCATTER_X_LABEL, fontsize=plt.rcParams["axes.labelsize"],
                      fontweight="bold")
        fig.supylabel(F.SCATTER_Y_LABEL, fontsize=plt.rcParams["axes.labelsize"],
                      fontweight="bold")

    _figure(rows, data, figsize, out_path, draw, decorate, line_legend=False)


def slope(data):
    _slope_figure(ROWS, data, SLOPE_JOINT_FIGSIZE,
                  f"{OUT_DIR}/cell_slope_by_split_two_row", SLOPE_JOINT_XLIM)
    for ds, row_data in zip(ROWS, data):
        _slope_figure([ds], [row_data], ROW_FIGSIZE,
                      f"{OUT_DIR}/cell_slope_by_split_{ds.name}", None)


def scatter(data):
    _scatter_figure(ROWS, data, SCATTER_JOINT_FIGSIZE,
                    f"{OUT_DIR}/cell_rank_scatter_by_split_two_row")
    for ds, row_data in zip(ROWS, data):
        _scatter_figure([ds], [row_data], ROW_FIGSIZE,
                        f"{OUT_DIR}/cell_rank_scatter_by_split_{ds.name}")


def split_stats(frames, split, where):
    stats = {}
    for conv, df in zip(CONVENTIONS, frames):
        s, err = panel(df, split, where)
        n = F.beats_pct_table(df, "cell", [split]).xs(split)["count"].reindex(F.CELL_ORDER)
        stats[conv] = (s, err, n)
    return stats


def table(rows, data):
    out = []
    for r, (ds, frames) in enumerate(zip(rows, data)):
        for c, split in enumerate(ds.splits):
            stats = split_stats(frames, split, ds.results_dir)
            for cell in F.CELL_ORDER:
                rec = {"dataset": ds.name, "view": ds.view,
                       "panel": chr(ord("A") + r * len(TITLES) + c),
                       "split": split, "split_label": TITLES[c],
                       "cell_emb": cell, "cell_emb_label": F.CELL_NICE.get(cell, cell)}
                for conv, (s, err, n) in stats.items():
                    rec[f"{conv}_beats_pct"] = s[cell]
                    rec[f"{conv}_sem"] = err[cell]
                    rec[f"{conv}_n"] = int(n[cell])
                rec["cell_line_minus_drug"] = (rec["within_cell_line_beats_pct"]
                                               - rec["within_drug_beats_pct"])
                out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["slope", "scatter", "both"], default="both")
    a = ap.parse_args()

    data = [collect(ds) for ds in ROWS]
    if a.kind in ("slope", "both"):
        slope(data)
    if a.kind in ("scatter", "both"):
        scatter(data)
    RC.save_csv(table(ROWS, data), OUT_CSV)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
