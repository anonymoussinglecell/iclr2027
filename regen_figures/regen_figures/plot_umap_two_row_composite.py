#!/usr/bin/env python
"""
(Re)generates UMAP projections, datasets as rows, each coloured by cell line.
  Row 1  shared-label setting
  Row 2  single-cell label setting

Each row reuses its own dataset's cached UMAP fit.
"""
import argparse
import importlib.util

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figure_style as FS

ROOT = # set path
TAHOE = f"{ROOT}/scripts_tahoe"
ZENODO = f"{ROOT}/scripts_zenodo"

from report_common import CELL_NICE  

OUT_STEM = f"{ROOT}/plots_final/umap_two_row_composite"
ORDER = ["pca", "scvi", "gene_jepa", "gformer_cancer", "scgpt_pan"]

LEGEND_COL_W = 0.62
FIG_WIDTH = 20.0
TITLE_FONTSIZE = 26
LEGEND_FONTSIZE = 8


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def datasets():
    tahoe = load_module(f"{TAHOE}/umap.py", "tahoe_umap")
    zenodo = load_module(f"{ZENODO}/plot_umap.py", "zenodo_umap").U
    for mod in (tahoe, zenodo):
        mod.ORDER = ORDER
    return [tahoe, zenodo]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", action="store_true",
                    help="recompute every UMAP fit instead of reading the cache.")
    a = ap.parse_args()

    rows = []
    for mod in datasets():
        pop = mod.load_population()
        xys = {n: mod.embedding_xy(n, pop, refit=a.refit) for n in ORDER}
        lines, colors = mod.cell_line_palette(pop)
        rows.append({"mod": mod, "pop": pop, "xys": xys,
                     "lines": lines, "colors": colors})

    n_cols = len(ORDER)
    width_ratios = [1.0] * n_cols + [LEGEND_COL_W]
    panel_w = FIG_WIDTH / sum(width_ratios)
    fig = plt.figure(figsize=(FIG_WIDTH, 2 * panel_w + 0.55))
    gs = fig.add_gridspec(2, n_cols + 1, width_ratios=width_ratios,
                          wspace=0.035, hspace=0.015)

    for r, row in enumerate(rows):
        for col, name in enumerate(ORDER):
            ax = fig.add_subplot(gs[r, col])
            row["mod"].draw_cell_line_panel(ax, row["xys"][name], row["pop"],
                                            row["colors"])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(CELL_NICE.get(name, name), fontsize=TITLE_FONTSIZE,
                             fontweight="bold", pad=8)

        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          color=row["colors"][c], label=c) for c in row["lines"]]
        leg_ax = fig.add_subplot(gs[r, n_cols])
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="center left", frameon=False,
                      fontsize=LEGEND_FONTSIZE, ncol=2 if len(handles) > 22 else 1,
                      markerscale=1.4, labelspacing=0.35, columnspacing=1.0,
                      handletextpad=0.3)

    FS.save(fig, OUT_STEM, formats=("png", "svg"), dpi=400, bbox_inches="tight")
    print(f"wrote {OUT_STEM}.{{png,svg}}")


if __name__ == "__main__":
    main()
