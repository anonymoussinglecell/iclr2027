#!/usr/bin/env python
"""
Paths, display names, dataset profiles and scoring helpers shared by the figure scripts.
"""
import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,       # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "svg.fonttype": "none",   # keep SVG text as text, editable
})

ROOT = ""  # set path
OUT_DIR = f"{ROOT}/plots_final"
FORMATS = ("png", "pdf", "svg")

CELL_ORDER = ["gene_jepa", "gformer_cancer", "scgpt_pan", "pca", "scvi"]
DRUG_ORDER = ["morgan", "biomed", "chemberta", "maccs", "materials", "molformer"]
MODEL_ORDER = ["mlp", "lightgbm", "xrfm"]

CELL_NICE = {"pca": "PCA", "scvi": "scVI", "gene_jepa": "GeneJepa",
             "gformer_cancer": "Geneformer", "scgpt_pan": "scGPT"}
# the "materials" run tag is the SMI-TED drug encoder
DRUG_NICE = {"morgan": "Morgan", "biomed": "Biomed", "chemberta": "ChemBERTa",
             "maccs": "MACCS", "materials": "SMI-TED", "molformer": "MoLFormer"}
MODEL_NICE = {"mlp": "MLP", "lightgbm": "LightGBM", "xrfm": "xRFM"}

# one shared column label; each dataset maps it to its own pair-zero-shot split
PAIRZS_TAG = "pairzs"
SPLIT_TAGS = ["random", PAIRZS_TAG, "loclo_loo", "lodo_loo"]
SPLIT_NICE = {
    "random":             "Random split",
    PAIRZS_TAG:           "Pair zero shot",
    "pairzs_withleak":    "Pair zero shot",
    "pairzs_withoutleak": "Pair zero shot",
    "loclo_loo":          "Cell line zero shot",
    "lodo_loo":           "Drug zero shot",
}
TITLES = [SPLIT_NICE[t] for t in SPLIT_TAGS]

METRIC_LABELS = {"MCC_drug": "MCC (within-drug)", "MCC_cellline": "MCC (within-cell line)"}


@dataclass(frozen=True)
class Dataset:
    """The split and view differ per dataset by design (see paper): the shared-label
    dataset is scored pairlevel on pairzs_withoutleak, the single-cell label dataset
    aggregate on pairzs_withleak."""
    name: str
    nice: str
    results_dir: str
    view: str          # "pairlevel" | "aggregate"
    pairzs: str
    report_dir: str
    bar_colors: tuple  # (within-drug, within-cell-line)
    pairs_parquet: str
    h5ad: str
    cell_emb_dir: str
    umap_cache_dir: str
    splits: list = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "splits", [self.split_for(t) for t in SPLIT_TAGS])

    @property
    def ablation_dir(self):
        return f"{self.report_dir}/ablations"

    @property
    def avg_csv(self):
        return f"{self.report_dir}/avg_min_max_mcc_by_split.csv"

    def split_for(self, tag):
        return self.pairzs if tag == PAIRZS_TAG else tag


TAHOE = Dataset(
    name="tahoe", nice="Tahoe",
    results_dir=f"{ROOT}/results/tahoe/bal15_drugs",
    view="pairlevel", pairzs="pairzs_withoutleak",
    report_dir=f"{ROOT}/tahoe_plots",
    bar_colors=("#274c77", "#6096ba"),
    pairs_parquet=f"{ROOT}/data/tahoe/embeddings/train_pairs_bal15.parquet",
    h5ad=f"{ROOT}/data/tahoe/controls2000_merged_hvg5000_log1p_umap.h5ad",
    cell_emb_dir=f"{ROOT}/data/tahoe/embeddings/cell",
    umap_cache_dir=f"{ROOT}/results/tahoe/umap",
)
ZENODO = Dataset(
    name="zenodo", nice="Zenodo",
    results_dir=f"{ROOT}/results/zenodo/grid",
    view="aggregate", pairzs="pairzs_withleak",
    report_dir=f"{ROOT}/zenodo_plots",
    bar_colors=("#453a49", "#6d3b47"),
    pairs_parquet="",   # set path
    h5ad="",            # set path
    cell_emb_dir="",    # set path
    umap_cache_dir="",  # set path
)
DATASETS = (TAHOE, ZENODO)   # row order in every two-row figure


def combo_score(run_dir, by, view, metric="MCC", pooling="pooled"):
    """One finished run's within-drug (by='drug') or within-cell-line (by='cell_line')
    score: the units are averaged within each seed first, then mean +/- SE across
    seeds. A unit with an undefined score (single-class) is dropped within its seed."""
    fname = "metrics_by_drug_raw.csv" if by == "drug" else "metrics_by_cell_line_raw.csv"
    path = os.path.join(run_dir, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    sub = df[(df["view"] == view) & (df["pooling"] == pooling) & (df["metric"] == metric)]
    per_seed = sub.dropna(subset=["value"]).groupby("seed")["value"].mean().to_numpy(dtype=float)
    if len(per_seed) == 0:
        return None
    se = per_seed.std(ddof=1) / np.sqrt(len(per_seed)) if len(per_seed) > 1 else np.nan
    return {"mean": float(per_seed.mean()), "se": float(se)}


def collect(ds, by, metric="MCC"):
    """One row per finished run of this dataset: split, cell, drug, model, mean, se."""
    rows = []
    for split in ds.splits:
        for d in sorted(glob.glob(f"{ds.results_dir}/{split}__*")):
            parts = os.path.basename(d).split("__")
            if not os.path.isdir(d) or len(parts) != 4:
                continue
            s = combo_score(d, by, ds.view, metric)
            if s is not None:
                rows.append({**dict(zip(["split", "cell", "drug", "model"], parts)), **s})
    return pd.DataFrame(rows, columns=["split", "cell", "drug", "model", "mean", "se"])


def save(fig, stem, dpi=300, formats=FORMATS, **kw):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=dpi, **kw)
    plt.close(fig)
    print(f"wrote {stem}.{{{','.join(formats)}}}")
