#!/usr/bin/env python
"""The profiles the figures are built from.

Each dataset's results dir, scoring view and pair-zero-shot split are set here once
and references across scripts for consistency. The split and view differ per dataset
by design (see paper): shared-label dataset is scored pairlevel on pairzs_withoutleak, 
single-cell label setting dataset is scored aggregate on pairzs_withleak.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

import figure_style as FS  
import report_common as RC  

ROOT = RC.ROOT

# one shared column label; each dataset maps it to its own split below
PAIRZS_TAG = "pairzs"
PAIRZS_LABEL = "Pair zero shot"
SPLIT_TAGS = ["random", PAIRZS_TAG, "loclo_loo", "lodo_loo"]

# SPLIT_NICE with both pairzs variants collapsed onto the unqualified label
SPLIT_NICE = {**RC.SPLIT_NICE,
              "pairzs_withleak": PAIRZS_LABEL,
              "pairzs_withoutleak": PAIRZS_LABEL}
TITLES = [SPLIT_NICE[RC.SPLIT_ORDER[0]], PAIRZS_LABEL,
          SPLIT_NICE["loclo_loo"], SPLIT_NICE["lodo_loo"]]


@dataclass(frozen=True)
class Dataset:
    name: str          # run-dir / file-stem tag
    nice: str          # display name
    results_dir: str
    view: str          # "pairlevel" | "aggregate"
    pairzs: str        # this dataset's pair-zero-shot split
    report_dir: str
    bar_colors: tuple  # (within-drug, within-cell-line), for grouped bars
    splits: list = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "splits",
                           ["random", self.pairzs, "loclo_loo", "lodo_loo"])

    @property
    def ablation_dir(self):
        return f"{self.report_dir}/ablations"

    @property
    def avg_csv(self):
        return f"{self.report_dir}/avg_min_max_mcc_by_split.csv"

    def split_for(self, tag):
        """SPLIT_TAGS entry -> this dataset's real split name."""
        return self.pairzs if tag == PAIRZS_TAG else tag


TAHOE = Dataset(
    name="tahoe", nice="Tahoe",
    results_dir=f"{ROOT}/results/tahoe/bal15_drugs",
    view="pairlevel", pairzs="pairzs_withoutleak",
    report_dir=f"{ROOT}/tahoe_plots",
    bar_colors=("#274c77", "#6096ba"),          # blue
)
ZENODO = Dataset(
    name="zenodo", nice="Zenodo",
    results_dir=f"{ROOT}/results/zenodo/grid",
    view="aggregate", pairzs="pairzs_withleak",  
    report_dir=f"{ROOT}/zenodo_plots",
    bar_colors=("#453a49", "#6d3b47"),           # mauve
)
DATASETS = (TAHOE, ZENODO)          # row order in every two-row figure
OUT_DIR = f"{ROOT}/plots_final"

_MISSING = object()


@contextmanager
def applied(modules, **attrs):
    """Set attrs on each module for the duration of the block, then restore."""
    saved = []
    for mod in modules:
        for k, v in attrs.items():
            saved.append((mod, k, getattr(mod, k, _MISSING)))
            setattr(mod, k, v)
    try:
        yield
    finally:
        for mod, k, old in reversed(saved):
            if old is _MISSING:
                delattr(mod, k)
            else:
                setattr(mod, k, old)
