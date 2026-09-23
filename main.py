#!/usr/bin/env python
# =====================================================================
# main.py  -->  main script for every (cell-embedding x drug-embedding x
#              predictive-model x split) combination.
# ---------------------------------------------------------------------
# CELL-LINE embeddings   (--cell-emb)
#     pca            : fit per split on the training cells only   (recomputed per fold)
#     scvi           : fit per split on the training cells only   (recomputed per fold)
#                      on raw counts (--scvi-input counts, tahoe) or log1p data
#                      (--scvi-input lognorm, zenodo)
#     gene_jepa     : precomputed, frozen  -> loaded once
#     gformer_cancer : precomputed, frozen  -> loaded once
#     scgpt_pan      : precomputed, frozen  -> loaded once
#
# DRUG embeddings        (--drug-emb)  -- all precomputed, keyed by drug name
#     morgan | biomed | chemberta | maccs | materials | molformer
#
# PREDICTIVE models       (--model)
#     mlp | lightgbm | xrfm
#
# SPLIT methods           (--split)
#     random             (01) StratifiedKFold on rows -- baseline, leaky, close to perfect MCC 
#     loclo_loo          (02) leave-one-cell-line-out (one fold per cell line)
#     lodo_loo           (03) leave-one-drug-out (one fold per drug)
#     pairzs_withleak    (04) grouped by (cell_line,drug) pair, single-cell leakage allowed
#     pairzs_withoutleak (05) pair zero-shot with double-blocking, no single-cell leakage
#
# --n-folds defaults (per split):
#     random                         : 3
#     pairzs_withleak                : 10  (-> 90% train/fold, matching lodo_loo;
#                                            ordinary k-fold)
#     pairzs_withoutleak             : 5   (double-blocked: train frac is ((n-1)/n)^2 and
#                                            total pairs ever tested (pooled) is 1/n, so
#                                            more folds trades test coverage for train data.
#                                            n=5 -> 64% train/fold, ~20% of pairs (~8/drug)
#                                            ever evaluated.)
#     loclo_loo, lodo_loo            : N/A -- fixed at #groups by LeaveOneGroupOut
#     --n-folds is possible to pass explicitly to override any of these.
#
# For every split the PCA/scVI embedding is fit on that split's training cells only.
# The fixed cell embeddings (gene_jepa / gformer_cancer / scgpt_pan) and the
# drug embeddings (also fixed) are loaded once and reused across folds.
#
# STAGES                 (--stage)
#     predict   : train + inference only. Writes the raw held-out predictions
#                 (one row per held-out cell x drug x seed) and stops.
#     evaluate  : reads those predictions back and computes every metric,
#                 table and plot. Doesn't touch h5ad and refits nothing, so
#                 scoring can be changed in the future without rerunning all.
#     all       : predict, then evaluate (the default)
#
# Outputs go to per-combination folders:
#     {out_root}/{split}__{cell_emb}__{drug_emb}__{model}/
#         predictions/seed{S}.parquet   raw inference  <- written by `predict`
#         run_manifest.json             what produced those predictions
#         metrics_*.csv, *.png          scores         <- written by `evaluate`
#
# =====================================================================

import os
import gc
import re
import sys
import glob
import hashlib
import json
import argparse
import datetime
import logging
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut, KFold,
    StratifiedShuffleSplit,
)
from sklearn.metrics import (
    matthews_corrcoef,          # MCC          (needs 0/1 predictions)
    roc_auc_score,              # AUROC        (needs probabilities)
    average_precision_score,    # AUPRC        (needs probabilities)
    balanced_accuracy_score,    # balanced acc (needs 0/1 predictions)
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# 1. Args  --  knobs from the .sh submit script
# =====================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Drug-sensitivity pipeline (cell-emb x drug-emb x model x split).")

    p.add_argument("--cell-emb", default="pca",
                   choices=["pca", "scvi", "gene_jepa", "gformer_cancer", "scgpt_pan"],
                   help="cell-line embedding")
    p.add_argument("--drug-emb", default="morgan",
                   choices=["morgan", "biomed", "chemberta", "maccs", "materials", "molformer"],
                   help="drug embedding (all precomputed, keyed by drug name).")
    p.add_argument("--model", default="mlp",
                   choices=["mlp", "lightgbm", "xrfm"],
                   help="predictive model.")
    p.add_argument("--split", default="loclo_loo",
                   choices=["random", "loclo_loo", "lodo_loo",
                            "pairzs_withleak", "pairzs_withoutleak"],
                   help="evaluation split / zero-shot protocol.")
    p.add_argument("--feature-mode", default="both",
                   choices=["both", "cell_only", "drug_only"],
                   help="Ablation: 'cell_only' zeroes out the drug-embedding block, so "
                        "predictions can only depend on which cell line it is (a marginal "
                        "per-cell-line propensity, not the drug). 'drug_only' zeroes out the "
                        "cell-embedding block the same way (a marginal per-drug propensity, "
                        "not the cell). Comparing either against the normal 'both' run tells "
                        "you whether the multimodal model is actually learning a cell x drug "
                        "interaction, or could match its score from one modality alone.")

    # which stage(s) to run 
    p.add_argument("--stage", default="all", choices=["all", "predict", "evaluate"],
                   help="predict = train + inference, save raw predictions and stop. "
                        "evaluate = load saved predictions and score them (no h5ad, "
                        "no refitting). all = both.")
    p.add_argument("--cache-dir", default=None,
                   help="on-disk cache of per-fold PCA/scVI embeddings, shared "
                        "across every drug-emb x model run. "
                        "Default: {root}/cache/cell_embeddings")
    p.add_argument("--no-cache", action="store_true",
                   help="always refit the cell embedding (ignore and do not "
                        "write the cache).")
    p.add_argument("--pred-dir", default=None,
                   help="where the raw prediction shards live. "
                        "Default: {out_dir}/predictions . Point --stage evaluate at "
                        "another run's folder to re-score it.")

    # ---- paths ----
    p.add_argument("--project-root", default=None)
    p.add_argument("--input-table", default=None,
                   help="row table with [barcode, cell_line, drug, sensitivity]. "
                        "Only those 4 columns are used, so one table serves every combination. "
                        "Default: {root}/data/tahoe/embeddings/train_pairs.parquet")
    p.add_argument("--h5ad", default=None,
                   help="single-cell h5ad; used only when --cell-emb is pca or scvi. "
                        "Default: {root}/data/tahoe/controls2000_merged_hvg5000_log1p_umap.h5ad")
    p.add_argument("--cell-emb-file", default=None,
                   help="precomputed frozen cell embedding parquet, indexed by barcode. "
                        "Used when --cell-emb is gene_jepa/gformer_cancer/scgpt_pan. "
                        "Default: {root}/data/tahoe/embeddings/cell/{cell_emb}_cell_embeddings.parquet")
    p.add_argument("--drug-emb-file", default=None,
                   help="drug embedding parquet, indexed by drug name. "
                        "Default: {root}/data/tahoe/embeddings/drug/{drug_emb}_drug_embeddings.parquet")
    p.add_argument("--out-root", default=None,
                   help="parent output dir. Default: {root}/results/tahoe")

    # ---- column names ----
    p.add_argument("--barcode-col", default="BARCODE_SUB_LIB_ID")
    p.add_argument("--log1p-layer", default="log1p",
                   help="adata layer for PCA (set to 'X' to use adata.X).")
    p.add_argument("--counts-layer", default="counts",
                   help="adata layer fed to scVI (its contents are declared by --scvi-input).")

    # ---- PCA / scVI knobs ----
    p.add_argument("--n-components", type=int, default=50,
                   help="cell-embedding dim for the refit-per-split embeddings "
                        "(PCA components / scVI latent size).")
    p.add_argument("--hvg-n", type=int, default=3000,
                   help="top-variance genes reselected on train cells (PCA only).")
    p.add_argument("--scvi-epochs", type=int, default=100,
                   help="max scVI epochs.")
    p.add_argument("--scvi-batch-size", type=int, default=None,
                   help="scVI minibatch size (scvi-tools default is 128). ")
    p.add_argument("--no-scvi-early-stopping", action="store_true",
                   help="train the full --scvi-epochs every fold.")
    p.add_argument("--scvi-patience", type=int, default=10,
                   help="early-stopping patience on elbo_validation.")
    p.add_argument("--verbose-scvi", action="store_true",
                   help="keep scVI/Lightning's per-fold log spam.")
    p.add_argument("--matmul-precision", default="high",
                   choices=["highest", "high", "medium"],
                   help="torch float32 matmul precision; 'high' uses the GPU's "
                        "Tensor Cores and is a free speedup on Ampere+.")
    p.add_argument("--scvi-n-top-genes", type=int, default=2000,
                   help="genes fed to scVI, selected by --scvi-hvg-flavor "
                        "(ignored for 'all'; default inherited).")
    p.add_argument("--scvi-input", default="counts", choices=["counts", "lognorm"],
                   help="What the --counts-layer array actually contains. "
                        "'counts' (default, tahoe): raw integer counts -> ZINB "
                        "likelihood, log_variational=True, seurat_v3 HVGs. "
                        "'lognorm' (zenodo): log1p(CPM) -> normal likelihood, "
                        "log_variational=False (no second log), no HVG reselection. "
                        "Sets the defaults for the three flags below; each can still "
                        "be overridden individually.")
    p.add_argument("--scvi-likelihood", default=None,
                   choices=["zinb", "nb", "poisson", "normal"],
                   help="Override the likelihood implied by --scvi-input.")
    p.add_argument("--scvi-log-variational", default=None, choices=["true", "false"],
                   help="Override whether scVI log1p-transforms its input internally. "
                        "Must be false for already-logged data.")
    p.add_argument("--scvi-hvg-flavor", default=None,
                   choices=["seurat_v3", "seurat", "cell_ranger", "variance", "all"],
                   help="Gene selection fed to scVI. seurat_v3 expects counts; "
                        "seurat/cell_ranger expect log data; 'variance' is the "
                        "train-only top-variance ranking PCAEmbedder uses; 'all' "
                        "skips selection and uses every gene.")
    p.add_argument("--allow-scvi-input-mismatch", action="store_true",
                   help="Downgrade the --scvi-input integrality check from an error "
                        "to a warning. Escape hatch only; do not use in a sweep.")

    # ---- training knobs ----
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    p.add_argument("--n-folds", type=int, default=None,
                   help="folds for random / pairzs_* splits. "
                        "Ignored by loclo_loo / lodo_loo (fixed by #groups). "
                        "Default (when omitted) is split-dependent -- 3 for "
                        "random, 10 for pairzs_withleak, 5 for "
                        "pairzs_withoutleak; see resolve_cfg() / the SPLIT "
                        "methods comment at the top of this file for why.")
    p.add_argument("--epochs", type=int, default=8, help="MLP epochs.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="fraction of train carved off as a validation set for xrfm/lightgbm.")

    # ---- xRFM hyperparameters (inherited defaults) ----
    p.add_argument("--xrfm-kernel", default="l2")
    p.add_argument("--xrfm-bandwidth", type=float, default=1.0)
    p.add_argument("--xrfm-exponent", type=float, default=1.0)
    p.add_argument("--xrfm-reg", type=float, default=1e-3)
    p.add_argument("--xrfm-iters", type=int, default=5)
    # ---- LightGBM device ('cpu' unless there is a GPU LightGBM build) ----
    p.add_argument("--lgbm-device", default="cpu", choices=["cpu", "cuda", "gpu"])
    p.add_argument("--xrfm-max-leaf-size", type=int, default=20_000,
                   help="rows per kernel leaf. Memory ~O(n^2), solve ~O(n^3), "
                        "so 60k needs ~14 GB on top of the data.")

    # ---- plotting / misc ----
    p.add_argument("--no-zoom", action="store_true",
                   help="disable y-axis auto-zoom on the metric plots.")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if unset).")
    return p

def panel_from_table(input_table):
    """Which drug panel is this row table? Derived from the filename so the
    output folder always matches the input.

    train_pairs.parquet        -> complete_drugpanel
    train_pairs_bal15.parquet  -> bal15_drugs
    zenodo_rows.parquet        -> "" (zenodo has a single panel; no extra
                                   folder level -- see out_dir below)
    anything else              -> the suffix after 'train_pairs_'
    """
    stem = os.path.splitext(os.path.basename(input_table))[0]
    if stem == "train_pairs":
        return "complete_drugpanel"
    if stem.startswith("train_pairs_"):
        suffix = stem[len("train_pairs_"):]
        return "bal15_drugs" if suffix == "bal15" else f"{suffix}_drugs"
    if stem == "zenodo_rows":
        return ""
    return _safe_name(stem)


# Split-dependent --n-folds default, used only when --n-folds isn't passed explicitly.
_DEFAULT_N_FOLDS_FLAT = 3
_DEFAULT_N_FOLDS_BY_SPLIT = {
    "pairzs_withleak": 10,
    "pairzs_withoutleak": 5,
}

_SCVI_INPUT_PRESETS = {
    "counts":  {"likelihood": "zinb",   "log_variational": True,  "hvg_flavor": "seurat_v3"},
    "lognorm": {"likelihood": "normal", "log_variational": False, "hvg_flavor": "all"},
}


def resolve_cfg(args):
    """Fill in path defaults that depend on other args."""
    root = args.project_root
    if args.input_table is None:
        args.input_table = f"{root}/data/tahoe/embeddings/train_pairs.parquet"
    if args.h5ad is None:
        args.h5ad = f"{root}/data/tahoe/controls2000_merged_hvg5000_log1p_umap.h5ad"
    if args.cell_emb_file is None:
        args.cell_emb_file = f"{root}/data/tahoe/embeddings/cell/{args.cell_emb}_cell_embeddings.parquet"
    if args.drug_emb_file is None:
        args.drug_emb_file = f"{root}/data/tahoe/embeddings/drug/{args.drug_emb}_drug_embeddings.parquet"
    if args.out_root is None:
        args.out_root = f"{root}/results/tahoe"
    if args.n_folds is None:
        args.n_folds = _DEFAULT_N_FOLDS_BY_SPLIT.get(args.split, _DEFAULT_N_FOLDS_FLAT)
    preset = _SCVI_INPUT_PRESETS[args.scvi_input]
    if args.scvi_likelihood is None:
        args.scvi_likelihood = preset["likelihood"]
    if args.scvi_log_variational is None:
        args.scvi_log_variational = preset["log_variational"]
    else:
        args.scvi_log_variational = (args.scvi_log_variational == "true")
    if args.scvi_hvg_flavor is None:
        args.scvi_hvg_flavor = preset["hvg_flavor"]
    args.zoom = not args.no_zoom
    args.panel = panel_from_table(args.input_table)
    if args.feature_mode == "cell_only":
        args.panel = f"{args.panel}_cellonly" if args.panel else "cellonly"
    elif args.feature_mode == "drug_only":
        args.panel = f"{args.panel}_drugonly" if args.panel else "drugonly"
    tag = f"{args.split}__{args.cell_emb}__{args.drug_emb}__{args.model}"
    # No panel level (e.g. zenodo, which has a single panel) -> results go
    # straight under out_root instead of out_root/{panel}/.
    args.out_dir = os.path.join(args.out_root, tag) if not args.panel \
        else os.path.join(args.out_root, args.panel, tag)
    args.tag = tag
    if args.pred_dir is None:
        args.pred_dir = os.path.join(args.out_dir, "predictions")
    if args.cache_dir is None:
        args.cache_dir = f"{root}/cache/cell_embeddings"
    return args


METRIC_NAMES = ["MCC", "AUROC", "AUPRC", "BalancedAcc"]
VIEWS = ["aggregate", "pairlevel"]


# =====================================================================
# 2. Reproducibility
# =====================================================================
def set_seed(seed):
    """Seed every RNG used here so runs are repeatable."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# 3. CELL-LINE EMBEDDINGS
# Turns cells into their embeddings. Every embedding method is wrapped
# into a class that will contain:
#       .barcodes                       -> the list of cells, in order, so we know which row is which (np.ndarray[str])
#       .per_split                      -> a yes/no flag: "does embedding need to be recomputed for every fold? (vs being fixed)"
#       .features(train_cell_ids, seed) -> (n_cells, dim) float32, aligned to .barcodes --> hands back a table of numbers, one row per cell.
#    train_cell_ids are indices into .barcodes; frozen embeddings ignore them.
# The point of this block is to deal with differences in embeddings here, so that the main loop can just call .features(...)
# and not worry about the details anymore.
# =====================================================================
def release_host_memory():
    gc.collect()
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


class PCAEmbedder:
    """Per-split PCA: train-only HVG reselection + PCA fit on train cells, transform all.
    Identical behaviour to fit_pca_features() in the original scripts."""
    per_split = True

    def __init__(self, h5ad_path, log1p_layer, hvg_n, n_components):
        import anndata as ad
        import scipy.sparse as sp
        print(f"Loading h5ad for PCA: {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        X = adata.X if log1p_layer in (None, "X") else adata.layers[log1p_layer]
        if sp.issparse(X):
            X = X.toarray()
        self.Xlog = np.ascontiguousarray(X, dtype=np.float32)     # (n_cells, n_genes)
        self.barcodes = adata.obs.index.astype(str).values
        self.hvg_n = hvg_n
        self.k = n_components
        print(f"  gene matrix: {self.Xlog.shape[0]:,} cells x {self.Xlog.shape[1]:,} genes")

    def features(self, train_cell_ids, seed):
        from sklearn.decomposition import PCA
        Xlog = self.Xlog
        # 1. reselect the top-HVG_N most variable genes, measured on TRAIN cells only
        var = Xlog[train_cell_ids].var(axis=0)
        if self.hvg_n < Xlog.shape[1]:
            hvg_idx = np.argpartition(var, -self.hvg_n)[-self.hvg_n:]
        else:
            hvg_idx = np.arange(Xlog.shape[1])
        # 2. fit PCA on train cells only
        pca = PCA(n_components=self.k, svd_solver="randomized", random_state=seed)
        pca.fit(Xlog[np.ix_(train_cell_ids, hvg_idx)])
        # 3. transform EVERY cell (held-out cells never shaped the fit)
        return pca.transform(Xlog[:, hvg_idx]).astype(np.float32)


def _sample_values(mat, seed=0, n_rows=2000):
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(mat.shape[0], min(n_rows, mat.shape[0]), replace=False))
    sub = mat[rows]
    return np.asarray(sub.tocsr().data if sp.issparse(sub) else sub).ravel()


class ScviEmbedder:
    """Per-split scVI, mostly matching inherited scripts:
        1. gene selection (hvg_flavor, n_top_genes) on the TRAIN cells
        2. fit SCVI(n_latent, likelihood, log_variational) on those train genes
        3. get_latent_representation on a var-matched AnnData to embed ALL cells
           with the now-frozen encoder (held-out cells never shaped the fit).

    Input is whatever --counts-layer holds ('X' for adata.X), declared by
    --scvi-input: raw counts (zinb, log_variational=True, seurat_v3) or log1p
    data (normal, log_variational=False, all genes). A sampled integrality check
    raises if the layer contradicts the configured likelihood.
    Requires scvi-tools, scanpy, and scikit-misc (for seurat_v3)."""
    per_split = True

    def __init__(self, h5ad_path, counts_layer, n_latent, epochs, n_top_genes,
                 hvg_max_cells=5000, batch_size=None, early_stopping=True,
                 patience=10, quiet=True, likelihood="zinb", log_variational=True,
                 hvg_flavor="seurat_v3", allow_mismatch=False):
        import anndata as ad
        import scipy.sparse as sp
        print(f"Loading h5ad for scVI: {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        self.barcodes = adata.obs.index.astype(str).values
        self.var_names = np.asarray(adata.var_names)

        X = adata.X if counts_layer in (None, "X") else adata.layers[counts_layer]
        if sp.issparse(X):
            self.counts = sp.csc_matrix(X, dtype=np.float32)
            nnz, size = self.counts.nnz, self.counts.data.nbytes * 1.5
            dens = nnz / (self.counts.shape[0] * self.counts.shape[1])
            fmt = f"sparse CSC, {100*dens:.1f}% non-zero"
        else:
            self.counts = np.ascontiguousarray(X, dtype=np.float32)
            size = self.counts.nbytes
            fmt = "dense"
        del adata, X
        gc.collect()

        n_cells, n_genes = self.counts.shape
        print(f"  counts: {n_cells:,} cells x {n_genes:,} genes ({fmt}, "
              f"~{size/1e9:.2f} GB, loaded once and reused for every fold)")

        self.n_latent = n_latent
        self.epochs = epochs
        self.n_top_genes = n_top_genes
        self.hvg_max_cells = hvg_max_cells
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.quiet = quiet
        self.likelihood = likelihood
        self.log_variational = log_variational
        self.hvg_flavor = hvg_flavor
        self.allow_mismatch = allow_mismatch
        self._logged_settings = False

        vals = _sample_values(self.counts)
        is_int = bool(np.all(np.abs(vals - np.rint(vals)) < 1e-6))
        has_neg = bool(np.any(vals < -1e-6))
        count_lik = self.likelihood in ("zinb", "nb", "poisson")

        problems = []
        if count_lik and not is_int:
            problems.append(f"gene_likelihood={self.likelihood!r} needs integer counts, "
                            f"but layer {counts_layer!r} is non-integer. "
                            f"Did you mean --scvi-input lognorm?")
        if count_lik and not self.log_variational:
            problems.append("count likelihoods expect log_variational=True.")
        if self.likelihood == "normal" and self.log_variational:
            problems.append("gene_likelihood='normal' with log_variational=True would log "
                            "already-logged input a second time.")
        if self.likelihood == "normal" and is_int:
            problems.append(f"gene_likelihood='normal' but layer {counts_layer!r} looks like "
                            "integer counts. Did you mean --scvi-input counts?")
        if has_neg:
            problems.append("input contains negative values; scVI expects counts or log1p data.")

        if problems:
            msg = "scVI input/model mismatch:\n  - " + "\n  - ".join(problems)
            if self.allow_mismatch:
                print(f"  [scVI] WARNING (--allow-scvi-input-mismatch): {msg}")
            else:
                raise ValueError(msg + "\n(override with --allow-scvi-input-mismatch)")
        print(f"  [scVI] input={counts_layer!r} integer={is_int} likelihood={self.likelihood} "
              f"log_variational={self.log_variational} hvg_flavor={self.hvg_flavor}")

    # ---- helpers -----------------------------------------------------------
    def _slice_rows(self, mat, rows):
        import scipy.sparse as sp
        return mat[rows].tocsr() if sp.issparse(mat) else mat[rows]

    def _select_cols(self, cols):
        import scipy.sparse as sp
        if sp.issparse(self.counts):
            return self.counts[:, cols].tocsr()
        return np.ascontiguousarray(self.counts[:, cols])

    def _quiet(self):
        import logging
        import scvi
        for name in ("lightning", "lightning.pytorch", "pytorch_lightning",
                     "lightning.pytorch.utilities.rank_zero",
                     "lightning.pytorch.accelerators.cuda",
                     "lightning.fabric.utilities.seed",
                     "pytorch_lightning.utilities.seed", "scvi"):
            logging.getLogger(name).setLevel(logging.ERROR)
        try:
            scvi.settings.verbosity = logging.ERROR
        except Exception:
            pass
        warnings.filterwarnings("ignore", module=r"lightning.*")

    def _train(self, model):
        """Train with early stopping when applicable/asked."""
        kw = {"max_epochs": self.epochs, "enable_progress_bar": False}
        if self.batch_size:
            kw["batch_size"] = self.batch_size
        if self.early_stopping:
            kw.update(early_stopping=True,
                      early_stopping_monitor="elbo_validation",
                      early_stopping_patience=self.patience,
                      check_val_every_n_epoch=1)
        try:
            model.train(**kw)
        except TypeError as e:
            print(f"    [scVI] train() rejected {sorted(kw)} ({e}); "
                  f"falling back to max_epochs only.")
            model.train(max_epochs=self.epochs, enable_progress_bar=False)
        try:
            return len(model.history["elbo_train"])
        except Exception:
            return None

    @staticmethod
    def _release(model):
        """Optimization helper: Without this the run grows by ~1.4 GB per fold."""
        import scvi
        try:
            model.deregister_manager()          # scvi-tools >= 0.20
        except Exception:
            pass
        for store in ("_setup_adata_manager_store", "_per_instance_manager_store"):
            s = getattr(scvi.model.SCVI, store, None)
            if isinstance(s, dict):
                s.clear()                      
        for cls_name in ("SCVI", "LinearSCVI"):
            cls = getattr(scvi.model, cls_name, None)
            for store in ("_setup_adata_manager_store", "_per_instance_manager_store"):
                s = getattr(cls, store, None)
                if isinstance(s, dict):
                    s.clear()
        release_host_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- the per-fold embedding -------------------------------------------
    def features(self, train_cell_ids, seed):
        import anndata as ad
        import scanpy as sc
        import scipy.sparse as sp
        import scvi
        import pandas as pd
        if self.quiet:
            self._quiet()
        scvi.settings.num_workers = 0
        scvi.settings.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. gene selection (hvg_flavor) on train cells only; hvg_cols stays sorted
        hvg_cells = train_cell_ids
        if len(hvg_cells) > self.hvg_max_cells:
            rng = np.random.default_rng(seed)
            hvg_cells = np.sort(rng.choice(hvg_cells, self.hvg_max_cells, replace=False))
        n_genes = self.counts.shape[1]
        # n_top_genes >= n_genes skips scanpy even for seurat_v3, which hashes to the legacy
        # cache key. Benign: seurat_v3 returns every gene in that case, so the fit is identical.
        if self.hvg_flavor == "all" or self.n_top_genes >= n_genes:
            hvg_cols = np.arange(n_genes)
        elif self.hvg_flavor == "variance":
            sub = self._slice_rows(self.counts, hvg_cells)
            if sp.issparse(sub):
                m = np.asarray(sub.mean(axis=0)).ravel()
                v = np.asarray(sub.multiply(sub).mean(axis=0)).ravel() - m ** 2
            else:
                v = sub.var(axis=0)
            hvg_cols = np.sort(np.argpartition(v, -self.n_top_genes)[-self.n_top_genes:])
        else:
            tr_ad = ad.AnnData(X=self._slice_rows(self.counts, hvg_cells),
                               var=pd.DataFrame(index=self.var_names))
            sc.pp.highly_variable_genes(tr_ad, n_top_genes=self.n_top_genes,
                                        flavor=self.hvg_flavor, subset=True)
            hvg_cols = np.where(np.isin(self.var_names, tr_ad.var_names.tolist()))[0]
            del tr_ad
            gc.collect()
        var_hvg = pd.DataFrame(index=self.var_names[hvg_cols])
        if not self._logged_settings:
            print(f"    [scVI] genes fed to scVI: {len(hvg_cols):,} of {n_genes:,} "
                  f"(hvg_flavor={self.hvg_flavor})")
            self._logged_settings = True

        counts_hvg = self._select_cols(hvg_cols)            # (n_cells, n_top_genes)

        # 2. fit scVI on the train cells' HVG counts
        train_hvg = ad.AnnData(X=self._slice_rows(counts_hvg, train_cell_ids),
                               var=var_hvg)
        scvi.model.SCVI.setup_anndata(train_hvg)
        try:
            model = scvi.model.SCVI(train_hvg, n_latent=self.n_latent,
                                    gene_likelihood=self.likelihood,
                                    log_variational=self.log_variational)
        except (TypeError, ValueError) as e:
            import importlib.metadata as _md
            raise RuntimeError(
                f"scvi-tools {_md.version('scvi-tools')} rejected "
                f"gene_likelihood={self.likelihood!r} / log_variational={self.log_variational}: {e}"
            ) from e

        # `log_variational` reaches VAE through **model_kwargs; assert it actually landed.
        mod = model.module
        assert getattr(mod, "log_variational") == self.log_variational, \
            "log_variational was not applied to the VAE"
        assert str(getattr(mod, "gene_likelihood")) == self.likelihood, \
            f"gene_likelihood was not applied (module reports {mod.gene_likelihood!r})"
        n_epochs = self._train(model)

        # 3. embed all cells with the frozen encoder 
        all_hvg = ad.AnnData(X=counts_hvg, var=var_hvg)
        latent = model.get_latent_representation(all_hvg).astype(np.float32)

        # 4. every reference back before the next fold starts
        self._release(model)
        del model, train_hvg, all_hvg, counts_hvg
        gc.collect()
        if n_epochs is not None:
            print(f"    [scVI] trained {n_epochs} epoch(s) of max {self.epochs}"
                  f"{' (early stopped)' if n_epochs < self.epochs else ''}")
        return latent


class PrecomputedCellEmbedder:
    """Frozen foundation-model cell embedding: load a parquet indexed by barcode, once."""
    per_split = False

    def __init__(self, path):
        print(f"Loading frozen cell embedding: {path}")
        df = pd.read_parquet(path)
        self.barcodes = df.index.astype(str).values
        self._feats = df.values.astype(np.float32)
        self._cache = None
        print(f"  cell embedding: {self._feats.shape[0]:,} cells x {self._feats.shape[1]:,} dims")

    def features(self, train_cell_ids, seed):
        return self._feats   # frozen: identical every fold/seed


# =====================================================================
# 3b. PER-FOLD EMBEDDING CACHE
#
# The cell embedding for a given (split, fold, seed) does not depend on the
# drug embedding or the predictive model. Without a cache, running the grid
#   6 drug-embs x 3 models  with  --cell-emb scvi --split loclo_loo
# means 18 runs x 129 fits = 2,322 scVI trainings, of which only 129 are
# actually distinct. This caches each fitted embedding to disk so the other
# 2,193 become the simpler task of a file read.
#
# The cache key is a hash of (embedder, its hyperparameters,
# seed, and the exact set of training cell indices). The training cells are
# what actually determine the fit.
# =====================================================================
_CACHE_VERSION = {"pca": 1, "scvi": 1}


def _safe_name(s, n=48):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s))[:n]


def h5ad_barcodes(path):
    """Reads just obs_names from an h5ad, without loading X.

    Lets a fully-cached run skip opening the count matrix entirely. For efficiency and speed."""
    import anndata as ad
    try:
        a = ad.read_h5ad(path, backed="r")
        bc = np.array(a.obs_names.astype(str).values, copy=True)
        try:
            a.file.close()
        except Exception:
            pass
        del a
        gc.collect()
        return bc
    except Exception as e:
        print(f"  [cache] backed read of {path} failed ({e}); falling back to a full read")
        return np.asarray(ad.read_h5ad(path).obs.index.astype(str).values)


class CachedCellEmbedder:
    """Wraps a refit-per-split embedder (PCA / scVI) with an on-disk cache.

    On a run where every fold hits the cache, the h5ad is never opened and
    scvi-tools is never imported."""
    per_split = True
    cacheable = True

    def __init__(self, factory, barcodes, cache_dir, tag, key_params, hashed_keys=None):
        self._factory = factory
        self._inner = None
        self.barcodes = barcodes
        self.dir = os.path.join(cache_dir, tag)
        self.tag = tag
        self.key_params = key_params
        # the subset of key_params that goes into the hash (default: all of it)
        self.hash_params = key_params if hashed_keys is None else \
            {k: key_params[k] for k in hashed_keys}
        self.context = (None, None)          # (split, fold_label), for filenames
        self.hits = self.misses = 0
        print(f"Cell-embedding cache ({tag}): {self.dir}")

    def _key(self, train_cell_ids, seed):
        ids = np.ascontiguousarray(np.unique(np.asarray(train_cell_ids, dtype=np.int64)))
        h = hashlib.sha256()
        h.update(json.dumps({"tag": self.tag, "cache_version": _CACHE_VERSION[self.tag],
                             "seed": int(seed), "params": self.hash_params},
                            sort_keys=True, default=str).encode())
        h.update(np.int64(ids.size).tobytes())
        h.update(ids.tobytes())
        return h.hexdigest()[:16]

    def _paths(self, key):
        split, fold = self.context
        prefix = f"{_safe_name(split or 'run')}__fold{_safe_name(fold or '?')}"
        stem = os.path.join(self.dir, f"{prefix}__{key}")
        return stem + ".npy", stem + ".json"

    def features(self, train_cell_ids, seed):
        key = self._key(train_cell_ids, seed)
        npy, sidecar = self._paths(key)

        # ---- try the cache ----
        hit = None
        for cand in ([npy] if os.path.exists(npy) else []) or glob.glob(
                os.path.join(self.dir, f"*__{key}.npy")):      # prefix may differ
            try:
                feats = np.load(cand)
            except Exception as e:
                print(f"    [cache] unreadable {os.path.basename(cand)} ({e}); refitting")
                continue
            if feats.shape[0] != len(self.barcodes):
                print(f"    [cache] {os.path.basename(cand)} has {feats.shape[0]:,} rows, "
                      f"expected {len(self.barcodes):,}; refitting")
                continue
            hit = feats
            break
        if hit is not None:
            self.hits += 1
            print(f"    [cache] hit  {self.tag} {key} "
                  f"({hit.shape[1]} dims) -- no refit")
            return np.ascontiguousarray(hit, dtype=np.float32)

        # ---- miss: build the real embedder (once) and fit ----
        self.misses += 1
        print(f"    [cache] miss {self.tag} {key} -- fitting")
        if self._inner is None:
            self._inner = self._factory()
            if not np.array_equal(np.asarray(self._inner.barcodes), self.barcodes):
                raise RuntimeError(
                    "barcode order from the backed h5ad read does not match the "
                    "full load -- the cache index would be wrong.")
        feats = np.ascontiguousarray(
            self._inner.features(train_cell_ids, seed), dtype=np.float32)

        def _write_npy(p):
            with open(p, "wb") as fh:
                np.save(fh, feats)
        atomic_write(npy, _write_npy)
        meta = {"tag": self.tag, "key": key, "cache_version": _CACHE_VERSION[self.tag],
                "split": self.context[0], "fold": str(self.context[1]),
                "seed": int(seed), "n_train_cells": int(np.unique(train_cell_ids).size),
                "shape": list(feats.shape), "params": self.key_params,
                "hashed_params": sorted(self.hash_params),
                "written_utc": datetime.datetime.now(datetime.timezone.utc)
                                       .isoformat(timespec="seconds")}
        atomic_write(sidecar,
                     lambda p: open(p, "w").write(json.dumps(meta, indent=2, default=str)))
        return feats

    def report(self):
        n = self.hits + self.misses
        if n:
            print(f"  [cache] {self.tag}: {self.hits}/{n} folds served from cache, "
                  f"{self.misses} fitted")


def make_cell_embedder(cfg):
    """Frozen embeddings load straight from parquet. PCA/scVI are wrapped in
    the on-disk cache unless --no-cache."""
    if cfg.cell_emb not in ("pca", "scvi"):
        return PrecomputedCellEmbedder(cfg.cell_emb_file)   # gene_jepa / gformer / scgpt

    try:
        h5_size = os.path.getsize(cfg.h5ad)
    except OSError:
        h5_size = -1
    hashed_keys = None

    if cfg.cell_emb == "pca":
        key_params = {"h5ad": os.path.abspath(cfg.h5ad), "h5ad_bytes": h5_size,
                      "layer": cfg.log1p_layer, "hvg_n": cfg.hvg_n,
                      "n_components": cfg.n_components}
        factory = lambda: PCAEmbedder(cfg.h5ad, cfg.log1p_layer,
                                      cfg.hvg_n, cfg.n_components)
    else:
        try:
            import importlib.metadata as _md
            scvi_version = _md.version("scvi-tools")    
        except Exception:
            scvi_version = "unknown"
        key_params = {"h5ad": os.path.abspath(cfg.h5ad), "h5ad_bytes": h5_size,
                      "layer": cfg.counts_layer, "n_latent": cfg.n_components,
                      "max_epochs": cfg.scvi_epochs, "n_top_genes": cfg.scvi_n_top_genes,
                      "batch_size": cfg.scvi_batch_size,
                      "early_stopping": not cfg.no_scvi_early_stopping,
                      "patience": cfg.scvi_patience,
                      "scvi_tools": scvi_version,
                      "scvi_input": cfg.scvi_input,
                      "likelihood": cfg.scvi_likelihood,
                      "log_variational": cfg.scvi_log_variational,
                      "hvg_flavor": cfg.scvi_hvg_flavor}
        # --scvi-input counts runs the pre-lognorm code path, so hash the pre-lognorm
        # dict and keep existing tahoe entries valid; any other config hashes everything.
        new_fields = {"scvi_input": "counts", **_SCVI_INPUT_PRESETS["counts"]}
        if all(key_params[k] == v for k, v in new_fields.items()):
            hashed_keys = [k for k in key_params if k not in new_fields]
        factory = lambda: ScviEmbedder(cfg.h5ad, cfg.counts_layer, cfg.n_components,
                                       cfg.scvi_epochs, cfg.scvi_n_top_genes,
                                       batch_size=cfg.scvi_batch_size,
                                       early_stopping=not cfg.no_scvi_early_stopping,
                                       patience=cfg.scvi_patience,
                                       quiet=not cfg.verbose_scvi,
                                       likelihood=cfg.scvi_likelihood,
                                       log_variational=cfg.scvi_log_variational,
                                       hvg_flavor=cfg.scvi_hvg_flavor,
                                       allow_mismatch=cfg.allow_scvi_input_mismatch)

    if cfg.no_cache:
        print(f"Cell-embedding cache disabled (--no-cache); {cfg.cell_emb} refits every fold.")
        return factory()
    return CachedCellEmbedder(factory, h5ad_barcodes(cfg.h5ad),
                              cfg.cache_dir, cfg.cell_emb, key_params, hashed_keys)


# =====================================================================
# 4. MLP  (in_dim = cell_dim + drug_dim, inferred)
# =====================================================================
class MLP(nn.Module):
    """(cell_dim + drug_dim) -> 256 -> 64 -> 1. One raw logit out."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# =====================================================================
# 5. Predictive models
#    fit_predict_xxx(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, y_np, tr, va, cfg, device, seed)
#    i.e.
#    given this fold's cell features, the drug matrix, the per-row cell/drug indices, labels y,
#    and train/val row indices -> returns val-row probabilities (np.ndarray).
#
# Fits on this fold's training rows and produces predictions for the held-out rows.
# =====================================================================
def _build_X(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, rows):
    """Materialize [cell-emb | drug-emb] for the given rows (for tree/kernel models)."""
    return np.concatenate(
        [cell_feats[cell_idx_np[rows]], drug_matrix[drug_idx_np[rows]]], axis=1
    ).astype(np.float32)


def fit_predict_mlp(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, y_np,
                    tr, va, cfg, device, seed, return_model=False):
    cell_t = torch.tensor(cell_feats, device=device)
    drug_t = torch.tensor(drug_matrix, device=device)
    cell_idx = torch.tensor(cell_idx_np, dtype=torch.long, device=device)
    drug_idx = torch.tensor(drug_idx_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    tr_t = torch.tensor(tr, dtype=torch.long, device=device)
    va_t = torch.tensor(va, dtype=torch.long, device=device)

    # single-class train fold -> predict the constant, no fit
    y_train = y[tr_t]
    uniq = torch.unique(y_train)
    if uniq.numel() == 1:
        const = float(uniq.item())
        preds = torch.full((va_t.numel(),), const, device=device)
        return (preds.cpu().numpy(), None) if return_model else preds.cpu().numpy()

    in_dim = cell_feats.shape[1] + drug_matrix.shape[1]
    model = MLP(in_dim).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_train = tr_t.numel()
    bs = cfg.batch_size
    for _ in range(cfg.epochs):
        model.train()
        perm = tr_t[torch.randperm(n_train, device=device)]
        for start in range(0, n_train, bs):
            batch = perm[start:start + bs]
            if batch.numel() < 2:                       # BatchNorm needs >=2 rows
                continue
            xb = torch.cat([cell_t[cell_idx[batch]], drug_t[drug_idx[batch]]], dim=1)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), y[batch])
            loss.backward()
            optimizer.step()

    model.eval()
    preds = torch.empty(va_t.numel(), device=device)
    with torch.no_grad():
        for start in range(0, va_t.numel(), bs):
            batch = va_t[start:start + bs]
            xb = torch.cat([cell_t[cell_idx[batch]], drug_t[drug_idx[batch]]], dim=1)
            preds[start:start + batch.numel()] = torch.sigmoid(model(xb))
    if return_model:
        return preds.cpu().numpy(), model
    return preds.cpu().numpy()


def fit_predict_lightgbm(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, y_np,
                         tr, va, cfg, device, seed):
    """LightGBM head, hyperparameters aligned to inherited wrapper, with
    the two inherited robustness guards: (1) if the training fold is single-class, skip
    fitting and predict that constant; (2) drop zero-variance columns on train.
    No class-weighting."""
    import lightgbm as lgb
    X_tr = _build_X(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, tr)
    X_va = _build_X(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, va)
    y_tr = y_np[tr].astype(int)

    # (1) single-class train fold -> predict the constant, no fit
    uniq = np.unique(y_tr)
    if uniq.size == 1:
        return np.full(len(va), float(uniq[0]), dtype=np.float32)

    # (2) drop zero-variance columns (fit mask on train, apply to val)
    keep = X_tr.std(axis=0) > 0
    if not keep.all():
        X_tr, X_va = X_tr[:, keep], X_va[:, keep]

    clf = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8, max_bin=63,
        reg_lambda=1.0, device_type=cfg.lgbm_device,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    clf.fit(X_tr, y_tr)
    classes = list(clf.classes_)
    proba = clf.predict_proba(X_va)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(va), np.float32)


def fit_predict_xrfm(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, y_np,
                     tr, va, cfg, device, seed):
    """xRFM head, matching inherited XRFMClassifier config.

    Standardization: cell and drug embeddings live on different raw scales
    xRFM has no way to correct for this on its own. z-score both
    blocks together using train statistics only, applied to train+val.
    """
    from xrfm import xRFM

    agop_cap = os.environ.get("XRFM_AGOP_SUBSET_CAP")
    if agop_cap and not getattr(xRFM, "_agop_subset_capped", False):
        cap = int(agop_cap)
        _orig_get_agop_on_subset = xRFM._get_agop_on_subset

        def _capped_get_agop_on_subset(self, X, y, subset_size=50_000, **kw):
            return _orig_get_agop_on_subset(self, X, y, subset_size=min(subset_size, cap), **kw)

        xRFM._get_agop_on_subset = _capped_get_agop_on_subset
        xRFM._agop_subset_capped = True
        print(f"    [xrfm] AGOP split-subset capped at {cap:,} rows (XRFM_AGOP_SUBSET_CAP)")

    X_tr = _build_X(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, tr)
    X_va = _build_X(cell_feats, drug_matrix, cell_idx_np, drug_idx_np, va)
    y_tr = y_np[tr]

    # single-class train fold -> predict the constant, no fit
    uniq_tr = np.unique(y_tr)
    if uniq_tr.size == 1:
        return np.full(len(va), float(uniq_tr[0]), dtype=np.float32)

    mu, sd = X_tr.mean(axis=0, keepdims=True), X_tr.std(axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)          # constant columns -> 0 after centering, safe
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    # xRFM.fit requires a validation set so carving a stratified one from train
    itr, ival = _internal_val_split(y_tr, cfg.val_frac, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rfm_params = {
        "model": {"kernel": cfg.xrfm_kernel, "bandwidth": cfg.xrfm_bandwidth,
                  "exponent": cfg.xrfm_exponent, "diag": False,
                  "bandwidth_mode": "adaptive"},
        "fit": {"reg": cfg.xrfm_reg, "iters": cfg.xrfm_iters,
                "verbose": False, "early_stop_rfm": True},
    }
    default_rfm_params = {
        "model": {"kernel": "l2_high_dim", "exponent": 1.0, "bandwidth": cfg.xrfm_bandwidth,
                  "diag": False, "bandwidth_mode": "adaptive"},
        "fit": {"get_agop_best_model": True, "return_best_params": False,
                "reg": 1e-3, "iters": 0, "early_stop_rfm": False, "verbose": False},
    }
    model = xRFM(rfm_params=rfm_params, device=device, tuning_metric="auc",
                 max_leaf_size=cfg.xrfm_max_leaf_size, default_rfm_params=default_rfm_params,
                 split_method="top_vector_agop_on_subset", n_trees=1)

    def to_t(a, dtype):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

    try:
        model.fit(to_t(X_tr[itr], torch.float32), to_t(y_tr[itr].astype(np.int64), torch.long),
                  to_t(X_tr[ival], torch.float32), to_t(y_tr[ival].astype(np.int64), torch.long))
        proba = model.predict_proba(to_t(X_va, torch.float32))
    except AttributeError as e:
        print(f"    [xrfm WARNING] fit failed ({e}); falling back to train "
              f"positive rate as a constant prediction for this fold.")
        return np.full(len(va), float(y_tr.mean()), dtype=np.float32)
    if torch.is_tensor(proba):
        proba = proba.detach().float().cpu().numpy()
    proba = np.asarray(proba)
    if proba.ndim == 2:
        return proba[:, -1]
    return proba.reshape(-1)


def _internal_val_split(y_tr, val_frac, seed):
    """Stratified train/val indices within the training rows (for xrfm/lightgbm)."""
    n = len(y_tr)
    if len(np.unique(y_tr)) < 2 or val_frac <= 0 or n < 10:
        idx = np.arange(n)
        return idx, idx                      # degenerate: reuse train as val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    itr, ival = next(sss.split(np.zeros(n), y_tr))
    return itr, ival


MODEL_DISPATCH = {
    "mlp": fit_predict_mlp,
    "lightgbm": fit_predict_lightgbm,
    "xrfm": fit_predict_xrfm,
}


# =====================================================================
# 6. Splits
#    Note: fold_label isn't always the same type.
#    For the grouped and k-fold splits it's just a fold number (1, 2, 3).
#    But for the leave-one-out splits, it's the name of the held-out group
# =====================================================================
def make_pair_key(cell_line, drug):
    return (pd.Series(cell_line).astype(str) + "||" + pd.Series(drug).astype(str)).values


def iter_splits(split, N, y_np, cell_line, drug, cell_idx_np, n_folds, seed):

    if split == "random":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(N), y_np), start=1):
            yield fold, tr, va

    elif split in ("loclo_loo", "lodo_loo"):
        groups = cell_line if split == "loclo_loo" else drug
        what = "cell line" if split == "loclo_loo" else "drug"
        logo = LeaveOneGroupOut()
        n = len(np.unique(groups))
        for i, (tr, va) in enumerate(logo.split(np.zeros(N), y_np, groups=groups), start=1):
            held = str(np.unique(groups[va])[0])
            assert held not in set(np.unique(groups[tr])), \
                f"The held-out {what} leaked into training!"
            print(f"  fold {i}/{n} [{held}] ({what})")
            yield held, tr, va                       # fold_label == held-out group name

    elif split == "pairzs_withleak":
        print(f"  n_folds={n_folds} -> ~{100*(n_folds-1)/n_folds:.0f}% train/fold; "
              f"every pair is tested exactly once, pooled across folds (ordinary k-fold).")
        pair_key = make_pair_key(cell_line, drug)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(sgkf.split(np.zeros(N), y_np, groups=pair_key), start=1):
            tr_p, va_p = set(np.unique(pair_key[tr])), set(np.unique(pair_key[va]))
            assert tr_p.isdisjoint(va_p), "A (cell_line,drug) pair leaked across the split!"
            train_cells = np.unique(cell_idx_np[tr])
            test_cells = np.unique(cell_idx_np[va])
            leaked = np.intersect1d(train_cells, test_cells, assume_unique=True)
            print(f"  fold {fold}/{n_folds}: {len(va_p)} unseen pairs | "
                  f"single-cell leakage: {len(leaked):,} barcodes in both "
                  f"({100*len(leaked)/max(len(test_cells),1):.1f}% of test cells)")
            yield fold, tr, va

    elif split == "pairzs_withoutleak":
        # "Double blocking": independent pair-fold and cell-fold; a row is test only if
        # both are on-fold, train only if both are off-fold, else dropped. Guarantees
        # disjoint pairs and disjoint cells to prevent single-cell leakage. (downside: smaller data size)
        train_frac = ((n_folds - 1) / n_folds) ** 2
        coverage_frac = 1.0 / n_folds
        pair_key = make_pair_key(cell_line, drug)
        uniq_cells = np.unique(cell_idx_np)
        pfold_row = np.full(N, -1, dtype=np.int64)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for f, (_, te) in enumerate(sgkf.split(np.zeros(N), y_np, groups=pair_key)):
            pfold_row[te] = f
        assert (pfold_row >= 0).all(), "some row got no pair-fold"

        cfold_of_cell = np.full(uniq_cells.max() + 1, -1, dtype=np.int64)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for f, (_, te) in enumerate(kf.split(uniq_cells)):
            cfold_of_cell[uniq_cells[te]] = f
        cfold_row = cfold_of_cell[cell_idx_np]
        assert (cfold_row >= 0).all(), "some row got no cell-fold"

        for fold in range(1, n_folds + 1):
            f = fold - 1
            tr = np.where((pfold_row != f) & (cfold_row != f))[0]
            va = np.where((pfold_row == f) & (cfold_row == f))[0]
            dropped = N - len(tr) - len(va)
            tr_p, va_p = set(np.unique(pair_key[tr])), set(np.unique(pair_key[va]))
            train_cells, test_cells = np.unique(cell_idx_np[tr]), np.unique(cell_idx_np[va])
            assert tr_p.isdisjoint(va_p), "A pair leakage across the split!"
            assert np.intersect1d(train_cells, test_cells, assume_unique=True).size == 0, \
                "A single cell leaked, double-blocking is broken."
            print(f"  fold {fold}/{n_folds}: test {len(va):,} rows / {len(va_p)} pairs, "
                  f"train {len(tr):,} rows / {len(tr_p)} pairs, dropped {dropped:,} "
                  f"off-diagonal rows | single-cell leakage: 0 (blocked)")
            yield fold, tr, va
    else:
        raise ValueError(f"unknown split: {split}")


# =====================================================================
# 7. Metrics & Plotting  
# =====================================================================
def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_bin = (y_prob >= 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        return {m: np.nan for m in METRIC_NAMES}
    return {
        "MCC": matthews_corrcoef(y_true, y_bin),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "BalancedAcc": balanced_accuracy_score(y_true, y_bin),
    }


def per_drug_variance_stats(probs, drug_idx):
    """Memorization check (inherited).

    Splits the variance of predicted probabilities into a part within each drug
    and a part between drugs. If within_drug_var_share is near 0, predictions
    barely change across cells for a given drug -> the model is memorizing a
    per-drug constant and ignoring the cell embedding, which would make the
    cell-embedding comparison meaningless. Only informative when drugs are seen
    in training (so we skip it for the LODO splits)."""
    probs = np.asarray(probs).reshape(-1)
    drug_idx = np.asarray(drug_idx).reshape(-1)
    if probs.size == 0:
        return {"within_drug_var": np.nan, "within_drug_var_share": np.nan}
    means, vars_, ns = [], [], []
    for d in np.unique(drug_idx):
        p = probs[drug_idx == d]
        if p.size < 2:
            continue
        means.append(p.mean()); vars_.append(p.var()); ns.append(p.size)
    if not vars_:
        return {"within_drug_var": np.nan, "within_drug_var_share": np.nan}
    means, vars_, ns = np.array(means), np.array(vars_), np.array(ns)
    within = float(np.average(vars_, weights=ns))
    grand = float(np.average(means, weights=ns))
    between = float(np.average((means - grand) ** 2, weights=ns))
    total = within + between
    return {"within_drug_var": within,
            "within_drug_var_share": (within / total) if total > 0 else np.nan}


def plot_view(summary, view, title, out_png, zoom):
    sub = summary[summary["view"] == view].set_index("metric").loc[METRIC_NAMES]
    means = sub["overall_mean"].to_numpy()
    ses = sub["SE"].to_numpy()

    finite = np.isfinite(means) & np.isfinite(ses)
    if not finite.all():
        missing = [m for m, ok in zip(METRIC_NAMES, finite) if not ok]
        print(f"  [plot_view WARNING] {view}: no finite score for {missing} "
              f"(too few folds had both classes) -- drawing those bars as 0.")

    tops = np.where(finite, means + ses, 0.0)
    bots = np.where(finite, means - ses, 0.0)
    LABEL_HEADROOM = 0.20
    if zoom and finite.any():
        data_top, data_bot = tops[finite].max(), bots[finite].min()
        span = max(data_top - data_bot, 0.02)
        pad = max(0.15 * span, 0.02)
        ymin = max(0.0, data_bot - pad) if data_bot >= 0 else min(data_bot - pad, 0.0)
        ymax = ymin + (data_top - ymin) / (1.0 - LABEL_HEADROOM)
    else:
        ymin, ymax = 0.0, 1.2
    if ymax <= ymin:
        ymax = ymin + 0.1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(METRIC_NAMES, np.where(finite, means, 0.0),
           yerr=np.where(finite, ses, 0.0),
           color=["#4C72B0" if ok else "#B0B0B0" for ok in finite],
           edgecolor="#2f4a70", linewidth=0.8,
           error_kw={"elinewidth": 1.6, "capsize": 8, "capthick": 1.6, "ecolor": "black"})
    ax.set_ylim(ymin, ymax)
    if ymin <= 0.0 <= ymax:
        ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_ylabel("score")
    ax.set_title(title)
    offset = (ymax - ymin) * 0.015
    for x, (m, s, t, ok) in enumerate(zip(means, ses, tops, finite)):
        label = f"{m:.3f}\n\u00b1{s:.3f}" if ok else "no data\n(all NaN)"
        ax.text(x, t + offset, label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    atomic_write(out_png, lambda p: fig.savefig(p, dpi=150, format="png"))
    plt.close(fig)
    print(f"  wrote {out_png}")


# =====================================================================
# 8. IO Helpers
#    os.replace() is atomic within a filesystem, so a job failed mid-write
#    leaves the previous completed file in place, not causing a half-written one.
#    Re-running a stage overwrites its outputs.
# =====================================================================
def atomic_write(path, write_fn):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}{os.path.splitext(path)[1]}"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def write_parquet(df, path):
    atomic_write(path, lambda p: df.to_parquet(p, index=False))
    return path


def write_csv(df, path):
    atomic_write(path, lambda p: df.to_csv(p, index=False))
    return path


def shard_path(pred_dir, seed):
    return os.path.join(pred_dir, f"seed{seed}.parquet")


# =====================================================================
# 9. Stage "predict"  -->  train + inference only.
#
#    Writes one parquet per seed:  {pred_dir}/seed{S}.parquet
#        row_idx    position in the input table
#        barcode    the single cell
#        cell_line, drug
#        y_true     the label
#        prob       predicted probability for this cell x drug
#        fold       fold number, or the held-out group name for the LOO splits
#        seed
# =====================================================================
def run_predict(cfg, device):
    print("Loading row table ...")
    df = pd.read_parquet(cfg.input_table,
                         columns=[cfg.barcode_col, "cell_line", "drug", "sensitivity"])
    N = len(df)
    print(f"  {N:,} rows loaded")

    # ---- drug embedding (precomputed, keyed by drug name) ----
    # feature-mode 'cell_only' makes the drug block invisible to the model 
    if cfg.feature_mode == "cell_only":
        print("feature-mode=cell_only: skipping drug embedding load (drug block is zeroed)")
        drug_to_idx = {d: 0 for d in df["drug"].unique()}
        drug_matrix = np.zeros((1, 0), dtype=np.float32)
    else:
        print(f"Loading drug embedding ({cfg.drug_emb}) ...")
        drug_df = pd.read_parquet(cfg.drug_emb_file)
        drug_to_idx = {d: i for i, d in enumerate(drug_df.index)}
        drug_matrix = drug_df.values.astype(np.float32)
        assert df["drug"].isin(drug_to_idx).all(), "A drug has no embedding in the drug file!"
        print(f"  drug matrix: {drug_matrix.shape[0]:,} drugs x {drug_matrix.shape[1]:,} dims")

    # ---- cell embedding provider ----
    cell_embedder = make_cell_embedder(cfg)

    # ---- map each row's barcode -> its position in the cell-embedding order ----
    barcodes = df[cfg.barcode_col].astype(str).values
    barcode_to_cell = {b: i for i, b in enumerate(cell_embedder.barcodes)}
    cell_idx_np = pd.Series(barcodes).map(barcode_to_cell).values
    assert not pd.isnull(cell_idx_np).any(), \
        "Some row's barcode is missing from the cell embedding -- check barcode alignment."
    cell_idx_np = cell_idx_np.astype(np.int64)
    if cfg.feature_mode == "cell_only":
        drug_idx_np = np.zeros(N, dtype=np.int64)       # unused: drug block is zero-width
    else:
        drug_idx_np = df["drug"].map(drug_to_idx).values.astype(np.int64)

    y_np = df["sensitivity"].values.astype(int)
    cell_line = df["cell_line"].astype(str).values
    drug = df["drug"].astype(str).values

    fit_predict = MODEL_DISPATCH[cfg.model]
    os.makedirs(cfg.pred_dir, exist_ok=True)

    frozen_feats = None                              # cache for frozen cell embeddings
    shards = []

    for seed in cfg.seeds:
        print(f"\n===== SEED {seed} =====")
        set_seed(seed)
        parts = []

        for fold_label, tr, va in iter_splits(
                cfg.split, N, y_np, cell_line, drug, cell_idx_np, cfg.n_folds, seed):

            # ---- this fold's cell features (refit per split, or frozen+cached) ----
            if cfg.feature_mode == "drug_only":
                cell_feats = np.zeros((len(cell_embedder.barcodes), 0), dtype=np.float32)
            elif cell_embedder.per_split:
                if getattr(cell_embedder, "cacheable", False):
                    cell_embedder.context = (cfg.split, fold_label)
                train_cells = np.unique(cell_idx_np[tr])
                cell_feats = cell_embedder.features(train_cells, seed)
            else:
                if frozen_feats is None:
                    frozen_feats = cell_embedder.features(None, seed)
                cell_feats = frozen_feats

            # ---- train + predict held-out rows ----
            preds = np.asarray(
                fit_predict(cell_feats, drug_matrix, cell_idx_np, drug_idx_np,
                            y_np, tr, va, cfg, device, seed),
                dtype=np.float32).reshape(-1)
            assert preds.shape[0] == len(va), \
                f"model returned {preds.shape[0]} predictions for {len(va)} held-out rows"

            parts.append(pd.DataFrame({
                "row_idx": np.asarray(va, dtype=np.int64),
                "fold": str(fold_label),
                "prob": preds,
            }))
            print(f"    predicted {len(va):,} held-out rows  "
                  f"(mean prob {preds.mean():.3f} | positives {y_np[va].mean():.3f})")

        # ---- assemble and write this seed's shard ----
        part = pd.concat(parts, ignore_index=True)
        del parts
        ridx = part["row_idx"].to_numpy()
        out = pd.DataFrame({
            "row_idx": ridx,
            "barcode": pd.Categorical(barcodes[ridx]),
            "cell_line": pd.Categorical(cell_line[ridx]),
            "drug": pd.Categorical(drug[ridx]),
            "y_true": y_np[ridx].astype(np.int8),
            "prob": part["prob"].to_numpy(),
            "fold": pd.Categorical(part["fold"].to_numpy()),
            "seed": np.int32(seed),
        })
        path = write_parquet(out, shard_path(cfg.pred_dir, seed))
        size_mb = os.path.getsize(path) / 1e6
        print(f"  wrote {path}  ({len(out):,} rows, {size_mb:.1f} MB)")
        shards.append({"seed": int(seed), "file": os.path.basename(path),
                       "n_rows": int(len(out)), "size_mb": round(size_mb, 2)})
        del part, out

    if hasattr(cell_embedder, "report"):
        cell_embedder.report()

    expected = {os.path.basename(shard_path(cfg.pred_dir, s)) for s in cfg.seeds}
    extras = [fn for fn in sorted(os.listdir(cfg.pred_dir))
              if fn.startswith("seed") and fn.endswith(".parquet")
              and fn not in expected]
    if extras:
        print(f"  note: {len(extras)} other shard(s) present ({', '.join(extras)}); "
              f"left untouched. --stage evaluate uses only --seeds.")

    # ---- manifest: record what produced these predictions ----
    manifest = {
        "tag": cfg.tag,
        "written_utc": datetime.datetime.now(datetime.timezone.utc)
                               .isoformat(timespec="seconds"),
        "n_input_rows": int(N),
        "shards": shards,
        "config": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                   for k, v in sorted(vars(cfg).items())},
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__,
                     "pandas": pd.__version__,
                     "torch": torch.__version__},
    }
    atomic_write(os.path.join(cfg.out_dir, "run_manifest.json"),
                 lambda p: open(p, "w").write(json.dumps(manifest, indent=2, default=str)))
    print(f"\nPredictions complete -> {cfg.pred_dir}")


# =====================================================================
# 10. Stage "evaluate"  -->  scoring only, from the saved predictions.
# =====================================================================
def load_predictions(cfg):
    """Read exactly the shards named by --seeds (never whatever happens to be
    lying in the folder), so a stale file can't contaminate the numbers."""
    parts = []
    for s in cfg.seeds:
        p = shard_path(cfg.pred_dir, s)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"no predictions for seed {s} at {p}. "
                f"Run --stage predict first, or point --pred-dir at the right folder.")
        parts.append(pd.read_parquet(p))
    pred = pd.concat(parts, ignore_index=True)
    for c in ("cell_line", "drug", "fold"):        # concat of categoricals -> object
        pred[c] = pred[c].astype("category")
    print(f"  loaded {len(pred):,} prediction rows "
          f"({pred['drug'].nunique()} drugs x {pred['cell_line'].nunique()} cell lines, "
          f"{len(cfg.seeds)} seed(s))")
    return pred


def make_view(pred, view):
    """aggregate = one row per held-out cell x drug (not used for Tahoe, used for Zenodo dataset).
    pairlevel = one row per (cell_line, drug) within a fold+seed: mean predicted
    probability, and a majority-vote ground truth (mean(y_true) >= 0.5) (used only for Tahoe)."""
    if view == "aggregate":
        return pred
    g = (pred.groupby(["seed", "fold", "cell_line", "drug"],
                      observed=True, sort=False)
             .agg(pos_frac=("y_true", "mean"), prob=("prob", "mean"),
                  n_cells=("y_true", "size"))
             .reset_index())
    g["y_true"] = (g["pos_frac"] >= 0.5).astype(int)
    g["mixed"] = (g["pos_frac"] > 0) & (g["pos_frac"] < 1)
    return g


def metrics_by(frame, keys):
    rows = []
    for k, g in frame.groupby(keys, observed=True, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        m = compute_metrics(g["y_true"].to_numpy(), g["prob"].to_numpy())
        base = dict(zip(keys, k))
        for metric in METRIC_NAMES:
            rows.append({**base, "metric": metric, "value": m[metric],
                         "n_rows": int(len(g))})
    return pd.DataFrame(rows)


def _mean_se(values):
    """mean / sd / SE over the finite entries only: a unit that was single-class
    for a metric contributes NaN and is dropped for that metric alone."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) >= 2:
        sd = v.std(ddof=1)
        return v.mean(), sd, sd / np.sqrt(len(v)), len(v)
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan, 1
    return np.nan, np.nan, np.nan, 0


def summarize_over_folds(raw, extra_keys=()):
    """Average the seeds within each fold, then
    take mean +/- SE across folds. Returns (summary, fold_means)."""
    keys = ["view", *extra_keys, "metric"]
    fold_means = (raw.groupby([*keys, "fold"], as_index=False, observed=True)["value"]
                     .mean().rename(columns={"value": "fold_mean"}))
    rows = []
    for k, g in fold_means.groupby(keys, observed=True, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        mean, sd, se, n = _mean_se(g["fold_mean"])
        rows.append({**dict(zip(keys, k)), "overall_mean": mean, "sd": sd,
                     "SE": se, "n_units": n})
    return pd.DataFrame(rows), fold_means


def summarize_over_seeds(raw, extra_keys=()):
    """The metric is computed once over all folds' predictions
    within a seed, then mean +/- SE across seeds."""
    keys = ["view", *extra_keys, "metric"]
    rows = []
    for k, g in raw.groupby(keys, observed=True, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        mean, sd, se, n = _mean_se(g["value"])
        rows.append({**dict(zip(keys, k)), "overall_mean": mean, "sd": sd,
                     "SE": se, "n_units": n})
    return pd.DataFrame(rows)


def breakdown(pred_views, group_col):
    """Per-drug (or per-cell-line) scores, both views, both pooling conventions.

    per_fold : metric within each fold, then averaged across folds
    pooled   : metric over all folds' predictions at once, per seed"""
    raw_parts, sum_parts = [], []
    for view, frame in pred_views.items():
        r_fold = metrics_by(frame, ["seed", "fold", group_col]).assign(view=view)
        r_pool = metrics_by(frame, ["seed", group_col]).assign(view=view, fold="(pooled)")
        s_fold, _ = summarize_over_folds(r_fold, extra_keys=(group_col,))
        s_pool = summarize_over_seeds(r_pool, extra_keys=(group_col,))
        raw_parts += [r_fold.assign(pooling="per_fold"), r_pool.assign(pooling="pooled")]
        sum_parts += [s_fold.assign(pooling="per_fold"), s_pool.assign(pooling="pooled")]
    cols = ["view", "pooling", group_col, "metric", "fold", "seed", "value", "n_rows"]
    raw = pd.concat(raw_parts, ignore_index=True)[cols]
    summ = pd.concat(sum_parts, ignore_index=True)[
        ["view", "pooling", group_col, "metric", "overall_mean", "sd", "SE", "n_units"]]
    return raw.sort_values(cols[:4]), summ.sort_values(["view", "pooling", group_col, "metric"])


def run_evaluate(cfg):
    print(f"Loading predictions from {cfg.pred_dir} ...")
    pred = load_predictions(cfg)
    views = {v: make_view(pred, v) for v in VIEWS}

    # ---- overall: one score per view x metric x fold x seed ----
    raw = pd.concat([metrics_by(f, ["seed", "fold"]).assign(view=v)
                     for v, f in views.items()], ignore_index=True)
    raw = raw[["view", "metric", "fold", "seed", "value", "n_rows"]]
    summary, fold_means = summarize_over_folds(raw)
    summary = summary.rename(columns={"sd": "std_fold_means", "n_units": "n_folds_used"})

    # ---- per-drug and per-cell-line breakdowns ----
    drug_raw, drug_summary = breakdown(views, "drug")
    line_raw, line_summary = breakdown(views, "cell_line")

    # ---- memorization check (only where test drugs were seen in training) ----
    mem_records = []
    if cfg.split != "lodo_loo":
        agg = views["aggregate"]
        for (seed, fold), g in agg.groupby(["seed", "fold"], observed=True, sort=False):
            mstats = per_drug_variance_stats(g["prob"].to_numpy(),
                                             g["drug"].astype(str).to_numpy())
            mem_records.append({"fold": fold, "seed": seed, **mstats})

    # ---- save everything ----
    os.makedirs(cfg.out_dir, exist_ok=True)
    write_csv(raw, f"{cfg.out_dir}/metrics_per_fold_seed.csv")
    write_csv(fold_means, f"{cfg.out_dir}/metrics_fold_means.csv")
    write_csv(summary, f"{cfg.out_dir}/metrics_summary.csv")
    write_csv(drug_summary, f"{cfg.out_dir}/metrics_by_drug.csv")
    write_csv(drug_raw, f"{cfg.out_dir}/metrics_by_drug_raw.csv")
    write_csv(line_summary, f"{cfg.out_dir}/metrics_by_cell_line.csv")
    write_csv(line_raw, f"{cfg.out_dir}/metrics_by_cell_line_raw.csv")
    if mem_records:
        write_csv(pd.DataFrame(mem_records), f"{cfg.out_dir}/memorization_check.csv")

    # ---- pairlevel label-purity diagnostic (always 0% mixed on tahoe) ----
    plevel = views["pairlevel"]
    if len(plevel):
        write_csv(plevel[["seed", "fold", "cell_line", "drug", "pos_frac",
                          "n_cells", "mixed", "y_true", "prob"]],
                  f"{cfg.out_dir}/pairlevel_pair_purity.csv")
        n_mixed, n_total = int(plevel["mixed"].sum()), len(plevel)
        print(f"\npairlevel purity: {n_mixed}/{n_total} pair-fold-seed instances "
              f"label-mixed ({100*n_mixed/max(n_total,1):.1f}%) -- y_true for these "
              f"uses a majority vote (mean(y_true) >= 0.5), not an arbitrary cell.")

    try:
        plot_view(summary, "aggregate", f"{cfg.tag}  aggregate - mean +/- SE",
                  f"{cfg.out_dir}/metrics_aggregate.png", cfg.zoom)
        plot_view(summary, "pairlevel", f"{cfg.tag}  pair-level - mean +/- SE",
                  f"{cfg.out_dir}/metrics_pairlevel.png", cfg.zoom)
    except Exception as e:
        print(f"  [plot_view WARNING] plotting failed, CSVs already saved. Error: {e}")

    # ---- fold x seed tables ----
    for view in VIEWS:
        print(f"\n================ {view} view: fold x seed tables ================")
        sub = raw[raw["view"] == view]
        for metric in METRIC_NAMES:
            piv = sub[sub["metric"] == metric].pivot(index="fold", columns="seed",
                                                     values="value")
            piv.columns = [f"seed{c}" for c in piv.columns]
            piv["FoldMean"] = piv.mean(axis=1)
            print(f"\n{metric}:")
            print(piv.round(3).to_string())

    # ---- memorization warnings ----
    if mem_records:
        shares = pd.DataFrame(mem_records)["within_drug_var_share"]
        shares = shares[np.isfinite(shares)]
        if len(shares):
            flag = "  <-- WARNING: near-constant per drug (memorizing)" \
                if shares.mean() < 0.05 else ""
            print(f"\nmemorization: mean within-drug var share = "
                  f"{100*shares.mean():.1f}%{flag}")

    # ---- per-drug preview: the question this stage exists to answer ----
    prev = drug_summary[(drug_summary["view"] == "aggregate") &
                        (drug_summary["pooling"] == "pooled") &
                        (drug_summary["metric"] == "MCC")]
    if 0 < len(prev) <= 60:
        print("\n---- per-drug MCC (aggregate view, pooled across folds) ----")
        print(prev[["drug", "overall_mean", "SE", "n_units"]]
              .sort_values("drug").round(3).to_string(index=False))

    # ---- final summary ----
    print(f"\n===== SUMMARY  [{cfg.tag}]  (mean +/- SE across folds) =====")
    for view in VIEWS:
        print(f"\n{view}:")
        sub = summary[summary["view"] == view].set_index("metric").loc[METRIC_NAMES]
        for m in METRIC_NAMES:
            n_used = int(sub.loc[m, "n_folds_used"])
            print(f"  {m:12s}: {sub.loc[m, 'overall_mean']:.3f} +/- "
                  f"{sub.loc[m, 'SE']:.3f}   (n={n_used} folds)")
    print(f"\nAll files written to: {cfg.out_dir}")


# =====================================================================
# 11. Main
# =====================================================================
def main():
    cfg = resolve_cfg(build_parser().parse_args())
    device = torch.device(cfg.device) if cfg.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    try:                       # Tensor Cores: free speedup for scVI and the MLP
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    except Exception:
        pass

    print("=" * 70)
    print(f"RUN: {cfg.tag}   [stage={cfg.stage}]")
    print(f"  cell-emb={cfg.cell_emb} | drug-emb={cfg.drug_emb} | "
          f"model={cfg.model} | split={cfg.split} | n_folds={cfg.n_folds}")
    print(f"  device={device} | seeds={cfg.seeds} | out={cfg.out_dir}")
    print("=" * 70)

    if cfg.stage in ("all", "predict"):
        run_predict(cfg, device)
    if cfg.stage in ("all", "evaluate"):
        # note: `all` still scores by reading the parquet back, so the numbers
        # from one combined run and from two separate runs are identical.
        print("\n" + "=" * 70)
        print("EVALUATE")
        print("=" * 70)
        run_evaluate(cfg)


if __name__ == "__main__":
    main()