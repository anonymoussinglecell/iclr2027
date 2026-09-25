# ICLR 2027 - "Evaluating Cell and Drug Representations for Generalizable Drug Response Predictions"

Code and data for the anonymous ICLR 2027 submission. The released tables are enough to regenerate the figures in the paper, and `main.py` can retrain any single configuration from the released inputs.

## Names used in the code

The code names each dataset after its source, not after the term used in the paper:
* `tahoe` : shared-label setting
* `zenodo` : single-cell label setting 

`zenodo` refers only to where the original single-cell label dataset is distributed. It is unrelated to the Zenodo record below, which holds this repository's data files.

Splits, embeddings and models also keep short names in files and folders:

| Code | Paper |
|---|---|
| `random` | Random split |
| `pairzs_withoutleak` (shared-label setting), `pairzs_withleak` (single-cell label setting) | Pair zero shot |
| `loclo_loo` | Cell line zero shot |
| `lodo_loo` | Drug zero shot |
| `gene_jepa`, `gformer_cancer`, `scgpt_pan`, `pca`, `scvi` | GeneJepa, Geneformer, scGPT, PCA, scVI |
| `morgan`, `maccs`, `chemberta`, `molformer`, `biomed`, `materials` | Morgan, MACCS, ChemBERTa, MoLFormer, Biomed, SMI-TED |
| `mlp`, `lightgbm`, `xrfm` | MLP, LightGBM, xRFM |

Two settings differ between the datasets, as described in the paper:
- **Pair zero shot.** Shared-label setting uses the leakage-free split. Single-cell label setting uses the split that allows single-cell leakage, which is negligible in this dataset.
- **Scoring view.** Shared-label setting is scored per cell line–drug pair (`pairlevel`). Single-cell label setting has only 17 such pairs, so it is scored per cell (`aggregate`).

## Repository contents

```
main.py                  training, inference and scoring for one configuration
regen_figures/
    common.py            paths, display names, per-dataset settings, scoring helpers
    figures.py           regenerates the paper figures
ablation_analyses/       scores behind the ablation figure, one pair of files per dataset
    {tahoe,zenodo}_ablation_scores_long.csv       full model and single-modality ablations, every configuration
    {tahoe,zenodo}_baselines_pairlevel_long.csv   prevalence baselines
```

The single-cell label settings' baselines file keeps the `_pairlevel` name, but like all scores of this dataset setting its values are computed per cell.

## Data

The larger files are in the Zenodo record 10.5281/zenodo.22962632.

**`data_input/`**: inputs to `main.py`
- `tahoe_train_pairs_bal15.parquet`: Tahoe shared-label setting labels, one row per cell and drug. 
- `tahoe_controls2000_merged_hvg5000_log1p_umap.h5ad`: Tahoe shared-label setting expression, 5,000 highly variable genes, with raw-count and log1p layers.
- `zenodo_rows.parquet`: Multiple single-cell label settings' labels.
- `zenodo_cells_log1p.h5ad`: Multiple single-cell label settings' expression, log1p(CPM).
- Frozen cell embeddings (GeneJepa, Geneformer, scGPT) and all six drug embeddings for both datasets, as parquet files. PCA and scVI have no file, because `main.py` refits them on each fold's training cells.

**`results/`**: metrics for every configuration, one folder per run named `<split>__<cell_emb>__<drug_emb>__<model>`
- 360 folders per dataset: 4 splits × 5 cell embeddings × 6 drug embeddings × 3 models. Each run used seeds 0, 1 and 42.
- Each folder holds `metrics_by_drug_raw.csv` and `metrics_by_cell_line_raw.csv`. Columns: `view`, `pooling`, drug or `cell_line`, `metric` (MCC, AUROC, AUPRC, BalancedAcc), `fold`, `seed`, `value`, `n_rows`.
- A configuration's score in the paper uses the rows with `pooling = pooled` and the dataset's view. The value is averaged over drugs (or cell lines) within each seed, then over seeds.
- `avg_min_max_mcc_by_split.csv`, one per dataset: the mean, minimum and maximum MCC over all configurations in each split.

## Setup

Python 3.11 with the versions used for the paper:

```
pip install numpy==2.2.6 pandas==2.3.3 pyarrow==24.0.0 scipy==1.17.1 matplotlib==3.11.0 \
    anndata==0.12.18 scanpy==1.11.5 umap-learn==0.5.12 \
    scikit-learn==1.8.0 lightgbm==4.6.0 torch==2.11.0 scvi-tools==1.4.2 xrfm==0.4.4
```

Regenerating the figures runs on a CPU. Retraining xRFM and scVI configurations works best on a GPU.

Download the Zenodo record, then set the data paths at the top of `regen_figures/common.py`.

## Retraining one configuration

Each call trains and scores one configuration for seeds 0, 1 and 42. Pass every path explicitly, because `main.py`'s built-in defaults assume a different folder layout. Write to a new output folder so the released results are not overwritten.

Tahoe:
```
python main.py --project-root . \
    --input-table data_input/tahoe_train_pairs_bal15.parquet \
    --h5ad data_input/tahoe_controls2000_merged_hvg5000_log1p_umap.h5ad \
    --cell-emb-file data_input/<cell embedding>.parquet \
    --drug-emb-file data_input/<drug embedding>.parquet \
    --out-root results_rerun/tahoe \
    --split lodo_loo --cell-emb gene_jepa --drug-emb morgan --model xrfm \
    --xrfm-iters 2 --xrfm-max-leaf-size 5000
```

Zenodo:
```
python main.py --project-root . \
    --input-table data_input/zenodo_rows.parquet \
    --h5ad data_input/zenodo_cells_log1p.h5ad \
    --cell-emb-file data_input/<cell embedding>.parquet \
    --drug-emb-file data_input/<drug embedding>.parquet \
    --out-root results_rerun/zenodo \
    --split pairzs_withleak --cell-emb scvi --drug-emb morgan --model lightgbm \
    --counts-layer log1p --scvi-input lognorm --scvi-batch-size 512
```

These settings differ from `main.py`'s defaults:

| When | Shared-label setting | Single-cell label setting |
|---|---|---|
| `--cell-emb scvi` | `--scvi-batch-size 512` | `--scvi-batch-size 512 --counts-layer log1p --scvi-input lognorm` (Single-cell label setting datasets have no raw counts) |
| `--model xrfm` | `--xrfm-iters 2 --xrfm-max-leaf-size 5000` | defaults (5 iterations, leaf size 20,000) |

All other settings are `main.py`'s defaults, including the number of folds per split (listed in the header of `main.py`).

- Results go to `<out-root>/<split>__<cell_emb>__<drug_emb>__<model>/`. Tahoe runs add one subfolder named after the input table.
- `--feature-mode cell_only` or `drug_only` runs the single-modality ablations. The unused embedding argument only names the output folder. The paper used `morgan` for cell-only runs and `gene_jepa` for drug-only runs.
- xRFM sizes its leaves and AGOP subsample from the free GPU memory, so its scores can differ slightly on other GPUs.
