# iclr2027
Code for the anonymous ICLR 2027 submission "Evaluating Cell and Drug Representations for Generalizable Drug Response Predictions."

The paper uses two datasets. In the code they are named after where they come from, not after the terms used in the paper: 
* "Tahoe" : Tahoe 100M dataset (shared-label setting)
* "Zenodo" : Zenodo record (single-cell label setting). The single-cell label dataset is called zenodo only because it is distributed through a Zenodo record.

main.py                 training, inference and scoring for one configuration
regen_figures/
    common.py           paths, display names, dataset settings, scoring helpers
    figures.py          regenerates the figures
ablation_analyses/      ablation and prevalence-baseline scores, one pair of files per dataset
    {tahoe,zenodo}_ablation_scores_long.csv
    {tahoe,zenodo}_baselines_pairlevel_long.csv

**Data**

The larger files are in this Zenodo record: 10.5281/zenodo.22945646

data_input/: the input tables and single-cell matrices
tahoe_train_pairs_bal15.parquet, tahoe_controls2000_merged_hvg5000_log1p_umap.h5ad
zenodo_rows.parquet, zenodo_cells_log1p.h5ad
the frozen cell embeddings (GeneJepa, Geneformer, scGPT) and all drug embeddings for both datasets, as parquet files
results/: the metrics tables for every configuration (cell embedding x drug embedding x model x split), one folder per run, plus each dataset's avg_min_max_mcc_by_split.csv

