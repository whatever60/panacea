import os

import numpy as np
from scipy import sparse as ss
from scipy.io import mmwrite
import pandas as pd
import anndata
import scanpy as sc
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

from process import setup


def index_of_y_in_x(x, y):
    # y must be a subset of x
    # assert np.isin(y, x).all()
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    return xsorted[ypos.clip(max=len(xsorted) - 1)]  # length same as y


def merge_datasets(datasets):
    genes = np.unique(np.concatenate([d.var.index.tolist() for d in datasets]))
    cell_ids = np.concatenate([d.obs.index.tolist() for d in datasets])
    sp_mats = []
    gene_lists = []
    for d in datasets:
        # expand the expression matrix to the unified gene list, and fill new genes with 0
        # create an empty sparse matrix
        idx = index_of_y_in_x(genes, d.var.index)
        mat = ss.csr_matrix(
            (d.X.data, idx[d.X.indices], d.X.indptr), shape=(d.X.shape[0], len(genes))
        )
        sp_mats.append(mat)
        gene_lists.append(np.sort(idx))
    return ss.vstack(sp_mats), cell_ids, genes, gene_lists


def add_label(X, data_idx, genes, label_obs, data, gene_list_ids=None) -> None:
    idx = index_of_y_in_x(data_idx, label_obs.index)
    idx_mask = data_idx[idx] == label_obs.index
    assert idx_mask.sum() <= X.shape[0]
    # get rid of extra cells in label_obs.
    obs_cp = label_obs.copy()[idx_mask].astype(str)
    idx = idx[idx_mask]
    X = X[idx]  # permute X so that its order follows obs_cp.
    if gene_list_ids is not None:
        assert len(gene_list_ids) == len(data_idx)
        obs_cp["gene_list_id"] = gene_list_ids[idx]
    adata = anndata.AnnData(X, obs=obs_cp, var=pd.DataFrame({}, index=genes), dtype=int)
    setup(out_dir, data)
    adata.write_h5ad(f"{out_dir}/{data}/adata/adata.h5ad")


if __name__ == "__main__":
    import pytorch_lightning as pl
    from rich import print as rprint
    from rich.traceback import install

    install()
    pl.seed_everything(42)

    out_dir = "/home/tiankang/wusuowei/data/single_cell/panacea"

    label_data_path = "/home/tiankang/SCALEX/results/transfer/mouse_atlas/adata.h5ad"
    label_obs = anndata.read_h5ad(label_data_path).obs

    # these data will be used for pretraining
    tabulas_muris_aging_droplet_path = (
        "/data/xionglei/data/mouse/tabula_muris_aging_droplet.h5ad"
    )
    tabulas_muris_aging_facs_path = (
        "/data/xionglei/data/mouse/tabula_muris_aging_facs.h5ad"
    )
    mouse_atlas_dir = "/data/tiankang/SCALEX/mouse_atlas/adata.h5ad"

    # these data will be used for finetuning
    mouse_kidney_gr_path = "/data/xionglei/data/mouse/mouse_kidney_GR.h5ad"
    mouse_lung_gr_path = "/data/xionglei/data/mouse/mouse_lung_GR.h5ad"

    # ====================== pretrain ======================
    dataset_paths = [
        tabulas_muris_aging_droplet_path,
        tabulas_muris_aging_facs_path,
        mouse_atlas_dir,
    ]
    datasets = [anndata.read_h5ad(p) for p in dataset_paths]

    X, cell_ids, genes, gene_lists = merge_datasets(datasets)

    gene_list_ids = np.concatenate(
        [np.full(len(d.obs), i) for i, d in enumerate(datasets)]
    )
    add_label(X, cell_ids, genes, label_obs, "mouse_atlas", gene_list_ids)
    # save gene_lists
    for i, g in enumerate(gene_lists):
        np.savetxt(f"{out_dir}/mouse_atlas/gene_list/{i}.txt", g, fmt="%d")

    # ====================== mouse kidney ======================
    mouse_kidney_data = anndata.read_h5ad(mouse_kidney_gr_path)
    X = mouse_kidney_data.X
    data_idx = mouse_kidney_data.obs.index
    genes = mouse_kidney_data.var.index
    add_label(X, data_idx, genes, label_obs, "mouse_kidney_gr")

    # ====================== mouse lung ======================
    mouse_lung_data = anndata.read_h5ad(mouse_lung_gr_path)
    X = mouse_lung_data.X
    data_idx = mouse_lung_data.obs.index
    genes = mouse_lung_data.var.index
    add_label(X, data_idx, genes, label_obs, "mouse_lung_gr")
