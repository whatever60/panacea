import argparse
import os

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from joblib import Parallel, delayed
import anndata
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tqdm.auto import tqdm

# python3 process.py --data pancreas --adata /data/tiankang/SCALEX/pancreas/adata.h5ad
# python3 process.py --data mouse_atlas --adata /home/tiankang/wusuowei/data/single_cell/panacea/mouse_atlas/adata/adata.h5ad
# python3 process.py --data mouse_lung_gr --adata /home/tiankang/wusuowei/data/single_cell/panacea/mouse_lung_gr/adata/adata.h5ad
# python3 process.py --data mouse_kidney_gr --adata /home/tiankang/wusuowei/data/single_cell/panacea/mouse_kidney_gr/adata/adata.h5ad
# python3 process.py --data mouse_brain_spatial --adata /data/tiankang/SCALEX/mouse_brain_spatial/adata.h5ad
# python3 process.py --data human_atlas_adult --adata /data/tiankang/SCALEX/human_atlas_adult/adata.h5ad


def setup(out_dir: str, data: str) -> None:
    [
        os.makedirs(f"{out_dir}/{p}", exist_ok=True)
        for p in [
            data,
            f"{data}/adata",
            f"{data}/data",
            f"{data}/gene_list",
            f"{data}/tokenizer",
        ]
    ]


def process_adata(adata, out_dir, data) -> None:
    # save adata as genes.txt, metadata.txt, matrix.mtxs
    obs = adata.obs.astype(str).fillna("nan").reset_index()
    obs.columns = ["cell_name"] + obs.columns[1:].tolist()
    adata = anndata.AnnData(adata.X, obs, adata.var, dtype=int)
    if not os.path.isfile(f"{out_dir}/{data}/adata/adata.h5ad"):
        adata.write_h5ad(f"{out_dir}/{data}/adata/adata.h5ad")
    # if not os.path.isfile(f"{out_dir}/{data}/adata/matrix.mtx"):
    #     mmwrite(f"{out_dir}/{data}/adata/matrix.mtx", adata.X)
    adata.obs.to_csv(f"{out_dir}/{data}/adata/metadata.txt", sep="\t")
    np.savetxt(
        f"{out_dir}/{data}/adata/genes.txt", adata.var.index.to_numpy(), fmt="%s"
    )


def train_tokenizer(gene_path, tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pad_token = tokenizer.token_to_id("[PAD]")
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train(gene_path, trainer)
    tokenizer.save(tokenizer_path)


def read_data(gene_path, cell_path, count_path):
    genes = pd.read_csv(gene_path, sep="\t", header=None).squeeze("columns")
    cell_meta = pd.read_csv(cell_path, sep="\t", dtype=object)
    # the first header of meta data is empty, so we change it to "cell_name"
    cell_meta.columns = ["cell_name"] + cell_meta.columns[1:].tolist()

    # load count_path with pandas and skip the first 3 rows, and set columns to ["cell", "gene", "count"]
    count_df = pd.read_csv(
        count_path,
        skiprows=3,
        sep=" ",
        header=None,
        names=["cell_idx", "gene", "count"],
    )
    count_df.cell_idx -= 1
    count_df.gene -= 1
    return genes, cell_meta, count_df


if __name__ == "__main__":
    import pytorch_lightning as pl
    from rich import print as rprint
    from rich.traceback import install

    install()
    tqdm.pandas()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--adata", type=str, default="")
    parser.add_argument("--val_type", action="store_true")
    parser.add_argument("--val_batch", action="store_true")
    args = parser.parse_args()
    data = args.data
    adata = args.adata
    pl.seed_everything(42)

    data_dir = "/home/tiankang/wusuowei/data/single_cell/panacea"
    adata_path = f"{data_dir}/{data}/adata/adata.h5ad"
    gene_path = f"{data_dir}/{data}/adata/genes.txt"
    cell_path = f"{data_dir}/{data}/adata/metadata.txt"
    count_path = f"{data_dir}/{data}/adata/matrix.mtx"
    tokenizer_path = f"{data_dir}/{data}/tokenizer/tokenizer.json"

    if adata:
        # if adata is given, process it first.
        setup(data_dir, data)
        process_adata(anndata.read_h5ad(adata), data_dir, data)
        rprint(f"Processing adata done")

    # Till now, everything is still (essentially) raw data.

    def get_cell_data(group) -> None:
        """For each cell, create an entry with all attributes in meta data, index in count file, the name of genes, and the count of each gene"""
        cell_idx = group.name
        # there should be only one corresponding row in meta data
        cell_data = cell_meta.iloc[cell_idx]
        cell_data = cell_data.to_dict()
        cell_data["cell_idx"] = cell_idx
        cell_data["gene"] = " ".join(genes[group.gene])
        cell_data["count"] = " ".join(group["count"].astype(str))
        # cell_data["gene"] = genes[group.gene].tolist()
        # cell_data["count"] = group["count"].tolist()
        json_list.append(cell_data)

    def get_cell_data_adata(cell_idx) -> None:
        cell_data = adata.obs.iloc[cell_idx].to_dict()
        cell_data["cell_idx"] = cell_idx
        count = adata.X[cell_idx]
        non_zero_pos = count.nonzero()
        cell_data["gene"] = " ".join(adata.var.index[non_zero_pos[1]])
        cell_data["count"] = " ".join(count[non_zero_pos].A[0].astype(str))
        return cell_data

    # ========= read data (from text files) =========
    # json_list = []
    # genes, cell_meta, count_df = read_data(gene_path, cell_path, count_path)
    # rprint(f"Max count of dataste {data}:", count_df["count"].max())
    # # construct pandas dataset
    # # groupby cells
    # count_df.groupby("cell_idx").progress_apply(get_cell_data)
    # json_df = pd.DataFrame(json_list).set_index("cell_idx")

    # ========== read data (from h5ad) ==========
    adata = anndata.read_h5ad(adata_path)
    rprint(f"Max count of dataste {data}:", adata.X.max())
    # adata.obs = adata.obs.reset_index()
    # adata.obs.columns = ["cell_name"] + adata.obs.columns[1:].tolist()
    json_list = Parallel(n_jobs=1)(
        delayed(get_cell_data_adata)(cell_idx) for cell_idx in tqdm(range(adata.n_obs))
    )
    json_df = pd.DataFrame(json_list).set_index("cell_idx")

    # ========= special manipulation =========
    if data == "mouse_brain_spatial":
        json_df = json_df.query("batch == 'MERFISH'")

    # ========= QC =========
    # remove those cells that have less than min_genes genes
    num_genes = json_df["gene"].str.count(" ") + 1
    # thres = num_genes.quantile(0.05)
    thres = 32
    rprint("QC threshold:", thres)
    rprint("Before quality control:", json_df.shape)
    json_df = json_df[num_genes >= thres]
    rprint("After quality control:", json_df.shape)

    # ========= train-val split =========
    # leave out a batch
    batches = json_df.batch.value_counts().sort_values()
    if len(batches) > 3 and args.val_batch:
        leave_out_batch = batches.index[len(batches) // 2]
        val_batch_df = json_df.query(f"batch == '{leave_out_batch}'")
    else:
        val_batch_df = None
    # leave out a type
    types = json_df.celltype.value_counts().sort_values()
    if len(types) > 3 and args.val_type:
        leave_out_type = types.index[len(types) // 2]
        val_type_df = json_df.query(f"celltype == '{leave_out_type}'")
    else:
        val_type_df = None
    # get the rest (train and val none)
    mask_batch = (
        json_df.batch != leave_out_batch
        if val_batch_df is not None
        else np.ones(len(json_df), dtype=bool)
    )
    mask_type = (
        json_df.celltype != leave_out_type
        if val_type_df is not None
        else np.ones(len(json_df), dtype=bool)
    )
    rest_df = json_df[mask_batch & mask_type]
    train_idx = np.random.binomial(1, 0.8, len(rest_df)).astype(bool)
    train_df = rest_df.loc[train_idx]
    val_none_df = rest_df.loc[~train_idx]

    # save to json
    dataset_dir = f"{data_dir}/{data}/data"
    train_df.to_json(
        f"{dataset_dir}/{data}_train.json.zip", orient="table", indent=2, index=False
    )
    val_none_df.to_json(
        f"{dataset_dir}/{data}_val_none.json.zip", orient="table", indent=2, index=False
    )
    if val_batch_df is not None:
        val_type_df.to_json(
            f"{dataset_dir}/{data}_val_type.json.zip",
            orient="table",
            indent=2,
            index=False,
        )
    if val_type_df is not None:
        val_batch_df.to_json(
            f"{dataset_dir}/{data}_val_batch.json.zip",
            orient="table",
            indent=2,
            index=False,
        )

    # test dataset
    data_files = dict(
        train=f"{dataset_dir}/{data}_train.json.zip",
        val_none=f"{dataset_dir}/{data}_val_none.json.zip",
    )
    if os.path.exists(f"{dataset_dir}/{data}_val_type.json.zip"):
        data_files["val_type"] = f"{dataset_dir}/{data}_val_type.json.zip"
    if os.path.exists(f"{dataset_dir}/{data}_val_batch.json.zip"):
        data_files["val_batch"] = f"{dataset_dir}/{data}_val_batch.json.zip"

    dataset = load_dataset("json", data_files=data_files, field="data")
    rprint(dataset)

    train_tokenizer([gene_path], tokenizer_path)
