import numpy as np
import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
import pytorch_lightning as pl
from rich import print as rprint
from rich.traceback import install


install()
pl.seed_everything(42)

data_dir = "/data/tiankang/SCALEX/pancreas"
gene_file = "genes.txt"
cell_file = "metadata.txt"
count_file = "matrix.mtx"
out_dir = "/home/tiankang/wusuowei/data/single_cell/panacea"


json_list = []

genes = pd.read_csv(f"{data_dir}/{gene_file}", sep="\t", header=None).squeeze("columns")

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pad_token = tokenizer.token_to_id("[PAD]")
trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.pre_tokenizer = WhitespaceSplit()
files = [f"{data_dir}/genes.txt"]
tokenizer.train(files, trainer)
tokenizer.save(f"{out_dir}/tokenizer.json")

cell_meta = pd.read_csv(f"{data_dir}/{cell_file}", sep="\t")
# the first header of meta data is empty, so we change it to "cell_name"
cell_meta.columns = ["cell_name"] + cell_meta.columns[1:].tolist()

# load count_file with pandas and skip the first 3 rows, and set columns to ["cell", "gene", "count"]
count_df = pd.read_csv(
    f"{data_dir}/{count_file}",
    skiprows=3,
    sep=" ",
    header=None,
    names=["cell_idx", "gene", "count"],
)
count_df.cell_idx -= 1
count_df.gene -= 1
rprint(count_df["count"].max())


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


# groupby cells
count_df.groupby("cell_idx").apply(get_cell_data)

# json
json_df = pd.DataFrame(json_list).set_index("cell_idx")

# train-val split
# leave out a batch
batches = json_df.batch.value_counts().sort_values()
leave_out_batch = batches.index[len(batches) // 2]
val_batch_df = json_df.query(f"batch == '{leave_out_batch}'")
# leave out a type
types = json_df.celltype.value_counts().sort_values()
leave_out_type = types.index[len(types) // 2]
val_type_df = json_df.query(f"celltype == '{leave_out_type}'")
# get the rest
rest_df = json_df.query(
    f"batch != '{leave_out_batch}' and celltype != '{leave_out_type}'"
)
# split train_df into train and val (none)
# bernulli
train_idx = np.random.binomial(1, 0.8, len(rest_df)).astype(bool)
train_df = rest_df.loc[train_idx]
val_none_df = rest_df.loc[~train_idx]

# save to json
train_df.to_json(
    f"{out_dir}/pancreas_train.json.zip", orient="table", indent=4, index=False
)
val_none_df.to_json(
    f"{out_dir}/pancreas_val_none.json.zip", orient="table", indent=4, index=False
)
val_type_df.to_json(
    f"{out_dir}/pancreas_val_type.json.zip", orient="table", indent=4, index=False
)
val_batch_df.to_json(
    f"{out_dir}/pancreas_val_batch.json.zip", orient="table", indent=4, index=False
)


data_files = dict(
    train=f"{out_dir}/pancreas_train.json.zip",
    val_none=f"{out_dir}/pancreas_val_none.json.zip",
    val_type=f"{out_dir}/pancreas_val_type.json.zip",
    val_batch=f"{out_dir}/pancreas_val_batch.json.zip",
)
# test dataset
dataset = load_dataset("json", data_files=data_files, field="data")
rprint(dataset)
