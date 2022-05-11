import json

import torch
import pytorch_lightning as pl
import yaml
from rich import print as rprint
from rich.traceback import install

from train import Panacea


install()
pl.seed_everything(42)

DATA_DIR = "/data/tiankang/wusuowei/data/single_cell/panacea"
DATA = "pancreas"

with open(f"configs/{DATA}.yaml") as f:
    config_dataset = yaml.safe_load(f)
with open("config_data.yaml") as f:
    config_data = yaml.safe_load(f)
with open("config_model.yaml") as f:
    config_model = yaml.safe_load(f)

config_data["data"] = "DATA"
config_model["gpus"] = []
config = {**config_model, **config_data, **config_dataset}
model = Panacea(**config)

trm_model = model.student.backbone

# ===================
# test data
with open("test_data.json") as f:
    test_data = json.load(f)
    gene = torch.tensor(test_data["gene"]).unsqueeze(0)
    count = torch.tensor(test_data["count"]).unsqueeze(0)
    mask_gene = torch.tensor(test_data["mask_gene"]).unsqueeze(0)
    mask_count = torch.tensor(test_data["mask_count"]).unsqueeze(0)

# [batch_size, 1 + length, emb_dim]
rprint(trm_model(gene, count, mask_gene, mask_count).shape)
