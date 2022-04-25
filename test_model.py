import json

import torch
import pytorch_lightning as pl
import yaml
from rich import print as rprint
from rich.traceback import install

from train import Panacea


install()

with open("config_data.yaml") as f:
    config_data = yaml.safe_load(f)
with open("config_model.yaml") as f:
    config_model = yaml.safe_load(f)


DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/panacea"
config_model["gpus"] = []
config = {**config_model, **config_data}
model = Panacea(**config)

trm_model = model.student.backbone

# ===================
# test data
with open("test_data.json") as f:
    test_data = json.load(f)
    gene = torch.tensor(test_data["gene"])
    count = torch.tensor(test_data["count"])
    mask_gene = torch.tensor(test_data["mask_gene"])
    mask_count = torch.tensor(test_data["mask_count"])

# [batch_size, 1 + length, emb_dim]
rprint(trm_model(gene, count, mask_gene, mask_count).shape)
