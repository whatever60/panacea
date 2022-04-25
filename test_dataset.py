import json

import pytorch_lightning as pl
import yaml
from rich import print as rprint
from rich.traceback import install

from datamodules import PanaceaDataModule


install()
pl.seed_everything(42)

with open("config_data.yaml") as f:
    config_data = yaml.safe_load(f)
with open("config_model.yaml") as f:
    config_model = yaml.safe_load(f)


DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/panacea"
config_model["gpus"] = []
datamodule = PanaceaDataModule(data_dir=DATA_DIR, **config_data)
datamodule.setup("fit")
dataloader = datamodule.train_dataloader()

for batch in iter(dataloader):
    if "D28.3_41" in batch["cell_name"]:
        rprint(batch["cell_name"])
        break

test_data = dict(
    gene=batch["gene"][0][5:7].tolist(),
    count=batch["count"][0][5:7].tolist(),
    mask_gene=(batch["mask"][0][5:7] == 1).tolist(),
    mask_count=(batch["mask"][0][5:7] == 2).tolist(),
)


with open("test_data.json", "w") as f:
    json.dump(test_data, f)
