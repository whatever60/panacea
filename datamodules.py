import os
from typing import List, Literal, Tuple, Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


class SingleCellDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data: str,
        split: str,
        classes: list,
        batches: list,
        train_classes: list,
        # ==== sample genes ====
        num_crops_g: int,
        min_length_g: int,
        max_length_g: int,
        mean_length_g: int,
        noise_ratio_g: float,
        dropout_p_g: float,
        # =========
        num_crops_l: int,
        min_length_l: int,
        max_length_l: int,
        mean_length_l: int,
        noise_ratio_l: float,
        dropout_p_l: float,
        # ==========
        length_dist: Literal["uniform", "sample"],
        # max_count: int,
        log_count: bool,
        count_temp: bool,
        zero_prob: float,
        # ==========
        mask_prop_gene: float,
        mask_prop_count: float,
        num_special_tokens: int = 5,
        vocab_size: int = 14900,
        **kwargs,  # not used, just for convinience.
    ):
        super().__init__()
        dataset = load_dataset(
            "json",
            data_files={split: f"{data_dir}/{data}/data/{data}_{split}.json.zip"},
            field="data",
            split=split,
        )

        self.split = split
        self.classes = classes
        self.batches = batches
        self.train_classes = train_classes

        self.num_crops_g = num_crops_g
        self.min_length_g = min_length_g
        self.max_length_g = max_length_g
        self.mean_length_g = mean_length_g
        self.noise_ratio_g = noise_ratio_g
        self.dropout_p_g = dropout_p_g

        self.num_crops_l = num_crops_l
        self.min_length_l = min_length_l
        self.max_length_l = max_length_l
        self.mean_length_l = mean_length_l
        self.noise_ratio_l = noise_ratio_l
        self.dropout_p_l = dropout_p_l

        self.length_dist = length_dist
        # self.max_count = max_count
        self.log_count = log_count
        self.count_temp = count_temp
        self.zero_prob = zero_prob

        self.mask_prop_gene = mask_prop_gene
        self.mask_prop_count = mask_prop_count
        self.num_special_tokens = num_special_tokens
        self.vocab_size = vocab_size
        self.num_genes = vocab_size - num_special_tokens

        self.dropout_p_g_l = [0] + [self.dropout_p_g] * (self.num_crops_g - 1)
        self.dropout_p_l_l = [self.dropout_p_l] * self.num_crops_l

        # self.tokenizer = Tokenizer.from_file(f"{data_dir}/tokenizer.json")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{data_dir}/{data}/tokenizer/tokenizer.json"
        )
        self.tokenizer.add_special_tokens(
            {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "unk_token": "[UNK]",
            }
        )
        if os.listdir(f"{data_dir}/{data}/gene_list"):
            self.gene_lists = [
                np.loadtxt(
                    f"{data_dir}/{data}/gene_list/{i}", dtype=int
                )
                    for i in os.listdir(f"{data_dir}/{data}/gene_list")
            ]
            self.exclude_genes = [
                np.ones(self.num_genes, dtype=bool) for _ in self.gene_lists
            ]
            for gene_list, exclude_gene in zip(self.gene_lists, self.exclude_genes):
                exclude_gene[gene_list] = False
        else:
            self.gene_lists = None
            self.exclude_genes = None
        self.dataset = dataset.map(self.tokenize_function, batched=True)

    def __len__(self):
        return len(self.dataset)

    # def pad_trunc(self, input_ids, counts, attention_mask):
    #     if len(input_ids) < self.max_length:  # pad
    #         input_ids = input_ids + [self.tokenizer.pad_token_id] * (
    #             self.max_length - len(input_ids)
    #         )
    #         attention_mask = attention_mask + [0] * (
    #             self.max_length - len(attention_mask)
    #         )
    #         counts = counts + [-1] * (self.max_length - len(counts))
    #     else:  # truncate
    #         return self._sample_random(input_ids, counts, attention_mask)

    def __getitem__(self, index):
        """raw data -> sample -> generating mask -> generating target -> padding"""
        sample = self.dataset[index]
        count = np.array(sample["count"].split(" ")).astype(int)
        gene = np.array(sample["input_ids"])
        gene_list_idx = sample.get("gene_list_idx")
        # set it to None, and extra genes would be considered just like zero expressed genes.
        gene_list_idx = None

        # 加noise和sampling谁先谁后会有很大区别吗
        # noise is only added to non-zero counts.
        count_g = self.add_noise(count, "global")
        count_l = self.add_noise(count, "local")

        gene_g, count_g = zip(
            *(self._sample_random(gene, count, gene_list_idx) for count in count_g)
        )
        if self.num_crops_l:
            gene_l, count_l = zip(
                *(
                    self._sample_random(
                        gene, count, is_global=False, gene_list_idx=gene_list_idx
                    )
                    for count in count_l
                )
            )
        else:
            gene_l = []
            count_l = []

        mask_g = [self.get_mask(len(gene)) for gene in gene_g]
        mask_l = [self.get_mask(len(gene)) for gene in gene_l]

        gene_g, count_g, mask_g = self.pad(gene_g, count_g, mask_g, self.max_length_g)
        gene_l, count_l, mask_l = self.pad(gene_l, count_l, mask_l, self.max_length_l)

        genes = gene_g + gene_l
        counts = count_g + count_l
        masks = mask_g + mask_l

        del sample["input_ids"]
        del sample["attention_mask"]
        del sample["token_type_ids"]

        sample["gene"], sample["count"], sample["mask"] = genes, counts, masks
        count_raw = self.to_raw(gene, count)
        if gene_list_idx is not None:
            count_raw[self.exclude_genes[gene_list_idx]] = -999
        sample["count_raw"] = count_raw
        # if sample["celltype"] is None:
        #     sample["celltype"] = "nan"
        sample["label"] = self.classes.index(sample["celltype"])
        sample["target"] = (
            self.train_classes.index(sample["label"]) if not "val" in self.split else -1
        )
        sample["batch_idx"] = self.batches.index(sample["batch"])

        return sample

    def tokenize_function(self, samples):
        return self.tokenizer(samples["gene"])

    def to_raw(self, g, c):
        count_raw = np.zeros(self.num_genes, dtype=c.dtype)
        count_raw[g - self.num_special_tokens] = c
        return count_raw

    def get_mask(self, length_input):
        mask_gene = np.random.binomial(
            1, self.mask_prop_gene + self.mask_prop_count, length_input
        )

        mask_count = np.random.binomial(
            1,
            self.mask_prop_count / (self.mask_prop_gene + self.mask_prop_count),
            length_input,
        )

        # 0 for non-mask, 1 for gene mask, 2 for count mask
        mask = mask_gene + mask_count * mask_gene
        return mask.astype(int)

    def add_noise(
        self, counts: np.ndarray, kind: Literal["global", "local"]
    ) -> List[np.ndarray]:
        """
        counts: 1d array
        noise_ratio: float
        dropout_p: 1d array
        """
        if kind == "global":
            noise_ratio = self.noise_ratio_g
            dropout_p = self.dropout_p_g_l
        elif kind == "local":
            noise_ratio = self.noise_ratio_l
            dropout_p = self.dropout_p_l_l
        else:
            raise ValueError(f"kind should be global or local, not {kind}")
        num_genes = len(counts)
        num_crops = len(dropout_p)
        if noise_ratio:
            noise = np.random.randn(num_crops, num_genes) * noise_ratio
            res = counts[None] * (1 + noise)  # + noise * 5
        else:
            res = counts[None]
        if dropout_p:
            dropout = np.stack([1 - np.random.rand(num_genes) * p for p in dropout_p])
            res = res * dropout
        res = res.clip(min=0).round().astype(int)
        return list(res)

    def _sample_random(
        self, gene, count, is_global: bool = True, gene_list_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if is_global:
            min_length = self.min_length_g
            mean_length = self.mean_length_g
            max_length = self.max_length_g
        else:
            min_length = self.min_length_l
            mean_length = self.mean_length_l
            max_length = self.max_length_l

        count_raw = self.to_raw(gene, count)
        if self.log_count:
            # give small prob to zero count.
            p = np.log1p(count_raw / self.count_temp) + np.log1p(
                self.zero_prob / self.count_temp
            )
        else:
            p = count_raw + self.zero_prob
        if gene_list_idx is not None:
            p[self.exclude_genes[gene_list_idx]] = 0
        p /= p.sum()

        if self.length_dist == "sample":
            p = p * mean_length
            while True:
                rands = np.random.rand(self.num_genes)
                mask = rands < p
                sampled_gene_ids = np.arange(self.num_genes)[mask]
                if min_length <= len(sampled_gene_ids) <= max_length:
                    break
        elif self.length_dist == "uniform":
            sample_size = np.random.randint(min_length, max_length + 1)
            sampled_gene_ids = np.random.choice(self.num_genes, sample_size, p=p)
        else:
            raise NotImplementedError

        return sampled_gene_ids + self.num_special_tokens, count_raw[sampled_gene_ids]

    def pad(
        self, genes: list, counts: list, masks: list, max_length: int
    ) -> Tuple[list, list, list]:
        genes_padded = []
        counts_padded = []
        masks_padded = []
        gene_pad = self.tokenizer.pad_token_id
        count_pad = -3
        mask_pad = -1
        for gene, count, mask in zip(genes, counts, masks):
            pad_len = max_length - len(gene)
            if pad_len:
                # np.concatenate([np.arange(2), []]).dtype
                # >>> dtype("float64")
                genes_padded.append(np.concatenate([gene, [gene_pad] * pad_len]))
                counts_padded.append(np.concatenate([count, [count_pad] * pad_len]))
                masks_padded.append(np.concatenate([mask, [mask_pad] * pad_len]))
            else:
                genes_padded.append(gene)
                counts_padded.append(count)
                masks_padded.append(mask)
        return genes_padded, counts_padded, masks_padded


class PanaceaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data: str = "pancreas",
        test: bool = False,
        **data_config_dict,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            kwargs = self.hparams.copy()

            kwargs["num_crops_g"] = 1
            kwargs["noise_ratio_g"] = kwargs["dropout_p_g"] = 0
            kwargs["num_crops_l"] = 0
            if not self.hparams.test:
                self.dataset_train = SingleCellDataset(split="train", **self.hparams)
            else:
                kwargs["length_dist"] = "uniform"
                kwargs["min_length_g"] = kwargs["mean_length_g"] = kwargs[
                    "max_length_g"
                ]
                self.dataset_train = SingleCellDataset(split="train", **kwargs)
            self.dataset_val_none = SingleCellDataset(split="val_none", **kwargs)
            try:
                self.dataset_val_type = SingleCellDataset(split="val_type", **kwargs)
            except FileNotFoundError:
                self.dataset_val_type = None
            try:
                self.dataset_val_batch = SingleCellDataset(split="val_batch", **kwargs)
            except FileNotFoundError:
                self.dataset_val_batch = None

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=False if self.hparams.test else True,
            num_workers=16,
            pin_memory=False if self.hparams.test else True,
        )

    def val_dataloader(self):
        ret = []
        loader_val_none = DataLoader(
            self.dataset_val_none,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=False if self.hparams.test else True,
        )
        ret.append(loader_val_none)
        if self.dataset_val_type is not None:
            loader_val_type = DataLoader(
                self.dataset_val_type,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=False if self.hparams.test else True,
            )
            ret.append(loader_val_type)
        if self.dataset_val_batch is not None:
            loader_val_batch = DataLoader(
                self.dataset_val_batch,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=False if self.hparams.test else True,
            )
            ret.append(loader_val_batch)
        return ret


if __name__ == "__main__":
    import json
    from typing import Sequence

    import yaml
    from rich import print as rprint
    from rich.traceback import install

    install()

    DATA = "pancreas"

    with open("config_data.yaml") as f:
        config = yaml.safe_load(f)
    with open(f"configs/{DATA}.yaml") as f:
        config.update(yaml.safe_load(f))

    datamodule = PanaceaDataModule(
        data_dir="/data/tiankang/wusuowei/data/single_cell/panacea",
        data=DATA,
        **config,
    )
    datamodule.setup(stage="fit")
    dataset = datamodule.dataset_train
    rprint(len(dataset))
    sample = dataset[123]
    for k, v in sample.items():
        rprint(k, type(v), ":")
        if isinstance(v, np.ndarray):
            rprint(v.shape, v.dtype)
        elif isinstance(v, list):
            rprint([(i.shape, i.dtype) for i in v])
        elif not isinstance(v, Sequence):
            rprint(v)

    test_data = {}
    test_data["gene"] = sample["gene"][0].tolist()
    test_data["count"] = sample["count"][0].tolist()
    test_data["mask_gene"] = (sample["mask"][0] == 1).tolist()
    test_data["mask_count"] = (sample["mask"][0] == 2).tolist()
    # save to json
    with open(f"test_data.json", "w") as f:
        json.dump(test_data, f)

    rprint(dataset.tokenizer.pad_token_id)

    # for sample in datamodule.train_dataloader():
