import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


class SingleCellDataset(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, split: str, max_length: int):
        super().__init__()
        dataset = load_dataset(
            "json",
            data_files={split: f"{data_dir}/{dataset_name}_{split}.json.zip"},
            field="data",
        )
        self.split = split
        self.max_length = max_length
        # self.tokenizer = Tokenizer.from_file(f"{data_dir}/tokenizer.json")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{data_dir}/tokenizer.json"
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.dataset = dataset.map(self.tokenize_function, batched=True)

    def tokenize_function(self, samples):
        return self.tokenizer(samples["gene"])

    def _sample_random(self, *seqs):
        idxs = np.random.choice(len(seqs[0]), self.max_length, replace=False)
        return [seq[idxs] for seq in seqs]

    def __len__(self):
        return len(self.dataset)

    def pad_trunc(self, input_ids, counts, attention_mask):
        if len(input_ids) < self.max_length:  # pad
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            )
            attention_mask = attention_mask + [0] * (
                self.max_length - len(attention_mask)
            )
            counts = counts + [-1] * (self.max_length - len(counts))
        else:  # truncate
            return self._sample_random(input_ids, counts, attention_mask)

    def __getitem__(self, index):
        sample = self.dataset[self.split][index]
        count = np.array(sample["count"].split(" ")).astype(int)
        gene = np.array(sample["input_ids"])
        am = np.array(sample["attention_mask"])
        sample["input_ids"], sample["count"], sample["attention_mask"] = self.pad_trunc(
            gene, count, am
        )
        return sample


if __name__ == "__main__":
    from rich import print as rprint

    dataset = SingleCellDataset(
        data_dir="/home/tiankang/wusuowei/data/single_cell/panacea",
        dataset_name="pancreas",
        split="val_type",
        max_length=512,
    )
    rprint(dataset[0].keys())
