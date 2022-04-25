# %%
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import yaml
from tqdm.auto import tqdm
from rich import print as rprint
from rich.traceback import install

from datamodules import SingleCellDataset


# %%
install()
# default style
sns.set_style("ticks")
sns.set_palette("Set1")
plt.rcParams["figure.dpi"] = 300
pl.seed_everything(2022)

# %%
def to_img(dataset, g, c, m, count_raw):
    interval_start = 5880
    interval_end = 5880 + 50
    c = dataset.to_raw(g, c)
    m = dataset.to_raw(g, m)
    c = c[interval_start:interval_end]
    m = m[interval_start:interval_end]
    norm = count_raw[interval_start:interval_end].max()
    g = g - dataset.num_special_tokens
    g = g[np.logical_and(interval_start <= g, g < interval_end)] - interval_start
    c = c / norm * 0.4 + 0.6  # within [0.6, 1]
    c_img = np.zeros(50) + 0.4  # unselected elements are 0.4
    c_img[g] = c[g]
    m_img = np.zeros(50)
    m_img[g] = m[g]
    rprint(pd.value_counts(m[g]))
    return c_img, m_img


def plot_custom_color(count, mask, pad_len):
    """
    Modified from https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    count_filter = count[count != 0.4]
    mask_filter = mask[count != 0.4]
    figh = 5
    fig, axs = plt.subplots(nrows=2, figsize=(30, figh))

    n = 4
    gradient = count.repeat(n)
    gradient[::n] = np.nan
    gradient = np.concatenate([[1, 0, np.nan], gradient, [np.nan, 0, 1]])

    def get_filtered_gradient(c_f):
        gradient_f = np.concatenate([c_f, [np.nan] * (len(count) - len(c_f))]).repeat(n)
        gradient_f[::n] = np.nan
        gradient_f = np.concatenate([[1, 0, np.nan], gradient_f, [np.nan, 0, 1]])
        return gradient_f

    filtered = get_filtered_gradient(count_filter)

    masked = (
        (mask_filter == 1).astype(float) - (mask_filter == 2).astype(float) + 1
    ) / 2
    masked[masked == 0.5] = np.nan
    masked = get_filtered_gradient(masked)
    sep = np.ones_like(gradient) * np.nan

    if pad_len:
        start = 3 + n * len(count_filter)
        end = 3 + n * pad_len
        filtered_pad = filtered.copy()
        filtered_pad[start:end] = 0.4
        filtered_pad[start:end:n] = np.nan
        gradient = np.stack([gradient, sep, filtered, sep, filtered_pad])
    else:
        gradient = np.stack([gradient, sep, filtered, sep, sep])
    gradient_mask = np.stack([masked, sep, sep, sep, sep])

    axs[0].imshow(1 - gradient, aspect="auto", cmap="RdGy")
    axs[1].imshow(gradient_mask, aspect="auto", cmap="cividis")
    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
    return fig


def _sample_random(self, gene, count, is_global) -> Tuple[np.ndarray, np.ndarray]:
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
    p /= p.sum()
    return p

    # if self.length_dist == "sample":
    #     p = p * mean_length
    #     # deal with the case when
    #     while True:
    #         rands = np.random.rand(self.num_genes)
    #         mask = rands < p
    #         sampled_gene_ids = np.arange(self.num_genes)[mask]
    #         if min_length <= len(sampled_gene_ids) <= max_length:
    #             break
    # elif self.length_dist == "uniform":
    #     sample_size = np.random.randint(min_length, max_length + 1)
    #     sampled_gene_ids = np.random.choice(self.num_genes, sample_size, p=p)
    # else:
    #     raise NotImplementedError

    # return sampled_gene_ids + self.num_special_tokens, count_raw[sampled_gene_ids]


# %%
get_interval = lambda x: x[5880 : 5880 + 100]

data_dir = "/home/tiankang/wusuowei/data/single_cell/panacea"
split = "train"

with open("config_data.yaml") as f:
    config = yaml.safe_load(f)

# %%
config_increase_sample = config.copy()
config_increase_sample.update(
    {
        "min_length_g": 4096,
        "mean_length_g": 6144,
        "max_length_g": 8192,
        "min_length_l": 2048,
        "mean_length_l": 3072,
        "max_length_l": 4096,
    }
)
dataset = SingleCellDataset(data_dir=data_dir, split=split, **config_increase_sample)

# non_sparsity = np.array(
#     [sum(dataset[i]["count_raw"] != 0) for i in tqdm(range(len(dataset)))]
# )
# rprint(
#     non_sparsity.argmax(),
#     "has the biggest non-zero proportion:",
#     non_sparsity.max() / dataset.num_genes,
# )
# index = non_sparsity.argmax()

# %%
# We take as example a sample with high non-zero proportion. (sample 6560 in
# training set has 5880 non-zero entries)
index = 6560
self = dataset
sample = self.dataset[self.split][index]
count = np.array(sample["count"].split(" ")).astype(int)
gene = np.array(sample["input_ids"])
count_raw = self.to_raw(gene, count)
sample["count_raw"] = count_raw

print((get_interval(count_raw) != 0).sum())

# %%
# ===================
# plot all genes
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
sns.lineplot(x=np.arange(dataset.num_genes), y=count_raw, ax=ax)
fig.savefig("figs/test_dataset/count_raw.jpg")
plt.close(fig)
# ===================


# %%
dropout_p_g = [0] + [self.dropout_p_g] * (self.num_crops_g - 1)
dropout_p_l = [self.dropout_p_l] * self.num_crops_l
count_g = self.add_noise(count, self.noise_ratio_g, dropout_p_g)
count_l = self.add_noise(count, self.noise_ratio_l, dropout_p_l)


# %%
# ===================
count_g_raw = [dataset.to_raw(gene, c) for c in count_g]
count_l_raw = [dataset.to_raw(gene, c) for c in count_l]
# plot a interval with high non-zero proportion
df = pd.DataFrame(dict(x=np.arange(100), y=get_interval(count_raw), hue="raw"))
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
for i in range(dataset.num_crops_g):
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                dict(
                    x=np.arange(100), y=get_interval(count_g_raw[i]), hue=f"global-{i}"
                )
            ),
        ]
    ).reset_index(drop=True)
for i in range(dataset.num_crops_l):
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                dict(x=np.arange(100), y=get_interval(count_l_raw[i]), hue=f"local-{i}")
            ),
        ]
    ).reset_index(drop=True)
sns.lineplot(data=df, x="x", y="y", hue="hue", ax=ax, alpha=0.5)
# remove legend
ax.legend_.remove()
fig.legend(fontsize="x-large", loc="upper right")
fig.savefig("figs/test_dataset/count_interval_noisy.jpg")
plt.close(fig)

# sample prob
p = _sample_random(dataset, gene, count, is_global=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
pd.Series(p, index=count_raw).sort_index().drop_duplicates().plot(ax=ax)
p = get_interval(p)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.lineplot(x=np.arange(100), y=p / p.max(), ax=ax)
fig.savefig("figs/test_dataset/sample_prob.jpg")
# ===================

# %%
gene_g, count_g = zip(
    *(
        self._sample_random(
            gene,
            count,
            is_global=True,
        )
        for count in count_g
    )
)
if self.num_crops_l:
    gene_l, count_l = zip(
        *(
            self._sample_random(
                gene,
                count,
                is_global=False,
            )
            for count in count_l
        )
    )
else:
    gene_l = []
    count_l = []

mask_g = [self.get_mask(len(gene)) for gene in gene_g]
mask_l = [self.get_mask(len(gene)) for gene in gene_l]

# gene_g, count_g, mask_g = self.pad(gene_g, count_g, mask_g, self.max_length_g)
# gene_l, count_l, mask_l = self.pad(gene_l, count_l, mask_l, self.max_length_l)

genes = gene_g + gene_l
counts = count_g + count_l
masks = mask_g + mask_l


# %%
# ===================
# plot sampling scheme
# before sampling (all genes are sampled)
fig = plot_custom_color(
    *to_img(
        dataset,
        g=np.arange(self.num_genes),
        c=count_raw,
        m=np.zeros(self.num_genes),
        count_raw=count_raw,
    ),
    pad_len=0,
)
fig.savefig("figs/test_dataset/count_raw_before_sampling.jpg")
plt.close(fig)
# after sampling
for i, (g, c, m) in enumerate(zip(genes, counts, masks)):
    pad_len = 25 if i < self.num_crops_g else 15
    fig = plot_custom_color(*to_img(dataset, g, c, m, count_raw), pad_len)
    fig.savefig(f"figs/test_dataset/count_raw_after_sampling_{i}.jpg")
    plt.close(fig)
# ===================


# %%
dataset = SingleCellDataset(data_dir=data_dir, split=split, **config)


# %%
def plot_count_dist(dataset, i, zero_prob=None, count_temp=None) -> None:
    zero_prob_o = dataset.zero_prob
    count_temp_o = dataset.count_temp
    dataset.zero_prob = zero_prob if zero_prob is not None else dataset.zero_prob
    dataset.count_temp = count_temp if count_temp is not None else dataset.count_temp

    sample = dataset[i]
    sample_raw = dataset.dataset[dataset.split][i]
    
    # plot histogram of count distribution before and after sampling
    df = pd.DataFrame(dict(x=sample["count_raw"], view="raw"))
    for i in range(dataset.num_crops_g + dataset.num_crops_l):
        x = sample["count"][i][sample["mask"][i] != -1]
        hue = (
            f"global-{i}"
            if i < dataset.num_crops_g
            else f"local-{i - dataset.num_crops_g}"
        )
        df = pd.concat([df, pd.DataFrame(dict(x=x, view=hue))]).reset_index(drop=True)
    df.x += 10
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.histplot(
        data=df,
        x="x",
        hue="view",
        log_scale=True,
        element="poly",
        fill=False,
        ax=ax,
        stat="proportion",
        common_norm=False,
        bins=30,
    )
    p = _sample_random(
        dataset,
        np.array(sample_raw["input_ids"]),
        np.array(sample_raw["count"].split(" ")).astype(int),
        is_global=True,
    )
    ax.set_ylim(-0.01, 0.31)
    # remove upper and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
    # https://stackoverflow.com/questions/66346542/customizing-legend-in-seaborn-histplot-subplots
    legend = ax.get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax.legend(
        handles,
        ["raw", "global-0", "global-1", "local-0", "local-1", "local-2"],
        title="Views",
        loc="upper right",
        bbox_to_anchor=(1.25, 0.95),
        ncol=1,
        fancybox=True,
        shadow=True,
    )
    # ax.get_legend().remove()
    # ax.legend(fontsize="x-large", loc="upper right")
    ax_t = ax.twinx()
    pd.Series(p, index=sample["count_raw"] + 10).sort_index().drop_duplicates().plot(
        ax=ax_t, color="gray", linestyle="--"
    )
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["left"].set_visible(False)
    fig.savefig(f"figs/test_dataset/count_raw_after_sampling_{i}.jpg")

    dataset.zero_prob = zero_prob_o
    dataset.count_temp = count_temp_o

# %%
# ===================
for i in [6560, 11087, 11198]:  # non-sparsity: [0.72, 0.51, 0.29]
# for i in [11087]:
    # for zp in [0.05, 0.1, 0.15, 0.2]:
    plot_count_dist(dataset, i)

# # %%
# for i in [11087]:
#     for cp in [0.5, 1, 2, 4]:
#         plot_count_dist(dataset, i, count_temp=cp)

# %%
non_sparsity = np.array(
    [
        len(dataset.dataset[dataset.split][i]["input_ids"]) / dataset.num_genes
        for i in tqdm(range(len(dataset)))
    ]
)

# %%
