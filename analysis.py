import os
from typing import Literal, Optional, Union

import numpy as np
from scipy import sparse as ss
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform
import pandas as pd
from joblib import Parallel, delayed
from pytorch_lightning import seed_everything
import anndata
from anndata import AnnData
from sklearn.utils import check_random_state, check_array
from sklearn.decomposition import PCA
from umap import UMAP
from umap.umap_ import simplicial_set_embedding, find_ab_params
import scanpy as sc
from scanpy._utils import NeighborsView
from scanpy.tools._utils import get_init_pos_from_paga, _choose_representation
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from rich import print as rprint
from rich.traceback import install


sns.color_palette("Spectral", as_cmap=True)

color_map_pastel = [
    "rgb(102, 197, 204)",
    "rgb(246, 207, 113)",
    "rgb(248, 156, 116)",
    "rgb(220, 176, 242)",
    "rgb(135, 197, 95)",
    "rgb(158, 185, 243)",
    "rgb(254, 136, 177)",
    "rgb(201, 219, 116)",
    "rgb(139, 224, 164)",
    "rgb(180, 151, 231)",
    "rgb(179, 179, 179)",
]


def plot(
    e: np.ndarray,
    labels: np.ndarray,
    output_file: Optional[str] = None,
) -> None:
    fig = go.Figure()
    for i in pd.value_counts(labels).sort_values(ascending=False).index:
        fig.add_trace(
            go.Scattergl(
                x=e[labels == i, 0],
                y=e[labels == i, 1],
                mode="markers",
                name=i,
                marker=dict(size=4, opacity=0.5),  # , color=color_map_pastel[i]),
            ),
        )
    fig.update_layout(
        width=1000,
        height=1000,
    )
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.write_image(output_file)
    return fig


# scanpy preprocessing
def scanpy_preprocessing(adata, min_cells=3, n_top_genes=2000, num_pcs=30):
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=num_pcs, n_neighbors=30)
    return adata


def scanpy_umap(
    adata: AnnData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Union[
        Literal["paga", "spectral", "random"], np.ndarray, None
    ] = "spectral",
    random_state=0,
    a: Optional[float] = None,
    b: Optional[float] = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",
    neighbors_key: Optional[str] = None,
) -> Optional[AnnData]:
    """\
    Embed the neighborhood graph using UMAP [McInnes18]_.
    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout Scanpy using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.  We use the
    implementation of `umap-learn <https://github.com/lmcinnes/umap>`__
    [McInnes18]_. For a few comparisons of UMAP with tSNE, see this `preprint
    <https://doi.org/10.1101/298430>`__.
    Parameters
    ----------
    adata
        Annotated data matrix.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out. The default of in the `umap-learn` package is
        0.1.
    spread
        The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called `n_epochs`
        in the original UMAP.
    alpha
        The initial learning rate for the embedding optimization.
    gamma
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Called `init` in the
        original UMAP. Options are:
        * Any key for `adata.obsm`.
        * 'paga': positions from :func:`~scanpy.pl.paga`.
        * 'spectral': use a spectral embedding of the graph.
        * 'random': assign initial embedding positions at random.
        * A numpy array of initial embedding positions.
    random_state
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` or `Generator`, `random_state` is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    a
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    b
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    copy
        Return a copy instead of writing to adata.
    method
        Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
    neighbors_key
        If not specified, umap looks .uns['neighbors'] for neighbors settings
        and .obsp['connectivities'] for connectivities
        (default storage places for pp.neighbors).
        If specified, umap looks .uns[neighbors_key] for neighbors settings and
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **X_umap** : `adata.obsm` field
        UMAP coordinates of data.
    """
    adata = adata.copy() if copy else adata

    if neighbors_key is None:
        neighbors_key = "neighbors"

    if neighbors_key not in adata.uns:
        raise ValueError(
            f'Did not find .uns["{neighbors_key}"]. Run `sc.pp.neighbors` first.'
        )
    # start = logg.info('computing UMAP')

    neighbors = NeighborsView(adata, neighbors_key)

    # if 'params' not in neighbors or neighbors['params']['method'] != 'umap':
    #     logg.warning(
    #         f'.obsp["{neighbors["connectivities_key"]}"] have not been computed using umap'
    #     )

    # # Compat for umap 0.4 -> 0.5
    # with warnings.catch_warnings():
    #     # umap 0.5.0
    #     warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
    #     import umap

    # if version.parse(umap.__version__) >= version.parse("0.5.0"):

    # def simplicial_set_embedding(*args, **kwargs):
    #     from umap.umap_ import simplicial_set_embedding

    #     X_umap, _ = simplicial_set_embedding(
    #         *args,
    #         densmap=False,
    #         densmap_kwds={},
    #         output_dens=False,
    #         **kwargs,
    #     )
    #     return X_umap

    # else:
    #     from umap.umap_ import simplicial_set_embedding

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
    adata.uns["umap"] = {"params": {"a": a, "b": b}}
    if isinstance(init_pos, str) and init_pos in adata.obsm.keys():
        init_coords = adata.obsm[init_pos]
    elif isinstance(init_pos, str) and init_pos == "paga":
        init_coords = get_init_pos_from_paga(
            adata, random_state=random_state, neighbors_key=neighbors_key
        )
    else:
        init_coords = init_pos  # Let umap handle it
    if hasattr(init_coords, "dtype"):
        init_coords = check_array(init_coords, dtype=np.float32, accept_sparse=False)

    if random_state != 0:
        adata.uns["umap"]["params"]["random_state"] = random_state
    random_state = check_random_state(random_state)

    neigh_params = neighbors["params"]
    X = _choose_representation(
        adata,
        neigh_params.get("use_rep", None),
        neigh_params.get("n_pcs", None),
        silent=True,
    )

    if method == "umap":
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding
        default_epochs = 500 if neighbors["connectivities"].shape[0] <= 10000 else 200
        n_epochs = default_epochs if maxiter is None else maxiter
        X_umap, _ = simplicial_set_embedding(
            X,
            neighbors["connectivities"].tocoo(),
            n_components,
            alpha,
            a,
            b,
            gamma,
            negative_sample_rate,
            n_epochs,
            init_coords,
            random_state,
            neigh_params.get("metric", "euclidean"),
            neigh_params.get("metric_kwds", {}),
            densmap=False,
            densmap_kwds={},
            output_dens=False,
        )
    else:
        raise NotImplementedError

    adata.obsm["X_umap"] = X_umap  # annotate samples with UMAP coordinates
    return X_umap
    # logg.info(
    #     '    finished',
    #     time=start,
    #     deep=('added\n' "    'X_umap', UMAP coordinates (adata.obsm)"),
    # )
    # return adata if copy else None


def dist(g1, g2):
    return ss.linalg.norm(g1 - g2) / g1.shape[0]


def get_scanpy(adata, seed=42):
    # scanpy preprocessing + scanpy umap
    seed_everything(seed)
    adata_scanpy = scanpy_preprocessing(adata)
    neighbors = NeighborsView(adata_scanpy, "neighbors")
    graph_scanpy = neighbors["connectivities"]
    embedding_scanpy = scanpy_umap(adata_scanpy, min_dist=0.1)
    return graph_scanpy, embedding_scanpy


def get_scanpy_umap(adata, seed=42):
    # scanpy preprocessing + umap
    seed_everything(seed)
    adata_scanpy = scanpy_preprocessing(adata)
    umap_scanpy = UMAP().fit(adata_scanpy.X)
    graph_scanpy_umap = umap_scanpy.graph_
    embedding_scanpy_umap = umap_scanpy.transform(adata_scanpy.X)
    return graph_scanpy_umap, embedding_scanpy_umap


def get_raw(adata, seed=42, hv=False):
    # raw data
    seed_everything(seed)
    if hv:
        adata = get_hv(adata)
    umap_raw = UMAP().fit(adata.X)
    graph_raw = umap_raw.graph_
    embedding_raw = umap_raw.transform(adata.X)
    return graph_raw, embedding_raw


def get_l1(adata, seed=42, hv=False):
    # normalize each row
    seed_everything(seed)
    data = adata.X / (adata.X.sum(axis=1) + 1)
    if hv:
        data = get_hv(data)
    umap_l1 = UMAP().fit(data)
    graph_l1 = umap_l1.graph_
    embedding_l1 = umap_l1.transform(data)
    return graph_l1, embedding_l1


def get_zscore(adata, seed=42, hv=False):
    # zscore
    seed_everything(seed)
    data = adata.X.todense()
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    if hv:
        data = get_hv(data)
    umap_zscore = UMAP().fit(data)
    graph_zscore = umap_zscore.graph_
    embedding_zscore = umap_zscore.transform(data)
    # zscore exaggerates batch effect
    return graph_zscore, embedding_zscore


def get_pca(adata, seed=42, hv=False):
    # PCA to 50
    seed_everything(seed)
    if hv:
        adata = get_hv(adata)
    data = PCA(n_components=50).fit_transform(adata.X.todense())
    umap_pca = UMAP().fit(data)
    graph_pca = umap_pca.graph_
    embedding_pca = umap_pca.transform(data)
    return graph_pca, embedding_pca


def get_l1_pca(adata, seed=42, hv=False):
    # l1 + PCA to 50
    seed_everything(seed)
    data = adata.X / (adata.X.sum(axis=1) + 1)
    if hv:
        data = get_hv(data)
    data = PCA(n_components=50).fit_transform(data)
    umap_l1_pca = UMAP().fit(data)
    graph_l1_pca = umap_l1_pca.graph_
    embedding_l1_pca = umap_l1_pca.transform(data)
    return graph_l1_pca, embedding_l1_pca


def get_hv(adata, n_top_genes=2000):
    if isinstance(adata, AnnData):
        stds = std(adata.X, axis=0)
    elif isinstance(adata, ss.spmatrix):
        stds = std(adata, axis=0)
    elif isinstance(adata, np.matrix):
        stds = adata.std(axis=0).A.flatten()
    else:
        raise NotImplementedError
    return adata[:, np.argsort(stds)[-n_top_genes:]].copy()


def plot_clustermap(graphs: ss.spmatrix, names: str, output_file: str) -> None:
    dists = np.zeros((len(graphs), len(graphs)))
    for i, g1 in enumerate(graphs):
        for j, g2 in enumerate(graphs):
            dists[j, i] = dists[i, j] = dist(g1, g2)

    fig = sns.clustermap(
        dists,
        cmap="magma",
        xticklabels=names,
        yticklabels=names,
        figsize=(15, 15),
        linewidths=0.5,
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file)


def std(spmatrix, axis):
    a = spmatrix.copy()
    a.data **= 2
    std = np.sqrt(a.mean(axis=axis) - spmatrix.mean(axis=axis).A ** 2).A.flatten()
    return std


def shuffle(spmatrix: ss.spmatrix) -> ss.spmatrix:
    """shuffle the values and their position in a symmetric sparse matrix (keep the symmetry)."""
    dense = spmatrix.todense()
    upper = squareform(dense)
    # diag = np.diag(dense)  # diag must be zero, or squareform will not work
    upper = np.random.permutation(upper)
    dense = squareform(upper)
    # dense[np.diag_indices_from(dense)] = diag
    if isinstance(spmatrix, ss.csr_matrix):
        return ss.csr_matrix(dense)
    elif isinstance(spmatrix, ss.csc_matrix):
        return ss.csc_matrix(dense)
    else:
        raise ValueError("Unknown matrix type")


def downsample_exper(adata, func, i, num_repeats, xs) -> None:
    name = func.__name__.replace("get_", "")
    if i > 6:
        name += "_hv"
    os.makedirs(f"data/downsample_exper/{name}", exist_ok=True)
    for repeat_i in tqdm(range(num_repeats)):
        adata_downsampled = adata
        for downsample in tqdm(xs[:-1]):
            down_ratio = adata.shape[1] * downsample / adata_downsampled.shape[1]
            sampled_genes = np.random.choice(
                adata_downsampled.shape[1],
                int(adata_downsampled.shape[1] * down_ratio),
                replace=False,
            )
            adata_downsampled = adata[:, sampled_genes]
            if i <= 6:
                graph, emb = func(adata_downsampled, seed=np.random.randint(0, 100))
            else:
                graph, emb = func(
                    adata_downsampled, hv=True, seed=np.random.randint(0, 100)
                )
            # save graph and emb
            ss.save_npz(f"data/downsample_exper/{name}/graph_{repeat_i}.npz", graph)
            np.save(f"data/downsample_exper/{name}/emb_{repeat_i}.npy", np.array(emb))


def jaccard(graph1, graph2):
    intersec = graph1.multiply(graph2).astype(bool).sum()
    union = graph1.astype(bool).sum() + graph2.astype(bool).sum() - intersec
    return intersec / union


def downsample_plot(adata, func, i, num_repeats, xs) -> None:
    name = func.__name__.replace("get_", "")
    if i > 6:
        name += "_hv"
    os.makedirs(f"figs/downsample_exper/{name}", exist_ok=True)
    dists = []
    corrs_raw = []
    corrs_nonzero = []
    jacs = []
    gs = []
    for repeat_i in tqdm(range(num_repeats)):
        graphs = []
        for downsample in tqdm(xs[:-1]):
            graph = ss.load_npz(f"data/downsample_exper/{name}/graph_{repeat_i}.npz")
            emb = np.load(f"data/downsample_exper/{name}/emb_{repeat_i}.npy")
            graphs.append(graph)
            plot(
                emb,
                adata.obs["batch"],
                f"figs/{name}/{repeat_i}/batch_{downsample:.1f}_emb.png",
            )
            plot(
                emb,
                adata.obs["celltype"],
                f"figs/{name}/{repeat_i}/label_{downsample:.1f}_emb.png",
            )

        graphs.append(shuffle(graphs[0]))

        gs.append(graphs)
        dists.append([dist(g_down, graphs[0]) for g_down in graphs])

        raws = []
        nonzeros = []
        g0 = squareform(graphs[0].todense())
        for g_down in graphs:
            g = squareform(g_down.todense())
            raws.append(spearmanr(g, g0).correlation)
            either_non_zero = np.logical_or(g, g0)
            g_ = g[either_non_zero]
            g0_ = g0[either_non_zero]
            nonzeros.append(spearmanr(g_, g0_).correlation)
        corrs_raw.append(raws)
        corrs_nonzero.append(nonzeros)
        jacs.append([jaccard(g_down, graphs[0]) for g_down in graphs])

    gs = np.array(gs)

    for i, x in enumerate(xs[:-1]):
        # "interconsistency"
        graphs = gs[:, i]
        ns = list(map(lambda x: f"rep-{x}", range(len(graphs))))
        graphs = np.append(graphs, shuffle(graphs[0]))
        ns.append("shuffled-0")
        plot_clustermap(graphs, ns, f"figs/{name}/consistency_{x:.1f}.png")

    xs = np.stack([xs for _ in range(num_repeats)])
    dists = np.array(dists)
    corrs = np.array(corrs)

    for ys, postfix in zip(
        [dists, corrs_raw, corrs_nonzero, jacs],
        ["dist", "corr_raw", "corr_nonzero", "jaccard"],
    ):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_xaxis()
        line_df = pd.DataFrame(
            dict(
                x=xs.flatten(),
                y=ys.flatten(),
                hue=np.arange(xs.shape[0]).repeat(xs.shape[1]),
            )
        )
        sns.lineplot(
            data=line_df,
            x="x",
            y="y",
            hue="hue",
            alpha=0.3,
            ax=ax,
        )
        mean = np.mean(ys, axis=0)
        std = np.std(ys, axis=0)
        sns.lineplot(
            x=xs[0],
            y=mean,
            ax=ax,
        )
        ax.fill_between(xs[0], mean - std, mean + std, alpha=0.2)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        sns.despine(fig)
        fig.savefig(f"figs/{name}/downsample_{postfix}.png")


if __name__ == "__main__":
    install()
    seed_everything(42)

    data_dir = "/data/tiankang/SCALEX/pancreas"

    adata = anndata.read_h5ad(f"{data_dir}/adata.h5ad")
    rprint(adata.shape)
    sc.pp.filter_cells(adata, min_genes=200)
    rprint(adata.shape)

    # all genes
    graph_raw, embedding_raw = get_raw(adata)
    graph_l1, embedding_l1 = get_l1(adata)
    graph_zscore, embedding_zscore = get_zscore(adata)
    graph_pca, embedding_pca = get_pca(adata)
    graph_l1_pca, embedding_l1_pca = get_l1_pca(adata)

    # scanpy feature selection
    adata_scanpy = scanpy_preprocessing(adata)
    graph_scanpy, embedding_scanpy = get_scanpy(adata)
    graph_scanpy_umap, embedding_scanpy_umap = get_scanpy_umap(adata)

    # 2000 highly variable genes
    graph_raw_hv, embedding_raw_hv = get_raw(adata, hv=True)
    graph_l1_hv, embedding_l1_hv = get_l1(adata, hv=True)
    graph_zscore_hv, embedding_zscore_hv = get_zscore(adata, hv=True)
    graph_pca_hv, embedding_pca_hv = get_pca(adata, hv=True)
    graph_l1_pca_hv, embedding_l1_pca_hv = get_l1_pca(adata, hv=True)

    # plot distance matrix
    graphs = [
        graph_raw,
        graph_l1,
        graph_zscore,
        graph_pca,
        graph_l1_pca,
        graph_scanpy,
        graph_scanpy_umap,
        graph_raw_hv,
        graph_l1_hv,
        graph_zscore_hv,
        graph_pca_hv,
        graph_l1_pca_hv,
    ]
    names = [
        "raw",
        "l1",
        "zscore",
        "pca",
        "l1_pca",
        "scanpy",
        "scanpy_umap",
        "raw_hv",
        "l1_hv",
        "zscore_hv",
        "pca_hv",
        "l1_pca_hv",
    ]
    plot_clustermap(graphs, names, f"figs/full/comp_preproc.png")

    for emb, name in zip(
        [
            embedding_raw,
            embedding_l1,
            embedding_l1_pca,
            embedding_zscore,
            embedding_pca,
            embedding_scanpy,
            embedding_scanpy_umap,
            embedding_raw_hv,
            embedding_l1_hv,
            embedding_zscore_hv,
            embedding_pca_hv,
            embedding_l1_pca_hv,
        ],
        [
            "raw",
            "l1",
            "l1+pca",
            "zscore",
            "pca",
            "scanpy",
            "scanpy+umap",
            "raw+hv",
            "l1+hv",
            "zscore+hv",
            "pca+hv",
            "l1+pca+hv",
        ],
    ):
        plot(emb, adata.obs["celltype"], f"figs/full/{name}_label.png")
        plot(emb, adata.obs["batch"], f"figs/full/{name}_batch.png")

    # the downsample experiment
    for f in [downsample_exper]:
        Parallel(n_jobs=8)(
            delayed(f)(adata, func, i, num_repeats=8, xs=np.linspace(1, 0, 11))
            for i, func in enumerate(
                tqdm(
                    [
                        get_raw,
                        get_l1,
                        get_zscore,
                        get_pca,
                        get_l1_pca,
                        get_scanpy,
                        get_scanpy_umap,
                        get_raw,
                        get_l1,
                        get_zscore,
                        get_pca,
                        get_l1_pca,
                    ]
                )
            )
        )

    # downsample_exper(adata, get_raw, 0)
