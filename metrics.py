import numpy as np
from sklearn import metrics
from munkres import Munkres
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class iBOTMetrics(Metric):
    """NMI, ARI and FMI"""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        assert preds.dim() == 1 and target.dim() == 1, (preds.dim(), target.dim())
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        nmi = metrics.normalized_mutual_info_score(target, preds)
        ari = metrics.adjusted_rand_score(target, preds)
        f = metrics.fowlkes_mallows_score(target, preds)
        return {
            f"{self.prefix}/ibot_nmi": nmi,
            f"{self.prefix}/ibot_ari": ari,
            f"{self.prefix}/ibot_f": f,
        }


def eval_pred(label, pred, calc_acc=True):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    if not calc_acc:
        return nmi, ari, f, -1
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, f, acc


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(
        y_true, cluster_assignments, labels=None
    )
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


if __name__ == "__main__":
    from pytorch_lightning import seed_everything
    from rich import print as rprint
    from rich.traceback import install

    seed_everything(42)
    install()

    y_true = np.random.choice(2, 20)
    pred = np.random.choice(4, 20)
    rprint(eval_pred(y_true, pred, calc_acc=False))
