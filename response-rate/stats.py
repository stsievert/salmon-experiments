"""
This file is a collection of various stat functions.

Input: embedding filenames.
Output: stats for each embedding.

The following stats are collected:

* Accuracy
* Distance from ground truth embedding
* Nearest neighbor accuracy

"""

from time import time
from typing import Tuple, Union, Dict, List
from numbers import Number as NumberType

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import SpectralEmbedding
import numpy.linalg as LA

import salmon.triplets.algs.adaptive.search.gram_utils as gram_utils
from salmon.triplets.offline import OfflineEmbedding
from salmon.triplets.algs import TSTE

ArrayLike = Union[list, np.ndarray]
Number = Union[NumberType, int, float, np.integer, np.floating]


def collect(
    embedding: ArrayLike, targets: List[int], X_test: ArrayLike
) -> Dict[str, float]:
    embedding = np.asarray(embedding)
    X_test = np.asarray(X_test)

    accuracy = _get_acc(embedding, X_test)
    nn_acc, nn_diffs = _get_nn_diffs(embedding, targets)

    diff_stats = {
        f"nn_diff_p{k}": np.percentile(nn_diffs, k)
        for k in [99, 98, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2, 1]
    }
    nn_dists = {f"nn_acc_radius_{k}": (nn_diffs <= k).mean() for k in range(30)}

    n, d = embedding.shape
    stats = {}
    if d > 1:
        reduce = SpectralEmbedding(n_components=1, affinity="nearest_neighbors")
        embedding = reduce.fit_transform(embedding)
    norm = np.linalg.norm
    if targets:
        ground_truth = np.array(targets)
        assert (np.diff(ground_truth) > 0).all()
        ground_truth = ground_truth.reshape(-1, 1)
    else:
        ground_truth = np.arange(n).reshape(-1, 1)
    Y1, Y2, disparity = procrustes(ground_truth, embedding)
    stats = {
        "embedding_error": norm(Y1 - Y2),
        "embedding_rel_error": norm(Y1 - Y2) / norm(Y1),
        "procrustes_disparity": disparity,
    }

    return {
        "accuracy": accuracy,
        "nn_diff_median": np.median(nn_diffs),
        "nn_diff_mean": nn_diffs.mean(),
        "nn_acc": nn_acc,
        **diff_stats,
        **stats,
        **nn_dists,
        **_dist_stats(ground_truth, embedding),
    }


def _dist_stats(ground_truth: np.ndarray, em: np.ndarray) -> Dict[str, Number]:
    D_star = pdist(ground_truth)
    D_hat = pdist(em)
    D_star /= D_star.max()
    D_hat /= D_hat.max()
    return {"dist_rel_error": LA.norm(D_hat - D_star) / LA.norm(D_star)}


def _get_acc(embedding: np.ndarray, X: np.ndarray) -> float:
    assert isinstance(embedding, np.ndarray)
    n, d = embedding.shape
    # X[i] is always [h, w, l] so zero is the right choice.
    y = np.zeros(len(X)).astype("int")
    assert X.ndim == 2 and X.shape[1] == 3, f"{type(X)}, {X.shape}"
    est = TSTE(n=n, d=d)
    y_hat = est.predict(X, embedding=embedding)
    assert all(_.dtype.kind in ["u", "i"] for _ in [y, y_hat])
    acc = (y == y_hat).mean()
    return acc


def _get_nn_diffs(embedding, targets: List[int]) -> Tuple[float, np.ndarray]:
    """
    Get the NN accuracy and the number of objects that are closer than the
    true NN.
    """
    true_nns = []
    t = np.array(targets)
    for ti in targets:
        true_dist = np.abs(t - ti).astype("float32")
        true_dist[true_dist <= 0] = np.inf
        true_nns.append(true_dist.argmin())
    true_nns = np.array(true_nns).astype("int")

    dists = gram_utils.distances(gram_utils.gram_matrix(embedding))
    dists[dists <= 0] = np.inf

    neighbors = dists.argmin(axis=1)
    neighbor_dists = np.abs(neighbors - true_nns)
    nn_acc = (neighbor_dists == 0).mean()
    return nn_acc, neighbor_dists
