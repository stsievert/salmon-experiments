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
from typing import Tuple, Union, Dict
from numbers import Number as NumberType

import numpy as np
from scipy.spatial import procrustes
from sklearn.manifold import SpectralEmbedding

import salmon.triplets.algs.adaptive.search.gram_utils as gram_utils
from salmon.triplets.offline import OfflineEmbedding
from salmon.triplets.algs import TSTE

ArrayLike = Union[list, np.ndarray]


def collect(embedding: ArrayLike, X_test: ArrayLike) -> Dict[str, float]:
    embedding = np.asarray(embedding)
    X_test = np.asarray(X_test)

    accuracy = _get_acc(embedding, X_test)
    nn_acc, nn_diffs = _get_nn_diffs(embedding)

    diff_stats = {
        f"nn_diff_p{k}": np.percentile(nn_diffs, k)
        for k in [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]
    }

    n, d = embedding.shape
    stats = {}
    if d > 1:
        reduce = SpectralEmbedding(n_components=1, affinity="rbf")
        embedding = reduce.fit_transform(embedding)
    norm = np.linalg.norm
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
    }


def _get_acc(embedding: np.ndarray, X: np.ndarray) -> float:
    assert isinstance(embedding, np.ndarray)
    n, d = embedding.shape
    # X[i] is always [h, w, l] so zero is the right choice.
    y = np.zeros(len(X)).astype("int")
    assert X.ndim == 2 and X.shape[1] == 3, f"{type(X)}, {X.shape}"
    est = TSTE(n, d)
    y_hat = est.predict(X, embedding=embedding)
    assert all(_.dtype.kind in ["u", "i"] for _ in [y, y_hat])
    acc = (y == y_hat).mean()
    return acc


def _get_nn_diffs(embedding) -> Tuple[float, np.ndarray]:
    dists = gram_utils.distances(gram_utils.gram_matrix(embedding))
    dists[dists <= 0] = np.inf
    neighbors = dists.argmin(axis=1)
    neighbor_dists = np.abs(neighbors - np.arange(len(neighbors)))
    nn_acc = (neighbor_dists == 1).mean()
    return nn_acc, neighbor_dists
