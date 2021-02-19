from numbers import Number as NumberType
from typing import Any, Dict, List, Union

import numpy as np
import numpy.linalg as LA
import torch
from salmon.triplets.algs import CKL as RunnerCKL
from salmon.triplets.algs.adaptive import CKL
from scipy.spatial import procrustes

Number = Union[int, float, NumberType]


def collect(
    embedding: List[List[float]], X: np.ndarray
) -> Dict[str, Union[Number, str]]:
    em = np.asarray(embedding)
    X = np.asarray(X)
    n, d = em.shape

    acc = _get_acc(em, X)
    probs = _get_probs(em, X)
    sim_stats = similarity(embedding, np.load("io/low-dim-features.npy"))
    return {"accuracy": acc, "prob_avg": probs, "prob_cls": "CKL", **sim_stats}


def similarity(est: np.ndarray, truth: np.ndarray) -> Dict[str, Number]:
    T, E, disparity = procrustes(truth, est)
    rel_error = LA.norm(T - E) / LA.norm(T)
    return {
        "disparity": disparity,
        "em_rel_error": rel_error,
        "em_error": LA.norm(E - T),
        "norm_E": LA.norm(E),
        "norm_T": LA.norm(T),
        "norm_e": LA.norm(est),
        "norm_t": LA.norm(truth),
    }


def _get_probs(embedding: np.ndarray, X: np.ndarray) -> float:
    n, d = embedding.shape
    est = CKL(n, d)
    est._embedding.data = torch.from_numpy(embedding.astype("float32"))
    win2, lose2 = est._get_dists(X)
    probs = est.probs(win2, lose2)
    return float(probs.mean().item())


def _get_acc(embedding: np.ndarray, X: np.ndarray) -> float:
    assert isinstance(embedding, np.ndarray)
    n, d = embedding.shape
    # X[i] is always [h, w, l] so zero is the right choice.
    y = np.zeros(len(X)).astype("int")
    assert X.ndim == 2 and X.shape[1] == 3, f"{type(X)}, {X.shape}"
    est = RunnerCKL(n, d)
    y_hat = est.predict(X, embedding=embedding)
    assert all(_.dtype.kind in ["u", "i"] for _ in [y, y_hat])
    acc = (y == y_hat).mean()
    return acc
