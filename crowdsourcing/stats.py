from datetime import datetime
from functools import lru_cache
from numbers import Number as NumberType
from typing import Dict, Any, Optional, Union, List, Tuple
from zipfile import ZipFile
import numpy.linalg as LA

import json
import pandas as pd
import numpy as np

import targets as targets_module
import torch

datetime_parser = lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
ArrayLike = Union[list, np.ndarray]
Number = Union[NumberType, int, float, np.integer, np.floating]
Array = Union[np.ndarray, torch.Tensor]



def accuracy(
    embedding: ArrayLike, X: ArrayLike
) -> Dict[str, float]:
    y = np.zeros(len(X)).astype("int")
    y_hat = predict(X, embedding)
    return (y == y_hat).mean()

def predict(X, embedding):
    head_idx = X[:, 0].flatten()
    head = embedding[head_idx]

    left_idx = X[:, 1].flatten()
    left = embedding[left_idx]

    right_idx = X[:, 2].flatten()
    right = embedding[right_idx]

    ldiff = LA.norm(head - left, axis=1)
    rdiff = LA.norm(head - right, axis=1)

    # 1 if right closer; 0 if left closer
    # (which matches the labeling scheme)
    right_closer = rdiff < ldiff
    return right_closer.astype("uint8")

def _flatten_query(q: Dict[str, Any]) -> Optional[Dict[str, Union[int, datetime]]]:
    if "index_winner" not in q:
        return None
    sent = {
        d["label"] if d["label"] != "center" else "head": d["index"]
        for d in q["target_indices"]
    }
    t = q["timestamp_query_generated"]
    dt = q["response_time"]
    label = q["alg_label"]
    return {
        "winner": q["index_winner"],
        "timestamp": datetime_parser(t),
        "alg": label,
        "response_time": dt,
        **sent,
    }

def _munge(fname: str) -> pd.DataFrame:
    if "zip" in fname:
        with ZipFile(fname) as zf:
            assert len(zf.filelist) == 1
            with zf.open(zf.filelist[0]) as f:
                raw = json.load(f)
    else:
        with open(fname, "r") as f:
            raw = json.load(f)

    assert raw.pop("meta") == {"status": "OK", "code": 200}
    assert len(raw) == 1
    rare = raw["participant_responses"]
    mrare = sum(rare.values(), [])
    medium = [_flatten_query(d) for d in mrare]
    mwell = [m for m in medium if m]
    df = pd.DataFrame(mwell)
    cols = ["head", "left", "right", "winner", "alg", "timestamp", "response_time"]
    assert set(df.columns) == set(cols)
    df["loser"] = df[["head", "left", "right", "winner"]].apply(
        lambda r: r["left"] if r["winner"] == r["right"] else r["right"], axis=1
    )
    return df[cols + ["loser"]]


def _next_responses(alg: str, targets: List[str]) -> np.ndarray:
    df = _munge("io/next-fig3.json.zip")
    cols = ["head", "winner", "loser"]
    alg = alg or "Test"
    assert alg in df.alg.unique()
    X_next_idx = df.loc[df.alg == alg, cols].to_numpy()
    # fmt: off
    idx_fname = {
        0:  "i0126.png", 1:  "i0208.png", 2:  "i0076.png", 3:  "i0326.png",
        4:  "i0526.png", 5:  "i0322.png", 6:  "i0312.png", 7:  "i0036.png",
        8:  "i0414.png", 9:  "i0256.png", 10: "i0074.png", 11: "i0050.png",
        12: "i0470.png", 13: "i0022.png", 14: "i0430.png", 15: "i0254.png",
        16: "i0572.png", 17: "i0200.png", 18: "i0524.png", 19: "i0220.png",
        20: "i0438.png", 21: "i0454.png", 22: "i0112.png", 23: "i0494.png",
        24: "i0194.png", 25: "i0152.png", 26: "i0420.png", 27: "i0142.png",
        28: "i0114.png", 29: "i0184.png",
    }
    # fmt: on
    spikes = {
        next_idx: int(fname.strip("i.png")) for next_idx, fname in idx_fname.items()
    }

    # These are the NEXT targets by filename
    X_spikes = np.vectorize(spikes.get)(X_next_idx)

    assert len(targets) == len(idx_fname)
    spike_idx = {
        t_idx: salmon_idx
        for salmon_idx, t_idx in enumerate(targets)
    }
    # Make sure the spikes on both sides are the same
    assert set(spike_idx.keys()) == set(np.unique(X_spikes))
    assert set(spike_idx.keys()) == set(spikes.values())
    X_salmon = np.vectorize(spike_idx.get)(X_spikes)
    return X_salmon

@lru_cache()
def __X_test(targets) -> np.ndarray:
    return _next_responses("Test", targets)

def _X_test(targets):
    return __X_test(tuple(targets))

@lru_cache()
def __gt_responses(n, targets):
    return tuple([
        (h_i, r_i, l_i) if abs(h - r) < abs(h - l) else (h_i, l_i, r_i)
        for h_i, h in enumerate(targets)
        for r_i, r in enumerate(targets)
        for l_i, l in enumerate(targets)
        if h_i != l_i and h_i != r_i and r_i != l_i
    ])

def _gt_responses(n, targets):
    ret = __gt_responses(n, tuple(targets))
    return list(ret)

@lru_cache()
def __simulation_responses(n):
    df_sim = pd.read_parquet(f"io/random/test/n={n}-responses.parquet")
    sim = df_sim[["head", "winner", "loser"]].to_numpy()
    return sim

def test_responses(n: int) -> Dict[str, np.ndarray]:
    targets = targets_module.get(n)
    gt_responses = _gt_responses(n, tuple(targets))
    sim = __simulation_responses(n)
    if n == 30:
        human = _X_test(targets)
        return {"ground_truth": gt_responses, "simulation": sim, "human": human}
    return {"ground_truth": gt_responses, "simulation": sim}

def nn_accs(embedding: np.ndarray, targets: List[int]):
    nn_acc, nn_diffs = _get_nn_diffs(embedding, targets)

    diff_stats = {
        f"nn_diff_p{k}": np.percentile(nn_diffs, k)
        for k in [99, 98, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2, 1]
    }
    nn_dists = {f"nn_acc_radius_{k}": (nn_diffs <= k).mean() for k in range(30)}
    return nn_dists
def gram_matrix(X: Array) -> Array:
    """
    from salmon.triplets.samplers.adaptive.search.gram_utils import gram_matrix
    """
    if isinstance(X, torch.Tensor):
        return X @ X.transpose(0, 1)
    return X @ X.T


def distances(G: Array) -> Array:
    """
    from salmon.triplets.samplers.adaptive.search.gam_utils import distances
    """
#     assert_gram(G)
    G1 = np.diag(G).reshape(1, -1)
    G2 = np.diag(G).reshape(-1, 1)

    D = -2 * G + G1 + G2
    return D

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

    dists = distances(gram_matrix(embedding))
    dists[dists <= 0] = np.inf

    neighbors = dists.argmin(axis=1)
    neighbor_dists = np.abs(neighbors - true_nns)
    nn_acc = (neighbor_dists == 0).mean()
    return nn_acc, neighbor_dists