import os
from pathlib import Path
from typing import Tuple, List, Any, Dict, Set
from distributed import Client, as_completed
from copy import deepcopy
import sys
from zipfile import ZipFile
import json
from time import time

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import msgpack

import targets
from salmon.triplets.offline import OfflineEmbedding

DEBUG = False
OUT_DIR = "/scratch/ssievert/io/crowdsourcing-mturk/out"
#  if DEBUG:
    #  OUT_DIR = "./tmp"


def _print_fmt(v):
    if isinstance(v, (str, int)):
        return v
    if isinstance(v, (float, np.floating)):
        return f"{v:0.3f}"
    return v


def fit_estimator(
    *,
    ident: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    em_init: np.ndarray,
    n: int,
    d: int,
    num_ans: int,
    sampling,
    noise_model: str,
    max_epochs: int = 1_000_000,
    shuffle_seed=None,
    meta=None,
    **kwargs,
) -> Tuple[OfflineEmbedding, Dict[str, Any]]:
    import torch

    """
    max_epochs=1000, verbose=200, 60 workers
    n_threads=1: 9:55
        544: 3:45
        1152: 7:30
    n_threads=2: 15:39
        158: 2:00
        712: 7:30
        1156: 12:00

    """
    n_threads = os.environ.get("OMP_NUM_THREADS", 1)
    torch.set_num_threads(int(n_threads))

    from salmon.triplets.offline import OfflineEmbedding

    if meta is None:
        meta = {}
    assert sampling is not None
    if sampling == "random":
        assert shuffle_seed is not None
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(X_train)
    X_train = X_train[:num_ans]
    assert X_train.shape == (num_ans, 3)
    X_train_minimal = X_train.astype("int8")
    X_test_minimal = X_test.astype("int8")
    assert np.allclose(X_train_minimal, X_train), "diff = " + str(
        np.abs(X_train_minimal - X_train).sum()
    )
    assert np.allclose(X_test_minimal, X_test)
    X_test = X_test_minimal
    X_train = X_train_minimal

    assert all(arr.dtype.name == "int8" for arr in [X_train, X_test])
    est_kwargs = {f"est__{k}": v for k, v in kwargs.items()}
    meta = {f"meta__{k}": v for k, v in meta.items()}
    est = OfflineEmbedding(
        n=n,
        d=d,
        random_state=400,
        max_epochs=max_epochs,
        noise_model=noise_model,
        **kwargs,
    )
    meta = {
        "ident": ident,
        "len_X_train": len(X_train),
        "len_X_test": len(X_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n": n,
        "d": d,
        "num_ans": num_ans,
        "sampling": sampling,
        "shuffle_seed": shuffle_seed,
        "noise_model": noise_model,
        "num_ans": num_ans,
        "est__random_state": est.random_state,
        "est__max_epochs": est.max_epochs,
        "est__noise_model": est.noise_model,
        **est_kwargs,
        **meta,
    }
    est.fit(X_train, X_test, embedding=em_init)
    return est, meta


def _get_kwargs(nm: str) -> Dict[str, float]:
    # From Sec 3.2/bottom of page 6 of the NEXT paper
    # https://papers.nips.cc/paper/2015/file/89ae0fe22c47d374bc9350ef99e01685-Paper.pdf
    if nm == "CKL":
        return {"module__mu": 0.05}
    elif nm in ["TSTE", "SOE"]:
        return {}
    raise ValueError(f"nm={nm} not in ['CKL', 'TSTE', 'SOE']")


def _get_num_ans(n_answers, n, lower=None):
    num_ans = list(range(10 * n, 100 * n, 10 * n))
    num_ans += list(range(100 * n, 300 * n, 30 * n))
    num_ans += list(range(300 * n, 1000 * n, 100 * n))
    num_ans = [n for n in num_ans if n <= n_answers]
    if lower:
        num_ans = [n for n in num_ans if lower <= n]
    if num_ans and max(num_ans) < 0.95 * n_answers:
        num_ans += [n_answers]
    return num_ans


def _serialize(d):
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, (np.float64, float, np.floating)):
        return float(d)
    if isinstance(d, list):
        return [_serialize(_) for _ in d]
    if isinstance(d, dict):
        return {k: _serialize(v) for k, v in d.items()}
    return d


def _check_version():
    import salmon

    assert "v0.7.0rc2+11" in salmon.__version__
    return True


def _verify(df: pd.DataFrame, n) -> pd.DataFrame:
    assert df["alg_ident"].nunique() == 1
    _fnames = [set(df[f"{k}_filename"]) for k in ["left", "right", "head"]]
    fnames = _fnames[0].union(*_fnames)
    assert len(fnames) == n
    smoothness = list(sorted([int(f.strip("i.png")) for f in fnames]))
    assert len(set(smoothness)) == n
    assert 0 <= min(smoothness) < max(smoothness) <= 600
    assert all(s % 2 == 0 for s in smoothness)
    return True


def _get_responses(df):
    df = df.drop(
        columns=["timestamp", "alg_ident", "puid", "response_time", "network_latency"]
    )
    df = df.drop(columns=["left_filename", "right_filename"])
    if "score" in df.columns:
        df = df.drop(columns="score")
    assert {"head_filename", "winner_filename", "loser_filename"}.issubset(
        set(df.columns)
    )
    #  target_ids = {t: k for k, t in enumerate(T)}
    for k in ["head", "winner", "loser"]:
        df[f"{k}_param"] = df[f"{k}_filename"].str.strip("i.png").astype(int)
    #  for k in ["head", "winner", "loser"]:
    #  df[k] = df[f"{k}_param"].apply(target_ids.get)

    ret = df[["head", "winner", "loser"]].to_numpy()
    assert ret.max() == n - 1 and ret.min() == 0
    return ret.astype("int8")


def _get_priority(d: Dict[str, Any]) -> float:
    base = 1
    fname = d["meta"]["fname"]

    p = 1.0 / (1 + (d["num_ans"] / 4_000))
    assert 0 <= p <= 1

    if d["meta"]["n"] == 30:
        p += 2 * base

    if d["noise_model"] == "CKL":
        p += 10 * base
    elif d["noise_model"] == "SOE":
        p -= 10 * base
    else:
        assert d["noise_model"] == "TSTE"
    assert d["d"] in [1, 2]
    if d["d"] == 2:
        p += 5 * base

    if d["sampling"] == "active":
        p += 5 * base
    elif d["sampling"] == "random":
        if d["shuffle_seed"] == 3:
            p += 2 * base
    else:
        raise ValueError(d)
    return p


def _write(est, meta):
    meta = _serialize(meta)
    save = {
        "embedding": est.embedding_.tolist(),
        "meta": meta,
        "performance": _serialize(est.history_[-1]),
        "history": _serialize(est.history_),
    }

    with open(f"{OUT_DIR}/{meta['ident']}.msgpack", "wb") as f2:
        msgpack.dump(save, f2)


def _get_ground_truth(T: Set[int], num) -> np.ndarray:
    rng = np.random.RandomState(412)
    queries = [rng.choice(n, replace=False, size=3).tolist() for _ in range(num)]
    S = list(sorted([int(t.strip("i.png")) for t in T]))
    answers = [
        [h, l, r] if np.abs(S[h] - S[l]) < np.abs(S[h] - S[r]) else [h, r, l]
        for h, l, r in queries
    ]
    X = np.array(answers)

    dW = np.abs(X[:, 0] - X[:, 1])
    dL = np.abs(X[:, 0] - X[:, 2])
    correct = dW < dL

    # Not exact because X is indices, not actual targets
    assert correct.mean() >= 0.95
    return X


def _get_em_init(n: int, d: int) -> np.ndarray:
    with open(f"io/initial/n={n}-d={d}.json", "r") as f:
        raw = json.load(f)
    em = np.array(raw["em"])
    return em


def _prepare_jobs(n):
    DIR = Path("io/responses")
    X_trains = {}
    X_tests = {}
    randoms = {}
    for f in DIR.glob("*.csv"):
        if n == 90 and ("NEXT" in f.name or "n=30" in f.name):
            continue
        if n == 30 and "n=90" in f.name:
            continue
        df = pd.read_csv(f)

        assert set(df["head"].unique()) == set(range(n))
        assert _verify(df, n)

        X = _get_responses(df)
        print(f.name)
        if "test" in f.name.lower():
            X_tests[f.name.lower()] = X
        elif "random" in f.name.lower():
            randoms[f.name.lower()] = X
        else:
            X_trains[f.name.lower()] = X

    X_tests["ground_truth"] = _get_ground_truth(
        set(df["head_filename"].unique()), num=sum(len(x) for x in X_tests.values())
    )
    X_test = np.concatenate(list(X_tests.values()))
    X_random = np.concatenate(list(randoms.values()))
    random_fnames = "&".join(list(randoms.keys()))

    static = dict(X_test=X_test, verbose=10_000)
    MAX_EPOCHS = 10_000_000
    if DEBUG:
        MAX_EPOCHS = 1000
        static["verbose"] = 200

    em_inits = {(n, d): _get_em_init(n, d) for n in [30, 90] for d in [1, 2]}

    active_searches = [
        dict(
            X_train=X_train,
            num_ans=num_ans,
            sampling="active",
            noise_model=nm,
            meta={"fname": fname, "n": n},
            max_epochs=MAX_EPOCHS,
            em_init=_get_em_init(n, d),
            n=n,
            d=d,
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "SOE", "TSTE"]
        for fname, X_train in X_trains.items()
        for d in [1, 2]
        for num_ans in _get_num_ans(len(X_train), n)
    ]
    random_searches = [
        dict(
            X_train=X_random,
            num_ans=num_ans,
            noise_model=nm,
            meta={"fname": random_fnames, "n": n},
            max_epochs=MAX_EPOCHS // 5,
            sampling="random",
            shuffle_seed=k,
            em_init=_get_em_init(n, d),
            d=d,
            n=n,
            **_get_kwargs(nm),
            **static,
        )
        for k in range(5)
        for nm in ["CKL", "SOE", "TSTE"]
        for d in [1, 2]
        for num_ans in _get_num_ans(len(X_random), n)
    ]
    searches = active_searches + random_searches

    print(f"\nfor n={n}, len(TOTAL_JOBS) =", len(searches), "\n")

    return searches


if __name__ == "__main__":
    searches = []
    for n in [30, 90]:
        _searches = _prepare_jobs(n)
        searches.extend(_searches)

    for k, search in enumerate(searches):
        search["ident"] = k

    searches = list(sorted(searches, key=lambda d: -1 * _get_priority(d)))
    keys = ["num_ans", "noise_model", "meta"]
    show = [{k: j[k] for k in keys} for j in searches]

    print(f"\n\nStarting to submit {len(searches)} jobs...")
    client = Client("127.0.0.1:8786")
    d = client.run(_check_version)
    assert all(list(d.values()))
    futures = []
    for k, kwargs in enumerate(searches):
        future = client.submit(fit_estimator, **kwargs, priority=len(searches) - k)
        futures.append(future)
    print("len(futures) =", len(futures))

    for i, future in enumerate(as_completed(futures)):
        est, meta = future.result()
        _keys = ["noise_model", "sampling", "num_ans", "meta__alg"]
        print(f"{i} / {len(searches)}", {k: meta.get(k, "") for k in _keys})
        _write(est, meta)
