import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict
from distributed import Client, as_completed
from copy import deepcopy
import random

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import msgpack

import response_model
import targets
from responses import _cook_next

from stats_next import collect

SAVE_DIR = Path("/scratch/ssievert/salmon-final-next/out/")


def random_responses(T: List[int], n=90, num=20_000, random_state=42) -> np.ndarray:
    X = []
    rng = np.random.RandomState(random_state)
    for _ in range(num):
        i_h, i_l, i_r = _random_query(n, rng)
        h, l, r = T[i_h], T[i_l], T[i_r]
        winner = response_model.alien_egg(h, l, r, random_state=rng)
        assert winner in [0, 1]
        ans = [i_h, i_l, i_r] if winner == 0 else [i_h, i_r, i_l]
        X.append(ans)
    return np.array(X).astype(int)


def _X_test(T: List[int]) -> np.ndarray:
    n = 90
    return random_responses(T, num=60_000, random_state=42)


def _X_test_viz(T: List[int], random_state=42 ** 2) -> np.ndarray:
    n = 90
    v0 = _X_test(T)

    rng = np.random.RandomState(random_state)
    _T = np.array(T).astype(int)

    #  idx = rng.choice(len(T), size=(60_000, 3)).astype(int)
    idx = rng.choice(len(T), size=(1_000_000, 3)).astype(int)
    h, w, l = idx[:, 0], idx[:, 1], idx[:, 2]
    repeats = (h == w) | (h == l) | (w == l)
    idx = idx[~repeats]
    for k, (i_h, i_0, i_1) in enumerate(idx):
        winner = response_model.alien_egg(T[i_h], T[i_0], T[i_1])
        if winner == 1:
            idx[k, 1], idx[k, 2] = i_1, i_0

    return np.vstack((v0, idx))


def _random_query(n: int, random_state) -> Tuple[int, int, int]:
    rng = check_random_state(random_state)
    h, a, b = rng.choice(n, size=3, replace=False).astype(int)
    return h, a, b


def fit_estimator(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n: int,
    num_ans: int,
    sampling,
    noise_model: str,
    d: int = 2,
    max_epochs: int = 1_000_000,
    shuffle_seed=None,
    meta=None,
    targets=None,
    ident_k=None,
    **kwargs,
) -> Tuple["OfflineEmbedding", Dict[str, Any]]:
    from salmon.triplets.offline import OfflineEmbedding

    assert targets is not None
    assert ident_k is not None
    import torch

    torch.set_num_threads(1)

    from salmon.triplets.offline import OfflineEmbedding

    if meta is None:
        meta = {}
    assert sampling in [None, "next", "salmon", "random", "salmon-tste"], sampling
    if sampling == "random":
        assert shuffle_seed is not None
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(X_train)
    X_train = X_train[:num_ans]
    assert X_train.shape == (num_ans, 3)
    X_train_minimal = X_train.astype("int8")
    X_test_minimal = X_test.astype("int8")
    assert np.allclose(X_train_minimal, X_train)
    assert np.allclose(X_test_minimal, X_test)
    X_test = X_test_minimal
    X_train = X_train_minimal

    assert all(arr.dtype.name == "int8" for arr in [X_train, X_test])

    params = dict(
        n=n,
        d=d,
        random_state=701,
        noise_model=noise_model,
        initial_batch_size=1 * 128,
        **kwargs,
    )

    n_initial = min(10 * n, len(X_train))
    initial = OfflineEmbedding(max_epochs=min(100_000, max_epochs), **params)
    initial.fit(
        X_train[:n_initial], X_test, get_stats=deepcopy(collect), targets=targets
    )

    est = OfflineEmbedding(max_epochs=max_epochs, **params)
    est.fit(
        X_train,
        X_test,
        get_stats=deepcopy(collect),
        targets=targets,
        embedding=initial.embedding_,
    )

    est_kwargs = {f"est__{k}": v for k, v in kwargs.items()}
    meta = {f"meta__{k}": v for k, v in meta.items()}
    ret_dict = {
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
        "ident_k": ident_k,
        **est_kwargs,
        **meta,
    }
    return est, ret_dict


def _get_kwargs(nm: str) -> Dict[str, float]:
    # From Sec 3.2/bottom of page 6 of the NEXT paper
    # https://papers.nips.cc/paper/2015/file/89ae0fe22c47d374bc9350ef99e01685-Paper.pdf
    if nm == "CKL":
        return {"module__mu": 0.05}
    elif nm in ["TSTE", "SOE"]:
        return {}
    raise ValueError(f"nm={nm} not in ['CKL', 'TSTE', 'SOE']")


def _get_num_ans(n_answers, n, lower=None):
    num_ans = [5 * n]
    num_ans += list(range(10 * n, 100 * n, 10 * n))
    num_ans += list(range(100 * n, 300 * n, 30 * n))
    num_ans += list(range(300 * n, 1000 * n, 100 * n))
    num_ans = [n for n in num_ans if n <= n_answers]
    if lower:
        num_ans = [n for n in num_ans if lower <= n]
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

    assert "v0.7.0rc4+19" in salmon.__version__, (salmon.__version__, salmon.__file__)
    return True


if __name__ == "__main__":
    assert SAVE_DIR.exists() and SAVE_DIR.is_dir()
    DEBUG = False
    n = 90
    T = targets.get(n)
    X_test = _X_test(T)
    dl = np.abs(X_test[:, 0] - X_test[:, 1])
    dr = np.abs(X_test[:, 0] - X_test[:, 2])
    correct = (dl < dr).mean()
    print(f"Correct {100 * correct}% of the time")

    # Generate random responses
    R = random_responses(T, num=30_000, random_state=2032 ** 2)
    N_ANS = _get_num_ans(len(R), n)
    MAX_EPOCHS = 4_000_000
    static = dict(
        X_test=X_test, n=n, verbose=MAX_EPOCHS // 100, max_epochs=MAX_EPOCHS, targets=T
    )

    if DEBUG:
        N_ANS = N_ANS[:10]
        static["max_epochs"] = 2000
        static["verbose"] = 2000
    random_futures = [
        dict(
            X_train=R,
            num_ans=num_ans,
            shuffle_seed=seed,
            sampling="random",
            noise_model=nm,
            meta={"fname": "random", "sampling_seed": seed, "alg": "random"},
            d=d,
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "TSTE"]
        for num_ans in N_ANS
        for seed in range(5)
        for d in [2]
    ]

    def _get_train_data(p: Path) -> np.ndarray:
        raw = msgpack.loads(p.read_bytes(), raw=False)
        df = _cook_next(raw)
        return df[["head", "winner", "loser"]].to_numpy().astype(int)

    X_nexts = {
        f: _get_train_data(f)
        for f in [Path("next/io/2021-05-24/") / "rate=1_responses.msgpack"]
    }
    assert len(X_nexts) == 1 and {f.name for f in X_nexts} == {
        "rate=1_responses.msgpack"
    }
    next_futures = [
        dict(
            X_train=X_train,
            num_ans=num_ans,
            sampling="next",
            noise_model=nm,
            d=d,
            meta={"alg": "TSTE", "vary": "rate", "fname": str(f)},
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "TSTE"]
        for f, X_train in X_nexts.items()
        for num_ans in _get_num_ans(len(X_train), n)
        for d in [2]
    ]

    X_salmons = {
        f: pd.read_csv(f)[["head", "winner", "loser"]].to_numpy().astype(int)
        #  for f in Path("salmon/io/2021-05-25/").glob("ARR-1_responses.csv.zip")
        for f in Path("salmon/io/2021-07-01-arr-search/").glob("*1_responses.csv.zip")
    }
    assert len(X_salmons) == 12

    salmon_rates = [
        dict(
            X_train=X_train,
            num_ans=num_ans,
            sampling="salmon",
            noise_model=nm,
            d=d,
            meta={"alg": "ARR", "vary": "rate", "fname": str(f)},
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "TSTE"]
        for f, X_train in X_salmons.items()
        for num_ans in _get_num_ans(len(X_train), n)
        for d in [2]
    ]
    X_salmon_tse = {
        f: pd.read_csv(f)[["head", "winner", "loser"]].to_numpy().astype(int)
        for f in Path("salmon/io/2021-05-26-search/").glob("*1_responses.csv.zip")
    }
    salmon_searches = [
        dict(
            X_train=X_train,
            num_ans=num_ans,
            sampling="salmon-tste",
            noise_model=nm,
            d=d,
            meta={"alg": "TSTE", "vary": "search", "fname": str(f)},
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "TSTE"]
        for f, X_train in X_salmon_tse.items()
        for num_ans in _get_num_ans(len(X_train), n)
        for d in [2]
    ]

    for name, job in [
        ("salmon", salmon_searches),
        ("salmon-rates", salmon_rates),
        ("next", next_futures),
        ("random", random_futures),
    ]:
        print(f"len({name}_job) =", len(job))
    job_kwargs = salmon_searches + salmon_rates + next_futures + random_futures

    for k, j in enumerate(job_kwargs):
        j["ident_k"] = k

    from _generate_viz_data import _get_embeddings

    hist_v0 = _get_embeddings("_io/embeddings-v0.zip")

    print("\nlen(TOTAL_JOBS) =", len(job_kwargs))

    def _get_priority(d: Dict[str, Any]) -> float:
        p = 1 / (1 + d["num_ans"] / 4000)
        assert 0 <= p <= 1, f"p={p:0.3e} not in [0, 1]"
        base = 4

        if d["noise_model"] == "CKL":
            p *= base

        assert d["d"] == 2
        fname = d["meta"].get("fname", "")

        if d["meta"]["alg"] == "ARR" and "n_top=1-" in fname:
            p *= base

            if "n_search=352" in fname:
                p *= base

        if d["meta"]["alg"] == "ARR" and (
            "n_search=1000-" in fname or "n_search=3000-" in fname
        ):
            p *= base

        if d["meta"]["alg"] == "TSTE" and any(
            f"n_search={n}" in fname for n in [30, 100, 300, "1k", "3k"]
        ):
            p *= base / 2

        if d["sampling"] == "random" and d["meta"]["sampling_seed"] == 1:
            p *= base / 2

        p += 100
        return p

    job_kwargs = list(sorted(job_kwargs, key=lambda d: -1 * _get_priority(d)))
    keys = ["noise_model", "sampling", "num_ans"]  # , "meta"]
    show = [{k: j[k] for k in keys} for j in job_kwargs]
    for s, j in zip(show, job_kwargs):
        s.update(j["meta"])

    print("with all jobs: len(job_kwargs) =", len(job_kwargs))
    already_run = [f.name for f in SAVE_DIR.glob("*.msgpack")]
    assert len(already_run) == 89
    job_kwargs = [j for j in job_kwargs if f"{j['ident_k']}.msgpack" not in already_run]
    print("after filtering for already run jobs: len(job_kwargs) =", len(job_kwargs))

    print(f"Starting to submit {len(job_kwargs)} jobs...")

    client = Client("localhost:8786")
    d = client.run(_check_version)
    assert all(list(d.values()))
    client.upload_file("stats_next.py")

    if DEBUG:
        random.shuffle(job_kwargs)
        job_kwargs = job_kwargs[:100]
    futures = []
    for k, kwargs in enumerate(job_kwargs):
        kwargs["X_train"] = client.scatter(kwargs["X_train"])
        kwargs["X_test"] = client.scatter(kwargs["X_test"])
        future = client.submit(fit_estimator, **kwargs, priority=len(job_kwargs) - k)
        futures.append(future)
    print("len(futures) =", len(futures))

    for i, future in enumerate(as_completed(futures)):
        est, meta = future.result()
        meta = _serialize(meta)
        save = {
            "embedding": est.embedding_.tolist(),
            "meta": meta,
            "performance": _serialize(est.history_[-1]),
            "history": _serialize(est.history_),
        }
        _keys = ["noise_model", "sampling", "num_ans", "meta__alg"]
        print(f"{i} / {len(job_kwargs)}", {k: meta.get(k, "") for k in _keys})
        fname = meta["ident_k"]
        with open(SAVE_DIR / f"{fname}.msgpack", "wb") as f2:
            msgpack.dump(save, f2)
