from pathlib import Path
from typing import Tuple, List, Any, Dict
from distributed import Client, as_completed
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import msgpack

import response_model
import targets
from responses import _cook_next


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

    idx = rng.choice(len(T), size=(60_000, 3)).astype(int)
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
    **kwargs,
) -> Tuple["OfflineEmbedding", Dict[str, Any]]:
    import torch

    torch.set_num_threads(1)

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
    assert np.allclose(X_train_minimal, X_train)
    assert np.allclose(X_test_minimal, X_test)
    X_test = X_test_minimal
    X_train = X_train_minimal

    assert all(arr.dtype.name == "int8" for arr in [X_train, X_test])
    est = OfflineEmbedding(
        n=n,
        d=d,
        random_state=400,
        max_epochs=max_epochs,
        noise_model=noise_model,
        **kwargs,
    )
    est.fit(X_train, X_test)

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
    num_ans = list(range(10 * n, 100 * n, 10 * n))
    num_ans += list(range(100 * n, 300 * n, 30 * n))
    num_ans += list(range(300 * n, 1000 * n, 100 * n))
    num_ans = [n for n in num_ans if n <= n_answers]
    if lower:
        num_ans = [n for n in num_ans if lower <= n]
    if max(num_ans) < 0.95 * n_answers:
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

    assert "v0.6.1rc3+9" in salmon.__version__
    return True


if __name__ == "__main__":
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
    static = dict(X_test=X_test, n=n, d=2, verbose=10_000)
    MAX_EPOCHS = 10_000_000
    R_EPOCHS = 1_000_000

    if DEBUG:
        N_ANS = N_ANS[-10:]
        static["max_epochs"] = 4
    random_futures = [
        dict(
            X_train=R,
            num_ans=num_ans,
            shuffle_seed=seed,
            sampling="random",
            noise_model=nm,
            meta={"sampling_seed": seed, "alg": "random"},
            max_epochs=R_EPOCHS,
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "SOE", "TSTE"]
        for num_ans in N_ANS
        for seed in range(10)
    ]

    def _get_train_data(p: Path) -> np.ndarray:
        raw = msgpack.loads(p.read_bytes(), raw=False)
        df = _cook_next(raw)
        return df[["head", "winner", "loser"]].to_numpy().astype(int)

    X_trains = {
        f: _get_train_data(f)
        for f in Path("next/io/2021-05-24/").glob("rate=*_responses.msgpack")
    }
    next_futures = [
        dict(
            X_train=X_train,
            num_ans=num_ans,
            sampling="next",
            noise_model=nm,
            max_epochs=MAX_EPOCHS,
            meta={"alg": "TSTE", "vary": "rate", "fname": str(f)},
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "SOE", "TSTE"]
        for f, X_train in X_trains.items()
        for num_ans in _get_num_ans(len(X_train), n)
    ]

    X_trains = {
        f: pd.read_csv(f)[["head", "winner", "loser"]].to_numpy().astype(int)
        for f in Path("salmon/io/2021-05-25/").glob("*responses.csv.zip")
    }
    salmon_rates = [
        dict(
            X_train=X_train,
            num_ans=num_ans,
            sampling="salmon",
            noise_model=nm,
            meta={"alg": "ARR", "vary": "rate", "fname": str(f)},
            max_epochs=MAX_EPOCHS,
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "SOE", "TSTE"]
        for f, X_train in X_trains.items()
        for num_ans in _get_num_ans(len(X_train), n)
    ]
    X_trains = {
        f: pd.read_csv(f)[["head", "winner", "loser"]].to_numpy().astype(int)
        for f in Path("salmon/io/2021-05-26-search/").glob("*responses.csv.zip")
    }
    salmon_searches = [
        dict(
            X_train=X_train,
            num_ans=num_ans,
            sampling="salmon",
            noise_model=nm,
            meta={"alg": "TSTE", "vary": "search", "fname": str(f)},
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "SOE", "TSTE"]
        for f, X_train in X_trains.items()
        for num_ans in _get_num_ans(len(X_train), n)
    ]

    for name, job in [
        ("salmon", salmon_searches),
        ("salmon-rates", salmon_rates),
        ("next", next_futures),
        ("random", random_futures),
    ]:
        print(f"len({name}_job) =", len(job))
    job_kwargs = salmon_searches + salmon_rates + next_futures + random_futures
    print("\nlen(TOTAL_JOBS) =", len(job_kwargs))

    def _get_priority(d: Dict[str, Any]) -> float:
        p = 20_000 / (1e-3 + d["num_ans"])
        base = 5
        if d["noise_model"] == "CKL":
            p *= base
        #  if d["meta"]["alg"] == "ARR" or d["sampling"] == "next":
        #  p *= base
        #  if "n_search=1" in d["meta"].get("fname", ""):
        #  p /= base ** 2
        if d["sampling"] == "random" and d["meta"]["sampling_seed"] != 1:
            p /= base ** 3
        return p

    job_kwargs = list(sorted(job_kwargs, key=lambda d: -1 * _get_priority(d)))
    keys = ["noise_model", "sampling", "num_ans"]  # , "meta"]
    show = [{k: j[k] for k in keys} for j in job_kwargs]
    for s, j in zip(show, job_kwargs):
        s.update(j["meta"])

    from zipfile import ZipFile

    def _get_already_run_jobs(fname: str, keys) -> List[Dict[str, Any]]:
        jobs = []
        with ZipFile(fname) as zf:
            for fname in zf.namelist():
                with zf.open(fname) as f:
                    raw = f.read()
                rare = msgpack.loads(raw, raw=False)
                mrare = rare["meta"]
                medium = {
                    k: mrare[k if "epochs" not in k else f"est__{k}"] for k in keys
                }
                medium["meta"] = {
                    k.replace("meta__", ""): mrare[k]
                    for k in mrare.keys()
                    if "meta__" in k
                }
                jobs.append(medium)
            return jobs

    ident_keys = ["num_ans", "sampling", "noise_model", "shuffle_seed"]
    already_run = _get_already_run_jobs("_io/embeddings-v1-save.zip", ident_keys)

    def _same_job(j1: Dict[str, Any], j2: Dict[str, Any]) -> bool:
        _j2 = {k: j2[k] for k in j2.keys() if "X_" not in k}
        j1 = deepcopy(j1)
        j2 = deepcopy(_j2)
        shuffle_seed = j1["shuffle_seed"]
        if shuffle_seed is None:
            j1.pop("shuffle_seed")
        else:
            if "shuffle_seed" not in j2:
                j2["shuffle_seed"] = j1["shuffle_seed"]

        assert set(j1.keys()).issubset(set(j2.keys()))
        for k in j1.keys():
            if j1[k] != j2[k]:
                return False
        return True

    to_run = [j for j in job_kwargs if not any(_same_job(ar, j) for ar in already_run)]
    print(f"{len(already_run)} jobs have finished")
    print(f"{len(to_run)} jobs will be launched")
    same_jobs = [j for j in to_run if any(_same_job(ar, j) for ar in already_run)]
    print(f"{len(same_jobs)} finished jobs are being re-submitted...", flush=True)
    assert not len(same_jobs)
    from time import sleep

    sleep(1)
    print(f"...that number is 0")
    print("Saving jobs to submit to disk to make sure...")
    job_kwargs = to_run
    with open("_to_submit.msgpack", "wb") as f:
        to_save = [{k: j.get(k, None) for k in ident_keys} for j in job_kwargs]
        msgpack.dump(to_save, f)

    print(f"Starting to submit those {len(job_kwargs)} jobs...")

    client = Client("localhost:8786")
    d = client.run(_check_version)
    assert all(list(d.values()))

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
        with open(
            f"/scratch/ssievert/next-comparison/io-cluster-v2/{i}-v2.msgpack", "wb"
        ) as f2:
            msgpack.dump(save, f2)