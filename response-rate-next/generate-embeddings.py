from pathlib import Path
from typing import Tuple, List, Any, Dict
from distributed import Client, as_completed

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import msgpack

import response_model
import targets
from salmon.triplets.offline import OfflineEmbedding
from responses import _cook_next


def random_responses(T: List[int], num=20_000, random_state=42) -> np.ndarray:
    X = []
    rng = np.random.RandomState(42)
    for _ in range(num):
        i_h, i_l, i_r = _random_query(n, rng)
        h, l, r = T[i_h], T[i_l], T[i_r]
        winner = response_model.alien_egg(h, l, r, random_state=rng)
        assert winner in [0, 1]
        ans = [i_h, i_l, i_r] if winner == 0 else [i_h, i_r, i_l]
        X.append(ans)
    return np.array(X).astype(int)


def _X_test(T: List[int]) -> np.ndarray:
    return random_responses(T, num=60_000, random_state=42)


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
) -> Tuple[OfflineEmbedding, Dict[str, Any]]:
    if meta is None:
        meta = {}
    assert sampling is not None
    if sampling == "random":
        assert shuffle_seed is not None
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(X_train)
    X_train = X_train[:num_ans]
    assert X_train.shape == (num_ans, 3)

    est = OfflineEmbedding(n=n, d=d, random_state=400, max_epochs=max_epochs, **kwargs)
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
    base = 10 * n
    #  num_ans = list(range(1 * n, 10 * n, n))
    num_ans = list(range(6 * n, 30 * n, 3 * n))
    num_ans += list(range(30 * n, 100 * n, 10 * n))
    num_ans += list(range(100 * n, 300 * n, 30 * n))
    num_ans += list(range(300 * n, 1_000 * n, 100 * n))
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

    assert "v0.6.1rc3+7" in salmon.__version__
    return True


if __name__ == "__main__":
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
    N_ANS = N_ANS[-10:]

    #  static = dict(X_test=X_test, n=n, d=2, max_epochs=2)  # TODO: change max_epochs
    static = dict(X_test=X_test, n=n, d=2, max_epochs=4_000_000)
    random_futures = [
        dict(
            X_train=R,
            num_ans=num_ans,
            shuffle_seed=seed,
            sampling="random",
            noise_model=nm,
            meta={"sampling_seed": seed, "alg": "random"},
            **_get_kwargs(nm),
            **static,
        )
        for nm in ["CKL", "SOE", "TSTE"]
        for num_ans in N_ANS
        for seed in range(20)
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

    for job in [salmon_searches, salmon_rates, next_futures, random_futures]:
        print("len(job) =", len(job))
    job_kwargs = salmon_searches + salmon_rates + next_futures + random_futures
    print("len(job_kwargs) =", len(job_kwargs))

    def _get_priority(d: Dict[str, Any]) -> float:
        p = 20_000 / d["num_ans"]
        if d["noise_model"] == "CKL":
            p *= 10
        if "n_search=1" in d["meta"].get("fname", ""):
            p /= 100
        return p

    job_kwargs = list(sorted(job_kwargs, key=lambda d: -1 * _get_priority(d)))
    keys = ["noise_model", "sampling", "num_ans"]  # , "meta"]
    show = [{k: j[k] for k in keys} for j in job_kwargs]
    for s, j in zip(show, job_kwargs):
        s.update(j["meta"])
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
            f"/scratch/ssievert/next-comparison/io-cluster/{i}.msgpack", "wb"
        ) as f2:
            msgpack.dump(save, f2)
