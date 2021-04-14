from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
import numpy as np
from dask.distributed import Future, Client, get_client, as_completed
import msgpack
from zipfile import ZipFile, ZIP_LZMA
from io import BytesIO
from typing import Tuple, Dict, Any

from salmon.triplets.offline import OfflineEmbedding


def _check_version():
    import salmon

    assert "v0.6.0+25" in salmon.__version__
    return salmon.__version__


def _get_responses(
    p: Path, n: int, shuffle: bool = True, seed: int = 42, length=None,
) -> np.ndarray:
    if p.is_dir():
        p = p / "responses.csv.zip"
    if not p.exists():
        raise FileNotFoundError(f"No such file '{p}'")
    if "csv" in p.name:
        df = pd.read_csv(p)
    elif "parquet" in p.name:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Filetype of {p} not CSV or Parquet")
    cols = ["head", "winner", "loser"]
    responses = df[cols].to_numpy().astype("int16")
    assert 0 == responses.min() < responses.max() == n - 1
    assert len(np.unique(responses)) == n
    if length:
        assert len(responses) >= length
    return responses


def _generate_embedding(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    num_ans: int,
    d: int = 2,
    seed: int = 42,
    shuffle: bool = False,
) -> Dict[str, Any]:
    return {
        "d": d,
        "shuffle": shuffle,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed": seed,
    }


def _get_num_response(n, limit=None):
    # 10 * n up to 130 * n (starting at 10 * n)
    num_ans = [i * n for i in range(1, 10, 1)]
    num_ans += [i * n for i in range(10, 20, 2)]
    num_ans += [i * n for i in range(20, 50, 5)]
    num_ans += [i * n for i in range(50, 100, 10)]
    num_ans += [i * n for i in range(100, 200, 20)]
    num_ans += [i * n for i in range(200, 500, 50)]
    num_ans += [i * n for i in range(500, 1000, 100)]
    num_ans += [i * n for i in range(1000, 2000, 200)]
    if limit:
        num_ans = [n for n in num_ans if n <= limit]
    return num_ans


def _get_estimator(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n: int,
    d: int = 2,
    num_ans: int,
    seed=None,
    sampling=None,
    max_epochs=2_000_000,
    **kwargs,
) -> Tuple[OfflineEmbedding, Dict[str, Any]]:
    assert sampling is not None
    if seed:
        rng = np.random.RandomState(seed)
        rng.shuffle(X_train)
    X_train = X_train[:num_ans]
    assert X_train.shape == (num_ans, 3)

    est = OfflineEmbedding(n=n, d=d, random_state=42, max_epochs=max_epochs, **kwargs)
    est.fit(X_train, X_test)
    est_kwargs = {f"est__{k}": v for k, v in kwargs.items()}
    ret_dict = {
        "n": n,
        "d": d,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed": seed,
        "num_ans": num_ans,
        "sampling": sampling,
        "est__random_state": est.random_state,
        **est_kwargs,
    }
    return est, ret_dict


def _launch_jobs(
    n: int,
    X_active: np.ndarray,
    X_random: np.ndarray,
    X_test: np.ndarray,
    d: int = 2,
    n_random=10,
) -> List[Future]:
    client = get_client()

    limits = {30: 10_000}

    active_num_ans = _get_num_response(n, limit=limits.get(n, len(X_active)))
    difficulty = np.round(10 * d * n * np.log(n)).astype(int)
    print("Active ratio:", max(active_num_ans) / difficulty, len(X_active))

    rand_num_ans = _get_num_response(n, limit=limits.get(n, len(X_random)))
    print("Random ratio:", max(rand_num_ans) / difficulty, len(X_random))

    kwargs = dict(n=n, d=d, X_test=X_test)
    X_active_f = client.scatter(X_active)
    active_kwargs = [
        {
            "X_train": X_active_f,
            "seed": None,
            "sampling": "active",
            "num_ans": num_ans,
            **kwargs,
        }
        for num_ans in active_num_ans
    ]

    X_random_f = client.scatter(X_random)
    random_kwargs = [
        {
            "X_train": X_random_f,
            "seed": i + 1,
            "num_ans": num_ans,
            "sampling": "random",
            **kwargs,
        }
        for i in range(n_random)
        for num_ans in rand_num_ans
    ]

    futures = client.map(lambda d: _get_estimator(**d), active_kwargs + random_kwargs)
    return futures


if __name__ == "__main__":
    TODAY = "2021-04-09"
    DIR = Path(f"io/{TODAY}/")
    N = [30, 90, 180, 300]
    DIRS = [DIR / f"n={n}" for n in N]
    assert all(d.exists() for d in DIRS)
    RANDOM_DIR = Path("io/random/train")
    TEST_DIR = Path("io/random/test")
    RANDOM = {n: RANDOM_DIR / f"n={n}-responses.parquet" for n in N}
    TEST = {n: TEST_DIR / f"n={n}-responses.parquet" for n in N}

    active = {n: _get_responses(p, n) for n, p in zip(N, DIRS)}
    random = {n: _get_responses(p, n) for n, p in RANDOM.items()}
    test = {n: _get_responses(p, n) for n, p in TEST.items()}

    client = Client("localhost:8786")
    _d = client.run(_check_version)
    print(set(_d.values()))
    assert all(list(_d.values()))

    futures = []
    for n in N:
        print(f"### n = {n}")
        _futures = _launch_jobs(n, active[n], random[n], test[n])
        futures.extend(_futures)
    print(len(futures), {type(f) for f in futures})

    zf_kwargs = dict(compression=ZIP_LZMA, compresslevel=9)
    OUT_DIR = Path("/scratch/ssievert/io/crowdsourcing")
    out_file = OUT_DIR / "embeddings.zip"
    with ZipFile(out_file, mode="w", **zf_kwargs) as zf:
        pass
    for k, future in enumerate(as_completed(futures)):
        est, meta = future.result()
        history = {k: [h.get(k, None) for h in est.history_] for k in est.history_[0]}
        to_save = {
            "embedding": est.embedding_.tolist(),
            "history": history,
            "perf": est.history_[-1],
            "params": est.get_params(),
            "meta": meta,
        }
        print(k, len(futures))
        with BytesIO() as f:
            msgpack.dump(to_save, f)
            out = f.getvalue()
        with ZipFile(out_file, mode="a", **zf_kwargs) as zf:
            zf.writestr(f"{k}.msgpack", out)
