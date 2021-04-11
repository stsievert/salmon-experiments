from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
import numpy as np
from dask.distributed import Future, Client


def _get_responses(
    p: Path, n: int, shuffle: bool = True, seed: int = 42, length=19_700,
) -> np.ndarray:
    if p.is_dir():
        p = p / "responses.csv.zip"
    if not p.exists():
        raise FileNotFoundError(f"No such file '{p}'")
    df = pd.read_csv(p)
    cols = ["head", "winner", "loser"]
    responses = df[cols].to_numpy().astype("int16")
    assert 0 == responses.min() < responses.max() == n - 1
    assert len(np.unique(responses)) == n
    assert len(responses) >= length
    print(f"{len(responses)} responses for verified n={n}")
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
    if limit:
        num_ans = [n for n in num_ans if n <= limit]
    return num_ans


def _launch_jobs(
    n: int, active_resp: np.ndarray, random_resp: np.ndarray, d: int = 2,
) -> List[Future]:
    active_num_ans = _get_num_response(n, limit=len(active_resp))
    difficulty = np.round(10 * d * n * np.log(n)).astype(int)
    print("Active ratio:", max(active_num_ans) / difficulty)

    rand_num_ans = _get_num_response(n, limit=len(random_resp))
    print("Random ratio:", max(rand_num_ans) / difficulty)
    return [None]


if __name__ == "__main__":
    DIR = Path("io/2021-04-09/")
    N = [30, 90, 180, 300]
    DIRS = [DIR / f"n={n}" for n in N]
    assert all(d.exists() for d in DIRS)
    RANDOM_DIR = Path("io/random")
    RANDOM = {n: RANDOM_DIR / f"n={n}-responses.csv.zip" for n in N}

    responses = {n: _get_responses(p, n) for n, p in zip(N, DIRS)}
    random_responses = {
        n: _get_responses(p, n, length=70_000 if n >= 180 else 30_000)
        for n, p in RANDOM.items()
    }

    #  client = Client()
    for n in N:
        print(f"### n = {n}")
        _launch_jobs(n, responses[n], random_responses[n])

    # - [x] verify that responses are correctly downloaded
    # - [x] Collect random responses (at least 19700 / 0.8 == 24625)
    # - [ ] choose num_ans
