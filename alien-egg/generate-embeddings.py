#!/usr/bin/env python
# coding: utf-8
import sys
from typing import List
from ast import literal_eval
from pathlib import Path
from typing import Tuple
import json

import pandas as pd
from dask.distributed import Client
from dask.distributed import as_completed
import numpy as np
import msgpack

for path in [
    Path.home() / "Developer" / "stsievert" / "salmon",
    Path.home() / "salmon",
]:
    sys.path.append(str(path / "examples"))

from run import _X_test
import offline


def _check_version():
    import salmon

    assert "v0.5.0+6.g2" in salmon.__version__
    return True


def _literal_eval(x):
    try:
        return literal_eval(x)
    except ValueError:
        return x


def _get_dict(s: str) -> dict:
    k_v = [kv.split("=") for kv in s.split("-")]
    k_v2 = {k: _literal_eval(v) for k, v in k_v}
    return k_v2


def _ident(d: dict) -> str:
    d2 = sorted(tuple(d.items()))
    d3 = [f"{k}={v}" for k, v in d2]
    return "-".join(d3)


def _get_responses(f: Path) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(f)
    ident = f.name.replace("responses:", "").replace(".csv", "")


def _serialize(d):
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.float):
        return float(d)
    if isinstance(d, list):
        return [_serialize(_) for _ in d]
    if isinstance(d, dict):
        return {k: _serialize(v) for k, v in d.items()}
    return d


def _prep():
    import os
    import torch

    threads = int(os.environ.get("OMP_NUM_THREADS", 5))
    torch.set_num_threads(threads)
    return threads


if __name__ == "__main__":

    DIR = Path("io/2021-02-16/responses")
    noise = "human"
    n = 30
    dfs = {
        f.name.replace(".csv", ""): pd.read_csv(f)
        for f in DIR.glob("*.csv")
        if "test" not in f.name
    }
    assert len(dfs) == 2

    keys = [k for k in dfs.keys() if "alg=RandomSampling" in k]
    assert len(keys) == 1, keys
    key = keys[0]
    print([len(df) for df in dfs.values()])

    ## Get test set from random responses
    #  test_size = int(0.2 * len(dfs[key]))
    #  dfs["test"] = dfs[key].iloc[-test_size:].copy()
    #  dfs[key] = dfs[key].iloc[:-test_size].copy()

    print([len(df) for df in dfs.values()])

    ## Set same initial questions
    limit = 10 * n
    initial = dfs[key].iloc[:limit]
    for k, df in dfs.items():
        if k == "test":
            continue
        df.iloc[:limit] = initial.copy()
    assert all((df.iloc[:limit].score <= -1000).all() for df in dfs.values())

    cols = ["head", "winner", "loser"]
    datasets = {k: df[cols].to_numpy() for k, df in dfs.items()}
    print([len(v) for v in datasets.values()])
    X_test = _X_test(n, num=20_000, noise=noise)
    print([len(v) for v in datasets.values()])
    assert len(datasets) == 2

    NUM_ANS = [
        300,
        500,
        750,
        1250,
        1500,
        1750,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        6000,
        7000,
        8000,
        9000,
        10_000,
    ]

    _randoms = [k for k in dfs.keys() if "alg=RandomSampling" in k]
    assert len(_randoms) == 1
    _random = _randoms[0]
    #  datasets[_random] = _X_test(n, num=30_000, seed=10**6)
    print([len(v) for v in datasets.values()])

    #  offline._get_trained_model(
    #  datasets[_random],
    #  n_responses=10_000,
    #  meta=_get_dict(_random),
    #  shuffle=True,
    #  noise_model="GNMDS",
    #  **static,
    #  )
    #  sys.exit(1)
    #  breakpoint()

    client = Client("localhost:8786")
    client.upload_file("offline.py")
    d = client.run(_check_version)
    assert all(list(d.values()))

    d = client.run(_prep)
    d = list(d.values())
    assert all(d), d
    print(d)
    threads = d[0]
    static = dict(X_test=X_test, d=2, max_epochs=500_000, threads=threads, dwell=500)

    r_dataset = client.scatter(datasets[_random])
    random_futures = [
        client.submit(
            offline._get_trained_model,
            r_dataset,
            n_responses=n_ans,
            meta=_get_dict(_random),
            shuffle=True,
            noise_model=nm,
            ident=f"random-{nm}",
            **static,
        )
        for n_ans in NUM_ANS
        for nm in ["TSTE", "SOE", "CKL", "GNMDS"]
    ]

    _actives = [k for k in dfs.keys() if "alg=RR" in k]
    assert len(_actives) == 1, len(_actives)
    _active = _actives[0]
    a_dataset = client.scatter(datasets[_active])
    active_futures = [
        client.submit(
            offline._get_trained_model,
            a_dataset,
            n_responses=n_ans,
            meta={"noise_model": nm, **_get_dict(_active)},
            noise_model=nm,
            shuffle=True,
            ident=f"active-{nm}",
            **static,
        )
        for n_ans in NUM_ANS
        for nm in ["TSTE", "SOE", "CKL", "GNMDS"]
    ]

    futures = random_futures + active_futures

    for i, future in enumerate(as_completed(futures)):
        est, meta = future.result()
        meta = _serialize(meta)
        fname = _ident(meta) + ".csv"
        save = {
            "embedding": est.embedding_.tolist(),
            "meta": meta,
            "performance": _serialize(est.history_[-1]),
            "history": _serialize(est.history_),
        }
        _show = {k: est.history_[-1][k] for k in ["score_test", "loss_test"]}
        show = {k: f"{v:0.3f}" for k, v in _show.items()}
        print(i, meta["alg"], meta["ident"], meta["n_train"], show)
        with open(f"/scratch/ssievert/io/{i}.msgpack", "wb") as f2:
            msgpack.dump(save, f2)
