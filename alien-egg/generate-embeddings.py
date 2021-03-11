#!/usr/bin/env python
# coding: utf-8
import sys
from typing import List, Any, Dict, Union
from ast import literal_eval
from pathlib import Path
from typing import Tuple
import json
import yaml

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

import run
import offline


def _check_version():
    import salmon

    assert "v0.5.2+3" in salmon.__version__
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


def _flatten(d, prefix=""):
    if isinstance(d, (int, str, bytes, float)):
        return d
    elif isinstance(d, dict):
        return {k: _flatten(v, prefix=f"{k}__") for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        raise ValueError()
    raise ValueError()


def _get_kwargs(nm: str) -> Dict[str, Any]:
    # From section 3.2 of NEXT paper (last full paragraph on page 6)
    if nm == "CKL":
        return {"module__mu": 0.05}

    assert nm in ("SOE", "GNMDS", "TSTE")
    return {}


def _get_config(suffix: str):
    with open(DIR / f"config_{suffix}.yaml") as f:
        raw = yaml.safe_load(f)
    raw.pop("targets")
    rare = _flatten(raw)
    return rare

if __name__ == "__main__":

    DIR = Path("io/2021-03-09/")
    noise = "human"
    n = 30
    dfs = {
        f.name.replace(".csv", ""): pd.read_csv(f)
        for f in DIR.glob("responses_*.csv")
        if "test" not in f.name
    }
    print([len(df) for df in dfs.values()])
    assert len(dfs) == 2
    assert set(dfs.keys()) == {"responses_RR", "responses_RandomSampling"}

    ## Get test set from random responses
    #  test_size = int(0.2 * len(dfs[key]))
    #  dfs["test"] = dfs[key].iloc[-test_size:].copy()
    #  dfs[key] = dfs[key].iloc[:-test_size].copy()

    ## Set same initial questions
    limit = 1 * n
    initial = dfs["responses_RR"].iloc[:limit]
    for k, df in dfs.items():
        df.iloc[:limit] = initial.copy()
    assert all((df.iloc[:limit].score <= -1000).all() for df in dfs.values())

    cols = ["head", "winner", "loser"]
    datasets = {k: df[cols].to_numpy() for k, df in dfs.items()}
    print([len(v) for v in datasets.values()])
    X_test = run._X_test()

    print([len(v) for v in datasets.values()])
    assert len(datasets) == 2

    NUM_ANS = [
        n * i
        for i in list(range(1, 30, 5))
        + list(range(30, 130, 10))
        + list(range(130, 333, 20))
    ]

    static = dict(
        X_test=X_test,
        d=2,
        max_epochs=20_000,
        dwell=100,
        verbose=200,
    )
    #  offline._get_trained_model(
        #  datasets["responses_RandomSampling"],
        #  n_responses=10_000,
        #  meta=_get_config("RandomSampling"),
        #  noise_model="CKL",
        #  module__mu=0.05,
        #  **static,
    #  )
    #  sys.exit(0)
    #  breakpoint()

    client = Client("localhost:8786")
    client.upload_file("offline.py")
    d = client.run(_check_version)
    assert all(list(d.values()))

    d = client.run(_prep)
    d = list(d.values())
    assert all(d), d
    print(d)
    static["threads"] = d[0]

    r_dataset = client.scatter(datasets["responses_RandomSampling"])
    #  r_dataset = datasets["responses_RandomSampling"]
    random_futures = [
        client.submit(
            offline._get_trained_model,
            r_dataset,
            n_responses=n_ans,
            meta=_get_config("RandomSampling"),
            noise_model=nm,
            ident=f"random-{nm}",
            alg="random",
            **static,
            **_get_kwargs(nm),
        )
        for n_ans in NUM_ANS
        for nm in ["TSTE", "SOE", "CKL", "GNMDS"]
    ]

    a_dataset = client.scatter(datasets["responses_RR"])
    #  a_dataset = datasets["responses_RR"]
    active_futures = [
        client.submit(
            offline._get_trained_model,
            a_dataset,
            n_responses=n_ans,
            meta=_get_config("RR"),
            noise_model=nm,
            ident=f"active-{nm}",
            alg="active",
            **static,
            **_get_kwargs(nm),
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
        with open(f"/scratch/ssievert/io/alien-eggs/{i}.msgpack", "wb") as f2:
            msgpack.dump(save, f2)
