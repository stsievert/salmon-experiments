#!/usr/bin/env python
# coding: utf-8
import sys
from typing import List, Any, Dict, Union, Optional
from copy import deepcopy
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

    assert "v0.5.2+9" in salmon.__version__
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


def _get_config(suffix: str, targets=False):
    dir_ = Path("io/2021-03-21")
    with open(dir_ / f"config_{suffix}.yaml") as f:
        raw = yaml.safe_load(f)
    if targets:
        return raw["targets"]
    raw.pop("targets")
    rare = _flatten(raw)
    return rare


def _get_futures(client, dfs, random_state: int, d=2, config=None):
    for k, df in dfs.items():
        if "RR" in k:
            continue
        elif "RandomSampling" in k:
            df = df.sample(frac=1, random_state=random_state)
        else:
            raise ValueError(f"unrecognized algorithm in key {k}")

    cols = ["head", "winner", "loser"]
    datasets = {k: df[cols].to_numpy() for k, df in dfs.items()}
    html_targets = _get_config("RandomSampling", targets=True)
    targets = [t.split("/")[-2].strip("i.png '") for t in html_targets]
    print([len(v) for v in datasets.values()])
    X_test = run._X_test()

    print([len(v) for v in datasets.values()])
    print(f"Total of {len(datasets)} items")

    NUM_ANS = [n * i for i in range(1, 10)]
    NUM_ANS += [n * i for i in range(10, 140, 10)]
    NUM_ANS += [n * i for i in range(140, 240 + 20, 20) if i * n <= 7202]

    static = dict(
        X_test=X_test,
        dwell=1000,
        verbose=1000,
        random_state=random_state,
        max_epochs=400_000,
        d=d,
    )

    d = client.run(_prep)
    d = list(d.values())
    assert all(d), d
    print(d)
    static["threads"] = d[0]

    if random_state == 1:
        keys = [k for k in datasets.keys() if "RR" in k]
        a_datasets = {k: client.scatter(datasets[k]) for k in keys}
        active_futures = [
            client.submit(
                offline._get_trained_model,
                a_datasets[k],
                n_responses=n_ans,
                meta=config or {},
                noise_model=nm,
                ident=f"active-{nm}",
                alg="active",
                fname=k,
                priority=100,
                **static,
                **_get_kwargs(nm),
            )
            for n_ans in deepcopy(NUM_ANS)
            for nm in ["TSTE", "SOE", "CKL", "GNMDS"]
            for k in keys
        ]
        print("len(af)", len(active_futures))
    else:
        active_futures = []

    assert all("Random" in k or "RR" in k for k in dfs.keys())
    r_dataset = client.scatter(datasets["RandomSampling-5_responses"])
    random_futures = [
        client.submit(
            offline._get_trained_model,
            r_dataset,
            n_responses=n_ans,
            meta=config or {},
            noise_model=nm,
            ident=f"random-{nm}",
            alg="random",
            **static,
            **_get_kwargs(nm),
        )
        for n_ans in deepcopy(NUM_ANS)
        for nm in ["TSTE", "SOE", "CKL", "GNMDS"]
    ]

    futures = random_futures + active_futures
    print(f"len(futures) = {len(futures)}")
    return futures


if __name__ == "__main__":

    DIR = Path("io/2021-03-24/")
    # From run.py
    config = {
        "n": 30,
        "d": 2,
        "R": 1,
        "dataset": "alien_eggs",
        "random_state": 42,
        "reaction_time": 0.0,
        "n_users": 20,
        "init": True,
        "max_queries": 8000,
    }
    noise = "human"
    n = 30
    dfs = {
        f.name.replace(".csv", ""): pd.read_csv(f) for f in DIR.glob("*_responses.csv")
    }
    print([len(df) for df in dfs.values()])

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
    #  r_dataset = datasets["responses_RandomSampling"]

    client = Client("localhost:8786")
    client.upload_file("offline.py")
    d = client.run(_check_version)
    SEEDS = [s + 1 for s in range(10)]
    assert 1 in SEEDS and SEEDS[0] == 1
    _futures = [
        _get_futures(client, dfs, random_state=rs, d=d, config=config)
        for rs in SEEDS
        for d in [1, 2]
    ]
    futures = sum(_futures, [])

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
