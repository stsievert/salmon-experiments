"""
This script runs a simulation with the Salmon server launched at SALMON.

Input: configuration.
Output: Responses in responses.csv
"""

import asyncio
import json
import sys
from pathlib import Path
from time import time, sleep
from typing import Optional, Tuple, Union, Dict, List, Any
from datetime import datetime
from copy import deepcopy

import httpx
import numpy as np
import msgpack
import pandas as pd
import yaml
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

import stats

datetime_parser = lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")

for path in [
    Path.home() / "Developer" / "stsievert" / "salmon",
    Path.home() / "salmon",
]:
    sys.path.append(str(path / "examples"))

import datasets

SALMON = "http://localhost:8421"
SALMON_BACKEND = "http://localhost:8400"
P_WRONG = 0.15  # p in [0, 1]


class SalmonExperiment(BaseEstimator):
    def __init__(
        self,
        salmon=SALMON,
        dataset="strange_fruit",
        n=200,
        d=2,
        R=1,
        init=True,
        alg="RR",
        random_state=None,
    ):
        self.salmon = salmon
        self.dataset = dataset
        self.n = n
        self.d = d
        self.R = R
        self.init = init
        self.alg = alg
        self.random_state = random_state

    def initialize(self):
        if self.init:
            httpx.get(
                self.salmon + "/reset?force=1",
                auth=("username", "password"),
                timeout=20,
            )
            sleep(4)
        if self.alg == "RR":
            sampler = {"RR": {"R": self.R, "random_state": self.random_state}}
        elif self.alg == "RandomSampling":
            sampler = {"RandomSampling": {}}
        else:
            raise ValueError(f"alg={self.alg} not in ['RR', 'RandomSampling']")

        assert self.random_state is not None, self.random_state
        init = {
            "d": self.d,
            "samplers": sampler,
        }
        eggs = Path(__file__).parent / "io" / "alienegg30.zip"
        assert eggs.exists()
        r = httpx.post(
            self.salmon + "/init_exp",
            data={"exp": yaml.dump(init)},
            files={"targets": eggs.read_bytes()},
            auth=("username", "password"),
            timeout=10,
        )
        assert r.status_code == 200, (r.status_code, r.text)

        r = httpx.get(self.salmon + "/config", auth=("username", "password"))
        assert r.status_code == 200, (r.status_code, r.text)
        self.config = r.json()
        return self


def _random_query(n: int, random_state=None) -> List[int]:
    rng = check_random_state(random_state)
    h, l, r = rng.choice(n, replace=False, size=3).tolist()
    return [int(h), int(l), int(r)]


def _answer(query: Tuple[int, int, int], random_state=None,) -> int:
    h, l, r = query
    dl = abs(h - l)
    dr = abs(h - r)

    winner_idx = datasets.alien_egg(h, l, r)
    return winner_idx


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


def _get_targets():
    """
    Inputs: none
    Outputs: list of fnames: ['i0126.png', 'i0208.png', ...]
    """
    today = datetime.now().isoformat()[:10]
    DIR = Path("io/2021-03-09/")
    if today != "2021-03-24":
        raise ValueError(f"Careful! Hard coded directory {DIR}. Fix me!")
    suffix = "RandomSampling"
    with open(DIR / f"config_{suffix}.yaml") as f:
        config = yaml.safe_load(f)

    rare = [t.split("/")[-2] for t in config["targets"]]
    mrare = [t.strip("'\" ") for t in rare]
    return mrare


def _next_responses(alg: str, targets: List[str]) -> np.ndarray:
    df = _munge("io/next-fig3.json.static")
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

    if targets is None:
        targets = _get_targets()
    assert len(targets) == len(idx_fname)
    spike_idx = {
        int(fname.strip("i.png")): salmon_idx
        for salmon_idx, fname in enumerate(targets)
    }
    # Make sure the spikes on both sides are the same
    assert set(spike_idx.keys()) == set(np.unique(X_spikes))
    assert set(spike_idx.keys()) == set(spikes.values())
    X_salmon = np.vectorize(spike_idx.get)(X_spikes)
    return X_salmon


def _X_test(targets: List[str] = None) -> np.ndarray:
    return _next_responses("Test", targets)


class Stats:
    async def run_until(self, n, event, *, config, targets):
        history = []
        X_test = _X_test()
        fname = config["fname"].format(**config)
        while True:
            deadline = time() + 10
            responses = await self.collect()
            alg = "RR"  # config["alg"]
            r = await self._get_endpoint(f"/model/{alg}", base=SALMON_BACKEND)
            if r.status_code == 200:
                d = r.json()
                stat = stats.collect(d["embedding"], targets, X_test)
                history.append({"n_responses": len(responses), **config, **stat})
                pd.DataFrame(history).to_csv(f"{fname}_history.csv")
            else:
                print(r.text)

            df = pd.DataFrame(responses)
            both = set(df.columns).intersection(set(config.keys()))
            assert both.issubset({"response_time"})
            for k, v in config.items():
                if k == "response_time":
                    k = "response_time_mean"
                df[k] = v
            df.to_csv(f"{fname}_responses.csv")

            if event.is_set():
                break
            while time() < deadline:
                await asyncio.sleep(3)

        return history

    async def collect(self):
        async with httpx.AsyncClient() as client:
            responses = await _get_responses(client)
        return responses

    @staticmethod
    async def _get_endpoint(endpoint, base=SALMON):
        async with httpx.AsyncClient() as http:
            r = await http.get(base + endpoint, auth=("username", "password"))
        return r


class User(BaseEstimator):
    def __init__(
        self,
        *,
        targets: List[str],
        salmon=SALMON,
        n_responses=100,
        response_time=1,
        reaction_time=0.75,
        random_state=None,
        http=None,
        uid="",
    ):
        self.salmon = salmon
        self.response_time = response_time
        self.n_responses = n_responses
        self.random_state = random_state
        self.reaction_time = reaction_time
        self.http = http
        self.uid = uid
        self.targets = targets

    def init(self):
        self.initialized_ = True
        self.random_state_ = check_random_state(self.random_state)
        self.data_ = []
        return self

    async def _partial_fit(self, X=None, y=None):
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.init()

        await asyncio.sleep(1)

        sleep_time = self.random_state_.normal(loc=8, scale=2)
        await asyncio.sleep(sleep_time)
        sleep_time = self.random_state_.normal(
            loc=self.reaction_time, scale=self.reaction_time / 4
        )
        await asyncio.sleep(sleep_time)
        for k in range(self.n_responses):
            try:
                datum = {"num_responses": k, "puid": self.uid, "salmon": self.salmon}

                _s = time()
                r = await self.http.get(self.salmon + "/query", timeout=20)
                datum.update({"time_get_query": time() - _s})
                assert r.status_code == 200, r.text

                query = r.json()

                h = self.targets[query["head"]]
                l = self.targets[query["left"]]
                r = self.targets[query["right"]]

                _ans = _answer((h, l, r))
                winner = query["left"] if _ans == 0 else query["right"]

                sleep_time = self.random_state_.normal(
                    loc=self.response_time, scale=0.25
                )
                sleep_time = max(self.reaction_time, sleep_time)
                answer = {
                    "winner": winner,
                    "puid": self.uid,
                    "response_time": sleep_time,
                    **query,
                }
                w = answer["winner"]
                if self.uid == "0":
                    msg = f"(h, l, r, w) = {(h, l, r, w)}"
                    print(f"{msg}, score={answer['score']}")
                datum.update({"h": h, "l": l, "r": r, "w": w})
                datum.update({"time": time()})
                await asyncio.sleep(sleep_time)
                datum.update({"sleep_time": sleep_time})
                _s = time()
                r = await self.http.post(
                    self.salmon + "/answer", data=json.dumps(answer), timeout=20
                )
                datum.update({"time_post_answer": time() - _s})
                assert r.status_code == 200, r.text
                self.data_.append(datum)
            except Exception as e:
                print("Exception!")
                print(e)
        return self

    def partial_fit(self, X=None, y=None):
        return self._partial_fit(X=X, y=y)


def _fmt(x: str) -> int:
    filename = x.split("/")[-2]
    numeric = filename.strip("i.png' ")
    return int(numeric)


async def main(**config):
    r = httpx.get(SALMON + "/reset?force=1", timeout=20)
    await asyncio.sleep(4)
    kwargs = {k: config[k] for k in ["dataset", "n", "d", "init", "alg", "R", "random_state"]}
    exp = SalmonExperiment(**kwargs)
    exp.initialize()

    exp.config["targets"] = [_fmt(t) for t in exp.config["targets"]]

    config["response_time"] = config["n_users"] / config["responses_per_sec"]
    n_responses = (config["max_queries"] // config["n_users"]) + 1

    kwargs = {k: config[k] for k in ["reaction_time", "response_time"]}
    completed = asyncio.Event()
    stats = Stats()
    task = asyncio.create_task(
        stats.run_until(
            config["n"], completed, config=config, targets=exp.config["targets"]
        )
    )
    async with httpx.AsyncClient() as client:
        users = [
            User(
                http=client,
                random_state=i**2 + i**3 + i**4,
                uid=str(i),
                n_responses=n_responses,
                targets=exp.config["targets"],
                **kwargs,
            )
            for i in range(config["n_users"])
        ]
        responses = [user.partial_fit() for user in users]
        algs = list(exp.config["samplers"].keys())
        assert len(algs) == 1, "len(algs) = {}".format(len(algs))
        await asyncio.gather(*responses)
        completed.set()
        while not task.done():
            await asyncio.sleep(0.1)

    return responses


async def _get_responses(http, base=SALMON):
    r = await http.get(base + f"/responses", auth=("username", "password"))
    return r.json()


def _ident(d: dict) -> str:
    d2 = sorted(tuple(d.items()))
    d3 = [f"{k}={v}" for k, v in d2]
    return "-".join(d3)


if __name__ == "__main__":
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
        "fname": "io/tmp/{alg}-{responses_per_sec}",
        #  "responses_per_sec": 5,
        #  "alg": "RandomSampling",
    }

    config["alg"] = "RR"
    #  for rate in [20, 10, 5, 2, 1]:
    for rate in [1, 2, 5, 10, 20]:
        config["responses_per_sec"] = rate
        responses = asyncio.run(main(**config))
        assert True
        print(f"\n#### Done with rate={rate}\n")

    config["alg"] = "RandomSampling"
    config["responses_per_sec"] = 5
    responses = asyncio.run(main(**config))
    assert True
