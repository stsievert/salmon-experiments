"""
This script runs a simulation with the Salmon server launched at SALMON.

Input: configuration.
Output: Responses in responses.csv
"""

import asyncio
import json
import sys
from pathlib import Path
from time import time
from typing import Optional, Tuple, Union, Dict, List

import httpx
import numpy as np
import msgpack
import pandas as pd
import yaml
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

import stats

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
        R=10,
        random_state=None,
        init=True,
        alg="RR",
    ):
        self.salmon = salmon
        self.dataset = dataset
        self.n = n
        self.d = d
        self.R = R
        self.random_state = random_state
        self.init = init
        self.alg = alg

    def initialize(self):
        self.random_state_ = check_random_state(self.random_state)
        if self.init:
            httpx.get(
                self.salmon + "/reset?force=1",
                auth=("username", "password"),
                timeout=20,
            )
        if self.alg not in ["RR", "RandomSampling"]:
            raise ValueError(f"alg={self.alg} not in ['RR', 'RandomSampling']")
        sampler = {"RR": {"random_state": self.random_state, "R": self.R}}
        if self.alg == "RandomSampling":
            sampler["RR"]["sampling"] = "random"

        init = {
            "d": self.d,
            "samplers": sampler,
            "targets": list(range(self.n)),
        }
        #  if not self.init:
            #  self.config = init
            #  return self
        r = httpx.post(
            self.salmon + "/init_exp",
            data={"exp": yaml.dump(init)},
            auth=("username", "password"),
        )
        assert r.status_code == 200, (r.status_code, r.text)

        self.config = init
        return self


def _random_query(n: int, random_state=None) -> List[int]:
    rng = check_random_state(random_state)
    h, l, r = rng.choice(n, replace=False, size=3).tolist()
    return [int(h), int(l), int(r)]


def _answer(
    query: Union[List[int], dict], noise="constant", p_wrong=P_WRONG, random_state=None
) -> List[int]:
    if isinstance(query, dict):
        h, l, r = (query[k] for k in ["head", "left", "right"])
    else:
        h, l, r = query
    dl = abs(h - l)
    dr = abs(h - r)

    if noise == "constant":
        winner, loser = (l, r) if dl < dr else (r, l)
        ans = [h, winner, loser]
        if check_random_state(random_state).uniform(0, 1) < p_wrong:
            ans = [h, loser, winner]
    elif noise == "human":
        winner_idx = datasets.alien_egg(h, l, r)
        winner, loser = (l, r) if winner_idx == 0 else (r, l)
        ans = [h, winner, loser]
    else:
        raise ValueError("noise='{noise}' not in ['human', 'constant']")
    return ans


def _X_test(n: int, num: int = 10_000, seed=0, **kwargs) -> np.ndarray:
    queries = [_random_query(n, random_state=i + seed) for i in range(num)]
    answers = [
        _answer(query, random_state=i, **kwargs) for i, query in enumerate(queries)
    ]
    return np.asarray(answers)


class Stats:
    async def run_until(self, n, event, *, config):
        history = []
        X_test = _X_test(n, num=20_000, noise=config["noise"])
        ident = _ident(config)
        while True:
            deadline = time() + 60
            responses = await self.collect()
            alg = "RR"  # config["alg"]
            r = await self._get_endpoint(f"/model/{alg}", base=SALMON_BACKEND)
            if r.status_code == 200:
                d = r.json()
                stat = stats.collect(d["embedding"], X_test)
                history.append({"n_responses": len(responses), **config, **stat})
                pd.DataFrame(history).to_csv(f"history*{ident}.csv")
            else:
                print(r.text)

            df = pd.DataFrame(responses)
            both = set(df.columns).intersection(set(config.keys()))
            assert both.issubset({"response_time"})
            for k, v in config.items():
                if k == "response_time":
                    k = "response_time_mean"
                df[k] = v
            ident = _ident(config)
            df.to_csv(f"responses*{ident}.csv")

            if event.is_set():
                break
            while time() < deadline:
                await asyncio.sleep(5)

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
        salmon=SALMON,
        noise="human",
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
        self.noise = noise
        self.http = http
        self.uid = uid

    def init(self):
        self.initialized_ = True
        self.random_state_ = check_random_state(self.random_state)
        self.data_ = []
        return self

    async def _partial_fit(self, X=None, y=None):
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.init()

        await asyncio.sleep(5)
        for k in range(self.n_responses):
            try:
                datum = {"num_responses": k, "puid": self.uid, "salmon": self.salmon}

                _s = time()
                r = await self.http.get(self.salmon + "/query", timeout=20)
                datum.update({"time_get_query": time() - _s})
                assert r.status_code == 200, r.text

                query = r.json()

                h = query["head"]
                l = query["left"]
                r = query["right"]
                dl = abs(h - l)
                dr = abs(h - r)

                _ans = _answer(query, noise=self.noise)
                winner = _ans[1]

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
                    if w == l:
                        msg = f"DL={dl}, dr={dr}. (h, l, r, w) = {(h, l, r, w)}"
                    elif w == r:
                        msg = f"DR={dr}, dl={dl}. (h, l, r, w) = {(h, l, r, w)}"
                    else:
                        raise ValueError(f"h, l, r, w = {(h, l, r, w)}")
                    print(f"{msg}, score={answer['score']}")
                datum.update({"h": h, "l": l, "r": r, "w": w, "dl": dl, "dr": dr})
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


async def main(**config):
    kwargs = {
        k: config[k] for k in ["random_state", "dataset", "n", "d", "init", "alg"]
    }
    exp = SalmonExperiment(**kwargs)
    exp.initialize()

    n_responses = (config["max_queries"] // config["n_users"]) + 1

    kwargs = {k: config[k] for k in ["reaction_time", "response_time", "noise"]}
    completed = asyncio.Event()
    stats = Stats()
    task = asyncio.create_task(stats.run_until(config["n"], completed, config=config))
    async with httpx.AsyncClient() as client:
        users = [
            User(
                http=client,
                random_state=i,
                uid=str(i),
                n_responses=n_responses,
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
        "R": 10,
        "dataset": "alien_eggs",
        "noise": "human",
        "random_state": 42,
        "reaction_time": 0.25,
        "response_time": 1.00,
        "init": True,
        "n_users": 10,
        "alg": "RR",
        "max_queries": 30_000 + 100,
        #  "alg": "RandomSampling",
        #  "max_queries": 100_000 + 100,
    }
    ## Make sure no data has been uploaded
    r = httpx.get(SALMON + "/", timeout=20)
    assert (
        r.json()["detail"] == "No data has been uploaded"
        or "Experiment keys are not correct" in r.json()["detail"]
    )

    responses = asyncio.run(main(**config))
    assert True
