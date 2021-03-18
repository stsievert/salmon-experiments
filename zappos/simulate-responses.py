import asyncio
import json
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Union

import httpx
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

import stats

Query = Dict[str, int]
Answer = Dict[str, int]
SALMON = "http://localhost:8421"
SALMON_BACKEND = "http://localhost:8400"


class ExpStats:
    async def run_until(self, X_test, event, *, config):
        ident = config["ident"]
        history = []

        await asyncio.sleep(5)
        while True:
            deadline = time() + 10
            responses = await self.collect()
            alg = "RR"  # config["alg"]
            r = await self._get_endpoint(f"/model/{alg}", base=SALMON_BACKEND)
            if r.status_code == 200:
                d = r.json()
                stat = stats.collect(d["embedding"], X_test)
                history.append({"n_responses": len(responses), **config, **stat})
                pd.DataFrame(history).to_csv(f"history_{ident}.csv")
            else:
                print("history error:", r.json())

            df = pd.DataFrame(responses)
            both = set(df.columns).intersection(set(config.keys()))
            assert both.issubset({"response_time"})
            for k, v in config.items():
                if k == "response_time":
                    k = "response_time_mean"
                df[k] = v
            df.to_csv(f"responses_{ident}.csv")

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


async def _get_responses(http, base=SALMON):
    r = await http.get(base + f"/responses", auth=("username", "password"))
    return r.json()


def init(module, n=85, d=2, R=1, random_state=None, alg="RR", server=SALMON):
    _seed = check_random_state(random_state).randint(2 ** 32 - 1)
    seed = int(_seed)
    if alg not in ["RR", "RandomSampling"]:
        raise ValueError(f"alg={alg} not in ['RR', 'RandomSampling']")
    sampler = {"RR": {"module": module, "R": R}}
    if alg == "RandomSampling":
        sampler["RR"]["sampling"] = "random"

    init = {
        "d": d,
        "samplers": sampler,
        "targets": config["n"],
    }
    print("Resetting...")
    httpx.post(
        server + "/reset?force=1", auth=("username", "password"),
    )
    print("Init'ing this experiment:\n", init)
    httpx.post(
        server + "/init_exp",
        data={"exp": yaml.dump(init)},
        auth=("username", "password"),
    )
    return True


def _find(q: Query, X: np.ndarray) -> List[int]:
    global QUERIES
    idx = X[:, 0] == q["head"]
    idx &= (X[:, 1] == q["left"]) | (X[:, 1] == q["right"])
    idx &= (X[:, 2] == q["left"]) | (X[:, 2] == q["right"])

    full_idx = np.arange(len(X)).astype(int)
    return full_idx[idx]


def _answer(query: Query, X: np.ndarray, rng: np.random.RandomState) -> Answer:
    idx = _find(query, X)
    for _ in range(len(idx) * 3):
        i = rng.choice(idx)
        if i in QUERIES:
            break
    try:
        QUERIES.remove(i)
    except KeyError:
        # Happens in two conditions:
        # 1. race condition (`i` has already been removed)
        # 2. This question must be repeated
        pass

    q = X[i]
    ret = deepcopy(query)
    ret.update({"winner": int(q[1])})
    return ret


class User(BaseEstimator):
    def __init__(
        self,
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

    def init(self):
        self.initialized_ = True
        self.random_state_ = check_random_state(self.random_state)
        self.data_ = []
        return self

    async def _partial_fit(self, X, y=None):
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.init()

        for k in range(self.n_responses):
            try:
                r = await self.http.get(self.salmon + "/query", timeout=20)
                assert r.status_code == 200, r.text

                query = r.json()
                answer = _answer(query, X, self.random_state_)

                sleep_time = self.random_state_.normal(
                    loc=self.response_time, scale=0.25
                )
                sleep_time = max(self.reaction_time, sleep_time)
                if self.uid == "0":
                    print(k)
                await asyncio.sleep(sleep_time)

                answer["puid"] = self.uid
                answer["response_time"] = sleep_time
                _s = time()
                r = await self.http.post(
                    self.salmon + "/answer", data=json.dumps(answer), timeout=20
                )
                assert r.status_code == 200, r.text

            except Exception as e:
                print("Exception!")
                print(e)
                await asyncio.sleep(2)
        return self

    def partial_fit(self, *args, **kwargs):
        return self._partial_fit(*args, **kwargs)


async def main(config, X_train, X_test):
    assert config["n"] == len(np.unique(X_train))
    kwargs = {k: config[k] for k in ["n", "d", "R"]}
    init(config["module"], alg=config["sampler"], **kwargs)
    completed = asyncio.Event()
    stat = ExpStats()
    task = asyncio.create_task(stat.run_until(X_test, completed, config=config))
    async with httpx.AsyncClient() as client:
        n_responses = config["max_responses"] // config["n_users"]
        users = [
            User(
                http=client,
                random_state=i,
                uid=str(i),
                response_time=config["response_time"],
                n_responses=n_responses,
            )
            for i in range(config["n_users"])
        ]
        responses = [user.partial_fit(X_train) for user in users]
        await asyncio.gather(*responses)
        completed.set()
        while not task.done():
            await asyncio.sleep(1)


if __name__ == "__main__":
    config = {
        "n": 85,
        "d": 5,
        "R": 1,
        "sampler": "RR",
        #  "sampler": "RandomSampling",
        "max_responses": 10_000,
        "n_users": 5,
        "response_time": 2.0,
        "module": "CKL",
        "random_state": 42,
    }
    config["ident"] = config["sampler"]
    X = np.load("io/X.npy")
    QUERIES = set(range(len(X)))
    out = asyncio.run(main(config, X, X))

    # Eek! Testing on the train set. But it follows heim2015active.
    #
    # * The Zappos discussion says "The right-most graph in Fig. 3 shows the
    #   mean likelihood of all triples responses for all learned models."
    # * Discussion on synthetic data "All queries by the methods are answered
    #   by this pool, and the entire pool is used to evaluate query
    #   prediction error."

    assert True, out
