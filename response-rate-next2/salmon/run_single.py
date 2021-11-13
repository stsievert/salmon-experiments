import asyncio
import json
import sys
from pathlib import Path
from pprint import pprint
from time import time, sleep
from typing import Optional, Tuple, Union, Dict, List, Any
from datetime import datetime
from copy import deepcopy
import zipfile

import httpx
import numpy as np
import msgpack
import pandas as pd
import yaml
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


datetime_parser = lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")

_SALMON_DIR = Path.home() / "Developer" / "stsievert" / "salmon"
for path in [
    _SALMON_DIR / "examples",
    _SALMON_DIR / "examples" / "alien-eggs",
    Path(__file__).absolute().parent.parent,
]:
    sys.path.append(str(path))

import response_model as datasets
import targets

#  SALMON = "http://localhost:8421"
#  SALMON_BACKEND = "http://localhost:8400"
today = datetime.now().isoformat()[:10]
DIR = Path(f"io/{today}")
if not DIR.exists():
    DIR.mkdir()


class SalmonExperiment(BaseEstimator):
    def __init__(
        self,
        url,
        dataset="strange_fruit",
        n=200,
        d=2,
        R=1,
        init=True,
        alg="ARR",
        random_state=None,
        config_fname=None,
        n_search=0,
        n_top=0,
        scores="random",
    ):
        self.url = url
        self.dataset = dataset
        self.n = n
        self.d = d
        self.R = R
        self.init = init
        self.alg = alg
        self.random_state = random_state
        self.config_fname = config_fname
        self.n_search = n_search
        self.n_top = n_top
        self.scores = scores

    def initialize(self):
        if self.init:
            self.auth_ = ("username", "password5929")
            username, pword = self.auth_
            httpx.post(self.url + f":8421/create_user/{username}/{pword}")
            httpx.get(
                self.url + ":8421/reset?force=1", auth=self.auth_, timeout=80,
            )
            sleep(4)

        if self.alg in {"ARR"}:
            sampler = {
                self.alg: {
                    "R": self.R,
                    "random_state": self.random_state,
                    "scores": self.scores,
                }
            }
            if self.n_search != 0:
                sampler[self.alg]["n_search"] = self.n_search
            if self.n_top != 0:
                sampler[self.alg]["n_top"] = self.n_top
        elif self.alg == "TSTE":
            sampler = {
                self.alg: {"random_state": self.random_state, "n_search": self.n_search}
            }
        elif self.alg == "RandomSampling":
            sampler = {"RandomSampling": {}}
        else:
            raise ValueError(f"alg={self.alg} not in ['RR', 'RandomSampling']")

        assert self.random_state is not None, self.random_state
        init = {
            "d": self.d,
            "samplers": sampler,
            "targets": targets.get(self.n),
        }
        #  eggs = Path(__file__).parent / "io" / "alienegg30.zip"
        #  assert eggs.exists()
        pprint({k: v for k, v in init.items() if k not in ["targets"]})
        c = input("Using config above. Is that okay? (y/N) ")
        if c.lower() != "y":
            print("Breaking")
            sys.exit(1)
        r = httpx.post(
            self.url + ":8421/init_exp",
            data={"exp": yaml.dump(init)},
            #  files={"targets": eggs.read_bytes()},
            auth=self.auth_,
            timeout=20,
        )
        assert r.status_code == 200, (r.status_code, r.text)

        r = httpx.get(self.url + ":8421/config", auth=self.auth_)
        assert r.status_code == 200, (r.status_code, r.text)
        self.config = r.json()
        return self


def _answer(query: Tuple[int, int, int], random_state=None,) -> int:
    h, l, r = query
    dl = abs(h - l)
    dr = abs(h - r)

    winner_idx = datasets.alien_egg(h, l, r)
    return winner_idx


async def _write(data: List[dict], fname: str) -> bool:
    with open(fname, "wb") as f:
        msgpack.dump(data, f)
    return True


async def _write_df(df: pd.DataFrame, fname: str) -> bool:
    df.to_csv(fname)
    return True


class Stats:
    async def run_until(self, n, event, *, config, targets):
        config = deepcopy(config)
        history = []
        fname = config["fname"].format(**config)
        dir = config.pop("dir")
        while True:
            if event.is_set():
                break
            deadline = time() + 60
            try:
                responses = await self.collect(config["url"])
            except Exception as e:
                print("exception collecting")
                print(e)
                continue
            alg = config["alg"]
            try:
                r = await self._get_endpoint(f"/model/{alg}", config["url"] + ":8400")
            except Exception as e:
                print("Exception getting model")
                print(e)
                continue
            if r.status_code == 200:
                meta = {"n_responses": len(responses), "time": time(), **config}
                d = r.json()
                em = d.pop("embedding")
                datum = {"meta": meta, "alg_model": d, "embedding": em}
                history.append(datum)
                await _write(history, f"{dir / fname}_history.msgpack")
            else:
                print(r.text)

            try:
                df = pd.DataFrame(responses)
            except Exception as e:
                print(e)
                print("Error getting model")
            both = set(df.columns).intersection(set(config.keys()))
            assert both.issubset({"response_time"})
            for k, v in config.items():
                if k == "response_time":
                    k = "response_time_mean"
                df[k] = v

            await _write_df(df, f"{dir / fname}_responses.csv.zip")

            while time() < deadline:
                await asyncio.sleep(3)

        return history

    async def collect(self, url):
        async with httpx.AsyncClient() as client:
            responses = await _get_responses(client, url + ":8421")
        return responses

    @staticmethod
    async def _get_endpoint(endpoint, url, auth=("username", "password5929")):
        async with httpx.AsyncClient() as http:
            r = await http.get(url + endpoint, auth=auth, timeout=10)
        return r


class User(BaseEstimator):
    def __init__(
        self,
        url,
        *,
        targets: List[str],
        n_responses=100,
        response_time=1,
        reaction_time=0.75,
        random_state=None,
        http=None,
        uid="",
    ):
        self.url = url
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

        sleep_time = self.random_state_.uniform(low=0, high=8)
        await asyncio.sleep(sleep_time)
        sleep_time = self.random_state_.normal(loc=self.response_time, scale=0.5)
        await asyncio.sleep(sleep_time)
        for k in range(self.n_responses):
            try:
                datum = {"num_responses": k, "puid": self.uid, "url": self.url}

                _s = time()
                try:
                    r = await self.http.get(self.url + ":8421/query", timeout=90)
                except:
                    continue
                datum.update({"time_get_query": time() - _s})
                assert r.status_code == 200, r.text

                query = r.json()

                h = self.targets[query["head"]]
                l = self.targets[query["left"]]
                r = self.targets[query["right"]]

                _ans = _answer((h, l, r))
                winner = query["left"] if _ans == 0 else query["right"]

                sleep_time = self.random_state_.normal(
                    loc=self.response_time, scale=0.20
                )
                sleep_time = float(np.clip(sleep_time, self.reaction_time, 10))
                answer = {
                    "winner": winner,
                    "puid": self.uid,
                    "response_time": sleep_time,
                    **query,
                }
                w = answer["winner"]
                if self.uid == "0":
                    msg = f"(h, l, r, w) = {(h, l, r, w)}"
                    dl = abs(h - l)
                    dr = abs(h - r)
                    ratio = max(dr, dl) / (dl + dr)
                    print(f"ratio={ratio:0.2f}, {msg}, score={answer['score']:0.2f}")
                datum.update({"h": h, "l": l, "r": r, "w": w})
                datum.update({"time": time()})
                await asyncio.sleep(sleep_time)
                datum.update({"sleep_time": sleep_time})
                _s = time()
                try:
                    _r = await self.http.post(
                        self.url + ":8421/answer", json=answer, timeout=20
                    )
                except:
                    pass
                else:
                    assert _r.status_code == 200, (_r.status_code, _r.text)
                datum.update({"time_post_answer": time() - _s})
                self.data_.append(datum)
            except Exception as e:
                print("Exception!")
                print(e)
                raise e
        return self

    def partial_fit(self, X=None, y=None):
        return self._partial_fit(X=X, y=y)


def _fmt(x: str) -> int:
    filename = x.split("/")[-2]
    numeric = filename.strip("i.png' ")
    return int(numeric)


async def main(**config):
    url = config["url"]
    r = httpx.get(url + ":8421/reset?force=1", timeout=80)
    await asyncio.sleep(4)
    kwargs = {
        k: config[k] for k in ["dataset", "n", "d", "init", "alg", "R", "random_state"]
    }
    if "n_search" in config:
        kwargs["n_search"] = config["n_search"]
    if "n_top" in config:
        kwargs["n_top"] = config["n_top"]
    if "scores" in config:
        kwargs["scores"] = config["scores"]

    exp = SalmonExperiment(config["url"], **kwargs)
    exp.initialize()
    fname = config["fname"].format(**config)
    #  targets = pd.read_csv("../targets.csv.zip", header=None)[0].tolist()
    exp.config["targets"] = [int(t) for t in exp.config["targets"]]
    #  exp.config["targets"] = list(sorted(exp.config["targets"]))
    #  assert {int(t.strip("i.png")) for t in targets}.issubset(set(exp.config["targets"]))

    config["response_time"] = 2
    config["n_users"] = int(2 * config["responses_per_sec"])
    assert config["responses_per_sec"] >= 0.5
    n_responses = (config["max_queries"] // config["n_users"]) + 1

    kwargs = {k: config[k] for k in ["response_time"]}
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
                config["url"],
                http=client,
                random_state=i + 1,
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


async def _get_responses(http, url):
    r = await http.get(
        url + f"/responses", auth=("username", "password5929"), timeout=20
    )
    return r.json()


async def run_salmon_searches():
    config = {
        "n": 90,
        "d": 2,
        "R": 1,
        "dataset": "alien_eggs",
        "random_state": 400,
        "init": True,
        "max_queries": 30_000,
        #  "max_queries": 200,
        "dir": DIR,
        "fname": "{alg}-n_top={n_top}-scores={scores}-{responses_per_sec}",
        "alg": "ARR",
        "responses_per_sec": 1,
    }

    jobs = []
    #  urls = {"approx": "localhost"}
    urls = {
        "approx": "54.149.189.191",
        "original": "34.211.47.230",
        "random": "35.161.254.217",
        #  "approx": "localhost",
        #  "original": "localhost",
        #  "random": "localhost",
    }
    for scores, url in urls.items():
        sleep(np.random.uniform(low=1, high=4))
        config2 = deepcopy(config)
        config2["url"] = f"http://{url}"
        config2["scores"] = scores
        config2["n_top"] = 1
        jobs.append(main(**config2))
        assert True
        print(f"\n#### Done with scores={scores}\n")

    out = await asyncio.gather(*jobs)
    assert all(out)
    return True


if __name__ == "__main__":
    #  asyncio.run(run_rates())
    #  asyncio.run(run_searches())
    asyncio.run(run_salmon_searches())
