import yaml
import sys
from pathlib import Path
import asyncio
import numpy as np
import httpx
import pandas as pd
import sys
from time import time
from typing import List
import msgpack
from datetime import datetime

import test_utils

THIS_DIR = Path(__file__).absolute().parent
sys.path.append(str(THIS_DIR.parent))

import response_model
import targets

# c4.8xlarge machine (36 vCPUs, 60GB RAM, $1.59/hr)
# m4.16xlarge (64 vCPUs, 256GB RAM, $3.20/hr)
#  base = "http://54.202.72.125:8000"  # rate=5
base = "http://34.218.248.255:8000"
rate = 5

#  base = "http://34.221.93.18:8000"  # rate=2
#  rate = 2

#  base = "http://52.27.107.211:8000"  # rate=1
#  rate = 1

#  base = "http://18.236.141.222:8000"  # rate=0.5
#  rate = 0.5


async def _write(data: List[dict], fname: str) -> bool:
    with open(fname, "wb") as f:
        msgpack.dump(data, f)
    return True


class Stats:
    async def run_until(self, event, exp_uid, fname):
        history = []
        while True:
            deadline = time() + 60
            try:
                d = await self.collect(exp_uid)
                r = await self.responses(exp_uid)
                responses = r.json()
            except Exception as e:
                print("exception getting model!")
                print(type(e), e)
                while time() < deadline - 30:
                    await asyncio.sleep(3)
                continue

            assert len(d["algorithms"]) == 1
            alg_data = d["algorithms"][0]
            em = alg_data["X"]
            n_responses = alg_data["num_reported_answers"]
            meta = {k: alg_data[k] for k in ["delta", "d", "n", "alg_id", "alg_label"]}
            datum = {"meta": {"n_responses": n_responses, **meta}, "embedding": em}
            history.append(datum)
            await _write(history, f"{fname}_history.msgpack")
            await _write(responses, f"{fname}_responses.msgpack")

            if event.is_set():
                break
            while time() < deadline:
                await asyncio.sleep(3)

        return history

    @staticmethod
    async def collect(exp_uid):
        async with httpx.AsyncClient() as client:
            r = await client.get(base + f"/api/experiment/{exp_uid}")
            return r.json()

    @staticmethod
    async def responses(exp_uid):
        async with httpx.AsyncClient() as client:
            r = await client.get(base + f"/api/experiment/{exp_uid}/participants", timeout=90)
            return r


async def run_user(
    *, puid, exp_uid, num_queries, http, response_time=1, random_state=None
):
    assert random_state is not None
    http_kw = dict(headers={"content-type": "application/json"}, timeout=90)
    rng = np.random.RandomState(random_state)
    sleep_time = rng.uniform(low=0, high=8)
    await asyncio.sleep(sleep_time)

    gq = {"args": {"participant_uid": puid, "widget": False}, "exp_uid": exp_uid}
    PA_R = []
    for k in range(num_queries):
        if int(puid) == 0:
            print(k)
        try:
            q_future = http.post(base + "/api/experiment/getQuery", json=gq, **http_kw)

            sleep_time = rng.normal(loc=response_time, scale=0.2)
            sleep_time = max(0.75, sleep_time)
            sleep_time = float(np.clip(sleep_time, 0.75, 10))
            sleep_future = asyncio.sleep(sleep_time)

            _query, _ = await asyncio.gather(q_future, sleep_future)
            query = _query.json()
        except Exception as e:
            print("error getting query")
            print(type(e), e)
            continue

        if _query.status_code != 200:
            print("get_query.status_code != 200", _query.text)
            continue

        objs = {t["label"]: t["primary_description"] for t in query["target_indices"]}

        target_ids = {t["label"]: t["target_id"] for t in query["target_indices"]}

        h, l, r = [int(objs[k].strip("i.png")) for k in ["center", "left", "right"]]

        winner = response_model.alien_egg(h, l, r)
        assert winner in {0, 1}
        winner_index = target_ids["left"] if winner == 0 else target_ids["right"]
        answer = {
            "exp_uid": exp_uid,
            "args": {
                "query_uid": query["query_uid"],
                "target_winner": winner_index,
                "response_time": sleep_time,
            },
        }
        try:
            r = await http.post(
                base + "/api/experiment/processAnswer", json=answer, **http_kw
            )
        except Exception as e:
            print("process_answer error!", e)
        else:
            if r.status_code != 200:
                print("process_answer error!", r.text)
    return True


async def main(config):
    exp = yaml.safe_load(Path("init.yaml").read_text())
    spikes = targets.get(config["n"])
    rendered_spikes =[{"primary_description": f"i0{s}.png"} for s in spikes]
    exp["args"]["targets"]["targetset"] = rendered_spikes

    rendered_exp, meta = test_utils.initExp(exp, url=base)
    config["response_time"] = 2
    config["n_users"] = int(2 * config["responses_per_sec"])
    assert config["n_users"] >= 1
    print(config)
    r = input("Continue? ")
    if r.lower() != "y":
        sys.exit(1)
    n_queries = (config["max_queries"] // config["n_users"]) + 1

    stats = Stats()
    completed = asyncio.Event()
    fname = f"io/{today}/rate={config['responses_per_sec']}"
    DIR = Path(f"io/{today}")
    if not DIR.exists():
        DIR.mkdir()
    task = asyncio.create_task(stats.run_until(completed, meta["exp_uid"], fname))


    async with httpx.AsyncClient() as client:
        user_futures = [
            run_user(
                puid=str(k),
                exp_uid=meta["exp_uid"],
                num_queries=n_queries,
                http=client,
                random_state=k + 1,
                response_time=config["response_time"],
            )
            for k in range(config["n_users"])
        ]
        await asyncio.gather(*user_futures)
        completed.set()
        while not task.done():
            await asyncio.sleep(0.1)
    r = httpx.get(base + f"/api/experiment/{meta['exp_uid']}/participants?zip=0")
    df = pd.DataFrame(r.json())
    fname = config["fname"].format(**config)
    df.to_csv(f"{fname}.csv.zip")
    return True


if __name__ == "__main__":
    next_targets = pd.read_csv("../targets.csv.zip", header=None)[0].tolist()
    today = datetime.now().isoformat()[:10]
    exp = yaml.safe_load(Path("init.yaml").read_text())
    fnames = [t["primary_description"] for t in exp["args"]["targets"]["targetset"]]

    assert set(next_targets) == set(fnames)
    config = {
        "n": 90,
        "max_queries": 22_000,
        "fname": "io/{today}/responses-next-{responses_per_sec}",
        "responses_per_sec": rate,
        "today": today,
    }
    asyncio.run(main(config))
