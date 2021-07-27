import asyncio
import json
import random
import zipfile
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional, Any, Union
from typing import List, Dict
from time import time

import httpx
import pandas as pd
import numpy as np

import response_model
import targets

datetime_parser = lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")


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
        "puid": q["puid"],
        **sent,
    }


def _munge(fname: str) -> pd.DataFrame:
    assert "zip" in fname, "Specify a zipped JSON file"
    with zipfile.ZipFile(fname) as zf:
        assert len(zf.filelist) == 1, "Only specify one file in zip file"
        _fname = zf.filelist[0]
        raw = json.loads(zf.read(_fname))

    assert raw.pop("meta") == {"status": "OK", "code": 200}
    assert len(raw) == 1
    rare = raw["participant_responses"]
    mrare = []
    for puid, responses in rare.items():
        for r in responses:
            r["puid"] = puid
        mrare += responses
    #  mrare = sum(rare.values(), [])
    medium = [_flatten_query(d) for d in mrare]
    mwell = [m for m in medium if m]
    df = pd.DataFrame(mwell)
    cols = [
        "puid",
        "head",
        "left",
        "right",
        "winner",
        "alg",
        "timestamp",
        "response_time",
    ]
    #  assert set(df.columns) == set(cols)
    df["loser"] = df[["head", "left", "right", "winner"]].apply(
        lambda r: r["left"] if r["winner"] == r["right"] else r["right"], axis=1
    )
    return df[cols + ["loser"]]


def launch_experiment(
    hostname: str, *, n: int = 30, d: int = 2, seed: int = 42, sampler="ARR", reset=True,
):
    username, password = "fu", "bar"
    r = httpx.post(hostname + f"/create_user/{username}/{password}")
    assert r.status_code in [200, 403]
    init = {
        "targets": targets.get(n),
        "d": d,
    }
    if sampler == "ARR":
        samplers = {"ARR": {"random_state": seed, "R": 1}}
    elif sampler == "RandomSampling":
        samplers = {"RandomSampling": {}}
    else:
        raise ValueError(f"sampler={sampler} not in ['ARR', 'RandomSampling']")
    init["samplers"] = samplers

    if reset:
        r = httpx.get(hostname + "/reset?force=1", auth=(username, password), timeout=30)
        r = httpx.post(
            hostname + "/init_exp",
            auth=(username, password),
            data={"exp": json.dumps(init)},
            timeout=30,
        )
        assert r.status_code == 200
    return init


def sleep_until(timestamp: int):
    return asyncio.sleep(timestamp - time())


async def simulate_user(
    *, config, client, hostname: str, responses: Dict[str, Any], puid: int = 0
) -> int:
    assert set(responses.keys()) == {
        "secs_till_start",
        "response_times",
    }
    await asyncio.sleep(responses["secs_till_start"])
    start = time()
    targets = config["targets"]
    await asyncio.sleep(np.random.uniform(low=0, high=5))
    assert set(np.diff(np.argsort(targets)).tolist()) == {1}
    for k, rt in enumerate(responses["response_times"]):
        __start = time()
        try:
            r = await client.get(hostname + "/query")
            assert r.status_code == 200
        except Exception as e:
            print(f"Exception in /query! {e}")
            continue

        ans = r.json()
        ans["network_latency"] = time() - __start
        eps = 2e-3
        rt += np.random.uniform(low=-eps, high=eps)
        ans["response_time"] = max(rt, 100e-3)
        ans["puid"] = str(puid)

        await asyncio.sleep(rt)

        h, l, r = [targets[ans[k]] for k in ["head", "left", "right"]]
        w = response_model.alien_egg(h, l, r)
        assert w in [0, 1]
        ans["winner"] = ans["left"] if w == 0 else ans["right"]

        try:
            r = await client.post(hostname + "/answer", data=json.dumps(ans))
            assert r.status_code == 200
        except Exception as e:
            print(f"Exception in /answer! {e}")

        msg = "user {} on it {} took {:0.2f}s w/ rt={:0.2f}"
        print(msg.format(puid, k, time() - __start, rt))

    return len(responses["response_times"])


async def run(n=30, hostname="http://127.0.0.1:8421"):
    if hostname[-1] == "/":
        raise ValueError("No '/'!")
    if ":8421" not in hostname:
        raise ValueError("8421")
    eps = n / 30
    await asyncio.sleep(eps + np.random.uniform(low=0, high=eps))
    config = launch_experiment(hostname, n=n)
    responses = _munge("io/next-fig3.json.zip")
    responses = responses.sort_values(by="timestamp")
    responses["secs_from_start"] = (
        responses["timestamp"] - responses["timestamp"].min()
    ).dt.total_seconds()
    users = {}
    for k, puid in enumerate(responses.puid.unique()):
        user = responses[responses.puid == puid]
        response_times = user["response_time"].to_numpy()

        # This script can send about 40-50 responses/sec to Salmon when
        # unconstrainted
        #
        # If all users come to next as expected (in about 8 hours) the response
        # rate is always less than 7 responses/sec.
        users[k] = {
            "secs_till_start": user["secs_from_start"].to_numpy().min(),
            "response_times": response_times,
        }

    print(f"TOTAL USERS: {len(users)}")
    async with httpx.AsyncClient() as client:
        running_users = [
            simulate_user(
                config=config,
                client=client,
                hostname=hostname,
                responses=user_responses,
                puid=k + 1,
            )
            for k, user_responses in users.items()
        ]
        num_responses = await asyncio.gather(*running_users)
    print(f"#### Total of {sum(num_responses)} responses gathered")
    return sum(num_responses)


async def main(hostnames: Dict[int, str]) -> int:
    tasks = [run(n=n, hostname=hostname) for n, hostname in hostnames.items()]
    num_responses = await asyncio.gather(*tasks)
    return sum(num_responses)


if __name__ == "__main__":
    # Saves alg state to database
    # Relaunches algs on /init_exp
    # Restart: database restored, /init_exp never run.
    hostnames = {
        30:  "35.81.82.247",
        90:  "35.80.16.47",
        180: "44.234.20.96",
        300: "34.222.200.22",
    }
    hostnames = {k: f"http://{v}:8421" for k, v in hostnames.items()}
    print(hostnames)
    asyncio.run(main(hostnames))
    #  asyncio.run(run())
