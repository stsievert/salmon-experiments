import warnings
import sys
import msgpack
from typing import List, Dict, Any
from zipfile import ZipFile
import json

import pandas as pd
from distributed import Client

from generate_embeddings import _X_test_viz, _X_test
import targets
import viz


def _get_embeddings(zip_fname: str, history=False, arr_priority=False) -> List[dict]:
    rares = []
    with ZipFile(zip_fname) as zf:
        for fname in zf.namelist():
            with zf.open(fname) as f:
                raw = f.read()
            rare = msgpack.loads(raw)
            if not history:
                _ = rare.pop("history")
            if (
                rare["meta"]["sampling"] == "salmon"
                and rare["meta"]["meta__alg"] == "TSTE"
            ):
                rare["meta"]["sampling"] = "salmon-tste"
            fname = rare["meta"].get("meta__fname", "[random]")

            if arr_priority and "ARR-" in fname and "scores=" in fname:
                i = fname.find("ARR-")
                fname = fname[i + 4 :]
                j = fname.find(".")
                _p = fname[:j]
                _, priority = _p.split("=")
                rare["meta"]["priority"] = priority
                rares.append(rare)
                continue

            if "rate" in fname:
                # fname = 'next/io/2021-05-24/rate=0.5_responses.msgpack'
                rate = fname.split("/")[-1].replace("_responses.msgpack", "")
                s = "dict(" + rate + ")"
                d = eval(s)
                rare["meta"].update(d)
            if "ARR-" in fname and "n_top" not in fname:
                i = fname.find("ARR-")
                fname = fname[i + 4 :]
                j = fname.find("_")
                rate = fname[:j]
                rare["meta"]["rate"] = float(rate)

            rares.append(rare)
    print("n_embeddings =", len(rares))
    return rares


def _cook(rare: Dict[str, Any], T, X_test) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        mrare = viz._stats(rare, T=T, X_test=X_test)
    return mrare


def _write_jobs(jobs, filename):
    jobs = [j["meta"] for j in jobs]
    with open(filename, "w") as f:
        json.dump(jobs, f)

if __name__ == "__main__":
    em = _get_embeddings("_io/embeddings-v5.zip")
    em += _get_embeddings("_io/embeddings-arr-priority.zip", arr_priority=True)
    #  sys.exit(1)

    rares = em

    T = targets.get(90)
    X_test = _X_test_viz(T)
    #  X_test = _X_test(T)
    print("X_test.shape =", X_test.shape)
    client = Client()
    T_f = client.scatter(T)
    X_test_f = client.scatter(X_test)
    futures = client.map(viz._stats, rares, T=T_f, X_test=X_test_f)
    data = client.gather(futures)
    df = pd.DataFrame(data)
    df.to_csv("_viz_data.csv")
