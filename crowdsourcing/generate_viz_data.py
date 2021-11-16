import warnings
import msgpack
from typing import List, Dict, Any
from zipfile import ZipFile
import json
from time import sleep

import yaml
import pandas as pd
import numpy as np
from distributed import Client

#  from generate_embeddings import _X_test_viz, _X_test
#  import targets
import viz


def _get_embeddings(zip_fname: str, history=False) -> List[dict]:
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


def _get_X_test(n=30, next_only=False) -> np.ndarray:
    if n == 30:
        df1 = pd.read_csv("io/responses/NEXT-Test.csv")
        df2 = pd.read_csv("io/responses/Salmon-n=30-alg_ident=testing.csv")
        df = df1.copy() if next_only else pd.concat((df1, df2))
    elif n == 90:
        df = pd.read_csv("io/responses/Salmon-n=90-alg_ident=testing.csv")

    X = df[["head", "winner", "loser"]].to_numpy()
    return X


def _get_targets(n):
    fnames = {30: "io/salmon-raw/m1/config.txt", 90: "io/salmon-raw/m3/config.txt"}
    if n not in fnames:
        raise ValueError(f"n={n} not recognized")
    with open(fnames[n], "r") as f:
        config = yaml.safe_load(f)
    targets = config["targets"]
    T = [int(t.split(" ")[1].split("=")[-1].split("/")[-1].strip("'i.png")) for t in targets]
    return list(sorted(T))


if __name__ == "__main__":
    em1 = _get_embeddings("io/embeddings.zip")
    rares = em1
    print("total embeddings: ", len(rares))
    sleep(1)

    targets = {n: _get_targets(n) for n in [30, 90]}

    dfs = []
    for next_only in [True, False]:
        X_tests = {n: _get_X_test(n=n, next_only=next_only) for n in [30, 90]}
        data = []
        for r in rares:
            n = r["meta"]["n"]
            if n == 90 and next_only == True:
                continue
            datum = viz._stats(r, X_test=X_tests[n], T=targets[n])
            data.append(datum)
        df = pd.DataFrame(data)
        df["next_only"] = next_only
        dfs.append(df)
    pd.concat(dfs).to_csv(f"_viz_data.csv")
