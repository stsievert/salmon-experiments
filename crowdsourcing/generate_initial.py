import pandas as pd
from pathlib import Path
import numpy as np
from distributed import Client, as_completed
import json

from salmon.triplets.offline import OfflineEmbedding


def _correct_answer(q):
    h, o1, o2 = q
    d1 = abs(h - o1)
    d2 = abs(h - o2)
    if d1 < d2:
        w, l = o1, o2
    else:
        w, l = o2, o1
    return h, w, l


def _random_query(n, i):
    rng = np.random.RandomState(i)
    return rng.choice(n, size=3, replace=False)


def _X_test(n=90, num=50_000):
    _X_test = [_correct_answer(_random_query(n, i)) for i in range(num)]
    X_test = np.array(_X_test)
    return X_test


def _get_em(X_train, n=30, d=2, max_epochs=1_000_000, verbose=None):
    if verbose is None:
        verbose = int(max_epochs // 100)
    est = OfflineEmbedding(
        n=n, d=d, max_epochs=max_epochs, verbose=verbose, random_state=42
    )

    X_test = _X_test()
    est.fit(X_train, X_test)
    meta = {
        "n": n,
        "d": d,
        "len_X_train": len(X_train),
        "max_epochs": max_epochs,
        "random_state": est.random_state,
    }
    return est, est.embedding_, meta


if __name__ == "__main__":
    DIR = Path("io/salmon-raw")

    client = Client()
    futures = []
    for n in [30, 90]:
        i = 1 if n == 30 else 3
        df = pd.read_csv(DIR / f"m{i}" / "responses.csv")
        assert df["head"].nunique() == n
        random = df[df.alg_ident == "RandomSampling"]
        X = random[["head", "winner", "loser"]].to_numpy()
        X_train = X[: 10 * n]
        _futures = [client.submit(_get_em, X_train, n=n, d=d) for d in [1, 2]]
        futures.extend(_futures)
        print(len(futures))

    for future in as_completed(futures):
        est, em, meta = future.result()
        fname = f"n={meta['n']}-d={meta['d']}"
        with open(f"{fname}.json", "w") as f:
            json.dump({"em": em.tolist(), "meta": meta, "history": est.history_}, f)
