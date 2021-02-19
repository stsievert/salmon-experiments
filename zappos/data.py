from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

Query = Tuple[int, int, int]


def _find(q: Query, responses: pd.DataFrame) -> List[int]:
    h, o1, o2 = q
    idx = responses["h"] == h
    idx &= responses["w"].isin([o1, o2])
    idx &= responses["l"].isin([o1, o2])
    return idx[idx].index.values.tolist()


def _get_split(X, random_state=42):
    rng = check_random_state(random_state)
    targets = np.unique(X)
    n = X.max() + 1
    targets = set(range(n))

    # Make sure we only have one copy of each query (not two)
    _queries = {
        (h, min(o1, o2), max(o1, o2))
        for h in targets
        for o1 in targets - {h}
        for o2 in targets - {h, o1}
    }
    queries = list(sorted(list(_queries)))
    rng.shuffle(queries)
    train = set()
    test = set()
    bad = 0
    for k, query in enumerate(queries):
        idx = _find(query, df)
        if len(idx) == 4 and rng.uniform() <= 0.75:
            # This branch happens for about 97% of the queries.
            r = rng.choice(len(idx))
            test_i = idx[r]
            idx.remove(idx[r])
            assert len(idx) == 3
            test.add(test_i)
            train.update([i for i in idx])
        elif len(idx):
            train.update(set(idx))
        if k % 1000 == 0:
            print(k, k / len(queries))

    train_idx = list(sorted(list(train)))
    test_idx = list(sorted(list(test)))
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    return X_train, X_test


if __name__ == "__main__":
    df = pd.read_csv("dataset/zappos.csv.zip")
    df = df[["head", "b", "c"]]
    df.columns = ["h", "w", "l"]
    X = df[["h", "w", "l"]].to_numpy()

    np.save("io/X.npy", X)

    X_train, X_test = _get_split(X)

    np.save("io/X_train.npy", X_train)
    np.save("io/X_test.npy", X_test)
