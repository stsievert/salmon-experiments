"""
This file takes in responses and generates embeddings.
It saves these embeddings to disk.

Input: Responses.
Output: Embeddings.
"""

import pandas as pd
from pathlib import Path
from dask.distributed import Client, as_completed
import torch
import numpy as np

import dask
from sklearn.model_selection import train_test_split
from salmon.triplets.offline import OfflineEmbedding


class Embedding:
    def __init__(self, n, d, max_epochs=400, **kwargs):
        self.n = n
        self.d = d
        self.max_epochs = max_epochs
        self.kwargs = kwargs

    def init(self):
        self.model_ = OfflineEmbedding(
            n=self.n, d=self.d, max_epochs=self.max_epochs, **self.kwargs
        )

    def fit(self, *args, **kwargs):
        self.init()
        self.model_.fit(*args, **kwargs)
        return self

    @property
    def embedding_(self):
        return self.model_.embedding_

    @property
    def history_(self):
        return self.model_.history_


def _get_unique(col: pd.Series):
    assert col.nunique() == 1
    return col.iloc[0]


def _get_trained_model(
    X_train,
    X_test=None,
    n_responses=None,
    d=2,
    meta=None,
    scores=None,
    threads=None,
    ident=None,
    noise_model=None,
    alg=None,
    fname=None,
    **kwargs,
):
    if meta is None:
        meta = {}
    if threads:
        torch.set_num_threads(threads)
    meta = meta.copy()
    meta["ident"] = ident
    meta.update({f"est__{k}": v for k, v in kwargs.items()})
    n = X_train.max() + 1

    if n_responses:
        X_train = X_train[:n_responses]
        if scores is not None:
            scores = scores[:n_responses]
    _update = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n": n,
        "d": d,
        "n_responses": n_responses,
        "noise_model": noise_model,
        "ident": ident,
        "threads": threads,
        "alg": alg,
        "fname": fname,
    }
    meta.update(_update)

    est = Embedding(n=n, d=d, ident=ident, **kwargs)
    est.fit(X_train, X_test)
    return est, meta
