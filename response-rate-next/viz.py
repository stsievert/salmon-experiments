import stats
import pandas as pd
import msgpack
import warnings

import targets
from generate_embeddings import _X_test


def _stats(datum, T=None, X_test=None):
    em = datum["embedding"]
    n_responses = datum["meta"]["n_train"]
    fnames = pd.read_csv("targets.csv.zip", header=None)[0].tolist()
   
    if T is None:
        T = targets.get(90)
    if X_test is None:
        X_test = _X_test(T)
        
        
    s = stats.collect(em, T, X_test)
    meta2 = {f"meta__{k}": v for k, v in datum["meta"].items()}
    return {**s, **meta2}

def process(file, **kwargs):
    with open(file, "rb") as f:
        history = msgpack.load(f)
    data = [_stats(hist) for hist in history]
    return [{**kwargs, **d} for d in data]


def lineplot(data, x, y, hue, style=None, hue_order=None, ci=0.25, ax=None, estimator="median"):
    if ax is None:
        fig, ax = plt.subplots()
    if hue_order is None:
        hue_order = sorted(data[hue].unique())
    for k, h in enumerate(hue_order):
        show = data[data[hue] == h]
        kwargs = dict(index=x, values=y)
        middle = show.pivot_table(aggfunc=estimator, **kwargs)
        if not len(middle):
            continue
        ax.plot(middle, f"{style}C{k}", label=h)
        lower = show.pivot_table(aggfunc=lambda x: x.quantile(q=ci), **kwargs)
        upper = show.pivot_table(aggfunc=lambda x: x.quantile(q=1 - ci), **kwargs)
        assert (lower.index == upper.index).all()
        ax.fill_between(lower.index.values, y1=lower.values.flatten(), y2=upper.values.flatten(), color=f"C{k}", alpha=0.2)
    ax.legend(loc="best")
    return ax