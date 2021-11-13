import stats_next as stats
import pandas as pd
import msgpack
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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


def lineplot(
    data, x, y, hue, style="-", hue_order=None, ci=0.25, ax=None, estimator="median", palette="copper"
):
    if ax is None:
        fig, ax = plt.subplots()
    if hue_order is None:
        hue_order = sorted(data[hue].unique())
    if isinstance(palette, list):
        colors = palette
    else:
        cmap = mpl.cm.get_cmap(palette)
        colors = [cmap(x) for x in np.linspace(0, 1, num=len(hue_order))]
    for k, (h, color) in enumerate(zip(hue_order, colors)):
        show = data[data[hue] == h]
        kwargs = dict(index=x, values=y)
        middle = show.pivot_table(aggfunc=estimator, **kwargs)
        if not len(middle):
            continue
        _style = style if "C" not in style else style.format(k=k)
        if isinstance(style, list):
            _style = style[k]
        
        ax.plot(middle, _style, label=h, color=color)
        if ci > 0:
            lower = show.pivot_table(aggfunc=lambda x: x.quantile(q=ci), **kwargs)
            upper = show.pivot_table(aggfunc=lambda x: x.quantile(q=1 - ci), **kwargs)
            assert (lower.index == upper.index).all()
            ax.fill_between(
                lower.index.values,
                y1=lower.values.flatten(),
                y2=upper.values.flatten(),
                color=color,
                alpha=0.2,
            )
    ax.legend(loc="best")
    return ax
