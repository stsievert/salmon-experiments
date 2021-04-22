import seaborn as sns
import matplotlib.pyplot as plt

def lineplot(data, x, y, hue, style, hue_order=None, ci=0.25, ax=None, estimator="median"):
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
        ax.plot(middle, label=h)
        lower = show.pivot_table(aggfunc=lambda x: x.quantile(q=ci), **kwargs)
        upper = show.pivot_table(aggfunc=lambda x: x.quantile(q=1 - ci), **kwargs)
        assert (lower.index == upper.index).all()
        ax.fill_between(lower.index.values, y1=lower.values.flatten(), y2=upper.values.flatten(), color=f"C{k}", alpha=0.2)
    ax.legend(loc="best")
    return ax