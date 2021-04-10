from copy import deepcopy
from typing import List

import numpy as np


def _fmt(img: str) -> int:
    start = img.find("'")
    img = img[start + 1 :]
    end = img.find("'")
    img = img[:end]
    fname = img.split("/")[-1].strip("i.png")
    return int(fname)


def _get_new_targets(n: int, base: List[int] = None, seed=42) -> List[int]:
    rng = np.random.RandomState(seed)
    base = base or []
    choices = [i for i in range(600) if i not in base]
    new_targets = rng.choice(choices, size=n - len(base), replace=False).tolist()
    targets = np.sort(np.unique(base + new_targets))
    return targets.tolist()


def _get_old_targets():
    imgs = [
        "<img src='/static/targets/i0022.png' />",
        "<img src='/static/targets/i0036.png' />",
        "<img src='/static/targets/i0050.png' />",
        "<img src='/static/targets/i0074.png' />",
        "<img src='/static/targets/i0076.png' />",
        "<img src='/static/targets/i0112.png' />",
        "<img src='/static/targets/i0114.png' />",
        "<img src='/static/targets/i0126.png' />",
        "<img src='/static/targets/i0142.png' />",
        "<img src='/static/targets/i0152.png' />",
        "<img src='/static/targets/i0184.png' />",
        "<img src='/static/targets/i0194.png' />",
        "<img src='/static/targets/i0200.png' />",
        "<img src='/static/targets/i0208.png' />",
        "<img src='/static/targets/i0220.png' />",
        "<img src='/static/targets/i0254.png' />",
        "<img src='/static/targets/i0256.png' />",
        "<img src='/static/targets/i0312.png' />",
        "<img src='/static/targets/i0322.png' />",
        "<img src='/static/targets/i0326.png' />",
        "<img src='/static/targets/i0414.png' />",
        "<img src='/static/targets/i0420.png' />",
        "<img src='/static/targets/i0430.png' />",
        "<img src='/static/targets/i0438.png' />",
        "<img src='/static/targets/i0454.png' />",
        "<img src='/static/targets/i0470.png' />",
        "<img src='/static/targets/i0494.png' />",
        "<img src='/static/targets/i0524.png' />",
        "<img src='/static/targets/i0526.png' />",
        "<img src='/static/targets/i0572.png' />",
    ]

    return list(sorted(_fmt(img) for img in imgs))


def get(num: int) -> List[int]:
    targets = _get_old_targets()
    new_targets = deepcopy(targets)
    n = len(targets)
    assert n == 30
    valid_nums = [n, n * 3, n * 6, n * 10]
    if num not in valid_nums:
        raise ValueError(f"num={num} not in {valid_nums}")
    out = {}
    for _num in valid_nums:
        new_targets = _get_new_targets(_num, base=new_targets, seed=_num)
        assert set(targets).issubset(set(new_targets))
        assert len(new_targets) == _num
        out[_num] = deepcopy(new_targets)
    assert len(out) == len(valid_nums)
    assert {len(v) for v in out.values()} == set(valid_nums)
    return out[num]

if __name__ == "__main__":
    n = 30
    for mul in [1, 3, 6, 10]:
        out = get(n * mul)
        assert len(out) == n * mul
