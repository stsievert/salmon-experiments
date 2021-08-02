import sys
import numpy as np
from copy import deepcopy
from pathlib import Path
import shutil
sys.path.append("../response-rate-next")

import targets

def _new_target(t, T, rng):
    if t - 1 not in T and t + 1 not in T:
        return rng.choice([t - 1, t + 1])
    elif t - 1 not in T and t + 1 in T:
        return t - 1
    elif t - 1 in T and t + 1 not in T:
        return t + 1
    else:
        sign = rng.choice([-1, 1])
        return _new_target(t + 2 * sign, T, rng)

if __name__ == "__main__":
    T = targets.get(90)

    rng = np.random.RandomState(42)
    out = deepcopy(T)
    out[-1] = 598
    for k, t in enumerate(out):
        if t % 2:
            out[k] = _new_target(t, out, rng)
    assert len(out) == len(set(out)) == 90

    IN_DIR = Path("alienegg300")
    OUT_DIR = Path("targets")
    for p in OUT_DIR.glob("*"):
        p.unlink()
    for idx in out:
        zeros = "0" * (3 - len(str(idx)))
        fname = f"i{zeros}{idx}.jpg"
        shutil.copy(IN_DIR / fname, OUT_DIR / fname)
