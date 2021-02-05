import numpy as np

def _divide(n):
    div = len(n) // 2
    idx = n.find(".", 2)
    return n[:idx - 1], n[idx + 1:]


def _reduce(x):
    out = []
    for xi in x:
        if isinstance(xi, tuple):
            for _ in xi:
                out.append(float(_))
        else:
            out.append(float(xi))
    return out

if __name__ == "__main__":
    n = 85
    with open("cnn_feats.csv") as f:
        lines = f.readlines()
    assert len(lines) == 1
    txt = lines[0]
    raw = txt.split(",")
    newlines = [i for i, n in enumerate(raw) if n.count(".") == 2]
    mrare = [(n, ) if i not in newlines else _divide(n) for i, n in enumerate(raw)]
    medium = _reduce(mrare)

    mwell = [medium[k * 4096 : (k + 1) * 4096] for k in range(n)]
    well_done = np.array(mwell)
