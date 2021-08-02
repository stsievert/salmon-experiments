import pandas as pd
from pathlib import Path
from typing import Dict, Any

def _get_meta(fname: str) -> Dict[str, Any]:
    _meta = fname.replace(".csv", "").replace("Salmon-", "")
    _meta2 = _meta.split("-")
    _meta3 = {m.split("=")[0]: m.split("=")[1] for m in _meta2}
    meta = {k: int(v) if v.isdigit() else v for k, v in _meta3.items()}
    return meta

if __name__ == "__main__":
    DIR = Path("responses")
    mappings = []
    for f in DIR.glob("*.csv"):
        if "Salmon" in f.name:
            meta = _get_meta(f.name)
        elif "NEXT" in f.name:
            meta = {"n": 30}
        df = pd.read_csv(f)

        assert set(df["head"].unique()) == set(range(meta["n"]))
        _mapping = df.groupby("head")["head_filename"].unique()
        mapping = {k: v[0] for k, v in _mapping.items()}
        mappings.append(mapping)

    assert len(mappings) == 14
    m30 = [m for m in mappings if len(m) == 30]
    m90 = [m for m in mappings if len(m) == 90]
    assert len(m30) == 10  # NEXT * 5 + Salmon * 5
    # NEXT   : CK   , random , TSTE   , Test , US.
    # Salmon : arr1 , arr2   , random , SAAR , test
    assert len(m90) == 4  # Salmon(arr, random, SAAR, testing)
