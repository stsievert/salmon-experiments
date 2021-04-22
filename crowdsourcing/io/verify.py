from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    ROOT = Path(__file__).absolute().parent / "2021-04-16"
    N = [30, 90, 180, 300]
    for n in N:
        print(f"Verifying n={n}...", end=" ")
        df = pd.read_csv(ROOT / f"n={n}" / "responses.csv")
        targets = df[["head", "left", "right"]]
        assert (targets.nunique() == n).all()
        print("done")
