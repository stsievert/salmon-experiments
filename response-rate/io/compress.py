from pathlib import Path
import pandas as pd
import msgpack
import io

def compress_and_write(in_file, out_dir):
    with open(in_file, "rb") as f:
        raw = msgpack.load(f)

    df = pd.DataFrame(raw["history"])
    with io.BytesIO() as f:
        df.to_parquet(f, compression="brotli")
        serialized_df = f.getvalue()

    raw.pop("history")

    with open(out_dir / in_file.name, "wb") as f:
        msgpack.dump(raw, f)

if __name__ == "__main__":
    DIR = Path("2021-03-27")
    IN = "embeddings"
    OUT = "out"
    files = (DIR / IN).glob("*.msgpack")

    for in_file in files:
        compress_and_write(in_file, DIR / OUT)

