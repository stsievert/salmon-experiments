from zipfile import ZipFile
from pathlib import Path
import msgpack

if __name__ == "__main__":
    with ZipFile("io/2021-04-09/embeddings.zip") as zf:
        print([f.filename for f in zf.filelist])
        for f in zf.filelist:
            assert ".msgpack" in f.filename
            ir = zf.read(f)
            data = msgpack.loads(ir)
            assert set(data.keys()) == {
                "embedding",
                "history",
                "perf",
                "params",
                "meta",
            }
