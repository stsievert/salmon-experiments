import os
from pathlib import Path

IN_DIR = "alienegg90"
OUT_DIR = "alienegg90-cropped"

for f in Path(IN_DIR).glob("*.jpg"):
    cmd = f"sips -c 256 256 {IN_DIR}/{f.name} --out {OUT_DIR}/"
    os.system(cmd)
