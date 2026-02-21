import os
from pathlib import Path

base = Path("data/dataset")

for split in ["train", "val"]:
    print("\n", split.upper())
    for cl in sorted([p.name for p in (base/split).iterdir() if p.is_dir()]):
        count = len(list((base/split/cl).glob("*.*")))
        print(cl, count)




