from pathlib import Path
import random
from re import S
import shutil

print("split_dataset.py started")
SEED = 42
VAL_RATIO = 0.2

def slipt_dataset():
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "data" / "images"
    out_root = project_root / "data" / "dataset"

    print("project root:",project_root)
    print("src root:", src_root)
    print("out root:", out_root)

    train_root = out_root / "train"
    val_root = out_root / "val"

    if not src_root.exists():
        raise FileNotFoundError(f"Missing {src_root}")

    random.seed(SEED)

    if out_root.exists():
        shutil.rmtree(out_root)
    train_root.mkdir(parents=True, exist_ok= True)
    val_root.mkdir(parents=True,exist_ok= True)

    classes = [p for p in src_root.iterdir() if p.is_dir()]
    if not classes: 
        raise RuntimeError(f"No class folder found in {src_root}")

    for class_dir in classes:
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]]
        random.shuffle(images)

        split = int(len(images) * (1 - VAL_RATIO))
        train_imgs = images[:split]
        val_imgs = images[split:]

        (train_root / class_dir.name).mkdir(parents= True,exist_ok= True)
        (val_root / class_dir.name).mkdir(parents= True,exist_ok= True)

        for p in train_imgs:
            shutil.copy2(p, train_root / class_dir.name / p.name)
        for p in val_imgs:
            shutil.copy2(p, val_root / class_dir.name / p.name)

        print(f"{class_dir.name}: train= {len(train_imgs)}, val={len(val_imgs)}")

    print("\nDone.")
    print("Train folder", train_root)
    print("Val folder", val_root)

if __name__ == "__main__":
   slipt_dataset()



