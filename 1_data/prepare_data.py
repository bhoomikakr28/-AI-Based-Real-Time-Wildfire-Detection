import os, shutil, random
from pathlib import Path

SRC   = Path("1_data/raw")
DEST  = Path("1_data/processed")
SPLIT = 0.8

for label in ["fire", "no_fire"]:
    images = list((SRC / label).glob("*.*"))
    random.shuffle(images)
    split  = int(len(images) * SPLIT)
    for phase, imgs in [("train", images[:split]), ("val", images[split:])]:
        out = DEST / phase / label
        out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, out / img.name)

print("✅ Data prepared successfully!")