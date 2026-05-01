"""Prepare CycleGAN training data: c001-c003 train -> c004 query.

Resizes images to 256x128 (PAT's native input resolution) and writes them to
/workspace/cyclegan_data/urban_c4/{trainA,trainB,testA,testB}/.

trainA: ALL c001-c003 train images (11,175)  - source style
trainB: ALL c004 query images (928)         - target style
testA: sample of trainA (for GAN visualization)
testB: sample of trainB (for GAN visualization)
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd

SRC = Path('/workspace/Urban2026')
DST = Path('/workspace/cyclegan_data/urban_c4')
TARGET_SIZE = (128, 256)  # (W, H) = PAT input

def resize_save(src_path: Path, dst_path: Path):
    img = Image.open(src_path).convert('RGB')
    img = img.resize(TARGET_SIZE, Image.BICUBIC)
    img.save(dst_path, 'JPEG', quality=95)

def main():
    if DST.exists():
        print(f"Removing existing {DST}")
        shutil.rmtree(DST)
    for sub in ('trainA', 'trainB', 'testA', 'testB'):
        (DST / sub).mkdir(parents=True)

    # trainA: all c001-c003 train images
    src_train = SRC / 'image_train'
    train_files = sorted(os.listdir(src_train))
    print(f"trainA: {len(train_files)} files (c001-c003)")
    for i, f in enumerate(train_files):
        resize_save(src_train / f, DST / 'trainA' / f)
        if i % 2000 == 0:
            print(f"  trainA {i}/{len(train_files)}")

    # trainB: all c004 query images
    src_query = SRC / 'image_query'
    query_files = sorted(os.listdir(src_query))
    print(f"trainB: {len(query_files)} files (c004)")
    for i, f in enumerate(query_files):
        resize_save(src_query / f, DST / 'trainB' / f)

    # testA / testB: small samples for visualization during training
    for f in train_files[:20]:
        resize_save(src_train / f, DST / 'testA' / f)
    for f in query_files[:20]:
        resize_save(src_query / f, DST / 'testB' / f)
    print(f"\nDone. Structure at {DST}/")
    for sub in ('trainA', 'trainB', 'testA', 'testB'):
        n = len(os.listdir(DST / sub))
        print(f"  {sub}: {n} images")

if __name__ == '__main__':
    main()
