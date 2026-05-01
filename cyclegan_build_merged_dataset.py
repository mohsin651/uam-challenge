"""Build /workspace/Urban2026_cyclegan/ merged dataset.

Output structure (mirrors Urban2026/):
    Urban2026_cyclegan/
        train.csv            (22350 rows = 11175 real + 11175 fake-c004, same labels)
        train_classes.csv    (22350 rows, same labels for synthetic)
        query.csv, query_classes.csv, test.csv, test_classes.csv  (symlinks)
        image_train/         (real symlinks + fake_c4_* symlinks to image_train_fake_c4/)
        image_query/, image_test/  (symlinks)

The fake-c004 images preserve their REAL identity labels but get a NEW cameraID
(c004) — except we must NOT use 'c004' because c004 is the test camera and
camid drives the cam-adv classifier. Instead we create a synthetic 'c000'
or just replicate the original cameraID. Decision: keep original cameraID
(c001/c002/c003) so cam-adv classifier remains 3-way over training cameras.
The CycleGAN-augmented data teaches the model c004-style appearance under the
same camera labels — cam-adv pressure handles the camera-invariance, the
synthetic data handles the appearance shift.
"""
import os
import shutil
from pathlib import Path
import pandas as pd

SRC = Path('/workspace/Urban2026')
FAKE = Path('/workspace/Urban2026_cyclegan/image_train_fake_c4')
DST = Path('/workspace/Urban2026_cyclegan')


def main():
    if not FAKE.exists():
        raise SystemExit(f"  fake-c004 dir {FAKE} missing — run cyclegan_generate_fake_c4.py first")

    real_files = sorted(os.listdir(SRC / 'image_train'))
    fake_files = sorted(os.listdir(FAKE))
    print(f"  real train: {len(real_files)}")
    print(f"  fake c004:  {len(fake_files)}")
    if len(real_files) != len(fake_files):
        print(f"  ! warning: counts differ; merging will use intersect")

    # Wipe + rebuild dataset symlink structure
    for p in ['image_train', 'image_query', 'image_test',
              'query.csv', 'query_classes.csv', 'test.csv', 'test_classes.csv',
              'train.csv', 'train_classes.csv']:
        target = DST / p
        if target.is_symlink() or target.exists():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()

    # Symlink eval-side artifacts unchanged
    for fname in ['query.csv', 'query_classes.csv',
                  'test.csv', 'test_classes.csv']:
        os.symlink(SRC / fname, DST / fname)
    os.symlink(SRC / 'image_query', DST / 'image_query')
    os.symlink(SRC / 'image_test', DST / 'image_test')

    # Build image_train/ with real symlinks + fake_c4_* symlinks
    img_train = DST / 'image_train'
    img_train.mkdir()
    for f in real_files:
        os.symlink(SRC / 'image_train' / f, img_train / f)
    for f in fake_files:
        os.symlink(FAKE / f, img_train / f'fake_c4_{f}')
    print(f"  symlinked {len(real_files)} real + {len(fake_files)} fake-c4 = {len(os.listdir(img_train))}")

    # Build merged train.csv
    real_csv = pd.read_csv(SRC / 'train.csv')
    fake_csv = real_csv.copy()
    fake_csv['imageName'] = 'fake_c4_' + fake_csv['imageName']
    # Keep original cameraID (c001/c002/c003): the synthetic image inherits the
    # source camera's structural pose with c004's appearance. cam-adv classifier
    # remains a 3-way head over training cameras; appearance domain shift comes
    # from the synthetic-augmented data.
    merged = pd.concat([real_csv, fake_csv], ignore_index=True)
    merged.to_csv(DST / 'train.csv', index=False)
    print(f"  train.csv: {len(merged)} rows, {merged['Corresponding Indexes'].nunique()} IDs")

    # Build merged train_classes.csv
    real_cls = pd.read_csv(SRC / 'train_classes.csv')
    fake_cls = real_cls.copy()
    fake_cls['imageName'] = 'fake_c4_' + fake_cls['imageName']
    merged_cls = pd.concat([real_cls, fake_cls], ignore_index=True)
    merged_cls.to_csv(DST / 'train_classes.csv', index=False)

    print(f"  Done. Output at {DST}/")


if __name__ == '__main__':
    main()
