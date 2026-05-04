"""Build a trafficsignal-only training subset at /workspace/Urban2026_trafficsignal/.

Filters Urban2026/train.csv + train_classes.csv to trafficsignal rows only,
symlinks the corresponding training images, and symlinks query/test data
unchanged (eval-mAP path is meaningless on this dataset anyway, but the
training dataset class still loads it).

Output structure (mirrors Urban2026/):
    Urban2026_trafficsignal/
        train.csv            (filtered: 7568 trafficsignal rows)
        train_classes.csv    (filtered, with Class column)
        query.csv            (symlink to original)
        query_classes.csv    (symlink to original)
        test.csv             (symlink to original)
        test_classes.csv     (symlink to original)
        image_train/         (dir with symlinks to trafficsignal images only)
        image_query/         (symlink to original dir)
        image_test/          (symlink to original dir)

PID handling: keep ORIGINAL pids in the CSVs. The dataset class
(UrbanElementsReID._process_dir) does its own relabeling via pid_container,
so it'll automatically build a contiguous 0..799 mapping at load time.
"""
import csv
import os
import shutil

SRC = '/workspace/Urban2026'
DST = '/workspace/Urban2026_trafficsignal'
TARGET_CLASS = 'trafficsignal'  # case-insensitive match


def main():
    if os.path.exists(DST):
        print(f"  {DST} exists; removing for clean rebuild")
        shutil.rmtree(DST)
    os.makedirs(DST)

    # 1. Read train_classes.csv (has Class column)
    with open(os.path.join(SRC, 'train_classes.csv'), newline='') as f:
        reader = csv.reader(f)
        header_classes = next(reader)
        rows_classes = list(reader)  # [cameraID, imageName, objectID, Class]

    # Filter to trafficsignal
    ts_rows = [r for r in rows_classes if r[3].lower() == TARGET_CLASS]
    ts_imgs = set(r[1] for r in ts_rows)
    ts_pids = set(r[2] for r in ts_rows)
    print(f"  trafficsignal: {len(ts_rows)} images, {len(ts_pids)} unique IDs")

    # 2. Write filtered train.csv (no Class column — matches base format)
    with open(os.path.join(SRC, 'train.csv'), newline='') as f:
        reader = csv.reader(f)
        header_train = next(reader)
        rows_train = list(reader)  # [cameraID, imageName, objectID]
    ts_train = [r for r in rows_train if r[1] in ts_imgs]
    assert len(ts_train) == len(ts_rows), f"mismatch: {len(ts_train)} vs {len(ts_rows)}"

    with open(os.path.join(DST, 'train.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header_train)
        w.writerows(ts_train)
    with open(os.path.join(DST, 'train_classes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header_classes)
        w.writerows(ts_rows)
    print(f"  wrote {DST}/train.csv ({len(ts_train)} rows)")

    # 3. Symlink query/test CSVs unchanged
    for fname in ['query.csv', 'query_classes.csv', 'test.csv', 'test_classes.csv']:
        os.symlink(os.path.join(SRC, fname), os.path.join(DST, fname))

    # 4. Symlink image_query and image_test directories unchanged
    os.symlink(os.path.join(SRC, 'image_query'), os.path.join(DST, 'image_query'))
    os.symlink(os.path.join(SRC, 'image_test'),  os.path.join(DST, 'image_test'))

    # 5. Create image_train/ with symlinks to ONLY trafficsignal images
    img_train_dir = os.path.join(DST, 'image_train')
    os.makedirs(img_train_dir)
    src_img_train = os.path.join(SRC, 'image_train')
    for img in ts_imgs:
        os.symlink(os.path.join(src_img_train, img), os.path.join(img_train_dir, img))
    print(f"  symlinked {len(ts_imgs)} images into {img_train_dir}")

    # 6. Sanity check
    print(f"\n  Sanity check on {DST}:")
    print(f"    train.csv rows:        {sum(1 for _ in open(os.path.join(DST, 'train.csv'))) - 1}")
    print(f"    image_train/ entries:  {len(os.listdir(img_train_dir))}")
    # Verify a couple of symlinks resolve
    sample = sorted(os.listdir(img_train_dir))[0]
    resolved = os.path.realpath(os.path.join(img_train_dir, sample))
    print(f"    sample symlink:        {sample} → {resolved}")
    print(f"    sample exists:         {os.path.exists(resolved)}")


if __name__ == '__main__':
    main()
