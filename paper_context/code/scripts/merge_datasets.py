"""Merge Urban2026 (challenge) + UAM_Unified (external UrbAM-ReID) training data.

Output goes to /workspace/Urban2026_merged/. Keeps the original challenge
query/gallery untouched (via symlink) — only the train split is augmented.

Steps:
1. Symlink each challenge training image into merged/image_train/ with its
   original name.
2. Symlink each UAM training image into merged/image_train/ with an 'uam_'
   prefix (e.g. uam_000001.jpg) to avoid filename collisions.
3. Offset UAM objectIDs by +1090 (max challenge ID) so identity spaces don't
   collide. New UAM IDs land in [1091, 1569].
4. Normalize UAM's 'trafficsign' class label to 'trafficsignal' (challenge form).
5. Concatenate train.csv + train_classes.csv files.

Run once; idempotent (overwrites existing symlinks + CSVs).
"""
import csv
import os
from pathlib import Path

CHAL_ROOT = Path('/workspace/Urban2026')
EXT_ROOT  = Path('/workspace/UAM_Unified_extract/UAM_Unified')
OUT_ROOT  = Path('/workspace/Urban2026_merged')

OUT_ROOT.mkdir(exist_ok=True)
img_out = OUT_ROOT / 'image_train'
img_out.mkdir(exist_ok=True)

# --- 1. Challenge training images: symlink with original names ---
chal_imgs = sorted((CHAL_ROOT / 'image_train').iterdir())
print(f"symlinking {len(chal_imgs)} challenge training images...")
for p in chal_imgs:
    dst = img_out / p.name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(p.resolve())

# --- 2. UAM training images: symlink with 'uam_' prefix ---
ext_imgs = sorted((EXT_ROOT / 'image_train').iterdir())
print(f"symlinking {len(ext_imgs)} UAM training images (with uam_ prefix)...")
for p in ext_imgs:
    dst = img_out / f"uam_{p.name}"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(p.resolve())

# --- 3. Figure out ID offset ---
def max_chal_id():
    with open(CHAL_ROOT / 'train.csv') as f:
        reader = csv.DictReader(f)
        return max(int(row['Corresponding Indexes']) for row in reader)
ID_OFFSET = max_chal_id()  # 1090 observed
print(f"ID_OFFSET (max challenge ID) = {ID_OFFSET}")

# --- 4. Merge train.csv ---
def read_rows(path, img_name_fn):
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = []
        for row in reader:
            rows.append((row['cameraID'],
                         img_name_fn(row['imageName']),
                         int(row.get('Corresponding Indexes', row.get('objectID')))))
        return fields, rows

chal_fields, chal_rows = read_rows(CHAL_ROOT / 'train.csv', lambda n: n)
ext_fields,  ext_rows  = read_rows(EXT_ROOT  / 'train.csv', lambda n: f"uam_{n}")

# Verify ext max ID
ext_max = max(r[2] for r in ext_rows)
print(f"UAM max objectID (pre-shift) = {ext_max}; will remap to [{ID_OFFSET+1}, {ID_OFFSET+ext_max}]")

# Write merged train.csv with a single consistent 3-column schema:
#   cameraID, imageName, Corresponding Indexes
merged_train = OUT_ROOT / 'train.csv'
with open(merged_train, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['cameraID', 'imageName', 'Corresponding Indexes'])
    for cam, name, pid in chal_rows:
        w.writerow([cam, name, f"{pid:04d}"])
    for cam, name, pid in ext_rows:
        w.writerow([cam, name, f"{pid + ID_OFFSET:04d}"])
print(f"wrote {merged_train} ({len(chal_rows)} + {len(ext_rows)} = {len(chal_rows)+len(ext_rows)} rows)")

# --- 5. Merge train_classes.csv with class normalization ---
CLASS_CANON = {
    'container':     'Container',
    'crosswalk':     'Crosswalk',
    'rubbishbins':   'RubbishBins',
    'trafficsign':   'trafficsignal',   # UAM uses 'trafficsign'; challenge uses 'trafficsignal'
    'trafficsignal': 'trafficsignal',
}

def read_class_rows(path, img_name_fn, pid_shift):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row.get('Corresponding Indexes', row.get('objectID')))
            cls_raw = row['Class'].strip()
            cls_canon = CLASS_CANON.get(cls_raw.lower())
            if cls_canon is None:
                raise ValueError(f"Unknown class label: {cls_raw!r} in {path}")
            rows.append((row['cameraID'],
                         img_name_fn(row['imageName']),
                         pid + pid_shift,
                         cls_canon))
    return rows

chal_cls = read_class_rows(CHAL_ROOT / 'train_classes.csv', lambda n: n,                0)
ext_cls  = read_class_rows(EXT_ROOT  / 'train_classes.csv', lambda n: f"uam_{n}", ID_OFFSET)

merged_classes = OUT_ROOT / 'train_classes.csv'
with open(merged_classes, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['cameraID', 'imageName', 'Corresponding Indexes', 'Class'])
    for cam, name, pid, cls in chal_cls + ext_cls:
        w.writerow([cam, name, f"{pid:04d}", cls])
print(f"wrote {merged_classes} ({len(chal_cls)} + {len(ext_cls)} = {len(chal_cls)+len(ext_cls)} rows)")

# --- 6. Summary ---
print()
print("=== MERGE COMPLETE ===")
print(f"merged train images:   {len(list(img_out.iterdir()))}")
print(f"merged train IDs total: {max_chal_id() + ext_max}")
