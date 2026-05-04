"""Build /workspace/Urban2026_pseudo/ — original train data + pseudo-labeled query+gallery.

Reads `results/pseudo_pairs.csv` (output of pseudo_label_extract.py) and merges
those rows into Urban2026/train.csv with the new pseudo-IDs (offset 1200+).

Output structure (mirrors Urban2026/):
    Urban2026_pseudo/
        train.csv            (11175 original rows + ~N_pseudo pseudo rows)
        train_classes.csv    (kept for reference, not used by training)
        query.csv, query_classes.csv, test.csv, test_classes.csv  (symlinks)
        image_train/         (dir containing symlinks to all original train images
                              + symlinks to pseudo-pair query and gallery images)
        image_query/, image_test/  (symlinks to original)

The pseudo-IDs (1200+) are higher than the max original ID, so the dataset
class's pid_container will assign them new contiguous indices automatically.
"""
import csv
import os
import shutil

import pandas as pd

SRC = '/workspace/Urban2026'
DST = '/workspace/Urban2026_pseudo'
PSEUDO_CSV = '/workspace/miuam_challenge_diff/results/pseudo_pairs.csv'


def main():
    if not os.path.exists(PSEUDO_CSV):
        raise SystemExit(f"  {PSEUDO_CSV} not found — run pseudo_label_extract.py first")

    if os.path.exists(DST):
        print(f"  {DST} exists; removing for clean rebuild")
        shutil.rmtree(DST)
    os.makedirs(DST)

    # 1. Load original train.csv (cameraID,imageName,Corresponding Indexes) + pseudo_pairs.csv
    orig = pd.read_csv(os.path.join(SRC, 'train.csv'))
    pseudo = pd.read_csv(PSEUDO_CSV)
    print(f"  original train rows: {len(orig)}, pseudo rows: {len(pseudo)}")
    print(f"  pseudo unique pseudo_ids: {pseudo['objectID'].nunique()}")
    PID_COL = 'Corresponding Indexes'

    # 3. Symlink other CSVs unchanged
    for fname in ['query.csv', 'query_classes.csv', 'test.csv',
                  'test_classes.csv', 'train_classes.csv']:
        os.symlink(os.path.join(SRC, fname), os.path.join(DST, fname))

    # 4. Symlink image_query and image_test directories unchanged
    os.symlink(os.path.join(SRC, 'image_query'), os.path.join(DST, 'image_query'))
    os.symlink(os.path.join(SRC, 'image_test'),  os.path.join(DST, 'image_test'))

    # 5. Build image_train/ with symlinks to:
    #    - all original training images
    #    - pseudo-paired query images (from image_query/)
    #    - pseudo-paired gallery images (from image_test/)
    img_train_dir = os.path.join(DST, 'image_train')
    os.makedirs(img_train_dir)
    src_train_imgs = os.path.join(SRC, 'image_train')
    for img in orig['imageName']:
        os.symlink(os.path.join(src_train_imgs, img), os.path.join(img_train_dir, img))

    # Pseudo: source can be 'query' or 'gallery'; use the source field to pick dir
    src_query = os.path.join(SRC, 'image_query')
    src_gallery = os.path.join(SRC, 'image_test')
    n_query = n_gallery = 0
    linked = set()  # names already symlinked, to dedupe gallery-claimed-twice cases
    drop_rows = []  # indices in pseudo to drop from final train.csv (lost in dedupe)
    for idx, row in pseudo.iterrows():
        src_dir = src_query if row['source'] == 'query' else src_gallery
        dst_name = f"pseudo_{row['source']}_{row['imageName']}"
        if dst_name in linked:
            # Same gallery image already claimed by another (more confident) pseudo-ID
            drop_rows.append(idx)
            continue
        os.symlink(os.path.join(src_dir, row['imageName']),
                   os.path.join(img_train_dir, dst_name))
        linked.add(dst_name)
        if row['source'] == 'query':
            n_query += 1
        else:
            n_gallery += 1
    if drop_rows:
        print(f"  deduped {len(drop_rows)} pseudo rows (gallery claimed twice)")
        pseudo = pseudo.drop(drop_rows).reset_index(drop=True)

    # Build the final combined train.csv with column name 'Corresponding Indexes'
    # to match the dataset class's reader (it parses row[2] regardless of name).
    fresh = orig.copy()
    pseudo_renamed = pseudo[['cameraID', 'imageName', 'objectID', 'source']].copy()
    pseudo_renamed['imageName'] = ('pseudo_' + pseudo_renamed['source'] + '_'
                                   + pseudo_renamed['imageName'])
    pseudo_renamed = pseudo_renamed.drop(columns=['source'])
    pseudo_renamed.columns = ['cameraID', 'imageName', PID_COL]
    final_combined = pd.concat([fresh, pseudo_renamed], ignore_index=True)
    final_combined.to_csv(os.path.join(DST, 'train.csv'), index=False)

    print(f"  symlinked {len(orig)} original train + {n_query} pseudo-query + {n_gallery} pseudo-gallery")
    print(f"  final train.csv: {len(final_combined)} rows, {final_combined[PID_COL].nunique()} unique IDs")

    # Sanity check
    sample_pseudo = sorted([f for f in os.listdir(img_train_dir) if f.startswith('pseudo_')])[0]
    resolved = os.path.realpath(os.path.join(img_train_dir, sample_pseudo))
    print(f"  sample pseudo symlink: {sample_pseudo} → {resolved}  exists: {os.path.exists(resolved)}")


if __name__ == '__main__':
    main()
