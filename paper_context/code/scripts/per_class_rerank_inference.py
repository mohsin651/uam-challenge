"""Per-class re-rank — isolate k-reciprocal computation to within-class galleries.

Standard pipeline applies global rerank then masks cross-class to inf. The
rerank's k-reciprocal step still uses ALL gallery as neighborhood context,
including cross-class galleries that get dropped at the final step. This is
"polluting" rerank by considering galleries that aren't valid candidates.

Per-class rerank: split queries+galleries by class group, run rerank on each
sub-problem independently. Each class's k-reciprocal context is now strictly
within-class.

Generates 2 high-EV variants:
  PRIMARY: per-class rerank with proven params (15, 4, 0.275) for all classes
  BACKUP: per-class rerank with class-tuned params (different K1/k2/lambda
          per class group based on gallery size)

Uses cached 8-ckpt features — runs in ~10 sec, no GPU needed.
"""
import csv
import os
import numpy as np
import pandas as pd

from utils.re_ranking import re_ranking


CACHE_DIR = '/workspace/miuam_challenge_diff/results/cache'
TEST_CACHE = f'{CACHE_DIR}/8ckpt_test.npz'
URBAN_ROOT = '/workspace/Urban2026/'

DBA_K = 8


def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def per_class_rerank(qf, gf, q_groups, g_groups, params_per_class):
    """For each class group, run re-ranking on its sub-problem independently.

    params_per_class: dict {group_name: (k1, k2, lambda)}
    Returns: (num_query, num_gallery) distance matrix, with cross-class set to +inf.
    """
    num_query, num_gallery = qf.shape[0], gf.shape[0]
    rrd = np.full((num_query, num_gallery), np.inf, dtype=np.float32)

    for grp in np.unique(q_groups):
        q_idx = np.where(q_groups == grp)[0]
        g_idx = np.where(g_groups == grp)[0]
        if len(q_idx) == 0 or len(g_idx) == 0:
            continue
        qf_g = qf[q_idx]
        gf_g = gf[g_idx]
        q_g = qf_g @ gf_g.T
        q_q = qf_g @ qf_g.T
        g_g = gf_g @ gf_g.T
        k1, k2, lam = params_per_class[grp]
        sub_rrd = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)
        # Place the sub-distance back into the global matrix
        for i, qi in enumerate(q_idx):
            rrd[qi, g_idx] = sub_rrd[i]
        print(f"  class '{grp}': {len(q_idx)} q × {len(g_idx)} g, params={params_per_class[grp]}")
    return rrd


def write_csv(rrd, label, num_query):
    indices = np.argsort(rrd, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, num_query + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"  → {out}")


def main():
    test_data = np.load(TEST_CACHE)
    qf = test_data['qf'].astype(np.float32)
    gf = test_data['gf'].astype(np.float32)
    num_query = int(test_data['num_query'])
    print(f"  loaded cache: qf {qf.shape}, gf {gf.shape}")

    gf = db_augment(gf, DBA_K)

    # Class-group setup
    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])

    print(f"  query class counts: {pd.Series(q_groups).value_counts().to_dict()}")
    print(f"  gallery class counts: {pd.Series(g_groups).value_counts().to_dict()}")

    # PRIMARY: per-class rerank with proven (k1=15, k2=4, lambda=0.275) for all classes.
    # Tests if class-restricted k-reciprocal context helps vs the global rerank.
    print("\n=== Variant A: per-class rerank, uniform proven params ===")
    params_uniform = {grp: (15, 4, 0.275) for grp in np.unique(q_groups)}
    rrd_a = per_class_rerank(qf, gf, q_groups, g_groups, params_uniform)
    write_csv(rrd_a, 'perclass_rerank_uniform_k15_lam0275', num_query)

    # BACKUP: per-class rerank with class-tuned params.
    # Trafficsignal (1836 gallery) — bigger pool, can afford larger k1
    # bin_like (654 gallery) — moderate
    # crosswalk (354 gallery) — smaller pool, smaller k1
    print("\n=== Variant B: per-class rerank, class-tuned params ===")
    params_tuned = {
        'trafficsignal': (20, 5, 0.30),
        'bin_like':      (12, 3, 0.25),
        'crosswalk':     (15, 4, 0.275),
    }
    rrd_b = per_class_rerank(qf, gf, q_groups, g_groups, params_tuned)
    write_csv(rrd_b, 'perclass_rerank_tuned', num_query)


if __name__ == '__main__':
    main()
