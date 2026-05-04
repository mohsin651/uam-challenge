"""Anisotropic DBA: class-density-aware k for each gallery image.

Standard DBA uses k=8 uniformly. Idea: sparse classes (crosswalk: 354 gallery,
bin_like: 654) may not have 8 same-identity neighbors, so k=8 averages across
identity boundaries within the class. Use k tuned to class density:
  trafficsignal (1836 gallery) → k=8 (proven, dense)
  bin_like (container∪rubbishbins, 654) → k=6 (medium density)
  crosswalk (354 gallery) → k=4 (sparse)

Two variants:
  PRIMARY: cross-class neighbors with class-aware k (each gallery's k determined by its class)
  BACKUP: within-class neighbors with class-aware k (stricter, smoothing only same-class)

Uses cached 8-ckpt features. ~30 sec runtime, training-free.
"""
import csv
import os
import numpy as np
import pandas as pd

from utils.re_ranking import re_ranking


CACHE_DIR = '/workspace/miuam_challenge_diff/results/cache'
TEST_CACHE = f'{CACHE_DIR}/8ckpt_test.npz'
URBAN_ROOT = '/workspace/Urban2026/'

RR_K1, RR_K2, RR_LAMBDA = 15, 4, 0.275

# Class-density-aware k values
K_PER_CLASS = {
    'trafficsignal': 8,   # dense (1836 gallery), proven k=8
    'bin_like': 6,        # medium (container 261 + rubbishbins 393 = 654)
    'crosswalk': 4,       # sparse (354)
}


def class_aware_dba_cross(gf, g_classes, k_per_class):
    """Each gallery feature replaced with mean of top-k nearest neighbors
    (across ALL classes), where k varies per gallery's class."""
    sim = gf @ gf.T  # (n, n)
    gf_smoothed = np.zeros_like(gf)
    for i in range(len(gf)):
        k = k_per_class[g_classes[i]]
        topk = np.argpartition(-sim[i], kth=k)[:k]
        avg = gf[topk].mean(axis=0)
        gf_smoothed[i] = avg / (np.linalg.norm(avg) + 1e-8)
    return gf_smoothed.astype(np.float32)


def class_aware_dba_within(gf, g_classes, k_per_class):
    """Each gallery feature replaced with mean of top-k nearest WITHIN-CLASS
    neighbors. Stricter — never averages across class boundaries."""
    gf_smoothed = np.zeros_like(gf)
    for grp, k in k_per_class.items():
        idx = np.where(g_classes == grp)[0]
        sub_gf = gf[idx]
        sim = sub_gf @ sub_gf.T
        # Effective k can't exceed within-class size
        eff_k = min(k, len(idx))
        topk = np.argpartition(-sim, kth=eff_k - 1, axis=1)[:, :eff_k]
        smoothed = sub_gf[topk].mean(axis=1)
        smoothed = smoothed / (np.linalg.norm(smoothed, axis=1, keepdims=True) + 1e-8)
        gf_smoothed[idx] = smoothed.astype(np.float32)
    return gf_smoothed.astype(np.float32)


def write_csv(qf, gf, label, num_query, q_groups, g_groups):
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)
    rrd[q_groups[:, None] != g_groups[None, :]] = np.inf
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
    print(f"  cache loaded: qf {qf.shape}, gf {gf.shape}")

    # Class groups
    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])

    print(f"  gallery distribution: {pd.Series(g_groups).value_counts().to_dict()}")
    print(f"  k per class: {K_PER_CLASS}")

    # PRIMARY: cross-class neighbors with class-aware k
    print("\n=== Variant A: cross-class neighbors, class-aware k ===")
    gf_a = class_aware_dba_cross(gf, g_groups, K_PER_CLASS)
    write_csv(qf, gf_a, 'aniso_dba_cross_8_6_4', num_query, q_groups, g_groups)

    # BACKUP: within-class neighbors with class-aware k
    print("\n=== Variant B: within-class neighbors, class-aware k ===")
    gf_b = class_aware_dba_within(gf, g_groups, K_PER_CLASS)
    write_csv(qf, gf_b, 'aniso_dba_within_8_6_4', num_query, q_groups, g_groups)


if __name__ == '__main__':
    main()
