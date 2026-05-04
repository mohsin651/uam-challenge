"""Hyperparameter-ensemble re-ranking: average distance matrices from multiple
(k1, k2, lambda) configs around the proven (15, 4, 0.275) peak.

Hedges against the parameter optimum being slightly off (the §44 sweep showed
a fairly flat plateau near our chosen config). Averaging across the plateau
should capture a more robust optimum.

Genuinely novel for our pipeline: nobody has ensemble-averaged rerank
distance matrices before. Different from CSV-rank averaging — operates at the
distance level so DBA + class-group filter still work normally.

Generates ONE CSV. ~30 sec runtime, training-free, cached features.
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

# Configs to ensemble (centered on proven (15, 4, 0.275))
RERANK_CONFIGS = [
    (15, 4, 0.275),     # proven center
    (14, 4, 0.275),     # k1 - 1
    (16, 4, 0.275),     # k1 + 1
    (15, 3, 0.275),     # k2 - 1
    (15, 5, 0.275),     # k2 + 1
    (15, 4, 0.250),     # lambda - 0.025
    (15, 4, 0.300),     # lambda + 0.025
]


def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def main():
    test_data = np.load(TEST_CACHE)
    qf = test_data['qf'].astype(np.float32)
    gf = test_data['gf'].astype(np.float32)
    num_query = int(test_data['num_query'])
    print(f"  loaded cache: qf {qf.shape}, gf {gf.shape}")

    gf_dba = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf_dba.T)
    q_q = np.dot(qf, qf.T)
    g_g = np.dot(gf_dba, gf_dba.T)

    # Compute rerank distance matrix for each config, then average
    print(f"\n  Computing rerank distance for {len(RERANK_CONFIGS)} configs:")
    rrd_avg = None
    for k1, k2, lam in RERANK_CONFIGS:
        print(f"    (k1={k1}, k2={k2}, lambda={lam})")
        rrd_i = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)
        rrd_avg = rrd_i if rrd_avg is None else rrd_avg + rrd_i
    rrd_avg = rrd_avg / len(RERANK_CONFIGS)
    print(f"  averaged: shape {rrd_avg.shape}, min={rrd_avg.min():.4f}, max={rrd_avg.max():.4f}")

    # Apply class-group filter
    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])
    rrd_avg[q_groups[:, None] != g_groups[None, :]] = np.inf

    indices = np.argsort(rrd_avg, axis=1)[:, :100]
    out = '/workspace/miuam_challenge_diff/results/hparam_ensemble_rerank_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, num_query + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"\n  → {out}")


if __name__ == '__main__':
    main()
