"""Gentle query expansion (α=0.1, K=1, mutual-NN gated).

We tested heavy QE (α=0.7, K=3) and it failed — too strong, polluted by wrong
identities. Untested: very gentle pull only when the query+top-1-gallery are
mutually each other's top-1 (high confidence). Same mechanism as QMV-mutual
but bridges query→gallery instead of query→query.

Generates 2 variants:
  PRIMARY: α=0.1, mutual-NN gated only (very conservative, low risk)
  BACKUP: α=0.2, mutual-NN gated (slightly more aggressive)

Uses cached 8-ckpt features. <30 sec runtime.
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
RR_K1, RR_K2, RR_LAMBDA = 15, 4, 0.275


def db_augment(gf, k):
    if k <= 0: return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def gentle_qe(qf, gf, q_groups, g_groups, alpha=0.1):
    """For each query Q, find top-1 same-class gallery G.
    If they're mutually each other's top-1 (with class-group respected), do:
        Q' = (Q + alpha * G) / ||Q + alpha * G||
    Otherwise leave Q unchanged."""
    sim = qf @ gf.T  # (n_q, n_g)
    # Mask out cross-class
    cross_mask = q_groups[:, None] != g_groups[None, :]
    sim_masked = sim.copy()
    sim_masked[cross_mask] = -np.inf

    # For each query: top-1 same-class gallery
    q_top1 = np.argmax(sim_masked, axis=1)  # (n_q,)
    # For each gallery: top-1 same-class query (transpose perspective)
    sim_g_to_q = sim.T.copy()  # (n_g, n_q)
    sim_g_to_q[cross_mask.T] = -np.inf
    g_top1 = np.argmax(sim_g_to_q, axis=1)  # (n_g,)

    # Mutual-NN check
    mutual = np.array([g_top1[q_top1[i]] == i for i in range(len(qf))])
    print(f"  mutual NN pairs (gated for QE): {mutual.sum()}/{len(qf)} ({100*mutual.mean():.1f}%)")

    qf_expanded = qf.copy()
    for i in range(len(qf)):
        if mutual[i]:
            g_idx = q_top1[i]
            new_q = qf[i] + alpha * gf[g_idx]
            qf_expanded[i] = new_q / (np.linalg.norm(new_q) + 1e-8)
    return qf_expanded.astype(np.float32)


def write_csv(qf, gf_dba, label, num_query, q_groups, g_groups):
    q_g = np.dot(qf, gf_dba.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf_dba, gf_dba.T)
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

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])

    gf_dba = db_augment(gf, DBA_K)

    print("\n=== Variant A: gentle QE α=0.1, mutual-NN gated ===")
    qf_a = gentle_qe(qf, gf, q_groups, g_groups, alpha=0.1)
    write_csv(qf_a, gf_dba, 'gentle_qe_alpha01_mutual', num_query, q_groups, g_groups)

    print("\n=== Variant B: gentle QE α=0.2, mutual-NN gated ===")
    qf_b = gentle_qe(qf, gf, q_groups, g_groups, alpha=0.2)
    write_csv(qf_b, gf_dba, 'gentle_qe_alpha02_mutual', num_query, q_groups, g_groups)


if __name__ == '__main__':
    main()
