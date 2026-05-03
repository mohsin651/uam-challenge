"""Similarity-weighted DBA + temperature-softened DBA.

Standard DBA averages top-K neighbors uniformly. Idea: weight by similarity so
closer neighbors dominate. Two variants:
  A. Pure weighted: weights ∝ cosine similarity (linear)
  B. Temperature-softened: weights = softmax(sim/T), T=0.1

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
DBA_K = 8
RR_K1, RR_K2, RR_LAMBDA = 15, 4, 0.275


def weighted_dba(gf, k=8):
    """Each gallery: weighted-mean of top-k cosine neighbors. Weights ∝ similarity."""
    sim = gf @ gf.T                                        # (n, n), self-sim ≈ 1
    topk_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k] # (n, k), incl. self
    rows = np.arange(len(gf))[:, None]
    topk_sim = sim[rows, topk_idx]                         # (n, k)
    # Make sure weights are non-negative (cosine can be slightly negative)
    topk_sim = np.clip(topk_sim, a_min=1e-6, a_max=None)
    weights = topk_sim / topk_sim.sum(axis=1, keepdims=True)
    neighbors = gf[topk_idx]                               # (n, k, dim)
    gf_w = (weights[..., None] * neighbors).sum(axis=1)
    gf_w = gf_w / (np.linalg.norm(gf_w, axis=1, keepdims=True) + 1e-8)
    return gf_w.astype(np.float32)


def softened_dba(gf, k=8, temp=0.1):
    """Each gallery: softmax(sim/T)-weighted mean of top-k neighbors."""
    sim = gf @ gf.T
    topk_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.arange(len(gf))[:, None]
    topk_sim = sim[rows, topk_idx]
    logits = topk_sim / temp
    logits = logits - logits.max(axis=1, keepdims=True)    # for numerical stability
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=1, keepdims=True)
    neighbors = gf[topk_idx]
    gf_w = (weights[..., None] * neighbors).sum(axis=1)
    gf_w = gf_w / (np.linalg.norm(gf_w, axis=1, keepdims=True) + 1e-8)
    return gf_w.astype(np.float32)


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

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])

    # Diagnostic: weight distribution at k=8
    sim = gf @ gf.T
    topk_idx = np.argpartition(-sim, kth=8, axis=1)[:, :8]
    rows = np.arange(len(gf))[:, None]
    topk_sim_sample = sim[rows, topk_idx][0]
    print(f"  sample top-8 sims (gallery[0]): {topk_sim_sample.round(3).tolist()}")
    print(f"  weight ratio top-1 / top-8: {(topk_sim_sample[0] / topk_sim_sample[-1]).round(3)}")

    # Variant A: pure similarity-weighted
    print("\n=== Variant A: similarity-weighted DBA k=8 ===")
    gf_a = weighted_dba(gf, k=8)
    write_csv(qf, gf_a, 'weighted_dba_k8', num_query, q_groups, g_groups)

    # Variant B: temperature-softened (T=0.1, mostly top-1-2)
    print("\n=== Variant B: softened DBA k=8 T=0.1 ===")
    gf_b = softened_dba(gf, k=8, temp=0.1)
    write_csv(qf, gf_b, 'softened_dba_k8_t01', num_query, q_groups, g_groups)


if __name__ == '__main__':
    main()
