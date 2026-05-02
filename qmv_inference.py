"""Query Majority Voting (QMV) — inspired by last year's winning paper.

For each query, find its top-K same-class query peers (other queries that
look very similar). Average their features into a consensus query feature.
Use that for retrieval.

Different from standard query expansion (which uses gallery neighbors —
biased toward gallery noise). QMV uses query neighbors — mutual verification
sidesteps gallery noise.

Generates 2 variants per the 1-2 high-EV rule:
  PRIMARY: mutual-NN top-1 peer averaging (most conservative, lowest risk)
  BACKUP: top-3 peers averaging (more aggressive consensus)

Pipeline runs on CACHED 8-ckpt features (no GPU needed; ~30 sec total).
"""
import csv
import os
import numpy as np
import pandas as pd

from utils.re_ranking import re_ranking


CACHE_DIR = '/workspace/miuam_challenge_diff/results/cache'
TEST_CACHE = f'{CACHE_DIR}/8ckpt_test.npz'

DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275
URBAN_ROOT = '/workspace/Urban2026/'


def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def write_csv(qf, gf, label, num_query):
    """qf, gf already L2-normalized numpy float32."""
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    # Query+gallery sequential filename ordering — verified across prior scripts
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])
    rrd[q_groups[:, None] != g_groups[None, :]] = np.inf

    indices = np.argsort(rrd, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, len(indices) + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"  → {out}")


def qmv_top1_mutual(qf, q_classes):
    """Average each query's feature with its mutual NN top-1 same-class peer."""
    num_query = qf.shape[0]
    same_class = q_classes[:, None] == q_classes[None, :]
    qq_sim = qf @ qf.T
    np.fill_diagonal(qq_sim, -np.inf)
    qq_sim[~same_class] = -np.inf

    top1_peer = np.argmax(qq_sim, axis=1)
    top1_sim = qq_sim[np.arange(num_query), top1_peer]
    mutual = top1_peer[top1_peer] == np.arange(num_query)

    n_mutual = int(mutual.sum())
    if n_mutual > 0:
        sims_of_mutual = top1_sim[mutual]
        print(f"  mutual NN pairs: {n_mutual}/{num_query} ({100*mutual.mean():.1f}%)"
              f" — sim min/median/max: {sims_of_mutual.min():.3f}/{np.median(sims_of_mutual):.3f}/{sims_of_mutual.max():.3f}")

    qf_enhanced = qf.copy()
    for i in range(num_query):
        if mutual[i]:
            j = top1_peer[i]
            avg = (qf[i] + qf[j]) / 2.0
            qf_enhanced[i] = avg / (np.linalg.norm(avg) + 1e-8)
    return qf_enhanced.astype(np.float32)


def qmv_topk(qf, q_classes, K=3, sim_threshold=0.5):
    """Average each query's feature with its top-K same-class peers (filter by sim threshold)."""
    num_query = qf.shape[0]
    same_class = q_classes[:, None] == q_classes[None, :]
    qq_sim = qf @ qf.T
    np.fill_diagonal(qq_sim, -np.inf)
    qq_sim[~same_class] = -np.inf

    qf_enhanced = qf.copy()
    n_avg_peers = 0
    for i in range(num_query):
        topk_idx = np.argpartition(-qq_sim[i], K)[:K]
        topk_sim = qq_sim[i, topk_idx]
        # Keep peers above threshold
        valid = topk_idx[topk_sim > sim_threshold]
        if len(valid) > 0:
            n_avg_peers += len(valid)
            avg = (qf[i] + qf[valid].sum(axis=0)) / (1 + len(valid))
            qf_enhanced[i] = avg / (np.linalg.norm(avg) + 1e-8)
    print(f"  avg peers per query: {n_avg_peers/num_query:.2f}")
    return qf_enhanced.astype(np.float32)


def main():
    test_data = np.load(TEST_CACHE)
    qf = test_data['qf'].astype(np.float32)         # (928, 1024)
    gf = test_data['gf'].astype(np.float32)         # (2844, 1024)
    num_query = int(test_data['num_query'])
    print(f"  loaded cache: qf {qf.shape}, gf {gf.shape}, num_query {num_query}")

    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    q_classes_list = [qcls.loc[qcls['imageName'] == f'{i:06d}.jpg', 'Class'].iloc[0].lower()
                      for i in range(1, num_query + 1)]
    q_classes = np.array(q_classes_list)
    print(f"  class distribution: {pd.Series(q_classes).value_counts().to_dict()}")

    # PRIMARY: mutual-NN top-1 averaging
    print("\n=== Variant A: mutual-NN top-1 averaging ===")
    qf_a = qmv_top1_mutual(qf, q_classes)
    write_csv(qf_a, gf, 'qmv_mutual_top1', num_query)

    # BACKUP: top-3 with sim threshold 0.5
    print("\n=== Variant B: top-3 peers, sim threshold 0.5 ===")
    qf_b = qmv_topk(qf, q_classes, K=3, sim_threshold=0.5)
    write_csv(qf_b, gf, 'qmv_top3_sim05', num_query)


if __name__ == '__main__':
    main()
