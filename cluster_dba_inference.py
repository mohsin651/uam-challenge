"""Cluster-aware DBA: KMeans-cluster the gallery first, then average each
gallery feature with its CLUSTER MEAN (instead of top-K nearest neighbors).

Standard DBA averages each gallery's feature with its top-K cosine neighbors,
which can pull features across identity boundaries. Cluster-aware DBA only
averages within explicit clusters (hypothesized identities), preserving
identity-discriminating signal while still smoothing.

Generates 2 variants:
  PRIMARY: hard cluster replacement (each gallery → cluster centroid)
  BACKUP:  soft cluster smoothing (each gallery → 0.5*own + 0.5*centroid)

Tries 3 cluster counts (500, 700, 900) and picks the one that gives the most
balanced cluster-size distribution.
"""
import csv
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils.re_ranking import re_ranking


CACHE_DIR = '/workspace/miuam_challenge_diff/results/cache'
TEST_CACHE = f'{CACHE_DIR}/8ckpt_test.npz'
URBAN_ROOT = '/workspace/Urban2026/'
DBA_K = 8
RR_K1, RR_K2, RR_LAMBDA = 15, 4, 0.275


def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def cluster_replace(gf, n_clusters, alpha=1.0):
    """Replace each gallery feature with (1-alpha)*own + alpha*cluster_mean."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    labels = km.fit_predict(gf)
    centroids = km.cluster_centers_  # (n_clusters, D)
    # Normalize centroids to unit norm
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    cluster_size = np.bincount(labels)
    print(f"    n_clusters={n_clusters}: median size {np.median(cluster_size):.1f}, "
          f"max {cluster_size.max()}, min {cluster_size.min()}, singletons {(cluster_size==1).sum()}")

    gf_smoothed = (1 - alpha) * gf + alpha * centroids[labels]
    gf_smoothed = gf_smoothed / (np.linalg.norm(gf_smoothed, axis=1, keepdims=True) + 1e-8)
    return gf_smoothed.astype(np.float32)


def write_csv(qf, gf, label, num_query):
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])
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
    print(f"  loaded cache: qf {qf.shape}, gf {gf.shape}")

    # Apply standard DBA first (proven 0.15884 starting point)
    gf_dba = db_augment(gf, DBA_K)

    # Try several cluster counts to gauge cluster-size distribution
    print("\n  KMeans clustering on DBA-smoothed gallery:")
    for n_c in [500, 700, 900]:
        cluster_replace(gf_dba, n_c, alpha=0.0)  # just diagnostic, alpha=0 = no replacement

    # PRIMARY: soft cluster smoothing at k=700, alpha=0.5
    print("\n=== Variant A: soft cluster smoothing (n=700, alpha=0.5) ===")
    gf_a = cluster_replace(gf_dba, n_clusters=700, alpha=0.5)
    write_csv(qf, gf_a, 'cluster_dba_n700_alpha05', num_query)

    # BACKUP: hard cluster replacement (k=700, alpha=1.0)
    print("\n=== Variant B: hard cluster replacement (n=700, alpha=1.0) ===")
    gf_b = cluster_replace(gf_dba, n_clusters=700, alpha=1.0)
    write_csv(qf, gf_b, 'cluster_dba_n700_alpha10', num_query)


if __name__ == '__main__':
    main()
