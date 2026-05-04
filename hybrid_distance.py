"""Hybrid distance: blend rerank distance with cosine sim.

  d_hybrid = (1-α) * rerank_d + α * (1 - cosine)

α=0.05, 0.1, 0.15. α=0 = pure rerank (0.15884 baseline). Tighter α = smaller
deviation from proven recipe."""
import csv
import os
import numpy as np
import pandas as pd

from utils.re_ranking import re_ranking


URBAN_ROOT = '/workspace/Urban2026/'
PERCKPT_CACHE = '/workspace/miuam_challenge_diff/results/cache/perckpt_test.npz'


def db_augment(gf, k):
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def write_csv(rrd_filtered, label, num_query):
    indices = np.argsort(rrd_filtered, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, num_query + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"  → {out}")


def main():
    d = np.load(PERCKPT_CACHE)
    qfs, gfs = d['qfs'], d['gfs']
    num_query = int(d['num_query'])
    qf = qfs.sum(axis=0); gf = gfs.sum(axis=0)
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    gf_dba = db_augment(gf, k=8)

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qc = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gc = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qc['imageName'], qc['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gc['imageName'], gc['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])
    cross_mask = q_groups[:, None] != g_groups[None, :]

    print("computing rerank distance...")
    q_g = qf @ gf_dba.T; q_q = qf @ qf.T; g_g = gf_dba @ gf_dba.T
    rrd = re_ranking(q_g, q_q, g_g, k1=15, k2=4, lambda_value=0.275)

    # cosine distance = 1 - cosine_sim
    cos = qf @ gf_dba.T
    cos_d = 1.0 - cos

    # Normalize both to similar scale (rerank is already in [0,2] roughly, cos_d in [0,2])
    print(f"rerank range: [{rrd.min():.4f}, {rrd.max():.4f}], mean {rrd.mean():.4f}")
    print(f"cos_d  range: [{cos_d.min():.4f}, {cos_d.max():.4f}], mean {cos_d.mean():.4f}")

    for alpha in [0.05, 0.10, 0.15]:
        hyb = (1 - alpha) * rrd + alpha * cos_d
        hyb_f = hyb.copy(); hyb_f[cross_mask] = np.inf

        base = rrd.copy(); base[cross_mask] = np.inf
        base_top1 = np.argmin(base, axis=1)
        hyb_top1 = np.argmin(hyb_f, axis=1)
        n_changed = (base_top1 != hyb_top1).sum()
        print(f"\n=== α={alpha:.2f}: top-1 changes vs baseline = {n_changed}/{num_query} ===")
        write_csv(hyb_f, f'hybrid_dist_alpha{int(alpha*100):03d}', num_query)


if __name__ == '__main__':
    main()
