"""Variant D: even tighter — only flip top-1 on the ~5% most uncertain queries
where cosine top-1 is in rerank top-3 (was top-5). Smaller blast radius."""
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


def write_csv(indices, label, num_query):
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
    rrd_f = rrd.copy(); rrd_f[cross_mask] = np.inf

    cos = qf @ gf_dba.T
    cos_f = cos.copy(); cos_f[cross_mask] = -np.inf

    base_idx = np.argsort(rrd_f, axis=1)[:, :100]
    sorted_d = np.sort(rrd_f, axis=1)
    margin = sorted_d[:, 1] - sorted_d[:, 0]

    # Bottom 5% margin
    n_uncertain = int(0.05 * num_query)
    uncertain_idx = np.argsort(margin)[:n_uncertain]
    print(f"uncertain queries (bottom 5% margin): {len(uncertain_idx)}")
    cos_idx = np.argsort(-cos_f, axis=1)[:, :100]

    # === Variant D: bottom 5%, cosine top-1 in rerank top-3 ===
    flips = 0
    idx_D = base_idx.copy()
    for i in uncertain_idx:
        rerank_top1 = base_idx[i, 0]
        cosine_top1 = cos_idx[i, 0]
        if rerank_top1 != cosine_top1 and cosine_top1 in base_idx[i, :3]:
            new_order = [cosine_top1] + [x for x in base_idx[i] if x != cosine_top1]
            idx_D[i] = np.array(new_order[:100])
            flips += 1
    print(f"\n=== Variant D: flip top-1 (bottom 5% margin, cosine top-1 in rerank top-3) ===")
    print(f"  flips: {flips}")
    write_csv(idx_D, 'unc_fallback_D_tight', num_query)

    # === Variant E: even tighter — bottom 3%, cosine top-1 must be rerank top-2 ===
    n_uncertain_e = int(0.03 * num_query)
    uncertain_e = np.argsort(margin)[:n_uncertain_e]
    flips_e = 0
    idx_E = base_idx.copy()
    for i in uncertain_e:
        rerank_top1 = base_idx[i, 0]
        cosine_top1 = cos_idx[i, 0]
        if rerank_top1 != cosine_top1 and cosine_top1 == base_idx[i, 1]:
            new_order = [cosine_top1, rerank_top1] + list(base_idx[i, 2:])
            idx_E[i] = np.array(new_order[:100])
            flips_e += 1
    print(f"\n=== Variant E: bottom 3% margin, swap top-1↔top-2 only if cosine agrees ===")
    print(f"  flips: {flips_e}")
    write_csv(idx_E, 'unc_fallback_E_swap12', num_query)


if __name__ == '__main__':
    main()
