"""Uncertainty-gated cosine fallback.

For queries where rerank distance has tiny top-1/top-2 margin (< threshold),
the rerank's k-reciprocal step can flip an originally-correct cosine top-1.
Idea: re-order top-K of those queries by raw cosine sim (DBA gallery), keep
top-1 if cosine agrees with rerank; else swap.

3 variants:
  A: only flip top-1 (most conservative) on bottom-10% margin queries
  B: re-sort full top-100 by cosine on bottom-10% margin queries
  C: re-sort top-20 by cosine on bottom-10% margin queries (compromise)
"""
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

    # Class-group setup
    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qc = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gc = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qc['imageName'], qc['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gc['imageName'], gc['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])
    cross_mask = q_groups[:, None] != g_groups[None, :]

    # Rerank distance
    print("computing rerank distance...")
    q_g = qf @ gf_dba.T; q_q = qf @ qf.T; g_g = gf_dba @ gf_dba.T
    rrd = re_ranking(q_g, q_q, g_g, k1=15, k2=4, lambda_value=0.275)
    rrd_f = rrd.copy(); rrd_f[cross_mask] = np.inf

    # Cosine sim (DBA gallery)
    cos = qf @ gf_dba.T
    cos_f = cos.copy(); cos_f[cross_mask] = -np.inf

    # Top-100 from rerank (this is the 0.15884 ordering)
    base_idx = np.argsort(rrd_f, axis=1)[:, :100]

    # Compute margin (top-1 vs top-2 rerank distance)
    sorted_d = np.sort(rrd_f, axis=1)
    margin = sorted_d[:, 1] - sorted_d[:, 0]

    # Which queries are "uncertain"? Bottom 10% margin.
    n_uncertain = int(0.1 * num_query)
    uncertain_idx = np.argsort(margin)[:n_uncertain]
    print(f"uncertain queries (bottom 10% margin): {len(uncertain_idx)}")

    # Cosine-based top-100 (used as the alternative ordering)
    cos_idx = np.argsort(-cos_f, axis=1)[:, :100]

    # === Variant A: flip ONLY top-1 if cosine top-1 differs (within top-5 of rerank to be safe) ===
    flips = 0
    idx_A = base_idx.copy()
    for i in uncertain_idx:
        rerank_top1 = base_idx[i, 0]
        cosine_top1 = cos_idx[i, 0]
        if rerank_top1 != cosine_top1:
            # Only swap if cosine top-1 is in rerank top-5 (sanity)
            if cosine_top1 in base_idx[i, :5]:
                # Move cosine_top1 to position 0; shift the rest down
                new_order = [cosine_top1] + [x for x in base_idx[i] if x != cosine_top1]
                idx_A[i] = np.array(new_order[:100])
                flips += 1
    print(f"\n=== Variant A: flip top-1 (uncertain + cosine top-1 in rerank top-5) ===")
    print(f"  flips: {flips}")
    write_csv(idx_A, 'unc_fallback_A_flip_top1', num_query)

    # === Variant B: re-sort top-100 by cosine on uncertain queries ===
    idx_B = base_idx.copy()
    for i in uncertain_idx:
        # Get rerank top-100 set, but re-sort within by cosine
        topk_set = base_idx[i]
        cos_within = cos_f[i, topk_set]
        order = np.argsort(-cos_within)
        idx_B[i] = topk_set[order]
    print(f"\n=== Variant B: re-sort top-100 by cosine on uncertain ===")
    print(f"  re-sorted queries: {len(uncertain_idx)}")
    write_csv(idx_B, 'unc_fallback_B_resort_top100', num_query)

    # === Variant C: re-sort top-20 by cosine, keep 21-100 from rerank ===
    idx_C = base_idx.copy()
    for i in uncertain_idx:
        top20 = base_idx[i, :20]
        cos_within = cos_f[i, top20]
        order = np.argsort(-cos_within)
        new_top20 = top20[order]
        idx_C[i] = np.concatenate([new_top20, base_idx[i, 20:]])
    print(f"\n=== Variant C: re-sort top-20 by cosine on uncertain ===")
    print(f"  re-sorted queries: {len(uncertain_idx)}")
    write_csv(idx_C, 'unc_fallback_C_resort_top20', num_query)

    # Diagnostic: how many top-1s actually changed in each variant?
    n_changed_A = (idx_A[:, 0] != base_idx[:, 0]).sum()
    n_changed_B = (idx_B[:, 0] != base_idx[:, 0]).sum()
    n_changed_C = (idx_C[:, 0] != base_idx[:, 0]).sum()
    print(f"\n=== Top-1 change count (out of 928) ===")
    print(f"  Variant A: {n_changed_A}")
    print(f"  Variant B: {n_changed_B}")
    print(f"  Variant C: {n_changed_C}")


if __name__ == '__main__':
    main()
