"""Error analysis on the 0.15884 ensemble — find where the model is uncertain
or systematically biased.

Without GT we can use:
  1. Rerank-distance margin (top-1 minus top-2) — low margin = uncertain
  2. Per-class breakdown of uncertain queries
  3. Per-camera breakdown
  4. Top-100 self-consistency: how clustered/spread are the top-N gallery features?
  5. Compare 0.15884 ranking to the failed-attempt rankings — find queries that
     all 3 attempts AGREED on (likely correct) vs disagreed (the open battleground)
"""
import csv
import os
import numpy as np
import pandas as pd
from collections import Counter

from utils.re_ranking import re_ranking


URBAN_ROOT = '/workspace/Urban2026/'
PERCKPT_CACHE = '/workspace/miuam_challenge_diff/results/cache/perckpt_test.npz'

BASELINE_CSV = '/workspace/miuam_challenge_diff/backup_score/camadv_baseline7_plus_camadv_ep60_0.15884.csv'
ATTEMPTS = {
    '0.15884_LOCKED': BASELINE_CSV,
    '0.14772_camadv_merged_fusion': '/workspace/miuam_challenge_diff/results/pat8_plus_camadv_merged_heavyaug_s2500_ep60_submission.csv',
    '0.153_solo_diag': '/workspace/miuam_challenge_diff/results/camadv_merged_heavyaug_s2500_ep60_solo_submission.csv',
    '0.14331_drop_s1234_ep30': '/workspace/miuam_challenge_diff/results/prune_drop_s1234_ep30_submission.csv',
}


def parse_csv(p):
    rows = []
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            rows.append(list(map(int, row[1].split())))
    return np.array(rows)


def main():
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    qcam = pd.read_csv(os.path.join(URBAN_ROOT, 'query.csv'))
    n_q = len(qcls)
    print(f"# queries: {n_q}")

    # Class distribution
    print("\n=== Query class distribution ===")
    print(qcls['Class'].value_counts())

    # Camera distribution
    print("\n=== Query camera distribution ===")
    print(qcam['cameraID'].value_counts())

    # Load each attempt's top-100
    rankings = {k: parse_csv(p) for k, p in ATTEMPTS.items()}

    # === Cross-attempt agreement analysis ===
    # For each query, how many attempts agree on top-1?
    base = rankings['0.15884_LOCKED']
    print("\n=== Cross-attempt agreement (per query) ===")
    top1_agreement = np.zeros(n_q, dtype=int)
    for k, r in rankings.items():
        if k == '0.15884_LOCKED':
            continue
        top1_agreement += (r[:, 0] == base[:, 0]).astype(int)
    print(f"Top-1 of 0.15884 also chosen by:")
    print(f"  all 3 failed attempts: {(top1_agreement == 3).sum()} queries (high-confidence)")
    print(f"  2 of 3:                 {(top1_agreement == 2).sum()} queries")
    print(f"  1 of 3:                 {(top1_agreement == 1).sum()} queries")
    print(f"  0 of 3 (only baseline): {(top1_agreement == 0).sum()} queries (FRAGILE TOP-1)")

    fragile_idx = np.where(top1_agreement == 0)[0]
    print(f"\n=== FRAGILE queries (only baseline picks this top-1) ===")
    fragile_classes = qcls['Class'].iloc[fragile_idx].value_counts()
    print(f"Class distribution of {len(fragile_idx)} fragile queries:")
    print(fragile_classes)

    # === Top-100 overlap @ 100 ===
    print("\n=== Top-100 overlap (mean @ 100, all queries) ===")
    for k, r in rankings.items():
        if k == '0.15884_LOCKED':
            continue
        ov = np.mean([len(set(b) & set(rr)) for b, rr in zip(base, r)])
        print(f"  baseline vs {k}: {ov:.1f}/100")

    # === Rerank-distance margin from per-ckpt cache (uncertainty signal) ===
    print("\n=== Loading per-ckpt cache for distance margin analysis ===")
    if not os.path.exists(PERCKPT_CACHE):
        print("  no per-ckpt cache; skip")
        return
    d = np.load(PERCKPT_CACHE)
    qfs, gfs = d['qfs'], d['gfs']
    num_query = int(d['num_query'])
    qf = qfs.sum(axis=0); gf = gfs.sum(axis=0)
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)

    # DBA k=8
    sim_gg = gf @ gf.T
    topk = np.argpartition(-sim_gg, kth=8, axis=1)[:, :8]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)

    q_g = qf @ gf_dba.T; q_q = qf @ qf.T; g_g = gf_dba @ gf_dba.T
    print("  computing rerank distance...")
    rrd = re_ranking(q_g, q_q, g_g, k1=15, k2=4, lambda_value=0.275)

    # Apply class-group filter
    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qc = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gc = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qc['imageName'], qc['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gc['imageName'], gc['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])
    rrd_f = rrd.copy()
    rrd_f[q_groups[:, None] != g_groups[None, :]] = np.inf
    sorted_d = np.sort(rrd_f, axis=1)
    margin_1_2 = sorted_d[:, 1] - sorted_d[:, 0]   # top-1 vs top-2 gap
    margin_1_5 = sorted_d[:, 4] - sorted_d[:, 0]   # top-1 vs top-5 gap

    print(f"\n=== Top-1 confidence (margin between top-1 and top-2 distances) ===")
    print(f"  Mean: {margin_1_2.mean():.4f}, Median: {np.median(margin_1_2):.4f}")
    print(f"  P10:  {np.percentile(margin_1_2, 10):.4f} (most uncertain 10%)")
    print(f"  P90:  {np.percentile(margin_1_2, 90):.4f} (most confident 10%)")

    # Bottom 10% margin (uncertain queries) — class breakdown
    n_uncertain = int(0.1 * num_query)
    uncertain_idx = np.argsort(margin_1_2)[:n_uncertain]
    print(f"\n=== Most uncertain {n_uncertain} queries — class breakdown ===")
    print(qc['Class'].iloc[uncertain_idx].value_counts())
    print(f"\n=== Camera breakdown of uncertain queries ===")
    print(qcam['cameraID'].iloc[uncertain_idx].value_counts())

    # Are uncertain queries also "fragile" (disagreement with failed attempts)?
    overlap_uncertain_fragile = len(set(uncertain_idx) & set(fragile_idx))
    print(f"\n=== Uncertain ∩ Fragile ===")
    print(f"  {overlap_uncertain_fragile} queries are BOTH uncertain (low margin) AND fragile (only baseline picks top-1)")
    print(f"  These are the highest-leverage queries to fix")


if __name__ == '__main__':
    main()
