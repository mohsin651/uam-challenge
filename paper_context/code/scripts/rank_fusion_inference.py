"""Rank-fusion + weighted-feature inference combining 7-ckpt + ArcFace.

Two ensemble strategies:
  (1) Weighted feature ensemble (still averages features, just downweights ArcFace):
      qf = qf_orig + alpha * qf_arc; gf = gf_orig + alpha * gf_arc
      → normal post-proc (DBA, rerank, class-filter)
  (2) Reciprocal Rank Fusion (RRF) — independently process each source,
      combine RANKINGS (1/(k+rank)) per gallery item, sort by fused score.
      This avoids feature-distribution mismatch entirely.

Generates 6+ CSVs. User picks one to submit.
"""
import csv
import os
import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.re_ranking import re_ranking


ORIG_7 = [
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth',
]
ARCFACE_CKPTS = [
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_arcface_seed300/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_arcface_seed300/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_arcface_seed300/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_arcface_seed300/part_attention_vit_60.pth',
]
# Proven post-processing params (from 0.15421)
DBA_K = 8
RR_K1, RR_K2, RR_LAMBDA = 15, 4, 0.275


def extract(model, loader, num_query):
    feats = []
    with torch.no_grad():
        for d in loader:
            ff = model(d['images'].cuda()).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    feats = torch.cat(feats, 0)
    return feats[:num_query], feats[num_query:]


def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def post_process_distance_matrix(qf_t, gf_t, cross_group_mask):
    """Apply DBA + rerank + class-group filter; return (rerank_dist, n_gallery)."""
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    re_rank_dist = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)
    re_rank_dist[cross_group_mask] = np.inf
    return re_rank_dist


def rank_map_from_distance(dist):
    """For each query, return rank_map[q, g] = position of g in query q's sorted-by-distance list (smaller dist = better rank)."""
    sorted_idx = np.argsort(dist, axis=1)             # (n_q, n_g)
    rank_map = np.empty_like(sorted_idx)
    n_q, n_g = sorted_idx.shape
    for q in range(n_q):
        rank_map[q, sorted_idx[q]] = np.arange(n_g)
    return rank_map


def write_csv(indices, label):
    out = f'/workspace/miuam_challenge_diff/results/rf_{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, len(indices)+1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName','Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t+1))])
    print(f"  → {out}")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    # Extract original 7-ckpt features (sum, then per-ckpt-norm so already correct)
    print("=== extracting original 7-ckpt features ===")
    qf_o, gf_o = None, None
    for c in ORIG_7:
        m = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0); m.load_param(c); m = m.cuda().eval()
        qf, gf = extract(m, val_loader, num_query)
        qf_o = qf if qf_o is None else qf_o + qf
        gf_o = gf if gf_o is None else gf_o + gf
        del m; torch.cuda.empty_cache()

    print("=== extracting ArcFace features ===")
    qf_a, gf_a = None, None
    for c in ARCFACE_CKPTS:
        m = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0); m.load_param(c); m = m.cuda().eval()
        qf, gf = extract(m, val_loader, num_query)
        qf_a = qf if qf_a is None else qf_a + qf
        gf_a = gf if gf_a is None else gf_a + gf
        del m; torch.cuda.empty_cache()

    # Class-group filter setup
    import pandas as pd
    CLASS_GROUP = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    q_cls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    g_cls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q2g = {n: CLASS_GROUP[c.lower()] for n, c in zip(q_cls['imageName'], q_cls['Class'])}
    g2g = {n: CLASS_GROUP[c.lower()] for n, c in zip(g_cls['imageName'], g_cls['Class'])}
    q_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g']=='query']
    g_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g']=='gallery']
    q_groups = np.array([q2g[os.path.basename(it[0])] for it in q_items])
    g_groups = np.array([g2g[os.path.basename(it[0])] for it in g_items])
    cross_group_mask = q_groups[:, None] != g_groups[None, :]

    # === STRATEGY 1: Weighted feature ensemble ===
    print("\n=== weighted feature ensemble ===")
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        qf_combined = qf_o + alpha * qf_a
        gf_combined = gf_o + alpha * gf_a
        rrd = post_process_distance_matrix(qf_combined, gf_combined, cross_group_mask)
        idx = np.argsort(rrd, axis=1)[:, :100]
        write_csv(idx, f'wfeat_alpha_{alpha:.2f}'.replace('.','p'))

    # === STRATEGY 2: Reciprocal Rank Fusion ===
    print("\n=== reciprocal rank fusion ===")
    rrd_o = post_process_distance_matrix(qf_o, gf_o, cross_group_mask)
    rrd_a = post_process_distance_matrix(qf_a, gf_a, cross_group_mask)
    rank_o = rank_map_from_distance(rrd_o)
    rank_a = rank_map_from_distance(rrd_a)
    K = 60   # standard RRF constant
    for w_a in [0.1, 0.2, 0.3, 0.5, 1.0]:
        score = 1.0 / (K + rank_o) + w_a / (K + rank_a)        # higher = better
        # Apply class-group filter again (they agreed at distance level, but make sure)
        score[cross_group_mask] = -np.inf
        idx = np.argsort(-score, axis=1)[:, :100]
        write_csv(idx, f'rrf_warc_{w_a:.2f}'.replace('.','p'))


if __name__ == '__main__':
    main()
