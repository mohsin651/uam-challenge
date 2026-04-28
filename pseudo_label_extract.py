"""Extract high-confidence pseudo-labels from the 0.15884 8-ckpt ensemble.

Output: a CSV `pseudo_pairs.csv` with high-confidence (query, gallery, pseudo_id)
triples that can be merged into Urban2026 training.

Method:
1. Extract features for query+gallery using 7-baseline + cam-adv ep60 ensemble.
2. Compute reranked distance matrix (DBA k=8, k1=15, k2=4, λ=0.275).
3. Apply class-group filter (cross-class = inf).
4. Mutual nearest neighbor: for each query Q, take its top-1 gallery G; check
   G's top-1 query is Q. Only mutual pairs survive.
5. Confidence threshold: keep pairs whose rerank distance is below median of
   mutual-NN distances (top half of mutual NNs).
6. Optional: also include top-K=2 gallery for each surviving query.

The pseudo_id assigned to each surviving pair is `1200 + i` to avoid colliding
with original Urban2026 IDs (max ~1100).
"""
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.re_ranking import re_ranking


CAMADV_EP60 = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42   = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'
ALL_CKPTS = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
    CAMADV_EP60,
]
DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275
TOP_K_PSEUDO = 2          # iter 2: top-2 gallery per query (was 1)
APPLY_MEDIAN_FILTER = False  # iter 2: keep all mutual NNs (was True → kept top half)
PSEUDO_ID_OFFSET = 1200
OUT_PATH = '/workspace/miuam_challenge_diff/results/pseudo_pairs.csv'


def extract(model, loader, num_query):
    feats = []
    with torch.no_grad():
        for data in loader:
            ff = model(data['images'].cuda()).float()
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


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    print(f"  num_query: {num_query}, gallery: {len(val_loader.dataset.img_items) - num_query}")

    print("\n  Extracting 8-ckpt ensemble features:")
    qf_sum = gf_sum = None
    for ckpt in ALL_CKPTS:
        print(f"    {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        qf_sum = qf if qf_sum is None else qf_sum + qf
        gf_sum = gf if gf_sum is None else gf_sum + gf
        del model; torch.cuda.empty_cache()

    qf = F.normalize(qf_sum, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_sum, p=2, dim=1).numpy().astype(np.float32)
    gf_dba = db_augment(gf, DBA_K)

    print("\n  Computing rerank distance matrix...")
    q_g = np.dot(qf, gf_dba.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf_dba, gf_dba.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

    # Class-group filter (same as 0.15884 pipeline)
    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']
    q_groups = np.array([q2g[os.path.basename(it[0])] for it in q_items])
    g_groups = np.array([g2g[os.path.basename(it[0])] for it in g_items])
    rrd[q_groups[:, None] != g_groups[None, :]] = np.inf

    # Top-1 gallery for each query
    q_top1_g = np.argmin(rrd, axis=1)         # shape (num_query,)
    q_top1_dist = rrd[np.arange(num_query), q_top1_g]
    # Top-1 query for each gallery (mutual NN check)
    g_top1_q = np.argmin(rrd, axis=0)         # shape (num_gallery,)

    # Mutual-NN filter
    mutual = np.array([g_top1_q[q_top1_g[i]] == i for i in range(num_query)])
    mutual_idx = np.where(mutual)[0]
    print(f"\n  Mutual NN pairs: {mutual.sum()} / {num_query} ({100*mutual.mean():.1f}%)")

    # Confidence threshold (optional)
    if len(mutual_idx) == 0:
        print("  ! No mutual NN pairs found — pseudo-labeling not viable")
        return
    if APPLY_MEDIAN_FILTER:
        med_dist = np.median(q_top1_dist[mutual_idx])
        keep_mask = mutual & (q_top1_dist <= med_dist)
        keep_idx = np.where(keep_mask)[0]
        print(f"  After distance filter (≤ median {med_dist:.4f}): {len(keep_idx)} pairs")
    else:
        keep_idx = mutual_idx
        print(f"  Skipping median filter; keeping all {len(keep_idx)} mutual NN pairs")

    # Map back to image filenames
    q_img_names = [os.path.basename(it[0]) for it in q_items]
    g_img_names = [os.path.basename(it[0]) for it in g_items]
    g_camids   = [int(it[3].get('camid', it[2])) if isinstance(it[3], dict) else None for it in g_items]
    # Use camid from the dataset tuple (it[2] is camid in the dataset structure)
    g_camids = [it[2] for it in g_items]
    q_camids = [it[2] for it in q_items]

    # Write pseudo_pairs.csv
    rows = []
    for i, qi in enumerate(keep_idx):
        pseudo_id = PSEUDO_ID_OFFSET + i
        # Pair: query
        rows.append({
            'cameraID': f'c{q_camids[qi]:03d}',
            'imageName': q_img_names[qi],
            'objectID': pseudo_id,
            'source': 'query',
            'rerank_dist': float(q_top1_dist[qi]),
        })
        # Pair: top-K galleries
        if TOP_K_PSEUDO == 1:
            gi = q_top1_g[qi]
            rows.append({
                'cameraID': f'c{g_camids[gi]:03d}',
                'imageName': g_img_names[gi],
                'objectID': pseudo_id,
                'source': 'gallery',
                'rerank_dist': float(rrd[qi, gi]),
            })
        else:
            top_k = np.argpartition(rrd[qi], TOP_K_PSEUDO)[:TOP_K_PSEUDO]
            for gi in top_k:
                rows.append({
                    'cameraID': f'c{g_camids[gi]:03d}',
                    'imageName': g_img_names[gi],
                    'objectID': pseudo_id,
                    'source': 'gallery',
                    'rerank_dist': float(rrd[qi, gi]),
                })

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\n  → Wrote {OUT_PATH}: {len(rows)} rows ({len(keep_idx)} pseudo-identities)")
    # Print summary
    df = pd.DataFrame(rows)
    print(f"  Source breakdown: {df['source'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
