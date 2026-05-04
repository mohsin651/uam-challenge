"""DBA + rerank-parameter sweep on the proven 7-ckpt ensemble.

Extracts features ONCE from the 7-ckpt list, then loops over post-processing
variants:
  - DBA on/off (k = 0, 3, 5, 8, 10)
  - Re-rank (k1, k2, lambda) — paper defaults vs a few alternatives

For each variant, applies class-group filter + writes a Kaggle CSV. User picks
the most-likely-winner CSVs to submit. Cheap probe of the post-processing
configuration space we've never explored.

DBA = Database-side Augmentation:
  For each gallery feature, replace with the mean of its top-K nearest
  neighbors (including itself). Smooths out per-image noise. Standard ReID
  trick, ~+0.3–1% mAP. Tunable K.
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


CHECKPOINT_LIST = [
    # Proven 7-ckpt set that produced 0.15421 with the optimal post-proc.
    # seed=200 (with GRAD_CLIP=1.0) was tested and dragged the ensemble down
    # from 0.15421 → 0.12608, same pattern as seed=100+EMA. The grad-clip
    # changes feature distribution enough to break the ensemble blend.
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth',
]

# Round 6 (3-seed ensemble: 11 ckpts). Test the proven 0.15421 winner config
# on the bigger ensemble + a few nearby configs in case optimum shifted.
VARIANTS = [
    # Proven post-processing winner from the 7-ckpt era — test if it transfers
    ('11ckpt_dba8_k15_k2_4_lambda027',   8,  15, 4, 0.275),    # proven winner
    ('11ckpt_dba8_k15_k2_4_lambda030',   8,  15, 4, 0.300),    # adjacent (was 0.15410)
    ('11ckpt_dba8_k15_k2_4_lambda025',   8,  15, 4, 0.250),    # adjacent (was 0.15288)
    # In case the bigger ensemble shifts the DBA optimum
    ('11ckpt_dba7_k15_k2_4_lambda027',   7,  15, 4, 0.275),
    ('11ckpt_dba9_k15_k2_4_lambda027',   9,  15, 4, 0.275),
]


def extract_feature(model, dataloader, num_query):
    feats = []
    with torch.no_grad():
        for data in dataloader:
            ff = model(data['images'].cuda()).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    feats = torch.cat(feats, 0)
    return feats[:num_query], feats[num_query:]


def db_augment(gf, k):
    """gf: (n, D), L2-normalized numpy. Returns DBA-smoothed gallery, L2-normed."""
    if k <= 0:
        return gf
    sim = gf @ gf.T                                    # (n, n)
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k] # (n, k) — top-k incl self
    gf_dba = gf[topk].mean(axis=1)                     # (n, D)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()

    # 1. Extract features once from all 7 ckpts.
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    qf_sum = None
    gf_sum = None
    for ckpt in CHECKPOINT_LIST:
        print(f"  loading {os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract_feature(model, val_loader, num_query)
        qf_sum = qf if qf_sum is None else qf_sum + qf
        gf_sum = gf if gf_sum is None else gf_sum + gf
        del model
        torch.cuda.empty_cache()

    qf = F.normalize(qf_sum, p=2, dim=1).numpy().astype(np.float32)
    gf_base = F.normalize(gf_sum, p=2, dim=1).numpy().astype(np.float32)
    print(f"\nensembled features: qf {qf.shape}  gf {gf_base.shape}")

    # 2. Class-group filter setup
    import pandas as pd
    CLASS_GROUP = {'trafficsignal': 'trafficsignal', 'crosswalk': 'crosswalk',
                   'container': 'bin_like', 'rubbishbins': 'bin_like'}
    q_cls_df = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    g_cls_df = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q_name_to_grp = {n: CLASS_GROUP[c.lower()] for n, c in zip(q_cls_df['imageName'], q_cls_df['Class'])}
    g_name_to_grp = {n: CLASS_GROUP[c.lower()] for n, c in zip(g_cls_df['imageName'], g_cls_df['Class'])}
    q_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']
    q_groups = np.array([q_name_to_grp[os.path.basename(it[0])] for it in q_items])
    g_groups = np.array([g_name_to_grp[os.path.basename(it[0])] for it in g_items])
    cross_group_mask = q_groups[:, None] != g_groups[None, :]

    # 3. For each variant: apply DBA (if k>0), compute distance matrices, rerank, classfilt, write CSV
    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)
    for label, dba_k, k1, k2, lam in VARIANTS:
        print(f"\n=== variant: {label}  (dba_k={dba_k}, k1={k1}, k2={k2}, lambda={lam}) ===")
        gf = db_augment(gf_base, dba_k)
        q_g = np.dot(qf, gf.T)
        q_q = np.dot(qf, qf.T)
        g_g = np.dot(gf, gf.T)
        re_rank_dist = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)
        re_rank_dist[cross_group_mask] = np.inf
        indices = np.argsort(re_rank_dist, axis=1)[:, :100]

        out_path = f'/workspace/miuam_challenge_diff/results/sweep_{label}_submission.csv'
        names = [f'{i:06d}.jpg' for i in range(1, len(indices) + 1)]
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['imageName', 'Corresponding Indexes'])
            for n, t in zip(names, indices):
                w.writerow([n, ' '.join(map(str, t + 1))])
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
