"""Feature-centering inference: strip per-camera mean bias before re-ranking.

The c004 query features and c001-c003 gallery features have different "mean
positions" in feature space due to camera-specific systematic bias. Subtracting
per-camera means BEFORE distance computation removes this bias, leaving only
the identity-discriminating direction of each feature.

Generates 3 variants for diagnostic comparison:
  A. PRIMARY: per-camera mean centering (q - mean_q, g - mean_g)
  B. Global mean centering (q - mean_all, g - mean_all)
  C. PCA debiasing — drop top-1 principal component (often the camera factor)

Pipeline:
  1. Extract 8-ckpt ensemble features (proven 0.15884 set)
  2. Apply centering transformation
  3. L2-normalize, then DBA + rerank + class-group filter

Per "1-2 high-EV variants" rule, the PRIMARY (variant A) is the recommended
submission. Variants B and C are diagnostics.
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


CAMADV_S500_EP60 = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'

ALL_CKPTS = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
    CAMADV_S500_EP60,
]
DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275


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


def write_csv(qf_arr, gf_arr, label, num_query, val_loader):
    """qf_arr, gf_arr already L2-normalized numpy float32."""
    qf = qf_arr; gf = db_augment(gf_arr, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

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

    indices = np.argsort(rrd, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, len(indices) + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"  → {out}")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    print(f"  num_query: {num_query}")

    print("\n  Extracting 8-ckpt ensemble:")
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

    qf_raw = F.normalize(qf_sum, p=2, dim=1).numpy().astype(np.float32)  # (928, 1024)
    gf_raw = F.normalize(gf_sum, p=2, dim=1).numpy().astype(np.float32)  # (2844, 1024)
    print(f"\n  qf_raw shape: {qf_raw.shape}, gf_raw shape: {gf_raw.shape}")

    # Compute means
    mean_q = qf_raw.mean(axis=0, keepdims=True)         # (1, 1024) — c004 mean
    mean_g = gf_raw.mean(axis=0, keepdims=True)         # (1, 1024) — c001-003 mean
    all_feats = np.vstack([qf_raw, gf_raw])
    mean_all = all_feats.mean(axis=0, keepdims=True)    # (1, 1024)
    print(f"  ||mean_q - mean_g||: {np.linalg.norm(mean_q - mean_g):.4f}")
    print(f"  cos(mean_q, mean_g): {(mean_q @ mean_g.T / (np.linalg.norm(mean_q) * np.linalg.norm(mean_g))).item():.4f}")

    # Variant A: per-camera mean centering
    qf_a = qf_raw - mean_q
    gf_a = gf_raw - mean_g
    qf_a = qf_a / (np.linalg.norm(qf_a, axis=1, keepdims=True) + 1e-8)
    gf_a = gf_a / (np.linalg.norm(gf_a, axis=1, keepdims=True) + 1e-8)
    write_csv(qf_a.astype(np.float32), gf_a.astype(np.float32),
              'centering_per_camera', num_query, val_loader)

    # Variant B: global mean centering
    qf_b = qf_raw - mean_all
    gf_b = gf_raw - mean_all
    qf_b = qf_b / (np.linalg.norm(qf_b, axis=1, keepdims=True) + 1e-8)
    gf_b = gf_b / (np.linalg.norm(gf_b, axis=1, keepdims=True) + 1e-8)
    write_csv(qf_b.astype(np.float32), gf_b.astype(np.float32),
              'centering_global', num_query, val_loader)

    # Variant C: PCA — remove top-1 principal component (often the camera factor)
    # Center on global mean, then SVD, then drop top-1 component
    centered = all_feats - mean_all
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    print(f"  PCA singular values (top 10): {S[:10]}")
    # Project out top-1 PC
    pc1 = Vt[0:1]                                     # (1, 1024)
    qf_c = qf_raw - mean_all
    gf_c = gf_raw - mean_all
    qf_c = qf_c - (qf_c @ pc1.T) @ pc1                # remove projection on pc1
    gf_c = gf_c - (gf_c @ pc1.T) @ pc1
    qf_c = qf_c / (np.linalg.norm(qf_c, axis=1, keepdims=True) + 1e-8)
    gf_c = gf_c / (np.linalg.norm(gf_c, axis=1, keepdims=True) + 1e-8)
    write_csv(qf_c.astype(np.float32), gf_c.astype(np.float32),
              'centering_pca_drop1', num_query, val_loader)


if __name__ == '__main__':
    main()
