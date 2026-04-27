"""ArcFace + 7-ckpt ensemble inference — 3 variants in one pass.

Extracts L2-normalized final-layer CLS features ONCE per unique checkpoint,
then combines them into 3 ensemble variants and writes 3 submission CSVs.
Applies the proven post-processing recipe (DBA k=8, k1=15, k2=4, λ=0.275)
+ class-group filter to each.

Variants:
  A. ArcFace solo (4 ckpts: ep30/40/50/60)
  B. 7-ckpt + ArcFace 4 ckpts (full cross-recipe ensemble; 11 ckpts total)
  C. 7-ckpt + ArcFace ep60 only (minimal addition; 8 ckpts total)
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


# Original 7-ckpt set that gave 0.15421 with optimal post-proc
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


def write_submission(qf, gf, label, val_loader, cfg):
    """Apply DBA + rerank + class-group filter, write Kaggle CSV."""
    qf = F.normalize(qf, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    re_rank_dist = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

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
    re_rank_dist[q_groups[:, None] != g_groups[None, :]] = np.inf

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/arcface_{label}_submission.csv'
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

    # Extract original 7-ckpt features (sum)
    print("=== extracting original 7-ckpt features ===")
    qf_orig, gf_orig = None, None
    for c in ORIG_7:
        m = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0); m.load_param(c); m = m.cuda().eval()
        qf, gf = extract(m, val_loader, num_query)
        qf_orig = qf if qf_orig is None else qf_orig + qf
        gf_orig = gf if gf_orig is None else gf_orig + gf
        del m; torch.cuda.empty_cache()
    print(f"  orig 7-ckpt sum shape: qf {qf_orig.shape}")

    # Extract ArcFace ckpt features individually
    print("\n=== extracting ArcFace 4 ckpts ===")
    qf_arc, gf_arc = [], []
    for c in ARCFACE_CKPTS:
        m = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0); m.load_param(c); m = m.cuda().eval()
        qf, gf = extract(m, val_loader, num_query)
        qf_arc.append(qf); gf_arc.append(gf)
        del m; torch.cuda.empty_cache()
    qf_arc_sum = sum(qf_arc); gf_arc_sum = sum(gf_arc)
    print(f"  arcface 4-ckpt sum shape: qf {qf_arc_sum.shape}")

    # Variant A: ArcFace solo
    print("\n=== variant A: ArcFace solo (4 ckpts) ===")
    write_submission(qf_arc_sum, gf_arc_sum, 'A_solo_4ckpt', val_loader, cfg)

    # Variant B: 7-ckpt + ArcFace 4 (11 total)
    print("\n=== variant B: 7-ckpt + ArcFace 4 (11 ckpts) ===")
    write_submission(qf_orig + qf_arc_sum, gf_orig + gf_arc_sum, 'B_orig7_plus_arcface4', val_loader, cfg)

    # Variant C: 7-ckpt + ArcFace ep60 only (8 total)
    print("\n=== variant C: 7-ckpt + ArcFace ep60 only (8 ckpts) ===")
    write_submission(qf_orig + qf_arc[3], gf_orig + gf_arc[3], 'C_orig7_plus_arcface_ep60', val_loader, cfg)


if __name__ == '__main__':
    main()
