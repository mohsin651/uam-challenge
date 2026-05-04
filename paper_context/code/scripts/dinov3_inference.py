"""DINOv3 ViT-L/16 + PAT-8 distance fusion.

DINOv3 (Aug 2025) — 1.7B-image self-supervised pretraining, first foundation
model to beat CLIP/SigLIP/Perception Encoder on retrieval. Used as FROZEN
feature extractor (no training) — pure semantic features complementary to
PAT's identity-discriminative features.

Same patch16, same 1024-d output as PAT — could do feature-level sum, but we
use distance-level fusion (safer per prior backbone-swap failures).

Generates 3 variants:
  PRIMARY: PAT-8 + DINOv3 distance fusion 85/15 (conservative, PAT-dominant)
  BACKUP A: PAT-8 + DINOv3 70/30 (more DINOv3 influence)
  BACKUP B: DINOv3 SOLO (diagnostic — does DINOv3 features have any signal?)
"""
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.re_ranking import re_ranking


CAMADV_S500_EP60 = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'
PAT_8 = [
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
URBAN_ROOT = '/workspace/Urban2026/'


def extract_pat(model, loader, num_query):
    feats = []
    with torch.no_grad():
        for data in loader:
            ff = model(data['images'].cuda()).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    feats = torch.cat(feats, 0)
    return feats[:num_query], feats[num_query:]


def extract_dinov3(loader, num_query):
    """Extract features from DINOv3 ViT-L/16 (frozen, ImageNet/DINOv3 stats)."""
    print("  loading DINOv3 ViT-L/16...")
    model = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0)
    state = torch.load('/workspace/miuam_challenge_diff/pretrained/dinov3_vitl16.pth')
    model.load_state_dict(state)
    model = model.cuda().eval()

    # DINOv3 typical preprocessing: ImageNet stats. Loader applies (0.5,0.5,0.5)
    # mean/std (PAT recipe). Adjust by re-normalizing on-the-fly.
    PAT_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    PAT_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    DINOV3_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    DINOV3_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    feats = []
    with torch.no_grad():
        for data in loader:
            img = data['images'].cuda()
            # Undo PAT normalization → re-apply DINOv3 normalization
            img_raw = img * PAT_STD + PAT_MEAN              # back to [0,1]
            img_dinov3 = (img_raw - DINOV3_MEAN) / DINOV3_STD
            ff = model(img_dinov3).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    feats = torch.cat(feats, 0)
    del model; torch.cuda.empty_cache()
    return feats[:num_query], feats[num_query:]


def db_augment(gf, k):
    if k <= 0: return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def compute_rerank_dist(qf, gf):
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    return re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)


def write_csv_from_dist(rrd, label, num_query):
    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, rrd.shape[1] + 1)])
    rrd_filt = rrd.copy()
    rrd_filt[q_groups[:, None] != g_groups[None, :]] = np.inf
    indices = np.argsort(rrd_filt, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, num_query + 1)]
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

    print("\n[1/2] PAT-8 features:")
    pat_qf_sum = pat_gf_sum = None
    for ckpt in PAT_8:
        print(f"  {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, num_class=0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract_pat(model, val_loader, num_query)
        pat_qf_sum = qf if pat_qf_sum is None else pat_qf_sum + qf
        pat_gf_sum = gf if pat_gf_sum is None else pat_gf_sum + gf
        del model; torch.cuda.empty_cache()
    pat_qf = pat_qf_sum.numpy().astype(np.float32)
    pat_gf = pat_gf_sum.numpy().astype(np.float32)

    print("\n[2/2] DINOv3 features (frozen):")
    dino_qf, dino_gf = extract_dinov3(val_loader, num_query)
    dino_qf = dino_qf.numpy().astype(np.float32)
    dino_gf = dino_gf.numpy().astype(np.float32)

    print("\n=== Computing rerank distances + writing CSVs ===")
    rrd_pat = compute_rerank_dist(pat_qf, pat_gf)
    rrd_dino = compute_rerank_dist(dino_qf, dino_gf)

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)
    write_csv_from_dist(0.85 * rrd_pat + 0.15 * rrd_dino,
                       'pat8_plus_dinov3_8515', num_query)
    write_csv_from_dist(0.7 * rrd_pat + 0.3 * rrd_dino,
                       'pat8_plus_dinov3_7030', num_query)
    write_csv_from_dist(rrd_dino, 'dinov3_solo', num_query)


if __name__ == '__main__':
    main()
