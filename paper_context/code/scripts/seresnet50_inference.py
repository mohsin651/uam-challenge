"""SE-ResNet-50 (BoT) inference — SOLO + cross-arch ensemble variants.

The SE-ResNet-50 produces 2048-d features (vs ViT-L PAT's 1024-d) — they CANNOT
be directly summed. Instead:
  1. Run BoT solo (sanity test; tells us if features have any signal at all)
  2. Cross-arch ensembles via DISTANCE-LEVEL fusion: compute rerank distance
     matrix separately for BoT and PAT-8, then average (or weighted-average)
     the matrices.

Generates 4 variants:
  A. BoT ep100 SOLO (diagnostic)
  B. BoT ep80+100 SOLO ensemble (within-arch)
  C. PAT-8 + BoT ep100 distance-fusion @ 0.5/0.5
  D. PAT-8 + BoT ep100 distance-fusion @ 0.7/0.3 (weight PAT more)
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


# SE-ResNet-50 BoT ckpts
SERESNET_DIR = '/workspace/miuam_challenge_diff/models/model_seresnet50_bot_seed1500'
BOT_EP100 = f'{SERESNET_DIR}/seresnet50_100.pth'
BOT_EP80 = f'{SERESNET_DIR}/seresnet50_80.pth'

# PAT-8 ensemble (proven 0.15884)
CAMADV_S500_EP60 = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42   = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'
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


def compute_rerank_dist(qf, gf):
    """Standard pipeline: DBA + cosine + rerank → returns rerank distance matrix."""
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    return re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)


def write_csv_from_dist(rrd, label, num_query):
    """Apply class-group filter and write top-100 CSV from a rerank distance matrix."""
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

    # === BoT extraction (uses different config — needs MODEL.NAME=seresnet50) ===
    print("\n[1/2] Extracting SE-ResNet-50 (BoT) features...")
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.defrost()
    cfg.MODEL.NAME = 'seresnet50'
    cfg.MODEL.PC_LOSS = False
    cfg.MODEL.SOFT_LABEL = False
    cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]   # SE-ResNet was trained with these
    cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    bot_qf_sum = bot_gf_sum = None
    bot_qf_100 = bot_gf_100 = None
    for ckpt, tag in [(BOT_EP100, 'ep100'), (BOT_EP80, 'ep80')]:
        if not os.path.exists(ckpt):
            print(f"  WARNING: {ckpt} not found — skipping {tag}")
            continue
        print(f"  {tag}: {ckpt}")
        model = make_model(cfg, cfg.MODEL.NAME, num_class=1088)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        del model; torch.cuda.empty_cache()
        if tag == 'ep100':
            bot_qf_100, bot_gf_100 = qf.numpy().astype(np.float32), gf.numpy().astype(np.float32)
        bot_qf_sum = qf if bot_qf_sum is None else bot_qf_sum + qf
        bot_gf_sum = gf if bot_gf_sum is None else bot_gf_sum + gf

    bot_qf_avg = (bot_qf_sum / 2).numpy().astype(np.float32) if bot_qf_sum is not None else None
    bot_gf_avg = (bot_gf_sum / 2).numpy().astype(np.float32) if bot_gf_sum is not None else None

    # === PAT-8 extraction (standard config) ===
    print("\n[2/2] Extracting PAT-8 features...")
    cfg.defrost()
    cfg.MODEL.NAME = 'part_attention_vit'
    cfg.MODEL.PC_LOSS = True   # PAT was trained with this
    cfg.MODEL.SOFT_LABEL = True
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    pat_qf_sum = pat_gf_sum = None
    for ckpt in PAT_8:
        print(f"  PAT: {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, num_class=0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        pat_qf_sum = qf if pat_qf_sum is None else pat_qf_sum + qf
        pat_gf_sum = gf if pat_gf_sum is None else pat_gf_sum + gf
        del model; torch.cuda.empty_cache()

    pat_qf = pat_qf_sum.numpy().astype(np.float32)
    pat_gf = pat_gf_sum.numpy().astype(np.float32)

    print("\n=== Computing rerank distance matrices ===")
    rrd_bot100 = compute_rerank_dist(bot_qf_100, bot_gf_100)
    rrd_bot_avg = compute_rerank_dist(bot_qf_avg, bot_gf_avg) if bot_qf_avg is not None else None
    rrd_pat = compute_rerank_dist(pat_qf, pat_gf)

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)

    # A: BoT solo (ep100)
    write_csv_from_dist(rrd_bot100, 'seresnet50_solo_ep100', num_query)
    # B: BoT ep80+100 average (intra-arch)
    if rrd_bot_avg is not None:
        write_csv_from_dist(rrd_bot_avg, 'seresnet50_solo_ep80_100', num_query)
    # C: PAT-8 + BoT distance-level fusion (50/50)
    rrd_fused_5050 = 0.5 * rrd_pat + 0.5 * rrd_bot100
    write_csv_from_dist(rrd_fused_5050, 'pat8_plus_seresnet50_5050', num_query)
    # D: PAT-8 weighted higher (70/30)
    rrd_fused_7030 = 0.7 * rrd_pat + 0.3 * rrd_bot100
    write_csv_from_dist(rrd_fused_7030, 'pat8_plus_seresnet50_7030', num_query)


if __name__ == '__main__':
    main()
