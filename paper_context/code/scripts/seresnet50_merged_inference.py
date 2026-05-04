"""SE-ResNet-50 + heavy aug + merged data inference — SOLO + cross-arch fusion.

Per the paper recipe (Diaz Benito et al. ICIPW 2025), submit SOLO. Backup
variants for safety.

Generates 4 CSVs:
  A. SE-ResNet-50 merged ep100 SOLO (primary; the paper-style submission)
  B. SE-ResNet-50 merged ep80+100 averaged (intra-arch ensemble)
  C. PAT-8 + SE-ResNet-50 merged 70/30 distance fusion (hedge)
  D. PAT-8 + SE-ResNet-50 merged 50/50 distance fusion (hedge)
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


SERESNET_DIR = '/workspace/miuam_challenge_diff/models/model_seresnet50_merged_seed2000'
BOT_EP100 = f'{SERESNET_DIR}/seresnet50_100.pth'
BOT_EP80 = f'{SERESNET_DIR}/seresnet50_80.pth'

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

    print("\n[1/2] Extracting SE-ResNet-50 (merged-trained) features...")
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.defrost()
    cfg.MODEL.NAME = 'seresnet50'
    cfg.MODEL.PC_LOSS = False
    cfg.MODEL.SOFT_LABEL = False
    cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    bot_qf_100 = bot_gf_100 = None
    bot_qf_avg = bot_gf_avg = None
    bot_sums_q = bot_sums_g = None
    n_extracted = 0
    for ckpt, tag in [(BOT_EP100, 'ep100'), (BOT_EP80, 'ep80')]:
        print(f"  {tag}: {ckpt}")
        # Use 1567 classes for the merged-trained model
        model = make_model(cfg, cfg.MODEL.NAME, num_class=1567)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        if tag == 'ep100':
            bot_qf_100, bot_gf_100 = qf.numpy().astype(np.float32), gf.numpy().astype(np.float32)
        bot_sums_q = qf if bot_sums_q is None else bot_sums_q + qf
        bot_sums_g = gf if bot_sums_g is None else bot_sums_g + gf
        n_extracted += 1
        del model; torch.cuda.empty_cache()
    bot_qf_avg = (bot_sums_q / n_extracted).numpy().astype(np.float32)
    bot_gf_avg = (bot_sums_g / n_extracted).numpy().astype(np.float32)

    print("\n[2/2] Extracting PAT-8 features (for cross-arch fusion variants)...")
    cfg.defrost()
    cfg.MODEL.NAME = 'part_attention_vit'
    cfg.MODEL.PC_LOSS = True
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

    print("\n=== Computing rerank distances + writing CSVs ===")
    rrd_bot100 = compute_rerank_dist(bot_qf_100, bot_gf_100)
    rrd_bot_avg = compute_rerank_dist(bot_qf_avg, bot_gf_avg)
    rrd_pat = compute_rerank_dist(pat_qf, pat_gf)

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)
    # A: SE-ResNet-50 merged solo ep100 (paper-style — primary)
    write_csv_from_dist(rrd_bot100, 'seresnet50_merged_solo_ep100', num_query)
    # B: ep80+100 average (intra-arch)
    write_csv_from_dist(rrd_bot_avg, 'seresnet50_merged_solo_ep80_100', num_query)
    # C: PAT-8 + BoT-merged 70/30
    write_csv_from_dist(0.7 * rrd_pat + 0.3 * rrd_bot100,
                       'pat8_plus_seresnet50_merged_7030', num_query)
    # D: PAT-8 + BoT-merged 50/50
    write_csv_from_dist(0.5 * rrd_pat + 0.5 * rrd_bot100,
                       'pat8_plus_seresnet50_merged_5050', num_query)


if __name__ == '__main__':
    main()
