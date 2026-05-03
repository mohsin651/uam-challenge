"""ViT-Huge inference — solo + cross-arch distance fusion with PAT-8.

ViT-H produces 1280-d features (vs PAT-L's 1024-d) — cannot directly sum.
Fusion at distance level: compute rerank distance matrices separately, then
weighted-average (per the BoT/seresnet50 pattern from §92).

Generates 4 variants:
  A. ViT-H solo ep40 (diagnostic)
  B. ViT-H ep30 + ep40 averaged (intra-arch)
  C. PAT-8 + ViT-H ep40 distance-fusion 70/30 (PAT-dominant, conservative)
  D. PAT-8 + ViT-H ep40 distance-fusion 50/50 (equal weight)
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


VIT_HUGE_DIR = '/workspace/miuam_challenge_diff/models/model_vit_huge_p14_seed1800'
VIT_HUGE_EP40 = f'{VIT_HUGE_DIR}/part_attention_vit_40.pth'
VIT_HUGE_EP30 = f'{VIT_HUGE_DIR}/part_attention_vit_30.pth'

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

    # === ViT-H extraction at 252×126 ===
    print("\n[1/2] Extracting ViT-Huge features (252×126, patch14)...")
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.defrost()
    cfg.MODEL.TRANSFORMER_TYPE = 'vit_huge_patch14_TransReID'
    cfg.MODEL.STRIDE_SIZE = [14, 14]
    cfg.INPUT.SIZE_TEST = [252, 126]
    cfg.INPUT.SIZE_TRAIN = [252, 126]
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    vith_feats = {}
    for ckpt, tag in [(VIT_HUGE_EP40, 'ep40'), (VIT_HUGE_EP30, 'ep30')]:
        if not os.path.exists(ckpt):
            print(f"  WARNING: {ckpt} missing"); continue
        print(f"  ViT-H {tag}: {ckpt}")
        model = make_model(cfg, 'part_attention_vit', num_class=0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        vith_feats[tag] = (qf.numpy().astype(np.float32), gf.numpy().astype(np.float32))
        del model; torch.cuda.empty_cache()

    # === PAT-8 extraction at 256×128 ===
    print("\n[2/2] Extracting PAT-8 features (256×128, patch16)...")
    cfg.defrost()
    cfg.MODEL.TRANSFORMER_TYPE = 'vit_large_patch16_224_TransReID'
    cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    pat_qf_sum = pat_gf_sum = None
    for ckpt in PAT_8:
        print(f"  PAT: {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, 'part_attention_vit', num_class=0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        pat_qf_sum = qf if pat_qf_sum is None else pat_qf_sum + qf
        pat_gf_sum = gf if pat_gf_sum is None else pat_gf_sum + gf
        del model; torch.cuda.empty_cache()
    pat_qf = pat_qf_sum.numpy().astype(np.float32)
    pat_gf = pat_gf_sum.numpy().astype(np.float32)

    # === Compute rerank distance matrices ===
    print("\n=== Computing rerank distances + writing CSVs ===")
    rrd_pat = compute_rerank_dist(pat_qf, pat_gf)
    rrd_vith_ep40 = compute_rerank_dist(*vith_feats['ep40'])
    rrd_vith_avg = None
    if 'ep30' in vith_feats:
        avg_qf = (vith_feats['ep30'][0] + vith_feats['ep40'][0]) / 2
        avg_gf = (vith_feats['ep30'][1] + vith_feats['ep40'][1]) / 2
        rrd_vith_avg = compute_rerank_dist(avg_qf, avg_gf)

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)
    # A: ViT-H solo
    write_csv_from_dist(rrd_vith_ep40, 'vit_huge_solo_ep40', num_query)
    # B: ViT-H ep30+ep40 average
    if rrd_vith_avg is not None:
        write_csv_from_dist(rrd_vith_avg, 'vit_huge_solo_ep30_40', num_query)
    # C: PAT-8 + ViT-H 70/30 (PAT dominant)
    write_csv_from_dist(0.7 * rrd_pat + 0.3 * rrd_vith_ep40,
                       'pat8_plus_vit_huge_7030', num_query)
    # D: PAT-8 + ViT-H 50/50
    write_csv_from_dist(0.5 * rrd_pat + 0.5 * rrd_vith_ep40,
                       'pat8_plus_vit_huge_5050', num_query)


if __name__ == '__main__':
    main()
