"""Inference for cam-adv + merged dataset + heavy-aug model (seed=2500).

Trained with: PAT-L + cam-adv (λ=0.1, NUM_CAMERAS=4) + Urban2026_merged (UAM)
+ heavy aug suite (perspective, rotation, color, erasing, LGT). 60 ep.

Generates 2 variants:
  PRIMARY: ep60 SOLO (diagnostic — does this 4-lever combo learn anything?)
  BACKUP: PAT-8 + ep60 fusion (sum-features, 9 ckpts total)
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


CAMADV_MERGED_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_merged_heavyaug_seed2500'
NEW_CKPT = f'{CAMADV_MERGED_DIR}/part_attention_vit_60.pth'

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
    if k <= 0: return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def write_csv(qf_t, gf_t, label, num_query):
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32) if torch.is_tensor(qf_t) else qf_t
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32) if torch.is_tensor(gf_t) else gf_t
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(URBAN_ROOT, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(URBAN_ROOT, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_groups = np.array([q2g[f'{i:06d}.jpg'] for i in range(1, num_query + 1)])
    g_groups = np.array([g2g[f'{i:06d}.jpg'] for i in range(1, gf.shape[0] + 1)])
    rrd[q_groups[:, None] != g_groups[None, :]] = np.inf

    indices = np.argsort(rrd, axis=1)[:, :100]
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

    print(f"\n[1/2] New ckpt (cam-adv + merged + heavy-aug seed=2500 ep60):")
    print(f"  {NEW_CKPT}")
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(NEW_CKPT)
    model = model.cuda().eval()
    new_qf, new_gf = extract(model, val_loader, num_query)
    del model; torch.cuda.empty_cache()

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)
    print("\n=== Variant A: NEW ckpt SOLO (diagnostic) ===")
    write_csv(new_qf.clone(), new_gf.clone(), 'camadv_merged_heavyaug_s2500_ep60_solo', num_query)

    print(f"\n[2/2] PAT-8 features (sum):")
    pat_qf = pat_gf = None
    for ckpt in PAT_8:
        print(f"  {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        pat_qf = qf if pat_qf is None else pat_qf + qf
        pat_gf = gf if pat_gf is None else pat_gf + gf
        del model; torch.cuda.empty_cache()

    print("\n=== Variant B: PAT-8 + new ckpt (9-ckpt fusion, sum-features) ===")
    write_csv(pat_qf + new_qf, pat_gf + new_gf,
              'pat8_plus_camadv_merged_heavyaug_s2500_ep60', num_query)


if __name__ == '__main__':
    main()
