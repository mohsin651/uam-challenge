"""Camera-adversarial run inference + ensemble experiments.

Extracts features from the 4 cam-adv ckpts (ep30/40/50/60) AND the proven
7-ckpt set, then writes CSVs for several variants:
  1. cam-adv solo (4 ckpts)
  2. cam-adv last-3 (ep40/50/60)
  3. cam-adv last-2 (ep50/60)
  4. cam-adv last-1 (ep60)
  5. 7-ckpt baseline + cam-adv solo (11 ckpts, equal weight)
  6. 7-ckpt baseline + cam-adv last-3 (10 ckpts)

All variants apply the proven post-processing (DBA k=8, rerank k1=15/k2=4/λ=0.275,
class-group filter). The cam_classifier head's weight is skipped by load_param's
"classifier in name → skip" rule, so the test config can have CAM_ADV=False.
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


CAMADV_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42   = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'

CAMADV_CKPTS = {
    'ep30': f'{CAMADV_DIR}/part_attention_vit_30.pth',
    'ep40': f'{CAMADV_DIR}/part_attention_vit_40.pth',
    'ep50': f'{CAMADV_DIR}/part_attention_vit_50.pth',
    'ep60': f'{CAMADV_DIR}/part_attention_vit_60.pth',
}

BASELINE_7CKPT = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
]

# Post-processing winner from the 7-ckpt era.
DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275


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
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def write_csv(qf_t, gf_t, label, num_query, val_loader):
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)

    import pandas as pd
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
    out = f'/workspace/miuam_challenge_diff/results/camadv_{label}_submission.csv'
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
    print(f"num_query: {num_query}, gallery: {len(val_loader.dataset.img_items) - num_query}")

    # Extract per-ckpt features (cam-adv).
    cam_feats = {}
    for tag, ckpt in CAMADV_CKPTS.items():
        print(f"  cam-adv {tag}: loading {os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract_feature(model, val_loader, num_query)
        cam_feats[tag] = (qf, gf)
        del model; torch.cuda.empty_cache()

    # Extract baseline-7 sum.
    print("\n  baseline 7-ckpt extraction:")
    base_qf = base_gf = None
    for ckpt in BASELINE_7CKPT:
        print(f"    {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract_feature(model, val_loader, num_query)
        base_qf = qf if base_qf is None else base_qf + qf
        base_gf = gf if base_gf is None else base_gf + gf
        del model; torch.cuda.empty_cache()

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)

    def sumf(tags):
        qf = sum(cam_feats[t][0] for t in tags)
        gf = sum(cam_feats[t][1] for t in tags)
        return qf, gf

    # Variants
    qf, gf = sumf(['ep30','ep40','ep50','ep60'])
    write_csv(qf, gf, 'solo_ep30_40_50_60', num_query, val_loader)

    qf, gf = sumf(['ep40','ep50','ep60'])
    write_csv(qf, gf, 'solo_ep40_50_60', num_query, val_loader)

    qf, gf = sumf(['ep50','ep60'])
    write_csv(qf, gf, 'solo_ep50_60', num_query, val_loader)

    qf, gf = sumf(['ep60'])
    write_csv(qf, gf, 'solo_ep60', num_query, val_loader)

    # Mixed: 7-ckpt baseline + cam-adv (4 ckpts) — 11 ckpts equal weight
    qf, gf = sumf(['ep30','ep40','ep50','ep60'])
    write_csv(base_qf + qf, base_gf + gf, 'baseline7_plus_camadv_4', num_query, val_loader)

    # Mixed: 7-ckpt baseline + cam-adv last-3 (10 ckpts)
    qf, gf = sumf(['ep40','ep50','ep60'])
    write_csv(base_qf + qf, base_gf + gf, 'baseline7_plus_camadv_3', num_query, val_loader)

    # Mixed: 7-ckpt baseline + cam-adv last-2 (9 ckpts)
    qf, gf = sumf(['ep50','ep60'])
    write_csv(base_qf + qf, base_gf + gf, 'baseline7_plus_camadv_2', num_query, val_loader)

    # Mixed: 7-ckpt baseline + cam-adv ep60 only (8 ckpts)
    qf, gf = sumf(['ep60'])
    write_csv(base_qf + qf, base_gf + gf, 'baseline7_plus_camadv_ep60', num_query, val_loader)


if __name__ == '__main__':
    main()
