"""DBSCAN-pseudo-tuned model + 8-ckpt baseline ensemble inference.

Generates submission CSVs combining the proven 8-ckpt baseline (0.15884) with
the DBSCAN-pseudo-tuned ckpts (ep5, ep10) at various weights.

Variants:
  1. baseline-8 + 1.0 * pseudo_ep10           (most direct test)
  2. baseline-8 + 1.0 * pseudo_ep5            (less-adapted)
  3. baseline-8 + 0.5 * pseudo_ep10           (lower weight, safer)
  4. baseline-8 + (pseudo_ep5 + pseudo_ep10)  (mini trajectory ensemble of pseudo)
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
PSEUDO_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_dbscan_pseudo_seed950'
PSEUDO_EP5  = f'{PSEUDO_DIR}/part_attention_vit_5.pth'
PSEUDO_EP10 = f'{PSEUDO_DIR}/part_attention_vit_10.pth'

BASELINE_8 = [
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
OUT_DIR = '/workspace/miuam_challenge_diff/results'


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


def write_csv(qf_t, gf_t, label, num_query, val_loader):
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)
    CG = {'trafficsignal': 'trafficsignal', 'crosswalk': 'crosswalk',
          'container': 'bin_like', 'rubbishbins': 'bin_like'}
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
    out = f'{OUT_DIR}/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, len(indices) + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"  -> {out}")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    print("\n  baseline-8 ckpt extraction:")
    base_qf = base_gf = None
    for ckpt in BASELINE_8:
        print(f"    {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        base_qf = qf if base_qf is None else base_qf + qf
        base_gf = gf if base_gf is None else base_gf + gf
        del model; torch.cuda.empty_cache()

    pseudo_feats = {}
    for tag, ckpt in [('ep5', PSEUDO_EP5), ('ep10', PSEUDO_EP10)]:
        if not os.path.exists(ckpt):
            print(f"  ! {ckpt} not found, skipping {tag}")
            continue
        print(f"\n  pseudo-tuned {tag}: {ckpt}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        pseudo_feats[tag] = (qf, gf)
        del model; torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    if 'ep10' in pseudo_feats:
        # 8-baseline + 1.0 * pseudo_ep10  (primary high-EV variant)
        write_csv(base_qf + pseudo_feats['ep10'][0], base_gf + pseudo_feats['ep10'][1],
                  'dbscan_pseudo_baseline8_plus_ep10_1x', num_query, val_loader)
        # 8-baseline + 0.5 * pseudo_ep10  (safer if angular threshold concern)
        write_csv(base_qf + 0.5 * pseudo_feats['ep10'][0], base_gf + 0.5 * pseudo_feats['ep10'][1],
                  'dbscan_pseudo_baseline8_plus_ep10_0p5x', num_query, val_loader)
    if 'ep5' in pseudo_feats:
        write_csv(base_qf + pseudo_feats['ep5'][0], base_gf + pseudo_feats['ep5'][1],
                  'dbscan_pseudo_baseline8_plus_ep5_1x', num_query, val_loader)
    if 'ep5' in pseudo_feats and 'ep10' in pseudo_feats:
        write_csv(base_qf + pseudo_feats['ep5'][0] + pseudo_feats['ep10'][0],
                  base_gf + pseudo_feats['ep5'][1] + pseudo_feats['ep10'][1],
                  'dbscan_pseudo_baseline8_plus_ep5_ep10', num_query, val_loader)
    # Solo (diagnostic only — pseudo alone should be much worse than ensemble)
    if 'ep10' in pseudo_feats:
        write_csv(pseudo_feats['ep10'][0], pseudo_feats['ep10'][1],
                  'dbscan_pseudo_solo_ep10', num_query, val_loader)


if __name__ == '__main__':
    main()
