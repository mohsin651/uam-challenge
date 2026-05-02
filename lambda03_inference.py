"""λ=0.3 cam-adv inference — test ALL 6 epochs individually.

The hypothesis (per user via Path 2): at stronger GRL pressure (λ=0.3 vs λ=0.1),
earlier epochs may converge faster to a camera-invariant manifold that's
ensemble-compatible. The original λ=0.1 found ONLY ep60 was compatible (§66).
At λ=0.3, maybe ep40 or ep50 is also compatible, or maybe ep60 itself is BETTER.

Generates 6 CSVs (one per cam-adv-λ03 epoch added to 7-baseline at 1× weight).
Plus 2 backup variants:
- λ03 ep60 at 0.5× weight (more conservative angular contribution)
- λ03 ep60 SOLO (diagnostic)

Per "1-2 high-EV variants" user preference: I'll RECOMMEND which to submit
first based on training Acc trajectory. User picks 1.
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


LAMBDA03_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_lambda03_seed1400'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42   = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'

BASELINE_7 = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
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
    out = f'{OUT_DIR}/{label}_submission.csv'
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

    print("\n  7-baseline extraction:")
    base_qf = base_gf = None
    for ckpt in BASELINE_7:
        print(f"    {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        base_qf = qf if base_qf is None else base_qf + qf
        base_gf = gf if base_gf is None else base_gf + gf
        del model; torch.cuda.empty_cache()

    print("\n  λ=0.3 cam-adv per-epoch extraction:")
    lambda03_feats = {}
    for ep in [10, 20, 30, 40, 50, 60]:
        ckpt = f'{LAMBDA03_DIR}/part_attention_vit_{ep}.pth'
        if not os.path.exists(ckpt):
            print(f"    ep{ep}: ckpt missing, skipping")
            continue
        print(f"    ep{ep}: {ckpt}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        lambda03_feats[ep] = (qf, gf)
        del model; torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n=== Writing per-epoch ensemble variants ===")
    # 7-baseline + each λ=0.3 ckpt at 1× weight (mirroring the proven pattern)
    for ep, (qf, gf) in lambda03_feats.items():
        write_csv(base_qf + qf, base_gf + gf,
                  f'lambda03_baseline7_plus_ep{ep}', num_query, val_loader)

    # 7-baseline + λ=0.3 ep60 at 0.5× weight (lower angular contribution)
    if 60 in lambda03_feats:
        qf60, gf60 = lambda03_feats[60]
        write_csv(base_qf + 0.5 * qf60, base_gf + 0.5 * gf60,
                  'lambda03_baseline7_plus_ep60_half', num_query, val_loader)

    # λ=0.3 ep60 SOLO (diagnostic — tests pure cam-adv-λ03 features)
    if 60 in lambda03_feats:
        qf60, gf60 = lambda03_feats[60]
        write_csv(qf60, gf60, 'lambda03_solo_ep60', num_query, val_loader)


if __name__ == '__main__':
    main()
