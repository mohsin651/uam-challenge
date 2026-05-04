"""Ensemble leave-one-out pruning.

Extract per-ckpt features (cache them), then for each subset of size 7
(removing one ckpt at a time), compute the rerank+filter score *signal*
on a HELD-OUT proxy (top-1 self-similarity entropy on gallery — proxy for
mAP without ground truth).

Wait — we have NO ground truth on test. The proxy approach is unreliable.

Instead: just generate 8 CSVs (one per leave-one-out), submit the one
predicted best. Or simpler — generate ALL 8, plus the full-8 baseline,
look at how each subset's top-1 list differs from the proven 0.15884 list
and pick the one with highest agreement (sanity) and highest distinctness
on hard cases (signal).

Honestly the cleanest: generate 8 leave-one-out CSVs, pick top 1-2 most
different from each other (and from the full-8 baseline) and submit those.

This script generates 8 leave-one-out submission CSVs.
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
LABELS = [
    's1234_ep30', 's1234_ep40', 's1234_ep50',
    's42_ep30',   's42_ep40',   's42_ep50',   's42_ep60',
    'camadv_s500_ep60',
]
DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275
URBAN_ROOT = '/workspace/Urban2026/'
PERCKPT_CACHE = '/workspace/miuam_challenge_diff/results/cache/perckpt_test.npz'


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


def write_csv_from_feats(qf, gf, label, num_query):
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
    return indices


def load_or_extract_perckpt():
    if os.path.exists(PERCKPT_CACHE):
        print(f"  loading per-ckpt cache from {PERCKPT_CACHE}")
        d = np.load(PERCKPT_CACHE)
        return d['qfs'], d['gfs'], int(d['num_query'])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    qfs, gfs = [], []
    for ckpt, lab in zip(PAT_8, LABELS):
        print(f"  extracting {lab}: {ckpt}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        qfs.append(qf.numpy().astype(np.float32))
        gfs.append(gf.numpy().astype(np.float32))
        del model; torch.cuda.empty_cache()

    qfs = np.stack(qfs, axis=0)  # (8, n_q, 1024)
    gfs = np.stack(gfs, axis=0)  # (8, n_g, 1024)
    os.makedirs(os.path.dirname(PERCKPT_CACHE), exist_ok=True)
    np.savez(PERCKPT_CACHE, qfs=qfs, gfs=gfs, num_query=num_query)
    print(f"  saved per-ckpt cache to {PERCKPT_CACHE}")
    return qfs, gfs, num_query


def main():
    print("[1/2] Per-ckpt features:")
    qfs, gfs, num_query = load_or_extract_perckpt()
    print(f"  qfs {qfs.shape}, gfs {gfs.shape}, num_query={num_query}")

    print("\n[2/2] Generating 8 leave-one-out CSVs + 1 full-8 baseline:")
    # Full-8 baseline (sanity — should match 0.15884)
    full_qf = qfs.sum(axis=0); full_gf = gfs.sum(axis=0)
    write_csv_from_feats(full_qf, full_gf, 'prune_full8_sanity', num_query)
    print(f"  → prune_full8_sanity (should match 0.15884 baseline)")

    # 8 leave-one-out
    for i, lab in enumerate(LABELS):
        keep = [j for j in range(8) if j != i]
        qf = qfs[keep].sum(axis=0)
        gf = gfs[keep].sum(axis=0)
        out_label = f'prune_drop_{lab}'
        idx = write_csv_from_feats(qf, gf, out_label, num_query)
        print(f"  → drop {lab}")


if __name__ == '__main__':
    main()
