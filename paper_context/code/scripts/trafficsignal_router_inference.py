"""Trafficsignal-specialist routed inference.

For each query, route to the appropriate model:
  - Trafficsignal queries (582 of 928, 63%) → specialist features
  - Non-trafficsignal queries (346)          → unified ensemble features

Each route runs its OWN sub-pipeline (DBA + rerank + class-group filter)
on its filtered query/gallery subset, then results are merged in original
query order to produce one Kaggle CSV.

Variants generated:
  A. spec(4-ckpt) for ts, 8-unified for non-ts                      (default)
  B. spec(4-ckpt) for ts, 7-baseline (no cam-adv) for non-ts        (hedge)
  C. spec(4-ckpt) + 8-unified for ts, 8-unified for non-ts          (stack)
  D. spec(4-ckpt) + cam-adv ep60 for ts, 8-unified for non-ts       (specialist + camera-invariance)
  E. spec_ep60-solo for ts, 8-unified for non-ts                     (analogous to cam-adv ep60-only)
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


SPEC_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_trafficsignal_seed800'
CAMADV_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42   = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'

SPECIALIST_CKPTS = [
    f'{SPEC_DIR}/part_attention_vit_30.pth',
    f'{SPEC_DIR}/part_attention_vit_40.pth',
    f'{SPEC_DIR}/part_attention_vit_50.pth',
    f'{SPEC_DIR}/part_attention_vit_60.pth',
]
SPECIALIST_EP60 = f'{SPEC_DIR}/part_attention_vit_60.pth'
CAMADV_EP60 = f'{CAMADV_DIR}/part_attention_vit_60.pth'
BASELINE_7CKPT = [
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


def extract_sum(ckpts, loader, num_query):
    qf_sum = gf_sum = None
    for ckpt in ckpts:
        print(f"    {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, loader, num_query)
        qf_sum = qf if qf_sum is None else qf_sum + qf
        gf_sum = gf if gf_sum is None else gf_sum + gf
        del model; torch.cuda.empty_cache()
    return qf_sum, gf_sum


def db_augment(gf, k):
    if k <= 0:
        return gf
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def rerank_block(qf_t, gf_t, cross_mask=None,
                 dba_k=DBA_K, k1=RR_K1, k2=RR_K2, lam=RR_LAMBDA):
    """Returns top-100 sorted gallery LOCAL indices for each query."""
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, dba_k)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)
    if cross_mask is not None:
        rrd[cross_mask] = np.inf
    return np.argsort(rrd, axis=1)[:, :100]


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    print(f"num_query: {num_query}, gallery: {len(val_loader.dataset.img_items) - num_query}")

    # ---- Load query/gallery class info ----
    qcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']

    q_name2cls = {n: c.lower() for n, c in zip(qcls['imageName'], qcls['Class'])}
    g_name2cls = {n: c.lower() for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_classes = np.array([q_name2cls[os.path.basename(it[0])] for it in q_items])
    g_classes = np.array([g_name2cls[os.path.basename(it[0])] for it in g_items])

    ts_q_idx = np.where(q_classes == 'trafficsignal')[0]
    nts_q_idx = np.where(q_classes != 'trafficsignal')[0]
    ts_g_idx = np.where(g_classes == 'trafficsignal')[0]
    nts_g_idx = np.where(g_classes != 'trafficsignal')[0]
    print(f"  TS queries: {len(ts_q_idx)}, TS gallery: {len(ts_g_idx)}")
    print(f"  non-TS queries: {len(nts_q_idx)}, non-TS gallery: {len(nts_g_idx)}")

    # Class-group filter for non-TS sub-problem (container ∪ rubbishbins → bin_like)
    CG = {'crosswalk': 'crosswalk', 'container': 'bin_like', 'rubbishbins': 'bin_like'}
    nts_q_grp = np.array([CG[c] for c in q_classes[nts_q_idx]])
    nts_g_grp = np.array([CG[c] for c in g_classes[nts_g_idx]])
    nts_cross = nts_q_grp[:, None] != nts_g_grp[None, :]

    # ---- Extract specialist features ----
    print("\n  specialist 4-ckpt extraction:")
    spec_qf_4, spec_gf_4 = extract_sum(SPECIALIST_CKPTS, val_loader, num_query)
    print("\n  specialist ep60-solo extraction:")
    spec_qf_60, spec_gf_60 = extract_sum([SPECIALIST_EP60], val_loader, num_query)

    # ---- Extract baseline-7 features ----
    print("\n  baseline 7-ckpt extraction:")
    base_qf, base_gf = extract_sum(BASELINE_7CKPT, val_loader, num_query)

    # ---- Extract cam-adv ep60 features ----
    print("\n  cam-adv ep60 extraction:")
    cam_qf, cam_gf = extract_sum([CAMADV_EP60], val_loader, num_query)

    # Convenience: 8-unified = baseline-7 + cam-adv ep60
    uni_qf = base_qf + cam_qf
    uni_gf = base_gf + cam_gf

    os.makedirs(OUT_DIR, exist_ok=True)

    def write_router(label, ts_qf, ts_gf, nts_qf, nts_gf):
        """Write a routed CSV. Each route uses its own features sub-pipeline."""
        # TS sub-problem: specialist features, no cross-class mask (all are trafficsignal)
        ts_idx_local = rerank_block(
            ts_qf[ts_q_idx], ts_gf[ts_g_idx], cross_mask=None
        )
        # Map TS local → global gallery index
        ts_idx_global = ts_g_idx[ts_idx_local]

        # Non-TS sub-problem: unified features (or baseline), with bin_like class-group filter
        nts_idx_local = rerank_block(
            nts_qf[nts_q_idx], nts_gf[nts_g_idx], cross_mask=nts_cross
        )
        nts_idx_global = nts_g_idx[nts_idx_local]

        # Stitch back into 928-row output, preserving original query order
        out_indices = np.zeros((num_query, 100), dtype=np.int64)
        out_indices[ts_q_idx] = ts_idx_global
        out_indices[nts_q_idx] = nts_idx_global

        out_path = f'{OUT_DIR}/router_{label}_submission.csv'
        names = [f'{i:06d}.jpg' for i in range(1, num_query + 1)]
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
            for n, t in zip(names, out_indices):
                w.writerow([n, ' '.join(map(str, t + 1))])
        print(f"  → {out_path}")

    print("\n=== generating routed variants ===")
    # A: spec(4-ckpt) for ts, 8-unified for non-ts (default expected best)
    write_router('A_spec4_ts_uni8_nts', spec_qf_4, spec_gf_4, uni_qf, uni_gf)

    # B: spec(4-ckpt) for ts, 7-baseline (no cam-adv) for non-ts
    write_router('B_spec4_ts_base7_nts', spec_qf_4, spec_gf_4, base_qf, base_gf)

    # C: spec(4-ckpt) + 8-unified for ts, 8-unified for non-ts (stack)
    write_router('C_spec4plusUni8_ts_uni8_nts',
                 spec_qf_4 + uni_qf, spec_gf_4 + uni_gf, uni_qf, uni_gf)

    # D: spec(4-ckpt) + cam-adv ep60 for ts, 8-unified for non-ts
    write_router('D_spec4plusCamAdv_ts_uni8_nts',
                 spec_qf_4 + cam_qf, spec_gf_4 + cam_gf, uni_qf, uni_gf)

    # E: spec_ep60-solo for ts, 8-unified for non-ts (analog of cam-adv ep60-only)
    write_router('E_specEp60_ts_uni8_nts', spec_qf_60, spec_gf_60, uni_qf, uni_gf)


if __name__ == '__main__':
    main()
