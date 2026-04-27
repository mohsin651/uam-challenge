"""Training-free probes around the 0.15884 winner (7-baseline + cam-adv ep60).

Extracts features ONCE (8 ckpts), then sweeps:
  A. Weighting on cam-adv ep60: 1.5x, 2x, 3x, plus de-weighted baseline
  B. DBA-k around 8: try k=6, 7, 9, 10
  C. Rerank lambda around 0.275: try 0.25, 0.30, 0.325
  D. Rerank k1 around 15: try 14, 16

All variants apply class-group filter at the end. CSVs land in results/.
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

BASELINE_7CKPT = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
]
CAMADV_EP60 = f'{CAMADV_DIR}/part_attention_vit_60.pth'


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


def _class_groups(val_loader, num_query):
    import pandas as pd
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
    return q_groups[:, None] != g_groups[None, :]


def write_csv(qf_t, gf_t, label, num_query, val_loader,
              dba_k=8, k1=15, k2=4, lam=0.275, cross_mask=None):
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, dba_k)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)
    rrd[cross_mask] = np.inf
    indices = np.argsort(rrd, axis=1)[:, :100]
    out = f'/workspace/miuam_challenge_diff/results/freeprobe_{label}_submission.csv'
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
    cross_mask = _class_groups(val_loader, num_query)
    print(f"num_query: {num_query}, cross-group mask shape: {cross_mask.shape}")

    # --- Extract baseline-7 sum and cam-adv ep60 separately ---
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

    print(f"\n  cam-adv ep60 extraction:")
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(CAMADV_EP60)
    model = model.cuda().eval()
    cam_qf, cam_gf = extract_feature(model, val_loader, num_query)
    del model; torch.cuda.empty_cache()

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)

    # === A. Weighting variants on cam-adv ep60, with proven post-proc (k=8, λ=0.275) ===
    print("\n=== A. weighting variants ===")
    for w in [1.5, 2.0, 3.0]:
        write_csv(base_qf + w * cam_qf, base_gf + w * cam_gf,
                  f'w{w:.1f}xcamadv_dba8_k15_lam0275', num_query, val_loader,
                  dba_k=8, k1=15, k2=4, lam=0.275, cross_mask=cross_mask)
    # baseline weight reduction (de-emphasize baseline so cam-adv has more relative weight)
    for bw in [0.5, 0.75]:
        write_csv(bw * base_qf + cam_qf, bw * base_gf + cam_gf,
                  f'b{bw:.2f}x_w1xcamadv_dba8_k15_lam0275', num_query, val_loader,
                  dba_k=8, k1=15, k2=4, lam=0.275, cross_mask=cross_mask)

    # === B. DBA-k sweep on the proven 1×+1× sum ===
    print("\n=== B. DBA-k sweep on 0.15884 ensemble ===")
    for k in [6, 7, 9, 10]:
        write_csv(base_qf + cam_qf, base_gf + cam_gf,
                  f'dba{k}_k15_lam0275', num_query, val_loader,
                  dba_k=k, k1=15, k2=4, lam=0.275, cross_mask=cross_mask)

    # === C. Rerank λ sweep ===
    print("\n=== C. rerank lambda sweep on 0.15884 ensemble ===")
    for lam in [0.250, 0.300, 0.325]:
        write_csv(base_qf + cam_qf, base_gf + cam_gf,
                  f'dba8_k15_lam{int(lam*1000):03d}', num_query, val_loader,
                  dba_k=8, k1=15, k2=4, lam=lam, cross_mask=cross_mask)

    # === D. Rerank k1 sweep ===
    print("\n=== D. rerank k1 sweep on 0.15884 ensemble ===")
    for k1 in [14, 16, 18]:
        write_csv(base_qf + cam_qf, base_gf + cam_gf,
                  f'dba8_k{k1}_lam0275', num_query, val_loader,
                  dba_k=8, k1=k1, k2=4, lam=0.275, cross_mask=cross_mask)


if __name__ == '__main__':
    main()
