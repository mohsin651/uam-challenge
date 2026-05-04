"""Cross-resolution ensemble: 8-baseline @ 256x128 + hi-res cam-adv @ 384x192.

The 8-baseline ckpts (7 supervised + 1 cam-adv s500) were trained at 256x128.
The hi-res cam-adv s800 ckpts are trained at 384x192. We can't share a
dataloader (different INPUT.SIZE_TEST), so we extract features in TWO passes,
each at the ckpt's native training resolution. The features come out as
1024-d L2-normed CLS tokens — same vector space, so summing is meaningful
(differs from §71 multi-scale TTA which used the SAME ckpt at off-train
resolutions, breaking pos_embed calibration).

Variants written:
  1. baseline8_plus_hires_ep60_1x        (most direct, single ep60 ckpt)
  2. baseline8_plus_hires_ep60_0p5x      (safer angular weight)
  3. baseline8_plus_hires_ep50_60        (2-ckpt hi-res trajectory)
  4. baseline8_plus_hires_ep40_50_60     (3-ckpt hi-res trajectory)
  5. baseline8_plus_hires_all4           (ep30+40+50+60)
  6. hires_solo_ep60                     (diagnostic)
"""
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import cfg as cfg_global  # mutable; we'll re-merge per group
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.re_ranking import re_ranking


CAMADV_EP60 = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth'
BASELINE_DIR_S1234 = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep'
BASELINE_DIR_S42   = '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42'
HIRES_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_hires_seed800'

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
HIRES_CKPTS = {
    'ep30': f'{HIRES_DIR}/part_attention_vit_30.pth',
    'ep40': f'{HIRES_DIR}/part_attention_vit_40.pth',
    'ep50': f'{HIRES_DIR}/part_attention_vit_50.pth',
    'ep60': f'{HIRES_DIR}/part_attention_vit_60.pth',
}

DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275
OUT_DIR = '/workspace/miuam_challenge_diff/results'

BASE_CFG  = '/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml'
HIRES_CFG = '/workspace/miuam_challenge_diff/config/UrbanElementsReID_test_hires.yml'


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


def write_csv(qf_t, gf_t, label, num_query, val_loader, root_dir):
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, DBA_K)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=RR_K1, k2=RR_K2, lambda_value=RR_LAMBDA)
    CG = {'trafficsignal': 'trafficsignal', 'crosswalk': 'crosswalk',
          'container': 'bin_like', 'rubbishbins': 'bin_like'}
    qcls = pd.read_csv(os.path.join(root_dir, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(root_dir, 'test_classes.csv'))
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


def extract_group(cfg_path, ckpts, label):
    """Build dataloader at the YAML's resolution, extract features per ckpt."""
    print(f"\n  === Group: {label}  cfg={os.path.basename(cfg_path)} ===")
    # Reset cfg from default + merge
    cfg_global.defrost()
    cfg_global.merge_from_file(cfg_path)
    cfg_global.DATALOADER.NUM_WORKERS = 0
    cfg_global.freeze()
    val_loader, num_query = build_reid_test_loader(cfg_global, cfg_global.DATASETS.TEST[0])
    feats = {}
    for tag, ckpt in ckpts.items():
        if not os.path.exists(ckpt):
            print(f"    ! missing {ckpt}, skipping {tag}")
            continue
        print(f"    {tag}: {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg_global, cfg_global.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        feats[tag] = (qf, gf)
        del model; torch.cuda.empty_cache()
    return feats, val_loader, num_query, cfg_global.DATASETS.ROOT_DIR


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Pass 1: baseline-8 at 256x128 — sum to single (qf, gf)
    base_ckpts = {f'b{i}': c for i, c in enumerate(BASELINE_8)}
    base_feats, base_loader, num_query, root_dir = extract_group(BASE_CFG, base_ckpts, 'baseline-8 @ 256x128')
    base_qf = base_gf = None
    for tag, (qf, gf) in base_feats.items():
        base_qf = qf if base_qf is None else base_qf + qf
        base_gf = gf if base_gf is None else base_gf + gf

    # Pass 2: hi-res cam-adv at 384x192 — keep per-epoch feats for variant generation
    hires_feats, hires_loader, _, _ = extract_group(HIRES_CFG, HIRES_CKPTS, 'hires cam-adv @ 384x192')

    os.makedirs(OUT_DIR, exist_ok=True)
    # Sanity check that q/g order matches across the two loaders.
    base_q = [os.path.basename(it[0]) for it in base_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    hires_q = [os.path.basename(it[0]) for it in hires_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    assert base_q == hires_q, "Query order mismatch between base and hi-res loaders!"
    base_g = [os.path.basename(it[0]) for it in base_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']
    hires_g = [os.path.basename(it[0]) for it in hires_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']
    assert base_g == hires_g, "Gallery order mismatch!"
    print("\n  Order sanity check passed.")

    # Variants
    if 'ep60' in hires_feats:
        h_qf60, h_gf60 = hires_feats['ep60']
        write_csv(base_qf + h_qf60, base_gf + h_gf60,
                  'camadv_hires_baseline8_plus_ep60_1x', num_query, base_loader, root_dir)
        write_csv(base_qf + 0.5 * h_qf60, base_gf + 0.5 * h_gf60,
                  'camadv_hires_baseline8_plus_ep60_0p5x', num_query, base_loader, root_dir)
        # Solo (diagnostic — should be much worse)
        write_csv(h_qf60, h_gf60, 'camadv_hires_solo_ep60', num_query, base_loader, root_dir)
    if 'ep50' in hires_feats and 'ep60' in hires_feats:
        h_qf, h_gf = (hires_feats['ep50'][0] + hires_feats['ep60'][0],
                      hires_feats['ep50'][1] + hires_feats['ep60'][1])
        write_csv(base_qf + h_qf, base_gf + h_gf,
                  'camadv_hires_baseline8_plus_ep50_60', num_query, base_loader, root_dir)
    if 'ep40' in hires_feats and 'ep50' in hires_feats and 'ep60' in hires_feats:
        h_qf = hires_feats['ep40'][0] + hires_feats['ep50'][0] + hires_feats['ep60'][0]
        h_gf = hires_feats['ep40'][1] + hires_feats['ep50'][1] + hires_feats['ep60'][1]
        write_csv(base_qf + h_qf, base_gf + h_gf,
                  'camadv_hires_baseline8_plus_ep40_50_60', num_query, base_loader, root_dir)
    if all(t in hires_feats for t in ['ep30', 'ep40', 'ep50', 'ep60']):
        h_qf = sum(hires_feats[t][0] for t in ['ep30', 'ep40', 'ep50', 'ep60'])
        h_gf = sum(hires_feats[t][1] for t in ['ep30', 'ep40', 'ep50', 'ep60'])
        write_csv(base_qf + h_qf, base_gf + h_gf,
                  'camadv_hires_baseline8_plus_all4', num_query, base_loader, root_dir)


if __name__ == '__main__':
    main()
