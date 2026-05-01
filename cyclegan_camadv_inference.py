"""CycleGAN-augmented cam-adv ensemble inference.

Extracts features from the 8-ckpt baseline (7 supervised + cam-adv s500 ep60)
and the new cam-adv-cyclegan ep30/40/50/60 ckpts, then writes 6 weighted
ensemble variants. Same proven post-processing: DBA k=8, rerank k1=15,
k2=4, lambda=0.275, class-group filter.

Following Session 7 finding (only converged ep60 cam-adv ckpts are
ensemble-compatible at 1x), the primary submission candidate is ep60 only.
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
CYCLEGAN_DIR = '/workspace/miuam_challenge_diff/models/model_vitlarge_camadv_cyclegan_seed1100'

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
CYCLEGAN_CKPTS = {
    'ep30': f'{CYCLEGAN_DIR}/part_attention_vit_30.pth',
    'ep40': f'{CYCLEGAN_DIR}/part_attention_vit_40.pth',
    'ep50': f'{CYCLEGAN_DIR}/part_attention_vit_50.pth',
    'ep60': f'{CYCLEGAN_DIR}/part_attention_vit_60.pth',
}

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
    cfg.DATALOADER.NUM_WORKERS = 0  # host fork limit
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    print("\n  baseline-8 ckpt extraction (proven 0.15884 set):")
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

    cyclegan_feats = {}
    for tag, ckpt in CYCLEGAN_CKPTS.items():
        if not os.path.exists(ckpt):
            print(f"  ! {tag}: missing {ckpt}, skipping")
            continue
        print(f"\n  cyclegan-cam-adv {tag}: {os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        cyclegan_feats[tag] = (qf, gf)
        del model; torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    # NOTE: Training crashed at ep50 in-domain eval. Most-converged ckpt is ep40.
    # Cyclegan-augmented training lacks the GRL escalation that made the §60.8
    # compatibility-window finding apply only to ep60. Ep40 should be reasonable.
    most_recent = max([k for k in cyclegan_feats.keys() if k.startswith('ep')],
                      key=lambda x: int(x[2:])) if cyclegan_feats else None
    print(f"\n  Most-converged ckpt available: {most_recent}")

    # Primary: use the most-converged ckpt at 1x and 0.5x (mirrors cam-adv s500 ep60 pattern)
    if most_recent is not None:
        h_qf, h_gf = cyclegan_feats[most_recent]
        write_csv(base_qf + h_qf, base_gf + h_gf,
                  f'cyclegan_baseline8_plus_{most_recent}_1x', num_query, val_loader)
        write_csv(base_qf + 0.5 * h_qf, base_gf + 0.5 * h_gf,
                  f'cyclegan_baseline8_plus_{most_recent}_0p5x', num_query, val_loader)
        write_csv(h_qf, h_gf,
                  f'cyclegan_solo_{most_recent}', num_query, val_loader)

    # Try ep30 add (less converged, lower distribution shift, may be ensemble-safer)
    if 'ep30' in cyclegan_feats:
        h_qf, h_gf = cyclegan_feats['ep30']
        write_csv(base_qf + h_qf, base_gf + h_gf,
                  'cyclegan_baseline8_plus_ep30_1x', num_query, val_loader)
        write_csv(base_qf + 0.5 * h_qf, base_gf + 0.5 * h_gf,
                  'cyclegan_baseline8_plus_ep30_0p5x', num_query, val_loader)

    # Combined: most_recent + ep30 at 0.5x each
    if most_recent != 'ep30' and 'ep30' in cyclegan_feats and most_recent is not None:
        h_qf = cyclegan_feats['ep30'][0] + cyclegan_feats[most_recent][0]
        h_gf = cyclegan_feats['ep30'][1] + cyclegan_feats[most_recent][1]
        write_csv(base_qf + 0.5 * h_qf, base_gf + 0.5 * h_gf,
                  f'cyclegan_baseline8_plus_ep30_{most_recent}_0p5x', num_query, val_loader)


if __name__ == '__main__':
    main()
