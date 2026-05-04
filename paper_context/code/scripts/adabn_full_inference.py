"""AdaBN applied to ALL 8 ckpts — broader test-time adaptation.

The cam-adv-only AdaBN (§prior) gave 0.15871 (within noise of 0.15884).
Hypothesis: applying AdaBN to ALL ckpts collectively realigns the entire
ensemble's BN stats to the query+gallery distribution, potentially yielding
a real improvement.

Generates 2 variants:
  PRIMARY: AdaBN on all 8 ckpts, full reset (most aggressive adaptation)
  BACKUP: AdaBN on all 8 with momentum=0.1 (gentle blend with training stats)
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
ALL_8 = [
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


def extract(model, loader, num_query):
    feats = []
    with torch.no_grad():
        for data in loader:
            ff = model(data['images'].cuda()).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    feats = torch.cat(feats, 0)
    return feats[:num_query], feats[num_query:]


def adabn_recalibrate(model, loader, momentum=None):
    """Recompute BN running stats on loader's data.
    momentum=None: cumulative average (full reset). momentum=0.1 etc: EMA blend."""
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            if momentum is None:
                module.reset_running_stats()
                module.momentum = None  # cumulative
            else:
                module.momentum = momentum  # EMA with old stats kept

    model.train()
    with torch.no_grad():
        for data in loader:
            _ = model(data['images'].cuda())
    model.eval()


def db_augment(gf, k):
    if k <= 0: return gf
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
    out = f'/workspace/miuam_challenge_diff/results/{label}_submission.csv'
    names = [f'{i:06d}.jpg' for i in range(1, len(indices) + 1)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n, t in zip(names, indices):
            w.writerow([n, ' '.join(map(str, t + 1))])
    print(f"  → {out}")


def extract_all_with_adabn(ckpts, loader, num_query, momentum=None):
    """For each ckpt, apply AdaBN then extract features. Returns summed features."""
    qf_sum = gf_sum = None
    for ckpt in ckpts:
        print(f"  AdaBN+extract: {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda()
        adabn_recalibrate(model, loader, momentum=momentum)
        qf, gf = extract(model, loader, num_query)
        qf_sum = qf if qf_sum is None else qf_sum + qf
        gf_sum = gf if gf_sum is None else gf_sum + gf
        del model; torch.cuda.empty_cache()
    return qf_sum, gf_sum


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    print("\n=== Variant A: AdaBN on ALL 8 ckpts (full reset) ===")
    qf_a, gf_a = extract_all_with_adabn(ALL_8, val_loader, num_query, momentum=None)
    write_csv(qf_a, gf_a, 'adabn_all8_full', num_query, val_loader)

    print("\n=== Variant B: AdaBN on ALL 8 ckpts (gentle, momentum=0.1) ===")
    qf_b, gf_b = extract_all_with_adabn(ALL_8, val_loader, num_query, momentum=0.1)
    write_csv(qf_b, gf_b, 'adabn_all8_momentum01', num_query, val_loader)


if __name__ == '__main__':
    main()
