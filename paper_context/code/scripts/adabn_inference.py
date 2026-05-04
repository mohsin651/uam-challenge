"""AdaBN inference — recompute BatchNorm running stats on test (query+gallery)
distribution, then re-extract features. Targets the c004 distribution shift
directly via test-time normalization adaptation.

Apply ONLY to cam-adv s500 ep60 (the +0.005 lever). Keep proven 7-baseline
ckpts unchanged so the proven feature manifold isn't disturbed.

Generates 2 variants:
  PRIMARY: 7-baseline (unchanged) + AdaBN-cam-adv ep60 → 8 ckpts
  BACKUP: AdaBN-cam-adv ep60 SOLO (diagnostic — does AdaBN help features at all?)
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


def extract(model, loader, num_query):
    feats = []
    with torch.no_grad():
        for data in loader:
            ff = model(data['images'].cuda()).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    feats = torch.cat(feats, 0)
    return feats[:num_query], feats[num_query:]


def adabn_recalibrate(model, loader):
    """Recompute BatchNorm running stats on the loader's data.
    Sets model.train() with gradients OFF — BN updates running mean/var, no
    other params change."""
    # Freeze ALL parameters
    for p in model.parameters():
        p.requires_grad = False

    # Reset BN running stats to start fresh on new distribution
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.reset_running_stats()
            module.momentum = None  # use cumulative average instead of EMA

    # Forward in train mode → BN updates running stats with new data
    model.train()
    with torch.no_grad():
        for data in loader:
            _ = model(data['images'].cuda())

    # Back to eval mode — BN now uses the adapted stats
    model.eval()


def extract_with_adabn(model, loader, num_query):
    """Recalibrate BN, then extract features."""
    print("  Recalibrating BN running stats on query+gallery...")
    adabn_recalibrate(model, loader)
    return extract(model, loader, num_query)


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


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    # === 7-baseline (UNMODIFIED — preserve proven feature manifold) ===
    print("\n[1/2] Extracting 7-baseline (no AdaBN, proven features):")
    base_qf = base_gf = None
    for ckpt in BASELINE_7:
        print(f"  {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        base_qf = qf if base_qf is None else base_qf + qf
        base_gf = gf if base_gf is None else base_gf + gf
        del model; torch.cuda.empty_cache()

    # === Cam-adv s500 ep60 WITH AdaBN ===
    print("\n[2/2] Cam-adv s500 ep60 WITH AdaBN (BN stats adapted to test):")
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(CAMADV_S500_EP60)
    model = model.cuda()
    cam_qf, cam_gf = extract_with_adabn(model, val_loader, num_query)
    del model; torch.cuda.empty_cache()

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)
    # PRIMARY: 7-baseline + AdaBN-cam-adv (mirrors 0.15884 recipe with AdaBN twist)
    write_csv(base_qf + cam_qf, base_gf + cam_gf,
              'adabn_baseline7_plus_camadv_ep60', num_query, val_loader)
    # BACKUP: AdaBN-cam-adv solo (diagnostic)
    write_csv(cam_qf, cam_gf,
              'adabn_camadv_solo_ep60', num_query, val_loader)


if __name__ == '__main__':
    main()
