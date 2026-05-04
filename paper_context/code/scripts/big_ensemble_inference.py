"""Big ensemble inference: 7-baseline + cam-adv s500 ep60 + new seeds 2100/2200/2300.

Per the user's intel that #2/#3 teams use PAT, the most plausible difference is
they have MORE baseline seeds. We add 3 new seeds (2100/2200/2300, 2 ckpts each
= ep40 + ep60, matching the proven pattern's late-epoch usage).

Final ensemble: 7 (proven) + 6 (new) + 1 (cam-adv) = 14 ckpts.
Per §72.5 angular math: cam-adv at 1/(1+√13) ≈ 22% — safely under 27% threshold.

Generates 2 high-EV variants:
  PRIMARY: 13 baseline + 1 cam-adv (mirrors 0.15884 recipe scaled up)
  BACKUP: 13 baseline only (no cam-adv) — diagnostic for whether cam-adv still helps
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
DIRS = {
    's1234': '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep',
    's42':   '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42',
    's2100': '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed2100',
    's2200': '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed2200',
    's2300': '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed2300',
}

# 13 BASELINE ckpts: proven 7 + 2 ckpts (ep40, ep60) per new seed
BASELINE_13 = [
    f"{DIRS['s1234']}/part_attention_vit_30.pth",
    f"{DIRS['s1234']}/part_attention_vit_40.pth",
    f"{DIRS['s1234']}/part_attention_vit_50.pth",
    f"{DIRS['s42']}/part_attention_vit_30.pth",
    f"{DIRS['s42']}/part_attention_vit_40.pth",
    f"{DIRS['s42']}/part_attention_vit_50.pth",
    f"{DIRS['s42']}/part_attention_vit_60.pth",
    f"{DIRS['s2100']}/part_attention_vit_40.pth",
    f"{DIRS['s2100']}/part_attention_vit_60.pth",
    f"{DIRS['s2200']}/part_attention_vit_40.pth",
    f"{DIRS['s2200']}/part_attention_vit_60.pth",
    f"{DIRS['s2300']}/part_attention_vit_40.pth",
    f"{DIRS['s2300']}/part_attention_vit_60.pth",
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
    print(f"  num_query: {num_query}")

    print(f"\n  Extracting {len(BASELINE_13)} baseline ckpts:")
    base_qf = base_gf = None
    for ckpt in BASELINE_13:
        print(f"    {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        qf, gf = extract(model, val_loader, num_query)
        base_qf = qf if base_qf is None else base_qf + qf
        base_gf = gf if base_gf is None else base_gf + gf
        del model; torch.cuda.empty_cache()

    print(f"\n  Extracting cam-adv s500 ep60:")
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(CAMADV_S500_EP60)
    model = model.cuda().eval()
    cam_qf, cam_gf = extract(model, val_loader, num_query)
    del model; torch.cuda.empty_cache()

    os.makedirs('/workspace/miuam_challenge_diff/results', exist_ok=True)
    # PRIMARY: 13 baseline + 1 cam-adv at 1× weight (mirrors proven 0.15884 recipe)
    write_csv(base_qf + cam_qf, base_gf + cam_gf,
              'big13_plus_camadv_ep60', num_query, val_loader)
    # BACKUP: 13 baseline only (no cam-adv) — does cam-adv still help with bigger baseline?
    write_csv(base_qf, base_gf,
              'big13_baseline_only', num_query, val_loader)


if __name__ == '__main__':
    main()
