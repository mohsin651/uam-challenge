"""Histogram-match c004 queries to the gallery (c001-c003) distribution, then
re-extract features and submit.

The 928 queries are ALL from c004 (sole query camera). The gallery is from
c001/c002/c003. There's a known camera-domain gap in color/exposure. We've
attacked this in *training* (cam-adv). Now attacking at *inference*: pre-
process the query images so their pixel statistics match the gallery dist.

Three variants:
  A: per-channel histogram match (skimage match_histograms)
  B: simpler — channel mean/std match
  C: per-image LAB L-channel match (illuminance only, preserves color)
"""
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.exposure import match_histograms
from torchvision import transforms

from config import cfg
from model import make_model
from utils.re_ranking import re_ranking


URBAN_ROOT = '/workspace/Urban2026/'
PERCKPT_CACHE = '/workspace/miuam_challenge_diff/results/cache/perckpt_test.npz'
HM_CACHE_DIR = '/workspace/miuam_challenge_diff/results/cache/histmatch_queries'

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
DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275


def compute_gallery_stats(gallery_dir, n_samples=500):
    """Sample N gallery images, build a representative reference image from
    their pooled pixel distribution."""
    print(f"  sampling {n_samples} gallery images for reference dist...")
    files = sorted(os.listdir(gallery_dir))[:n_samples]
    imgs = []
    for f in files:
        img = np.array(Image.open(os.path.join(gallery_dir, f)).convert('RGB').resize((128, 256)))
        imgs.append(img)
    pooled = np.concatenate([im.reshape(-1, 3) for im in imgs], axis=0)  # (N*256*128, 3)
    pooled_2d = pooled.reshape(-1, 1, 3)  # treat as 1-pixel-wide image for match_histograms
    return pooled, np.array(imgs)  # return both for variants


def histmatch_per_channel(query_img, ref_imgs):
    """Match each channel's histogram independently to a representative ref."""
    ref = ref_imgs.mean(axis=0).astype(np.uint8)  # mean image as reference
    matched = match_histograms(query_img, ref, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)


def channel_mean_std_match(query_img, ref_pooled):
    """Z-score per channel, then re-shift to gallery channel mean/std."""
    q = query_img.astype(np.float32)
    ref_mean = ref_pooled.mean(axis=0)  # (3,)
    ref_std = ref_pooled.std(axis=0)
    q_mean = q.reshape(-1, 3).mean(axis=0)
    q_std = q.reshape(-1, 3).std(axis=0)
    out = (q - q_mean) / (q_std + 1e-8) * ref_std + ref_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def lab_l_match(query_img, ref_imgs):
    """Match only the L channel in LAB space — preserves color, normalizes exposure."""
    from skimage import color
    ref = ref_imgs.mean(axis=0).astype(np.uint8)
    q_lab = color.rgb2lab(query_img / 255.0)
    r_lab = color.rgb2lab(ref / 255.0)
    q_lab[..., 0] = match_histograms(q_lab[..., 0], r_lab[..., 0])
    out = color.lab2rgb(q_lab) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_queries(variant, query_dir, n_query, gallery_dir):
    """Apply variant transform to all query images, save as a cached batch tensor."""
    out_path = f'{HM_CACHE_DIR}/queries_{variant}.npy'
    if os.path.exists(out_path):
        print(f"  cache hit: {out_path}")
        return out_path

    os.makedirs(HM_CACHE_DIR, exist_ok=True)
    ref_pooled, ref_imgs = compute_gallery_stats(gallery_dir)

    print(f"  applying variant '{variant}' to {n_query} queries...")
    out = np.zeros((n_query, 256, 128, 3), dtype=np.uint8)
    for i in range(1, n_query + 1):
        img = np.array(Image.open(os.path.join(query_dir, f'{i:06d}.jpg')).convert('RGB').resize((128, 256)))
        if variant == 'perchan':
            img2 = histmatch_per_channel(img, ref_imgs)
        elif variant == 'meanstd':
            img2 = channel_mean_std_match(img, ref_pooled)
        elif variant == 'labL':
            img2 = lab_l_match(img, ref_imgs)
        else:
            raise ValueError(variant)
        out[i-1] = img2

    np.save(out_path, out)
    print(f"  saved {out_path}")
    return out_path


def extract_for_queries(model, query_imgs):
    """Given (N, H, W, 3) uint8 array, run through model. Use PAT (0.5,0.5,0.5) norm."""
    BATCH = 64
    PAT_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    PAT_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
    feats = []
    n = len(query_imgs)
    with torch.no_grad():
        for s in range(0, n, BATCH):
            batch = query_imgs[s:s+BATCH]
            t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().cuda() / 255.0
            t = (t - PAT_MEAN) / PAT_STD
            ff = model(t).float()
            ff = F.normalize(ff, p=2, dim=1)
            feats.append(ff.cpu())
    return torch.cat(feats, 0)


def db_augment(gf, k):
    sim = gf @ gf.T
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    gf_dba = gf[topk].mean(axis=1)
    gf_dba = gf_dba / (np.linalg.norm(gf_dba, axis=1, keepdims=True) + 1e-8)
    return gf_dba.astype(np.float32)


def write_csv(qf, gf, label, num_query):
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    gf = db_augment(gf, DBA_K)
    q_g = qf @ gf.T; q_q = qf @ qf.T; g_g = gf @ gf.T
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
    print(f"  → {out}")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.merge_from_file('/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml')
    cfg.freeze()

    n_query = 928
    query_dir = '/workspace/Urban2026/image_query'
    gallery_dir = '/workspace/Urban2026/image_test'

    # Step 1: pre-process queries with each variant
    variants = ['perchan', 'meanstd', 'labL']
    paths = {}
    for v in variants:
        print(f"\n[preproc] variant={v}")
        paths[v] = preprocess_queries(v, query_dir, n_query, gallery_dir)

    # Step 2: load existing per-ckpt cache for gallery features (unchanged)
    d = np.load(PERCKPT_CACHE)
    gfs = d['gfs']  # (8, n_g, 1024)
    gf_summed = gfs.sum(axis=0)  # original gallery features (PAT-8 sum)

    # Step 3: re-extract query features with each variant via 8-ckpt ensemble
    for v in variants:
        print(f"\n[extract] variant={v}")
        query_imgs = np.load(paths[v])
        qf_sum = None
        for ckpt in PAT_8:
            print(f"  {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
            model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
            model.load_param(ckpt)
            model = model.cuda().eval()
            qf = extract_for_queries(model, query_imgs).numpy().astype(np.float32)
            qf_sum = qf if qf_sum is None else qf_sum + qf
            del model; torch.cuda.empty_cache()

        write_csv(qf_sum, gf_summed.copy(), f'histmatch_{v}', n_query)


if __name__ == '__main__':
    main()
