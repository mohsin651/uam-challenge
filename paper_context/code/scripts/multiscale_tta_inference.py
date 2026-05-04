"""Multi-scale TTA inference on the 0.15884 8-ckpt ensemble.

For each ckpt, extract features at 3 input sizes (224x112, 256x128, 288x144),
L2-normalize, and average per-ckpt. The averaged features then go through the
usual ensemble + DBA + rerank + class-group filter pipeline.

Implementation notes:
- The PAT model has a hardcoded input-size assertion (PatchEmbed_overlap),
  so we rebuild the model per scale.
- The saved checkpoints' pos_embed is sized for 16x8 (256x128 input). We
  interpolate it bilinearly to match each scale's patch grid:
    224x112 -> 14x7,   256x128 -> 16x8,   288x144 -> 18x9.
- patch_embed.proj is conv kernel — shape-invariant to img_size, no resize needed.
- All other backbone params copy directly.
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

ALL_CKPTS = [
    f'{BASELINE_DIR_S1234}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S1234}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_30.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_40.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_50.pth',
    f'{BASELINE_DIR_S42}/part_attention_vit_60.pth',
    CAMADV_EP60,
]

# Three scales: native, zoom-out (smaller patches), zoom-in (more patches)
SCALES = [(224, 112), (256, 128), (288, 144)]
GRID = {  # (h_grid, w_grid) per scale
    (224, 112): (14, 7),
    (256, 128): (16, 8),
    (288, 144): (18, 9),
}
SAVED_GRID = (16, 8)  # checkpoints were trained at 256x128

DBA_K, RR_K1, RR_K2, RR_LAMBDA = 8, 15, 4, 0.275
OUT_DIR = '/workspace/miuam_challenge_diff/results'


def interpolate_pos_embed(saved_pe, new_h, new_w, old_h=16, old_w=8, n_extra=4):
    """saved_pe: (1, n_extra + old_h*old_w, dim). Returns (1, n_extra + new_h*new_w, dim).

    Bilinearly resizes the spatial grid; n_extra leading tokens (CLS + 3 part) are kept as-is.
    """
    extra = saved_pe[:, :n_extra]
    grid = saved_pe[:, n_extra:].reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_h, new_w), mode='bilinear', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
    return torch.cat([extra, grid], dim=1)


def load_param_with_pos_embed_resize(model, ckpt_path, target_h, target_w):
    """Load ckpt weights, interpolating pos_embed if its grid differs from target."""
    state = torch.load(ckpt_path, map_location='cpu')
    msd = model.state_dict()
    for k, v in state.items():
        k = k.replace('module.', '')
        if 'classifier' in k:
            continue
        if k not in msd:
            continue
        if msd[k].shape == v.shape:
            msd[k].copy_(v)
        elif 'pos_embed' in k:
            v_resized = interpolate_pos_embed(v, target_h, target_w,
                                              old_h=SAVED_GRID[0], old_w=SAVED_GRID[1])
            assert v_resized.shape == msd[k].shape, \
                f"pos_embed resize shape mismatch: got {v_resized.shape}, expected {msd[k].shape}"
            msd[k].copy_(v_resized)
        else:
            print(f"  shape mismatch on {k}: {v.shape} vs {msd[k].shape} — SKIPPED")


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


def write_csv(qf_t, gf_t, label, num_query, val_loader_for_groups,
              dba_k=DBA_K, k1=RR_K1, k2=RR_K2, lam=RR_LAMBDA):
    qf = F.normalize(qf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = F.normalize(gf_t, p=2, dim=1).numpy().astype(np.float32)
    gf = db_augment(gf, dba_k)
    q_g = np.dot(qf, gf.T); q_q = np.dot(qf, qf.T); g_g = np.dot(gf, gf.T)
    rrd = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lam)

    CG = {'trafficsignal':'trafficsignal','crosswalk':'crosswalk','container':'bin_like','rubbishbins':'bin_like'}
    qcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    gcls = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q2g = {n: CG[c.lower()] for n, c in zip(qcls['imageName'], qcls['Class'])}
    g2g = {n: CG[c.lower()] for n, c in zip(gcls['imageName'], gcls['Class'])}
    q_items = [it for it in val_loader_for_groups.dataset.img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in val_loader_for_groups.dataset.img_items if it[3]['q_or_g'] == 'gallery']
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

    # Per-ckpt accumulator: {ckpt_path: {'qf': ..., 'gf': ...}} averaged across scales.
    # We'll also track: per-scale ensemble (sum across all ckpts) for diagnosis.
    per_scale_qf = {}
    per_scale_gf = {}

    # Cache a "groups" loader at default size for class-group lookup at the end
    cfg.defrost(); cfg.INPUT.SIZE_TEST = [256, 128]; cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.freeze()
    val_loader_default, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    print(f"num_query: {num_query}, gallery: {len(val_loader_default.dataset.img_items) - num_query}")

    # Loop scales (outer) so we only rebuild loader 3 times total
    for h, w in SCALES:
        gh, gw = GRID[(h, w)]
        print(f"\n========= scale {h}x{w} (grid {gh}x{gw}) =========")
        cfg.defrost(); cfg.INPUT.SIZE_TEST = [h, w]; cfg.INPUT.SIZE_TRAIN = [h, w]; cfg.freeze()
        loader, num_q = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
        assert num_q == num_query

        scale_qf_sum = scale_gf_sum = None
        for ckpt in ALL_CKPTS:
            print(f"  ckpt: {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
            model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
            load_param_with_pos_embed_resize(model, ckpt, gh, gw)
            model = model.cuda().eval()
            qf, gf = extract(model, loader, num_q)
            del model; torch.cuda.empty_cache()
            scale_qf_sum = qf if scale_qf_sum is None else scale_qf_sum + qf
            scale_gf_sum = gf if scale_gf_sum is None else scale_gf_sum + gf
        per_scale_qf[(h, w)] = scale_qf_sum
        per_scale_gf[(h, w)] = scale_gf_sum

    os.makedirs(OUT_DIR, exist_ok=True)

    # Variant 1 — multi-scale TTA: average across all 3 scales (each scale's full ensemble), then DBA+rerank.
    qf_msr = sum(per_scale_qf[s] for s in SCALES) / len(SCALES)
    gf_msr = sum(per_scale_gf[s] for s in SCALES) / len(SCALES)
    write_csv(qf_msr, gf_msr, 'multiscale_3sizes_dba8_k15_lam0275',
              num_query, val_loader_default)

    # Variant 2 — sanity: just the native 256x128 scale (should reproduce ~0.15884)
    write_csv(per_scale_qf[(256, 128)], per_scale_gf[(256, 128)],
              'multiscale_native256_dba8_k15_lam0275',
              num_query, val_loader_default)

    # Variant 3 — zoom-out only (224x112)
    write_csv(per_scale_qf[(224, 112)], per_scale_gf[(224, 112)],
              'multiscale_only224_dba8_k15_lam0275',
              num_query, val_loader_default)

    # Variant 4 — zoom-in only (288x144)
    write_csv(per_scale_qf[(288, 144)], per_scale_gf[(288, 144)],
              'multiscale_only288_dba8_k15_lam0275',
              num_query, val_loader_default)

    # Variant 5 — averaged 256+288 only (drop 224 if it's noisy)
    qf_2 = (per_scale_qf[(256, 128)] + per_scale_qf[(288, 144)]) / 2
    gf_2 = (per_scale_gf[(256, 128)] + per_scale_gf[(288, 144)]) / 2
    write_csv(qf_2, gf_2, 'multiscale_256_288_dba8_k15_lam0275',
              num_query, val_loader_default)


if __name__ == '__main__':
    main()
