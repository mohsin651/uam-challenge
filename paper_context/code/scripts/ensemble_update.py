"""Ensemble inference: average features from multiple checkpoints.

Same feature-extraction logic as update.py (final-layer CLS, L2-normed, 2-pass),
but run once per checkpoint in CHECKPOINT_LIST. Features from each checkpoint
are L2-normalized, then averaged, then re-normalized — standard ensemble.
Re-ranking and class-group filter applied on the averaged features.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking


# Edit this list to change which checkpoints participate.
# 11-checkpoint ensemble: 7 from the 0.13361 base (seed1234 + seed42) +
# 4 from the new seed100+EMA run.
CHECKPOINT_LIST = [
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed100_ema/part_attention_vit_30.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed100_ema/part_attention_vit_40.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed100_ema/part_attention_vit_50.pth',
    '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed100_ema/part_attention_vit_60.pth',
]


def extract_feature(model, dataloaders, num_query):
    """Same as update.py: final-layer CLS, 2-pass sum (no-op but kept for fidelity),
    L2-normed. Returns (qf, gf) as torch tensors on CPU."""
    features = []
    for data in dataloaders:
        img, _, _, _, _ = data.values()
        ff = None
        for _ in range(2):
            outputs = model(img.cuda())
            f = outputs.float()
            ff = f if ff is None else ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)
    features = torch.cat(features, 0)
    return features[:num_query].cpu(), features[num_query:].cpu()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./config/UrbanElementsReID_test.yml", type=str)
    parser.add_argument("--track", default="./results/ensemble", type=str)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)
    logger.info("Ensemble checkpoints: %s", CHECKPOINT_LIST)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    assert cfg.MODEL.NAME == 'part_attention_vit'

    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    # Accumulate features across checkpoints. Each checkpoint's features are
    # already L2-normed inside extract_feature; we sum and renormalize at the end.
    qf_sum = None
    gf_sum = None
    for i, ckpt in enumerate(CHECKPOINT_LIST):
        print(f"\n=== Ensemble member {i+1}/{len(CHECKPOINT_LIST)}: {os.path.basename(ckpt)} ===")
        model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt)
        model = model.cuda().eval()
        with torch.no_grad():
            qf, gf = extract_feature(model, val_loader, num_query)
        if qf_sum is None:
            qf_sum = qf.clone()
            gf_sum = gf.clone()
        else:
            qf_sum += qf
            gf_sum += gf
        # free GPU memory before loading the next checkpoint
        del model
        torch.cuda.empty_cache()

    qf = F.normalize(qf_sum, p=2, dim=1).numpy()
    gf = F.normalize(gf_sum, p=2, dim=1).numpy()
    print(f"\nEnsembled {len(CHECKPOINT_LIST)} checkpoints; qf shape {qf.shape}, gf shape {gf.shape}")

    q_g_dist = np.dot(qf, gf.T)
    q_q_dist = np.dot(qf, qf.T)
    g_g_dist = np.dot(gf, gf.T)
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    # Class-group filter (same as update.py)
    import pandas as pd
    CLASS_GROUP = {
        'trafficsignal': 'trafficsignal',
        'crosswalk':     'crosswalk',
        'container':     'bin_like',
        'rubbishbins':   'bin_like',
    }
    q_cls_df = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'query_classes.csv'))
    g_cls_df = pd.read_csv(os.path.join(cfg.DATASETS.ROOT_DIR, 'test_classes.csv'))
    q_name_to_grp = {n: CLASS_GROUP[c.lower()] for n, c in zip(q_cls_df['imageName'], q_cls_df['Class'])}
    g_name_to_grp = {n: CLASS_GROUP[c.lower()] for n, c in zip(g_cls_df['imageName'], g_cls_df['Class'])}
    q_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in val_loader.dataset.img_items if it[3]['q_or_g'] == 'gallery']
    q_groups = np.array([q_name_to_grp[os.path.basename(it[0])] for it in q_items])
    g_groups = np.array([g_name_to_grp[os.path.basename(it[0])] for it in g_items])
    cross_group_mask = q_groups[:, None] != g_groups[None, :]
    print(f"class-group filter: masked {cross_group_mask.sum()} / {cross_group_mask.size} cells "
          f"({100*cross_group_mask.mean():.1f}%)")
    re_rank_dist[cross_group_mask] = np.inf

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]
    m, _ = indices.shape

    output_path = args.track + "_submission.csv"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    lista_nombres = ["{:06d}.jpg".format(i) for i in range(1, len(indices) + 1)]
    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for name, track in zip(lista_nombres, indices):
            w.writerow([name, ' '.join(map(str, track + 1))])

    print(f"\nSubmission saved to: {output_path}")
    print(f"Total queries written: {m}")


if __name__ == "__main__":
    main()
