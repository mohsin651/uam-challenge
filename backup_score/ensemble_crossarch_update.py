"""Cross-architecture ensemble: mixes checkpoints from different model configs.

Iterates over GROUPS, each with its own test-config YAML and a list of
checkpoints that belong to that architecture. For each group, rebuilds the
dataloader (input size may differ per arch) and the model. Extracts
L2-normalized final-layer CLS features per checkpoint, accumulates across
all groups/checkpoints, then re-normalizes, re-ranks, applies class-group
filter, writes the Kaggle submission CSV.

Current groups:
  - ViT-L/16 supervised @ 256x128 (the 7-ckpt base that gives 0.13361)
  - ViT-L/14 DINOv2      @ 224x112 (the new DINOv2-pretrained run)
"""
import argparse
import copy
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from yacs.config import CfgNode

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.re_ranking import re_ranking


GROUPS = [
    {
        'name': 'vitl16_supervised',
        'config_file': '/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml',
        'checkpoints': [
            '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth',
        ],
    },
    {
        'name': 'dinov2_p14',
        'config_file': '/workspace/miuam_challenge_diff/config/UrbanElementsReID_test_dinov2.yml',
        # ep40/50/60 have NaN in cls_token/part_tokens/pos_embed — AMP loss-scaler
        # didn't catch a numerical blowup mid-training. Only pre-blowup ckpts here.
        'checkpoints': [
            '/workspace/miuam_challenge_diff/models/model_vitlarge_dinov2_p14_seed42/part_attention_vit_10.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_dinov2_p14_seed42/part_attention_vit_20.pth',
            '/workspace/miuam_challenge_diff/models/model_vitlarge_dinov2_p14_seed42/part_attention_vit_30.pth',
        ],
    },
]


def extract_feature(model, dataloader, num_query):
    """Per-checkpoint features (L2-normalized final-layer CLS), on CPU."""
    features = []
    with torch.no_grad():
        for data in dataloader:
            img = data['images'].cuda()
            ff = model(img).float()
            ff = F.normalize(ff, p=2, dim=1)
            features.append(ff)
    features = torch.cat(features, 0).cpu()
    return features[:num_query], features[num_query:]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_group(group):
    """Build dataloader + model(s) for one arch group, return list of (qf, gf) per ckpt."""
    # Load a fresh cfg from this group's config file so each group can have
    # different TRANSFORMER_TYPE, INPUT.SIZE_TEST, STRIDE_SIZE, etc.
    # We can't mutate the global cfg more than once (frozen after first merge),
    # so we build a local CfgNode from yacs.
    from config.defaults import _C as _BASE_CFG
    local_cfg = _BASE_CFG.clone()
    local_cfg.merge_from_file(group['config_file'])
    local_cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = local_cfg.MODEL.DEVICE_ID

    val_loader, num_query = build_reid_test_loader(local_cfg, local_cfg.DATASETS.TEST[0])

    group_feats = []
    for ckpt_path in group['checkpoints']:
        print(f"\n[{group['name']}] loading {os.path.basename(ckpt_path)}")
        model = make_model(local_cfg, local_cfg.MODEL.NAME, 0, 0, 0)
        model.load_param(ckpt_path)
        model = model.cuda().eval()
        qf, gf = extract_feature(model, val_loader, num_query)
        group_feats.append((qf, gf))
        del model
        torch.cuda.empty_cache()

    # val_loader's dataset.img_items is needed for class-group filter alignment.
    # Since it's the same query/gallery across all groups (only image resolution
    # differs per arch, but identities are the same), we can use any group's
    # val_loader.dataset.img_items. Return it along with num_query.
    return group_feats, val_loader.dataset.img_items, num_query


def main():
    set_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="./results/ensemble_crossarch", type=str)
    args = parser.parse_args()

    # Need at least one group config to bootstrap the global cfg (for DATASETS.ROOT_DIR, etc.)
    cfg.merge_from_file(GROUPS[0]['config_file'])
    cfg.freeze()

    all_qf = []
    all_gf = []
    img_items = None
    num_query = None
    total_ckpts = 0

    for group in GROUPS:
        group_feats, group_img_items, group_num_query = run_group(group)
        if img_items is None:
            img_items = group_img_items
            num_query = group_num_query
        for qf, gf in group_feats:
            all_qf.append(qf)
            all_gf.append(gf)
            total_ckpts += 1

    print(f"\n=== Total: {total_ckpts} checkpoints across {len(GROUPS)} arch groups ===")

    # Sum per-checkpoint L2-normalized features, then renormalize (standard ensemble).
    qf_sum = torch.stack(all_qf, dim=0).sum(dim=0)
    gf_sum = torch.stack(all_gf, dim=0).sum(dim=0)
    qf = F.normalize(qf_sum, p=2, dim=1).numpy()
    gf = F.normalize(gf_sum, p=2, dim=1).numpy()
    print(f"ensembled feats shapes: qf {qf.shape}, gf {gf.shape}")

    q_g_dist = np.dot(qf, gf.T)
    q_q_dist = np.dot(qf, qf.T)
    g_g_dist = np.dot(gf, gf.T)
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    # Class-group filter (same as update.py / ensemble_update.py)
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
    q_items = [it for it in img_items if it[3]['q_or_g'] == 'query']
    g_items = [it for it in img_items if it[3]['q_or_g'] == 'gallery']
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
