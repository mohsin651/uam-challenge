"""Multi-scale + multi-checkpoint + multi-architecture ensemble inference.

For each (architecture group, scale, checkpoint) triple, extracts
L2-normalized final-layer CLS features. All features are summed (with optional
per-group weight) then L2-renormalized. Re-ranking + class-group filter on the
combined distance matrix, then standard Kaggle-CSV output.

Why multi-scale: inferring the SAME trained ViT-L checkpoint at 256x128 AND
288x144 produces complementary features (different receptive fields), with no
retraining cost. Known +0.3-1% gain in ReID literature.
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
from utils.re_ranking import re_ranking


# Each group: one or more checkpoints at one or more inference scales.
# Scales are [H, W] pairs; patches must be divisible by patch_size per model.
GROUPS = [
    {
        'name': 'vitl16_supervised',
        'config_file': '/workspace/miuam_challenge_diff/config/UrbanElementsReID_test.yml',
        'scales': [[256, 128]],   # multi-scale requires pos_embed resize support in outer load_param — TODO
        'weight': 1.0,
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
        'name': 'eva_p14',
        'config_file': '/workspace/miuam_challenge_diff/config/UrbanElementsReID_test_eva.yml',
        'scales': [[224, 112]],   # EVA trained at 224×112; stick with the trained size
        'weight': 1.0,
        'checkpoints': [
            '/workspace/miuam_challenge_diff/models/model_eva_large_p14_seed77/part_attention_vit_30.pth',
            '/workspace/miuam_challenge_diff/models/model_eva_large_p14_seed77/part_attention_vit_40.pth',
            '/workspace/miuam_challenge_diff/models/model_eva_large_p14_seed77/part_attention_vit_50.pth',
            '/workspace/miuam_challenge_diff/models/model_eva_large_p14_seed77/part_attention_vit_60.pth',
        ],
    },
]


def extract_feature(model, dataloader, num_query):
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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_group(group):
    """For each scale in the group, build dataloader once, then iterate ckpts."""
    from config.defaults import _C as _BASE_CFG
    base_cfg = _BASE_CFG.clone()
    base_cfg.merge_from_file(group['config_file'])

    os.environ['CUDA_VISIBLE_DEVICES'] = base_cfg.MODEL.DEVICE_ID

    group_feats = []  # list of (qf, gf) across (ckpt × scale)
    img_items = None
    num_query = None

    for scale in group['scales']:
        # Fresh cfg per scale because we mutate SIZE_TEST.
        local_cfg = base_cfg.clone()
        local_cfg.defrost()
        local_cfg.INPUT.SIZE_TEST = list(scale)
        local_cfg.INPUT.SIZE_TRAIN = list(scale)   # also affects dataloader's resize
        local_cfg.freeze()

        val_loader, nq = build_reid_test_loader(local_cfg, local_cfg.DATASETS.TEST[0])
        if img_items is None:
            img_items = val_loader.dataset.img_items
            num_query = nq

        for ckpt in group['checkpoints']:
            print(f"  [{group['name']}] scale={scale}  {os.path.basename(ckpt)}")
            model = make_model(local_cfg, local_cfg.MODEL.NAME, 0, 0, 0)
            model.load_param(ckpt)
            model = model.cuda().eval()
            qf, gf = extract_feature(model, val_loader, num_query)
            group_feats.append((qf, gf))
            del model
            torch.cuda.empty_cache()

    return group_feats, img_items, num_query, group.get('weight', 1.0)


def main():
    set_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="./results/ensemble_multiscale", type=str)
    args = parser.parse_args()

    cfg.merge_from_file(GROUPS[0]['config_file'])
    cfg.freeze()

    all_qf = []
    all_gf = []
    img_items = None
    num_query = None
    total = 0

    for group in GROUPS:
        feats, grp_img_items, grp_num_query, w = run_group(group)
        if img_items is None:
            img_items = grp_img_items
            num_query = grp_num_query
        for qf, gf in feats:
            all_qf.append(qf * w)
            all_gf.append(gf * w)
            total += 1

    print(f"\nTotal feature extractions (ckpt × scale × group): {total}")

    qf_sum = torch.stack(all_qf, dim=0).sum(dim=0)
    gf_sum = torch.stack(all_gf, dim=0).sum(dim=0)
    qf = F.normalize(qf_sum, p=2, dim=1).numpy()
    gf = F.normalize(gf_sum, p=2, dim=1).numpy()

    q_g = np.dot(qf, gf.T)
    q_q = np.dot(qf, qf.T)
    g_g = np.dot(gf, gf.T)
    re_rank_dist = re_ranking(q_g, q_q, g_g)

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
    print(f"class-group filter: masked {cross_group_mask.sum()} / {cross_group_mask.size} cells")
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
