"""Inference entrypoint for Urban Elements ReID.

Loads a trained part_attention_vit checkpoint, extracts features for
query + gallery, applies re-ranking, and writes the Kaggle-format CSV.
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
from processor.part_attention_vit_processor import do_inference as do_inf_pat
from utils.logger import setup_logger
from utils.re_ranking import re_ranking


def extract_feature(model, dataloaders, num_query):
    """Extract L2-normalized final-layer CLS features (1024-d).

    This is the 0.12927 configuration. All feature-engineering variants tried
    this session (multi-layer concat, part tokens, h-flip TTA, deep-sup,
    query expansion) regressed on Kaggle. Sticking with this exact setup.
    """
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
    return features[:num_query], features[num_query:]


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
    parser = argparse.ArgumentParser(description="Urban Elements ReID - Inference")
    parser.add_argument("--config_file", default="./config/UrbanElementsReID_test.yml",
                        help="path to config file", type=str)
    parser.add_argument("--track", default="./results/track",
                        help="output path prefix (e.g. ./results/track → ./results/track_submission.csv)",
                        type=str)
    parser.add_argument("opts", help="Modify config options from the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    assert cfg.MODEL.NAME == 'part_attention_vit', \
        f"This clean baseline only supports part_attention_vit (got {cfg.MODEL.NAME})"

    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inf_pat(cfg, model, val_loader, num_query)

    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)

    qf = qf.cpu().numpy()
    gf = gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    # Class-based filtering with equivalence groups. Strict per-class filtering
    # dropped mAP (0.12873 → 0.12683) because 24 training identities span both
    # "container" and "rubbishbins" — visually ambiguous industrial bins. The
    # strict filter was removing correct matches. Solution: merge the ambiguous
    # pair into one group so same-identity gallery images aren't pruned.
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
    assert len(q_groups) == re_rank_dist.shape[0] and len(g_groups) == re_rank_dist.shape[1], \
        f"group-array / distance shape mismatch: q={len(q_groups)} vs {re_rank_dist.shape[0]}, g={len(g_groups)} vs {re_rank_dist.shape[1]}"
    cross_group_mask = q_groups[:, None] != g_groups[None, :]
    per_query_masked = cross_group_mask.sum(axis=1)
    print(f"class-group filter: masked {cross_group_mask.sum()} / {cross_group_mask.size} cells "
          f"({100*cross_group_mask.mean():.1f}%), avg {per_query_masked.mean():.0f} / {re_rank_dist.shape[1]} per query")
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

    print(f"Submission saved to: {output_path}")
    print(f"Total queries written: {m}")


if __name__ == "__main__":
    main()
