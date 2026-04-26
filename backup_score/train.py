"""Training entrypoint for Urban Elements ReID.

Only the part_attention_vit model is supported in this clean baseline
(matches the configuration that produced the 0.12072 submission).
"""
import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader
from loss.build_loss import build_loss
from model import make_model
from processor.part_attention_vit_processor import part_attention_vit_do_train_with_amp
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger
import loss as Patchloss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Urban Elements ReID - Training")
    parser.add_argument("--config_file", default="./config/UrbanElementsReID_train.yml",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options from the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("PAT", output_dir, if_train=True)
    logger.info("Saving model to: {}".format(output_dir))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader = build_reid_train_loader(cfg)
    val_name = cfg.DATASETS.TEST[0]
    val_loader, num_query = build_reid_test_loader(cfg, val_name)

    num_classes = len(train_loader.dataset.pids)
    model_name = cfg.MODEL.NAME
    assert model_name == 'part_attention_vit', \
        f"This clean baseline only supports part_attention_vit (got {model_name})"

    model = make_model(cfg, modelname=model_name, num_class=num_classes,
                       camera_num=None, view_num=None)

    # Warm-start fine-tune: if FINETUNE_FROM is set, override the ImageNet
    # init with weights from a previously-trained ReID checkpoint.
    # load_param drops the 'classifier' head, so the new classifier is
    # trained from scratch (matches the standard ReID fine-tune pattern).
    if cfg.MODEL.FINETUNE_FROM:
        print(f"====== warm-start fine-tune from: {cfg.MODEL.FINETUNE_FROM} ======")
        model.load_param(cfg.MODEL.FINETUNE_FROM)

    if cfg.MODEL.FREEZE_PATCH_EMBED and 'resnet' not in cfg.MODEL.NAME:
        model.base.patch_embed.proj.weight.requires_grad = False
        model.base.patch_embed.proj.bias.requires_grad = False
        print("====== freeze patch_embed for stability ======")

    loss_func, center_cri = build_loss(cfg, num_classes=num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    # Patch-guided loss components
    patch_centers = Patchloss.PatchMemory(momentum=0.1, num=1)
    pc_criterion = Patchloss.Pedal(scale=cfg.MODEL.PC_SCALE, k=cfg.MODEL.CLUSTER_K).cuda()
    if cfg.MODEL.SOFT_LABEL:
        print("======== using soft label ========")

    part_attention_vit_do_train_with_amp(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
        patch_centers=patch_centers,
        pc_criterion=pc_criterion,
    )


if __name__ == '__main__':
    main()
