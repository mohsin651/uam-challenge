import logging
import os
import random
import time
import torch
import torch.nn as nn
from model.make_model import make_model
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.ema import ModelEMA
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def part_attention_vit_do_train_with_amp(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank,
             patch_centers = None,
             pc_criterion= None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PAT.train")
    logger.info('start training')
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    total_loss_meter = AverageMeter()
    reid_loss_meter = AverageMeter()
    pc_loss_meter = AverageMeter()
    # ds_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    scaler = amp.GradScaler(init_scale=512)
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    # EMA wrapper (if enabled). Updates after every optimizer step; shadow
    # weights swapped in at checkpoint-save time, then restored.
    ema = None
    if cfg.SOLVER.EMA_DECAY > 0:
        ema = ModelEMA(model, decay=cfg.SOLVER.EMA_DECAY)
        logger.info(f"EMA enabled with decay={cfg.SOLVER.EMA_DECAY}")
    # train
    if cfg.MODEL.PC_LOSS:
        print('initialize the centers')
        model.train()
        for i, informations in enumerate(train_loader):
            # measure data loading time
            with torch.no_grad():
                #input = input.cuda(non_blocking=True)
                input = informations['images'].cuda(non_blocking=True)
                vid = informations['targets']
                camid = informations['camid']
                path = informations['img_path']
                #input = input.view(-1, input.size(2), input.size(3), input.size(4))

                # compute output
                model_out = model(input, label=vid.cuda(non_blocking=True))
                if cfg.MODEL.DEEP_SUP or cfg.MODEL.CAM_ADV:
                    _, _, layerwise_feat_list, _ = model_out
                else:
                    _, _, layerwise_feat_list = model_out
                patch_centers.get_soft_label(path, layerwise_feat_list[-1], vid=vid, camid=camid)
        print('initialization done')
    
    best_mAP = 0.0
    best_index = 1
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss_meter.reset()
        reid_loss_meter.reset()
        acc_meter.reset()
        pc_loss_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            camid = informations['camid']
            img_path = informations['img_path']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = camid.to(device)
            t_domains = t_domains.to(device)

            model.to(device)
            with amp.autocast(enabled=True):
                # Pass label so ArcFace can compute cos(theta+margin) on the
                # true class. With ID_LOSS_TYPE='softmax' (default), label is
                # ignored and plain classifier is used.
                model_out = model(img, label=target)
                cam_logits = None
                if cfg.MODEL.DEEP_SUP:
                    score, layerwise_global_feat, layerwise_feat_list, aux_scores = model_out
                elif cfg.MODEL.CAM_ADV:
                    score, layerwise_global_feat, layerwise_feat_list, cam_logits = model_out
                else:
                    score, layerwise_global_feat, layerwise_feat_list = model_out

                ############## patch learning ######################
                l_ploss = cfg.MODEL.PC_LR
                if cfg.MODEL.PC_LOSS:
                    patch_agent, position = patch_centers.get_soft_label(img_path, layerwise_feat_list[-1], vid=vid, camid=camid)
                    feat = torch.stack(layerwise_feat_list[-1], dim=0)
                    feat = feat[:,::1,:]
                    '''
                    loss1: clustering loss(for patch centers)
                    '''
                    ploss, all_posvid = pc_criterion(feat, patch_agent, position, patch_centers, vid=target, camid=target_cam)
                    '''
                    loss2: reid-specific loss
                    (ID + Triplet loss)
                    '''
                    reid_loss = loss_fn(score, layerwise_global_feat[-1], target, all_posvid=all_posvid, soft_label=cfg.MODEL.SOFT_LABEL, soft_weight=cfg.MODEL.SOFT_WEIGHT, soft_lambda=cfg.MODEL.SOFT_LAMBDA)
                else:
                    ploss = torch.tensor([0.]).cuda()
                    reid_loss = loss_fn(score, layerwise_global_feat[-1], target, soft_label=cfg.MODEL.SOFT_LABEL)

                # Deep supervision: CE+Triplet on last NUM_AUX_LAYERS intermediate
                # layers' CLS tokens. aux_scores[i] corresponds to layer -(i+2).
                if cfg.MODEL.DEEP_SUP:
                    aux_loss = 0.0
                    for i in range(cfg.MODEL.NUM_AUX_LAYERS):
                        aux_feat = layerwise_global_feat[-(i + 2)]
                        aux_loss = aux_loss + loss_fn(aux_scores[i], aux_feat, target, soft_label=False)
                    aux_loss = aux_loss / cfg.MODEL.NUM_AUX_LAYERS
                    reid_loss = reid_loss + cfg.MODEL.AUX_LOSS_WEIGHT * aux_loss

                # Camera-adversarial loss. The grad-reversal layer inside the
                # model already flipped the gradient sign for the backbone's
                # path, so adding a POSITIVE cam_loss to total_loss makes the
                # camera classifier improve while the backbone is pushed to be
                # camera-invariant (negative effective gradient on backbone).
                cam_loss = torch.tensor(0.0, device=img.device)
                if cfg.MODEL.CAM_ADV and cam_logits is not None:
                    # camid is 1-indexed in the dataset (cameras start at c001).
                    # Subtract 1 to match cam_classifier output indices [0, num_cameras-1].
                    cam_target = (target_cam - 1).clamp(0, cfg.MODEL.NUM_CAMERAS - 1)
                    cam_loss = F.cross_entropy(cam_logits, cam_target)
                    reid_loss = reid_loss + cfg.MODEL.CAM_ADV_WEIGHT * cam_loss

                total_loss = reid_loss + l_ploss*ploss

            # Per-iteration NaN-loss guard: if total_loss became NaN before we
            # even backprop, abort immediately — otherwise training wastes GPU
            # running further corrupted steps.
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                logger.error(f"NaN/Inf in total_loss at epoch {epoch} iter {n_iter}. Aborting.")
                torch.save(model.state_dict(), os.path.join(log_path, cfg.MODEL.NAME + f'_PRENAN_ep{epoch}_it{n_iter}.pth'))
                return

            scaler.scale(total_loss).backward()

            # Optional gradient clipping (safety net against AMP missing NaN
            # gradients — was the root cause of the DINOv2 ep40+ blowup).
            # Unscale first so the clipping norm is in the true grad scale.
            grad_norm = None
            if cfg.SOLVER.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.SOLVER.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            # score = scores[-1]
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            total_loss_meter.update(total_loss.item(), img.shape[0])
            reid_loss_meter.update(reid_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            pc_loss_meter.update(ploss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0 and grad_norm is not None:
                logger.info(f"  [grad_norm pre-clip: {grad_norm.item():.3f} / clip_max: {cfg.SOLVER.GRAD_CLIP}]")
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] total_loss: {:.3f}, reid_loss: {:.3f}, pc_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader), total_loss_meter.avg,
                reid_loss_meter.avg, pc_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/reid_loss', reid_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar("train/pc_loss", pc_loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        # End-of-epoch NaN guard on the params DINOv2 blew up on (cls/part/pos_embed).
        # If any are NaN, something went wrong during this epoch — abort before saving.
        nan_params = []
        if hasattr(model, 'base'):
            for pname in ['cls_token', 'part_token1', 'part_token2', 'part_token3', 'pos_embed']:
                p = getattr(model.base, pname, None)
                if p is not None and torch.isnan(p).any().item():
                    nan_params.append(pname)
        if nan_params:
            logger.error(f"NaN detected in {nan_params} at end of epoch {epoch}. Aborting (not saving this checkpoint).")
            return

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
        
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                cmc, mAP = do_inference(cfg, model, val_loader, num_query)
                tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
                tbWriter.add_scalar('val/mAP', mAP, epoch)

        if epoch % checkpoint_period == 0:
            if best_mAP < mAP:
                best_mAP = mAP
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
            # If EMA is on, save the EMA (shadow) weights — those are the ones
            # that ensemble well and produce the best inference result. Restore
            # raw training weights afterwards so training continues correctly.
            if ema is not None:
                ema.apply_shadow(model)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            if ema is not None:
                ema.restore(model)
        torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_{}.pth'.format(best_index))
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None)
    eval_model.load_param(load_path)
    print('load weights from {}_{}.pth'.format(cfg.MODEL.NAME, best_index))
    for testname in cfg.DATASETS.TEST:
        if 'ALL' in testname:
            testname = 'DG_' + testname.split('_')[1]
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        do_inference(cfg, eval_model, val_loader, num_query)
    
    # DISABLED: original code deleted every .pth in log_path before saving the final
    # checkpoint, so a crash at the final epoch wiped all training progress.
    # Keep all per-epoch checkpoints; just save the best-epoch re-save alongside them.
    # del_list = os.listdir(log_path)
    # for fname in del_list:
    #     if '.pth' in fname:
    #         os.remove(os.path.join(log_path, fname))
    #         print('removing {}. '.format(os.path.join(log_path, fname)))
    print('saving final checkpoint (best epoch re-saved, per-epoch checkpoints retained).')
    torch.save(eval_model.state_dict(), os.path.join(log_path, cfg.MODEL.NAME + '_best_{}.pth'.format(best_index)))
    print('done!')

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("PAT.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    t0 = time.time()
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        # domains = informations['others']['domains']
        with torch.no_grad():
            img = img.to(device)
            # camids = camids.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camids))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP