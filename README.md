# miuam_challenge_diff — Clean PAT baseline for Urban Elements ReID 2026

Minimal, modular copy of the Part-Aware-Transformer code path we actually use.
All unused datasets (Market1501, DukeMTMC, VeRi, …), unused processors, and
unused configs have been stripped. Everything below is required to reproduce
the **0.12072** Kaggle submission.

## Layout

```
miuam_challenge_diff/
├── train.py                  # entrypoint: training
├── update.py                 # entrypoint: inference → track_submission.csv
├── evaluate_csv.py           # entrypoint: local mAP / CMC evaluation
├── enviroments.sh            # dependency install (reference, from PAT repo)
│
├── config/
│   ├── __init__.py
│   ├── defaults.py           # full yacs config schema
│   ├── UrbanElementsReID_train.yml
│   └── UrbanElementsReID_test.yml
│
├── data/
│   ├── __init__.py
│   ├── build_DG_dataloader.py   # build_reid_train_loader / build_reid_test_loader
│   ├── common.py
│   ├── data_utils.py
│   ├── transforms/              # build + Resize/Flip/Pad/Erasing/LGT + autoaug
│   ├── samplers/                # RandomIdentitySampler, InferenceSampler, …
│   └── datasets/
│       ├── __init__.py          # DATASET_REGISTRY — ONLY Urban datasets
│       ├── bases.py
│       ├── UrbanElementsReID.py
│       └── UrbanElementsReID_test.py
│
├── model/
│   ├── __init__.py
│   ├── make_model.py
│   └── backbones/
│       ├── __init__.py
│       ├── vit_pytorch.py       # ViT-Base/Large + part_attention_vit
│       ├── resnet.py            # pulled in by make_model.py imports
│       ├── resnet_ibn.py
│       └── IBN.py
│
├── loss/
│   ├── __init__.py              # exposes PatchMemory (smooth.py) + Pedal (myloss.py)
│   ├── build_loss.py            # called by train.py
│   ├── make_loss.py
│   ├── triplet_loss.py
│   ├── softmax_loss.py
│   ├── ce_labelSmooth.py
│   ├── center_loss.py
│   ├── arcface.py
│   ├── smooth.py
│   ├── myloss.py
│   └── metric_learning.py
│
├── processor/
│   ├── __init__.py
│   └── part_attention_vit_processor.py  # training loop + do_inference
│
├── solver/
│   ├── __init__.py              # make_optimizer + WarmupMultiStepLR
│   ├── make_optimizer.py
│   ├── scheduler_factory.py     # create_scheduler used by train.py
│   ├── scheduler.py             # cosine + warmup
│   ├── cosine_lr.py
│   └── lr_scheduler.py
│
├── utils/
│   ├── __init__.py
│   ├── comm.py
│   ├── logger.py
│   ├── meter.py
│   ├── metrics.py               # R1_mAP_eval
│   ├── re_ranking.py            # k-reciprocal re-ranking
│   └── registry.py
│
├── pretrained/                  # drop jx_vit_large_p16_224-4ee7a4dc.pth here
├── models/                      # checkpoints get saved here
├── results/                     # submission CSVs
└── logs/                        # training logs (also under models/<LOG_NAME>/)
```

## What was removed vs. original

- 40+ unused person/vehicle ReID dataset loaders (Market1501, DukeMTMC,
  MSMT17, VeRi, VehicleID, CUHK03, iLIDS, GRID, PRID, PRAI, RandPerson,
  SenseReID, SYSU_mm, ThermalWorld, PeS3D, CAVIARa, VIPeR, LPW, Shinpuhkan,
  AirportALERT, PKU, the DG_* variants, …) and their imports in
  `data/datasets/__init__.py`.
- `processor/ori_vit_processor_with_amp.py` — we always use
  `part_attention_vit`.
- Fallback branches in `train.py` / `update.py` that dispatched to the
  ori_vit processor; both scripts now assert `cfg.MODEL.NAME == 'part_attention_vit'`.
- `config/PAT.yml` and `config/vit.yml` (person ReID configs, unused).
- `visualization/`, `test.py`, `run.sh` — not part of our pipeline.
- `tb_log/`, `__pycache__/`, stray `.npy` / old submission CSVs.

Nothing in the active code path was modified — `make_model.py`,
`vit_pytorch.py`, `part_attention_vit_processor.py`, `re_ranking.py`,
losses, transforms, samplers, scheduler, etc. are **byte-identical** to the
originals. That guarantees the 0.12072 result is reproducible.

## Prerequisites

1. Conda env `reid` (same one you've been using on srv-02):
   ```bash
   conda activate reid
   ```
   If setting up fresh, use `enviroments.sh` as a reference.

2. Dataset (not copied — use your existing one):
   ```
   /home/raza.imam/miuam_challenge/urban-elements-re-id-challenge-2026/Urban2026/
   ├── image_train/
   ├── image_query/
   ├── image_test/
   ├── train.csv
   ├── query.csv
   └── test.csv
   ```
   Path is set in both YAMLs via `DATASETS.ROOT_DIR`.

3. Pretrained ViT-Large backbone (not copied — it's 1.2 GB):
   ```
   miuam_challenge_diff/pretrained/jx_vit_large_p16_224-4ee7a4dc.pth
   ```
   Either symlink your existing one or copy it:
   ```bash
   ln -s /home/raza.imam/miuam_challenge/Part-Aware-Transformer/pretrained/jx_vit_large_p16_224-4ee7a4dc.pth \
         /home/raza.imam/miuam_challenge_diff/pretrained/
   ```
   Then update `MODEL.PRETRAIN_PATH` in the train YAML to this folder.

4. A trained checkpoint to run `update.py` (not copied — point
   `TEST.WEIGHT` at an existing `.pth` such as your backup
   `backups/score_0.12072/part_attention_vit_60.pth`).

## Training

```bash
cd /home/raza.imam/miuam_challenge_diff
conda activate reid

# Edit config/UrbanElementsReID_train.yml:
#   - MODEL.PRETRAIN_PATH → /home/raza.imam/miuam_challenge_diff/pretrained
#   - DATASETS.ROOT_DIR   → your Urban2026 path
#   - MODEL.DEVICE_ID     → free GPU (NEVER 7)
#   - LOG_NAME            → folder name for this run

nohup python train.py \
  --config_file config/UrbanElementsReID_train.yml \
  > logs/train.log 2>&1 &
```

Checkpoints land in `models/<LOG_NAME>/part_attention_vit_<epoch>.pth`.

## Inference (generate Kaggle submission)

```bash
# Edit config/UrbanElementsReID_test.yml:
#   - TEST.WEIGHT → path to the checkpoint you want to evaluate
#   - TEST.RE_RANKING → True  (this is what gave us 0.12072)

CUDA_VISIBLE_DEVICES=0 python update.py \
  --config_file config/UrbanElementsReID_test.yml \
  --track results/ep60_rerank
# → results/ep60_rerank_submission.csv
```

## Local evaluation (sanity check before Kaggle upload)

```bash
python evaluate_csv.py \
  --path /home/raza.imam/miuam_challenge/urban-elements-re-id-challenge-2026/Urban2026/ \
  --track results/ep60_rerank_submission.csv
```

Note: local mAP will show ~100% on the *query/gallery* split because
`UrbanElementsReID_test.py` assigns `pid=-1` to every query & gallery image
(labels are hidden for Kaggle). Real score only comes from the Kaggle
leaderboard. This was documented in the session notes — ignore the
suspiciously-high local number.

## Reproducing the 0.12072 submission

1. Pretrained weights: `jx_vit_large_p16_224-4ee7a4dc.pth` (ViT-L/16
   ImageNet).
2. Train with `config/UrbanElementsReID_train.yml` as-is (ViT-Large,
   256×128, Adam lr=3.5e-4, warmup 10 ep, 60 epochs, batch 64, LGT on,
   REA off, seed 1234).
3. Use the epoch-60 checkpoint.
4. Run `update.py` with `TEST.RE_RANKING: True`, `FEAT_NORM: True`,
   `NECK_FEAT: 'before'`, 256×128 test size.
5. Upload the generated CSV to Kaggle.
