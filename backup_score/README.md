# Backup — Urban Elements ReID 2026 (SkyNet)

**Current best: 0.15421 mAP@100** (set 2026-04-26, post-processing on 7-ckpt ensemble).

## Layout of this folder

### Best submission CSV (the one that scored 0.15421)
```
sweep_dba8_k15_k2_4_lambda027_submission.csv          ← THE WINNER
```
Recipe: 7-checkpoint ensemble + DBA(k=8) + rerank(k1=15, k2=4, λ=0.275) + class-group filter.

### The 7 production checkpoints (renamed for clarity in the backup)
```
seed1234_ep30.pth     (1.2 GB, copy of models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth)
seed1234_ep40.pth     (1.2 GB, copy of models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth)
seed1234_ep50.pth     (1.2 GB, copy of models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth)
seed42_ep30.pth       (1.2 GB, copy of models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth)
seed42_ep40.pth       (1.2 GB, copy of models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth)
seed42_ep50.pth       (1.2 GB, copy of models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth)
seed42_ep60.pth       (1.2 GB, copy of models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth)
```
Total: ~8 GB.

### Code (all the scripts needed to reproduce)
```
ensemble_dba_rerank_sweep.py         ← primary tool — produces 0.15421
ensemble_update.py                   ← simpler 7-ckpt ensemble (produces 0.13361)
ensemble_crossarch_update.py         ← cross-architecture ensemble (DINOv2/EVA — both failed)
ensemble_multiscale_update.py        ← multi-scale ensemble
update.py                            ← single-checkpoint inference
train.py                             ← training entrypoint
processor_part_attention_vit_processor.py  ← training loop (with NaN guards + EMA hooks + grad clip)
utils_ema.py                         ← EMA wrapper module
loss_circle_loss.py                  ← Circle Loss module (untested at safe gamma)
```

### Configs (every YAML used in any experiment, including dead ones)
```
config/UrbanElementsReID_train.yml                       (base recipe)
config/UrbanElementsReID_train_seed42.yml                (seed=42 — produced ckpts in production)
config/UrbanElementsReID_train_seed100_ema.yml           (FAILED — kept for ref)
config/UrbanElementsReID_train_seed100_ema_circle.yml    (FAILED — Circle Loss + grad_clip conflict)
config/UrbanElementsReID_train_dinov2.yml                (FAILED — NaN blowup)
config/UrbanElementsReID_train_eva.yml                   (FAILED — features unsuitable)
config/UrbanElementsReID_train_eva_smoke.yml             (2-epoch verification YAML)
config/UrbanElementsReID_train_merged.yml                (FAILED — UAM negative transfer)
config/UrbanElementsReID_train_heavyaug.yml              (FAILED — too aggressive)
config/UrbanElementsReID_train_deepsup.yml               (FAILED — aux weight too low)
config/UrbanElementsReID_test.yml                        (main test config)
config/UrbanElementsReID_test_dinov2.yml                 (DINOv2 test config)
config/UrbanElementsReID_test_eva.yml                    (EVA test config)
config/defaults.py                                       (full yacs schema)
```

### Documentation
```
README.md           ← this file
notes.md            ← Kaggle submission tally with all scores
context_1.md        ← FULL session history (1146 lines, 5 sessions)
CLAUDE.md           ← Claude Code auto-load file (next-session quick reference)
RESUME.md           ← 1-screen quick-resume guide
OFFSITE_BACKUP.md   ← scp commands to back up to your laptop (RECOMMENDED)
```

### Older intermediate-best CSVs (kept for reference)
```
ep50_rerank_submission.csv                       ← scored 0.12873 (yesterday-of-yesterday best)
ep50_rerank_classfilt_v2_submission.csv          ← scored 0.12927 (after class-group filter)
ensemble_ep30_40_50_classfilt_submission.csv     ← scored 0.13019 (single-run ensemble)
ensemble_crossrun_classfilt_submission.csv       ← scored 0.13361 (cross-seed ensemble — pre-DBA peak)
sweep_dba5_k15_submission.csv                    ← scored 0.13707 (first DBA win)
sweep_dba8_k15_submission.csv                    ← scored 0.14880 (DBA peak found)
sweep_dba8_k15_lambda025_submission.csv          ← scored 0.14976 (+ lambda tuning)
sweep_dba8_k15_k2_4_lambda025_submission.csv     ← scored 0.15288 (+ k2 tuning)
sweep_dba8_k15_k2_4_lambda027_submission.csv     ← scored 0.15421 (CURRENT BEST)
```

## To reproduce the 0.15421 submission

After scp'ing this folder to a new machine (see OFFSITE_BACKUP.md):

1. Restore the 7 ckpts to their expected paths:
   ```bash
   mkdir -p models/model_vitlarge_256x128_60ep
   mkdir -p models/model_vitlarge_256x128_60ep_seed42
   cp backup_score/seed1234_ep30.pth models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth
   cp backup_score/seed1234_ep40.pth models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth
   cp backup_score/seed1234_ep50.pth models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth
   cp backup_score/seed42_ep30.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth
   cp backup_score/seed42_ep40.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth
   cp backup_score/seed42_ep50.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth
   cp backup_score/seed42_ep60.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth
   ```
2. Activate venv and run the sweep:
   ```bash
   cd /workspace/miuam_challenge_diff && source .venv/bin/activate
   python ensemble_dba_rerank_sweep.py
   ```
3. The output CSV `results/sweep_dba8_k15_k2_4_lambda027_submission.csv` should match `backup_score/sweep_dba8_k15_k2_4_lambda027_submission.csv` byte-for-byte.

## Critical "do not retry" reminders (cross-reference: context_1.md §31, §44, §47)

- UAM merged training data (UAM is a different city — negative transfer)
- DINOv2 backbone (failed at LR 3.5e-4 due to NaN; even at LR 1e-4 produced weak ReID features)
- EVA-L backbone (clean training, but MIM+CLIP pretrain doesn't help for fine-grained ReID)
- Circle Loss with gamma=256 + grad_clip=1.0 (they conflict catastrophically)
- EMA-trained ckpt added to ensemble (regressed)
- Multi-layer CLS concat (intermediate layers not ReID-trained)
- H-flip TTA (directional traffic signs)
- PAT part-token concat at inference
- Deep supervision at aux_weight=0.1
- Heavy-aug fine-tune from trained ckpt
- Query expansion α=0.7/K=3 pre-rerank
- DBA k≥10 (over-smooths gallery)
- λ<0.25 in re-rank
- k1=12 in re-rank with current setup

See `notes.md` for the chronological list and `context_1.md` for full reasoning.
