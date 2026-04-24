# Backup — Urban Elements ReID 2026

**Current best: 0.13361** (7-checkpoint cross-seed ensemble + class-group filter)

Score history (this folder is the moving snapshot of the best CSVs and recipe):
- `ep50_rerank_submission.csv` — 0.12873 (ViT-L ep50 + rerank)
- `ep50_rerank_classfilt_v2_submission.csv` — 0.12927 (+ class-group filter)
- `ensemble_ep30_40_50_classfilt_submission.csv` — 0.13019 (trajectory ensemble: seed1234 ep30/40/50)
- `ensemble_crossrun_classfilt_submission.csv` — **0.13361** (cross-seed ensemble: seed1234 ep30/40/50 + seed42 ep30/40/50/60)

The ensemble script (`ensemble_update.py`) is bundled here; `CHECKPOINT_LIST` inside it contains the exact 7-checkpoint recipe for the 0.13361 submission. Expects the per-epoch checkpoints in `models/model_vitlarge_256x128_60ep/` (seed 1234) and `models/model_vitlarge_256x128_60ep_seed42/` (seed 42) — not copied, too big.

**Kaggle score:** 0.12873 mAP@100
**Date:** 2026-04-20
**Team:** SkyNet

## Files in this folder

| File | What it is |
|---|---|
| `part_attention_vit_50.pth` | Trained ViT-Large checkpoint (epoch 50) used to generate the submission |
| `ep50_rerank_submission.csv` | The exact CSV uploaded to Kaggle (928 queries × 100 indices) |
| `UrbanElementsReID_train.yml` | Training config used |
| `UrbanElementsReID_test.yml` | Inference config used |
| `README.md` | This file |

## Recipe

- **Backbone:** ViT-Large/16, pretrained on ImageNet (`jx_vit_large_p16_224-4ee7a4dc.pth`)
- **Input size:** 256×128 (train and test)
- **Part-aware attention:** enabled (PAT architecture, 1 CLS + 4 part tokens)
- **Optimizer:** Adam, base LR 3.5e-4, warmup 10 epochs (linear, factor 0.01), weight decay 1e-4
- **Batch size:** 64 (16 identities × 4 images/id via `RandomIdentitySampler`)
- **Loss:** Triplet (soft margin, hard mining) + Cross-Entropy (label-smoothed) + PatchCenter clustering loss
- **Augmentation:** LGT on (prob 0.5), REA off, pixel mean/std 0.5
- **Epochs planned:** 60 — training was externally killed at epoch 58 (tmux/session issue, not OOM/crash); last saved checkpoint is epoch 50
- **Loss at epoch 50:** total_loss ~1.35, Acc ~99%. Loss was fully plateaued by epoch 40 → epoch 50 ≈ epoch 60.
- **Seed:** 1234
- **Hardware:** RTX 4090 (24 GiB), peak VRAM 12.2 GiB

## Inference

- **Feature norm:** on (L2)
- **Neck feat:** before BNNeck
- **Re-ranking:** k-reciprocal, k1=20, k2=6, λ=0.3
- **TTA:** none effective (update.py's 2× pass is degenerate — identical forward summed)

## Submission

```
kaggle competitions submit -c urban-elements-re-id-challenge-2026 \
  -f ep50_rerank_submission.csv \
  -m "ViT-Large ep50 rerank — 0.12873"
```

## Pipeline fixes applied (vs. the clean-baseline zip)

These fixes are required to make training/inference run at all — keep them if reproducing:

1. **`data/data_utils.py`** — replaced `PathManager.open` → built-in `open` (fvcore import missing from strip-down).
2. **`utils/metrics.py`** — fixed import `utils.reranking` → `utils.re_ranking`; converted re-ranking call from feature form to distance-matrix form with k1=20/k2=6/λ=0.3.
3. **`processor/part_attention_vit_processor.py` (lines 204–213)** — commented out the checkpoint-deletion block that wiped every `.pth` at end of training; now per-epoch checkpoints are retained and the best-epoch is additionally saved as `part_attention_vit_best_<N>.pth`.
4. **Both YAMLs** — retargeted paths from `/home/raza.imam/...` → `/workspace/...`, `DEVICE_ID=('0')`, `LOG_NAME` made consistent with actual config.

## Reproducing this score

```bash
cd /workspace/miuam_challenge_diff
source .venv/bin/activate
mkdir -p results
CUDA_VISIBLE_DEVICES=0 python update.py \
  --config_file config/UrbanElementsReID_test.yml \
  --track results/ep50_rerank
# → results/ep50_rerank_submission.csv  (should match the one in this folder byte-for-byte, given the same checkpoint)
```

## Notes

- Local mAP prints as 100% — artifact of `UrbanElementsReID_test.py` assigning `pid=-1` to all query/gallery (labels hidden for the Kaggle competition). Only the Kaggle leaderboard is real.
- Beat the prior 0.12072 baseline with the same recipe — likely seed-variance win, not a pipeline change (submission path was unchanged).
- Next experiments to try (no retrain needed for #1 and #2): class-filtered retrieval, intermediate-layer CLS concat, real h-flip TTA.
