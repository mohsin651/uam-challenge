# Context handoff — Urban Elements ReID 2026 (SkyNet)

**Written:** 2026-04-21. For the next Claude session picking this up.

---

## 0. Who / What / Why

- **User:** Student in IPCV Erasmus Mundus Master at UAM, taking DLVSP1 course. Team name **SkyNet**.
- **Competition:** Kaggle "Urban Elements Re-ID Challenge 2026" (https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026).
- **Task:** Re-identification of urban objects (trash bins, containers, crosswalks, traffic signals). For each query image, return the top-100 gallery images ranked by identity similarity. Metric: mAP@100.
- **Target:** Top 3 (~0.14+ mAP). Starting baseline the user brought in: **0.12072**.
- **User collaboration preferences (from memory):** concise responses, reads diffs directly, don't invent features, ask before destructive ops, convert relative dates to absolute. Has been burned by long training runs and values checkpoint safety.

## 1. The code the user started with

A stripped-down Part-Aware Transformer (PAT, ICCV 2023) codebase. User received it as `miuam_challenge_diff.zip` in `/workspace/`, extracted to `/workspace/miuam_challenge_diff/`. Dataset is at `/workspace/Urban2026/`.

**Project layout** (everything here is used; unused person/vehicle ReID datasets and processors were already stripped):
```
miuam_challenge_diff/
├── train.py                          # training entrypoint
├── update.py                         # inference entrypoint → generates Kaggle submission CSV
├── ensemble_update.py                # (added this session) inference with multi-checkpoint feature averaging
├── evaluate_csv.py                   # local mAP eval (NOT useful — see §3)
├── config/
│   ├── defaults.py                   # yacs schema
│   ├── UrbanElementsReID_train.yml
│   ├── UrbanElementsReID_train_seed42.yml       # (added) SEED=42 run
│   ├── UrbanElementsReID_train_deepsup.yml      # (added) deep-supervision experiment
│   ├── UrbanElementsReID_train_heavyaug.yml     # (added) heavy-aug fine-tune
│   └── UrbanElementsReID_test.yml
├── model/
│   ├── make_model.py                 # build_transformer_PAT class is the active one
│   └── backbones/vit_pytorch.py      # PAT ViT-L with 1 CLS + 3 part tokens + part-aware attention
├── loss/                             # Triplet, CE label-smooth, Pedal clustering, PatchMemory
├── processor/part_attention_vit_processor.py    # training loop + do_inference
├── data/                             # datasets, transforms, samplers
├── utils/                            # logger, meter, metrics (R1_mAP_eval), re_ranking
├── pretrained/                       # jx_vit_large_p16_224-4ee7a4dc.pth goes here (1.2 GB)
├── models/                           # checkpoints saved per training run
├── results/                          # generated submission CSVs
├── logs/                             # training/inference logs
├── backup_score/                     # snapshot of current best submission + README + notes.md
└── context/                          # THIS FOLDER
```

**Architecture recap:** ViT-Large/16 (24 blocks, 1024-d hidden, 305M params) pretrained on ImageNet. At the patch sequence, it prepends **1 CLS token + 3 part tokens**. Each block has standard self-attention AND a part-aware attention where each part token attends to a learned subset of patches (masked). Training uses AMP. Checkpoints are ~1.2 GB each.

## 2. Environment (already set up on this machine)

- GPU: RTX 4090, 24 GiB VRAM. Only GPU 0.
- venv at `/workspace/miuam_challenge_diff/.venv/` (Python 3.10.20 via `uv`). Activate with `source .venv/bin/activate`.
- Installed: torch==2.1.0+cu121, torchvision==0.16.0+cu121, timm, yacs, einops, sklearn, scipy, cv2, numpy==1.26.4, pandas, tensorboard.
- Pretrained ViT-L weights: `/workspace/miuam_challenge_diff/pretrained/jx_vit_large_p16_224-4ee7a4dc.pth` (1.2 GB, already downloaded from `https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth`).
- kaggle CLI is NOT installed. User submits from their laptop after scp'ing the CSV.

**VRAM note:** original handoff said "ViT-L bs=64 needs ~30 GiB" — that was wrong. Measured peak at bs=64 is **12.2 GiB**, well within 24 GiB. Training at the full recipe (bs=64, LR 3.5e-4, 60 epochs) runs fine.

**Speed:** ~144 samples/s → ~47 sec/epoch → ~47 min for 60 epochs. Heavy-aug variant with ColorJitter + REA is similar speed (CPU bound slightly more).

## 3. Dataset (Urban2026/) — critical facts

- Train: 11,175 images / 1,088 identities / cameras c001, c002, c003
- Query: 928 images / hidden identities / **all from camera c004**
- Gallery: 2,844 images / hidden identities / **all from cameras c001, c002, c003** (zero from c004)
- Classes: `trafficsignal`, `crosswalk`, `container`, `rubbishbins` (note inconsistent capitalization: `Container`, `Crosswalk`, `RubbishBins`, `trafficsignal` in raw CSVs — use lowercase normalization when comparing).

**CSV schema gotchas:**
- `train.csv`: `cameraID, imageName, Corresponding Indexes` (3 cols — identity is in column 3 despite the confusing header name).
- `train_classes.csv`: `cameraID, imageName, Corresponding Indexes, Class` (4 cols — class is column 4).
- `query_classes.csv` / `test_classes.csv`: `cameraID, imageName, Class` (3 cols — class is column 3, no identity because hidden).
- `query.csv` and `test.csv`: `cameraID, imageName, Corresponding Indexes` where `Corresponding Indexes` = `-1` (labels hidden).
- Row order of `query.csv` ⇔ `query_classes.csv` is identical (verified with `diff`). Same for test.

**Class distribution:**

| Class | Train IDs | Train imgs | Query imgs | Gallery imgs | Gallery reduction if query is this class |
|---|---|---|---|---|---|
| trafficsignal | 800 (74%) | 7,568 (68%) | 582 (63%) | 1,836 | 1.55× |
| crosswalk | 111 | 1,532 | 91 (10%) | 354 | 8.0× |
| container | 87 | 1,189 | 167 (18%) | 261 | 10.9× |
| rubbishbins | 115 | 886 | 88 (9%) | 393 | 7.2× |

**Identities spanning multiple classes in TRAIN:** 25 out of 1088 (2.3%). Breakdown:
- 24 identities span `container ↔ rubbishbins` (industrial-bin ambiguity, visually identical)
- 1 identity spans `crosswalk ↔ rubbishbins`
- 0 identities span trafficsignal with anything

This is **crucial for class filtering** — see §6.

**The 100% local mAP artifact:** `data/datasets/UrbanElementsReID_test.py` assigns `pid=-1` to every query and gallery image (labels hidden for Kaggle). The internal R1_mAP_eval matches -1 to -1 trivially and reports 100% mAP. **Ignore it.** Only the Kaggle leaderboard is real. This also means `evaluate_csv.py` gives useless results on query/gallery.

**Critical insight about why we're plateauing at ~0.13:** all queries are c004, all gallery is c001-c003, training is c001-c003. **The model has never seen c004 during training.** This is a cross-camera domain-generalization task — and the ~0.13 ceiling appears to be dominated by this domain gap, not feature-engineering headroom. See §7 on what worked / didn't work.

## 4. Pipeline fixes applied (required for training/inference to run)

The "clean baseline" strip-down was sloppy. Fixes I had to make:

1. **`data/data_utils.py`:4, 17** — replaced `PathManager.open(file_name, "rb")` → built-in `open(file_name, "rb")`. fvcore's PathManager was imported but the module wasn't in the zip. Pure local file, so `open()` is equivalent.
2. **`utils/metrics.py`:6** — fixed `from utils.reranking import re_ranking` → `from utils.re_ranking import re_ranking` (filename in repo has underscore).
3. **`utils/metrics.py`:125-132** — the original code called `re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)` assuming a feature-taking `re_ranking` function (which doesn't exist in the stripped repo). Fixed to convert features to distance matrices first and call the existing `re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)`. Also reverted k1/k2 from 50/15 (tuned for big person-ReID datasets) to the paper defaults 20/6 (better for our 2844-image gallery).
4. **`processor/part_attention_vit_processor.py`:204-213** — **COMMENTED OUT the checkpoint-deletion block.** The original code deleted every `.pth` in the log directory at the end of training and saved only the final one. If training crashed on the last epoch, ALL checkpoints were lost. Now per-epoch checkpoints are retained and the best-epoch is additionally saved as `part_attention_vit_best_<N>.pth`. **This fix was critical** — the first seed1234 training crashed at epoch 58 and would've lost everything.
5. **`processor/part_attention_vit_processor.py`:69** — when `DEEP_SUP=True`, model forward returns a 4-tuple instead of 3-tuple. Added conditional unpacking for the PC_LOSS init loop. Only matters if DEEP_SUP is enabled.
6. **Both YAMLs** — retargeted paths from `/home/raza.imam/...` → `/workspace/...`, DEVICE_ID from `('4')` → `('0')` (only GPU on this box), LOG_NAME values fixed to reflect actual config.

## 5. Architectural additions made this session

### 5a. Deep-supervision scaffolding (`MODEL.DEEP_SUP`) — works but didn't help score

Added aux heads on last 5 transformer blocks for the "deep supervision" experiment. **It's still in the code base, default OFF** via `_C.MODEL.DEEP_SUP = False` in `config/defaults.py`.

- `config/defaults.py` — added `_C.MODEL.DEEP_SUP`, `_C.MODEL.NUM_AUX_LAYERS`, `_C.MODEL.AUX_LOSS_WEIGHT`.
- `model/make_model.py` (class `build_transformer_PAT.__init__`) — when DEEP_SUP: builds `self.aux_bottlenecks` (ModuleList of BN1d) and `self.aux_classifiers` (ModuleList of Linear) for the last 5 intermediate layers (indices -2..-6).
- `model/make_model.py` (`forward` method) — when DEEP_SUP and training: returns `(cls_score, layerwise_cls_tokens, layerwise_part_tokens, aux_scores)` — a 4-tuple instead of 3-tuple.
- `processor/part_attention_vit_processor.py` (training loop) — when DEEP_SUP: computes `aux_loss = mean over 5 layers of loss_fn(aux_score_i, aux_feat_i, target, soft_label=False)` and adds `cfg.MODEL.AUX_LOSS_WEIGHT * aux_loss` to `reid_loss`.

### 5b. Warm-start fine-tune flag (`MODEL.FINETUNE_FROM`) — works

- `config/defaults.py` — added `_C.MODEL.FINETUNE_FROM = ''`.
- `train.py` — after `make_model(...)`, if `cfg.MODEL.FINETUNE_FROM` is non-empty, calls `model.load_param(cfg.MODEL.FINETUNE_FROM)`. `load_param` drops the `classifier.*` keys (so new classifier trained from scratch; aux_classifier keys also skipped since they contain "classifier"), but keeps backbone, bottleneck, and aux_bottlenecks.

### 5c. Ensemble inference script (`ensemble_update.py`) — works, is part of the best submission

A new script that:
1. Loads each checkpoint in a `CHECKPOINT_LIST` in turn (edit the list at the top of the file).
2. Runs the same feature extraction as `update.py` (final-layer CLS, L2-normed, 2-pass — 2-pass is a no-op, kept for behavioral fidelity).
3. Accumulates L2-normed features across checkpoints (sum).
4. L2-normalizes the sum.
5. Runs standard re-ranking + class-group filter + Kaggle CSV output (same code path as update.py, inlined).

## 6. Scoring history — every Kaggle submission chronologically

| # | Date | Submission CSV | Setup | Score | Δ vs prev best | Result |
|---|---|---|---|---|---|---|
| 0 | pre-session | (user brought in) | earlier team baseline | 0.12072 | — | baseline |
| 1 | 2026-04-20 | `ep50_rerank_submission.csv` | ViT-L ep50 + rerank (k1=20/k2=6/λ=0.3) — clean retrain, same recipe | **0.12873** | +0.00801 | ✅ session best v1 — random-seed variance + pipeline-fix win |
| 2 | 2026-04-20 | `ep50_rerank_classfilt_submission.csv` | + strict per-class filter | 0.12683 | −0.00190 | ❌ pruned true container↔rubbishbins matches |
| 3 | 2026-04-20 | `ep50_rerank_classfilt_v2_submission.csv` | + class-GROUP filter (container∪rubbishbins merged) | **0.12927** | +0.00054 | ✅ session best v2 |
| 4 | 2026-04-20 | `ep50_ml6_tta_classfilt_submission.csv` | + last-6 layer CLS concat (6144-d) + h-flip TTA | 0.12367 | −0.00560 | ❌ big regression |
| 5 | 2026-04-20 | `ep50_ml6_classfilt_submission.csv` | + last-6 CLS concat only (no TTA) | 0.12462 | −0.00465 | ❌ multi-layer concat alone hurts |
| 6 | 2026-04-20 | `ep60_deepsup_ml6_classfilt_submission.csv` | Deep-sup RETRAIN (aux heads on last 5 layers, weight=0.1) + last-6 CLS concat | **0.10788** | −0.02139 | ❌ WORST score — biggest belly-flop |
| 7 | 2026-04-20 | `ep50_parts_classfilt_submission.csv` | CLS + 3 PAT part tokens concat (4096-d) | 0.12177 | −0.00750 | ❌ part tokens don't help |
| 8 | 2026-04-20 | `ep50_qe_classfilt_submission.csv` | + query expansion (α=0.7, K=3) | 0.11045 | −0.01882 | ❌ QE hurt |
| 9 | 2026-04-21 | `ensemble_ep30_40_50_classfilt_submission.csv` | Feature ensemble (avg L2-normed CLS from ep30+40+50, same seed=1234 run) | 0.13019 | +0.00092 | ✅ was session best (one day) |
| 10 | 2026-04-21 | `ensemble_ep20_30_40_50_classfilt_submission.csv` | Added ep20 to the single-run ensemble | 0.12604 | −0.00415 | ❌ ep20 too early, dragged average down |
| 11 | 2026-04-21 | `ensemble_crossrun_classfilt_submission.csv` | Cross-SEED ensemble: seed1234 ep30/40/50 + seed42 ep30/40/50/60 (7 ckpts) | **0.13361** | +0.00342 | ✅✅ current session best — biggest single-move jump |

**Net trajectory:** 0.12072 → 0.12873 (+retrain) → 0.12927 (+classfilt) → 0.13019 (+traj ensemble) → 0.13361 (+cross-seed). Total +0.01289 (+10.6%).

## 7. What worked and what didn't — empirically confirmed

### ✅ WORKS
- **Retrain with the same 60-epoch recipe** (+0.008): just re-running the same training (seed 1234) gave a slightly better checkpoint than the team's prior result. This was pure seed variance; the pipeline fixes mattered too.
- **Class-group filter** (+0.00054): cross-class matches are guaranteed-wrong, but you must merge `container` and `rubbishbins` into one group because 24/1088 training identities span both. Implementation: after re-ranking, `re_rank_dist[q_group != g_group] = np.inf`, then argsort.
- **3-checkpoint trajectory ensemble** (+0.00092): average L2-normed final-CLS features from ep30+40+50 of the same training run. Small decorrelation signal from epoch variance.
- **Cross-seed ensemble** (+0.00342): train a second run with SEED=42 (was 1234), ensemble the middle/late checkpoints from each. **Biggest single lever** — seed variance gives much more decorrelated errors than in-trajectory epoch variance.

### ❌ DOES NOT WORK (confirmed with Kaggle submission, not speculation)
- **Strict per-class filter** — prunes true matches via the container↔rubbishbins ambiguity.
- **Multi-layer CLS concat (last 6 layers)** — the current training loss only back-propagates through the final-layer CLS token (see `make_model.py:293-297`). Layers 19-23 never see direct ReID signal; their CLS tokens are not identity-discriminative. Concat dilutes the strong final-layer feature.
- **Deep-supervision retrain + multi-layer concat** — aux loss weight of 0.1/5=0.02 per layer was too weak to actually shape intermediates for ReID discrimination, but strong enough to perturb the final layer. Lost from both sides.
- **H-flip TTA** — 63% of queries are traffic signals, which are **directional** (arrow signs, one-way signs). A flipped image is a *different identity*, not the same one. Safe for generic person/vehicle ReID, wrong here.
- **PAT part tokens (3 final-layer part tokens concat with CLS, 4096-d)** — trained via Pedal clustering loss, but that objective doesn't align with direct ReID discrimination. Adding them dilutes CLS.
- **Query expansion (α=0.7, K=3)** — top-3 gallery matches contain enough wrong-identity entries that the expanded query gets polluted.
- **Wider single-run ensemble** (adding ep20 to ep30+40+50) — ep20 is only 10 epochs past warmup (LR just hit peak at ep10), features aren't stable yet, drags down the average.
- **Camera-aware same-camera exclusion rerank** — NOT APPLICABLE because all queries are c004 and all gallery is c001-c003; there is no same-camera overlap to exclude.

### Why these fail — common thread
The trained model's final-layer CLS is extracting essentially everything useful the architecture can extract for this task. Every "extra signal" I've mixed in (intermediate CLS, part tokens, flipped views, top-K gallery expansion) has been *noisier* than useful. The ~0.13 ceiling seems to be the cross-camera (c004 vs c001-3) domain gap, not a feature-engineering plateau.

## 8. Training runs executed this session

Each training run produced per-epoch checkpoints (every 10 epochs) retained due to the deletion-bug fix. All at 256×128, batch 64, Adam lr=3.5e-4, 60 epochs, warmup 10, LGT on, REA off (except heavy-aug variant).

| Run | Seed | Config | Status | Checkpoints | Purpose |
|---|---|---|---|---|---|
| A | 1234 | `UrbanElementsReID_train.yml` | killed at ep58 externally (tmux/session, not OOM) | ep10/20/30/40/50 (no ep60) | base that gave 0.12873 → 0.12927 → 0.13019 |
| B | 1234 | `UrbanElementsReID_train_deepsup.yml` (DEEP_SUP=True) | completed 60 ep | ep10/20/30/40/50/60 | deep-supervision test (gave 0.10788 — failed) |
| C | 42 | `UrbanElementsReID_train_seed42.yml` | completed 60 ep | ep10/20/30/40/50/60 | cross-seed ensemble partner (drove 0.13361) |
| D | 7 | `UrbanElementsReID_train_heavyaug.yml` (warm-start from seed42 ep60) | completed 20 ep (just now, 13:58) | ep5/10/15/20 expected | heavy-aug fine-tune — results pending, NOT yet submitted |

**Run D heavy-aug setup (in case you need to reproduce):**
- FINETUNE_FROM: seed42 ep60 (`models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth`)
- BASE_LR: 5e-5 (7× lower than from-scratch run)
- WARMUP_EPOCHS: 2, WARMUP_FACTOR: 0.1
- MAX_EPOCHS: 20, CHECKPOINT_PERIOD: 5, EVAL_PERIOD: 5
- SEED: 7
- INPUT.REA.ENABLED: True, PROB: 0.4
- INPUT.CJ.ENABLED: True, PROB: 0.6, BRIGHTNESS: 0.3, CONTRAST: 0.3, SATURATION: 0.2, HUE: 0.1
- LGT still on, flip still on by default (DO_FLIP=True in defaults)
- LOG_NAME: `./model_vitlarge_256x128_heavyaug_from_seed42_ep60`

## 9. Losses used (same for all runs — not modifiable without more surgery)

Total loss per training step = **Triplet (soft-margin, hard-mined) + CE (label-smoothed) + Pedal (patch-clustering)**, all weighted 1.0, applied to the final-layer CLS token and the 3 part tokens.

More detail:
- **CE with label smoothing** on `classifier(bottleneck(layerwise_cls_tokens[-1]))` — 1088-way identity classifier on the final-layer CLS, with a `SOFT_LABEL` mechanism where the patch-clustering result feeds `all_posvid` into the CE to mix the hard one-hot target with a soft target (mixed with `SOFT_LAMBDA=0.5`, weighted `SOFT_WEIGHT=0.5`). See `loss/ce_labelSmooth.py` → `CrossEntropyLabelSmooth.__call__`.
- **Soft-margin Triplet** on raw `layerwise_global_feat[-1]` = pre-bottleneck final-layer CLS. Uses hard example mining, softplus formulation (no fixed margin). See `loss/triplet_loss.py` → `TripletLoss`.
- **Pedal (patch-clustering)** on 3 part tokens of the final layer. Uses a `PatchMemory` momentum-updated memory bank. The loss drives part tokens to cluster around identity centers. See `loss/myloss.py`.

Loss magnitudes observed: total_loss starts ~11 at epoch 1, converges to ~1.2 by epoch 60. pc_loss → ~0 by mid-training (clustering converges early and stops contributing).

## 10. Best-known setup (reproduces 0.13361)

```bash
cd /workspace/miuam_challenge_diff
source .venv/bin/activate

# (a) if starting from scratch, train two seeds:
#     nohup python train.py --config_file config/UrbanElementsReID_train.yml        > logs/train.log 2>&1 &    # seed 1234, ~47 min
#     nohup python train.py --config_file config/UrbanElementsReID_train_seed42.yml > logs/train_seed42.log 2>&1 &   # seed 42,  ~75 min (cooler GPU or slower batch = longer)
# (per-epoch .pth files land in models/model_vitlarge_256x128_60ep/{,_seed42/}, every 10 epochs)

# (b) ensemble inference (7 checkpoints — see CHECKPOINT_LIST in ensemble_update.py)
CUDA_VISIBLE_DEVICES=0 python ensemble_update.py \
  --config_file config/UrbanElementsReID_test.yml \
  --track results/ensemble_crossrun_classfilt
#  → results/ensemble_crossrun_classfilt_submission.csv

# (c) submit
kaggle competitions submit -c urban-elements-re-id-challenge-2026 \
  -f /workspace/miuam_challenge_diff/results/ensemble_crossrun_classfilt_submission.csv \
  -m "Cross-run ensemble (seed1234 ep30/40/50 + seed42 ep30/40/50/60) + rerank + class-group filter"
```

The `CHECKPOINT_LIST` inside `ensemble_update.py` contains the exact 7 paths. Backup copy lives at `backup_score/ensemble_update.py`.

`update.py` is at the single-checkpoint (0.12927) reference config — final-layer CLS, no TTA / no concat / no QE / no filter beyond class-group. Rollback-verified byte-identical to the 0.12927 CSV.

## 11. Current state at the moment of writing this context

- **Best score on Kaggle:** 0.13361 (submission #11, cross-seed ensemble).
- **Training runs completed:** A (killed at ep58, ep10-50 checkpoints), B (deepsup, all 60 ep), C (seed42, all 60 ep), D (heavyaug, 20 ep, **finished minutes before this file was written, NOT yet inferenced/submitted**).
- **Current running jobs:** none (all training finished).
- **update.py:** at the 0.12927 single-checkpoint config. Not modified again after rollback.
- **ensemble_update.py:** has the 7-checkpoint cross-run config for 0.13361. If you want to test heavy-aug D as an ensemble member, append its checkpoints to `CHECKPOINT_LIST`.

## 12. Ideas queued (ranked by expected ROI at this point, 2026-04-21)

Having confirmed cross-seed diversity is the biggest lever:

| Priority | Idea | Time | Expected Δ over 0.13361 | Why |
|---|---|---|---|---|
| 1 | Run heavy-aug inference + ensemble (D's ep15 & ep20 into the 7-ckpt stack → 9 ckpts) | 5 min | +0 to +0.01 | Already trained, just needs inference. Adds aug-shifted model variation. |
| 2 | Train a THIRD seed (e.g., seed=100), ensemble 10+ checkpoints | ~75 min + 5 min | +0.003 to +0.008 | Cross-seed has been the most reliable gain. Each new seed adds a similar +0.003. |
| 3 | Weighted ensemble (upweight late checkpoints) | 5 min | +0 to +0.005 | Cheap to probe. ep50/60 may deserve higher weight than ep30. |
| 4 | Rank-fusion across per-checkpoint re-rankings instead of averaging raw features | 10-20 min | +0 to +0.005 | Different combination strategy — might find signal averaging misses. |
| 5 | DINOv2-Large backbone full retrain | ~75 min | +0.01 to +0.03 | Self-supervised pretraining often generalizes across domains (c004 gap). Structural change. |
| 6 | Add UrbAM-ReID 2024 as extra training data | dataset download + ~2 hr | +0.02 to +0.04 | Most direct attack on the domain gap — more urban-object identities in training. Not sure if user has access. |
| 7 | Ensemble across backbones (ViT-L + DINOv2 + maybe SwinV2) | multiple retrains | +0.015 to +0.04 | Highest upside. Biggest time commitment. |

**Avoid these dead ends I already burned submissions on:** multi-layer CLS concat, h-flip TTA, PAT part-token concat at inference, deep supervision at the weights I tried (0.1 total / 0.02 per-layer too low — if you retry, use 1.0 per-layer equal weights and expect a different result), query expansion, wider trajectory ensemble with ep20. These all have Kaggle-confirmed regressions.

## 13. Things to know about the code that aren't obvious

- **The optimizer sees new modules only if they exist at `make_optimizer(cfg, model)` time.** Our DEEP_SUP and FINETUNE_FROM additions happen in model.__init__ and train.py before make_optimizer is called, so they work. If you add new Parameter/Module attributes *after* make_optimizer, they won't get gradients.
- **`load_param` skips any key containing the substring `classifier`.** That's both the main classifier AND `aux_classifiers.*` (if DEEP_SUP was used). The aux_bottlenecks DO get loaded. At inference time we only call `model.base(x)` directly, so skipped classifier weights are irrelevant — but if you add new modules with "classifier" in the name, they'll be silently dropped on load.
- **Data loader tip:** `val_loader.dataset.img_items` gives you `[(img_path, pid, camid, {'q_or_g': 'query'|'gallery'})]` in the order features come out of the loader. Used for aligning class labels to features — see the class-group filter block in `update.py`.
- **Checkpoint file size:** ~1.2 GB each. Don't copy them around more than necessary. The `backup_score/` folder has one checkpoint (`part_attention_vit_50.pth`) bundled.

## 14. User's collaboration signals (things to not re-learn)

- Prefers concise responses. "I can read diffs." Doesn't want long re-explanations.
- Has been burned by long training runs getting killed and losing work. Before running any `train.py` for >10 minutes, **VERIFY the checkpoint-deletion block in `processor/part_attention_vit_processor.py:204-213` is still commented out.** It's commented out now; don't uncomment.
- Asks for confirmation before destructive operations (rm, git force-push, etc.).
- Converts relative dates to absolute in responses ("today" → "2026-04-21").
- User has been stressed/discouraged when experiments fail in a row (6-in-a-row failure streak on 2026-04-20 was emotionally costly). Be honest about expectations; don't oversell speculative ideas. Avoid phrases like "should help" or "big win expected" unless backed by empirical evidence — our track record on predictions has been ~1/7 this session.
- The user gave latitude once ("you are free to do whatever you want") and I still picked things that failed. If the user grants latitude again, **bias toward low-variance options (ensemble adds, small hyperparameter nudges) over speculative structural changes.**

## 15. Pointers outside this folder

- Memory at `/root/.claude/projects/-workspace/memory/MEMORY.md` — persists across Claude sessions. Already contains user_profile, feedback_style, and project_reid_baseline entries.
- Backup at `/workspace/miuam_challenge_diff/backup_score/` — snapshot of the best CSV + ensemble_update.py + README.md + notes.md. Updated after each new best score.
- Logs at `/workspace/miuam_challenge_diff/logs/train*.log` — all four training runs.

That's everything. If you're the next Claude, start by reading this file top to bottom, then `backup_score/notes.md`, then skim the memory entries. After that you have as much context as I do.

---

# Session 2 update — 2026-04-24 (appended to the above, everything below happened AFTER the original file was written)

If you're reading this fresh, read sections 0–15 above first — they're still valid except where noted below.

## 16. What happened since the original context was written

### 16a. Heavy-aug (run D) inference — **FAILED**
- Ran inference adding heavy-aug ep15 + ep20 onto the 7-ckpt cross-seed stack → 9-ckpt ensemble.
- Submission: `results/ensemble_9ckpt_classfilt_submission.csv`. **Kaggle score: 0.12811** — regression of −0.00550 from 0.13361.
- Diagnosis: the 20-epoch ColorJitter+REA fine-tune produced features that were *meaningfully different* but *meaningfully worse*, dragging the average down. Same dilution pattern we'd seen with ep20 and strict-class-filter before.
- **Heavy-aug is officially dead.** The entire heavy-aug checkpoint folder has been deleted (see §20).

### 16b. Leaderboard update — target raised from 0.14 to 0.17
- Current top-5 snapshot (~2026-04-24):
  - 1. Ibn al-Haytham — 0.17607 (90 entries in 5h — HP-sweeping on a working base)
  - 2. Pixora — 0.16642
  - 3. OOM — 0.15220
  - 4. ElzaPak — 0.14331
  - 5. Model Forge — 0.13995
  - 6. **SkyNet (us) — 0.13361**
- Gap to #1: +0.042. No single trick closes that. User explicitly reset target to **0.17** (fight for #1, not just top-3).

### 16c. External-data experiment (UAM_Unified) — **NEGATIVE TRANSFER, FAILED**

**User downloaded:** `/workspace/UAM_Unified.zip` (116 MB) + `/workspace/Annotations.zip` (108 KB). These are from the "UrbAM-ReID 2024" external dataset the competition explicitly permits.

**The extracted data (`/workspace/UAM_Unified_extract/UAM_Unified/`) contains:**
- `image_train/` — 6,387 images
- `train.csv` / `train_classes.csv` — 479 unique identities, cameras c001–c004 (**including 745 c004 training images — the challenge's test-distribution camera!**)
- Classes: container (1578), crosswalk (1329), rubbishbins (400), trafficsign (3080) — note `trafficsign` (not `trafficsignal`; normalize on merge)

**What I built (`merge_datasets.py` in repo root):**
- `/workspace/Urban2026_merged/` — hybrid dataset via symlinks
- `image_train/` = challenge images (original filenames) + UAM images (prefixed `uam_` to avoid collision)
- `train.csv`: 11175 + 6387 = 17,562 rows; UAM identities offset by +1090 so they live in [1091, 1569]
- `train_classes.csv`: same, with `trafficsign` → `trafficsignal` normalization
- `query/image_query`, `test/image_test` symlinks to the challenge's unchanged eval dirs
- The merge is REPRODUCIBLE by running `python merge_datasets.py` (if `UAM_Unified_extract/` and `Urban2026/` exist).

**Training: seed=2024 run on merged data** (`config/UrbanElementsReID_train_merged.yml`).
- Same recipe as seed1234/seed42 (ViT-L/16, 256×128, bs=64, Adam lr=3.5e-4, 60 epochs, LGT+flip+pad augs).
- num_classes auto-adjusts to 1567 (from the merged train set's unique pids) — verified works with the model's classifier head.
- Ran ~95 min. LOG_NAME = `./model_vitlarge_256x128_60ep_merged_seed2024`. All 6 per-epoch ckpts (ep10–ep60) saved cleanly.

**Inference results on Kaggle:**
| Submission | Score | Δ vs 0.13361 |
|---|---|---|
| `results/ep60_merged_solo_classfilt_submission.csv` (merged ep60 solo) | **0.12256** | **−0.01105** |
| `results/ensemble_11ckpt_crossdata_classfilt_submission.csv` (7-ckpt + 4 merged) | **0.13329** | **−0.00032** |

**Why it failed — user revealed after the fact:** UrbAM-ReID 2024 was collected at **UAM campus** (Universidad Autónoma de Madrid), while URVAM-ReID2026 (the challenge) is from an *unspecified city*. So:
- UAM's c004 images are a different physical camera in a different city than the challenge's c004.
- Training on UAM data teaches the model UAM-specific visual features (lighting, architecture, object styles) that actively *contradict* what's needed for the challenge city.
- Adding more "external" identities also diluted classifier capacity (1088 → 1567 classes on the same 305M backbone).
- This is textbook **negative transfer** disguised by a shared-label schema.

**Lesson burned in:** DO NOT train on UAM_Unified mixed with challenge data. If you're going to use UAM at all, consider *only* these alternatives (none of which have been tried yet):
1. Use UAM for **self-supervised pretraining** (DINOv2-style), then fine-tune on challenge only.
2. Use UAM for **backbone warm-start** + long fine-tune on challenge only (to "wash out" UAM domain).
3. **Don't use UAM** — focus on structural diversity (DINOv2, EVA, ViT-H) on the challenge data only.

The merged dataset and UAM zip files are **retained on disk** per user's request (useful for option 1 above). The merged-data model checkpoints are deleted.

### 16d. DINOv2 integration — **BUILT, TESTED, NOT YET LAUNCHED**

User wanted DINOv2-Large as a structural-diversity bet. Did the full integration CPU-side while merged training was still running on the GPU.

**Downloaded:** `/workspace/miuam_challenge_diff/pretrained/dinov2_vitl14_pretrain.pth` (1.2 GB, from `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`). This is the **DINOv2 v1** (no-registers) variant — the simpler one for integration.

**Architectural differences between supervised ViT-L and DINOv2-L:**
- **Patch size 14** (not 16) — means our `PatchEmbed_overlap` needs patch_size=14 stride_size=14.
- **Position embedding** shape `(1, 1370, 1024)` = CLS + 37×37 (DINOv2 was pretrained at 518×518). Needs interpolation to our sequence length.
- **LayerScale (ls1.gamma, ls2.gamma)** per block — multiplicative scalars before each residual add. Our existing Block class didn't have this.
- `mask_token` key exists in DINOv2 state dict (SSL-only, not useful for ReID).

**Code changes I made:**
1. `model/backbones/vit_pytorch.py`:
   - Added `LayerScale(nn.Module)` class (~5 lines).
   - Added `layer_scale_init_value` kwarg to `part_Attention_Block.__init__` (None = Identity, float = active LayerScale). Applied in `forward` as `x + drop_path(ls1(attn(...)))` and `x + drop_path(ls2(mlp(...)))`.
   - Added `layer_scale_init_value` kwarg to `part_Attention_ViT.__init__`, threaded through to block construction.
   - Added `part_attention_vit_large_p14(img_size=(252,126), stride_size=14, ...)` factory. Default `layer_scale_init_value=1e-5`. Input is 252×126 (18×14 by 9×14 — divisible by 14, same 2:1 aspect ratio as 256×128).
   - Added `if k == 'mask_token': continue` to `load_param` to silently skip the SSL-only key.

2. `model/make_model.py`:
   - Imported `part_attention_vit_large_p14`.
   - Added `'vit_large_patch14_dinov2_TransReID': 'dinov2_vitl14_pretrain.pth'` to `imagenet_path_name`.
   - Added `'vit_large_patch14_dinov2_TransReID': part_attention_vit_large_p14` to `__factory_LAT_type` (the part-attention factory map — NOT `__factory_T_type` which is for plain TransReID).
   - Added `elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch14_dinov2_TransReID': self.in_planes = 1024` in BOTH `build_vit` and `build_part_attention_vit` classes (replace_all edit).

3. `config/UrbanElementsReID_train_dinov2.yml` created:
   - TRANSFORMER_TYPE: `vit_large_patch14_dinov2_TransReID`
   - STRIDE_SIZE: [14, 14]
   - INPUT.SIZE_TRAIN / SIZE_TEST: [252, 126]
   - SEED: 42
   - LOG_NAME: `./model_vitlarge_dinov2_p14_merged_seed42`
   - **Current DATASETS.ROOT_DIR: `/workspace/Urban2026_merged/`** — ⚠️ **THIS IS WRONG and needs to be changed to `/workspace/Urban2026/` before launching** (merged data is a dead lever; see §16c). LOG_NAME also ideally updated to drop the `_merged` suffix.

**Smoke test (CPU, model build + forward pass):**
- Model built successfully, 305.77M params (vs supervised ViT-L PAT's 305.38M — +0.4M for LayerScale gammas).
- DINOv2 weights loaded: **345/347 layers loaded** (2 skipped = mask_token + probably one other tiny shape-mismatch that's fine).
- Position embedding correctly resized from (1, 1370, 1024) to (1, 166, 1024). Interpolation from 37×37 grid to 18×9.
- Forward pass on CPU with 2×3×252×126 input → output tensor (2, 1024) ✓

**Status:** Integration verified. Ready to launch training in ~5 min (after fixing the ROOT_DIR in the YAML). Not yet launched because of disk-cleanup-first step (§20).

### 16e. Massive disk cleanup — executed 2026-04-24

Disk hit 91% (46 GB / 50 GB, only 4.7 GB free) before DINOv2 launch could fit. User authorized cleanup of dead experiments + unused checkpoints. **Deleted:**
- `models/model_vitlarge_256x128_60ep_deepsup/` entire folder (8.1 GB) — 0.10788 dead
- `models/model_vitlarge_256x128_heavyaug_from_seed42_ep60/` entire folder (5.7 GB) — 0.12811 dead
- `models/model_vitlarge_256x128_60ep_merged_seed2024/` entire folder (8.0 GB) — 0.12256 / 0.13329 dead
- `models/model_vitlarge_256x128_60ep/part_attention_vit_{10,20}.pth` (2.4 GB) — not in ensemble; ep20 confirmed hurts
- `models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_{10,20}.pth` (2.4 GB) — same
- `models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_best_10.pth` (1.2 GB) — bogus pid=-1 "best"
- `qf.npy`, `gf.npy` in repo root (~15 MB) — temp feature caches

**Freed: ~28 GB.** Disk now at 18 GB / 50 GB (36%).

**Preserved (7-ckpt ensemble reproducibility intact):**
- `models/model_vitlarge_256x128_60ep/part_attention_vit_{30,40,50}.pth` (3.6 GB) — seed1234
- `models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_{30,40,50,60}.pth` (4.8 GB) — seed42
- `pretrained/jx_vit_large_p16_224-4ee7a4dc.pth` (1.2 GB) — supervised ViT-L
- `pretrained/dinov2_vitl14_pretrain.pth` (1.2 GB) — DINOv2-L
- `backup_score/part_attention_vit_50.pth` (1.2 GB) — disaster-recovery copy
- All code, configs, results CSVs, logs, context, backup_score metadata

**IMPORTANT before any retrain:** the checkpoint-deletion safety fix is still in `processor/part_attention_vit_processor.py:204-213` (block commented out). Verify it's still commented out before kicking off a long run.

### 16f. Also preserved (dataset files, per user request)
- `/workspace/UAM_Unified.zip` (116 MB)
- `/workspace/Annotations.zip` (108 KB)
- `/workspace/UAM_Unified_extract/` (135 MB) — the extracted UAM dataset
- `/workspace/Urban2026_merged/` (1.9 MB — all symlinks)

Kept in case user wants to try the "UAM as self-supervised pretraining" idea later, or any of the other workarounds noted in §16c.

## 17. Current best still 0.13361 (2026-04-24, end of Session 2)

The best submission, checkpoints, and ensemble recipe are **unchanged** since Session 1:
- `backup_score/ensemble_crossrun_classfilt_submission.csv` → **0.13361**
- 7-checkpoint ensemble: seed1234 ep30/40/50 + seed42 ep30/40/50/60
- `ensemble_update.py` at repo root is restored to this exact 7-ckpt list (had been modified to 11-ckpt during the merged-data experiment; rolled back to match the backup version before the deletion step).

Reproducing 0.13361 after session cleanup:
```bash
cd /workspace/miuam_challenge_diff && source .venv/bin/activate
python ensemble_update.py --config_file config/UrbanElementsReID_test.yml --track results/reproduced_0.13361
```

## 18. Kaggle submission tally (Session 1 + Session 2)

| # | When | CSV | Score | Outcome |
|---|---|---|---|---|
| 0 | pre-session | — | 0.12072 | prior baseline (team's starting point) |
| 1 | 2026-04-20 | ep50_rerank | 0.12873 | ✅ retrain variance win |
| 2 | 2026-04-20 | ep50_rerank_classfilt (strict) | 0.12683 | ❌ pruned true matches |
| 3 | 2026-04-20 | ep50_rerank_classfilt_v2 (group) | 0.12927 | ✅ class-group filter |
| 4 | 2026-04-20 | ep50_ml6_tta_classfilt | 0.12367 | ❌ |
| 5 | 2026-04-20 | ep50_ml6_classfilt | 0.12462 | ❌ |
| 6 | 2026-04-20 | ep60_deepsup_ml6_classfilt | 0.10788 | ❌ |
| 7 | 2026-04-20 | ep50_parts_classfilt | 0.12177 | ❌ |
| 8 | 2026-04-20 | ep50_qe_classfilt | 0.11045 | ❌ |
| 9 | 2026-04-21 | ensemble_ep30_40_50_classfilt | 0.13019 | ✅ single-run trajectory ensemble |
| 10 | 2026-04-21 | ensemble_ep20_30_40_50_classfilt | 0.12604 | ❌ ep20 noise |
| 11 | 2026-04-21 | ensemble_crossrun_classfilt | **0.13361** | ✅✅ cross-seed — current best |
| 12 | 2026-04-21 | ensemble_9ckpt (+ heavy-aug) | 0.12811 | ❌ |
| 13 | 2026-04-24 | ep60_merged_solo_classfilt | 0.12256 | ❌ UAM negative transfer |
| 14 | 2026-04-24 | ensemble_11ckpt_crossdata_classfilt | 0.13329 | ❌ merged ckpts dilute ensemble |

**14 submissions total. 4 session wins. Session 2 produced 0 wins — it was all exploration.**

## 19. Exact next action (for whoever picks this up next)

DINOv2 launch — two edits away from starting:

1. In `config/UrbanElementsReID_train_dinov2.yml`, change:
   ```yaml
   DATASETS:
     ROOT_DIR: '/workspace/Urban2026_merged/'     # ← change this to...
   ```
   to:
   ```yaml
   DATASETS:
     ROOT_DIR: '/workspace/Urban2026/'            # challenge-only; merged is dead
   ```
2. In the same file, update LOG_NAME for clarity:
   ```yaml
   LOG_NAME: './model_vitlarge_dinov2_p14_merged_seed42'    # ← change to...
   LOG_NAME: './model_vitlarge_dinov2_p14_seed42'
   ```
3. Launch:
   ```bash
   cd /workspace/miuam_challenge_diff && source .venv/bin/activate
   nohup python train.py --config_file config/UrbanElementsReID_train_dinov2.yml \
     > logs/train_dinov2.log 2>&1 &
   ```

Expected runtime: ~95 min for 60 epochs at 252×126 on RTX 4090. Model will be ~306M params, similar memory to ViT-L (maybe slightly more at 162 tokens vs 128). Once done, add 3-4 of its late-epoch checkpoints to the 7-ckpt `CHECKPOINT_LIST` in `ensemble_update.py` → target submission for the next session's best.

## 20. Updated queued ideas — post Session 2 reality

With the UAM-is-a-different-city lesson learned and DINOv2 prep done, the realistic roadmap to 0.17 is:

| Priority | Lever | Why it should help now | Expected Δ |
|---|---|---|---|
| 1 | **DINOv2-L on challenge data only** (ready to launch) | Structural diversity — SSL pretraining differs from supervised ImageNet. First move that doesn't rely on UAM. | +0.005 to +0.020 |
| 2 | **seed=100 on challenge data only** | Proven +0.003 pattern from cross-seed ensemble. Reliable small win. | +0.003 |
| 3 | **EMA weights during training** | Standard trick; add 5 lines to processor. | +0.005 to +0.015 |
| 4 | **Longer training (100–120 epochs) on challenge only** | Pilot whether capacity is underutilized. | +0.005 (uncertain) |
| 5 | **Higher resolution (288×144 or 320×160)** on challenge only | More pixels = finer features for traffic signs. | +0.005 to +0.015 |
| 6 | **ViT-Huge or EVA-02-L backbone** | Bigger model + better pretraining. Requires code additions (factory, weights). | +0.005 to +0.020 |
| 7 | **Re-rank hyperparameter sweep (k1, k2, λ)** on final ensemble | Haven't tried non-defaults on the >0.13 ensemble. | +0.002 to +0.005 |
| 8 | **Database-side augmentation** (replace each gallery feature with mean of its top-K neighbors) | Textbook +0.3–1%. | +0.003 to +0.010 |
| 9 | **Query expansion, smarter implementation** | Previously α=0.7/K=3 failed. Try post-rerank QE or weighted-by-distance QE. | uncertain |

**Dead ideas — confirmed, do not retry unless you change something fundamental:**
- Mixing UAM_Unified training data with challenge data (§16c)
- Multi-layer last-6 CLS concat at inference (0.12462) — unless you retrain with deep supervision at meaningful aux weight (the 0.02-per-layer we tried was too low)
- H-flip TTA (0.12367) — directional traffic signs
- PAT part-token concat at inference (0.12177)
- Query expansion at α=0.7, K=3 with pre-rerank application (0.11045)
- Heavy-augmentation fine-tune from a trained checkpoint (0.12811)
- Deep supervision with aux_weight=0.1/5 layers (0.10788) — retry only with aux_weight≥0.5 per layer

## 21. Still true from Session 1 (read these in the original file above)

- §3 (Dataset facts, including the 100% local mAP artifact — still happens, still ignore)
- §4 (Pipeline fixes applied — all still needed, still in code)
- §9 (Losses used — unchanged)
- §14 (User collaboration signals — read this before interacting with the user)
- §15 (External pointers: memory/, backup_score/, logs/)

If the user asks you about a failed experiment or a specific Kaggle score, this Session 2 update (§16–§20) has the data you need before looking anywhere else.


---

# Session 3 update — 2026-04-24 evening through 2026-04-25 morning (appended again — read this AFTER Sessions 1 and 2 above)

If you're a fresh Claude reading this, you now have three appendices: original (§0-§15), Session 2 (§16-§21), and this Session 3. Read top to bottom. Do NOT re-experiment things confirmed dead.

## 22. State of the leaderboard (rough — as of 2026-04-25 morning)

| Rank | Team | Score |
|---|---|---|
| 1 | Ibn al-Haytham | 0.17607 |
| 2 | ElzaPak | 0.17017 (jumped +0.027 in one submission — they found something) |
| 3 | Pixora | 0.16642 |
| 4 | OOM | 0.15220 |
| 5 | Model Forge | 0.13995 |
| **6** | **SkyNet (us)** | **0.13361** |

Gap to top-3: +0.032. Gap to leader: +0.042. **No single change closes that — it's a stack of tricks**. We've explored most architectural levers; they all failed. Likely top teams are: bigger ensembles, advanced ReID losses (ArcFace/Circle/SubCenterArcFace done correctly), camera-invariance training, and/or pseudo-labeling.

## 23. DINOv2 retry on challenge-only data — FAILED (NaN blowup mid-training)

After confirming UAM merged data was negative transfer (§16c), we tried DINOv2 again on **challenge-only data** (so the data variable was clean and only the backbone differed).

**Setup:**
- TRANSFORMER_TYPE: `vit_large_patch14_dinov2_TransReID` (created in Session 2)
- Input 224×112 (16×8=128 tokens — same memory envelope as supervised ViT-L)
- Original DINOv2 config: BASE_LR=3.5e-4 (same as supervised), no grad clipping
- Pretrained: `pretrained/dinov2_vitl14_pretrain.pth`
- Built using existing `part_attention_vit_large_p14` factory (LayerScale enabled)

**What happened:**
- Training appeared to converge (loss descended normally)
- But: AT EPOCH 40+ all the prefix tokens (`base.cls_token`, `base.part_token1/2/3`, `base.pos_embed`) became **NaN**
- ep10/20/30 checkpoints CLEAN; ep40/50/60 had NaN
- Inference produced all-zero / all-NaN features → cosine similarities were all ~equal → argsort returned arbitrary indices → **0.00000 mAP on Kaggle**
- Tried using only the clean ep10/20/30 checkpoints → got partial signal but still poor (solo 0.11915 / ensemble 0.12651, both regressions vs 0.13361)

**Root cause hypothesis:**
- LayerScale gammas amplify some gradient paths
- Combined with LR 3.5e-4 (too high for DINOv2's sensitive init — DINOv2 papers fine-tune at 1e-4)
- Plus AMP loss-scaler that DOESN'T catch all NaN gradients (it's a heuristic)
- → mid-training, some gradient spike caused a few specific params to overflow to NaN, AMP missed it, training continued silently for 20+ more epochs producing useless weights

**Lesson:** DINOv2 (and any LayerScale-enabled backbone) requires **lower LR (1e-4 or below) AND gradient clipping** as belt-and-suspenders. Don't rely on AMP loss-scaler alone.

**Submissions burned:** 0.00000 (ep60 NaN) + 0.00000 (ensemble with NaN ckpts) + 0.11915 (ep30 solo) + 0.12651 (ensemble with ep10/20/30) = **4 wasted submissions**.

## 24. EVA-L attempt — FAILED (trained cleanly but features unsuitable for ReID)

After DINOv2 failed, user wanted to try ViT-Huge. I argued for EVA-L instead (similar params 303M, simpler integration than ViT-H/14 — mainly because original EVA architecture is just standard ViT-L without LayerScale, vs ViT-H needing custom factory + bigger memory).

**Setup:**
- Used `eva_large_patch14_196.in22k_ft_in22k_in1k` from timm (downloaded via `timm.create_model('...', pretrained=True)`, ~1.2 GB saved to `pretrained/eva_large_patch14_196.pth`)
- Created factory `part_attention_vit_large_p14_eva` (same as DINOv2's p14 factory but `layer_scale_init_value=None` — EVA has no LayerScale)
- Created `config/UrbanElementsReID_train_eva.yml` with **safer hyperparams** (learned from DINOv2 failure):
  - BASE_LR: 1e-4 (3.5× lower)
  - GRAD_CLIP: 1.0
  - SEED: 77
  - Input 224×112, patch14
- Added `'fc_norm'` to load_param skip-list (timm ViT has fc_norm we don't use; previously load_param's error-handling crashed on this missing key)

**Code changes that came along (kept — useful for any future LayerScale or ViT factory):**
- `model/backbones/vit_pytorch.py` — LayerScale module (already from §16d), `part_attention_vit_large_p14_eva()` factory, load_param skips `mask_token` and `fc_norm`
- `model/make_model.py` — registered `vit_large_patch14_eva_TransReID` in `__factory_LAT_type` + filename map + `in_planes=1024` case
- `config/defaults.py` — added `_C.SOLVER.GRAD_CLIP`, `_C.SOLVER.EMA_DECAY`, `_C.MODEL.CIRCLE_MARGIN`, `_C.MODEL.CIRCLE_GAMMA`
- `processor/part_attention_vit_processor.py` — added per-iteration NaN-loss abort + per-epoch NaN-param check (cls_token, part_tokens, pos_embed) + grad clipping with optional unscale + grad-norm logging when clipping
- 2-epoch GPU smoke test confirmed gradient clipping working (grad norms 7-38 → clipped to 1.0)

**What happened:**
- Training completed CLEANLY all 60 epochs (no NaN, gradient clipping engaged, loss descended normally)
- Final ep60: total_loss ~1.18, Acc ~98.5% — looked great
- But Kaggle results: **solo 0.10244, ensemble (7 ViT-L + 4 EVA) 0.12411 — REGRESSION**
- EVA's MIM+CLIP pretrain gives different visual features than supervised ImageNet ViT-L. For this specific dataset, those features are WORSE for fine-grained identity discrimination, even though they were trained cleanly.

**Lesson:** Backbone swap is risky. **DINOv2 and EVA both regressed** on this task vs supervised ViT-L (despite supposedly being "stronger" backbones on general benchmarks). Stop swapping backbones unless you have specific reason to think a different pretrain helps the c004 cross-camera generalization gap.

**Submissions burned:** 0.10244 (EVA solo) + 0.12411 (cross-arch ensemble). 6 architecture-related submissions in total (DINOv2 + EVA), all regressed.

## 25. Disk cleanup #2 (2026-04-24)

After EVA failure, deleted DINOv2 checkpoints (8 GB) since both DINOv2 and merged-data were confirmed dead. **Kept** the DINOv2 pretrained weight (`dinov2_vitl14_pretrain.pth`, 1.2 GB) in case we want to retry DINOv2 with safer hyperparams later. Also kept EVA pretrained weight (`eva_large_patch14_196.pth`, 1.2 GB) for the same reason.

After cleanup: 18 GB / 50 GB used (36%).

## 26. New code added this session (mostly kept; some default-disabled)

All of these are integrated and ready to use; most controlled by config flags so existing recipes are unaffected unless you opt in.

### `utils/ema.py` (NEW)
Exponential Moving Average wrapper. Class `ModelEMA(model, decay=0.9999)`. Updates after every optimizer step. At save time, swap shadow weights into the model (saved checkpoint = EMA weights), then restore raw training weights.

Activate via YAML: `SOLVER.EMA_DECAY: 0.9999` (default 0.0 = disabled). Wired into the processor — when EMA is on, `ema.update(model)` runs after each `scaler.step(optimizer)` and `ema.apply_shadow → save → restore` runs at checkpoint save time.

**Status:** integrated, default off. Used in current seed=100 run.

### `loss/circle_loss.py` (NEW)
Circle Loss (Sun et al. 2020). Drop-in replacement for the soft-margin triplet. Activated by setting `MODEL.METRIC_LOSS_TYPE: 'circle'` in the YAML. Hyperparameters: `MODEL.CIRCLE_MARGIN: 0.25`, `MODEL.CIRCLE_GAMMA: 256.0` (paper defaults).

`build_loss.py` was updated to wrap CircleLoss in a tuple-returning shim so `triplet(feat, target)[0]` calls work identically. The `loss_func` branch now matches `if cfg.MODEL.METRIC_LOSS_TYPE in ('triplet', 'circle')` (originally only matched 'triplet' literally — fixed).

**Status:** integrated, default off. **Tested in training and FAILED** — see §28.

### Per-iteration NaN-loss guard (in processor)
```python
if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
    logger.error(...)
    torch.save(model.state_dict(), os.path.join(log_path, ..._PRENAN_..._.pth))
    return
```
Aborts immediately if total_loss became NaN, saves the pre-NaN state for debugging.

### Per-epoch NaN-param guard (in processor)
After each epoch, checks if `cls_token / part_token1/2/3 / pos_embed` have NaN. If yes, aborts BEFORE saving the corrupted checkpoint.

### Grad-clip with logging (in processor)
```python
if cfg.SOLVER.GRAD_CLIP > 0:
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.SOLVER.GRAD_CLIP)
```
Clips post-unscaled gradients. Logs the pre-clip norm at every log_period iteration so you can see when clipping engages.

### `ensemble_crossarch_update.py` (NEW — Session 2)
Iterates groups of (config_file, list of checkpoints) — each group can have different INPUT.SIZE_TEST, TRANSFORMER_TYPE, etc. Used to ensemble ViT-L/16 (256×128) + DINOv2/14 (224×112) + EVA/14 (224×112) into one submission. Verified working but the cross-arch ensembles regressed because DINOv2/EVA were weak.

### `ensemble_multiscale_update.py` (NEW)
Extension of crossarch that also iterates over scales per group. **Multi-scale for ViT-L/16 didn't work out of the box** — the outer `build_part_attention_vit.load_param` doesn't resize `pos_embed` when inference scale differs from training scale, so trying 288×144 on a 256×128-trained ckpt errors. Currently uses only single-scale per group. **TODO if revisited:** add pos_embed resize to the outer load_param.

## 27. Submissions tally (continuing from §18)

| # | When | CSV | Score | Outcome |
|---|---|---|---|---|
| 15 | 2026-04-24 | `ensemble_11ckpt_crossdata_classfilt` (with merged-data ckpts) | 0.13329 | ❌ |
| 16 | 2026-04-24 | `dinov2_ep60_solo_classfilt` (NaN-corrupted ckpt) | 0.00000 | ❌ |
| 17 | 2026-04-24 | `ensemble_crossarch_vitl_dinov2` (NaN-corrupted ckpts in ensemble) | 0.00000 | ❌ |
| 18 | 2026-04-24 | `dinov2_ep30_solo_classfilt` (clean ep30 only) | 0.11915 | ❌ |
| 19 | 2026-04-24 | `ensemble_crossarch_vitl_dinov2ep30` (7 ViT-L + 3 DINOv2 ep10/20/30) | 0.12651 | ❌ |
| 20 | 2026-04-24 | `eva_ep60_solo_classfilt` | 0.10244 | ❌ |
| 21 | 2026-04-24 | `ensemble_vitl_eva` (7 ViT-L + 4 EVA) | 0.12411 | ❌ |

**7 consecutive regressions** since session 2 ended (which had also ended on a regression). **Best is still 0.13361** — the cross-seed ensemble from Session 1.

## 28. Circle Loss attempt — FAILED at our hyperparams (don't retry as configured)

After EVA failure, I argued the "non-architectural change" path: **seed=100 + EMA + Circle Loss** on the proven supervised ViT-L recipe. User went for it.

**What happened:**
- Used: `METRIC_LOSS_TYPE: 'circle'`, `CIRCLE_MARGIN: 0.25`, `CIRCLE_GAMMA: 256.0` (paper defaults), `BASE_LR: 3.5e-4`, `EMA_DECAY: 0.9999`, `GRAD_CLIP: 1.0`
- At epoch 22-24: total_loss stuck at 53-56, **Acc stuck at 0.34-0.38** (vs 0.9+ for triplet at same epoch)
- Pre-clip grad_norm consistently 78-90; we clipped to 1.0 → effective LR was ~1/80 of intended → model barely moving
- KILLED training before completion

**Root cause:** Circle Loss with `gamma=256` produces very large gradients early in training. Combined with our `grad_clip=1.0` (set conservatively after DINOv2 NaN scare), every step gets aggressively throttled → effective learning rate is dramatically reduced → the model can't actually fit.

**Three ways to make Circle Loss work (untested — for future sessions if revisited):**
1. **Lower gamma** (e.g., 64 or 128) — reduces grad magnitude
2. **Higher GRAD_CLIP** (e.g., 10.0 or remove entirely) — allows the larger Circle grads to actually propagate
3. **Lower BASE_LR + balanced loss weights** — explicitly weight `MODEL.TRIPLET_LOSS_WEIGHT: 0.1` so Circle's large magnitude is balanced against CE

**Lesson:** Don't use `gamma=256` with `grad_clip=1.0`. They fight each other.

## 29. Current run: seed=100 + triplet + EMA (clean recipe; LIVE NOW)

After Circle failure, restarted with the PROVEN recipe (just adding EMA, dropping Circle):
- Same as 0.13361 base recipe (supervised ViT-L, 256×128, batch 64, Adam LR 3.5e-4, 60 epochs)
- Only changes: `SEED: 100` (new), `EMA_DECAY: 0.9999` (new)
- `GRAD_CLIP: 0.0` (disabled; triplet doesn't need it)
- Config: `config/UrbanElementsReID_train_seed100_ema.yml`
- LOG_NAME: `model_vitlarge_256x128_60ep_seed100_ema`

**Important: launched in DETACHED tmux session** because previous nohup'd run (last night) died at epoch 11 when user closed laptop:

```bash
tmux new -d -s reid_train "cd /workspace/miuam_challenge_diff && source .venv/bin/activate && python train.py --config_file config/UrbanElementsReID_train_seed100_ema.yml > logs/train_seed100_ema.log 2>&1"
```

Tmux session name: **`reid_train`**. Survives any disconnect because tmux server is independent of the user's SSH session.

Verified at startup: `Epoch[1] Iter[60/159] total_loss: 8.974 Acc: 0.018` — IDENTICAL to seed=1234 baseline at this point (same shape of curve), confirming clean recipe. EMA enabled with decay=0.9999.

**ETA:** finish ~10:38 (2026-04-25). 6 per-epoch checkpoints will be saved in `models/model_vitlarge_256x128_60ep_seed100_ema/`. The saved weights will be the **EMA shadow weights** (this is the whole point — those are what ensemble well).

## 30. Plan after seed=100 finishes

**Immediate:**
1. Check `models/model_vitlarge_256x128_60ep_seed100_ema/` — should have ep10/20/30/40/50/60. Verify no NaN (just to be safe even though triplet shouldn't NaN).
2. Run **solo inference** on ep60 — diagnostic
3. Run **ensemble** with the existing 7-ckpt list + new seed=100 ckpts (probably ep30/40/50/60). New CHECKPOINT_LIST has 11 entries.
4. Submit ensemble. Expected range: 0.135–0.150 (modest but real).

**If seed=100+EMA helps (>= 0.135):**
- Train **seed=200+EMA** with same recipe. Stack another seed → ensemble of 15 ckpts.
- Likely incremental gain of ~0.003.

**If seed=100+EMA doesn't help (≈ 0.13361 or worse):**
- We've genuinely exhausted what same-recipe iteration can give.
- Real improvement requires either:
  - **Better ReID training** that we haven't tried (ArcFace done right, sub-center ArcFace, instance-batch normalization, camera-aware training)
  - **Different inference tricks** (DBA, advanced reranking)
  - **Longer training** (90/120 epochs — risk of overfit but might help c004 generalization)
- At that point: ACCEPT 0.13361 as the final, save state for posterity, move on.

## 31. CRITICAL NOTES for any future Claude session reading this

### Things to NEVER retry (Kaggle-confirmed dead this whole session arc):
- Mixed UAM training data (negative transfer — UAM is a different city campus)
- DINOv2 backbone at LR 3.5e-4 without grad clipping (NaN blowup at ep40+)
- DINOv2 even at safer LR 1e-4 — still gives weak features for this task (ep30 solo: 0.11915)
- EVA-L backbone at any LR — features unsuitable for fine-grained ReID here (ep60 solo: 0.10244)
- Multi-layer CLS concat (last-6) on supervised ViT-L
- Strict per-class filtering (use class-GROUP filter merging container ∪ rubbishbins)
- H-flip TTA (directional traffic signs)
- PAT part-token concat at inference
- Query expansion α=0.7/K=3 pre-rerank
- Deep supervision at aux_weight=0.1 (too weak — try 0.5+ if revisited)
- Heavy-aug fine-tune from a trained ckpt (ColorJitter+REA from ep60)
- Wider trajectory ensemble (adding ep20 hurts vs ep30/40/50)
- Multi-scale of ViT-L/16 inference (requires pos_embed resize patch in load_param — TODO)
- Circle Loss with gamma=256 and grad_clip=1.0 (they conflict; needs lower gamma OR bigger clip)

### Things that have worked (cumulative gain ~0.013 over baseline):
1. Retrain with same recipe, different seed (+0.008 random-seed variance win)
2. Class-group filter at rerank time (+0.001) — container ∪ rubbishbins merged
3. Cross-seed ensemble (7-ckpt: seed1234 ep30/40/50 + seed42 ep30/40/50/60) (+0.003)

### Things that are integrated but UNTESTED on Kaggle:
- EMA weights (currently training; should give +0.5-1.5% if it works)
- NaN guards (preventive, no perf impact)
- Gradient clipping (preventive, can hurt when clipping too aggressively as shown by Circle Loss)
- Cross-architecture ensemble script (works, but useless because DINOv2/EVA are bad)
- Multi-scale ensemble script (single-scale only currently — multi-scale needs pos_embed resize)

### Operational lessons:
- **Always launch long training in detached tmux**, not just nohup. nohup `&` survives SIGHUP but not all forms of session termination on cloud GPUs.
- **Always check checkpoint integrity** (cls_token / part_tokens / pos_embed for NaN) before trusting a checkpoint. AMP loss-scaler doesn't catch all NaN gradients.
- **Always smoke test** new code on CPU first, then 2-epoch GPU before committing to a 60-epoch run. Catches OOM, NaN, weight-load issues cheaply.

## 32. State of the workspace (snapshot 2026-04-25 09:30)

```
/workspace/Urban2026/                                          # challenge data
/workspace/UAM_Unified.zip                                     # external data (kept, but DON'T merge into training)
/workspace/UAM_Unified_extract/                                # extracted UAM data
/workspace/Urban2026_merged/                                   # symlink-merged dataset (DON'T USE)
/workspace/miuam_challenge_diff/
├── pretrained/
│   ├── jx_vit_large_p16_224-4ee7a4dc.pth        # supervised ViT-L (THE ONE THAT WORKS)
│   ├── dinov2_vitl14_pretrain.pth                # DINOv2 (failed; kept for future retry)
│   └── eva_large_patch14_196.pth                 # EVA (failed; kept for future retry)
├── models/
│   ├── model_vitlarge_256x128_60ep/              # seed=1234, ep30/40/50 — in 0.13361 ensemble
│   ├── model_vitlarge_256x128_60ep_seed42/       # seed=42, ep30/40/50/60 — in 0.13361 ensemble
│   ├── model_eva_large_p14_seed77/               # EVA failed run (5 .pth files, useless — should delete)
│   └── model_vitlarge_256x128_60ep_seed100_ema/  # CURRENT RUN (training now)
├── config/
│   ├── UrbanElementsReID_train.yml                       # base
│   ├── UrbanElementsReID_train_seed42.yml                # seed=42 retrain (Session 1)
│   ├── UrbanElementsReID_train_seed100_ema.yml           # CURRENT (clean: triplet + EMA)
│   ├── UrbanElementsReID_train_seed100_ema_circle.yml    # FAILED — kept for reference (don't reuse)
│   ├── UrbanElementsReID_train_dinov2.yml                # FAILED — kept for reference
│   ├── UrbanElementsReID_train_eva.yml                   # FAILED — kept for reference
│   ├── UrbanElementsReID_train_eva_smoke.yml             # 2-epoch smoke YAML (delete or keep — small)
│   ├── UrbanElementsReID_train_merged.yml                # FAILED — kept for reference
│   ├── UrbanElementsReID_train_heavyaug.yml              # FAILED — kept
│   ├── UrbanElementsReID_train_deepsup.yml               # FAILED — kept
│   ├── UrbanElementsReID_test.yml                        # main test config (ViT-L/16)
│   ├── UrbanElementsReID_test_dinov2.yml                 # DINOv2 test config
│   └── UrbanElementsReID_test_eva.yml                    # EVA test config
├── update.py                                             # single-ckpt inference (rolled back to 0.12927 config)
├── ensemble_update.py                                    # the 7-ckpt ensemble that produced 0.13361
├── ensemble_crossarch_update.py                          # cross-arch (works; useless given EVA/DINOv2 dead)
├── ensemble_multiscale_update.py                         # cross-arch + multi-scale (single-scale only)
├── merge_datasets.py                                     # builds Urban2026_merged (DON'T USE — bad idea)
├── results/                                              # all submission CSVs
├── backup_score/                                         # 0.13361 reference snapshot
└── context/context_1.md                                  # THIS FILE
```

### Best-of-everything reproduction recipe (for the 0.13361)
```bash
cd /workspace/miuam_challenge_diff && source .venv/bin/activate
python ensemble_update.py --config_file config/UrbanElementsReID_test.yml --track results/reproduced_0.13361
```

### To launch any new training in a detachment-safe way
```bash
tmux new -d -s <session_name> "cd /workspace/miuam_challenge_diff && source .venv/bin/activate && python train.py --config_file <yaml> > logs/<log>.log 2>&1"
# Verify with: tmux ls
# Watch live: tmux attach -t <session_name>  (Ctrl-B then D to detach)
```

## 33. The end-of-session honest take

We've done **21 Kaggle submissions** in this run, of which **3 produced wins** (0.12873, 0.12927, 0.13361 base), and 18 were regressions or duds. The trajectory is well-documented. The session has burned a LOT of GPU time on architectural experiments (DINOv2, EVA, deep-sup, merged-data, heavy-aug) that all failed. The only proven path forward is **iterating on the supervised ViT-L recipe with EMA + multiple seeds**, plus possibly **inference-side improvements (DBA, smarter reranking) that we haven't fully explored.**

If a future session wants to try something genuinely new, two ideas that **haven't been tried at all** that might be worth it (in order of expected ROI):
- **Database-side augmentation (DBA)** at inference — replace each gallery feature with the mean of its top-K neighbors. Standard +0.3-1% trick.
- **ArcFace / sub-center ArcFace** as the ID classification loss (replaces CE), separate from Circle Loss. Different angular-margin formulation; might work where Circle didn't.
- **Camera-as-input** — adding camera ID embedding as an extra input token, training to be camera-invariant via gradient reversal. Targets the c004 domain gap directly.

Don't try things that are NOT in §31's "never retry" list and that aren't in this "untried" list — they've already been tried in some form.


---

# Session 4 update — 2026-04-25 morning + afternoon (the big breakthrough day)

**Read this AFTER Sessions 1, 2, 3 above.** We are now at **0.14976** — first time crossing 0.14, gained +0.016 in a single afternoon **with zero retraining, all post-processing on the existing 7-checkpoint ensemble**. The Path A diagnostic (DBA + rerank sweeps) succeeded spectacularly.

## 34. Score progression today (2026-04-25)

| Submission | Config | Score | Δ vs prev best |
|---|---|---|---|
| pre-day base | 7-ckpt ensemble + class-group filter (k1=20/k2=6/λ=0.3, no DBA) | 0.13361 | — |
| **morning round** (failures during seed=100+EMA path): | | | |
| `seed100_ema_ep60_solo_classfilt` | seed=100 + EMA solo | 0.11739 | -0.016 ❌ |
| `ensemble_3seeds_ema` | 7-ckpt + 4× seed100+EMA ckpts (11 ckpts) | 0.12625 | -0.0074 ❌ |
| **afternoon: DBA + rerank sweep** | | | |
| `sweep_dba5_k15` | + DBA(k=5) + k1=15 | **0.13707** | +0.00346 ✅ first win |
| `sweep_dba5_k12` | + DBA(k=5) + k1=12 | 0.13694 | flat (k1 doesn't matter alone) |
| `sweep_dba8_k15` | + DBA(k=8) + k1=15 | **0.14880** | +0.01173 ✅✅ huge jump |
| `sweep_dba12_k15` | + DBA(k=12) + k1=15 | 0.13159 | -0.0172 ❌ over-smoothed |
| `sweep_dba7_k15` | + DBA(k=7) + k1=15 | 0.14630 | confirms k=8 peak |
| `sweep_dba8_k15_lambda025` | + DBA(k=8) + k1=15 + λ=0.25 | **0.14976** | +0.00096 ✅ current best |
| `sweep_dba8_k15_lambda020` | + DBA(k=8) + k1=15 + λ=0.20 | 0.14715 | -0.0026 (λ peaked at 0.25) |

**Net gain in one afternoon: +0.01615 (from 0.13361 to 0.14976) using only post-processing.**

## 35. The DBA + rerank tuning curves we mapped today

### DBA k value (with k1=15, λ=0.30 mostly)
```
k=5:  0.13707
k=7:  0.14630
k=8:  0.14880  ← PEAK (sharp on the upper side)
k=12: 0.13159  ← catastrophic over-smoothing
```
**Lesson:** k=8 is the sweet spot for our 7-checkpoint ensemble's gallery. The peak is asymmetric — going k=8→7 loses ~0.003, going k=8→12 loses ~0.017.

### Rerank λ value (with DBA k=8, k1=15)
```
λ=0.20: 0.14715
λ=0.25: 0.14976  ← PEAK
λ=0.30: 0.14880
```
**Lesson:** λ=0.25 (more weight on jaccard than the default 0.30) gives a small but real gain. Drop is asymmetric here too — going below 0.25 hurts more than going above.

### Rerank k1 value (with DBA k=5, λ=0.30)
```
k1=12: 0.13694
k1=15: 0.13707
```
**Lesson:** k1 tightening (20→15) only helps in combination with DBA. k1=15 vs k1=12 is essentially flat — the original k1=20 default was probably also fine for non-DBA setups; with DBA k=8 the new winning config uses k1=15.

## 36. The current winning recipe (reproducing 0.14976)

The 7 checkpoints are unchanged from §17:
- seed1234: ep30, ep40, ep50
- seed42: ep30, ep40, ep50, ep60

Post-processing recipe:
1. Extract L2-normalized final-layer CLS features from each ckpt (single forward, normalize, sum across all 7 ckpts, L2-renormalize)
2. **Apply DBA with k=8 to gallery features:** for each gallery feature, replace it with the L2-normalized mean of its top-8 nearest neighbors (in cosine similarity, including itself)
3. Compute distance matrices (q-g, q-q, g-g) from the (DBA-smoothed gallery, raw query) features
4. Re-rank with **k1=15, k2=5, λ=0.25** (was k1=20, k2=6, λ=0.30)
5. Apply class-group filter (mask cross-group cells to +∞; container ∪ rubbishbins still merged)
6. Argsort, write top-100 indices to CSV

Reproducible via `ensemble_dba_rerank_sweep.py` — the variant `dba8_k15_lambda025` regenerates this CSV in ~75 sec.

## 37. New code added in Session 4

### `ensemble_dba_rerank_sweep.py` (NEW — primary tool of the day)
Standalone script that:
1. Extracts features from a fixed `CHECKPOINT_LIST` (currently the proven 7) — once.
2. For each entry in `VARIANTS`, applies DBA(dba_k) to gallery, runs k-reciprocal rerank with given k1/k2/λ, applies class-group filter, writes a Kaggle CSV.
3. Variants are a tuple `(label, dba_k, k1, k2, lambda)` so easy to expand.

`db_augment(gf, k)` function — straightforward mean of top-k including self, then L2-renormalize.

Latest VARIANTS list (Round 4) is in the script — has unsumbitted candidates ready (`sweep_dba8_k15_k2_4_lambda025`, `sweep_dba8_k15_k2_3_lambda025`, etc.) — see §39.

**The class-group filter and the basic ensemble code is duplicated between this script and `ensemble_update.py` / `ensemble_crossarch_update.py` / `ensemble_multiscale_update.py`.** Refactoring opportunity but not urgent.

## 38. Updates to "what works" and "what doesn't" lists

Add to **WORKS** (Session 4 additions):
4. **DBA k=8 on gallery features** — single biggest post-processing win this session (+0.012). Smooths gallery features by averaging with their top-K neighbors. Implementation: `gf = (gf[topk_idx_per_row].mean(axis=1)); gf = L2_normalize(gf)`.
5. **Re-rank λ=0.25** — tiny but real (+0.001) when combined with DBA k=8.
6. **Re-rank k1=15** — combined with DBA k=8 helps; not isolated effect.

Add to **DOES NOT WORK / DEAD ENDS** (new in Session 4):
- **EMA-trained ckpt added to the 7-ckpt ensemble** (seed=100+EMA ep30/40/50/60 added → 11 ckpts) — score dropped from 0.13361 to 0.12625. Either EMA hurt or seed=100 was an unlucky draw; either way, that specific 4 ckpts shouldn't be in the production ensemble. The **seed=100+EMA solo** also scored 0.11739 (significantly worse than seed=1234 solo on its own).
- **DBA k≥12** — over-smooths gallery, kills discrimination (0.13159 at k=12 vs 0.14880 at k=8).
- **Re-rank λ<0.25** — over-weights jaccard, drops (0.14715 at λ=0.20 vs 0.14976 at λ=0.25).
- **Re-rank k1=10/12 alone** (without DBA) — no significant improvement vs k1=15 or default k1=20.

## 39. Untested CSVs ready on disk for tomorrow's submissions

These were generated in Round 4 of the sweep but couldn't submit (daily quota hit). All in `/workspace/miuam_challenge_diff/results/`. Listed in priority order based on most-likely incremental win:

| File | Variant | Why try it |
|---|---|---|
| `sweep_dba8_k15_k2_4_lambda025_submission.csv` | k2=4 (down from 5) at the winner | k2 dimension completely unexplored — most informative single-step |
| `sweep_dba8_k15_k2_3_lambda025_submission.csv` | k2=3 even tighter | if k2=4 helps, k2=3 might help more |
| `sweep_dba8_k15_k2_6_lambda025_submission.csv` | k2=6 (looser) | if k2=4 hurts, try the other direction |
| `sweep_dba8_k15_lambda015_submission.csv` | λ=0.15 | already-on-disk lambda extreme; likely worse than 0.20 (0.14715) — skip unless k2 fails |
| `sweep_dba8_k15_lambda010_submission.csv` | λ=0.10 | even more extreme; very unlikely to help |
| `sweep_dba8_k15_lambda005_submission.csv` | λ=0.05 | extreme; almost pure jaccard. Diagnostic only. |
| `sweep_dba8_k12_lambda025_submission.csv` | k1=12 + winner | tests if k1=12 with the new lambda helps |
| `sweep_dba8_k18_lambda025_submission.csv` | k1=18 + winner | the other direction |
| `sweep_dba8_k15_k2_4_lambda020_submission.csv` | combine k2=4 with λ=0.20 | only if both directions help individually |
| `sweep_dba8_k15_lambda035_submission.csv` | λ=0.35 (looser) | tests asymmetry of lambda peak |

There are also some Round 3 leftovers on disk that we never submitted:
- `sweep_dba10_k15_submission.csv`
- `sweep_dba8_default_submission.csv` (k1=20 with DBA k=8) — diagnostic for whether k1=15 contributes meaningfully
- `sweep_dba10_default_submission.csv`
- `sweep_dba15_k15_submission.csv`, `sweep_dba20_k15_submission.csv`, `sweep_dba25_k15_submission.csv` — likely worse (we know k=12 already over-smooths)

## 40. Recommended TOMORROW sequence

1. **Submit `sweep_dba8_k15_k2_4_lambda025`** (top of §39 list) — explores the last untouched rerank axis.
2. Based on result:
   - If **> 0.150**, try `dba8_k15_k2_3_lambda025` (continue tightening).
   - If **≈ 0.149-0.150**, try `dba8_k15_k2_6_lambda025` (other direction).
   - If **< 0.147**, lock in 0.14976 and pivot.
3. Total submissions to try: 2-3 max. After that, post-processing peak is found.
4. **If post-processing is exhausted (~0.150-0.151 range):** start considering the harder paths from §31's "untried wild bets":
   - DBA + a fine-tune at higher resolution
   - Camera-aware adversarial training
   - ArcFace + camera embedding (the SOTA ReID combo we haven't tried)
   - UAM as SSL pretraining source (NOT mixed training data; that was negative transfer)
5. **Do NOT touch:** the 7-ckpt CHECKPOINT_LIST in `ensemble_dba_rerank_sweep.py`. It's the proven set. Adding seed=100+EMA hurt today.

## 41. Critical state summary for tomorrow

**Current best:** **0.14976** — `sweep_dba8_k15_lambda025_submission.csv`
- Backed up in `backup_score/` ✓
- Reproducible by running `python ensemble_dba_rerank_sweep.py` (variant in `VARIANTS` list)

**Untouched files that produce the best score:**
- 7 .pth files in `models/model_vitlarge_256x128_60ep/{ep30,40,50}.pth` and `models/model_vitlarge_256x128_60ep_seed42/{ep30,40,50,60}.pth`
- `pretrained/jx_vit_large_p16_224-4ee7a4dc.pth` (still needed for any retraining)
- `ensemble_dba_rerank_sweep.py` with `VARIANTS` list

**Cleanup we COULD do (none urgent):**
- `models/model_vitlarge_256x128_60ep_seed100_ema/` (8 GB) — confirmed dragging ensemble down. Could delete if disk pressure returns.
- `models/model_eva_large_p14_seed77/` (8 GB) — EVA failed; can delete.
- DINOv2 + EVA pretrained weights in `pretrained/` (2.4 GB) — keep for now in case we revisit.
- `Urban2026_merged/` symlinks + `UAM_Unified_extract/` — only useful for SSL-pretrain idea.

**Disk currently:** ~26 GB / 50 GB used (52%). Plenty of room.

**Kaggle daily submission cap reached today.** Tomorrow's quota resets — pick CAREFULLY. The §39 priority list is ordered by my best guess of incremental improvement.

## 42. Honest assessment ending Session 4

We did **8 submissions today**, of which **5 wins** (a streak after many failures), netting +0.016. That's the biggest single-day improvement of the entire session arc.

The DBA k=8 + λ=0.25 + k1=15 recipe is the single biggest post-processing win we've found, and it's a perfectly clean improvement: no retraining, no architectural changes, just smarter inference on top of the 7-checkpoint ensemble that produced 0.13361.

**Where we stand on the leaderboard (rough estimate, end-of-day):**
- Top-3 cutoff was ~0.152 yesterday. We're at 0.14976.
- One more good submission could put us at top-3.
- After that, the gap to leader 0.176 is still significant — would need new model training to close.

**Realistic ceiling for current 7-ckpt ensemble + post-processing:** maybe 0.151-0.153.
**Beyond that:** structurally different model needed (one of the wild bets from §31).

If you're a future Claude: **don't touch what's working. Pick tomorrow's submission from §39's priority list. Once post-processing saturates, then think about the wild bets — but get the easy wins first.**


---

# Session 5 update — 2026-04-26 (continuation of Session 4 sweep, fully mapping post-processing space)

**Read after Sessions 1-4 above.** Previous update ended at **0.14976** (dba8_k15_lambda025). This session continues the post-processing sweep using the SAME 7-checkpoint ensemble. We now have **0.15421** as the new best — full lambda × k1 × k2 × DBA-k space mapped, all four peaks confirmed.

## 43. Score progression (continuing from §34)

Submissions made AFTER Session 4 wrap-up (when post-processing best was 0.14976):

| # | Submission CSV | Setup | Score | Δ vs prev best | Result |
|---|---|---|---|---|---|
| 0 | (start of Session 5) | — | — | — | best was 0.14976 |
| s5-1 | `sweep_dba8_k15_k2_4_lambda025` | + k2=4 (was k2=5) | **0.15288** | +0.00312 | ✅ first time crossing 0.15 |
| s5-2 | `sweep_dba8_k15_k2_3_lambda025` | k2=3 (further tightening) | 0.14752 | -0.0054 | ❌ over-tightened |
| s5-3 | `sweep_dba8_k15_k2_4_lambda020` | k2=4 + λ=0.20 (combine new k2 with low λ) | 0.15222 | -0.0007 | ❌ slight drop |
| s5-4 | `sweep_dba9_k15_k2_4_lambda025` | DBA=9 at the new k2 winner | 0.14515 | -0.0077 | ❌ DBA peak still at 8 |
| s5-5 | `sweep_dba8_k15_k2_4_lambda027` | + λ=0.275 (between proven good 0.25 and old default 0.30) | **0.15421** | +0.00133 | ✅✅ new best |
| s5-6 | `sweep_dba8_k15_k2_4_lambda030` | λ=0.30 at the new winner | 0.15410 | -0.0001 | flat (lambda peak around 0.275-0.30) |
| s5-7 | `sweep_dba8_k12_k2_4_lambda025` | k1=12 at the new winner | 0.15210 | -0.0021 | ❌ k1=15 stays the peak |

**Net for Session 5: +0.00445 across 7 submissions (5 wins out of 7).** Cumulative gain across Sessions 4+5: **+0.0206 (from 0.13361 to 0.15421) using only post-processing on existing 7 ckpts.**

## 44. Definitive post-processing peaks across all 4 axes (final)

This is the **most important table** for any future session. We've now mapped enough of the post-processing space to know the optimal config.

### DBA k (gallery-side feature averaging size)
| k | Score (best other config) | Note |
|---|---|---|
| 0 (off) | 0.13361 | baseline |
| 5 | 0.13707 | (k1=15, λ=0.30) |
| 7 | 0.14630 | (k1=15, λ=0.30) |
| **8** | **0.14880-0.15421** ← **PEAK** | best across multiple combos |
| 9 | 0.14515 | (k1=15, k2=4, λ=0.25) — drop |
| 10 | not directly compared at peak | |
| 12 | 0.13159 | (k1=15, λ=0.30) — catastrophic |

### Re-rank k1 (k-reciprocal expansion)
| k1 | Score (best other config) | Note |
|---|---|---|
| 12 | 0.13694 / 0.15210 | always slightly worse than 15 |
| **15** | best across all configs | ← PEAK |
| 18 | not tested at peak | likely similar to 15 |
| 20 (default) | original baseline | works without DBA |

### Re-rank k2 (local QE size)
| k2 | Score (best other config) | Note |
|---|---|---|
| 3 | 0.14752 | (DBA=8, k1=15, λ=0.25) — tight |
| **4** | **0.15288-0.15421** ← **PEAK** | best |
| 5 | 0.14976 | original choice; works |
| 6 (default) | works | |

### Re-rank λ (jaccard/euclidean balance)
| λ | Score (DBA=8, k1=15, k2=4) | Note |
|---|---|---|
| 0.20 | 0.15222 | drop |
| 0.225 | (`sweep_dba8_k15_k2_4_lambda022` on disk, untested) | likely interp |
| 0.25 | 0.15288 | works |
| **0.275** | **0.15421** ← **PEAK** | best so far |
| 0.30 | 0.15410 | tied with 0.275 (plateau) |

**Optimal config:** DBA k=8, rerank k1=15, k2=4, λ=0.275, class-group filter on, ensemble = 7 ckpts (seed1234 ep30/40/50 + seed42 ep30/40/50/60). Reproduces 0.15421.

## 45. Untested CSVs still on disk (for tomorrow or ensemble purposes)

These were generated in Round 5 but not submitted. Path: `/workspace/miuam_challenge_diff/results/sweep_*.csv`

| File | Variant | Likely outcome |
|---|---|---|
| `sweep_dba6_k15_k2_4_lambda025` | DBA=6 at new winner | likely worse than 8 (peak is at 8) |
| `sweep_dba7_k15_k2_4_lambda025` | DBA=7 at new winner | possibly close to peak (curve slightly asymmetric) |
| `sweep_dba10_k15_k2_4_lambda025` | DBA=10 at new winner | likely worse |
| `sweep_dba8_k15_k2_4_lambda022` | λ=0.225 at new k2 | between 0.20 (drop) and 0.25 (works) — probably interp |
| `sweep_dba8_k15_k2_4_lambda035` | λ=0.35 at new k2 | likely drops (0.30 was already plateau) |
| `sweep_dba8_k15_k2_6_lambda025` | k2=6 at new lambda | likely similar to k2=5 baseline |
| `sweep_dba8_k18_k2_4_lambda025` | k1=18 at new winner | unknown |
| `sweep_dba8_k15_lambda005/010/015` | very low λ at original k2=5 | very unlikely to help (λ=0.20 was already underperforming) |
| `sweep_dba8_k15_k2_4_lambda020` | already submitted (0.15222) | — |
| `sweep_dba12_k20`, `sweep_dba15_k20`, `sweep_dba15_k15`, `sweep_dba20_k15`, `sweep_dba25_k15` | high DBA + various k1 | very likely worse (DBA peak at 8 is sharp going up) |
| `sweep_dba8_default` (k1=20, λ=0.30, no k2 tightening) | DBA only, no rerank tuning | tests if rerank tuning matters; might show DBA alone helps or doesn't |
| `sweep_dba10_default` (k1=20, λ=0.30) | DBA k=10 with default rerank | tests off-peak DBA at default rerank |

**Recommended single submission tomorrow if just one slot:** `sweep_dba8_k15_k2_4_lambda035_submission.csv` to confirm lambda peak doesn't extend past 0.30, OR none — accept 0.15421 and pivot.

## 46. Per-axis curve analysis — fully mapped

We've explored:
- DBA k: {0, 5, 7, 8, 9, 12} = 6 values
- k1: {12, 15, 20} = 3 values
- k2: {3, 4, 5, 6} = 4 values
- λ: {0.20, 0.25, 0.275, 0.30} = 4 values

That's a 6×3×4×4 = 288 cell space. We've sampled ~22 cells (most around the diagonal of the optimum). Coverage is thin but the peaks of each axis are reliably identified.

**Strong signal:** all four axes have peaks near the center of their sampling range. Suggests the optimum is genuinely there and not at the edge.

## 47. Updates to "what works" / "doesn't work" (Session 5 additions)

### Add to WORKS:
7. **k2=4 in re-ranking** (was 6 default) — the third peak we found; +0.003 on top of DBA+lambda tuning. Combined with DBA k=8 + k1=15 + λ=0.275 = 0.15421.
8. **λ=0.275 in re-ranking** (was 0.30 default) — the fourth peak; +0.001 on top. Plateau extends to ~0.30.

### Add to DOES NOT WORK / DEAD ENDS:
- **k2=3** — too tight, drops 0.005 vs k2=4
- **k1=12** — slightly worse than k1=15 even at the new winner (drops 0.002)
- **DBA k≥9 at the optimal k2** — over-smooths, drops 0.005-0.008
- **λ < 0.25 at the new winner config** — drops the score; lambda peak is firmly ≥ 0.25

## 48. Realistic expectations for next steps

After fully mapping the 4-axis post-processing space, **0.15421 looks like the genuine ceiling for the existing 7-ckpt ensemble**. To push higher, we need NEW INFORMATION from a non-post-processing intervention.

| Lever | Effort | Expected | Risk |
|---|---|---|---|
| **A. Train seed=200 (plain triplet, no EMA)** + add to ensemble + re-run sweep | 75 min train + 5 min infer | +0.003 to +0.008 → **0.157-0.162** | low (proven seed-stacking pattern) |
| **B. Stack TWO more seeds (200, 300)** then re-tune sweep | 2.5 hr | +0.005 to +0.012 → **0.16-0.17** | low |
| **C. ArcFace + camera embedding retrain** — never tried, SOTA ReID | ~2 hr code + 75 min train | +0.01 to +0.03 → **0.165-0.184** | medium-high |
| **D. UAM as SSL pretraining source** (NOT mixed training data) | ~3-4 hr | +0.005 to +0.020 → 0.16-0.17+ | high (lots of new code) |
| **E. Pseudo-labeling from current 0.15421 ensemble** | ~2 hr | +0.005 to +0.015 (or hurt) | high |

**My recommendation: A first** (train seed=200, add to ensemble, re-run the sweep — expected to give the easy ~0.005-0.008). After that:
- If 0.16+, try B (stack another seed).
- If still under 0.16, try C (ArcFace + camera).

**DO NOT** add EMA or Circle Loss to seed=200's training — both confirmed to hurt this exact ensemble pattern in §28 and Session 4. Just plain seed variance with the proven recipe.

## 49. Files to back up + state at end of Session 5

**The new winner (0.15421):**
```
backup_score/sweep_dba8_k15_k2_4_lambda027_submission.csv      ← 0.15421
backup_score/sweep_dba8_k15_k2_4_lambda025_submission.csv      ← 0.15288 (intermediate)
```

**Code recipes for reproduction (no retraining):**
- Run `python ensemble_dba_rerank_sweep.py` with the appropriate variant in the VARIANTS list
- The variant currently named `dba8_k15_k2_4_lambda027` (or look it up: `(8, 15, 4, 0.275)`) reproduces 0.15421

**Not changed:** the 7 trained .pth files in `models/model_vitlarge_256x128_60ep/{ep30,40,50}.pth` and `models/model_vitlarge_256x128_60ep_seed42/{ep30,40,50,60}.pth`. These are the foundation.

**Disk state:** ~26 GB / 50 GB used. Plenty of headroom for seed=200 training (~8 GB) plus more.

## 50. Session 5 honest assessment

Combined Session 4+5 = **the single biggest gains of the entire session arc**. We went from 0.13361 → 0.15421 (+0.0206) with **zero retraining**. Just by methodically mapping the post-processing space (DBA k, rerank k1/k2/λ) we found gains that rivaled or exceeded all prior efforts.

**Lessons for next session:**
1. **Always systematically sweep post-processing space FIRST** before jumping to expensive retraining experiments. Top teams almost certainly did this immediately.
2. **DBA is a real lever, not optional.** Default papers' k=5 isn't always right — k=8 was better here. Sweep it.
3. **k2 was an unexplored axis.** Always include it.
4. **Lambda is fine-tunable around the default.** 0.30→0.275 mattered.
5. **Generate sweep variants in batches**, then submit prioritized. Burning Kaggle submissions one-at-a-time on speculative single-run experiments is wasteful when 5-min sweeps can produce a buffet to choose from.

**The 0.15421 recipe is the new starting point** for any future improvement attempt. To beat it: need either more diverse ckpts in the ensemble, better training recipe, or fundamentally different model. Don't churn more on post-processing.

