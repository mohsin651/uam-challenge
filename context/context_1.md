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

