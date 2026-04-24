# Experiments log — Urban Elements ReID 2026

Chronological list of every Kaggle submission. Score = mAP@100.

Target: **0.14+** (top 3). Best so far: **0.13361** (set 2026-04-21, 7-checkpoint cross-seed ensemble).

## Score history

| # | Submission CSV | Setup | Score | Δ vs prev best | Result |
|---|---|---|---|---|---|
| 0 | — | Prior baseline (pre-session) | 0.12072 | — | baseline |
| 1 | `ep50_rerank_submission.csv` | ViT-L ep50 + rerank (k1=20/k2=6/λ=0.3) — clean retrain, same recipe | **0.12873** | +0.00801 | ✅ session best v1 |
| 2 | `ep50_rerank_classfilt_submission.csv` | + strict per-class filter | 0.12683 | −0.00190 | ❌ filter pruned true matches |
| 3 | `ep50_rerank_classfilt_v2_submission.csv` | + class-GROUP filter (container∪rubbishbins merged) | **0.12927** | +0.00054 | ✅ session best (still holds) |
| 4 | `ep50_ml6_tta_classfilt_submission.csv` | + last-6 CLS concat + h-flip TTA | 0.12367 | −0.00560 | ❌ big regression |
| 5 | `ep50_ml6_classfilt_submission.csv` | + last-6 CLS concat only (no TTA) | 0.12462 | −0.00465 | ❌ multi-layer concat alone hurts |
| 6 | `ep60_deepsup_ml6_classfilt_submission.csv` | Deep-sup retrain (aux heads on last 5 layers, weight=0.1) + last-6 CLS concat | **0.10788** | −0.02139 | ❌ worst score of session |
| 7 | `ep50_parts_classfilt_submission.csv` | CLS + 3 PAT part tokens (4096-d) | 0.12177 | −0.00750 | ❌ part tokens don't help |
| 8 | `ep50_qe_classfilt_submission.csv` | + query expansion (α=0.7, K=3) | 0.11045 | −0.01882 | ❌ QE hurt |
| 9 | `ensemble_ep30_40_50_classfilt_submission.csv` | Feature ensemble (avg L2-normed CLS from ep30+40+50) + rerank + classfilt — **next day, 2026-04-21** | 0.13019 | +0.00092 | ✅ (was session best) |
| 10 | `ensemble_ep20_30_40_50_classfilt_submission.csv` | + added ep20 to the ensemble | 0.12604 | −0.00415 | ❌ ep20 too noisy, dragged average down |
| 11 | `ensemble_crossrun_classfilt_submission.csv` | Cross-seed ensemble: seed1234 ep30/40/50 + seed42 ep30/40/50/60 (7 checkpoints) | **0.13361** | +0.00342 | ✅ session best — seed variance is the real lever |
| 12 | `ensemble_9ckpt_classfilt_submission.csv` | + 2 heavy-aug fine-tune checkpoints (ep15, ep20 from warm-started ColorJitter+REA fine-tune of seed42 ep60) | 0.12811 | −0.00550 | ❌ heavy aug made a meaningfully-different but meaningfully-worse model |

**Net:** 6 consecutive regressions after submission #3, then ensemble (#9) broke through; wider single-run ensemble (#10) regressed; cross-seed ensemble (#11) gave the biggest jump of the session.

## What worked

- **Retrain with same recipe** (#1): +0.008 — random-seed variance win
- **Class-group filter** (#3): +0.00054 — merge container ∪ rubbishbins into one group, since 24/1088 training identities span both (visually ambiguous industrial bins). Strict per-class filter (#2) hurt because it pruned correct matches.
- **3-checkpoint feature ensemble** (#9): +0.00092 — average L2-normed final-layer CLS features from ep30+40+50 (same training run, different epochs), then renormalize, then rerank+classfilt. Modest but genuine. Implemented in `ensemble_update.py`.
- **Cross-seed ensemble** (#11): +0.00342 — train a second run with different random seed (SEED=42 instead of 1234), then ensemble the 3 middle/late checkpoints from each seed (7 total with seed42's bonus ep60). Biggest single-move gain of the session — seed variance decorrelates errors far more than in-trajectory epoch variance.

## What did NOT work (confirmed empirically)

- **Strict per-class filter** — prunes true matches (container ↔ rubbishbins ambiguity)
- **Multi-layer CLS concat** (last-6) — intermediate layers (19-23) are never supervised by ReID loss in this training recipe, so their CLS tokens aren't discriminative for identity. Concat dilutes the strong final-layer feature.
- **h-flip TTA** — 63% of queries are traffic signals which are DIRECTIONAL (arrows, one-way signs); a flipped image is a different identity, not the same one. Safe for generic ReID, wrong here.
- **Deep-supervision retrain + multi-layer concat** — aux loss weight 0.1/5 = 0.02 per layer was too weak to shape intermediates for ReID, but strong enough to perturb the final layer. Worst of both worlds.
- **PAT part tokens (3 part tokens from final layer)** — trained via Pedal clustering loss, but that loss objective doesn't align with ReID discrimination. Adding them dilutes CLS.
- **Query expansion (α=0.7, K=3)** — top-3 contains enough wrong matches to pollute the expanded query.

## Pipeline fixes made this session (required for training/inference to run)

- `data/data_utils.py` — replaced `PathManager.open` → built-in `open` (missing fvcore)
- `utils/metrics.py` — fixed `utils.reranking` → `utils.re_ranking` + converted re-rank call from feature-form to distance-matrix form (k1=20/k2=6/λ=0.3)
- `processor/part_attention_vit_processor.py:204-213` — commented out checkpoint-deletion block that wiped all .pth at end of training
- Both YAMLs retargeted from `/home/raza.imam/...` → `/workspace/...`, DEVICE_ID=('0'), LOG_NAME cleaned up

## Key insight about why we're stuck

All 928 queries are from camera **c004**; all 2844 gallery images are from c001/c002/c003; training set is c001/c002/c003. **The model has never seen c004 during training.** The ~0.13 ceiling is likely a domain-generalization gap between c004 and the training cameras — not a feature-engineering problem. No amount of post-processing on the CLS features can fix this.

## Final state of code

`update.py` is at the 0.12927 configuration (single-checkpoint reference). A separate `ensemble_update.py` script produces the 0.13019 ensemble submission:
- `extract_feature`: final-layer CLS (1024-d), L2-normalized — same as update.py
- Looped over CHECKPOINT_LIST = [ep30, ep40, ep50] of `models/model_vitlarge_256x128_60ep/`
- Sum the per-checkpoint normalized features → renormalize → rerank → class-group filter
- Re-ranking: k1=20, k2=6, λ=0.3
- Class-group filter: container∪rubbishbins merged

Best CSV: `results/ensemble_ep30_40_50_classfilt_submission.csv` = byte-identical copy in `backup_score/`. Reproduction command (env activated, working dir = repo root):

```bash
python ensemble_update.py --config_file config/UrbanElementsReID_test.yml --track results/ensemble_ep30_40_50_classfilt
```

## Options not yet tried (queued for future sessions)

| Option | Type | Est. time | Expected |
|---|---|---|---|
| Multi-checkpoint ensemble (ep30/40/50/60 of same run, averaged features) | post-processing | ~15 min, no retrain | +0.003 to +0.01 |
| Heavy-augmentation fine-tune (color jitter, grayscale, 20 ep from ep50) | fine-tune | ~25 min | +0.005 to +0.02 |
| DINOv2 backbone retrain (replace supervised ViT with self-supervised pretrain) | full retrain | ~75 min | +0.01 to +0.03 |
| Ensemble across backbones (ViT-L + DINOv2 + ViT-B) | multi-retrain + ensemble | hours | +0.015 to +0.04 |
| Add UrbAM-ReID 2024 dataset as extra training data | full retrain + data download | ~2 hr | +0.02 to +0.04 |

## Submission count (approx)

8 submissions made this session. Check Kaggle daily-cap before the next batch.
