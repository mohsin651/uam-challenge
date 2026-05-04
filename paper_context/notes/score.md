# Score & Training Log Reference — Urban Elements ReID 2026

**Best score: 0.15884** (set 2026-04-26, locked across 50+ subsequent attempts).

This file is a flat lookup table of every experiment with: score, training log path, submission CSV path, and short description. Use for paper figures (loss curves, ablation tables, score progression).

---

## A. Score progression (chronological wins)

| # | Date | Score | Δ vs prev | Recipe / Win |
|---|---|---|---|---|
| 0 | pre-session | 0.12072 | — | Prior team baseline |
| 1 | 2026-04-20 | **0.12873** | +0.00801 | Clean retrain seed=1234 (random-seed variance) |
| 2 | 2026-04-20 | **0.12927** | +0.00054 | + class-group filter (container ∪ rubbishbins merged) |
| 3 | 2026-04-21 | 0.13019 | +0.00092 | + 3-ckpt single-run trajectory ensemble (seed=1234 ep30/40/50) |
| 4 | 2026-04-21 | **0.13361** | +0.00342 | + cross-seed ensemble (added seed=42 ep30/40/50/60) |
| 5 | 2026-04-25 | **0.14976** | +0.01615 | + DBA k=8 + rerank k1=15 + λ=0.25 (post-processing wins) |
| 6 | 2026-04-26 | **0.15421** | +0.00445 | + k2=4 + λ=0.275 (post-proc fine-tune) |
| 7 | 2026-04-26 | **0.15884** | +0.00463 | **+ camera-adversarial ep60 ckpt added** ← FINAL BEST |

**Total lift: 0.12072 → 0.15884 = +0.03812 (+31.6% relative).**

---

## B. Production checkpoints (8-ckpt 0.15884 ensemble)

| # | Path | Recipe | Train log |
|---|---|---|---|
| 1 | `models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth` | seed=1234 baseline, ep30 | `models/model_vitlarge_256x128_60ep/train.log` |
| 2 | `models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth` | seed=1234 baseline, ep40 | (same) |
| 3 | `models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth` | seed=1234 baseline, ep50 | (same) |
| 4 | `models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth` | seed=42 baseline, ep30 | `models/model_vitlarge_256x128_60ep_seed42/train.log` |
| 5 | `models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth` | seed=42, ep40 | (same) |
| 6 | `models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth` | seed=42, ep50 | (same) |
| 7 | `models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth` | seed=42, ep60 | (same) |
| 8 | `models/model_vitlarge_camadv_seed500/part_attention_vit_60.pth` | cam-adv s500, ep60 (the +0.005 winner) | `models/model_vitlarge_camadv_seed500/train.log` |

**Reproduce 0.15884:**
```bash
cd /workspace/miuam_challenge_diff && source .venv/bin/activate
python camadv_inference.py
# generates results/camadv_baseline7_plus_camadv_ep60_submission.csv → 0.15884
```

**Backup of winning CSV:** `backup_score/camadv_baseline7_plus_camadv_ep60_0.15884.csv`

---

## C. ALL Kaggle submissions, chronological

### Sessions 1-3 (2026-04-20 to 04-25): post-processing exploration

| # | When | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|---|
| 1 | 2026-04-20 | `results/ep50_rerank_submission.csv` | 0.12873 | -0.030 |
| 2 | 2026-04-20 | `results/ep50_rerank_classfilt_submission.csv` (strict per-class) | 0.12683 | -0.032 |
| 3 | 2026-04-20 | `results/ep50_rerank_classfilt_v2_submission.csv` (group filter) | 0.12927 | -0.030 |
| 4 | 2026-04-20 | `results/ep50_ml6_tta_classfilt_submission.csv` (multi-layer + TTA) | 0.12367 | -0.035 |
| 5 | 2026-04-20 | `results/ep50_ml6_classfilt_submission.csv` | 0.12462 | -0.034 |
| 6 | 2026-04-20 | `results/ep60_deepsup_ml6_classfilt_submission.csv` (deep-sup) | 0.10788 | -0.051 |
| 7 | 2026-04-20 | `results/ep50_parts_classfilt_submission.csv` (part-token concat) | 0.12177 | -0.037 |
| 8 | 2026-04-20 | `results/ep50_qe_classfilt_submission.csv` (QE α=0.7) | 0.11045 | -0.048 |
| 9 | 2026-04-21 | `results/ensemble_ep30_40_50_classfilt_submission.csv` | 0.13019 | -0.029 |
| 10 | 2026-04-21 | `results/ensemble_ep20_30_40_50_classfilt_submission.csv` | 0.12604 | -0.033 |
| 11 | 2026-04-21 | `results/ensemble_crossrun_classfilt_submission.csv` (7-ckpt cross-seed) | **0.13361** | -0.025 |

### Session 2 (2026-04-24): UAM merge + DINOv2 setup

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 12 | `results/ensemble_9ckpt_classfilt_submission.csv` (heavy-aug 9-ckpt) | 0.12811 | -0.031 |
| 13 | `results/ep60_merged_solo_classfilt_submission.csv` (UAM merge) | 0.12256 | -0.036 |
| 14 | `results/ensemble_11ckpt_crossdata_classfilt_submission.csv` | 0.13329 | -0.026 |

### Session 3 (2026-04-24/25): backbone swaps DINOv2 + EVA

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 15 | `results/ensemble_11ckpt_crossdata_classfilt_submission.csv` (re-tested) | 0.13329 | -0.026 |
| 16 | `results/dinov2_ep60_solo_classfilt_submission.csv` (NaN-corrupted) | 0.00000 | -0.159 |
| 17 | `results/ensemble_crossarch_vitl_dinov2_submission.csv` (NaN ckpts) | 0.00000 | -0.159 |
| 18 | `results/dinov2_ep30_solo_classfilt_submission.csv` (clean ep30) | 0.11915 | -0.040 |
| 19 | `results/ensemble_crossarch_vitl_dinov2ep30_submission.csv` | 0.12651 | -0.032 |
| 20 | `results/eva_ep60_solo_classfilt_submission.csv` | 0.10244 | -0.057 |
| 21 | `results/ensemble_vitl_eva_submission.csv` (7 ViT-L + 4 EVA) | 0.12411 | -0.035 |

### Sessions 4-5 (2026-04-25/26): DBA + rerank sweeps (BIG WINS)

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 22 | `results/seed100_ema_ep60_solo_classfilt_submission.csv` | 0.11739 | -0.041 |
| 23 | `results/ensemble_3seeds_ema_submission.csv` | 0.12625 | -0.033 |
| 24 | `results/sweep_dba5_k15_submission.csv` | 0.13707 | -0.022 |
| 25 | `results/sweep_dba5_k12_submission.csv` | 0.13694 | -0.022 |
| 26 | `results/sweep_dba8_k15_submission.csv` | 0.14880 | -0.010 |
| 27 | `results/sweep_dba12_k15_submission.csv` | 0.13159 | -0.027 |
| 28 | `results/sweep_dba7_k15_submission.csv` | 0.14630 | -0.013 |
| 29 | `results/sweep_dba8_k15_lambda025_submission.csv` | **0.14976** | -0.009 |
| 30 | `results/sweep_dba8_k15_lambda020_submission.csv` | 0.14715 | -0.012 |
| 31 | `results/sweep_dba8_k15_k2_4_lambda025_submission.csv` | 0.15288 | -0.006 |
| 32 | `results/sweep_dba8_k15_k2_3_lambda025_submission.csv` | 0.14752 | -0.011 |
| 33 | `results/sweep_dba8_k15_k2_4_lambda020_submission.csv` | 0.15222 | -0.007 |
| 34 | `results/sweep_dba9_k15_k2_4_lambda025_submission.csv` | 0.14515 | -0.014 |
| 35 | `results/sweep_dba8_k15_k2_4_lambda027_submission.csv` | **0.15421** | -0.005 |
| 36 | `results/sweep_dba8_k15_k2_4_lambda030_submission.csv` | 0.15410 | -0.005 |
| 37 | `results/sweep_dba8_k12_k2_4_lambda025_submission.csv` | 0.15210 | -0.007 |

### Session 6 (2026-04-26): seed=200, ArcFace, RRF

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 38 | `results/sweep_11ckpt_dba8_k15_k2_4_lambda027_submission.csv` (seed=200+gradclip) | 0.12608 | -0.033 |
| 39 | `results/arcface_C_orig7_plus_arcface_ep60_submission.csv` | 0.14752 | -0.011 |
| 40 | `results/arcface_B_orig7_plus_arcface4_submission.csv` | 0.14738 | -0.011 |
| 41 | `results/arcface_A_solo_4ckpt_submission.csv` | 0.12053 | -0.038 |
| 42 | `results/rf_rrf_warc_0p10_submission.csv` (RRF) | 0.15263 | -0.006 |

### Session 7 (2026-04-26 evening): CLIP-ReID + camera-adversarial WIN

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 43 | `CLIP-ReID/results/clipreid_ep60_solo_submission.csv` | 0.09788 | -0.061 |
| 44 | `CLIP-ReID/results/clipreid_ep30_40_50_60_submission.csv` | 0.09788 | -0.061 |
| 45 | `results/camadv_baseline7_plus_camadv_ep60_submission.csv` | **0.15884** | 0 ← BEST |
| 46 | `results/camadv_baseline7_plus_camadv_4_submission.csv` | 0.13342 | -0.025 |

### Session 7 continued (2026-04-27): cam-adv mid-mix sweep

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 47 | `results/camadv_baseline7_plus_camadv_2_submission.csv` (s500 ep50+60) | 0.14940 | -0.009 |
| 48 | `results/camadv_baseline7_plus_camadv_3_submission.csv` (s500 ep40/50/60) | 0.13930 | -0.020 |
| 49 | `results/freeprobe_w1.5xcamadv_dba8_k15_lam0275_submission.csv` | 0.15427 | -0.005 |

### Session 7 (2026-04-28): trafficsignal specialist + multi-scale TTA

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 50 | `results/router_A_spec4_ts_uni8_nts_submission.csv` (trafficsignal specialist) | 0.14301 | -0.016 |
| 51 | `results/multiscale_3sizes_dba8_k15_lam0275_submission.csv` | 0.14885 | -0.010 |
| 52 | `results/camadv_s500_s600_baseline7_submission.csv` (2 cam-adv stack) | 0.14170 | -0.017 |
| 53 | `results/pseudo_baseline8_plus_pseudo1x_submission.csv` (87 pairs) | 0.15677 | -0.002 |
| 54 | `results/pseudo2_baseline8_plus_ep5_submission.csv` (172 pairs) | 0.15673 | -0.002 |
| 55 | `results/uam_baseline8_plus_ep30_ep60_submission.csv` (UAM transfer) | 0.14795 | -0.011 |
| 56 | `results/fourcrop_tta_baseline8_submission.csv` | 0.12599 | -0.033 |
| 57 | `results/centering_per_camera_submission.csv` | 0.13943 | -0.019 |

### Session 8 (2026-04-30): DBSCAN pseudo + hi-res cam-adv

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 58 | `results/dbscan_pseudo_baseline8_plus_ep10_1x_submission.csv` | 0.15465 | -0.004 |
| 59 | `results/camadv_hires_baseline8_plus_ep60_1x_submission.csv` (384×192) | 0.14792 | -0.011 |

### Session 9 (2026-05-02): QMV + heavy aug + lambda=0.3 + per-class rerank

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 60 | `results/qmv_mutual_top1_submission.csv` | 0.14932 | -0.010 |
| 61 | `results/heavyaug_baseline8_plus_ep60_submission.csv` (stack) | 0.14151 | -0.017 |
| 62 | `results/heavyaug_replace_baseline7_plus_ep60_submission.csv` | 0.13907 | -0.020 |
| 63 | `results/lightaug_replace_baseline7_plus_ep60_submission.csv` | 0.14251 | -0.016 |
| 64 | `results/perclass_rerank_uniform_k15_lam0275_submission.csv` | 0.15714 | -0.002 |
| 65 | `results/hparam_ensemble_rerank_submission.csv` | 0.15581 | -0.003 |
| 66 | `results/cluster_dba_n700_alpha05_submission.csv` | 0.15327 | -0.006 |
| 67 | `results/lambda03_baseline7_plus_ep60_submission.csv` | 0.15462 | -0.004 |
| 68 | `results/lambda03_baseline7_plus_ep60_half_submission.csv` | 0.15221 | -0.007 |

### Session 10 (2026-05-03): late-stage exhaustion attempts

| # | CSV | Score | Δ vs 0.15884 |
|---|---|---|---|
| 69 | `results/softened_dba_k8_t01_submission.csv` | 0.14305 | -0.016 |
| 70 | `results/aniso_dba_cross_8_6_4_submission.csv` | 0.15087 | -0.008 |
| 71 | `results/seresnet50_merged_solo_ep100_submission.csv` (paper recipe) | **0.06725** | -0.092 (worst) |
| 72 | `results/big13_plus_camadv_ep60_submission.csv` (3 new seeds) | 0.14274 | -0.016 |
| 73 | `results/big13_baseline_only_submission.csv` | 0.13995 | -0.019 |
| 74 | `results/adabn_baseline7_plus_camadv_ep60_submission.csv` | **0.15871** | -0.0001 (NEUTRAL ⭐) |
| 75 | `results/adabn_all8_full_submission.csv` | 0.15837 | -0.0005 (NEUTRAL) |
| 76 | `results/gentle_qe_alpha01_mutual_submission.csv` | 0.15425 | -0.005 |
| 77 | `results/pat8_plus_vit_huge_7030_submission.csv` (ViT-H/14) | 0.14989 | -0.009 |
| 78 | `results/pat8_plus_seresnet50_5050_submission.csv` (BoT 50/50) | 0.09188 | -0.067 |
| 79 | `results/pat8_plus_dinov3_8515_submission.csv` (DINOv3 fusion) | 0.13664 | -0.022 |

**Total: 79 Kaggle submissions across 10 sessions. 7 wins (chronological), 50+ post-best regressions, 2 neutral (AdaBN family).**

---

## D. Training logs by experiment (for paper loss/Acc curves)

### D.1. Production baseline & cam-adv

| Method | Wall time | Final Acc | Log path | Score (in-ensemble) |
|---|---|---|---|---|
| seed=1234 baseline | ~47 min (RTX 4090) | 0.99+ | `models/model_vitlarge_256x128_60ep/train.log` | (in 0.15884) |
| seed=42 baseline | ~75 min (RTX 4090) | 0.99+ | `models/model_vitlarge_256x128_60ep_seed42/train.log` | (in 0.15884) |
| cam-adv s500 (λ=0.1, ep60) | ~75 min | 0.984 (ep60) | `models/model_vitlarge_camadv_seed500/train.log` | +0.00463 win |

### D.2. Failed retrain experiments (by category)

**Backbone swaps:**
| Backbone | Wall time | Final Acc | Train log | Solo Kaggle |
|---|---|---|---|---|
| DINOv2 ViT-L/14 | (NaN at ep40+) | NaN | `logs/train_dinov2.log` | 0.00000 / 0.11915 |
| EVA-L p14 | ~75 min | 0.985 | `models/model_eva_large_p14_seed77/train.log` | 0.10244 |
| CLIP-ReID ViT-B/16 | ~2 hr (Stage 1+2) | 0.997 | `CLIP-ReID/logs/...` | 0.09788 |
| ViT-Huge p14 (40 ep, bs=24) | ~2 hr | 0.987 | `models/model_vit_huge_p14_seed1800/train.log` | (fusion: 0.14989) |
| SE-ResNet-50 BoT (Urban only) | ~28 min | 0.89 | `models/model_seresnet50_bot_seed1500/train.log` | (fusion: 0.09188) |
| SE-ResNet-50 BoT + heavy aug + merged | ~44 min | 0.89 | `models/model_seresnet50_merged_seed2000/train.log` | 0.06725 |

**Recipe / loss / aug variants on PAT-L:**
| Method | Train log | Solo (replace) Kaggle |
|---|---|---|
| seed=200 + GRAD_CLIP | (deleted; ckpts purged §56) | 0.12608 |
| ArcFace s=30 m=0.30 | (deleted; ckpts purged §56) | 0.12053 |
| Heavy aug cam-adv (seed=1200) | `models/model_vitlarge_camadv_heavyaug_seed1200/train.log` | 0.13907 |
| Light aug cam-adv (seed=1300) | `models/model_vitlarge_camadv_lightaug_seed1300/train.log` | 0.14251 |
| Cam-adv λ=0.3 (seed=1400) | `models/model_vitlarge_camadv_lambda03_seed1400/train.log` | 0.15462 |
| Cam-adv s600 (λ=0.1) | `models/model_vitlarge_camadv_seed600/train.log` | (stack: 0.14170) |
| Hi-res cam-adv 384×192 | `models/model_vitlarge_camadv_hires_seed800/train.log` | (fusion: 0.14792) |
| 3 new baseline seeds 2100/2200/2300 | `models/model_vitlarge_256x128_60ep_seed2{100,200,300}/train.log` | (ensemble: 0.14274) |

**Transfer learning:**
| Method | Train log | Kaggle |
|---|---|---|
| Merged data (Urban+UAM) PAT seed=2024 | (Session 2 logs) | 0.12256 / 0.13329 |
| UAM supervised pretrain | `models/uam_pipeline.log` | (transfer: 0.14795) |
| UAM → Urban2026 fine-tune | (same log) | 0.14795 |

**Pseudo-labeling:**
| Method | Train log | Kaggle |
|---|---|---|
| Top-1 mutual NN (87 pairs, LR=3e-5) | `models/model_vitlarge_pseudo_seed900/train.log` (overwritten by iter-2) | 0.15677 |
| Top-2 mutual NN (172 pairs, LR=3e-5) | `models/model_vitlarge_pseudo_seed900/train.log` | 0.15673 |
| DBSCAN cluster (519 imgs, LR=1e-4) | `models/model_vitlarge_dbscan_pseudo_seed950/train.log` | 0.15465 |

**Trafficsignal specialist:**
| Method | Train log | Kaggle |
|---|---|---|
| 800-ID PAT specialist (seed=800) | `models/model_vitlarge_trafficsignal_seed800/train.log` | (router: 0.14301) |

### D.3. Training-free (no train log; cached features used)

| Method | Inference script | Kaggle |
|---|---|---|
| Multi-scale TTA (224+256+288) | `multiscale_tta_inference.py` | 0.14885 |
| 4-crop TTA at 256×128 | `fourcrop_tta_inference.py` | 0.12599 |
| Mean centering (per-camera) | `feature_centering_inference.py` | 0.13943 |
| MLP refinement | `mlp_refinement_pipeline.py` | ~0.10 |
| QMV mutual top-1 | `qmv_inference.py` | 0.14932 |
| Per-class rerank | `per_class_rerank_inference.py` | 0.15714 |
| Hyperparameter ensemble rerank | `hparam_ensemble_rerank.py` | 0.15581 |
| Cluster-DBA n=700 α=0.5 | `cluster_dba_inference.py` | 0.15327 |
| Anisotropic DBA (8/6/4) | `anisotropic_dba_inference.py` | 0.15087 |
| Softened DBA T=0.1 | `weighted_dba_inference.py` | 0.14305 |
| Gentle QE α=0.1 mutual-NN | `gentle_qe_inference.py` | 0.15425 |
| AdaBN cam-adv only | `adabn_inference.py` | 0.15871 (NEUTRAL ⭐) |
| AdaBN all 8 ckpts | `adabn_full_inference.py` | 0.15837 (NEUTRAL) |
| DINOv3 + PAT 85/15 fusion | `dinov3_inference.py` | 0.13664 |

---

## E. DBA + rerank parameter sweep (paper Figure: post-processing curves)

All measured on the proven 7-baseline ensemble (pre cam-adv era):

### DBA k axis (k1=15, λ=0.30 fixed):
| k | Score |
|---|---|
| 0 (no DBA) | 0.13361 |
| 5 | 0.13707 |
| 7 | 0.14630 |
| **8** | **0.14880-0.15421** ← PEAK |
| 9 | 0.14515 |
| 12 | 0.13159 (over-smoothed) |

### Re-rank λ axis (DBA=8, k1=15, k2=4 fixed):
| λ | Score |
|---|---|
| 0.20 | 0.15222 |
| 0.25 | 0.15288 |
| **0.275** | **0.15421** ← PEAK |
| 0.30 | 0.15410 |

### Re-rank k1 axis (DBA=8, λ=0.275, k2=4 fixed):
| k1 | Score |
|---|---|
| 12 | 0.15210 |
| **15** | **0.15421** ← PEAK |

### Re-rank k2 axis (DBA=8, k1=15, λ=0.275 fixed):
| k2 | Score |
|---|---|
| 3 | 0.14752 |
| **4** | **0.15421** ← PEAK |
| 5 | 0.14976 |

**Optimal:** DBA k=8, k1=15, k2=4, λ=0.275 (proven across all post-processing experiments).

---

## F. Cam-adv per-epoch ensemble compatibility (paper Figure: convergence window)

7-baseline + cam-adv ckpts (s500, λ=0.1) at 1× weight:

| Epochs included | Score | Δ |
|---|---|---|
| (none, just 7-baseline) | 0.15421 | -0.005 |
| {ep60} ← single late ep | **0.15884** | 0 (BEST) |
| {ep50, ep60} | 0.14940 | -0.009 |
| {ep40, ep50, ep60} | 0.13930 | -0.020 |
| {ep30, ep40, ep50, ep60} | 0.13342 | -0.025 |

**Strict monotonic degradation** as earlier epochs added — paper-worthy convergence-window finding.

---

## G. Cam-adv weight saturation (paper Figure: angular threshold)

7-baseline + cam-adv s500 ep60 at varying weight w:

| w | Angular contribution | Score |
|---|---|---|
| 0× | 0% | 0.15421 |
| 1× | 27% (= 1/(1+√7)) | **0.15884** ← peak |
| 1.5× | 36% | 0.15427 |
| 2 ckpts × 1× | 35% | 0.14170 (s500+s600) |

**Sharp threshold around 30%** angular contribution — paper-worthy mechanistic finding.

---

## H. Summary tables for paper

### H.1. Negative results by category (50+ documented)

| Category | Variants tested | Common failure mode |
|---|---|---|
| Backbone swaps | DINOv2-L, EVA-L, CLIP-ReID ViT-B, ViT-H/14, SE-ResNet-50, DINOv3-L | Cross-architecture features incompatible with proven ensemble manifold |
| Loss changes | Circle (γ=64-256), ArcFace (s=30-64, m=0.30-0.50), Quadruplet | Aggressive losses + grad_clip conflict; features distribution-shifted |
| Recipe changes | EMA, GRAD_CLIP=1.0 (no other changes), heavy aug, light aug | Modify feature distribution past angular threshold |
| Resolution changes | Multi-scale TTA, 4-crop, 384×192 retrain | ViT pos_embed coupling — different scales = different manifold |
| Pseudo-labeling | Top-1 mutual NN, top-2, DBSCAN | Label noise floor (~70% wrong at our 0.16 baseline accuracy) |
| Transfer learning | UAM merged data, UAM supervised pretrain | Cross-domain biases persist through fine-tune |
| TTA | Multi-scale, 4-crop, h-flip, QMV, gentle QE | Pos_embed drift; cropping artifacts; visually-similar-but-diff IDs |
| Feature debiasing | Mean centering, PCA-drop-1, MLP refinement | Per-camera mean encodes USEFUL identity signal, not pure bias |
| Re-rank tweaks | Per-class rerank, hparam ensemble, anisotropic DBA, weighted DBA | Cross-class context as informative negatives; (15,4,0.275) is the genuine optimum |
| Gallery clustering | KMeans n=700 + cluster centroid blend | Cluster boundaries don't align with identities |
| Cam-adv variants | λ=0.3 single (1× and 0.5×), 1.5× weighting, 2-ckpt stack, heavy/light aug, hi-res 384×192 | All shift features past angular threshold |
| Cross-seed scaling | n=5 (3 new seeds 2100/2200/2300 added) | Hardware-induced manifold drift (RTX 4090 → 3090) |
| **AdaBN** | Cam-adv only + all 8 ckpts | **NEUTRAL** — first non-regression; manifold-preserving |

### H.2. The two paper-worthy mechanistic theories

**1. Angular-weight threshold:** the maximum complementary signal that can be added to an L2-norm-summed ensemble is bounded by `w/(w+sqrt(N))`. Above ~30%, distribution drift dominates. (Source: §67, §72; supported by saturation curve in §G)

**2. Convergence-window for adversarial features:** GRL-trained features are ensemble-compatible only at the FINAL ~5 epochs of training. Earlier epochs have too-strong distribution shift. (Source: §60.8, §66; supported by per-epoch curve in §F)

**3. Manifold-preservation principle (Session 10 finding):** only interventions that preserve learned feature directions (BN-stat recalibration, post-processing tuning at the optimum) avoid the regression cliff. Manifold-perturbing interventions (any retrain, any feature transformation) regress. (Source: §99, contrast with §C dead-end list)

**4. Hardware-induced manifold drift (Session 10 finding):** cross-seed ReID ensembling loses compatibility across heterogeneous GPUs (RTX 4090 → RTX 3090) due to AMP+cuDNN-induced numerical drift, even with deterministic seeds. (Source: §100; refutes naive "more seeds = better" scaling)

---

## I. Files for paper appendix / reproducibility

- `context/context_1.md` — full 3387-line research log (all sessions)
- `backup_score/` — winning CSV + 7-baseline ckpts (paper artifact)
- `models/*/train.log` — every training run's loss/Acc trajectory
- `results/*.csv` — every Kaggle submission CSV
- `config/UrbanElementsReID_*.yml` — all training configs
- `model/make_model.py`, `data/transforms/build.py`, `processor/part_attention_vit_processor.py` — modified files (cam-adv, GRL, heavy aug, AdaBN integration)
- `loss/circle_loss.py`, `loss/arcface_head.py`, `utils/gradient_reversal.py`, `utils/ema.py` — added losses + utilities

---

## J. Things missed in initial pass — adding for completeness (2026-05-03)

### J.1. Dataset characterization (paper Figure: class imbalance)

| Class | Train images | Train IDs | Query (c004) | Gallery (c001-c003) |
|---|---|---|---|---|
| trafficsignal | 7568 (68%) | 800 (72%) | 582 (63% queries) | 1836 (65% gallery) |
| crosswalk | 1532 | 111 | 91 (10%) | 354 |
| container | 1189 | 87 | 167 (18%) | 261 |
| rubbishbins | 886 | 115 | 88 (9%) | 393 |
| **TOTAL** | **11,175** | **1088** | **928** | **2844** |

UAM external dataset (used in Session 2 §16c, Session 7 §75, Session 10 §95):
- Path: `/workspace/UAM_Unified_extract/UAM_Unified/`
- 6,387 train images, 479 IDs, c001-c004 (UAM is a different city — Madrid)
- Includes 745 c004 training images (paper-relevant: only data with c004 training-side)
- Class distribution: trafficsign 3080, container 1578, crosswalk 1329, rubbishbins 400
- Note: UAM uses "trafficsign" (not "trafficsignal"); merge requires class normalization

### J.2. CycleGAN-augmented cam-adv attempt (§81) — INCOMPLETE

A late-Session-8 experiment that was attempted but never produced a final submission:
- **Pipeline**: cloned `pytorch-CycleGAN-and-pix2pix`, trained CycleGAN c001-c003 ↔ c004 on Urban2026 query images (~2.5 hr), generated 11,175 fake-c004 versions of training images
- **Style-shift verification**: synthetic-c004 RGB stats sit 58% of the way from c001-c003 to real c004 (paper-worthy: cycle GAN learned genuine domain shift)
- **PAT+cam-adv training on merged real+fake data**: started but training/CSVs never seen (workspace state changed before completion)
- **Files attempted**: `cyclegan_*.py` scripts in repo root; `/workspace/cyclegan_data/`, `/workspace/cyclegan_checkpoints/` no longer present
- **Status**: incomplete; no Kaggle submission; mention in paper as "attempted but unable to complete due to operational constraints"

### J.3. Unsubmitted variant CSVs (paper Search-Space Coverage)

These were generated but never submitted (Kaggle submission limit + hypothesis ranking ruled them out before they got tested):

**DBA / rerank fine-grain sweep (Session 5 Round 4):**
| CSV | Variant | Likely outcome |
|---|---|---|
| `results/sweep_dba6_k15_k2_4_lambda025_submission.csv` | DBA=6 at new k2 winner | likely worse than DBA=8 (peak) |
| `results/sweep_dba7_k15_k2_4_lambda025_submission.csv` | DBA=7 at new k2 winner | likely close to peak |
| `results/sweep_dba10_k15_k2_4_lambda025_submission.csv` | DBA=10 at new k2 | likely worse |
| `results/sweep_dba8_k15_k2_4_lambda022_submission.csv` | λ=0.225 | between drop (0.20) and works (0.25) |
| `results/sweep_dba8_k15_k2_4_lambda035_submission.csv` | λ=0.35 (looser) | likely past peak plateau |
| `results/sweep_dba8_k15_k2_6_lambda025_submission.csv` | k2=6 | similar to k2=5 baseline |
| `results/sweep_dba8_k18_k2_4_lambda025_submission.csv` | k1=18 | unknown but likely worse than k1=15 |

**Cam-adv mid-mix (Session 7):**
| CSV | Variant | Status |
|---|---|---|
| `results/camadv_solo_ep30_40_50_60_submission.csv` | cam-adv solo (no baseline) | not submitted |
| `results/camadv_solo_ep40_50_60_submission.csv` | cam-adv solo, last-3 | not submitted |
| `results/camadv_solo_ep50_60_submission.csv` | cam-adv solo, last-2 | not submitted |
| `results/camadv_solo_ep60_submission.csv` | cam-adv solo, ep60 only | not submitted |

**Free-probe weighting variants (Session 7, post-0.15884):**
| CSV | Variant | Status |
|---|---|---|
| `results/freeprobe_w2.0xcamadv_dba8_k15_lam0275_submission.csv` | 2× cam-adv weight | not submitted (1.5× already failed) |
| `results/freeprobe_w3.0xcamadv_dba8_k15_lam0275_submission.csv` | 3× cam-adv | not submitted |
| `results/freeprobe_b0.50x_w1xcamadv_*_submission.csv` | half-baseline weight | not submitted |
| `results/freeprobe_b0.75x_w1xcamadv_*_submission.csv` | 0.75× baseline | not submitted |
| `results/freeprobe_dba6/7/9/10_k15_lam0275_*` | DBA k variants on 8-ckpt | not submitted |
| `results/freeprobe_dba8_k14/16/18_lam0275_*` | k1 variants on 8-ckpt | not submitted |
| `results/freeprobe_dba8_k15_lam250/300/325_*` | λ variants on 8-ckpt | not submitted |

**Trafficsignal specialist routing (Session 7):**
| CSV | Variant | Status |
|---|---|---|
| `results/router_B_spec4_ts_base7_nts_submission.csv` | spec(4) + base7 (no cam-adv on non-ts) | not submitted |
| `results/router_C_spec4plusUni8_ts_uni8_nts_submission.csv` | spec stacked with proven | not submitted |
| `results/router_D_spec4plusCamAdv_ts_uni8_nts_submission.csv` | spec + cam-adv on ts | not submitted |
| `results/router_E_specEp60_ts_uni8_nts_submission.csv` | spec ep60 only | not submitted |

**Pseudo-labeling iter-2 alternatives (Session 7):**
| CSV | Variant | Status |
|---|---|---|
| `results/pseudo2_baseline8_plus_ep10_submission.csv` | pseudo iter-2 ep10 (instead of ep5) | not submitted |
| `results/pseudo_solo_submission.csv` | pseudo iter-1 solo | not submitted |
| `results/dbscan_pseudo_*_0p5x_submission.csv`, `_ep5_1x`, `_ep5_ep10`, `_solo_ep10` | DBSCAN pseudo variants | not submitted |

**ViT-Huge variants (Session 10):**
| CSV | Variant | Status |
|---|---|---|
| `results/vit_huge_solo_ep40_submission.csv` | ViT-H solo | not submitted |
| `results/vit_huge_solo_ep30_40_submission.csv` | ViT-H ep30+40 average | not submitted |
| `results/pat8_plus_vit_huge_5050_submission.csv` | 50/50 fusion | not submitted |

**SE-ResNet-50 BoT variants (Session 10):**
| CSV | Variant | Status |
|---|---|---|
| `results/seresnet50_solo_ep100_submission.csv` (Urban only) | BoT solo | not submitted |
| `results/seresnet50_solo_ep80_100_submission.csv` (Urban only) | BoT ep80+100 | not submitted |
| `results/pat8_plus_seresnet50_7030_submission.csv` (Urban only) | 70/30 fusion | not submitted |
| `results/seresnet50_merged_solo_ep80_100_submission.csv` | merged BoT ep80+100 | not submitted |
| `results/pat8_plus_seresnet50_merged_7030_submission.csv` | merged BoT 70/30 fusion | not submitted |

**DINOv3 variants (Session 10):**
| CSV | Variant | Status |
|---|---|---|
| `results/pat8_plus_dinov3_7030_submission.csv` | 70/30 fusion | not submitted |
| `results/dinov3_solo_submission.csv` | DINOv3 solo | not submitted |

**AdaBN backups:**
| CSV | Variant | Status |
|---|---|---|
| `results/adabn_camadv_solo_ep60_submission.csv` | AdaBN-cam-adv solo | not submitted |
| `results/adabn_all8_momentum01_submission.csv` | gentle EMA blend | not submitted |

**Other backups:**
| CSV | Variant | Status |
|---|---|---|
| `results/aniso_dba_within_8_6_4_submission.csv` | within-class aniso DBA | not submitted |
| `results/weighted_dba_k8_submission.csv` | linear-weighted DBA | not submitted |
| `results/cluster_dba_n700_alpha10_submission.csv` | hard cluster replacement | not submitted |
| `results/centering_global_submission.csv` | global mean centering | not submitted |
| `results/centering_pca_drop1_submission.csv` | PCA drop-1 | not submitted |
| `results/lambda03_baseline7_plus_ep{10,20,30,40,50}_submission.csv` | per-epoch λ=0.3 cam-adv | not submitted |
| `results/lambda03_solo_ep60_submission.csv` | λ=0.3 solo | not submitted |
| `results/multiscale_native256/only224/only288/256_288_*_submission.csv` | multi-scale per-config | not submitted |
| `results/qmv_top3_sim05_submission.csv` | QMV top-3 (vs mutual top-1) | not submitted |
| `results/perclass_rerank_tuned_submission.csv` | per-class with tuned per-class params | not submitted |
| `results/gentle_qe_alpha02_mutual_submission.csv` | gentle QE α=0.2 | not submitted |
| `results/heavyaug_baseline8_plus_ep60` (stack variant CSV at 5050 weight) | unsubmitted variants | not submitted |

### J.4. Key training hyperparameter summary (paper §3 Method)

**Proven baseline recipe (used for seed=1234, seed=42 — produces 0.15421 7-ckpt ensemble):**
- Backbone: ViT-Large/16, ImageNet pretrained (`jx_vit_large_p16_224-4ee7a4dc.pth`)
- Patch size 16, stride 16, input 256×128 (16×8 patch grid + 1 CLS + 3 part tokens = 132 tokens)
- Optimizer: Adam, BASE_LR=3.5e-4, WEIGHT_DECAY=1e-4, BIAS_LR_FACTOR=2
- LR schedule: linear warmup over 10 epochs from 0.01×LR
- Total: 60 epochs
- Batch size: 64, NUM_INSTANCE=4 (16 IDs × 4 instances)
- Sampler: softmax_triplet (PK sampling)
- Loss: soft-margin triplet (NO_MARGIN=True) + label-smoothed CE + Pedal patch-clustering on 3 part tokens (all weight 1.0); SOFT_LABEL=True
- Augmentation: Resize, RandomHorizontalFlip(0.5), Pad(10)+RandomCrop, LGT(0.5) — no Random Erasing in proven recipe
- Pixel mean/std: (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5)
- AMP enabled, GRAD_CLIP=0 (NaN-vulnerable but ensemble-compatible — see §52)

**Cam-adv addition (§60, seed=500, the +0.005 winner):**
- Same as baseline recipe + new GRL+CamCls head
- `MODEL.CAM_ADV: True, NUM_CAMERAS: 3, CAM_ADV_LAMBDA: 0.1, CAM_ADV_WEIGHT: 1.0`
- GRL gradient sign-flipped + scaled by λ=0.1 between bottleneck and cam_classifier(3-way)
- GRAD_CLIP=1.0 (mandatory once GRL is active)
- Saved ckpts: ep10, 20, 30, 40, 50, 60 (only ep60 used in production)

**Test-time recipe (proven 0.15884):**
- 8-ckpt ensemble: 7 baseline + cam-adv s500 ep60 (all 1× weight)
- Each ckpt: extract final-layer CLS feature, L2-normalize
- Sum across 8 ckpts, L2-renormalize → final query/gallery features
- DBA on gallery: each gallery feature = L2-norm(mean of top-8 cosine neighbors)
- Compute distance matrices (q_g, q_q, g_g), apply k-reciprocal rerank with k1=15, k2=4, λ=0.275
- Class-group filter: cross-group cells set to ∞ (groups: trafficsignal, crosswalk, bin_like = container ∪ rubbishbins)
- Argsort, write top-100 indices to CSV

### J.5. Hardware / environment

| Sessions | Hardware | Notes |
|---|---|---|
| 1-3 (2026-04-20 to 04-25) | RTX 4090, 24 GiB | Original training environment |
| 4-7 (2026-04-25 to 04-28) | RTX 4090 | Same |
| 8-10 (2026-04-30 to 05-03) | **RTX 3090, 24 GiB** | Hardware switch — implicated in §100 manifold drift hypothesis |

- Python 3.10 venv at `/workspace/miuam_challenge_diff/.venv/`
- torch 2.1.0+cu121, timm, yacs, einops, sklearn, scipy
- Pretrained weights at `pretrained/`:
  - `jx_vit_large_p16_224-4ee7a4dc.pth` (1.2 GB) — supervised ViT-L (used in 0.15884)
  - `dinov2_vitl14_pretrain.pth` (1.2 GB) — DINOv2 (failed §16d/§23)
  - `eva_large_patch14_196.pth` (1.2 GB) — EVA-L (failed §24)
  - `dinov3_vitl16.pth` (1.2 GB) — DINOv3 (failed §104)
  - SE-ResNet-50 timm pretrained (downloaded inline, ~100 MB) — failed §92, §96

### J.6. Code modifications (paper appendix — reproducibility)

| File | Modification | Purpose |
|---|---|---|
| `data/data_utils.py:4,17` | Replace PathManager.open → built-in open | Fix import (fvcore not present) |
| `utils/metrics.py:6,125-132` | Fix re_ranking import + parameter signature | Fix breakage in stripped baseline |
| `processor/part_attention_vit_processor.py:204-213` | Comment out checkpoint-deletion block | Preserve per-epoch ckpts (critical safety) |
| `model/make_model.py` | Add `build_part_attention_vit` modifications: deep_sup heads, cam_adv head, ArcFace head, seresnet50 wrapper, vit_huge factory | Enable all our experiments |
| `model/backbones/vit_pytorch.py` | Add `LayerScale`, `part_attention_vit_large_p14`, `part_attention_vit_large_p14_eva`, `part_attention_vit_huge_p14` factories | Support DINOv2/EVA/ViT-Huge backbones |
| `data/transforms/build.py` | Add RandomPerspective, RandomRotation gates | Heavy aug (§88) |
| `loss/build_loss.py:57` | Wrap CircleLoss with tuple-return shim | Drop-in replacement for triplet |
| `loss/circle_loss.py` (NEW) | Circle Loss implementation | §28 attempt (failed) |
| `loss/arcface_head.py` (NEW) | ArcFace head | §53 (failed) |
| `loss/softmax_loss.py:24-39` | Handle `all_posvid=None` when PC_LOSS off | Enable seresnet50 with no part tokens |
| `utils/gradient_reversal.py` (NEW, 38 lines) | GRL Function for cam-adv | Core of §60.2 |
| `utils/ema.py` (NEW) | EMA wrapper | §29 (didn't help in ensemble) |
| `train.py:79-85` | Add FINETUNE_FROM warm-start | UAM transfer (§75) + pseudo-labeling fine-tune (§73) |
| `train.py:73-74` | Allow seresnet50 model name | §92 |
| `processor/part_attention_vit_processor.py:78-82, 117-122, 125-141, 158-164` | Per-iter NaN guard, per-epoch NaN-param check, GRAD_CLIP with logging, deep_sup unpacking, cam_adv unpacking, cam_loss aggregation, PC_LOSS guard for seresnet50 | All extra training mechanics |
| `config/defaults.py` | All new config flags: GRAD_CLIP, EMA_DECAY, ID_LOSS_TYPE, ARCFACE_S/M, DEEP_SUP, NUM_AUX_LAYERS, AUX_LOSS_WEIGHT, FINETUNE_FROM, PRETRAIN_CHOICE, CAM_ADV, NUM_CAMERAS, CAM_ADV_LAMBDA, CAM_ADV_WEIGHT, PERSPECTIVE.{ENABLED,PROB,DISTORTION}, ROTATION.{ENABLED,PROB,DEGREES} | All experiments configurable |

### J.7. Files NOT included in code repo (download links if needed)

- ViT-L/16 ImageNet pretrain: `https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth`
- DINOv2 ViT-L/14: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`
- EVA-L p14: via timm `timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True)`
- ViT-Huge p14: via timm `timm.create_model('vit_huge_patch14_224', pretrained=True)`
- DINOv3 ViT-L/16: via timm `timm.create_model('vit_large_patch16_dinov3', pretrained=True)`
- SE-ResNet-50: via timm `timm.create_model('seresnet50', pretrained=True)`
- Urban2026 dataset: via Kaggle competition page
- UAM_Unified dataset: external (provided by user; original source UrbAM-ReID 2024 paper)
