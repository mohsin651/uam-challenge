# paper_context/ — INVENTORY

Generated 2026-05-03 for the URVAM-ReID 2026 arXiv paper drafting session.

All paths below are absolute, rooted at `/workspace/miuam_challenge_diff/paper_context/`.
Original files were **copied, not moved**. Sources are preserved in-place under
`/workspace/miuam_challenge_diff/`.

## Top-level summary

| Folder | Files | Size | Description |
|---|---|---|---|
| `logs/`         | 49 (top-level + per_run + inference)  | 2.6 MB  | Training stdout/stderr + per-run train/test logs + inference logs |
| `tensorboard/`  | 39 event files in 29 run dirs         | 652 KB  | TensorBoard `events.out.tfevents.*` per training run |
| `csvs/`         | 201                                   | 85 MB   | Kaggle submission CSVs (200) + 1 pseudo-pair table |
| `notebooks/`    | 0                                     | 0       | **NONE FOUND** — no `.ipynb` exists in the repo (analysis was script-driven) |
| `configs/`      | 37                                    | 152 KB  | YAML training/test configs |
| `code/`         | 104 files (48 root scripts + modules) | 816 KB  | All inference/training/post-processing scripts + module dirs |
| `notes/`        | 8                                     | 304 KB  | Markdown notes: CLAUDE.md, RESUME.md, README.md, context_1.md, score.md, backup_score notes |
| **TOTAL**       | **~437 files**                        | **89 MB** | |

---

## 1. `logs/` — 22 top-level + 4 inference + per_run/22 dirs

### 1a. `logs/` (top-level — 18 files, ~830 KB)
Training stdout from `nohup`/tmux runs. Each is a single training run's full console output.

| File | Bytes | What experiment |
|---|---|---|
| `train.log` | 31,858 | Original baseline (seed=1234, ViT-L/16 PAT, 60 ep) |
| `train_seed42.log` | 35,174 | Cross-seed retrain (seed=42) — partner for the 0.13361 ensemble |
| `train_deepsup.log` | 35,344 | Deep-supervision experiment (DEEP_SUP=True, aux=0.1) — failed (0.10788) |
| `train_heavyaug.log` | 18,711 | Heavy-aug fine-tune from seed=42 ep60 (CJ + REA) — failed (0.12811) |
| `train_merged.log` | 53,477 | UAM-merged training (seed=2024, 17,562 imgs) — negative transfer (0.12256) |
| `train_dinov2.log` | 35,275 | DINOv2-L p14 retry (NaN blowup at ep40+) — 0.00000 / 0.11915 ep30 |
| `_eva_smoke.log` | 11,822 | EVA-L 2-epoch GPU smoke test (gradient clipping verification) |
| `train_eva.log` | 45,766 | EVA-L full training (seed=77) — failed (0.10244 solo) |
| `train_seed100_ema.log` | 35,601 | Seed=100 + EMA — solo 0.11739, ensemble dragged down |
| `train_seed100_ema_circle.log` | 24,791 | Circle Loss attempt (γ=256 + clip=1.0) — Acc stuck at 0.34, killed |
| `train_seed200.log` | 45,753 | Plain triplet seed=200 — encountered NaN at ep38 |
| `train_arcface.log` | 46,094 | ArcFace head retrain (seed=300) — see `arcface_*` CSVs for results |
| `cyclegan_train.log` | 94,290 | CycleGAN training (c001-3 ↔ c004 stylization) for fake-c4 augmentation |
| `train_camadv_cyclegan.log` | 73,238 | Cam-adversarial + CycleGAN-augmented training (seed=1100) |
| `train_camadv_hires.log` | 89,477 | Cam-adversarial hi-res 384×192 (seed=800) |
| `train_dbscan_pseudo.log` | 16,003 | DBSCAN pseudo-label experiment (seed=950) |
| `uam_pipeline.log` | 76,008 | UAM-pretrain → after-UAM fine-tune chain (seeds 1000, 1100) |
| `seeds_chain_orchestrator.log` | 104,489 | Orchestration log for the seed=2100/2200/2300 chained baseline retrains |

### 1b. `logs/inference/` — 4 files
| File | Bytes | Inference run |
|---|---|---|
| `camadv_inference.log` | 8,782 | Cam-adversarial ensemble inference |
| `freeprobes.log` | 7,541 | `camadv_freeprobes.py` — DBA/k1/λ free-probe sweep |
| `multiscale_tta.log` | 15,142 | `multiscale_tta_inference.py` — multi-scale TTA |
| `router_inference.log` | 10,288 | `trafficsignal_router_inference.py` — class-conditional routing |

### 1c. `logs/per_run/<run_name>/` — 22 dirs, 38 files total
For each model folder under `models/`, copied:
- `train.log` (stdout, when present)
- `train_log.txt` (Python logger structured log — has loss/Acc/iter timing per step)
- `test_log.txt` (only `model_vitlarge_256x128_60ep` has one)

| Run dir | # files | Notes |
|---|---|---|
| `model_vitlarge_256x128_60ep`             | 2 (train_log + test_log) | Seed=1234 baseline (in 0.13361 ensemble) |
| `model_vitlarge_256x128_60ep_seed42`      | 1 (train_log)            | Seed=42 baseline (in 0.13361 ensemble) |
| `model_vitlarge_camadv_seed500`           | 2 | First cam-adversarial run (λ=0.1) |
| `model_vitlarge_camadv_seed600`           | 2 | Cam-adv variant (different hyperparams) |
| `model_vitlarge_camadv_lambda03_seed1400` | 2 | Cam-adv λ=0.3 variant |
| `model_vitlarge_camadv_lightaug_seed1300` | 2 | Cam-adv + light aug |
| `model_vitlarge_camadv_heavyaug_seed1200` | 2 | Cam-adv + heavy aug |
| `model_vitlarge_camadv_hires_seed800`     | 1 | Cam-adv at 384×192 |
| `model_vitlarge_camadv_cyclegan_seed1100` | 1 | Cam-adv + CycleGAN-fake-c4 augmentation |
| `model_vitlarge_camadv_merged_heavyaug_seed2500` | 2 | Cam-adv + merged data + heavy aug |
| `model_vitlarge_pseudo_seed900`           | 2 | Pseudo-label retrain |
| `model_vitlarge_dbscan_pseudo_seed950`    | 1 | DBSCAN pseudo-label retrain |
| `model_vitlarge_uam_pretrain_seed1000`    | 1 | UAM SSL-style pretrain stage |
| `model_vitlarge_after_uam_seed1100`       | 1 | Fine-tune after UAM pretrain |
| `model_vitlarge_trafficsignal_seed800`    | 2 | Class-specialist (trafficsignal-only training) |
| `model_vitlarge_256x128_60ep_seed1700`    | 2 | Cross-seed baseline #4 |
| `model_vitlarge_256x128_60ep_seed2100`    | 2 | Chained cross-seed baseline (orchestrator run) |
| `model_vitlarge_256x128_60ep_seed2200`    | 2 | Chained cross-seed baseline |
| `model_vitlarge_256x128_60ep_seed2300`    | 2 | Chained cross-seed baseline |
| `model_seresnet50_bot_seed1500`           | 2 | SE-ResNet50 BoT-style ReID (non-transformer baseline) |
| `model_seresnet50_merged_seed2000`        | 2 | SE-ResNet50 on UAM-merged data |
| `model_vit_huge_p14_seed1800`             | 2 | ViT-Huge/14 retrain |

### MISSING / FLAGS for `logs/`
- **No `.log` for `model_vitlarge_256x128_60ep_seed42`** in `logs/per_run/` — only `train_log.txt` (the structured log). The matching stdout log is in `logs/train_seed42.log` at the top level.
- **No top-level stdout log for cam-adv seed=500, seed=600, λ=0.3, lightaug, heavyaug, merged_heavyaug, hires** — they all only have per-run `train.log` files (these runs were probably launched directly from a script that wrote into the model dir).
- **No EMA-only no-Circle stdout** for seed=100 outside `train_seed100_ema.log` (single file is correct here).
- **No `.log` for `model_vitlarge_dinov2_p14_seed42`** in per_run/ — that run dir only kept TensorBoard events (the .log went into `logs/train_dinov2.log`).
- **No EVA `train_log.txt`** in per_run/ — EVA training output went only to `logs/train_eva.log`.

---

## 2. `tensorboard/` — 29 run dirs, 39 event files

Each subfolder is one training run (preserves `tb_log/<run_name>/events.out.tfevents.*` structure).
Some runs have multiple event files because tmux/training was restarted (each restart creates a new event file).

| Run dir | # event files | Size | Experiment + final mAP@100 (from session notes) |
|---|---|---|---|
| `model_vitlarge_256x128_60ep` | 2 | 24K | Seed=1234 baseline — solo ~0.128 |
| `model_vitlarge_256x128_60ep_seed42` | 1 | 20K | Seed=42 baseline — partner in 0.13361 ensemble |
| `model_vitlarge_256x128_60ep_deepsup` | 1 | 20K | Deep-sup aux=0.1 — failed (ensemble: 0.10788) |
| `model_vitlarge_256x128_heavyaug_from_seed42_ep60` | 1 | 8K | Heavy-aug fine-tune — failed (ensemble: 0.12811) |
| `model_vitlarge_256x128_60ep_merged_seed2024` | 1 | 40K | UAM-merged seed=2024 — failed (solo: 0.12256) |
| `model_vitlarge_dinov2_p14_seed42` | 2 | 24K | DINOv2-L (NaN at ep40+) — failed |
| `_eva_smoke` | 1 | 4K | EVA 2-ep smoke (no mAP — sanity check only) |
| `model_eva_large_p14_seed77` | 1 | 20K | EVA-L 60-ep — failed (solo: 0.10244) |
| `model_vitlarge_256x128_60ep_seed100_ema` | 2 | 24K | Seed=100 + EMA — solo 0.11739 |
| `model_vitlarge_256x128_60ep_seed100_ema_circle` | 1 | 12K | Circle Loss — killed early (Acc stuck) |
| `model_vitlarge_256x128_60ep_seed200` | 2 | 32K | Seed=200 (NaN at ep38) — partial |
| `model_vitlarge_256x128_60ep_arcface_seed300` | 2 | 24K | ArcFace seed=300 — see arcface_* CSVs |
| `model_vitlarge_camadv_seed500` | 2 | 24K | Cam-adv first run — see camadv_* CSVs (one variant scored 0.15884) |
| `model_vitlarge_camadv_seed600` | 1 | 20K | Cam-adv seed=600 |
| `model_vitlarge_camadv_lambda03_seed1400` | 1 | 20K | Cam-adv λ=0.3 |
| `model_vitlarge_camadv_lightaug_seed1300` | 1 | 20K | Cam-adv + light aug |
| `model_vitlarge_camadv_heavyaug_seed1200` | 2 | 24K | Cam-adv + heavy aug |
| `model_vitlarge_camadv_merged_heavyaug_seed2500` | 1 | 12K | Cam-adv + merged + heavy aug |
| `model_vitlarge_pseudo_seed900` | 2 | 8K | Pseudo-label retrain |
| `model_vitlarge_uam_pretrain_seed1000` | 1 | 12K | UAM-only pretrain stage |
| `model_vitlarge_after_uam_seed1100` | 1 | 20K | After-UAM fine-tune |
| `model_vitlarge_trafficsignal_seed800` | 1 | 12K | Class-specialist trafficsignal |
| `model_vitlarge_256x128_60ep_seed1700` | 1 | 4K | Cross-seed #4 |
| `model_vitlarge_256x128_60ep_seed2100` | 1 | 20K | Chained baseline (orchestrator) |
| `model_vitlarge_256x128_60ep_seed2200` | 1 | 20K | Chained baseline |
| `model_vitlarge_256x128_60ep_seed2300` | 1 | 20K | Chained baseline |
| `model_seresnet50_bot_seed1500` | 1 | 32K | SE-ResNet50 baseline |
| `model_seresnet50_merged_seed2000` | 1 | 64K | SE-ResNet50 merged |
| `model_vit_huge_p14_seed1800` | 3 | 64K | ViT-Huge/14 (3 restarts) |

### MISSING / FLAGS for `tensorboard/`
- **No tb log for `model_vitlarge_camadv_cyclegan_seed1100`** — the cam-adv+CycleGAN run did not write tensorboard events (likely because it was orchestrated by a different launcher).
- **No tb log for `model_vitlarge_camadv_hires_seed800`** — same; per-run train.log exists but tb absent.
- **No tb log for `model_vitlarge_dbscan_pseudo_seed950`** — same.

---

## 3. `csvs/` — 201 files, 85 MB

All but one (`pseudo_pairs.csv`) are **Kaggle submission CSVs**, format: 929 rows (1 header + 928 query images), 2 columns:
- `imageName` (e.g. `000001.jpg`)
- `Corresponding Indexes` (space-separated string of top-100 gallery filenames per query)

### Special files (non-submission)
| File | Rows | Columns | Purpose |
|---|---|---|---|
| `pseudo_pairs.csv` | 520 | `cameraID, imageName, objectID, source, rerank_dist` | Pseudo-label pairs from `pseudo_label_extract.py` (top query-gallery pairs above similarity threshold for self-training) |

### Headline submission CSVs (matched to known Kaggle scores)
| Score | CSV | Recipe |
|---|---|---|
| **0.15884** | `camadv_baseline7_plus_camadv_ep60_0.15884.csv` (also in `csvs/`) | **HIGHEST KNOWN** — 7-ckpt baseline + cam-adv ep60 ensemble (NEW HIGH beyond the 0.15421 documented in context_1.md) |
| 0.15421 | `sweep_dba8_k15_k2_4_lambda027_submission.csv` | DBA(k=8) + rerank(k1=15, k2=4, λ=0.275) + class-group filter |
| 0.15288 | `sweep_dba8_k15_k2_4_lambda025_submission.csv` | k2=4 sweep (λ=0.25) |
| 0.14976 | `sweep_dba8_k15_lambda025_submission.csv` | DBA(k=8) + rerank(λ=0.25) |
| 0.14880 | `sweep_dba8_k15_submission.csv` | DBA(k=8) + rerank(k1=15) |
| 0.13707 | `sweep_dba5_k15_submission.csv` | DBA(k=5) — first DBA win |
| 0.13361 | `ensemble_crossrun_classfilt_submission.csv` | 7-ckpt cross-seed ensemble (pre-DBA) |
| 0.13019 | `ensemble_ep30_40_50_classfilt_submission.csv` | Single-seed trajectory ensemble |
| 0.12927 | `ep50_rerank_classfilt_v2_submission.csv` | Class-group filter introduced |
| 0.12873 | `ep50_rerank_submission.csv` | Original retrain baseline |

### Submission CSV families (count per family — all 928-query/100-gallery shape)
| Prefix | Count | What family |
|---|---|---|
| `sweep_dba*` / `freeprobe_dba*` | ~50 | DBA + rerank hyperparameter sweeps |
| `camadv_*` | ~12 | Cam-adversarial training submissions |
| `cyclegan_*` | ~6 | CycleGAN-augmented cam-adv submissions |
| `arcface_*` | 3 | ArcFace head retrain submissions |
| `dbscan_pseudo_*` | 5 | DBSCAN pseudo-label submissions |
| `pseudo_*` | (mixed) | Pseudo-label submissions |
| `adabn_*` | 4 | Adaptive BatchNorm at inference |
| `aniso_dba_*` | 2 | Anisotropic DBA (different k for within/cross-camera) |
| `ensemble_*` | ~7 | Cross-arch, cross-data, EMA-stacked ensembles |
| `centering_*` | 3 | Feature centering (global / per-camera / PCA-drop) |
| `cluster_dba_*` | 2 | Cluster-based DBA (n=700) |
| `fourcrop_tta_*` | 1 | 4-crop test-time aug |
| `dinov2_*` / `eva_*` / `dinov3_*` | 4 | Failed backbone-swap submissions |
| `big*` / `seresnet50_*` / `vit_huge_*` | (mixed) | Other architecture variants |

### MISSING / FLAGS for `csvs/`
- **No JSON/parquet result tables** — no per-experiment hyperparameter sweep CSVs were dumped; sweep results are recoverable only by re-parsing inference logs (`logs/inference/freeprobes.log` + the comprehensive submission table in `notes/score.md` if present).
- **No training-metric CSVs** (loss/Acc per epoch) — those metrics are only in TensorBoard events and the structured `train_log.txt` files.
- The `0.15884` cam-adv variant suggests the score chain extended past the 0.15421 documented in `context/context_1.md` Session 5. **The current best CSV may actually be 0.15884 or higher, not 0.15421.** Worth verifying with the user.

---

## 4. `notebooks/` — **EMPTY**

**No `.ipynb` files exist in the repo.** All exploration was done via Python scripts (e.g.
`ensemble_dba_rerank_sweep.py` for the DBA/rerank sweep, `dbscan_pseudo_extract.py` for
clustering, `pseudo_label_extract.py` for pseudo-labeling). The closest notebook-equivalents
are these scripts in `code/scripts/`:

| Equivalent script | What it does |
|---|---|
| `ensemble_dba_rerank_sweep.py` | DBA-k × rerank-k1/k2/λ post-processing parameter sweep |
| `hparam_ensemble_rerank.py` | Hyperparameter ensemble rerank |
| `dbscan_pseudo_extract.py` | DBSCAN clustering analysis on the gallery features |
| `pseudo_label_extract.py` | Pseudo-pair extraction (produces `pseudo_pairs.csv`) |
| `evaluate_csv.py` | Local mAP eval (produces 100% on hidden labels — see context §3) |

---

## 5. `configs/` — 37 YAML files

### Training configs (33)
| File | Purpose |
|---|---|
| `UrbanElementsReID_train.yml` | Baseline ViT-L/16 PAT (seed=1234) |
| `UrbanElementsReID_train_seed42.yml` | Cross-seed retrain partner |
| `UrbanElementsReID_train_seed200.yml` | Plain triplet seed=200 |
| `UrbanElementsReID_train_seed1700.yml` | Cross-seed #4 |
| `UrbanElementsReID_train_seed2100.yml` | Chained baseline #1 |
| `UrbanElementsReID_train_seed2200.yml` | Chained baseline #2 |
| `UrbanElementsReID_train_seed2300.yml` | Chained baseline #3 |
| `UrbanElementsReID_train_seed100_ema.yml` | EMA baseline |
| `UrbanElementsReID_train_seed100_ema_circle.yml` | EMA + Circle Loss (failed) |
| `UrbanElementsReID_train_arcface.yml` | ArcFace head retrain |
| `UrbanElementsReID_train_camadv.yml` | Cam-adversarial (CAM_ADV=True, λ=0.1) |
| `UrbanElementsReID_train_camadv_seed600.yml` | Cam-adv seed=600 variant |
| `UrbanElementsReID_train_camadv_lambda03.yml` | Cam-adv λ=0.3 |
| `UrbanElementsReID_train_camadv_lightaug.yml` | Cam-adv + light aug |
| `UrbanElementsReID_train_camadv_heavyaug.yml` | Cam-adv + heavy aug |
| `UrbanElementsReID_train_camadv_hires.yml` | Cam-adv at 384×192 |
| `UrbanElementsReID_train_camadv_cyclegan.yml` | Cam-adv + CycleGAN-augmented |
| `UrbanElementsReID_train_camadv_merged_heavyaug.yml` | Cam-adv + merged + heavy aug |
| `UrbanElementsReID_train_dbscan_pseudo.yml` | DBSCAN pseudo-label retrain |
| `UrbanElementsReID_train_pseudo.yml` | Generic pseudo-label retrain |
| `UrbanElementsReID_train_trafficsignal.yml` | Trafficsignal class-specialist |
| `UrbanElementsReID_train_dinov2.yml` | DINOv2-L p14 (failed) |
| `UrbanElementsReID_train_eva.yml` | EVA-L p14 (failed) |
| `UrbanElementsReID_train_eva_smoke.yml` | EVA 2-epoch smoke |
| `UrbanElementsReID_train_vit_huge.yml` | ViT-Huge/14 |
| `UrbanElementsReID_train_seresnet50_bot.yml` | SE-ResNet50 BoT |
| `UrbanElementsReID_train_seresnet50_merged.yml` | SE-ResNet50 + merged data |
| `UrbanElementsReID_train_merged.yml` | UAM-merged seed=2024 (failed) |
| `UrbanElementsReID_train_after_uam.yml` | After-UAM fine-tune |
| `UrbanElementsReID_pretrain_uam.yml` | UAM SSL-style pretrain stage |
| `UrbanElementsReID_train_heavyaug.yml` | Plain heavy-aug fine-tune |
| `UrbanElementsReID_train_deepsup.yml` | Deep-supervision (failed) |
| `UrbanElementsReID_train_deepsup_smoke.yml` | Deep-sup smoke |

### Test/inference configs (4)
| File | Purpose |
|---|---|
| `UrbanElementsReID_test.yml` | Standard inference (256×128 ViT-L/16) |
| `UrbanElementsReID_test_hires.yml` | 384×192 hi-res inference |
| `UrbanElementsReID_test_dinov2.yml` | DINOv2 inference (224×112 p14) |
| `UrbanElementsReID_test_eva.yml` | EVA inference (224×112 p14) |

### MISSING / FLAGS for `configs/`
- **No CLIP-ReID or ViT-H test config** — ViT-H training config exists but no dedicated test YAML (used `UrbanElementsReID_test.yml` adapted in inference scripts).
- **Hi-res 384×192 retrain config**: `UrbanElementsReID_train_camadv_hires.yml` is the only hi-res training config (cam-adv + 384×192). No standalone hi-res baseline retrain config.

---

## 6. `code/` — 104 files, 816 KB

### `code/scripts/` — 51 files (48 Python + 3 shell)
**Training entry point:**
- `train.py` — main training entry, supports CAM_ADV, EMA, GRAD_CLIP, FINETUNE_FROM, ArcFace head

**Single-checkpoint inference:**
- `update.py` — single-ckpt inference (rolled back to 0.12927 reference config)

**Ensemble inference scripts:**
- `ensemble_update.py` — 7-ckpt cross-seed ensemble (produces 0.13361)
- `ensemble_dba_rerank_sweep.py` — **THE post-processing tool of Sessions 4–5**; sweeps DBA k × k1 × k2 × λ on a fixed checkpoint list (produces 0.15421)
- `ensemble_crossarch_update.py` — cross-architecture ensemble (e.g. ViT-L + DINOv2 + EVA)
- `ensemble_multiscale_update.py` — multi-scale TTA ensemble (currently single-scale per group)
- `big_ensemble_inference.py` — 13-ckpt big ensemble (`big13_*` submissions)
- `hparam_ensemble_rerank.py` — hyperparameter rerank sweep on the ensemble

**Cam-adversarial inference family:**
- `camadv_inference.py` — main cam-adv ensemble inference (gave the 0.15884 jump)
- `camadv_s600_inference.py` — cam-adv seed=600 variant
- `camadv_hires_inference.py` — cam-adv 384×192 hi-res
- `camadv_freeprobes.py` — DBA/k1/λ free-probe sweep at hi-res (`freeprobe_*` submissions)

**CycleGAN c001-3 ↔ c004 stylization:**
- `cyclegan_prep_data.py` — split data for CycleGAN training
- `cyclegan_generate_fake_c4.py` — generate fake-c004 images from c001-3
- `cyclegan_build_merged_dataset.py` — build merged dataset with fake-c4 augmentation
- `cyclegan_camadv_inference.py` — inference with CycleGAN-trained model

**Pseudo-label / DBSCAN clustering:**
- `pseudo_label_extract.py` — extract pseudo-pairs (produces `pseudo_pairs.csv`)
- `pseudo_inference.py` — inference with pseudo-label-trained model
- `dbscan_pseudo_extract.py` — DBSCAN clustering on gallery features for pseudo-IDs
- `dbscan_pseudo_inference.py` — inference with DBSCAN-pseudo-trained model
- `build_pseudo_train_subset.py` — build pseudo-labeled training subset

**Class-conditional / specialist models:**
- `build_trafficsignal_subset.py` — build trafficsignal-only training subset
- `trafficsignal_router_inference.py` — class-conditional routing at inference
- `per_class_rerank_inference.py` — per-class rerank parameter tuning

**Backbone-specific inference:**
- `dinov3_inference.py` — DINOv3 (failed)
- `vit_huge_inference.py` — ViT-Huge (failed)
- `seresnet50_inference.py`, `seresnet50_merged_inference.py` — SE-ResNet50 (failed)
- `arcface_inference.py` — ArcFace head ensemble
- `uam_inference.py` — UAM-pretrained model inference

**Inference-time tricks (each typically a Kaggle submission family):**
- `adabn_inference.py`, `adabn_full_inference.py` — Adaptive BatchNorm
- `anisotropic_dba_inference.py` — anisotropic DBA (within/cross-camera different k)
- `cluster_dba_inference.py` — cluster-based DBA
- `weighted_dba_inference.py` — weighted DBA
- `fourcrop_tta_inference.py` — 4-crop test-time aug
- `feature_centering_inference.py` — feature centering (global / per-camera / PCA-drop)
- `gentle_qe_inference.py` — gentle query expansion
- `qmv_inference.py` — query mean vector
- `rank_fusion_inference.py` — rank-fusion across per-checkpoint reranks
- `multiscale_tta_inference.py` — multi-scale TTA
- `lambda03_inference.py` — λ=0.3 specific inference
- `heavyaug_inference.py`, `heavyaug_replace_inference.py` — heavy-aug ensemble
- `lightaug_inference.py` — light-aug ensemble
- `mlp_refinement_pipeline.py` — MLP head refinement on features

**Evaluation / dataset scaffolding:**
- `evaluate_csv.py` — local mAP eval (100% artifact on hidden labels — see context §3)
- `merge_datasets.py` — UAM-merged dataset builder

**Shell:**
- `run_3_seeds.sh` — orchestrator for chained seed=2100/2200/2300 baseline retrains
- `run_uam_pipeline.sh` — orchestrator for UAM pretrain → fine-tune chain
- `enviroments.sh` — env setup [sic, original spelling]

### `code/model/` — 7 files
- `make_model.py` — `build_transformer_PAT` (active class) + `build_vit` + `build_part_attention_vit` factories. Includes CAM_ADV head + ArcFace head registration.
- `backbones/` — `vit_pytorch.py` with PAT ViT-L/16 + LayerScale + p14 factories for DINOv2/EVA. Contains `part_Attention_ViT`, `LayerScale`, `PatchEmbed_overlap`, position-embedding interpolation logic.

### `code/processor/` — 2 files
- `part_attention_vit_processor.py` — main training loop. Contains: AMP scaler, EMA update hook, cam-adv loss with GRL, NaN guards (per-iteration loss + per-epoch param), grad clipping, **commented-out checkpoint-deletion block at lines ~204–213** (the safety fix mentioned in CLAUDE.md).

### `code/loss/` — 13 files
- `make_loss.py`, `build_loss.py` — loss-builder dispatch
- `triplet_loss.py` — soft-margin hard-mined triplet
- `ce_labelSmooth.py` — CE with label smoothing + soft-label mechanism (`SOFT_LABEL`, `SOFT_LAMBDA`)
- `myloss.py` — Pedal patch-clustering loss with `PatchMemory` momentum bank
- `arcface.py`, `arcface_head.py` — ArcFace formulation
- `circle_loss.py` — Circle Loss (γ=256 default — failed at clip=1.0)
- `center_loss.py`, `metric_learning.py`, `softmax_loss.py`, `smooth.py`

### `code/utils/` — 9 files
- `metrics.py` — R1_mAP_eval (with the local-eval 100% artifact on pid=-1)
- `re_ranking.py` — k-reciprocal rerank (k1, k2, λ)
- `ema.py` — `ModelEMA(decay=0.9999)` shadow-weights wrapper
- `meter.py`, `logger.py`, `iotools.py`, `reranking_gpu.py`, `lr_scheduler.py`, `__init__.py`

### `code/solver/` — 6 files
- `make_optimizer.py` — Adam optimizer w/ WeightDecay + LARGE_FC_LR option
- `lr_scheduler.py`, `cosine_lr.py`, `scheduler_factory.py`

### `code/data/` — 16 files
- `build_DG_dataloader.py` — main DataLoader factory
- `data_utils.py` — fvcore PathManager workaround (uses built-in `open` — see context §4 fix #1)
- `datasets/` — `UrbanElementsReID.py`, `UrbanElementsReID_test.py`, `bases.py`, `dataset_loader.py`
- `samplers/` — `RandomIdentitySampler` (P×K sampling)
- `transforms/` — `build_transforms.py` (LGT, REA, CJ, flip, padding)

---

## 7. `notes/` — 8 markdown files

| File | Purpose |
|---|---|
| `context_1.md` | The 1146-line full session history (Sessions 1–5, written 2026-04-21 through 2026-04-26) |
| `score.md` | (in `context/`) — supplementary score notes |
| `CLAUDE.md` | Repo's CLAUDE.md (current best, dead-ends, critical paths, user preferences) |
| `RESUME.md` | Resume-style summary of the session arc |
| `README.md` | Repo README |
| `backup_score_README.md` | README inside `backup_score/` (snapshot at 0.13361 era) |
| `backup_score_notes.md` | Notes about the best-score backup snapshot |
| `OFFSITE_BACKUP.md` | Offsite-backup checklist |

---

## 8. Cross-references — score → ckpt → CSV → tensorboard run

**Best documented score in `context_1.md`: 0.15421**, reproducer:
```bash
python ensemble_dba_rerank_sweep.py
# variant: dba8_k15_k2_4_lambda025 (the lambda027 variant in VARIANTS list)
```
- CSV: `csvs/sweep_dba8_k15_k2_4_lambda027_submission.csv`
- Checkpoints: `models/model_vitlarge_256x128_60ep/{30,40,50}.pth` + `models/model_vitlarge_256x128_60ep_seed42/{30,40,50,60}.pth` (NOT in paper_context — too large; see source paths)
- Tensorboard: `tensorboard/model_vitlarge_256x128_60ep/` + `tensorboard/model_vitlarge_256x128_60ep_seed42/`

**Apparent NEW HIGH found in artifacts: 0.15884** —
- CSV: `csvs/camadv_baseline7_plus_camadv_ep60_0.15884.csv`
- This score is **NOT documented in `context_1.md`** (which ends at Session 5 / 0.15421). User may want to verify whether 0.15884 is the actual current best — it would correspond to adding cam-adversarial-trained checkpoints to the 7-ckpt baseline.
- Recipe (inferred): 7-ckpt baseline + cam-adv seed=500 ep60 → ensemble. Reproducer is likely `code/scripts/camadv_inference.py`.

---

## 9. NOT included (intentionally excluded)

| Excluded | Reason | Source path (still on disk) |
|---|---|---|
| `*.pth` checkpoints (~26 GB total across model dirs + `pretrained/`) | Too large for paper context; not needed for paper text | `/workspace/miuam_challenge_diff/models/`, `/workspace/miuam_challenge_diff/pretrained/` |
| `*.npy`, `*.npz` feature caches (~60 MB) | Reproducible feature dumps; not needed for paper | `qf.npy`, `gf.npy`, `results/cache/8ckpt_*.npz` |
| `__pycache__/` | Compiled bytecode | (everywhere) |
| Raw image data (`/workspace/Urban2026/`, `/workspace/UAM_Unified_extract/`) | Image data; outside paper context scope | `/workspace/Urban2026/`, `/workspace/UAM_Unified_extract/` |
| `.venv/` | Python virtualenv | `/workspace/miuam_challenge_diff/.venv/` |
| Backup checkpoint copy in `backup_score/seed*.pth` | 7×1.2GB each | `/workspace/miuam_challenge_diff/backup_score/seed*.pth` |

---

## 10. Things worth flagging for the paper-drafting session

1. **The 0.15884 cam-adversarial CSV** in `csvs/` is **above** the 0.15421 documented in `context_1.md`. Either:
   - `context_1.md` is outdated (Session 6+ work happened: the cam-adversarial line of experiments — including λ=0.1/0.3 sweeps, hi-res, CycleGAN, heavy-aug variants — appears to have netted a real gain past 0.15421)
   - or the 0.15884 file is misnamed / unsubmitted
   - **Recommend:** verify with the user before writing the paper's "results" section.

2. **Cam-adversarial training** is a major experimental track that's NOT in `context_1.md`. There are 9 cam-adv configs, 7 cam-adv model folders, 12+ cam-adv submission CSVs. This is the missing Session 6+ material.

3. **Pseudo-labeling and DBSCAN clustering** were also tried (seed=900 pseudo, seed=950 dbscan_pseudo). Neither is in `context_1.md`. Submissions: `pseudo_*` and `dbscan_pseudo_*` — likely all regressed but worth a paper "negative results" section.

4. **CycleGAN c001-3 ↔ c004 stylization** as a c004-domain-gap attack is novel for this dataset and worth highlighting. Failure / partial success is documented in `logs/cyclegan_train.log` + `logs/train_camadv_cyclegan.log` + `cyclegan_*` CSVs.

5. **No notebooks were ever written.** All analysis was Python-script-driven. If the paper wants reproducible per-experiment plots, you'll need to (a) replay tensorboard events with `tbparse`, or (b) parse `train_log.txt` files (each line has `Epoch[E] Iter[I/N] total_loss: L Acc: A`).

6. **Architectures explored beyond ViT-L/16 PAT:** ViT-Huge/14, DINOv2-L/14, EVA-L/14, SE-ResNet50 BoT (the baseline non-transformer architecture), DINOv3. All except ViT-L/16 PAT regressed. This is a strong "negative results" catalog for the paper.

7. **Inference-time tricks fully explored** (each has its own script + submission family): DBA (k=5,7,8,9,10,12), rerank sweep (k1, k2, λ), AdaBN, anisotropic DBA, cluster DBA, weighted DBA, 4-crop TTA, multi-scale TTA, feature centering, query expansion, MLP head refinement. The DBA-k=8 + rerank tuning is the headline result; the others are negative-result entries.
