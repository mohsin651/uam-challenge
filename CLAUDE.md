# Urban Elements ReID 2026 — SkyNet (Mohsin)

Active Kaggle competition: https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026

**Current best Kaggle score: 0.15421** (set 2026-04-26).
Recipe: 7-checkpoint ensemble + DBA(k=8) + rerank(k1=15, k2=4, λ=0.275) + class-group filter.
The exact CSV is at `backup_score/sweep_dba8_k15_k2_4_lambda027_submission.csv`.

## BEFORE doing anything in this project — READ THIS

Read **`context/context_1.md`** top-to-bottom. It's the complete session
history (5 sessions, 1146 lines) including:
- Score progression: 0.12072 → 0.12873 → 0.12927 → 0.13361 → 0.14976 → 0.15421
- §31 + §44 + §47 are the most critical "do not retry" + "what works" tables
- §48 has the realistic next-step paths (A through E)

**DO NOT re-experiment with anything in the dead-end lists.** Things confirmed
to fail: UAM merged training data, DINOv2 backbone, EVA backbone, Circle Loss
(at gamma=256 with grad clip), EMA-trained ckpts in this ensemble, multi-layer
CLS concat, h-flip TTA (directional traffic signs), part-token concat,
deep supervision at aux=0.1, heavy-aug fine-tune, query expansion (α=0.7/K=3),
DBA k≥10, λ<0.25, k1=12 in re-ranking.

## Things that have worked (verified on Kaggle)

1. Cross-seed ensemble (seed=1234 + seed=42)
2. Class-group filter (container ∪ rubbishbins merged)
3. DBA on gallery features with k=8
4. Re-rank with k1=15, k2=4, λ=0.275

## Critical paths

- **Training:** `cd /workspace/miuam_challenge_diff && source .venv/bin/activate && python train.py --config_file config/<yaml>`
  - **Always launch in detached tmux** for any long run; nohup `&` died on session disconnect (see Session 3 §29).
- **Reproduce 0.15421:** `python ensemble_dba_rerank_sweep.py` (variant `dba8_k15_k2_4_lambda027` is in the VARIANTS list)
- **Backup of best:** `backup_score/`
- **venv:** Python 3.10, torch==2.1.0+cu121
- **GPU:** single RTX 4090, 24 GiB. Batch 64 at 256×128 uses ~12 GiB; safe.
- **Dataset:** `/workspace/Urban2026/` (challenge), `/workspace/UAM_Unified_extract/` (external — DON'T merge into training; UAM is a different city)

## User collaboration preferences (from memory)

- Concise responses, can read diffs directly, no need for over-explanation
- Ask before destructive operations (rm, force-push, etc.)
- Convert relative dates to absolute ("today" → "2026-04-26")
- Has been burned by long training runs dying overnight — verify checkpoint deletion bug at processor/part_attention_vit_processor.py:204-213 is still commented out before any training
- Prefers to know honest expected gains; doesn't want hype after a long session of failures

## Architecture (1-paragraph recap)

PAT (Part-Aware Transformer): ViT-Large/16 backbone (305M params, 24 blocks, 1024-d hidden, ImageNet pretrained). On top: 1 CLS token + 3 part tokens added to patch sequence. Each block has standard self-attention AND part-aware attention (each part token attends to a learned subset of patches). Loss = CE (label-smoothed) + Triplet (soft margin, hard mining) + Pedal (patch clustering on part tokens), all weighted 1.0. AMP for training. Test-time: extract final-layer CLS feature, L2-normalize, ensemble across multiple checkpoints, k-reciprocal re-rank.

## Recommended next-step paths (from §48 of context_1.md)

| Path | Effort | Expected | Risk |
|---|---|---|---|
| A. Train seed=200 (plain triplet, NO EMA, NO Circle) + ensemble | 75 min | +0.003 to +0.008 | low |
| B. Stack TWO more seeds (200, 300) | 2.5 hr | +0.005 to +0.012 | low |
| C. ArcFace + camera embedding retrain | ~3 hr | +0.01 to +0.03 | medium |
| D. UAM as SSL pretraining source (NOT merged data) | ~4 hr | +0.005 to +0.020 | high |
| E. Pseudo-labeling from current 0.15421 ensemble | ~2 hr | +0.005 to +0.015 | high (can hurt) |

**Default recommendation: A first.** Lower risk, proven pattern. After that, stack or pivot.
