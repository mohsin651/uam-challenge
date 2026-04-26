# RESUME — Urban Elements ReID 2026

**Last updated:** 2026-04-26.

## Where we are

| Score | Recipe |
|---|---|
| **0.15421 (best)** | 7-ckpt ensemble + DBA k=8 + rerank k1=15/k2=4/λ=0.275 + class-group filter |
| 0.13361 (cross-seed ensemble baseline, no DBA) | seed1234 + seed42 ckpts |
| 0.12072 (prior) | starting point user brought in |

CSV of best: `backup_score/sweep_dba8_k15_k2_4_lambda027_submission.csv`.

Reproduce: `python ensemble_dba_rerank_sweep.py` (variant `dba8_k15_k2_4_lambda027` is in the VARIANTS list).

## The 7 production checkpoints (don't touch)

```
models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth
models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth
models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth
models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth
models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth
models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth
models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth
```

## Next move (recommended)

**Train seed=200, plain triplet, NO EMA, NO Circle Loss, same as 0.13361 recipe.** Expected +0.003-0.008.

```bash
# 1. Copy config and change SEED + LOG_NAME
cp config/UrbanElementsReID_train.yml config/UrbanElementsReID_train_seed200.yml
# Edit: SEED: 200 and LOG_NAME: './model_vitlarge_256x128_60ep_seed200'

# 2. Verify checkpoint-deletion fix is still in place
grep -A 5 "DISABLED" processor/part_attention_vit_processor.py | head

# 3. Launch in DETACHED TMUX (not nohup — that dies on disconnect)
tmux new -d -s reid_train "cd /workspace/miuam_challenge_diff && source .venv/bin/activate && python train.py --config_file config/UrbanElementsReID_train_seed200.yml > logs/train_seed200.log 2>&1"

# 4. After ~75 min: add ckpts to ensemble_dba_rerank_sweep.py CHECKPOINT_LIST and re-run
```

## DO NOT retry (Kaggle-confirmed dead, see context/context_1.md §31, §47)

UAM merged training • DINOv2 backbone • EVA backbone • Circle Loss (γ=256+clip=1.0) • EMA-trained ckpts in this ensemble • multi-layer CLS concat • h-flip TTA (directional traffic signs) • part-token concat at inference • deep supervision at aux=0.1 • query expansion α=0.7/K=3 • DBA k≥10 • λ<0.25 • k1=12 with current setup

## Read for full context

- `CLAUDE.md` — auto-loaded by Claude Code, includes architecture + collaboration prefs
- `context/context_1.md` — 1146-line full history (Sessions 1-5)
- `backup_score/notes.md` — Kaggle submission tally + scores
- `backup_score/README.md` — backup folder index
