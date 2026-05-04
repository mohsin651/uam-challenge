# Off-machine backup recommendation

Disk-internal backup (this folder) only protects against **accidental
deletion of files in the project**. It does NOT protect against:
- Cloud GPU instance being terminated (everything on this disk is lost)
- Disk corruption
- Filesystem-wide rm

**Real disaster recovery = pull these to your laptop / external drive.**

## Minimum (recommended): code + best CSVs + docs (~5 MB)

These are tiny and let you reproduce the recipe (you'd need to retrain the 7
ckpts, but config + code + best CSV is enough to verify reproduction).

```bash
# Run these FROM YOUR LAPTOP, not from the GPU box.
mkdir -p ~/Desktop/reid_offsite_2026-04-26

# All non-ckpt files in the backup folder
scp -r root@<HOST>:/workspace/miuam_challenge_diff/backup_score/{*.csv,*.py,*.md,config} \
       ~/Desktop/reid_offsite_2026-04-26/

# The full session history
scp root@<HOST>:/workspace/miuam_challenge_diff/context/context_1.md \
    ~/Desktop/reid_offsite_2026-04-26/
```

## Full (8 GB): everything including the 7 production checkpoints

These are 1.2 GB each. Total ~8 GB. Required to reproduce the 0.15421 score
without retraining from scratch (~9 hours).

```bash
# From your laptop:
scp -r root@<HOST>:/workspace/miuam_challenge_diff/backup_score \
    ~/Desktop/reid_offsite_2026-04-26/
```

(8 GB over typical home internet = 30-60 min depending on your bandwidth.)

## Replace `<HOST>` with your cloud GPU's address

If you SSH'd in with `ssh root@some-host -p 12345`, then:
```bash
scp -P 12345 -r root@some-host:/workspace/miuam_challenge_diff/backup_score \
    ~/Desktop/reid_offsite_2026-04-26/
```

## What to do if the cloud GPU instance dies

1. Spin up a fresh GPU instance (any provider with PyTorch + CUDA 12.1)
2. Recreate venv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (or copy from this backup's package list)
3. Download the 7 ckpts from your laptop back to the new instance:
   ```bash
   scp -r ~/Desktop/reid_offsite_2026-04-26/backup_score \
       root@<NEW-HOST>:/workspace/miuam_challenge_diff/backup_score/
   ```
4. Move `seed1234_*.pth` and `seed42_*.pth` back to their original locations:
   ```bash
   mkdir -p models/model_vitlarge_256x128_60ep models/model_vitlarge_256x128_60ep_seed42
   mv backup_score/seed1234_ep30.pth models/model_vitlarge_256x128_60ep/part_attention_vit_30.pth
   mv backup_score/seed1234_ep40.pth models/model_vitlarge_256x128_60ep/part_attention_vit_40.pth
   mv backup_score/seed1234_ep50.pth models/model_vitlarge_256x128_60ep/part_attention_vit_50.pth
   mv backup_score/seed42_ep30.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_30.pth
   mv backup_score/seed42_ep40.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_40.pth
   mv backup_score/seed42_ep50.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_50.pth
   mv backup_score/seed42_ep60.pth   models/model_vitlarge_256x128_60ep_seed42/part_attention_vit_60.pth
   ```
5. You'll also need:
   - The challenge dataset at `/workspace/Urban2026/` — re-download from Kaggle
   - The supervised ViT-L pretrained weights (`pretrained/jx_vit_large_p16_224-4ee7a4dc.pth`) — re-download from `https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth`
6. Run `python ensemble_dba_rerank_sweep.py` (with the dba8_k15_k2_4_lambda027 variant) to verify 0.15421 reproduction.
