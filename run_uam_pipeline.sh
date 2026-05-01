#!/bin/bash
# Two-stage chained training: UAM pretrain → Urban2026 fine-tune.
# Run inside a tmux session so it survives disconnect.
set -e
cd /workspace/miuam_challenge_diff
source .venv/bin/activate

echo "===================================================="
echo "STAGE 1: UAM supervised pretrain (~45 min, 60 epochs)"
echo "===================================================="
python train.py --config_file config/UrbanElementsReID_pretrain_uam.yml

# Verify ep60 ckpt exists before chaining
UAM_EP60=/workspace/miuam_challenge_diff/models/model_vitlarge_uam_pretrain_seed1000/part_attention_vit_60.pth
if [ ! -f "$UAM_EP60" ]; then
    echo "ERROR: UAM ep60 ckpt not found at $UAM_EP60 — aborting fine-tune"
    exit 1
fi
echo "UAM ep60 ckpt confirmed: $(ls -lh $UAM_EP60)"

echo ""
echo "===================================================="
echo "STAGE 2: Urban2026 fine-tune from UAM (~75 min, 60 epochs)"
echo "===================================================="
python train.py --config_file config/UrbanElementsReID_train_after_uam.yml

echo ""
echo "===================================================="
echo "PIPELINE DONE"
echo "===================================================="
ls -lh /workspace/miuam_challenge_diff/models/model_vitlarge_uam_pretrain_seed1000/*.pth
ls -lh /workspace/miuam_challenge_diff/models/model_vitlarge_after_uam_seed1100/*.pth
