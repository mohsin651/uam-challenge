#!/bin/bash
# Chained training: seed=2100 → 2200 → 2300 (proven baseline recipe).
# Use `||` so a NaN in one seed doesn't kill the chain — others still run.
set -u  # error on unset variable, but no -e (let chain continue past failures)
cd /workspace/miuam_challenge_diff
source .venv/bin/activate

mkdir -p models/model_vitlarge_256x128_60ep_seed2100
mkdir -p models/model_vitlarge_256x128_60ep_seed2200
mkdir -p models/model_vitlarge_256x128_60ep_seed2300

echo "==================== seed=2100 START $(date) ===================="
python train.py --config_file config/UrbanElementsReID_train_seed2100.yml \
    2>&1 | tee models/model_vitlarge_256x128_60ep_seed2100/train.log
RET2100=$?
echo "==================== seed=2100 END  $(date) (exit $RET2100) ==="

echo "==================== seed=2200 START $(date) ===================="
python train.py --config_file config/UrbanElementsReID_train_seed2200.yml \
    2>&1 | tee models/model_vitlarge_256x128_60ep_seed2200/train.log
RET2200=$?
echo "==================== seed=2200 END  $(date) (exit $RET2200) ==="

echo "==================== seed=2300 START $(date) ===================="
python train.py --config_file config/UrbanElementsReID_train_seed2300.yml \
    2>&1 | tee models/model_vitlarge_256x128_60ep_seed2300/train.log
RET2300=$?
echo "==================== seed=2300 END  $(date) (exit $RET2300) ==="

echo ""
echo "==================== ALL DONE  $(date) ===================="
echo "seed=2100 exit: $RET2100  (success=0, NaN=non-zero)"
echo "seed=2200 exit: $RET2200"
echo "seed=2300 exit: $RET2300"
echo ""
echo "Saved ckpts:"
ls -lh models/model_vitlarge_256x128_60ep_seed2{100,200,300}/*.pth 2>/dev/null
