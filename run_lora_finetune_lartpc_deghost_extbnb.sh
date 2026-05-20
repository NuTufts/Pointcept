#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ----------------------------------------------------------------------------
# LoRA fine-tuning for de-ghosting from SONATA pretrained weights
#
# Same pretrained SONATA checkpoint as SSNet fine-tuning — the
# LoRASonataCheckpointLoader hook remaps keys automatically.
# The de-ghosting LoRA A/B matrices and deghost_head are new and will
# be initialised from scratch (expected missing keys, logged at startup).
# ----------------------------------------------------------------------------
#
WEIGHTS=/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_run1/model/epoch_18.pth
# ----------------------------------------------------------------------------
# Full 10-epoch production run (6 GPUs, full batch size, WandB enabled).
# Uncomment this block and comment out the test block below when ready.
# ----------------------------------------------------------------------------
python3 tools/train.py \
    --config configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-deghost-extbnb-larmatch.py \
    --num-gpus 2 \
    --options weight=$WEIGHTS

# ----------------------------------------------------------------------------
# 2-epoch test run (4 GPUs, reduced batch, WandB disabled).
# Verifies: checkpoint loading, loss decreasing, no shape errors,
# SemSegEvaluator reporting IoU for both real and ghost classes.
# ----------------------------------------------------------------------------
#python3 tools/train.py \
#    --config configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v5-deghost.py \
#    --num-gpus 4 \
#    --options weight=$WEIGHTS \
#        epoch=2 \
#        batch_size=16 \
#        batch_size_val=8

# ----------------------------------------------------------------------------
# To resume a paused/crashed run, uncomment below and comment out the block
# above. Point RESUME_WEIGHTS at the last saved epoch checkpoint.
# ----------------------------------------------------------------------------
#RESUME_WEIGHTS=/cluster/tufts/wongjiradlabnu/vdasil01/Pointcept/sonata/lora_finetune_v6_deghost_coordfix/model/model_last.pth
#python3 tools/train.py \
#    --config configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-deghost.py \
#    --num-gpus 6 \
#    --options weight=$RESUME_WEIGHTS resume=True
