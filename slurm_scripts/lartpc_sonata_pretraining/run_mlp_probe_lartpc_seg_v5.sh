#!/bin/bash
cd /cluster/tufts/wongjiradlabnu/vdasil01/Pointcept
source setenv_pointcept_only.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ----------------------------------------------------------------------------
# MLP Probe (parameter-matched) from SONATA pretrained weights
#
# The backbone is fully FROZEN — only the MLP head (~1.215M params) trains.
# Set WEIGHTS to the same pretrained SONATA checkpoint used for LoRA so that
# the backbone features are directly comparable.
# The SonataCheckpointLoader hook remaps keys automatically.
# ----------------------------------------------------------------------------
WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v5_h200_noghosts/model/model_last.pth

python3 tools/train.py \
    --config configs/lartpc/linearprobe-sonata-v1m1-segmentation-v5.py \
    --num-gpus 4 \
    --options weight=$WEIGHTS
# ----------------------------------------------------------------------------
# To resume a paused/crashed run, uncomment below and comment out the block
# above. Point RESUME_WEIGHTS at the last saved epoch checkpoint.
# ----------------------------------------------------------------------------
#RESUME_WEIGHTS=/cluster/tufts/wongjiradlabnu/vdasil01/Pointcept/sonata/mlp_probe_parammatched/model/model_last.pth
#python3 tools/train.py \
#    --config configs/lartpc/linearprobe-sonata-v1m1-segmentation-v5.py \
#    --num-gpus 6 \
#    --options weight=$RESUME_WEIGHTS resume=True
