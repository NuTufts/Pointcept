#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/vdasil01/Pointcept
source setenv_pointcept_only.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ----------------------------------------------------------------------------
# LoRA fine-tuning from SONATA pretrained weights
#
# Set WEIGHTS to your pretrained SONATA checkpoint.
# The LoRASonataCheckpointLoader hook remaps the keys automatically,
# so you can point directly at a raw pretraining checkpoint (epoch_N.pth
# or model_best.pth from sonata/lartpc_v*/model/).
# ----------------------------------------------------------------------------

#WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v5_h200_noghosts/model/model_last.pth
WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/model_last.pth
#export WANDB_API_KEY="wandb_v1_JzdGdSIxN1z8lEAT8URJxXolzDo_lvZ"
python3 tools/train.py \
    --config configs/lartpc/lorafinetune-sonata-v1m1-lartpc-v6-fixed.py \
    --num-gpus 4 \
    --options weight=$WEIGHTS

# ----------------------------------------------------------------------------
# To resume a paused/crashed run, uncomment below and comment out the block
# above. Point RESUME_WEIGHTS at the last saved epoch checkpoint.
# ----------------------------------------------------------------------------

#RESUME_WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lora_finetune_v1_p100_noghost/model/model_last.pth
#python3 tools/train.py \
#    --config configs/lartpc/lorafinetune-sonata-v1m1-lartpc.py \
#    --num-gpus 6 \
#    --options weight=$RESUME_WEIGHTS resume=True
