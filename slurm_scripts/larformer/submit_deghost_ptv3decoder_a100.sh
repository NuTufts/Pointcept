#!/bin/bash
# PTv3-decoder deghoster production training -- 2x A100-80GB.
#
# MEMORY SIZING (from smoke 2169834): 31.2 GB at batch 32 on one GPU -> the
# config's batch_size=96 does NOT fit one 80GB card (the lr=0-frozen encoder
# still stores activations for backward). Run 2 GPUs x 48/GPU = 96 total,
# ~45-50 GB/GPU expected.
#
# Estimated wall: ~0.33 s/iter at 32/GPU -> ~4.3k iters/epoch at batch 96
# -> roughly 12-15 h for the 20-epoch OneCycle. Single 24 h window with
# auto-resume as a safety net (no chaining needed at this length; resubmit
# manually if it ever times out -- auto-resume picks up from model_last.pth).
#
#   sbatch submit_deghost_ptv3decoder_a100.sh

#SBATCH --job-name=lf-deghost-ptv3dec
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-deghost-ptv3dec.%j.%N.log
#SBATCH --error=logs/lf-deghost-ptv3dec.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage1_deghost/deghost-ptv3decoder-v1-frozenenc-extbnb.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SAVE=${WORKDIR}/exp/deghost_ptv3decoder_v1_frozenenc_extbnb
CHECKPOINT=${SAVE}/model/model_last.pth

module load apptainer
mkdir -p "${SAVE}"

# Auto fresh-vs-resume. Fresh: cfg.weight = the Sonata pretrain (loaded by
# SonataFinetuneCheckpointLoader). Resume: override weight to model_last +
# resume=True (Sonata loader no-ops, CheckpointLoader resumes).
if [ -f "${CHECKPOINT}" ]; then
  echo "[submit] resuming from ${CHECKPOINT}"
  RESUME_OPTS="--options resume=True weight=${CHECKPOINT}"
else
  echo "[submit] fresh start (Sonata pretrain via config weight key)"
  RESUME_OPTS=""
fi

apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 2 ${RESUME_OPTS}"
