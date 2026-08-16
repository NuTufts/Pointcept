#!/bin/bash
# LoRA deghoster on the P5B.3 MIXED sim+data encoder — LoRA arm of the
# encoder-swap A/B (see lorafinetune-sonata-p5b3mix-deghost.py). Trains on
# the v3 LANTERN lists (original prod4 data deleted). 2 GPUs, 50 epochs.
#   sbatch slurm_scripts/larformer/submit_lora_deghost_p5b3mix.sh

#SBATCH --job-name=lora-deghost-p5b3mix
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lora-deghost-p5b3mix.%j.%N.log
#SBATCH --error=logs/lora-deghost-p5b3mix.%j.%N.err

set -u
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage1_deghost/lorafinetune-sonata-p5b3mix-deghost.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SAVE=${WORKDIR}/sonata/lora_deghost_p5b3mix_hasmatch
CHECKPOINT=${SAVE}/model/model_last.pth

module load apptainer
mkdir -p "${SAVE}"
if [ -f "${CHECKPOINT}" ]; then
  echo "[submit] resuming from ${CHECKPOINT}"
  RESUME_OPTS="--options resume=True weight=${CHECKPOINT}"
else
  echo "[submit] fresh start (P5B.3 pretrain via LoRASonataCheckpointLoader)"
  RESUME_OPTS=""
fi
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 2 ${RESUME_OPTS}"
