#!/bin/bash
# PTv3-decoder deghoster on the P5B.3 MIXED sim+data encoder — crop stage of
# the encoder-swap A/B (see deghost-ptv3decoder-p5b3mix-v1.py). Byte-identical
# recipe to submit_deghost_ptv3decoder_a100.sh; only config + save dir differ.
#   sbatch slurm_scripts/larformer/submit_deghost_ptv3decoder_p5b3mix_a100.sh

#SBATCH --job-name=lf-deghost-p5b3mix
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-deghost-p5b3mix.%j.%N.log
#SBATCH --error=logs/lf-deghost-p5b3mix.%j.%N.err

set -u
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage1_deghost/deghost-ptv3decoder-p5b3mix-v1.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SAVE=${WORKDIR}/exp/deghost_ptv3decoder_p5b3mix_v1
CHECKPOINT=${SAVE}/model/model_last.pth

module load apptainer
mkdir -p "${SAVE}"
if [ -f "${CHECKPOINT}" ]; then
  echo "[submit] resuming from ${CHECKPOINT}"
  RESUME_OPTS="--options resume=True weight=${CHECKPOINT}"
else
  echo "[submit] fresh start (P5B.3 pretrain via config weight key)"
  RESUME_OPTS=""
fi
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 2 ${RESUME_OPTS}"
