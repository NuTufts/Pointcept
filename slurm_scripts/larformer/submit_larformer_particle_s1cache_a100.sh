#!/bin/bash
# Stage-3 particle segmenter M2F-recipe retrain -- 4x A100, 8-epoch OneCycle,
# on the v2 tau=0.20 cache (382,768 train events -> 23,923 iters/epoch at
# batch 16; est ~4.5-6h/epoch with vecloss -> ~2 chained 48h windows).
# Same pre-submit afterany chaining as the slicer v2 script.
#   TO STOP THE CHAIN:  touch <SAVE>/STOP_AUTORESUBMIT
#   sbatch submit_larformer_particle_s1cache_a100.sh

#SBATCH --job-name=lf-particle-s1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=48
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-particle-s1.%j.%N.log
#SBATCH --error=logs/lf-particle-s1.%j.%N.err

set -u
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage3_particle/larformer-particle-s1cache-m2frecipe.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SCRIPT_DIR=${WORKDIR}/slurm_scripts/larformer
SELF=${SCRIPT_DIR}/submit_larformer_particle_s1cache_a100.sh

SAVE=${WORKDIR}/exp/larformer_particle_s1cache_m2frecipe
CHECKPOINT=${SAVE}/model/model_last.pth
FINAL_CKPT=${SAVE}/model/epoch_8.pth
STOP_FILE=${SAVE}/STOP_AUTORESUBMIT
PROG_FILE=${SAVE}/.autoresubmit_last_ckpt_mtime

module load apptainer
mkdir -p "${SAVE}" "${SCRIPT_DIR}/logs"

chain_next() {
  [ -z "${SLURM_JOB_ID:-}" ] && return 0
  [ -f "${STOP_FILE}" ] && { echo "[chain] STOP file present; not chaining."; return 0; }
  [ -f "${FINAL_CKPT}" ] && { echo "[chain] training complete; not chaining."; return 0; }
  local cur last
  cur=$(stat -c %Y "${CHECKPOINT}" 2>/dev/null || echo 0)
  last=$(cat "${PROG_FILE}" 2>/dev/null || echo -1)
  if [ "${cur}" = "${last}" ] && [ "${cur}" != "0" ]; then
    echo "[chain] no progress since last hop -> self-halting."; touch "${STOP_FILE}"; return 0
  fi
  echo "${cur}" > "${PROG_FILE}"
  local next
  next=$( cd "${SCRIPT_DIR}" && sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "${SELF}" )
  echo "[chain] queued successor ${next}."
}

[ -f "${STOP_FILE}" ] && { echo "[chain] STOP file present; exiting."; exit 0; }
chain_next

if [ -f "${CHECKPOINT}" ]; then
  echo "[submit] resuming from ${CHECKPOINT}"
  RESUME_OPTS="--options resume=True weight=${CHECKPOINT}"
else
  echo "[submit] fresh start"
  RESUME_OPTS=""
fi

apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 4 ${RESUME_OPTS}"
