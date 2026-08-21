#!/bin/bash
# LArFormer Stage-2 slicer CELL S1 (mix-enriched + completed labels + masked-no-object + v6-lantern cascade) -- 4x A100-80GB, 5-epoch OneCycle.
#
# V2 = v1 recipe + vectorized pair loss (2x step speed) + num_queries 48 +
# per-layer Hungarian matching (match_per_layer). NOT TO BE LAUNCHED until the
# v1 run is stopped/finished (it holds the GPUs) AND the v2 smoke
# (smoke_larformer_slicer_m2frecipe_v2_a100.sh) passed.
#
# Same pre-submit afterany chaining as the v1 script. With the vectorized
# loss, epochs are expected ~22h -> ~2 epochs per 48h window.
#
#   TO STOP THE CHAIN:  touch <SAVE>/STOP_AUTORESUBMIT
#
#   sbatch submit_larformer_slicer_s1_a100.sh

#SBATCH --job-name=lf-slicer-s1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=48
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:4
# pax007 excluded 2026-07-30: two consecutive resume jobs (1823692, 1979840)
# hung there at the FIRST NCCL allreduce (all 4 ranks, last completed work =
# -1) and were killed by the 1h watchdog — node-level GPU-interconnect fault
# that SLURM doesn't detect (sinfo shows it healthy).
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-slicer-s1.%j.%N.log
#SBATCH --error=logs/lf-slicer-s1.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage2_slicer/larformer-slicer-s1-mixenriched-v1.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SCRIPT_DIR=${WORKDIR}/slurm_scripts/larformer
SELF=${SCRIPT_DIR}/submit_larformer_slicer_s1_a100.sh

SAVE=${WORKDIR}/exp/larformer_slicer_s1_mixenriched_v1
CHECKPOINT=${SAVE}/model/model_last.pth
FINAL_CKPT=${SAVE}/model/epoch_5.pth           # config epoch=5 -> training done
STOP_FILE=${SAVE}/STOP_AUTORESUBMIT
PROG_FILE=${SAVE}/.autoresubmit_last_ckpt_mtime

module load apptainer
mkdir -p "${SAVE}" "${SCRIPT_DIR}/logs"

chain_next() {
  [ -z "${SLURM_JOB_ID:-}" ] && return 0          # manual run -> never chain
  if [ -f "${STOP_FILE}" ]; then
    echo "[chain] STOP file present (${STOP_FILE}); not chaining."; return 0
  fi
  if [ -f "${FINAL_CKPT}" ]; then
    echo "[chain] final checkpoint ${FINAL_CKPT} exists; training complete, not chaining."
    return 0
  fi
  local cur last
  cur=$(stat -c %Y "${CHECKPOINT}" 2>/dev/null || echo 0)
  last=$(cat "${PROG_FILE}" 2>/dev/null || echo -1)
  if [ "${cur}" = "${last}" ] && [ "${cur}" != "0" ]; then
    echo "[chain] model_last.pth mtime unchanged since last hop (${cur}) -> previous"
    echo "        window made no progress (crash?). Self-halting (touch ${STOP_FILE})."
    touch "${STOP_FILE}"; return 0
  fi
  echo "${cur}" > "${PROG_FILE}"
  local next
  next=$( cd "${SCRIPT_DIR}" && sbatch --parsable \
          --dependency=afterany:${SLURM_JOB_ID} "${SELF}" )
  echo "[chain] queued successor ${next} (afterany:${SLURM_JOB_ID})."
  echo "[chain] stop the chain any time with:  touch ${STOP_FILE}"
}

if [ -f "${STOP_FILE}" ]; then
  echo "[chain] STOP file present (${STOP_FILE}); not training. Exiting."
  echo "        (remove it to re-enable auto-resubmit.)"
  exit 0
fi

chain_next

if [ -f "${CHECKPOINT}" ]; then
  echo "[submit] resuming from ${CHECKPOINT}"
  RESUME_OPTS="--options resume=True weight=${CHECKPOINT}"
else
  echo "[submit] fresh start (no checkpoint at ${CHECKPOINT})"
  RESUME_OPTS=""
fi

apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd ${WORKDIR} && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config ${CONFIG} --num-gpus 4 ${RESUME_OPTS}"
