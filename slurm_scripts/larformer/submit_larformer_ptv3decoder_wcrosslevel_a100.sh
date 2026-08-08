#!/bin/bash
# LArFormer Stage-2 slicer retrain -- 300k max_spacepoints cap on 4x A100-80GB.
#
# Runs from the kpv2_pointcept checkout (which carries the raised cap + the
# intra-epoch val-loss probe). batch_size=16 total = 4/GPU physical,
# gradient_accumulation_steps=1 -> real batch 16 (like the old run, no accum).
# The a100 nodes are 8-GPU (pax007/049/050/051/052/105/106/142/143/144), so 4 fit
# on one node. Worst-case smoke peaked at 41.5 GB/GPU. Auto fresh-start on first
# launch, auto-resume once a model_last.pth exists.
#
# SELF-RESUBMIT (unattended multi-window training):
#   Epochs are ~45h but the wall-clock cap is 48h, so each job does ~1 epoch.
#   SLURM's SIGUSR1 does NOT propagate through apptainer->bash->python here
#   (verified: SignalCheckpointHook never fires), so we do NOT rely on the
#   RESUBMIT marker. Instead each job PRE-SUBMITS a successor gated
#   afterany:$SLURM_JOB_ID -- it starts when this job ends for ANY reason
#   (including a hard TIME-LIMIT kill) and auto-resumes from model_last.pth.
#
#   TO STOP THE CHAIN:  touch <SAVE>/STOP_AUTORESUBMIT
#     (the next queued job no-ops and exits; the currently-running job finishes
#      its window unless you also scancel it).
#   Crash-loop guard: if model_last.pth's mtime did not advance since the last
#   chain hop, the window is assumed to have crashed and the chain self-halts.
#
#   sbatch submit_larformer_ptv3decoder_wcrosslevel_a100.sh

#SBATCH --job-name=lf-slicer-cap300k
#SBATCH --mem=192G
#SBATCH --cpus-per-task=48
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=pax141
#SBATCH --output=logs/lf-slicer-cap300k.%j.%N.log
#SBATCH --error=logs/lf-slicer-cap300k.%j.%N.err

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage2_slicer/larformer-slicer-v1-cascaded-ptv3hybrid-crosslevel.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SCRIPT_DIR=${WORKDIR}/slurm_scripts/larformer
SELF=${SCRIPT_DIR}/submit_larformer_ptv3decoder_wcrosslevel_a100.sh

# save_path is relative in the config -> resolves under ${WORKDIR}/exp/...
SAVE=${WORKDIR}/exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_nonzeroinit_maskdn_noamp_cap300k
CHECKPOINT=${SAVE}/model/model_last.pth
FINAL_CKPT=${SAVE}/model/epoch_50.pth          # config epoch=50 -> training done
STOP_FILE=${SAVE}/STOP_AUTORESUBMIT
PROG_FILE=${SAVE}/.autoresubmit_last_ckpt_mtime

module load apptainer
mkdir -p "${SAVE}" "${SCRIPT_DIR}/logs"

# ---- self-resubmit chaining -------------------------------------------------
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

# Queue the successor NOW (before training) so a hard TIME-LIMIT kill still
# leaves a job waiting to take over.
chain_next
# ---------------------------------------------------------------------------

# Auto fresh-vs-resume: resume iff a checkpoint already exists.
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
