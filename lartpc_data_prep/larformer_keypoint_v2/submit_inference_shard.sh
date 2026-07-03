#!/bin/bash
#
# Sharded, GPU-array DETERMINISTIC full-cascade keypoint-v2 INFERENCE over the
# 10k valdata sample. The merged_sp list is split into NSHARDS contiguous chunks;
# each array task runs the cascade (deghoster + slicer + Stage-3 segmenter +
# keypoint model) over its chunk via --start-event/--n-events and writes per-event
# keypoint2_event{i:05d}_0.h5 (i = SORTED index in the merged_sp list) into
# OUTPUT_DIR.
#
# REPRODUCIBILITY: runs with --deterministic (set_deterministic + reseed_per_event)
# pinned to A100 (Ampere -- the conforming family). Without this the slicer's
# query->slice assignment + downstream hard cuts churn ~9% run-to-run. See
# docs/LArFormer_Reproducibility.md. reseed_per_event makes each event's output
# independent of its position, so sharding is safe (bit-identical to a serial run).
#
# Clean the output dir ONCE before submitting the array (the orchestrator does
# this; do NOT clean per-task or tasks delete each other's output).
#
# Submit:
#   NSHARDS=16 sbatch --array=0-15 submit_inference_shard.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=kp2inf
#SBATCH --output=logs/inference/kp2inf.%A_%a.%N.log
#SBATCH --error=logs/inference/kp2inf.%A_%a.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,preempt,wongjiradlab
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --array=0-15

set -eu

NSHARDS=${NSHARDS:-16}
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc_data_prep/larformer_keypoint_v2

CONFIG=${CONFIG:-configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade.py}
INPUT_LIST=${INPUT_LIST:-${RECODIR}/inputlists/merged_sp_valdata_all.txt}
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/valdata_all_with_score_maps/}

# ---- per-shard contiguous range over the merged_sp list ---------------------
NLINES=$(grep -c . "${INPUT_LIST}")
PER_SHARD=$(( (NLINES + NSHARDS - 1) / NSHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
N=${PER_SHARD}

echo ">>> shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}: ${NLINES} events; "
echo "    start-event=${START} n-events=${N} -> ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}" "${WORKDIR}/logs/inference"
if [ "${START}" -ge "${NLINES}" ]; then
  echo ">>> start ${START} >= ${NLINES}; nothing to do."; exit 0
fi

module load apptainer 2>/dev/null || true

# Transient "CUDA unknown error" on model.to(cuda) hits some nodes even though
# nvidia-smi works. Retry a few times (re-running nvidia-modprobe) so one flaky
# node doesn't waste the shard; --deterministic + reseed_per_event makes a rerun
# of the same range bit-identical, so a retry is safe.
rc=1
for attempt in 1 2 3; do
  echo ">>> attempt ${attempt} on $(hostname)"
  nvidia-modprobe -u -c=0 2>/dev/null || true
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  apptainer exec --nv --bind /cluster:/cluster "${container}" bash -c "
    cd ${WORKDIR} && \
    source setenv_pointcept_only.sh && \
    python3 tools/run_larformer_keypoint2_cascade_inference.py \
      --config ${CONFIG} \
      --input-list ${INPUT_LIST} \
      --output-dir ${OUTPUT_DIR} \
      --start-event ${START} \
      --n-events ${N} \
      --deterministic \
      --save-score-maps \
      --device cuda
  " && { rc=0; break; }
  echo ">>> attempt ${attempt} failed (rc=$?); retrying in 10s"; sleep 10
done
[ "${rc}" -eq 0 ] || { echo ">>> shard ${SLURM_ARRAY_TASK_ID} FAILED after retries"; exit 1; }

echo "DONE shard ${SLURM_ARRAY_TASK_ID} -> ${OUTPUT_DIR}"
