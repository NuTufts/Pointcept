#!/bin/bash
#
# Sharded, GPU-array SLURM submission of the full-cascade keypoint-v2 INFERENCE
# (deghoster + slicer + Stage-3 particle segmenter + keypoint model) over a large
# input list. The list is split into NSHARDS contiguous chunks; each array task
# processes one chunk via --start-event / --n-events on
# tools/run_larformer_keypoint2_cascade_inference.py.
#
# Output filenames carry the GLOBAL dataset index (keypoint2_event{i:05d}_{ei}.h5),
# so all shards can safely write into the SAME OUTPUT_DIR without collisions.
#
# ---------------------------------------------------------------------------
# IMPORTANT: keep the #SBATCH --array range in sync with NSHARDS below.
#   NSHARDS=5  ->  --array=0-4
# The Tufts GPU-job cap is 6 concurrent jobs, so keep NSHARDS <= 6 (or add
# a %<N> throttle to --array, e.g. --array=0-9%6).
#
# You can override NSHARDS/array at submit time WITHOUT editing this file:
#   NSHARDS=5 sbatch --array=0-4 submit_larformer_keypoint2_fullcascade_inference_shard.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=kp2infer
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint="a100|l40s|l40|h100|rtx_6000_ada|rtx_a6000"
#SBATCH --constraint="a100|l40s|l40"
#SBATCH --output=logs/keypoint2_cascade_inference/kp2_infer_shard.%A_%a.%N.log
#SBATCH --error=logs/keypoint2_cascade_inference/kp2_infer_shard.%A_%a.%N.err
#SBATCH --array=0-4

set -eu

# ---- knobs ------------------------------------------------------------------
# Total number of events to spread across the shards. The final shard is clamped
# to the real dataset length inside the python tool, so over-provisioning here is
# harmless (e.g. TOTAL_EVENTS larger than the list just processes to the end).
NSHARDS=${NSHARDS:-5}
TOTAL_EVENTS=${TOTAL_EVENTS:-10000}

# ---- paths ------------------------------------------------------------------
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

CONFIG=${CONFIG:-configs/lartpc/larformer-keypoint2-fullcascade.py}
INPUT_LIST=${INPUT_LIST:-lartpc_data_prep/larformer_keypoint_v2/inputlists/merged_sp_valdata_all.txt}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKDIR}/lartpc_data_prep/larformer_keypoint_v2/output/valdata_all_with_score_maps/}

# Stage-3 particle ckpt + keypoint ckpt default from the config; override here.
PARTICLE_WEIGHTS=${PARTICLE_WEIGHTS:-}
KEYPOINT_WEIGHTS=${KEYPOINT_WEIGHTS:-}

# Set SAVE_SCORE_MAPS=1 to also write the dense head score maps (+ raw GT
# keypoints) for the visualizer's score-map panel.
SAVE_SCORE_MAPS=${SAVE_SCORE_MAPS:-1}

# ---- per-shard event range --------------------------------------------------
# ceil(TOTAL_EVENTS / NSHARDS) events per shard, contiguous, non-overlapping.
PER_SHARD=$(( (TOTAL_EVENTS + NSHARDS - 1) / NSHARDS ))
START_EVENT=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
N_EVENTS=${PER_SHARD}

echo ">>> shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}: start_event=${START_EVENT} n_events=${N_EVENTS}"

# ---- optional CLI overrides passed only when set ----------------------------
EXTRA_ARGS=""
[ -n "${PARTICLE_WEIGHTS}" ] && EXTRA_ARGS="${EXTRA_ARGS} --particle-weights ${PARTICLE_WEIGHTS}"
[ -n "${KEYPOINT_WEIGHTS}" ] && EXTRA_ARGS="${EXTRA_ARGS} --keypoint-weights ${KEYPOINT_WEIGHTS}"
[ -n "${SAVE_SCORE_MAPS}" ] && EXTRA_ARGS="${EXTRA_ARGS} --save-score-maps"

mkdir -p "${OUTPUT_DIR}"

# ---- run inside the apptainer container -------------------------------------
module load apptainer 2>/dev/null || true

# Ensure the nvidia-uvm device nodes exist on the host before apptainer --nv
# binds them; otherwise CUDA fails to initialize inside the container with
# "CUDA unknown error" even though nvidia-smi works.
nvidia-modprobe -u -c=0 2>/dev/null || true

apptainer exec --nv --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 tools/run_larformer_keypoint2_cascade_inference.py \
    --config ${CONFIG} \
    --input-list ${INPUT_LIST} \
    --output-dir ${OUTPUT_DIR} \
    --start-event ${START_EVENT} \
    --n-events ${N_EVENTS} \
    --device cuda \
    ${EXTRA_ARGS}
"

echo "DONE shard ${SLURM_ARRAY_TASK_ID} -> ${OUTPUT_DIR}"
