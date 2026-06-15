#!/bin/bash
#
# submit_capture.sh — Tier-A capture (capture_cascade_tensors.py) for the cross-GPU test
# with the same checkpoint-env setup run_stageB_capped.sh uses. Parametrized by env:
#   CAPTURE_DIR  (required)  where capture_*.npz go
#   EVENT_LIST   (default cascade_inputs_1g0X.txt)
#   MAXEV        (default empty = all)
#   DEGHOST_THRESHOLD_VAL (optional τ override)
#   DETERM       (default 1; set 0 for same-GPU run-to-run capture)
# Pin to a node/SKU at submit time:  sbatch --nodelist=pax052 --export=ALL,CAPTURE_DIR=... submit_capture.sh

#SBATCH --job-name=sp_capture
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/repeatability_tests/slurm/logs/capture/cap.%j.log
#SBATCH --error=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/larformer_physics/repeatability_tests/slurm/logs/capture/cap.%j.err

POINTCEPT=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
REPEAT=${POINTCEPT}/lartpc_data_prep/larformer_physics/repeatability_tests
SP=${POINTCEPT}/lartpc_data_prep/larformer_physics/single_photon   # input lists live here
CONFIG=${POINTCEPT}/lartpc_data_prep/larformer_scripts/larformer_configs/single_photon_scale1500.conf
SIF=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

: "${CAPTURE_DIR:?set CAPTURE_DIR}"
EVENT_LIST=${EVENT_LIST:-${SP}/workdir_scale/cascade_inputs_1g0X.txt}
DETERM=${DETERM:-1}
mkdir -p ${CAPTURE_DIR} ${REPEAT}/slurm/logs/capture

DET_FLAG=""; [ "${DETERM}" = "0" ] && DET_FLAG="--no-deterministic"   # capture is deterministic by default
DG_FLAG=""; [ -n "${DEGHOST_THRESHOLD_VAL:-}" ] && DG_FLAG="--deghost-threshold-val ${DEGHOST_THRESHOLD_VAL}"
FP64_FLAG=""; [ "${DEGHOST_FP64:-0}" = "1" ] && FP64_FLAG="--deghost-fp64"
NOSHUF_FLAG=""; [ "${DEGHOST_NOSHUFFLE:-0}" = "1" ] && NOSHUF_FLAG="--deghost-no-shuffle"
MAX_FLAG=""; [ -n "${MAXEV:-}" ] && MAX_FLAG="--max-events ${MAXEV}"

module load apptainer/1.4.0 2>/dev/null || true
apptainer exec --nv --bind /cluster:/cluster ${SIF} bash -c "
set -e
source ${CONFIG}
# Optional attention-backend override (FLASH_BACKEND=xformers|flash_attn).
${FLASH_BACKEND:+export LARFORMER_FLASH_BACKEND=${FLASH_BACKEND}}
echo \"attn backend = \${LARFORMER_FLASH_BACKEND:-flash_attn}\"
# env the cascade config reads for its sub-checkpoints (same as run_stageB_capped.sh)
export LARFORMER_DEGHOSTER_CKPT LARFORMER_SLICER_CKPT LARFORMER_PARTICLE_CKPT \
       LARFORMER_SONATA_PRETRAIN LARFORMER_GT_SOURCE LARFORMER_FLASH_BACKEND
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd \${UBDL_DIR} && source setenv_pointcept_container.sh >/dev/null 2>&1
export PYTHONPATH=${POINTCEPT}:\${PYTHONPATH}
# re-prefix the standalone stage-3 ckpt -> particle_segmenter.* (cached in CAPTURE_DIR)
REPREFIXED=${CAPTURE_DIR}/\$(basename \${LARFORMER_PARTICLE_CKPT%.pth}).particle_segmenter.pth
[ -f \${REPREFIXED} ] || python3 ${POINTCEPT}/lartpc_data_prep/larformer_scripts/prefix_particle_ckpt.py \
    --in \${LARFORMER_PARTICLE_CKPT} --out \${REPREFIXED}
echo 'GPU:'; nvidia-smi --query-gpu=name --format=csv,noheader
python3 ${REPEAT}/capture_cascade_tensors.py \
    --config \${LARFORMER_CASCADE_CONFIG:-${POINTCEPT}/configs/lartpc/larformer-particle-fullcascade-ptv3crosslevel.py} \
    --weights \${REPREFIXED} \
    --input-list ${EVENT_LIST} \
    --capture-dir ${CAPTURE_DIR} \
    --class-prob-threshold \${CLASS_PROB_THRESHOLD:-0.3} \
    ${DET_FLAG} ${DG_FLAG} ${FP64_FLAG} ${NOSHUF_FLAG} ${MAX_FLAG} --device cuda
"
echo "CAPTURE DONE -> ${CAPTURE_DIR}"
