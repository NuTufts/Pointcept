#!/bin/bash
#
# run_stageB_capped.sh
# Pass-2 Stage B for the single-photon study: run the full LArFormer cascade on a
# CAPPED, explicit list of per-event merged H5 (the signal events selected by
# select_cascade_events.py), rather than per-file like run_stepB_cascade_wconfig.sh.
#
# Runs INSIDE the pointcept container (call it from submit_stageB_cascade.sh via
#   apptainer exec --nv ... bash -c "source run_stageB_capped.sh <config> <list> <out>")
# It mirrors run_stepB_cascade_wconfig.sh's env + checkpoint-reprefix + inference.
#
# Usage:
#   source run_stageB_capped.sh <config.conf> <input_list.txt> <output_dir>
#
# NOTE: no `set -u` — the sourced ubdl env scripts (config/setup.sh) reference
# unset vars (e.g. FORCE_LARLITE_BASEDIR) and are not set-u-safe, same as the
# proven run_stepB_cascade_wconfig.sh.
CONFIG=$1
INPUT_LIST=$2
ARG_OUTPUT_DIR=$3   # keep separate: the config also defines OUTPUT_DIR (Stage-A
                    # merged_sp), and sourcing it would clobber the passed value.

if [ ! -f "${CONFIG}" ];     then echo "ERROR: no config ${CONFIG}";     exit 1; fi
if [ ! -s "${INPUT_LIST}" ]; then echo "ERROR: empty/missing input list ${INPUT_LIST}"; exit 1; fi

# Pull POINTCEPT_DIR / UBDL_DIR / ckpt / threshold from the dataset config.
source "${CONFIG}"

# Restore the caller's output dir (the config's OUTPUT_DIR is Stage-A's merged_sp).
OUTPUT_DIR=${ARG_OUTPUT_DIR}
mkdir -p "${OUTPUT_DIR}"
LARFORMER_SCRIPTS=${POINTCEPT_DIR}/lartpc/data_prep/uboone_official
CLASS_PROB_THRESHOLD=${CLASS_PROB_THRESHOLD:-0.3}
LARFORMER_GT_SOURCE=${LARFORMER_GT_SOURCE:-deghost}
LARFORMER_CASCADE_CONFIG=${LARFORMER_CASCADE_CONFIG:-${POINTCEPT_DIR}/configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py}

echo "=========================================================="
echo "Stage B (capped cascade)"
echo "  config       : ${LARFORMER_CASCADE_CONFIG}"
echo "  particle ckpt: ${LARFORMER_PARTICLE_CKPT}"
echo "  input list   : ${INPUT_LIST}  ($(wc -l < ${INPUT_LIST}) events)"
echo "  output dir   : ${OUTPUT_DIR}"
echo "  thr=${CLASS_PROB_THRESHOLD}  gt_source=${LARFORMER_GT_SOURCE}"
echo "=========================================================="

# Cascade checkpoints + attention backend read from env by the python config.
# LARFORMER_FLASH_BACKEND: leave unset for flash_attn (Ampere+); the P100 submit
# script sets it to "xformers" (Pascal has no flash-attn kernel).
export LARFORMER_DEGHOSTER_CKPT LARFORMER_SLICER_CKPT LARFORMER_PARTICLE_CKPT \
       LARFORMER_SONATA_PRETRAIN LARFORMER_GT_SOURCE LARFORMER_FLASH_BACKEND

# Reduce CUDA fragmentation (helps avoid OOM on the 16 GB P100). The inference
# tool also guards each event with a per-event OOM skip.
# STABLE_ALLOC=1 (allocator-stability study): use the native allocator (no
# expandable_segments) AND skip the per-event empty_cache, to test whether the
# history-dependent allocator layout is the source of the list-order dependence.
if [ "${STABLE_ALLOC:-0}" = "1" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
    export LARFORMER_SKIP_EMPTY_CACHE=1
    echo "  STABLE_ALLOC: native allocator + no per-event empty_cache"
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# ubdl env (larlite/larcv for dataset slice_labels + h5py) + Pointcept on path.
cd "${UBDL_DIR}" && source setenv_pointcept_container.sh
export PYTHONPATH=${POINTCEPT_DIR}:${PYTHONPATH}

# Re-prefix the standalone Stage-3 checkpoint to particle_segmenter.* (cached).
REPREFIXED_CKPT="${OUTPUT_DIR}/$(basename ${LARFORMER_PARTICLE_CKPT%.pth}).particle_segmenter.pth"
if [ ! -f "${REPREFIXED_CKPT}" ]; then
    echo "Re-prefixing Stage-3 checkpoint -> particle_segmenter.*"
    python3 "${LARFORMER_SCRIPTS}/prefix_particle_ckpt.py" \
        --in "${LARFORMER_PARTICLE_CKPT}" --out "${REPREFIXED_CKPT}" || {
        echo "ERROR: checkpoint re-prefix failed"; exit 1; }
fi

# Optional slicer-side query dedup (SLICER_DEDUP_IOU>0 enables it; 0/unset = off).
SLICER_DEDUP_FLAG=""
if [ -n "${SLICER_DEDUP_IOU:-}" ] && awk "BEGIN{exit !(${SLICER_DEDUP_IOU}+0 > 0)}" 2>/dev/null; then
    SLICER_DEDUP_FLAG="--slicer-dedup-iou ${SLICER_DEDUP_IOU}"
    echo "  slicer dedup: ${SLICER_DEDUP_FLAG}"
fi

# Optional Stage-1 deghoster keep-threshold override (DEGHOST_THRESHOLD_VAL).
# Lower than the 0.5 default keeps more (lower-confidence) spacepoints.
DEGHOST_FLAG=""
if [ -n "${DEGHOST_THRESHOLD_VAL:-}" ]; then
    DEGHOST_FLAG="--deghost-threshold-val ${DEGHOST_THRESHOLD_VAL}"
    echo "  deghost τ override: ${DEGHOST_THRESHOLD_VAL}"
fi

# Optional deterministic (repeatable) inference for physics production.
# DETERMINISTIC=1 fixes seeds + disables TF32 + forces deterministic cuBLAS/cuDNN.
# CUBLAS_WORKSPACE_CONFIG must be exported BEFORE the process starts (cuBLAS reads
# it at init), so set it here rather than only inside Python.
DETERMINISTIC_FLAG=""
if [ "${DETERMINISTIC:-0}" = "1" ]; then
    DETERMINISTIC_FLAG="--deterministic"
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    echo "  deterministic: ON (TF32 off, deterministic cuBLAS/cuDNN; ~1.3-2x slower)"
fi

# Optional flash-recovery expanded segmenter flow (FLASH_RECOVER_K>0 enables it).
FLASH_RECOVER_FLAG=""
if [ -n "${FLASH_RECOVER_K:-}" ] && [ "${FLASH_RECOVER_K}" -gt 0 ] 2>/dev/null; then
    FLASH_RECOVER_FLAG="--flash-recover-k ${FLASH_RECOVER_K} \
        --flash-recover-chi2-max ${FLASH_RECOVER_CHI2_MAX:-500} \
        --flash-recover-oob-max ${FLASH_RECOVER_OOB_MAX:-0.05} \
        --flash-recover-gamma ${FLASH_RECOVER_GAMMA:-5.25}"
    [ "${FLASH_RECOVER_AUGMENT_ALL:-0}" = "1" ] && FLASH_RECOVER_FLAG="${FLASH_RECOVER_FLAG} --flash-recover-augment-all"
    echo "  flash recover: K=${FLASH_RECOVER_K} chi2<=${FLASH_RECOVER_CHI2_MAX:-500} oob<=${FLASH_RECOVER_OOB_MAX:-0.05} augment_all=${FLASH_RECOVER_AUGMENT_ALL:-0}"
fi

python3 "${POINTCEPT_DIR}/tools/run_larformer_stage3_inference.py" \
    --input-mode full-cascade \
    --config "${LARFORMER_CASCADE_CONFIG}" \
    --weights "${REPREFIXED_CKPT}" \
    --input-list "${INPUT_LIST}" \
    --output-dir "${OUTPUT_DIR}" \
    --class-prob-threshold "${CLASS_PROB_THRESHOLD}" \
    ${SLICER_DEDUP_FLAG} ${FLASH_RECOVER_FLAG} ${DETERMINISTIC_FLAG} ${DEGHOST_FLAG} \
    --device cuda --no-gt
rc=$?
echo "Stage B exit status: ${rc}"
ls -lh "${OUTPUT_DIR}"/stage3pred_*.h5 2>/dev/null
exit ${rc}
