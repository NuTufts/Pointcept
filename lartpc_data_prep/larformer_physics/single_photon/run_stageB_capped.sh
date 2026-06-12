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
LARFORMER_SCRIPTS=${POINTCEPT_DIR}/lartpc_data_prep/larformer_scripts
CLASS_PROB_THRESHOLD=${CLASS_PROB_THRESHOLD:-0.3}
LARFORMER_GT_SOURCE=${LARFORMER_GT_SOURCE:-deghost}
LARFORMER_CASCADE_CONFIG=${LARFORMER_CASCADE_CONFIG:-${POINTCEPT_DIR}/configs/lartpc/larformer-particle-fullcascade-ptv3crosslevel.py}

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

python3 "${POINTCEPT_DIR}/tools/run_larformer_stage3_inference.py" \
    --input-mode full-cascade \
    --config "${LARFORMER_CASCADE_CONFIG}" \
    --weights "${REPREFIXED_CKPT}" \
    --input-list "${INPUT_LIST}" \
    --output-dir "${OUTPUT_DIR}" \
    --class-prob-threshold "${CLASS_PROB_THRESHOLD}" \
    --device cuda --no-gt
rc=$?
echo "Stage B exit status: ${rc}"
ls -lh "${OUTPUT_DIR}"/stage3pred_*.h5 2>/dev/null
exit ${rc}
