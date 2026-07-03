#!/bin/bash
#
# run_stepB_cascade_wconfig.sh
# Stage B of the LArFormer cascade data path. Runs inside the pointcept
# container (GPU). Config-driven.
#
# Runs the full LArFormer cascade (LoRA deghoster -> ptv3crosslevel slicer ->
# ptv3crosslevel particle segmenter) on the per-event merged_*.h5 produced by
# Stage A, via tools/larformer/run_larformer_stage3_inference.py --input-mode full-cascade.
# Writes stage3pred_*.h5 (slicer half + particle half) into the workdir.
#
# Two invocation modes:
#   1) Driver mode:
#        source run_stepB_cascade_wconfig.sh <workdir> <fileno> <tag> <isdata> [keep]
#   2) Standalone:
#        source run_stepB_cascade_wconfig.sh <config_file> [lineno]
#

WCONFIG_CONTAINER_TYPE=pointcept
source "$(dirname "${BASH_SOURCE[0]}")/_wconfig_common.sh"
wconfig_bootstrap "$@"; _bs_rc=$?
if [ "${WCONFIG_RE_EXECED:-0}" = "1" ] || [ ${_bs_rc} -ne 0 ]; then
    return ${_bs_rc} 2>/dev/null || exit ${_bs_rc}
fi
if [ "${WCONFIG_STANDALONE:-0}" = "1" ]; then
    ISDATAFLAG=${ISDATAFLAG:-""}
    set -- "${WCONFIG_WORKDIR}" "${WCONFIG_FILENO}" "${TAG}" "${ISDATAFLAG}" \
           "${KEEP_INTERMEDIATES}"
fi

WORKDIR_PATH=$1
FILENO=$2
FILETAG=$3
ISDATAFLAG=${4:-""}
KEEP_INTERMEDIATES=${5:-0}

if [ -z "${WORKDIR_PATH}" ] || [ -z "${FILENO}" ] || [ -z "${FILETAG}" ]; then
    echo "ERROR: usage: source run_stepB_cascade_wconfig.sh <workdir> <fileno> <tag> ..."
    return 1 2>/dev/null || exit 1
fi
ZFILENO=$(printf "%05d" ${FILENO})

if [ -z "${SCRIPT_DIR:-}" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
POINTCEPT_DIR=${POINTCEPT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}

# Cascade config + checkpoints (config has env-overridable defaults).
LARFORMER_CASCADE_CONFIG=${LARFORMER_CASCADE_CONFIG:-${POINTCEPT_DIR}/configs/lartpc/larformer/stage3_particle/larformer-particle-fullcascade-ptv3crosslevel.py}
LARFORMER_PARTICLE_CKPT=${LARFORMER_PARTICLE_CKPT:-${POINTCEPT_DIR}/exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_lr1e4_bugfixed/model_iter_98652.pth}
CLASS_PROB_THRESHOLD=${CLASS_PROB_THRESHOLD:-0.3}
DEVICE=${DEVICE:-cuda}

# Pure-inference path for ALL dataset types: the frozen 3-class slicer cannot
# consume 7-class particle GT in its eval matcher (CUDA-asserts), so we run the
# whole cascade GT-less. gt_source="deghost" emits no instances (slicer/particle
# matchers stay clean) and `particle_class_id` falls back to a -1 stub (the
# trained per-level cls head still runs for mixed_query_selection; only its
# unused loss target is masked). GT-matched eval metrics are a separate concern
# — use the cached inference path for those.
LARFORMER_GT_SOURCE=${LARFORMER_GT_SOURCE:-deghost}

# Export checkpoint + dataset overrides so the python config picks them up.
export LARFORMER_DEGHOSTER_CKPT LARFORMER_SLICER_CKPT LARFORMER_PARTICLE_CKPT \
       LARFORMER_SONATA_PRETRAIN LARFORMER_GT_SOURCE

cd "${WORKDIR_PATH}" || { echo "cannot cd to ${WORKDIR_PATH}"; return 1 2>/dev/null || exit 1; }
echo "Stage B workdir: $(pwd)"
echo "  config  : ${LARFORMER_CASCADE_CONFIG}"
echo "  stage3  : ${LARFORMER_PARTICLE_CKPT}"
echo "  isdata='${ISDATAFLAG}'  class_prob_threshold=${CLASS_PROB_THRESHOLD}"

# Stage-level skip
has_pred=$(ls "${WORKDIR_PATH}"/stage3pred_*.h5 2>/dev/null | head -1)
if [ "${KEEP_INTERMEDIATES}" = "1" ] && [ -n "${has_pred}" ]; then
    echo "Stage B: skipped (stage3pred H5 present)"
    return 0 2>/dev/null || exit 0
fi

# Input list of merged H5s for this file.
ls "${WORKDIR_PATH}"/merged_${FILETAG}_fileno${ZFILENO}_entry*.h5 > "${WORKDIR_PATH}/stageB_inputs.txt" 2>/dev/null
if [ ! -s "${WORKDIR_PATH}/stageB_inputs.txt" ]; then
    echo "ERROR: no merged_${FILETAG}_fileno${ZFILENO}_entry*.h5 in workdir"
    return 1 2>/dev/null || exit 1
fi
echo "  $(wc -l < ${WORKDIR_PATH}/stageB_inputs.txt) merged H5 events to process"

# ubdl env (larlite/larcv for the dataset's slice_labels + h5py) + Pointcept.
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
cd "${UBDL_DIR}"
source setenv_pointcept_container.sh
export PYTHONPATH=${POINTCEPT_DIR}:${PYTHONPATH}
cd "${WORKDIR_PATH}"

# Re-prefix the standalone Stage-3 checkpoint to particle_segmenter.* (cached).
REPREFIXED_CKPT="${WORKDIR_PATH}/$(basename ${LARFORMER_PARTICLE_CKPT%.pth}).particle_segmenter.pth"
if [ ! -f "${REPREFIXED_CKPT}" ]; then
    echo "Re-prefixing Stage-3 checkpoint -> particle_segmenter.*"
    python3 "${SCRIPT_DIR}/prefix_particle_ckpt.py" \
        --in "${LARFORMER_PARTICLE_CKPT}" --out "${REPREFIXED_CKPT}" || {
        echo "ERROR: checkpoint re-prefix failed"; return 1 2>/dev/null || exit 1; }
fi

# Always GT-less (see LARFORMER_GT_SOURCE rationale above).
NOGT="--no-gt"

CMD="python3 ${POINTCEPT_DIR}/tools/larformer/run_larformer_stage3_inference.py \
    --input-mode full-cascade \
    --config ${LARFORMER_CASCADE_CONFIG} \
    --weights ${REPREFIXED_CKPT} \
    --input-list ${WORKDIR_PATH}/stageB_inputs.txt \
    --output-dir ${WORKDIR_PATH} \
    --class-prob-threshold ${CLASS_PROB_THRESHOLD} \
    --device ${DEVICE} ${NOGT}"
echo "${CMD}"
${CMD}
stepB_status=$?
echo "Stage B exit status: ${stepB_status}"
if [ ${stepB_status} -ne 0 ]; then
    echo "WARNING: Stage B cascade failed (exit ${stepB_status}). Stage A merged"
    echo "         H5 outputs are unaffected."
    return ${stepB_status} 2>/dev/null || exit ${stepB_status}
fi

echo "Stage B output files:"
ls -lh "${WORKDIR_PATH}"/stage3pred_*.h5 2>/dev/null
echo "Stage B complete"
