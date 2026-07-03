#!/bin/bash
#
# STUDY: run the Stage-3 particle segmenter + keypoint model SEPARATELY on EVERY
# slice (nu + each cosmic slice) for the single-visible-photon sample, to see what
# the downstream stages do when handed the cosmic photon slices. Writes one
# keypoint2_event{i}_{slice}_{ei}.h5 per (event, slice). Very expensive (one full
# Stage-3 pass per slice) -- only for this small 85-event study.
#
#   sbatch submit_all_slices_singlephoton.sh                 # full 85 events
#   N_EVENTS=3 sbatch submit_all_slices_singlephoton.sh      # smoke test
# ---------------------------------------------------------------------------

#SBATCH --job-name=allslices
#SBATCH --output=logs/all_slices/allslices.%j.%N.log
#SBATCH --error=logs/all_slices/allslices.%j.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,preempt,wongjiradlab
# Pin to A100 (Ampere): the ONLY conforming family for a repeatable measurement.
# H100/H200 (Hopper) diverge ~1.9% at the event level -- see
# docs/LArFormer_Reproducibility.md sec 4.3. L40S also conforms if you need it.
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue

set -eu
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc_data_prep/larformer_keypoint_v2

CONFIG=${CONFIG:-configs/lartpc/larformer-keypoint2-fullcascade.py}
INPUT_LIST=${INPUT_LIST:-${RECODIR}/inputlists/merged_sp_valdata_singlephoton.txt}
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/all_slices_singlephoton/}
N_EVENTS=${N_EVENTS:--1}
SLICE_MIN_POINTS=${SLICE_MIN_POINTS:-20}
# LOOSE=1 -> add --loose-fallback: for a slice with NO above-threshold instance,
#   recover the single below-threshold query with highest P(gamma) (loose_class_id=1),
#   flagged loose_pass in the output. loose_conf holds that P(gamma).
EXTRA_ARGS=""
[ "${LOOSE:-0}" = "1" ] && EXTRA_ARGS="${EXTRA_ARGS} --loose-fallback"
# DETERMINISTIC=1 (default) -> --deterministic: set_deterministic() before build +
#   reseed_per_event() each event (TF32 off, deterministic algorithms, seeded).
#   REQUIRED for a repeatable slice/instance assignment -- without it, the slicer's
#   query->slice mapping churns run-to-run (docs/LArFormer_Reproducibility.md).
#   Set DETERMINISTIC=0 only for fast exploration where run-to-run drift is OK.
[ "${DETERMINISTIC:-1}" = "1" ] && EXTRA_ARGS="${EXTRA_ARGS} --deterministic"

mkdir -p "${OUTPUT_DIR}" "${WORKDIR}/logs/all_slices"
# Clean prior per-slice outputs: slicer query indices differ run-to-run (GPU
# nondeterminism), so stale files from an earlier run would carry cosmicNN labels
# that no longer match this run's sidecar. Start fresh for a self-consistent set.
rm -f "${OUTPUT_DIR}"/keypoint2_event*.h5 "${OUTPUT_DIR}"/sliceid_event*.h5
module load apptainer/1.2.4-suid 2>/dev/null || module load apptainer 2>/dev/null || true
nvidia-modprobe -u -c=0 2>/dev/null || true

apptainer exec --nv --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 tools/run_larformer_keypoint2_cascade_inference.py \
    --config ${CONFIG} \
    --input-list ${INPUT_LIST} \
    --output-dir ${OUTPUT_DIR} \
    --n-events ${N_EVENTS} \
    --all-slices \
    --slice-min-points ${SLICE_MIN_POINTS} \
    --save-score-maps \
    ${EXTRA_ARGS} \
    --device cuda
"
echo "DONE -> ${OUTPUT_DIR}"
