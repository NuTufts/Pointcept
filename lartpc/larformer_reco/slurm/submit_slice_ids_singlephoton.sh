#!/bin/bash
#
# Dump per-event full-cloud SLICE-ID sidecars for the single-visible-photon
# sample. Runs ONLY the cascade deghoster+slicer (--slice-ids-only) per event and
# writes sliceid_event{i}.h5 with, for every spacepoint, slice_id in
#   {-2 ghost, -1 kept-unclustered, 0 nu-slice, 1..K cosmic-slice-k} + deghost
# P(real). Written for EVERY event, including those with no nu slice -- so you can
# see whether a lost photon was deghosted, mis-sliced as cosmic, or unclustered.
#
#   sbatch submit_slice_ids_singlephoton.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=sliceids
#SBATCH --output=logs/slice_ids/sliceids.%j.%N.log
#SBATCH --error=logs/slice_ids/sliceids.%j.%N.err
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -eu
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco

CONFIG=${CONFIG:-configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade.py}
INPUT_LIST=${INPUT_LIST:-${RECODIR}/inputlists/merged_sp_valdata_singlephoton.txt}
SID_DIR=${SID_DIR:-${RECODIR}/output/slice_ids_singlephoton/}

mkdir -p "${SID_DIR}" "${WORKDIR}/logs/slice_ids"
module load apptainer/1.2.4-suid 2>/dev/null || module load apptainer 2>/dev/null || true
# Ensure nvidia-uvm device nodes exist before apptainer --nv (see nu_reco script).
nvidia-modprobe -u -c=0 2>/dev/null || true

apptainer exec --nv --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 tools/larformer/run_larformer_keypoint2_cascade_inference.py \
    --config ${CONFIG} \
    --input-list ${INPUT_LIST} \
    --output-dir ${SID_DIR} \
    --slice-id-dir ${SID_DIR} \
    --slice-ids-only \
    --device cuda
"
echo "DONE -> ${SID_DIR}"
