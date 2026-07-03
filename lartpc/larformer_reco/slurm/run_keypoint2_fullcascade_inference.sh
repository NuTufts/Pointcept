#!/bin/bash
#
# Single-config full-cascade keypoint-v2 INFERENCE.
#
# Runs the whole attempt-2 keypoint pipeline (deghoster + slicer + Stage-3
# particle segmenter + keypoint model) on raw merged_h5 events from ONE config,
# writing per-event keypoint H5s (per-particle start/end + dense nu vertex, in
# detector cm) to OUTPUT_DIR.
#
# Run interactively ON an a100 node (you already hold one):
#     cd lartpc/larformer_reco
#     bash run_keypoint2_fullcascade_inference.sh
#
# Everything (cascade composition, keypoint model, test dataset, the Stage-3
# particle ckpt + keypoint ckpt paths, max_spacepoints=500000) is baked into the
# config; override per run with the env vars below.

set -eu

# ---- paths (edit WORKDIR if your checkout lives elsewhere) -------------------
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

CONFIG=${CONFIG:-configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-fullcascade.py}
#INPUT_LIST=${INPUT_LIST:-lartpc/larformer_reco/inputlists/merged_sp_mcc9_v29e_dl_run3b_bnb_nu_overlay_50event_test.txt}
#OUTPUT_DIR=${OUTPUT_DIR:-${WORKDIR}/lartpc/larformer_reco/output/test_50events_with_score_maps/}

#INPUT_LIST=lartpc/larformer_reco/inputlists/merged_sp_valdata_pi0filtered_50events.txt
#OUTPUT_DIR=${OUTPUT_DIR:-${WORKDIR}/lartpc/larformer_reco/output/valdata_pi0filtered_50events_with_score_maps/}

INPUT_LIST=lartpc/larformer_reco/inputlists/merged_sp_valdata_all.txt
OUTPUT_DIR=${OUTPUT_DIR:-${WORKDIR}/lartpc/larformer_reco/output/valdata_all_with_score_maps/}

N_EVENTS=${N_EVENTS:--1}        # -1 = all events in the list

# Stage-3 particle ckpt + keypoint ckpt default from the config; override here.
PARTICLE_WEIGHTS=${PARTICLE_WEIGHTS:-}
KEYPOINT_WEIGHTS=${KEYPOINT_WEIGHTS:-}

# Set SAVE_SCORE_MAPS=1 to also write the object/no-object + nu-vertex dense
# head score maps (+ raw GT keypoints) for the visualizer's score-map panel.
SAVE_SCORE_MAPS=${SAVE_SCORE_MAPS:-}

# Optional CLI overrides passed only when set.
EXTRA_ARGS=""
[ -n "${PARTICLE_WEIGHTS}" ] && EXTRA_ARGS="${EXTRA_ARGS} --particle-weights ${PARTICLE_WEIGHTS}"
[ -n "${KEYPOINT_WEIGHTS}" ] && EXTRA_ARGS="${EXTRA_ARGS} --keypoint-weights ${KEYPOINT_WEIGHTS}"
[ -n "${SAVE_SCORE_MAPS}" ] && EXTRA_ARGS="${EXTRA_ARGS} --save-score-maps"

mkdir -p "${OUTPUT_DIR}"

# ---- run inside the apptainer container --------------------------------------
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
    --n-events ${N_EVENTS} \
    --device cuda \
    --save-score-maps \
    ${EXTRA_ARGS}
"

echo "DONE -> ${OUTPUT_DIR}"
