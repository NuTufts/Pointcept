#!/bin/bash
#
# Glue step between deterministic inference and nu_reco: rebuild the keypoint2
# output list from the freshly-written per-event files, and clean the downstream
# (nu_reco + eval) output dirs so the next stages start from a self-consistent set.
# CPU-only, fast. Chain it after the inference array:
#   sbatch --dependency=afterok:<infJID> regen_kp2_list.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=regenkp2
#SBATCH --output=logs/inference/regenkp2.%j.log
#SBATCH --error=logs/inference/regenkp2.%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=batch,preempt

set -eu
RECODIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/lartpc_data_prep/larformer_keypoint_v2
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/valdata_all_with_score_maps/}
KP2_LIST=${KP2_LIST:-${RECODIR}/outputlists/keypoint2_out_valdata_all.txt}
NU_RECO_DIR=${NU_RECO_DIR:-${RECODIR}/output/nu_reco_valdata_all/}
EVAL_DIR=${EVAL_DIR:-${RECODIR}/output/eval_reco_valdata_all/}

mkdir -p "$(dirname "${KP2_LIST}")"
# Only the nu-slice standard outputs: keypoint2_event<digits>_0.h5 (exclude any
# per-slice cosmic files from other studies). Absolute paths, sorted.
ls "${OUTPUT_DIR}"/keypoint2_event*_0.h5 2>/dev/null \
  | grep -E 'keypoint2_event[0-9]+_0\.h5$' | sort > "${KP2_LIST}"
echo ">>> keypoint2 list: $(grep -c . "${KP2_LIST}") files -> ${KP2_LIST}"

# clean downstream so stale shards can't survive into the new results
rm -f "${NU_RECO_DIR}"/nu_reco_shard*.h5 2>/dev/null || true
rm -f "${EVAL_DIR}"/eval_shard*.npz 2>/dev/null || true
echo ">>> cleaned ${NU_RECO_DIR} and ${EVAL_DIR}"
