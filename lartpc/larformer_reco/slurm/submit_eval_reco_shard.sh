#!/bin/bash
#
# Sharded, CPU-array SLURM submission of the RECO PERFORMANCE EVAL
# (eval_reco_performance.py). The MERGED_SP list (the denominator) is split into
# NSHARDS contiguous chunks; each array task runs the eval over its chunk via
# --start/--n and writes one partial record table (eval_shard{START:07d}.npz)
# into OUTPUT_DIR. A follow-up merge job (submit_eval_reco_merge.sh) concatenates
# them, prints the summary, and makes the plots.
#
# The eval is pure h5py+numpy (CPU-only, no GPU). Each shard independently builds
# the (small) keypoint2 src_file->file index and preloads the nu_reco shards;
# that work is redundant across shards but cheap and runs in parallel, so no
# shared cache is needed.
#
# Submit the array, then a merge job that waits for it:
#   JID=$(NSHARDS=10 sbatch --parsable --array=0-9 submit_eval_reco_shard.sh)
#   sbatch --dependency=afterok:${JID} submit_eval_reco_merge.sh
# (keep --array=0-(NSHARDS-1) consistent with NSHARDS.)
# ---------------------------------------------------------------------------

#SBATCH --job-name=evalreco
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/eval_reco/eval_reco_shard.%A_%a.%N.log
#SBATCH --error=logs/eval_reco/eval_reco_shard.%A_%a.%N.err
#SBATCH --array=0-9

set -eu

# ---- knobs ------------------------------------------------------------------
NSHARDS=${NSHARDS:-10}
# denominator restriction; blank it (PRIMARIES_ONLY="") for all nu-origin.
PRIMARIES_ONLY=${PRIMARIES_ONLY:---primaries-only}
# reco stream to evaluate ('nu' or 'flashmatch'); the KEYPOINT2_LIST must be
# the matching per-stream list (see regen_kp2_list.sh) and NU_RECO_DIR the
# nu_reco run made from that same list.
STREAM=${STREAM:-nu}
# photon denominator visibility cut [MeV]; 0 disables (see --gamma-min-evis)
GAMMA_MIN_EVIS=${GAMMA_MIN_EVIS:-20}

# ---- paths ------------------------------------------------------------------
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco

KEYPOINT2_LIST=${KEYPOINT2_LIST:-${RECODIR}/outputlists/keypoint2_out_valdata_all.txt}
MERGED_SP_LIST=${MERGED_SP_LIST:-${RECODIR}/inputlists/merged_sp_valdata_all.txt}
NU_RECO_DIR=${NU_RECO_DIR:-${RECODIR}/output/nu_reco_valdata_all/}
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/eval_reco_valdata_all/}

# ---- per-shard range over the MERGED_SP list --------------------------------
# ceil(NLINES / NSHARDS) files per shard, contiguous, non-overlapping. The eval
# clamps the last shard to the real list length.
NLINES=$(grep -c . "${MERGED_SP_LIST}")
PER_SHARD=$(( (NLINES + NSHARDS - 1) / NSHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
N=${PER_SHARD}
OUT=$(printf "%s/eval_shard%07d.npz" "${OUTPUT_DIR}" "${START}")

echo ">>> shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}: merged_sp list has ${NLINES} "
echo "    files; start=${START} n=${N} -> ${OUT}"

mkdir -p "${OUTPUT_DIR}" "${WORKDIR}/logs/eval_reco"

# stop cleanly if this shard starts past the end of the list
if [ "${START}" -ge "${NLINES}" ]; then
  echo ">>> start ${START} >= ${NLINES}; nothing to do."
  exit 0
fi

module load apptainer 2>/dev/null || true

apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 ${RECODIR}/eval/eval_reco_performance.py \
    --keypoint2-list ${KEYPOINT2_LIST} \
    --merged-sp-list ${MERGED_SP_LIST} \
    --nu-reco-dir ${NU_RECO_DIR} \
    --out ${OUT} \
    --start ${START} \
    --n ${N} \
    --stream ${STREAM} \
    --gamma-min-evis ${GAMMA_MIN_EVIS} \
    ${PRIMARIES_ONLY}
"

echo "DONE shard ${SLURM_ARRAY_TASK_ID} -> ${OUT}"
