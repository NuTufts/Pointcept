#!/bin/bash
#
# Sharded, CPU-array SLURM submission of the NU-INTERACTION RECO (vertices +
# tracks + showers + 4-momenta) over the keypoint2-cascade output list. The
# KEYPOINT2 list is split into NSHARDS contiguous chunks; each array task runs
# run_nu_reco.py over its chunk via --start/--n and writes one H5 per shard
# (nu_reco_shard{START:07d}.h5) into OUTPUT_DIR.
#
# The reco is CPU-only (no GPU) -> use a CPU partition and as many shards as the
# queue allows. Each keypoint2 file is linked to its parent merged_sp (for
# charge/truth) by the `src_file` attr, resolved by basename against MERGED_SP_LIST.
#
# Prereqs (generated files, must exist under trajfit/data/):
#   range2ke_lar.npz  -> `python trajfit/make_range2ke_npz.py`
#   calo_calib.npz    -> `python trajfit/particle_momentum.py`  (refit on the
#                        full valdata sample for production; the shipped one is
#                        from the small pi0 dev set).
#
# Submit (override NSHARDS/array without editing):
#   NSHARDS=20 sbatch --array=0-19 submit_nu_reco_shard.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=nureco
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/nu_reco/nu_reco_shard.%A_%a.%N.log
#SBATCH --error=logs/nu_reco/nu_reco_shard.%A_%a.%N.err
#SBATCH --array=0-19

set -eu

# ---- knobs ------------------------------------------------------------------
NSHARDS=${NSHARDS:-20}
# extra run_nu_reco.py flags, e.g. EXTRA_ARGS="--shower-mode llr"
EXTRA_ARGS=${EXTRA_ARGS:-}

# ---- paths ------------------------------------------------------------------
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco

KEYPOINT2_LIST=${KEYPOINT2_LIST:-${RECODIR}/outputlists/keypoint2_out_valdata_all.txt}
MERGED_SP_LIST=${MERGED_SP_LIST:-${RECODIR}/inputlists/merged_sp_valdata_all.txt}
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/nu_reco_valdata_all/}

# ---- per-shard range over the KEYPOINT2 list --------------------------------
# ceil(NLINES / NSHARDS) events per shard, contiguous, non-overlapping. The
# driver clamps the last shard to the real list length.
NLINES=$(grep -c . "${KEYPOINT2_LIST}")
PER_SHARD=$(( (NLINES + NSHARDS - 1) / NSHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
N=${PER_SHARD}

echo ">>> shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}: keypoint2 list has ${NLINES} "
echo "    files; start=${START} n=${N} -> ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}" "${WORKDIR}/logs/nu_reco"

# stop cleanly if this shard starts past the end of the list
if [ "${START}" -ge "${NLINES}" ]; then
  echo ">>> start ${START} >= ${NLINES}; nothing to do."
  exit 0
fi

module load apptainer 2>/dev/null || true

apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 ${RECODIR}/scripts/run_nu_reco.py \
    --keypoint2-list ${KEYPOINT2_LIST} \
    --merged-sp-list ${MERGED_SP_LIST} \
    --output-dir ${OUTPUT_DIR} \
    --start ${START} \
    --n ${N} ${EXTRA_ARGS}
"

echo "DONE shard ${SLURM_ARRAY_TASK_ID} -> ${OUTPUT_DIR}"
