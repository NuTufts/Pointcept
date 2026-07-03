#!/bin/bash
#
# Sharded, CPU-array SLURM submission of the SHOWER CALO CALIBRATION collection.
# Each array task sums de-double-counted pixel charge per predicted shower cluster
# vs true KE over its chunk of the keypoint2 list and writes the (Q,KE) pairs to
# OUTPUT_DIR/calo_data_shard{START:07d}.npz (collect mode -- no fit).
#
# After ALL shards finish, MERGE + FIT into trajfit/data/calo_calib.npz:
#   apptainer exec --bind /cluster:/cluster <container> bash -c "
#     cd <WORKDIR> && source setenv_pointcept_only.sh && \
#     python3 -m lartpc.larformer_reco.trajfit.particle_momentum (PYTHONPATH=repo root) \
#       --merge-data '<OUTPUT_DIR>/calo_data_shard*.npz' "
# (the merge is one fast process; run it interactively or as a dependent job.)
#
# No interaction reco is run -- this only reads the predicted shower clusters +
# their merged_sp image charge. CPU-only.
#
#   NSHARDS=10 sbatch --array=0-9 submit_calo_calib_shard.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=calocalib
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/calo_calib/calo_calib_shard.%A_%a.%N.log
#SBATCH --error=logs/calo_calib/calo_calib_shard.%A_%a.%N.err
#SBATCH --array=0-9

set -eu

NSHARDS=${NSHARDS:-10}

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco

KEYPOINT2_LIST=${KEYPOINT2_LIST:-${RECODIR}/outputlists/keypoint2_out_valdata_all.txt}
MERGED_SP_LIST=${MERGED_SP_LIST:-${RECODIR}/inputlists/merged_sp_valdata_all.txt}
OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/calo_calib_data/}

NLINES=$(grep -c . "${KEYPOINT2_LIST}")
PER_SHARD=$(( (NLINES + NSHARDS - 1) / NSHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
N=${PER_SHARD}

echo ">>> calib shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}: ${NLINES} files; "
echo "    start=${START} n=${N} -> ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}" "${WORKDIR}/logs/calo_calib"
if [ "${START}" -ge "${NLINES}" ]; then
  echo ">>> start ${START} >= ${NLINES}; nothing to do."; exit 0
fi

module load apptainer 2>/dev/null || true

apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  PYTHONPATH=${WORKDIR} python3 -m lartpc.larformer_reco.trajfit.particle_momentum \
    --keypoint2-list ${KEYPOINT2_LIST} \
    --merged-sp-list ${MERGED_SP_LIST} \
    --start ${START} --n ${N} \
    --out-data ${OUTPUT_DIR}/calo_data_shard$(printf '%07d' ${START}).npz
"

echo "DONE calib shard ${SLURM_ARRAY_TASK_ID} -> ${OUTPUT_DIR}"
