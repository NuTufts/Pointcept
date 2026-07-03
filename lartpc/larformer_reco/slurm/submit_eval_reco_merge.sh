#!/bin/bash
#
# Merge step for the sharded RECO PERFORMANCE EVAL. Concatenates the per-shard
# record tables (eval_shard*.npz) written by submit_eval_reco_shard.sh into one
# table (--out), prints the per-species efficiency summary, and writes the plots.
# Pure h5py+numpy+matplotlib, single CPU, fast.
#
# Run after the shard array finishes (chain with a SLURM dependency):
#   JID=$(NSHARDS=10 sbatch --parsable --array=0-9 submit_eval_reco_shard.sh)
#   sbatch --dependency=afterok:${JID} submit_eval_reco_merge.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=evalmerge
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --partition=batch
#SBATCH --output=logs/eval_reco/eval_reco_merge.%j.%N.log
#SBATCH --error=logs/eval_reco/eval_reco_merge.%j.%N.err

set -eu

# ---- paths ------------------------------------------------------------------
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
RECODIR=${WORKDIR}/lartpc/larformer_reco

OUTPUT_DIR=${OUTPUT_DIR:-${RECODIR}/output/eval_reco_valdata_all/}
OUT=${OUT:-${RECODIR}/results_eval_reco.npz}
PLOTS=${PLOTS:-${RECODIR}/plots/eval_reco/}

mkdir -p "${WORKDIR}/logs/eval_reco" "${PLOTS}"

module load apptainer 2>/dev/null || true

apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 ${RECODIR}/eval/eval_reco_performance.py \
    --merge '${OUTPUT_DIR}/eval_shard*.npz' \
    --out ${OUT} \
    --plots ${PLOTS}
"

echo "DONE merge -> ${OUT} ; plots -> ${PLOTS}"
