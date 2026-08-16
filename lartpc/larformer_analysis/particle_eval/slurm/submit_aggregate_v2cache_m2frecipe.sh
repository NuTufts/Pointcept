#!/bin/bash
#
# Aggregate the v2cache_val x particle_m2frecipe_ep8 per-event records into
# the summary JSON + parquet bundle (aggregate_metrics.py). Submit after the
# valtest array finishes:  sbatch submit_aggregate_v2cache_m2frecipe.sh
# ---------------------------------------------------------------------------

#SBATCH --job-name=agg_pev2
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=batch
#SBATCH --output=logs/particle_eval/aggregate_v2cache.%j.log
#SBATCH --error=logs/particle_eval/aggregate_v2cache.%j.err

set -eu

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
ADIR=${WORKDIR}/exp/larformer_particle_v2_cached_ptv3crosslevel_m2frecipe/valtest_ep8_v2cache/analysis/v2cache_val

mkdir -p "${WORKDIR}/logs/particle_eval"
module load apptainer 2>/dev/null || true

apptainer exec --bind /cluster:/cluster "${container}" bash -c "
  cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 lartpc/larformer_analysis/particle_eval/aggregate_metrics.py \
    --analysis-dir ${ADIR} \
    --tag v2cache_val --model-tag particle_m2frecipe_ep8
"
echo "DONE aggregate"
