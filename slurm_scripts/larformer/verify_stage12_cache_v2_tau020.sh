#!/bin/bash
# Full verification of the v2 tau=0.20 stage-1+2 cache (runs after the
# build arrays finish). Phases: coverage, FULL integrity scan (~420k h5
# opens, 8 workers), content sample, dataset read-back, augment check.
#   sbatch [--dependency=afterany:<build jobs>] verify_stage12_cache_v2_tau020.sh
#SBATCH --job-name=lf-cache-verify
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/lf-cache-verify.%j.log
#SBATCH --error=logs/lf-cache-verify.%j.err
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CACHE=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__m2fv2ep4_ftdeghost_tau020
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
module load apptainer
apptainer exec --bind /cluster:/cluster $container bash -c "\
  cd $K && source setenv_pointcept_only.sh && \
  python3 tools/larformer/verify_stage12_cache.py \
    --cache-root $CACHE --splits train,val \
    --expect-train 410000 --expect-val 10000 \
    --sample 1000 --dataset-sample 300 --workers 8"
RC=$?
echo "verification exit code: $RC"
echo "cache size: $(du -sh --apparent-size $CACHE 2>/dev/null | cut -f1)"
df -h /cluster/tufts/wongjiradlab | tail -1
exit $RC
