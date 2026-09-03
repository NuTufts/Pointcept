#!/bin/bash
# particle_class_id augmentation for the v2 tau=0.20 cache (train + val),
# followed inline by a quick verification (augment phase must flip to
# present). CPU-only, in-place, idempotent.
#SBATCH --job-name=lf-cache-s1-augment
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=batch,preempt
#SBATCH --output=logs/lf-cache-s1-augment.%j.log
#SBATCH --error=logs/lf-cache-s1-augment.%j.err
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CACHE=/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__s1ep2_v6lantern_tau020
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
module load apptainer
for SPLIT in train val; do
  echo "=== augmenting ${SPLIT} ==="
  apptainer exec --bind /cluster:/cluster $container bash -c "\
    cd $K && source setenv_pointcept_only.sh && \
    python3 tools/larformer/augment_stage12_cache_particle_class_id.py \
      --cache-root $CACHE/$SPLIT --workers 8 --progress-every 20000"
  RC=$?
  echo "=== ${SPLIT} augment exit: $RC ==="
  [ $RC -ne 0 ] && exit $RC
done
echo "=== post-augment quick verification ==="
apptainer exec --bind /cluster:/cluster $container bash -c "\
  cd $K && source setenv_pointcept_only.sh && \
  python3 tools/larformer/verify_stage12_cache.py \
    --cache-root $CACHE --splits train,val \
    --expect-train 406320 --expect-val 1500 \
    --quick --sample 300 --dataset-sample 50"
RC=$?
echo "post-augment verification exit: $RC"
exit $RC
