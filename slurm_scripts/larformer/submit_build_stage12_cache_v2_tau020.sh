#!/bin/bash
# Stage-1+2 cache build, PRODUCTION v2 chain @ tau=0.20.
#   deghoster: PTv3-decoder full-event-ft (xformers)   slicer: m2frecipe-v2 ep4
#   config: larformer-fullcascade-production-v2-tau020.py
# Cache semantics: inclusion = deghost_keep(tau=0.2, eval mode) AND
# (slicer nu-mask prob > tau_loose_floor OR GT-nu). tau-loose knobs are the
# SLICER-mask thresholds (floor 0.2 / nominal 0.5 / delta 0.2 = same as the
# old caches, so LArFormerStage12CacheDataset defaults keep working) — the
# DEGHOST tau lives in the cascade config, not here.
#
# TRAIN (default): 410k events, 20 shards x ~10h.
#   sbatch submit_build_stage12_cache_v2_tau020.sh
# VAL: 10k events, 2 shards.
#   sbatch --array=0-1 --export=ALL,SPLIT=val,NSHARDS=2 submit_build_stage12_cache_v2_tau020.sh
# Idempotent: re-submitting skips events that already have a cache file or
# .skipped marker.

#SBATCH --job-name=lf-cache-v2-tau020
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/stage12_cache_v2/build.%A_%a.%N.log
#SBATCH --error=logs/stage12_cache_v2/build.%A_%a.%N.err
#SBATCH --array=0-19

set -u

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage3_particle/larformer-fullcascade-production-v2-tau020.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif

SPLIT=${SPLIT:-train}
NSHARDS=${NSHARDS:-20}
OUTPUT_ROOT="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__m2fv2ep4_ftdeghost_tau020/"

LISTDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists
if [ "${SPLIT}" = "train" ]; then
  INPUTLIST=${LISTDIR}/h5list_mcall_lantern_train.txt
else
  INPUTLIST=${LISTDIR}/h5list_mcall_lantern_val.txt
fi

module load apptainer
mkdir -p "${OUTPUT_ROOT}"

apptainer exec --nv --bind /cluster:/cluster $container bash -c "\
  cd ${WORKDIR} && source setenv_pointcept_only.sh && \
  python3 tools/larformer/build_stage12_cache_shard.py \
    --config ${CONFIG} \
    --inputlist ${INPUTLIST} \
    --cache-root ${OUTPUT_ROOT} \
    --split ${SPLIT} \
    --shard-id ${SLURM_ARRAY_TASK_ID} --n-shards ${NSHARDS} \
    --max-spacepoints 300000 \
    --tau-loose-floor 0.2 --tau-loose-nominal 0.5 --tau-loose-delta 0.2"
