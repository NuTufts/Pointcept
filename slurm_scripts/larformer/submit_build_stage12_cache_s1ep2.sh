#!/bin/bash
# Stage-1+2 cache build, S1 chain (pilot): v6-lantern deghoster ep25 +
# S1 mix-enriched slicer EPOCH 2, over the MIX v1 corpus (186.5k overlay
# + 219.8k LANTERN enriched, label-completed). SLICER_RETRAIN_PLAN
# cache-rebuild plan, 2026-08-21. Old v2 cache is KEPT (user decision).
#
# TRAIN (default): 406,320 events, 26 shards.
#   sbatch submit_build_stage12_cache_s1ep2.sh
# VAL (completed-copy 1500):
#   sbatch --array=0 --export=ALL,SPLIT=val,NSHARDS=1 submit_build_stage12_cache_s1ep2.sh
# Idempotent: re-submitting skips events with a cache file or .skipped marker.
#SBATCH --job-name=lf-cache-s1ep2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/stage12_cache_s1/build.%A_%a.%N.log
#SBATCH --error=logs/stage12_cache_s1/build.%A_%a.%N.err
#SBATCH --array=0-25
set -u
W=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=${W}/configs/lartpc/larformer/stage3_particle/larformer-fullcascade-s1ep2-tau020.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SPLIT=${SPLIT:-train}
NSHARDS=${NSHARDS:-26}
OUTPUT_ROOT="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12__s1ep2_v6lantern_tau020/"
LED=${W}/lartpc/data_prep/uboone_official/training_data_ledger
if [ "${SPLIT}" = "train" ]; then
  INPUTLIST=${LED}/h5list_mix_enriched_train_v1.txt
else
  INPUTLIST=${LED}/h5list_lantern_val_completed_1500.txt
fi
module load apptainer
mkdir -p "${OUTPUT_ROOT}"
apptainer exec --nv --bind /cluster:/cluster $container bash -c "\
  cd ${W} && source setenv_pointcept_only.sh && \
  python3 tools/larformer/build_stage12_cache_shard.py \
    --config ${CONFIG} \
    --inputlist ${INPUTLIST} \
    --cache-root ${OUTPUT_ROOT} \
    --split ${SPLIT} \
    --shard-id ${SLURM_ARRAY_TASK_ID} --n-shards ${NSHARDS} \
    --max-spacepoints 300000 \
    --tau-loose-floor 0.2 --tau-loose-nominal 0.5 --tau-loose-delta 0.2"
