#!/bin/bash

#SBATCH --job-name=larformer_build_cache
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint="a100|l40|l40s|h100|rtx_a5000|rtx_6000|rtx_6000_ada|rtx_a6000"
#SBATCH --constraint="a100"
#SBATCH --output=logs/stage12_cache/larformer_build_cache_train.%j.%N.log
#SBATCH --error=logs/stage12_cache/larformer_build_cache_train.%j.%N.err
#SBATCH --array=0-9

set -u

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/
#CONFIG=${WORKDIR}/configs/lartpc/larformer-particle-v1-cached-ptv3crosslevel.py
CONFIG=${WORKDIR}/configs/lartpc/larformer-particle-v1.py

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer
#module load apptainer/1.2.4-suid

NSHARDS=10
OUTPUT_ROOT="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12_highermaxsp_ptv3crosslevelslicer_iter_75750/"

# Training set: 410000 events
TRAIN_FILE_LIST="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_train.txt"
INPUTLIST=${TRAIN_FILE_LIST}
SPLIT="train"

# Validation set: 10000 events
# VAL_FILE_LIST="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists/h5list_mcall_lantern_val.txt"
# INPUTLIST=${VAL_FILE_LIST}
# SPLIT="val"

# start job inside container
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && \
  cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/ && \
  source setenv_pointcept_only.sh && \
  python3 tools/build_stage12_cache_shard.py --config ${CONFIG} \
  --n-shards ${NSHARDS} --shard-id $SLURM_ARRAY_TASK_ID \
  --inputlist ${INPUTLIST} \
  --cache-root ${OUTPUT_ROOT} \
  --split ${SPLIT} \
  --max-spacepoints 750000
  --tau-loose-floor 0.2 --tau-loose-nominal 0.5 --tau-loose-delta 0.2"
