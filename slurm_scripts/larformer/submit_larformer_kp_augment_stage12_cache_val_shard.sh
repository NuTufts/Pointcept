#!/bin/bash

#SBATCH --job-name=larformer_augment_cache
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint="a100|l40|l40s|h100|rtx_a5000|rtx_6000|rtx_6000_ada|rtx_a6000"
#SBATCH --constraint="a100"
#SBATCH --output=logs/stage3_augment_cache_val/larformer_augment_cache.%j.%N.log
#SBATCH --error=logs/stage3_augment_cache_val/larformer_augment_cache.%j.%N.err
#SBATCH --array=0-9

set -u

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/
POINTCEPT_WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/
KP_CONFIG=${WORKDIR}/configs/lartpc/larformer-keypoint2-particle-v1.py
PARTICLE_CONFIG=${WORKDIR}/configs/lartpc/larformer-particle-fullcascade-ptv3crosslevel.py
PARTICLE_WEIGHTS=${POINTCEPT_WORKDIR}/exp/larformer_particle_v1_cached_ptv3crosslevel_smallbatch_resume3_cosinedecay/model/epoch_12.pth

# location of the sl7 container here
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer
#module load apptainer/1.2.4-suid

NSHARDS=10
#CACHE_ROOT="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12_highermaxsp_predmask_ptv3crosslevelslicer_iter_75750_test/"
CACHE_ROOT="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12_highermaxsp_ptv3crosslevelslicer_iter_75750/"
OUTPUT_ROOT="/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/hdf5/larformer_cache_stage12_highermaxsp_predmask_ptv3crosslevelslicer_iter_75750/"

# start job inside container
#SBATCH --array=0-63
# python tools/augment_stage12_cache_pred_masks_shard.py \
#   --keypoint-config configs/lartpc/larformer-keypoint2-particle-v1.py \
#   --cache-root /path/to/stage123_cache/train \
#   --particle-config configs/lartpc/larformer-particle-fullcascade-ptv3crosslevel.py \
#   --particle-weights /path/to/model_iter_98652.pth \
#   --shard-id $SLURM_ARRAY_TASK_ID --n-shards 64

apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 tools/augment_stage12_cache_pred_masks_shard.py \
  --keypoint-config ${KP_CONFIG} \
  --cache-root ${CACHE_ROOT} \
  --particle-config ${PARTICLE_CONFIG} \
  --particle-weights ${PARTICLE_WEIGHTS} \
  --n-shards ${NSHARDS} --shard-id $SLURM_ARRAY_TASK_ID
  --output-dir ${OUTPUT_ROOT}"
