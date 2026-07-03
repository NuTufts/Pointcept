#!/bin/bash

#SBATCH --job-name=kpqlarformer
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --signal=USR1@600
#SBATCH --output=logs/trainlogs_keypoint2/larformer_keypoint2_particle_cachedpredmask_v1.%j.%N.log
#SBATCH --error=logs/trainlogs_keypoint2/larformer_keypoint2_particle_cachedpredmask_v1.%j.%N.err

set -u

# set the location of your copy of the repo here
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept/
POINTCEPT_WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/
CONFIG=${WORKDIR}/configs/lartpc/larformer/stage4_keypoint/larformer-keypoint2-particle-predmask-cached-v1.py
SAVE_PATH_DIR=larformer_keypoint2_particle_cachedpredmask_v1

# location of the sl7 container here
#container=/projects/u6jo/containers/pointcept-sandbox/
container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif

# setup singularity on the node
module load apptainer

# command used during laptop training tests
# python tools/train.py --config-file configs/lartpc/larformer-keypoint2-partic
# le-predmask-cached-v1.py --num-gpus 1 --options save_path=/mnt/ddrive/pointcept_larformer_kp/larformer_keypoint2_predmask_680sample epoch=50 e
# val_epoch=50 schedule.warmup_iters=300 resume=True weight=/mnt/ddrive/pointcept_larformer_kp/larformer_keypoint2_predmask_680sample/model_epoc
# h_10.pth

# to start job inside container
# apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && \
#   source setenv_pointcept_only.sh && \
#   python3 tools/train.py --config ${CONFIG} --num-gpus 2 \
#   --options save_path=${WORKDIR}/exp/${SAVE_PATH_DIR} epoch=10 eval_epoch=10"
  
CHECKPOINT="${WORKDIR}/exp/larformer_keypoint2_particle_cachedpredmask_v1/model/epoch_20.pth"
# to resume job script inside container
apptainer exec --nv --bind /cluster:/cluster $container bash -c "cd ${WORKDIR} && \
  source setenv_pointcept_only.sh && \
  python3 tools/train.py --config ${CONFIG} --num-gpus 2 \
  --options save_path=${WORKDIR}/exp/${SAVE_PATH_DIR} epoch=30 eval_epoch=30 \
  resume=True weight=${CHECKPOINT}"
