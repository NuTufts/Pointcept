#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

# ---------------------------------------------------------------------
# TRAIN v5
# Attempt: 390k training set, batchsize 64, larger model
#python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v5.py --num-gpus 2
# 
#CHECKPOINT_FILE=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v5_h200_noghosts/model/epoch_16.pth
#CHECKPOINT_FILE=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v5_h200_noghosts/model/epoch_32.pth
#python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v5.py --num-gpus 2 --options resume=True weight=${CHECKPOINT_FILE}

# ---------------------------------------------------------------------
# TRAIN v6
# Attempt: 390k training set, batchsize 64, larger model
#python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6.py --num-gpus 2
#
#CHECKPOINT_FILE=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v6_h200_noghosts_pretrain/model/epoch_14.pth
#CHECKPOINT_FILE=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v6_h200_noghosts_pretrain/model/epoch_28.pth
# CHECKPOINT_FILE=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v6_h200_noghosts_pretrain/model/epoch_42.pth
# python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v6.py --num-gpus 2 --options resume=True weight=${CHECKPOINT_FILE}

# RE-RUN TRAIN v6 with log-space fix
config=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/configs/lartpc/sonata_pretrain/pretrain-sonata-v1m1-lartpc-v6-logspace-resume.py
#CHECKPOINT_FILE=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_14.pth
CHECKPOINT_FILE=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v6_h200_noghosts_pretrain_logspace_resume/model/epoch_28.pth
#python3 tools/train.py --config ${config} --num-gpus 2 
python3 tools/train.py --config ${config} --num-gpus 2 --options resume=True weight=${CHECKPOINT_FILE}
