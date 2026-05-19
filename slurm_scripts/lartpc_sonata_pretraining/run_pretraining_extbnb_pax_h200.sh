#!/bin/bash

cd /cluster/tufts/wongjiradlab/twongj01/mphys
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept


CONFIG="/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/configs/lartpc/pretrain-sonata-v6-extbnb.py"

# Attempt: 822k training set, batchsize 72, use float32 for CPE, no grad scalar
#python3 tools/train.py --config $CONFIG --num-gpus 2

# resume
CHECKPOINT="/cluster/tufts/wongjiradlab/twongj01/mphys/Pointcept/sonata/lartpc_v6_h200_extbnb_run1/model/epoch_5.pth"
python3 tools/train.py --config ${CONFIG} --num-gpus 2 --options resume=True weight=${CHECKPOINT}

