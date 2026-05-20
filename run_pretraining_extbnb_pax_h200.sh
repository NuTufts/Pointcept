#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept


CONFIG="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/configs/lartpc/pretrain-sonata-v7-extbnb-larmatch.py"

#python3 tools/train.py --config $CONFIG --num-gpus 2

# resume
#CHECKPOINT="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_lowerlr/model/epoch_6.pth"
CHECKPOINT="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v7_h200_extbnb_larmatch_lowerlr/model/epoch_12.pth"
python3 tools/train.py --config ${CONFIG} --num-gpus 2 --options resume=True weight=${CHECKPOINT}

