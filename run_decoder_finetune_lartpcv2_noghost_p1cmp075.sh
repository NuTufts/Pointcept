#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_v2_p100_noghosts_extended_epoch_200.pth

python3 tools/train.py --config configs/lartpc/semseg-sonata-v1m1-lartpc-v2-finetune.py --num-gpus 4 --options weight=$WEIGHTS
