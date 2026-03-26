#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py

# Training start
python3 tools/train.py --config ${CONFIG} --num-gpus 4
