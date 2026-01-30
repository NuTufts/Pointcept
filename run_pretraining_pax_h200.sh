#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

# Attempt: 390k training set, batchsize 72
python3 tools/train.py --config configs/lartpc/pretrain-sonata-v1m1-lartpc-v4.py --num-gpus 1