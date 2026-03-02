#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

# Training start
python3 tools/train.py --config configs/lartpc/shower-origin-sonata-v1m1-v3.py --num-gpus 6
