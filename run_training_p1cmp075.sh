#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

#python3 tools/train.py --config configs/lartpc/pretrain-sonata-v1m1-lartpc-v2.py --num-gpus 4
python3 tools/train.py --config configs/lartpc/pretrain-sonata-v1m1-lartpc-v2.py --num-gpus 4 --options weight=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/epoch_90_v2_p100_noghosts.pth resume=True

##python3 tools/train.py --config configs/lartpc/pretrain-sonata-v1m1-lartpc-restart-p100.py --options weight=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/epoch_87.pth epoch=200 eval_epoch=200

