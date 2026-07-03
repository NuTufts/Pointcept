#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

# Attempt 1
##python3 tools/train.py --config configs/lartpc/pretrain-sonata-v1m1-lartpc-restart-p100.py --options weight=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/epoch_87.pth epoch=200 eval_epoch=200

# Attempt 2
#python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v2.py --num-gpus 4
#python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v2.py --num-gpus 4 --options weight=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/epoch_90_v2_p100_noghosts.pth resume=True

# Attempt 3: 250k training set, deeper layers, smaller batchsize
python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v3.py --num-gpus 4
# extended training. With larger number of events: 375. Includes more charged pions. Increase rate of nu-only examples.
python3 tools/train.py --num-gpus 4 --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc-v3.py --options resume=True weight=sonata/lartpc_v3_p100_noghosts/epoch_22.pth
