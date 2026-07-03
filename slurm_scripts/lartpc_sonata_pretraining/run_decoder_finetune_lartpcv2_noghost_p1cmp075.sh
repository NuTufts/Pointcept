#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

#WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v3_p100_noghosts/model/epoch_22.pth
#python3 tools/train.py --config configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v3-decoder-finetune.py --num-gpus 4 --options weight=$WEIGHTS

#WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/semseg-decoder-finetune-v3-noghost-p100/model/model_last.pth
#python3 tools/train.py --config configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v3-decoder-finetune.py --num-gpus 4 --options weight=$WEIGHTS

#WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/semseg-decoder-finetune-v3-noghost-p100-resume-dropcosmics/model/epoch_11.pth
#python3 tools/train.py --config configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v3-decoder-finetune.py --num-gpus 4 --options weight=$WEIGHTS resume=True

#WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/semseg-decoder-finetune-v3-noghost-p100-resume-dropcosmics/model/epoch_32.pth
#python3 tools/train.py --config configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v3-decoder-finetune.py --num-gpus 4 --options weight=$WEIGHTS

WEIGHTS=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/sonata/lartpc_v4_p100_noghosts/epoch_13.pth
python3 tools/train.py --config configs/lartpc/semseg/archive/semseg-sonata-v1m1-lartpc-v4-decoder-finetune.py --num-gpus 6 --options weight=$WEIGHTS