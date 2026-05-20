#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py
RESUME_CONFIG=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075-resume.py

# Training start
#python3 tools/train.py --config ${CONFIG} --num-gpus 4

# resume
weights=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/shower_origin/sonata_v1m1_v3_v6logfixbackbone_newtargets/model/model_epoch74.pth
python3 tools/train.py --config ${RESUME_CONFIG} --num-gpus 4 --options resume=True weight=${weights}
