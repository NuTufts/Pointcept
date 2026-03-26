#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

config=$WORKDIR/configs/lartpc/shower-origin-sonata-v1m1-v3-p1cmp075.py
checkpoint_file=$WORKDIR/shower_origin/sonata_v1m1_v3_v6backbone_pax_pi0filter/model/model_last.pth
dataset_file=$WORKDIR/lartpc_data_prep/hdflist_showerfragment_bnb_nu_pi0filter_val.txt

python3 tools/run_shower_origin_inference.py -c $config --checkpoint $checkpoint_file --data-list $dataset_file --output shower_origin_results_epoch152.h5
