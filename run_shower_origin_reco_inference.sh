#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

config=$WORKDIR/configs/lartpc/shower-origin-sonata-v1m1-v3-reco-fragments-p1cmp075.py
checkpoint_file=$WORKDIR/shower_origin/sonata_v1m1_v3_v6backbone_pax_pi0filter_recofragments/model/model_best.pth
dataset_file=$WORKDIR/lartpc_data_prep/shower_origin_reco_scripts/h5list_showeroriginreco_ncpi0filter_validated_val.txt

python3 tools/run_shower_origin_inference.py \
    -c $config \
    --checkpoint $checkpoint_file \
    --data-list $dataset_file \
    --output shower_origin_reco_results.h5
