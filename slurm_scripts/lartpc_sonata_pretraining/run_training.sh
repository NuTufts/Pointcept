#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
python3 tools/train.py --config configs/lartpc/sonata_pretrain/archive/pretrain-sonata-v1m1-lartpc.py
