#!/bin/bash
# Mini smoke: new-deghoster cascade config, 4 events, 1 GPU.
#SBATCH --job-name=lf-cascade-smoke
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-cascade-smoke.%j.%N.log
#SBATCH --error=logs/lf-cascade-smoke.%j.%N.err
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
OLD=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
SAVE=$K/exp/_smoke_cascade_ptv3deghost
mkdir -p $SAVE
awk -F',' 'NR>1 && NR<=5 {print $5}' $OLD/lartpc_data_prep/larformer_analysis/valtest/manifest/bnb_nu_pi0filter_corsika_valtest.csv > $SAVE/list4.txt
echo "input events:"; cat $SAVE/list4.txt
module load apptainer
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd $K && source setenv_pointcept_only.sh && \
   python3 tools/larformer/run_slicer_inference.py \
     --config $K/configs/lartpc/larformer/stage2_slicer/larformer-slicer-m2frecipe-v2-ptv3deghost.py \
     --weights $K/exp/larformer_slicer_v1_cascaded_ptv3hybrid_crosslevel_cap300k_m2frecipe_v2/model/epoch_4.pth \
     --input-list $SAVE/list4.txt \
     --output-dir $SAVE/inference \
     --split val"
RC=$?
echo "exit code: $RC"; ls -la $SAVE/inference/ 2>/dev/null | head
