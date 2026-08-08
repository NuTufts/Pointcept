#!/bin/bash
# Full-event fine-tune of the PTv3-decoder deghoster (v2) -- 2x A100-80GB.
# Creates the 80k/2k subset lists, then trains 2 epochs (~13-22h) warm-started
# from the crop-trained v1 model_best. Auto-resume on resubmit.
#   sbatch submit_deghost_fullevent_ft_a100.sh
#SBATCH --job-name=lf-deghost-ftfull
#SBATCH --mem=160G
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-deghost-ftfull.%j.%N.log
#SBATCH --error=logs/lf-deghost-ftfull.%j.%N.err
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
CONFIG=$K/configs/lartpc/larformer/stage1_deghost/deghost-ptv3decoder-v2-fullevent-ft.py
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
LANTERN=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists
SAVE=$K/exp/deghost_ptv3decoder_v2_fullevent_ft
CHECKPOINT=$SAVE/model/model_last.pth
mkdir -p $SAVE
[ -f $SAVE/ft_train_80k.txt ] || head -80000 $LANTERN/h5list_mcall_lantern_train.txt > $SAVE/ft_train_80k.txt
[ -f $SAVE/ft_val_2k.txt ]    || head -2000  $LANTERN/h5list_mcall_lantern_val.txt   > $SAVE/ft_val_2k.txt
module load apptainer
if [ -f "$CHECKPOINT" ]; then
  RESUME_OPTS="--options resume=True weight=$CHECKPOINT"
  echo "[submit] resuming from $CHECKPOINT"
else
  RESUME_OPTS=""
  echo "[submit] fresh fine-tune (warm start from v1 model_best via config weight)"
fi
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd $K && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config $CONFIG --num-gpus 2 $RESUME_OPTS"
