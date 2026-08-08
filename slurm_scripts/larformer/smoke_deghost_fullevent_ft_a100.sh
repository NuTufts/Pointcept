#!/bin/bash
#SBATCH --job-name=lf-dgft-smoke
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=pax141,pax007
#SBATCH --output=logs/lf-dgft-smoke.%j.%N.log
#SBATCH --error=logs/lf-dgft-smoke.%j.%N.err
set -u
K=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/kpv2_pointcept
LANTERN=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/lartpc_data_prep/lantern_scripts/h5lists
SAVE=$K/exp/_smoke_deghost_ftfull
mkdir -p $SAVE
head -40 $LANTERN/h5list_mcall_lantern_train.txt > $SAVE/smoke_train.txt
head -8  $LANTERN/h5list_mcall_lantern_val.txt   > $SAVE/smoke_val.txt
container=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif
module load apptainer
MEMLOG=$SAVE/gpumem.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 3 > $MEMLOG 2>&1 &
SMPID=$!
apptainer exec --nv --bind /cluster:/cluster $container bash -c \
  "cd $K && source setenv_pointcept_only.sh && \
   python3 tools/train.py --config $K/configs/lartpc/larformer/stage1_deghost/deghost-ptv3decoder-v2-fullevent-ft.py --num-gpus 1 --options \
     save_path=$SAVE \
     data.train.data_list_file=$SAVE/smoke_train.txt \
     data.val.data_list_file=$SAVE/smoke_val.txt \
     epoch=1 eval_epoch=1 evaluate=True enable_wandb=False \
     batch_size=2 batch_size_val=2 num_worker=4 num_worker_val=2"
RC=$?
kill $SMPID 2>/dev/null
echo "exit: $RC"
awk -F',' 'BEGIN{m=0}{gsub(/ /,"",$2); if($2+0>m)m=$2+0} END{print "PEAK GPU MB: "m" (batch 2/GPU; production same)"}' $MEMLOG
