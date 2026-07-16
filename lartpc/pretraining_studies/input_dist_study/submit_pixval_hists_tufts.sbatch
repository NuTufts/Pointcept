#!/bin/bash
# Stage-1 array job for the input-distribution study (P05F) on the Tufts
# CPU farm. Adjust partition/account and the three paths below, then:
#
#   sbatch --array=0-199 submit_pixval_hists_tufts.sbatch
#
# Each task processes every 200th file of FILELIST (~2,100 files/task for
# the full 415k MC train list; expect ~5-15 min/task depending on FS).
# When all tasks finish, run stage 2 on a login/single node:
#
#   python3 merge_and_analyze_pixval_hists.py --indir $OUTDIR --outdir $OUTDIR/analysis
#
#SBATCH --job-name=p05f_pixval
#SBATCH --partition=batch          # ADJUST to your Tufts CPU partition
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/p05f_pixval.%A_%a.log

set -u

REPO=/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept   # ADJUST
FILELIST=${REPO}/lartpc/filelists/h5list_v3_mc_only_train_tufts.txt            # ADJUST (Tufts-remapped list)
OUTDIR=${REPO}/exp/p05f_input_dist                                             # ADJUST

NUM_SHARDS=${NUM_SHARDS:-200}

mkdir -p "$OUTDIR" logs
python3 ${REPO}/lartpc/pretraining_studies/input_dist_study/accumulate_pixval_hists.py \
    --filelist "$FILELIST" \
    --num-shards "$NUM_SHARDS" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --outdir "$OUTDIR"
