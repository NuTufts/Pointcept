#!/bin/bash
# Stage-1 array job for the input-distribution study (P05F) on the Tufts
# CPU farm. Adjust partition/account and the three paths below, then:
#
#   mkdir -p logs   # BEFORE sbatch: SLURM opens the --output file before the
#                   # script runs; a missing logs/ kills jobs instantly (exit 53)
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

# The Tufts login/compute nodes have no h5py outside the container.
CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif

NUM_SHARDS=${NUM_SHARDS:-200}

# apptainer is already on PATH on the batch nodes; load it anyway to match
# start_container.sh (modtree/deprecated must come first or the load fails).
module load modtree/deprecated
module load apptainer/1.2.4-suid

mkdir -p "$OUTDIR" logs
apptainer exec --bind /cluster:/cluster "$CONTAINER" \
    python3 ${REPO}/lartpc/pretraining_studies/input_dist_study/accumulate_pixval_hists.py \
        --filelist "$FILELIST" \
        --num-shards "$NUM_SHARDS" \
        --shard "$SLURM_ARRAY_TASK_ID" \
        --outdir "$OUTDIR"
