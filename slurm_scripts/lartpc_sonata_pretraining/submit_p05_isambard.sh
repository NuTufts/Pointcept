#!/bin/bash
# Parameterized SLURM batch script for Phase 0.5 runs on Isambard (WP5/WP8).
#
# Do not sbatch directly — use launch_p05_run.sh, which sets the job name,
# log paths, and registry entry:
#     ./launch_p05_run.sh configs/lartpc/p05/<config>.py
#
# Machinery is inherited from submit_pretraining_isambard_extbnbmc_ghop.sh:
# SIGUSR1-triggered mid-epoch checkpointing (SignalCheckpointHook) with
# resubmit chaining, capped at MAX_RESUBMITS.

#SBATCH --job-name=p05
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --signal=USR1@600
#SBATCH --output=/projects/u6jo/work/pointcept/exp/logs/p05.%x.%j.%N.log
#SBATCH --error=/projects/u6jo/work/pointcept/exp/logs/p05.%x.%j.%N.err

set -u

CONFIG=${1:?usage: sbatch submit_p05_isambard.sh <config.py>}
MAX_RESUBMITS=${2:-6}

WORKDIR=/projects/u6jo/work/pointcept
SQASHFILE=/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch.sqsh
container=/projects/u6jo/containers/pointcept-sandbox/

# save_path is parsed from the (generated) config; every P05 config defines it
# as a single top-level line: save_path = "sonata/p05/<RUN_ID>"
SAVE_REL=$(grep -m1 '^save_path' "$CONFIG" | sed 's/.*=\s*"\(.*\)".*/\1/')
if [ -z "$SAVE_REL" ]; then
    echo "ERROR: could not parse save_path from $CONFIG"; exit 1
fi
SAVE_DIR=${WORKDIR}/${SAVE_REL}
CKPT=${SAVE_DIR}/model/model_last.pth
RESUBMIT_MARKER=${SAVE_DIR}/RESUBMIT
COUNTER_FILE=${SAVE_DIR}/.resubmit_count

RESUME_OPTS=""
if [ -f "$CKPT" ]; then
    echo "[$(date)] Found existing checkpoint at $CKPT; resuming"
    RESUME_OPTS="--options resume=True weight=${CKPT}"
else
    echo "[$(date)] No existing checkpoint; starting fresh run"
    mkdir -p "${SAVE_DIR}/model"
fi

rm -f "$RESUBMIT_MARKER"

# srun + exec: see submit_pretraining_isambard_extbnbmc_ghop.sh for why this
# shape is required for SIGUSR1 to reach python.
srun apptainer exec --nv --bind $SQASHFILE:/data:image-src=,ro,/projects/u6jo:/projects/u6jo $container \
    bash -c "cd ${WORKDIR} && \
             source setenv_isambard_project_repo.sh && \
             exec python3 tools/train.py --config ${CONFIG} --num-gpus 4 ${RESUME_OPTS}"
TRAIN_RC=$?
echo "[$(date)] training exited with code $TRAIN_RC"

# Resubmit on the RESUBMIT marker (graceful signal-triggered save) OR on any
# nonzero exit (wall-clock kill, crash). The marker path alone proved
# insufficient: with --num-gpus 4 the trainer ranks are mp.spawn children of
# a launcher parent that had no USR1 handler, so SLURM's pre-timeout signal
# killed the parent (exit 138) before any rank could save+mark (observed on
# P05B.1, job 5663946). IterCheckpointSaver keeps model_last.pth at most
# save_iter_freq iters stale, so resuming after a hard kill is safe; the
# counter caps runaway crash loops.
if [ -f "$RESUBMIT_MARKER" ] || [ "$TRAIN_RC" -ne 0 ]; then
    COUNT=0
    [ -f "$COUNTER_FILE" ] && COUNT=$(cat "$COUNTER_FILE")
    if [ "$COUNT" -ge "$MAX_RESUBMITS" ]; then
        echo "[$(date)] Reached MAX_RESUBMITS=$MAX_RESUBMITS; not resubmitting"
    else
        echo $((COUNT + 1)) > "$COUNTER_FILE"
        rm -f "$RESUBMIT_MARKER"
        echo "[$(date)] Resubmitting (rc=$TRAIN_RC, count=$((COUNT + 1))/$MAX_RESUBMITS)"
        sbatch --job-name="$SLURM_JOB_NAME" --dependency=afterany:$SLURM_JOB_ID "$0" "$CONFIG" "$MAX_RESUBMITS"
    fi
else
    echo "[$(date)] Clean exit (rc=0) with no RESUBMIT marker: training finished."
    echo "[$(date)] Not resubmitting."
fi
