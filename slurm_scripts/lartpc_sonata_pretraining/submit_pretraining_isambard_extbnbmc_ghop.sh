#!/bin/bash

#SBATCH --job-name=larsonata
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --signal=USR1@600
#SBATCH --output=logs/larsonata.v8.isambard_ghopper_extbnb_mc_combined_larmatch.%j.%N.log
#SBATCH --error=logs/larsonata.v8.isambard_ghopper_extbnb_mc_combined_larmatch.%j.%N.err

set -u

# set the location of your copy of the repo here
WORKDIR=/home/u6jo/twongj01.u6jo/ubpointcept/pointcept
CONFIG=${WORKDIR}/configs/lartpc/pretrain-sonata-v8-extbnb-mc-combined-larmatch.py
SQASHFILE=/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch.sqsh

# location of the sl7 container here
container=/projects/u6jo/containers/pointcept-sandbox/

# Must match the `save_path` value inside the config file. Relative to WORKDIR
# because that's where `python tools/train.py` is launched from below.
SAVE_REL=sonata/lartpc_v8_isambard_ghopper_extbnb_mc_combined_larmatch
SAVE_DIR=${WORKDIR}/${SAVE_REL}
CKPT=${SAVE_DIR}/model/model_last.pth
RESUBMIT_MARKER=${SAVE_DIR}/RESUBMIT
COUNTER_FILE=${SAVE_DIR}/.resubmit_count
MAX_RESUBMITS=5

# Resume detection: if a checkpoint already exists, override cfg.resume and
# cfg.weight via --options. On the first launch the file doesn't exist, so the
# config's defaults (resume=False, weight=None) take effect and training
# starts from scratch.
RESUME_OPTS=""
if [ -f "$CKPT" ]; then
    echo "[$(date)] Found existing checkpoint at $CKPT; resuming"
    RESUME_OPTS="--options resume=True weight=${CKPT}"
else
    echo "[$(date)] No existing checkpoint; starting fresh run"
    mkdir -p "${SAVE_DIR}/model"
fi

# Clean any stale marker from a previous slot before launching this slot.
rm -f "$RESUBMIT_MARKER"

# Why this invocation shape:
#   - `srun` makes python the SLURM "step" so --signal=USR1 is delivered to it
#     (without srun, SLURM only signals srun-launched steps, and the batch
#     script gets nothing unless `B:` is set).
#   - `exec python3 ...` inside `bash -c` replaces bash with python via
#     execve(2). Same PID, no fork: apptainer's child becomes python directly,
#     so the signal reaches python instead of being queued by bash.
#   - `source setenv.sh` runs *before* exec so env vars (PYTHONPATH, ROOT
#     bindings, etc.) are exported into the process that exec replaces.
srun apptainer exec --nv --bind $SQASHFILE:/data:image-src=,ro $container \
    bash -c "cd ${WORKDIR} && \
             source setenv.sh && \
             exec python3 tools/train.py --config ${CONFIG} --num-gpus 4 ${RESUME_OPTS}"
TRAIN_RC=$?
echo "[$(date)] training exited with code $TRAIN_RC"

# Resubmit chaining: only if SignalCheckpointHook wrote the RESUBMIT marker
# (i.e. we exited because of the wall-clock signal, not natural completion or
# a crash). Cap chain length to MAX_RESUBMITS as a safety net.
if [ -f "$RESUBMIT_MARKER" ]; then
    COUNT=0
    [ -f "$COUNTER_FILE" ] && COUNT=$(cat "$COUNTER_FILE")
    if [ "$COUNT" -ge "$MAX_RESUBMITS" ]; then
        echo "[$(date)] Reached MAX_RESUBMITS=$MAX_RESUBMITS; not resubmitting"
    else
        echo $((COUNT + 1)) > "$COUNTER_FILE"
        rm -f "$RESUBMIT_MARKER"
        echo "[$(date)] Resubmitting (count=$((COUNT + 1))/$MAX_RESUBMITS)"
        sbatch --dependency=afterany:$SLURM_JOB_ID "$0"
    fi
else
    echo "[$(date)] No RESUBMIT marker; training either finished or crashed."
    echo "[$(date)] Not resubmitting."
fi
