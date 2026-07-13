#!/bin/bash
# P0.7 smoke test for a Phase 0.5 config: short GPU run on a small file list,
# writing under sonata/p05/smoke/ so the real run directory is untouched.
#
#   ./launch_p05_smoke.sh configs/lartpc/p05/supervised-ceiling-p05a1-mc-noghost.py
#   ./launch_p05_smoke.sh configs/lartpc/p05/pretrain-sonata-p05b1-mc-noghost-freerot.py
#
# Uses the MC val list (5,000 files -> ~104 iters/epoch at batch 48) as train
# data and the diag1k list as val, 1 epoch:
#   - supervised configs: exercises mid-epoch SemSegEvaluator (eval_freq=50,
#     Iter-axis logging), CheckpointSaver, M5 hook
#   - SSL configs: exercises the multi-view pipeline, weights-only snapshots
#     (forced at iters 20 and 60 via hooks override), M5 hook
# Afterwards, verify:
#   - supervised: val/mIoU points at iters 50 and 100 in the log / wandb
#   - SSL: sonata/p05/smoke/<RUN_ID>/snapshot/ has snapshot_iter0000020_* and
#     snapshot_iter0000060_*, and one loads at Tufts with the probe config.

set -eu

WORKDIR=/projects/u6jo/work/pointcept
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
FILELISTS=${WORKDIR}/lartpc/filelists
SMOKE_TRAIN=${FILELISTS}/h5list_v3_mc_only_val.txt
SMOKE_VAL=${FILELISTS}/h5list_v3_mc_diag1k.txt

CONFIG_ARG=${1:?usage: ./launch_p05_smoke.sh <config.py>}
CONFIG=$(cd "$WORKDIR" && readlink -f "$CONFIG_ARG")
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG_ARG"; exit 1; }

SAVE_REL=$(grep -m1 '^save_path' "$CONFIG" | sed 's/.*=\s*"\(.*\)".*/\1/')
RUN_ID=$(basename "$SAVE_REL")
TAG=smoke_$(echo "$RUN_ID" | tr '.' '_' | tr -cd 'A-Za-z0-9_-')

OPTS="save_path=sonata/p05/smoke/${RUN_ID}"
OPTS="$OPTS epoch=1 eval_epoch=1"
OPTS="$OPTS data.train.data_list_file=${SMOKE_TRAIN}"
OPTS="$OPTS data.val.data_list_file=${SMOKE_VAL}"

BASENAME=$(basename "$CONFIG")
if [[ "$BASENAME" == pretrain-sonata-* ]]; then
    # hooks index 6 = IterCheckpointSaver in the generated SSL configs:
    # force two early snapshots so the snapshot path is exercised.
    OPTS="$OPTS hooks.6.snapshot_at_iters=[20,60]"
elif [[ "$BASENAME" == supervised-ceiling-* ]]; then
    # hooks index 3 = SemSegEvaluator in the generated supervised configs:
    # evaluate twice within the short epoch.
    OPTS="$OPTS hooks.3.eval_freq=50"
fi

mkdir -p "${WORKDIR}/exp/logs" "${WORKDIR}/sonata/p05/smoke"

cd "$SCRIPT_DIR"
JOBID=$(sbatch --parsable --job-name="$TAG" --time=04:00:00 <<EOF
#!/bin/bash
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --output=${WORKDIR}/exp/logs/p05.%x.%j.%N.log
#SBATCH --error=${WORKDIR}/exp/logs/p05.%x.%j.%N.err
set -u
SQASHFILE=/projects/u6jo/datasets/combined_pretrain-sonata-v7-extbnb-larmatch.sqsh
container=/projects/u6jo/containers/pointcept-sandbox/
# NOTE: single line on purpose — backslash-newline continuations inside a
# heredoc within \$( ) command substitution collapse into literal backslashes,
# which apptainer then receives as an escaped-space command (" ").
srun apptainer exec --nv --bind \$SQASHFILE:/data:image-src=,ro,/projects/u6jo:/projects/u6jo \$container bash -c "cd ${WORKDIR} && source setenv_isambard_project_repo.sh && exec python3 tools/train.py --config ${CONFIG} --num-gpus 4 --options ${OPTS}"
EOF
)
echo "submitted smoke test ${TAG}: job ${JOBID}"
echo "outputs: ${WORKDIR}/sonata/p05/smoke/${RUN_ID}"
