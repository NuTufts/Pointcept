#!/bin/bash
#
# submit_valtest.sh — submit a SLURM array that runs LArFormer inference +
# per-event analysis for one (dataset TAG × model) on the val+test set.
#
# Usage:
#   1. Edit a per-TAG config (see valtest_pi0_crosslevel.conf) with paths
#      to your model + Stage-0 outputs.
#   2. Submit (from the repo root, on a Tufts head node):
#          sbatch lartpc_data_prep/larformer_analysis/slurm/submit_valtest.sh \
#                 lartpc_data_prep/larformer_analysis/slurm/valtest_pi0_crosslevel.conf
#      The script auto-determines the array size from RERUN_LINES_FILE.
#
# What it does (per SLURM array task):
#   - reads the SLURM_ARRAY_TASK_ID-th line from RERUN_LINES_FILE (= a fileno)
#   - selects manifest rows for that fileno
#   - runs `tools/run_slicer_inference.py` once on those merged H5s
#   - runs `analyze_event.py` per surviving event
#
# Notes:
#   - The SBATCH directives below are PLACEHOLDERS overridden via
#     `sbatch --partition=… --time=… …` on the command line OR via
#     `--export=ALL,VALTEST_CONFIG=<conf>` if you wrap it in a launcher.
#   - The simplest way to set #SBATCH partition/time/etc. from the conf
#     is the launcher form: this script accepts the conf as $1 and
#     re-issues itself via sbatch with the SBATCH overrides from the
#     conf. See the "auto-resubmit" block at the top.

# Detect: are we already running under sbatch? If not, re-exec ourselves
# under sbatch using the SBATCH knobs from the conf.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${VALTEST_RESUBMITTED:-}" ]; then
    set -e
    CONF="${1:?Usage: $0 <valtest_*.conf>}"
    if [ ! -f "$CONF" ]; then
        echo "ERROR: conf file not found: $CONF" >&2
        exit 1
    fi
    # Source to learn SBATCH knobs + array size.
    # shellcheck disable=SC1090
    source "$CONF"

    if [ ! -f "$RERUN_LINES_FILE" ]; then
        echo "ERROR: RERUN_LINES_FILE not found: $RERUN_LINES_FILE" >&2
        exit 1
    fi
    N_FILENOS=$(grep -c '^[^#[:space:]]' "$RERUN_LINES_FILE")
    if [ "$N_FILENOS" -le 0 ]; then
        echo "ERROR: RERUN_LINES_FILE is empty: $RERUN_LINES_FILE" >&2
        exit 1
    fi
    STRIDE="${STRIDE:-1}"
    if [ "$STRIDE" -le 0 ]; then
        echo "ERROR: STRIDE must be a positive integer (got $STRIDE)" >&2
        exit 1
    fi
    # ceil(N_FILENOS / STRIDE) = (N_FILENOS + STRIDE - 1) / STRIDE
    N_TASKS=$(( (N_FILENOS + STRIDE - 1) / STRIDE ))
    LAST_TASK=$((N_TASKS - 1))

    mkdir -p "${SBATCH_LOG_DIR:-./logs}"
    CONF_ABS="$(readlink -f "$CONF")"
    SCRIPT_ABS="$(readlink -f "${BASH_SOURCE[0]}")"

    echo "Launching SLURM array: 0-${LAST_TASK} (=${N_TASKS} tasks)"
    echo "  N_FILENOS  = ${N_FILENOS}"
    echo "  STRIDE     = ${STRIDE}  (filenos per task)"
    echo "  conf       = ${CONF_ABS}"
    echo "  log dir    = ${SBATCH_LOG_DIR}"
    echo "  partition  = ${SBATCH_PARTITION}"
    echo "  time       = ${SBATCH_TIME}"

    exec sbatch \
        --job-name="valtest_${TAG}" \
        --array=0-${LAST_TASK} \
        --partition="${SBATCH_PARTITION}" \
        --gres=gpu:1 \
        --constraint="${SBATCH_GPU_CONSTRAINT}" \
        --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
        --mem-per-cpu="${SBATCH_MEM_PER_CPU}" \
        --time="${SBATCH_TIME}" \
        --output="${SBATCH_LOG_DIR}/task_%A_%a.out" \
        --error="${SBATCH_LOG_DIR}/task_%A_%a.err" \
        --export=ALL,VALTEST_CONFIG="${CONF_ABS}",VALTEST_RESUBMITTED=1 \
        "${SCRIPT_ABS}"
fi

# ----- below here: actually running on a compute node under sbatch ----
set -eu

CONF="${VALTEST_CONFIG:?VALTEST_CONFIG env var must be set (re-exec from launcher)}"
# shellcheck disable=SC1090
source "$CONF"

echo "======================================"
echo "valtest task SLURM_JOB_ID=${SLURM_JOB_ID:-?}  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "TAG=${TAG}  MODEL_TAG=${MODEL_TAG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "Started: $(date)"
echo "======================================"

module load apptainer # use this for production cluster
#module load apptainer/1.2.4-suid # use this on old cluster

# In-container per-task command.
INNER_CMD="\
cd ${WORKDIR} && \
python lartpc_data_prep/larformer_analysis/slurm/run_valtest_per_fileno.py \
    --tag ${TAG} \
    --rerun-lines-file ${RERUN_LINES_FILE} \
    --manifest-csv ${MANIFEST_CSV} \
    --model-config ${MODEL_CONFIG} \
    --model-weights ${MODEL_WEIGHTS} \
    --model-tag ${MODEL_TAG} \
    --output-dir ${OUTPUT_DIR} \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --stride ${STRIDE:-1} \
    --gamma-beam ${GAMMA_BEAM} \
    --gamma-cosmic ${GAMMA_COSMIC} \
    --charge-source ${CHARGE_SOURCE:-raw_adc} \
    --charge-share-norm ${CHARGE_SHARE_NORM:-per-slice}"

apptainer exec --nv --bind /cluster:/cluster \
    "${CONTAINER}" bash -c "${INNER_CMD}"
RC=$?

echo "Finished: $(date)  exit=${RC}"
exit ${RC}
