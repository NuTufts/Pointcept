#!/bin/bash
#
# submit_valtest.sh — submit a SLURM array that runs LArFormer Stage-3
# particle-segmenter inference + per-event analysis on the val (or test)
# split for one (dataset TAG x model) pair.
#
# Usage:
#   1. Edit a per-(TAG x model) config (see valtest_pi0_ptv3crosslevel.conf)
#      with paths to your model + cache (or rerun_lines for full-cascade).
#   2. Submit (from the repo root, on a Tufts head node):
#          sbatch lartpc/larformer_analysis/particle_eval/slurm/submit_valtest.sh \
#                 lartpc/larformer_analysis/particle_eval/slurm/valtest_pi0_ptv3crosslevel.conf
#      The script auto-determines the array size from the input source.
#
# What each SLURM array task does:
#   - In cached mode: takes its share (stride consecutive events) of the
#     enumerated cache events, runs Stage-3 inference once on the whole
#     chunk, then loops `analyze_event.py` per surviving event.
#   - In full-cascade mode: takes its share of filenos from RERUN_LINES_FILE,
#     selects manifest rows, runs the cascade end-to-end, analyzes.
#
# Notes:
#   - SBATCH knobs come from the conf via the auto-resubmit block below.

# Detect: are we already running under sbatch? If not, re-exec ourselves
# under sbatch using the SBATCH knobs from the conf.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${VALTEST_RESUBMITTED:-}" ]; then
    set -e
    CONF="${1:?Usage: $0 <valtest_*.conf>}"
    if [ ! -f "$CONF" ]; then
        echo "ERROR: conf file not found: $CONF" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONF"

    INPUT_MODE="${INPUT_MODE:-cached}"
    if [ "$INPUT_MODE" = "cached" ]; then
        if [ ! -d "${CACHE_DIR}/${SPLIT}" ]; then
            echo "ERROR: cache dir not found: ${CACHE_DIR}/${SPLIT}" >&2
            exit 1
        fi
        # Count cache event files. The cache builder writes
        # *.h5 in a hashed tree under ${CACHE_DIR}/${SPLIT}/.
        N_EVENTS=$(find "${CACHE_DIR}/${SPLIT}" -name '*.h5' | wc -l)
        if [ "$N_EVENTS" -le 0 ]; then
            echo "ERROR: no cache events found under ${CACHE_DIR}/${SPLIT}" >&2
            exit 1
        fi
    elif [ "$INPUT_MODE" = "full-cascade" ]; then
        if [ ! -f "$RERUN_LINES_FILE" ]; then
            echo "ERROR: RERUN_LINES_FILE not found: $RERUN_LINES_FILE" >&2
            exit 1
        fi
        N_EVENTS=$(grep -c '^[^#[:space:]]' "$RERUN_LINES_FILE")
        if [ "$N_EVENTS" -le 0 ]; then
            echo "ERROR: RERUN_LINES_FILE is empty: $RERUN_LINES_FILE" >&2
            exit 1
        fi
    else
        echo "ERROR: unknown INPUT_MODE='${INPUT_MODE}' (expected cached|full-cascade)" >&2
        exit 1
    fi

    STRIDE="${STRIDE:-1}"
    if [ "$STRIDE" -le 0 ]; then
        echo "ERROR: STRIDE must be a positive integer (got $STRIDE)" >&2
        exit 1
    fi
    # ceil(N_EVENTS / STRIDE)
    N_TASKS=$(( (N_EVENTS + STRIDE - 1) / STRIDE ))
    LAST_TASK=$((N_TASKS - 1))

    mkdir -p "${SBATCH_LOG_DIR:-./logs}"
    CONF_ABS="$(readlink -f "$CONF")"
    SCRIPT_ABS="$(readlink -f "${BASH_SOURCE[0]}")"

    echo "Launching SLURM array: 0-${LAST_TASK} (=${N_TASKS} tasks)"
    echo "  INPUT_MODE = ${INPUT_MODE}"
    echo "  N_EVENTS   = ${N_EVENTS}"
    echo "  STRIDE     = ${STRIDE}  (events per task)"
    echo "  conf       = ${CONF_ABS}"
    echo "  log dir    = ${SBATCH_LOG_DIR}"
    echo "  partition  = ${SBATCH_PARTITION}"
    echo "  time       = ${SBATCH_TIME}"

    if [[ "${SKIP_INFERENCE}" != "true" ]]; then
        echo "Launching SLURM array for inference + analysis..."
        exec sbatch \
            --job-name="valtest_s3_${TAG}" \
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
    else
        echo "Launching SLURM array for analysis only..."
        exec sbatch \
            --job-name="valtest_s3_${TAG}" \
            --array=0-${LAST_TASK} \
            --partition=batch \
            --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
            --mem-per-cpu="${SBATCH_MEM_PER_CPU}" \
            --time="${SBATCH_TIME}" \
            --output="${SBATCH_LOG_DIR}/task_%A_%a.out" \
            --error="${SBATCH_LOG_DIR}/task_%A_%a.err" \
            --export=ALL,VALTEST_CONFIG="${CONF_ABS}",VALTEST_RESUBMITTED=1 \
            "${SCRIPT_ABS}"
    fi
fi

# ----- below here: actually running on a compute node under sbatch ----
set -eu

CONF="${VALTEST_CONFIG:?VALTEST_CONFIG env var must be set (re-exec from launcher)}"
# shellcheck disable=SC1090
source "$CONF"

INPUT_MODE="${INPUT_MODE:-cached}"

echo "======================================"
echo "valtest task SLURM_JOB_ID=${SLURM_JOB_ID:-?}  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "TAG=${TAG}  MODEL_TAG=${MODEL_TAG}  INPUT_MODE=${INPUT_MODE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "Started: $(date)"
echo "======================================"

# the -suid variant only exists on some nodes (wongjiradlab); gpu/preempt
# nodes carry a different default -- tolerate whatever is available
module load apptainer/1.2.4-suid 2>/dev/null \
  || module load apptainer 2>/dev/null || true

# In-container per-task command. The driver script handles both input
# modes; only the knobs it needs are different.
if [ "$INPUT_MODE" = "cached" ]; then
    EXTRA_ARGS=" \
        --cache-dir ${CACHE_DIR} \
        --split ${SPLIT:-val}"
else
    EXTRA_ARGS=" \
        --rerun-lines-file ${RERUN_LINES_FILE} \
        --manifest-csv ${MANIFEST_CSV}"
fi

if [ "${SKIP_ANALYSIS}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-analysis"
fi
if [ "${SKIP_INFERENCE}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-inference"
fi

INNER_CMD="\
cd ${WORKDIR} && \
python lartpc/larformer_analysis/particle_eval/slurm/run_valtest_per_task.py \
    --tag ${TAG} \
    --model-config ${MODEL_CONFIG} \
    --model-weights ${MODEL_WEIGHTS} \
    --model-tag ${MODEL_TAG} \
    --output-dir ${OUTPUT_DIR} \
    --input-mode ${INPUT_MODE} \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --stride ${STRIDE:-1} \
    --class-prob-threshold ${CLASS_PROB_THRESHOLD:-0.0} \
    ${EXTRA_ARGS}"

apptainer exec --nv --bind /cluster:/cluster \
    "${CONTAINER}" bash -c "${INNER_CMD}"
RC=$?

echo "Finished: $(date)  exit=${RC}"
exit ${RC}
