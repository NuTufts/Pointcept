#!/bin/bash
#
# SLURM wrapper for tar_and_transfer.py with optional self-chaining.
#
# Before submitting (or restarting): refresh the Isambard certificate with
# `clifton`. Refresh again at any point during the previous job's run so the
# chained successor picks up a valid cert when it starts.
#
# Usage:
#   sbatch submit_transfer.sh [--chain] [--max-chain N] \
#       <file_list> <source_prefix> [extra args to tar_and_transfer.py]
#
# Example (one-shot):
#   sbatch submit_transfer.sh \
#     /cluster/.../hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
#     /cluster/tufts/wongjiradlab/larbys/data/
#
# Example (auto-resubmit until complete):
#   sbatch submit_transfer.sh --chain --max-chain 30 \
#     /cluster/.../hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
#     /cluster/tufts/wongjiradlab/larbys/data/
#
# To stop a running chain without killing the active job, create a stop file:
#   touch <staging-dir>/<list-name>.stop_chain
# (where list-name is the file list basename without extension).
#
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --job-name=isambard_xfer
#SBATCH --output=logs/isambard_xfer-%j.out
#SBATCH --error=logs/isambard_xfer-%j.err

set -uo pipefail

# Capture original args verbatim so we can re-pass them to the chained job.
ORIG_ARGS=("$@")

CHAIN=0
MAX_CHAIN=20
while [[ $# -gt 0 ]]; do
    case "$1" in
        --chain) CHAIN=1; shift ;;
        --max-chain) MAX_CHAIN="$2"; shift 2 ;;
        --) shift; break ;;
        --*) break ;;
        *) break ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 [--chain] [--max-chain N] <file_list> <source_prefix> [tar_and_transfer.py args]" >&2
    exit 1
fi

FILE_LIST="$1"
SOURCE_PREFIX="$2"
shift 2
EXTRA_ARGS=("$@")

# Resolve script directory. On Tufts, sbatch stages this script into
# /var/spool/slurm/jobXXXX/ and runs it from there, so BASH_SOURCE points at
# the spool copy. Prefer SLURM_SUBMIT_DIR when it actually contains
# tar_and_transfer.py; otherwise fall back to BASH_SOURCE (interactive use).
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/tar_and_transfer.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "${SCRIPT_DIR}/tar_and_transfer.py" ]]; then
    echo "ERROR: tar_and_transfer.py not found in ${SCRIPT_DIR}" >&2
    echo "       Submit with: cd <isambard-dir> && sbatch submit_transfer.sh ..." >&2
    exit 1
fi
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
mkdir -p "${SCRIPT_DIR}/logs"

STAGING_DIR="${ISAMBARD_STAGING_DIR:-/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging}"
REMOTE_HOST="${ISAMBARD_REMOTE_HOST:-u6jo.aip2.isambard}"
REMOTE_STAGING_DIR="${ISAMBARD_REMOTE_STAGING_DIR:-/projects/u6jo/staging}"

mkdir -p "${STAGING_DIR}"

LIST_STEM="$(basename "${FILE_LIST}")"
LIST_STEM="${LIST_STEM%.*}"
COMPLETE_FILE="${STAGING_DIR}/${LIST_STEM}.complete"
STOP_FILE="${STAGING_DIR}/${LIST_STEM}.stop_chain"
CHAIN_COUNT_FILE="${STAGING_DIR}/${LIST_STEM}.chain_count"

echo "host:               $(hostname)"
echo "job_id:             ${SLURM_JOB_ID:-<not in slurm>}"
echo "file_list:          ${FILE_LIST}"
echo "source_prefix:      ${SOURCE_PREFIX}"
echo "staging_dir:        ${STAGING_DIR}"
echo "remote_host:        ${REMOTE_HOST}"
echo "remote_staging_dir: ${REMOTE_STAGING_DIR}"
echo "chain:              ${CHAIN} (max=${MAX_CHAIN})"
echo

# Short-circuit if already complete: no work, no chain.
if [[ -f "${COMPLETE_FILE}" ]]; then
    echo "Already complete (${COMPLETE_FILE}); nothing to do."
    rm -f "${CHAIN_COUNT_FILE}"
    exit 0
fi

# Probe ssh / cert. Failure here aborts before doing real work and breaks the
# chain (so we don't pile up doomed jobs on an expired cert).
echo "Probing ssh to ${REMOTE_HOST}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "mkdir -p ${REMOTE_STAGING_DIR} && echo OK"; then
    echo "ERROR: ssh to ${REMOTE_HOST} failed. Refresh the clifton cert and resubmit." >&2
    exit 2
fi

PYTHON="${PYTHON:-python3}"

set +e
"${PYTHON}" "${SCRIPT_DIR}/tar_and_transfer.py" \
    --file-list "${FILE_LIST}" \
    --source-prefix "${SOURCE_PREFIX}" \
    --staging-dir "${STAGING_DIR}" \
    --remote-host "${REMOTE_HOST}" \
    --remote-staging-dir "${REMOTE_STAGING_DIR}" \
    --zstd-threads "${SLURM_CPUS_PER_TASK:-0}" \
    "${EXTRA_ARGS[@]}"
PY_EXIT=$?
set -e

echo
echo "tar_and_transfer.py exited with status ${PY_EXIT}"

# ---- Chain logic ----
if (( CHAIN == 0 )); then
    exit ${PY_EXIT}
fi

if [[ -f "${COMPLETE_FILE}" ]]; then
    echo "Sentinel ${COMPLETE_FILE} present; chain complete."
    rm -f "${CHAIN_COUNT_FILE}"
    exit ${PY_EXIT}
fi

if [[ -f "${STOP_FILE}" ]]; then
    echo "Stop file ${STOP_FILE} present; chain halted by user."
    exit ${PY_EXIT}
fi

count=$(cat "${CHAIN_COUNT_FILE}" 2>/dev/null || echo 0)
count=$((count + 1))
if (( count >= MAX_CHAIN )); then
    echo "Chain count ${count} reached --max-chain ${MAX_CHAIN}; not chaining further."
    exit ${PY_EXIT}
fi

# Only chain if we made progress OR ran cleanly. If python crashed before
# doing anything (e.g. argparse error), don't pile on identical failures.
if (( PY_EXIT != 0 && PY_EXIT != 2 )); then
    echo "tar_and_transfer.py crashed (exit=${PY_EXIT}); not chaining."
    exit ${PY_EXIT}
fi

echo "${count}" > "${CHAIN_COUNT_FILE}"
DEP="${SLURM_JOB_ID:-}"
if [[ -z "${DEP}" ]]; then
    echo "WARNING: no SLURM_JOB_ID; cannot chain (run via sbatch, not directly)."
    exit ${PY_EXIT}
fi
NEXT=$(sbatch --parsable --chdir="${SCRIPT_DIR}" --dependency=afterany:"${DEP}" "${SCRIPT_PATH}" "${ORIG_ARGS[@]}") || NEXT=""
if [[ -n "${NEXT}" ]]; then
    echo "Submitted chained successor: jobid=${NEXT} (chain ${count}/${MAX_CHAIN})"
    echo "Remember to refresh the clifton cert before this job ends."
else
    echo "WARNING: failed to submit chained successor."
fi

exit ${PY_EXIT}
