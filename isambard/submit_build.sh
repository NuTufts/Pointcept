#!/bin/bash
#
# SLURM wrapper for build_squashfs.py with optional self-chaining.
#
# Build-only job: produces .sqsh shards locally, no rsync, no Isambard cert
# required. Use during Isambard downtime; pair with submit_transfer_shards.sh
# once Isambard is reachable again.
#
# Usage:
#   sbatch submit_build.sh [--chain] [--max-chain N] \
#       <file_list> <source_prefix> [extra args to build_squashfs.py]
#
# Example (one-shot):
#   sbatch submit_build.sh \
#     /cluster/.../hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
#     /cluster/tufts/wongjiradlab/larbys/data/
#
# Example (auto-resubmit until complete):
#   sbatch submit_build.sh --chain --max-chain 10 \
#     /cluster/.../hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt \
#     /cluster/tufts/wongjiradlab/larbys/data/
#
# To stop a running chain without killing the active job, create a stop file:
#   touch <staging-dir>/<list-name>.stop_chain
#
#SBATCH --partition=batch
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --job-name=sqsh_build
#SBATCH --output=logs/sqsh_build_bnb_nue_corsika-%j.out
#SBATCH --error=logs/sqsh_build_bnb_nue_corsika-%j.err

set -uo pipefail

ORIG_ARGS=("$@")

CHAIN=0
MAX_CHAIN=10
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
    echo "Usage: sbatch $0 [--chain] [--max-chain N] <file_list> <source_prefix> [build_squashfs.py args]" >&2
    exit 1
fi

FILE_LIST="$1"
SOURCE_PREFIX="$2"
shift 2
EXTRA_ARGS=("$@")

# Resolve script directory (mirrors submit_transfer.sh).
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/build_squashfs.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "${SCRIPT_DIR}/build_squashfs.py" ]]; then
    echo "ERROR: build_squashfs.py not found in ${SCRIPT_DIR}" >&2
    echo "       Submit with: cd <isambard-dir> && sbatch submit_build.sh ..." >&2
    exit 1
fi
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
mkdir -p "${SCRIPT_DIR}/logs"

STAGING_DIR="${ISAMBARD_STAGING_DIR:-/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging}"

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
echo "chain:              ${CHAIN} (max=${MAX_CHAIN})"
echo

if [[ -f "${COMPLETE_FILE}" ]]; then
    echo "Already complete (${COMPLETE_FILE}); nothing to do."
    rm -f "${CHAIN_COUNT_FILE}"
    exit 0
fi

# Sanity: mksquashfs must be on PATH.
if ! command -v mksquashfs >/dev/null 2>&1; then
    echo "ERROR: mksquashfs not in PATH on this node." >&2
    exit 2
fi

PYTHON="${PYTHON:-python3}"

set +e
"${PYTHON}" "${SCRIPT_DIR}/build_squashfs.py" \
    --file-list "${FILE_LIST}" \
    --source-prefix "${SOURCE_PREFIX}" \
    --staging-dir "${STAGING_DIR}" \
    --processors "${SLURM_CPUS_PER_TASK:-0}" \
    "${EXTRA_ARGS[@]}"
PY_EXIT=$?
set -e

echo
echo "build_squashfs.py exited with status ${PY_EXIT}"

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

# Don't chain on hard crashes (argparse error, mksquashfs missing, etc).
# Exit 2 means "some shards failed but the worker handled it cleanly".
if (( PY_EXIT != 0 && PY_EXIT != 2 )); then
    echo "build_squashfs.py crashed (exit=${PY_EXIT}); not chaining."
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
else
    echo "WARNING: failed to submit chained successor."
fi

exit ${PY_EXIT}
