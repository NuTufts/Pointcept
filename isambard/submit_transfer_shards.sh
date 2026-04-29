#!/bin/bash
#
# SLURM wrapper for transfer_shards.py with optional self-chaining.
#
# Transfers .sqsh shards already produced by build_squashfs.py to Isambard.
# Mirrors submit_transfer.sh's cert + chain logic — refresh `clifton` before
# submitting and again partway through, so the chained successor inherits a
# valid cert.
#
# Usage:
#   sbatch submit_transfer_shards.sh [--chain] [--max-chain N] \
#       <file_list> [extra args to transfer_shards.py]
#
# (Note: unlike submit_transfer.sh, no <source_prefix> argument — the shards
# already exist on disk; only the file_list is needed to derive the per-list
# stem used to find the checksums file.)
#
# Example (auto-resubmit until complete):
#   sbatch submit_transfer_shards.sh --chain --max-chain 30 \
#     /cluster/.../hdlist_extbnb_larmatch_run3_g1_sonata_validated.txt
#
#SBATCH --partition=batch
#SBATCH --time=11:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --job-name=sqsh_xfer
#SBATCH --output=logs/sqsh_xfer-%j.out
#SBATCH --error=logs/sqsh_xfer-%j.err

set -uo pipefail

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

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 [--chain] [--max-chain N] <file_list> [transfer_shards.py args]" >&2
    exit 1
fi

FILE_LIST="$1"
shift
EXTRA_ARGS=("$@")

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/transfer_shards.py" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [[ ! -f "${SCRIPT_DIR}/transfer_shards.py" ]]; then
    echo "ERROR: transfer_shards.py not found in ${SCRIPT_DIR}" >&2
    exit 1
fi
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
mkdir -p "${SCRIPT_DIR}/logs"

STAGING_DIR="${ISAMBARD_STAGING_DIR:-/cluster/tufts/wongjiradlab/larbys/data/ub_on_tufts/isambard_staging}"
REMOTE_HOST="${ISAMBARD_REMOTE_HOST:-u6jo.aip2.isambard}"
REMOTE_STAGING_DIR="${ISAMBARD_REMOTE_STAGING_DIR:-/projects/u6jo/staging}"

LIST_STEM="$(basename "${FILE_LIST}")"
LIST_STEM="${LIST_STEM%.*}"
COMPLETE_FILE="${STAGING_DIR}/${LIST_STEM}.transfer_complete"
STOP_FILE="${STAGING_DIR}/${LIST_STEM}.stop_chain"
CHAIN_COUNT_FILE="${STAGING_DIR}/${LIST_STEM}.transfer_chain_count"

echo "host:               $(hostname)"
echo "job_id:             ${SLURM_JOB_ID:-<not in slurm>}"
echo "file_list:          ${FILE_LIST}"
echo "staging_dir:        ${STAGING_DIR}"
echo "remote_host:        ${REMOTE_HOST}"
echo "remote_staging_dir: ${REMOTE_STAGING_DIR}"
echo "chain:              ${CHAIN} (max=${MAX_CHAIN})"
echo

if [[ -f "${COMPLETE_FILE}" ]]; then
    echo "Already complete (${COMPLETE_FILE}); nothing to do."
    rm -f "${CHAIN_COUNT_FILE}"
    exit 0
fi

# Probe ssh / cert before any real work; abort the chain on cert expiry so we
# don't pile up failed jobs.
echo "Probing ssh to ${REMOTE_HOST}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "${REMOTE_HOST}" "mkdir -p ${REMOTE_STAGING_DIR} && echo OK"; then
    echo "ERROR: ssh to ${REMOTE_HOST} failed. Refresh the clifton cert and resubmit." >&2
    exit 2
fi

PYTHON="${PYTHON:-python3}"

set +e
"${PYTHON}" "${SCRIPT_DIR}/transfer_shards.py" \
    --file-list "${FILE_LIST}" \
    --staging-dir "${STAGING_DIR}" \
    --remote-host "${REMOTE_HOST}" \
    --remote-staging-dir "${REMOTE_STAGING_DIR}" \
    "${EXTRA_ARGS[@]}"
PY_EXIT=$?
set -e

echo
echo "transfer_shards.py exited with status ${PY_EXIT}"

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

if (( PY_EXIT != 0 && PY_EXIT != 2 )); then
    echo "transfer_shards.py crashed (exit=${PY_EXIT}); not chaining."
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
