#!/bin/bash
#
# Verify .sqsh shards delivered by transfer_shards.py against their
# sha256sums on Isambard. No extraction — shards are mounted at runtime via
# Apptainer (--bind <shard>.sqsh:/data:image-src=/,ro), so verification is
# the only thing the destination side needs to do before training.
#
# Run on Isambard after a transfer chain completes, or after each chunk if
# you want incremental checks.
#
# Usage:
#   verify_shards_on_isambard.sh [staging_dir]
#
# Default:
#   staging_dir = /projects/u6jo/staging
#
# Processes every <staging>/*.sha256sums.txt file. For each, runs
# `sha256sum -c` on the shards listed there. Exits 0 if all pass, 2 if any
# shard failed verification, 1 on setup error.
#
# Note: with 5-10 TB of shards this is read-IO bound and can take several
# hours. Run it in screen/tmux or as a background job.

set -uo pipefail

STAGING_DIR="${1:-/projects/u6jo/staging}"

if [[ ! -d "${STAGING_DIR}" ]]; then
    echo "ERROR: staging dir does not exist: ${STAGING_DIR}" >&2
    exit 1
fi

shopt -s nullglob
CHECKSUM_FILES=("${STAGING_DIR}"/*.sha256sums.txt)

if (( ${#CHECKSUM_FILES[@]} == 0 )); then
    echo "ERROR: no checksum files (*.sha256sums.txt) in ${STAGING_DIR}" >&2
    echo "       Has the transfer phase pushed any shards yet?" >&2
    exit 1
fi

echo "staging_dir:    ${STAGING_DIR}"
echo "checksum files:"
total_listed=0
for f in "${CHECKSUM_FILES[@]}"; do
    n=$(grep -cv '^[[:space:]]*\(#\|$\)' "${f}" || true)
    total_listed=$((total_listed + n))
    echo "  ${f}  (${n} shards)"
done
echo "total shards across all datasets: ${total_listed}"
echo

cd "${STAGING_DIR}"

verify_failed=0
n_passed_files=0
n_failed_files=0
for cs in "${CHECKSUM_FILES[@]}"; do
    cs_basename="$(basename "${cs}")"
    echo "==> ${cs_basename}"
    # Capture sha256sum -c output to count OK / FAILED lines per dataset.
    tmp_out="$(mktemp)"
    if sha256sum -c "${cs}" 2>&1 | tee "${tmp_out}"; then
        ok_count=$(grep -c ': OK$' "${tmp_out}" || true)
        echo "    ${cs_basename}: ${ok_count} shard(s) OK"
        n_passed_files=$((n_passed_files + ok_count))
    else
        verify_failed=1
        ok_count=$(grep -c ': OK$' "${tmp_out}" || true)
        bad_count=$(grep -cE ': (FAILED|FAILED open or read)$' "${tmp_out}" || true)
        n_passed_files=$((n_passed_files + ok_count))
        n_failed_files=$((n_failed_files + bad_count))
        echo "    ${cs_basename}: ${ok_count} OK, ${bad_count} FAILED"
    fi
    rm -f "${tmp_out}"
    echo
done

echo "================================================================"
echo "Summary:  passed=${n_passed_files}  failed=${n_failed_files}  total_listed=${total_listed}"

if (( verify_failed )); then
    echo
    echo "ERROR: at least one shard failed verification." >&2
    echo "       Recovery on Tufts:" >&2
    echo "         1. Identify the offending shard(s) from the FAILED lines above." >&2
    echo "         2. Delete the local .sqsh under \${ISAMBARD_STAGING_DIR}." >&2
    echo "         3. Remove the matching line in" >&2
    echo "              <stem>.sha256sums.txt, <stem>.manifest.txt," >&2
    echo "              and <stem>.transfer_manifest.txt." >&2
    echo "         4. Also delete <stem>.complete and <stem>.transfer_complete." >&2
    echo "         5. Resubmit submit_build.sh and submit_transfer_shards.sh." >&2
    exit 2
fi

echo
echo "All ${total_listed} shards verified OK across ${#CHECKSUM_FILES[@]} dataset(s)."
echo "Mount at training time via:"
echo "  apptainer exec --bind <shard>.sqsh:/data:image-src=/,ro <pointcept.sif> ..."
