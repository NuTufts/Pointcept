#!/bin/bash
#
# Verify and unpack tarballs delivered by tar_and_transfer.py into the final
# datasets directory on Isambard.
#
# Each tarball lives under <staging>/<rel_dir>.tar.zst. Tarballs were created
# with paths relative to the original Tufts source prefix, so extracting from
# <dest_root> reproduces the original directory structure.
#
# Usage:
#   unpack_on_isambard.sh [staging_dir] [dest_root]
#
# Defaults:
#   staging_dir = /projects/u6jo/staging
#   dest_root   = /projects/u6jo/datasets
#
# All checksum files matching <staging>/*.sha256sums.txt (and the legacy
# sha256sums.txt) are processed. The script:
#   1. Verifies sha256sums against the tarballs in the staging area.
#   2. Extracts each tarball into dest_root.
#   3. Records extracted tarballs in <staging>/extracted.txt so re-runs skip
#      already-unpacked entries.

set -uo pipefail

STAGING_DIR="${1:-/projects/u6jo/staging}"
DEST_ROOT="${2:-/projects/u6jo/datasets}"

EXTRACTED_LOG="${STAGING_DIR}/extracted.txt"

if [[ ! -d "${STAGING_DIR}" ]]; then
    echo "ERROR: staging dir does not exist: ${STAGING_DIR}" >&2
    exit 1
fi

mkdir -p "${DEST_ROOT}"
touch "${EXTRACTED_LOG}"

shopt -s nullglob
CHECKSUM_FILES=("${STAGING_DIR}"/*.sha256sums.txt)
if [[ -f "${STAGING_DIR}/sha256sums.txt" ]]; then
    CHECKSUM_FILES+=("${STAGING_DIR}/sha256sums.txt")
fi

if (( ${#CHECKSUM_FILES[@]} == 0 )); then
    echo "ERROR: no checksum files (*.sha256sums.txt or sha256sums.txt) in ${STAGING_DIR}" >&2
    exit 1
fi

echo "staging_dir: ${STAGING_DIR}"
echo "dest_root:   ${DEST_ROOT}"
echo "checksum files:"
for f in "${CHECKSUM_FILES[@]}"; do
    echo "  ${f}"
done
echo

cd "${STAGING_DIR}"

echo "==> Verifying sha256 checksums..."
verify_failed=0
for cs in "${CHECKSUM_FILES[@]}"; do
    echo "-- ${cs}"
    if ! sha256sum -c "${cs}"; then
        verify_failed=1
    fi
done
if (( verify_failed )); then
    echo "ERROR: checksum verification failed. Aborting before any extraction." >&2
    exit 2
fi
echo "All checksums OK."
echo

# Read which tarballs are already extracted so we can skip them on re-run.
declare -A DONE=()
while IFS= read -r line; do
    [[ -n "${line}" ]] && DONE["${line}"]=1
done < "${EXTRACTED_LOG}"

n_total=0
n_skipped=0
n_extracted=0
n_failed=0

for cs in "${CHECKSUM_FILES[@]}"; do
    while read -r digest tarball_rel; do
        [[ -z "${tarball_rel}" ]] && continue
        n_total=$((n_total + 1))
        if [[ -n "${DONE[${tarball_rel}]:-}" ]]; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        if [[ ! -f "${tarball_rel}" ]]; then
            echo "WARNING: tarball missing in staging: ${tarball_rel}" >&2
            n_failed=$((n_failed + 1))
            continue
        fi
        echo "==> Extracting ${tarball_rel} into ${DEST_ROOT}"
        if tar -I zstd -xf "${tarball_rel}" -C "${DEST_ROOT}"; then
            echo "${tarball_rel}" >> "${EXTRACTED_LOG}"
            DONE["${tarball_rel}"]=1
            n_extracted=$((n_extracted + 1))
        else
            echo "ERROR: extraction failed for ${tarball_rel}" >&2
            n_failed=$((n_failed + 1))
        fi
    done < "${cs}"
done

echo
echo "Summary: total=${n_total} extracted=${n_extracted} skipped=${n_skipped} failed=${n_failed}"
if (( n_failed > 0 )); then
    exit 3
fi
