#!/bin/bash
#
# write_completion_sentinels.sh
#
# One-shot recovery: scan MERGEFILE_OUTPUT_DIR for fully-complete files and
# write the per-line `${TAG}_fileno${ZFILENO}.complete` sentinels that the
# resume logic in run_lantern_wconfig.sh checks before processing each line.
#
# Usage:
#   bash write_completion_sentinels.sh <config_file> [extra args...]
#
# Examples:
#   # Dry run — print which lines are complete, partial, or missing entries:
#   bash write_completion_sentinels.sh lantern_configs/bnbnu_coriska.conf --report-only
#
#   # Actually write sentinels:
#   bash write_completion_sentinels.sh lantern_configs/bnbnu_coriska.conf
#
#   # Limit to a line range:
#   bash write_completion_sentinels.sh lantern_configs/bnbnu_coriska.conf --lines 1:50
#
#   # Skip the input-ROOT entry-count check (mark any line with contiguous
#   # existing entries as complete):
#   bash write_completion_sentinels.sh lantern_configs/bnbnu_coriska.conf --force
#
#   # Also write a rerun list of original line numbers for incomplete lines.
#   # Feed this back into the driver via RERUN_LINES_FILE in a rerun config to
#   # resubmit only the lines that still need work (no scanning of complete
#   # lines from inside SLURM jobs):
#   bash write_completion_sentinels.sh lantern_configs/bnbnu_coriska.conf \
#       --report-only --write-rerun-list /path/to/rerun_lines.txt
#
# Runs the Python helper inside the pointcept container so larlite is
# available for counting entries in dlmerged ROOT files.
#

set -euo pipefail

CONFIG_FILE="${1:-}"
shift || true

if [ -z "${CONFIG_FILE}" ] || [ ! -f "${CONFIG_FILE}" ]; then
    echo "Usage: $0 <config_file> [--report-only] [--lines a:b] [--force]" >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
POINTCEPT_DIR=${POINTCEPT_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept}
export REPO_ROOT UBDL_DIR POINTCEPT_DIR

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

# Some configs only set OUTPUT_DIR; the driver writes merged H5s under
# MERGEFILE_OUTPUT_DIR. If unset, fall back to OUTPUT_DIR.
MERGEFILE_OUTPUT_DIR=${MERGEFILE_OUTPUT_DIR:-${OUTPUT_DIR:-}}

for v in INPUTLIST TAG MERGEFILE_OUTPUT_DIR POINTCEPT_CONTAINER; do
    if [ -z "${!v:-}" ]; then
        echo "ERROR: config must set ${v} (or OUTPUT_DIR)" >&2
        exit 1
    fi
done

echo "Scanning ${MERGEFILE_OUTPUT_DIR} for ${TAG} ..."

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer/1.2.4-suid 2>/dev/null || module load apptainer
fi

PY="${SCRIPT_DIR}/write_completion_sentinels.py"

# Run inside the pointcept container with the larlite/larcv env in place.
# setenv_pointcept_container.sh prepends the right PYTHONPATH so `import
# larlite` works (same setup the Step 2-4 driver uses).
#
# Pass UBDL_DIR + PY + python args as positional args after `--`; the inner
# bash -c uses single quotes so nothing is expanded by the outer shell,
# avoiding empty-arg artifacts when "$@" is empty.
apptainer exec --bind /cluster:/cluster "${POINTCEPT_CONTAINER}" \
    bash -c '
set -e
UBDL_DIR=$1; PY=$2; shift 2
cd "$UBDL_DIR"
source setenv_pointcept_container.sh
exec python3 "$PY" "$@"
' _ "${UBDL_DIR}" "${PY}" \
    --inputlist "${INPUTLIST}" \
    --output-dir "${MERGEFILE_OUTPUT_DIR}" \
    --tag "${TAG}" \
    "$@"
