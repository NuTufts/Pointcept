#!/bin/bash
#
# validate_shower_clustering_data.sh
#
# Wrapper that runs validate_shower_clustering_data.py inside the pointcept
# container with the right env so `from pointcept.datasets.shower_clustering
# import ShowerClusteringDataset` works for the functional check.
#
# Usage:
#   bash validate_shower_clustering_data.sh PATH [PATH ...] [--functional] [...]
#
# PATH may be an H5 file or a directory (recursively scanned for
# merged_*_entry*.h5). All other flags are passed through.
#
# Examples:
#   # Static checks across an entire dataset:
#   bash validate_shower_clustering_data.sh \\
#       /cluster/.../v3_larmatch/bnb_nu_corsika_prod2
#
#   # Static + dataset-loader pass on the first 50 files:
#   bash validate_shower_clustering_data.sh \\
#       /cluster/.../v3_larmatch/bnb_nu_corsika_prod2 \\
#       --functional --max-files 50 --verbose
#
#   # Validate a specific list and write the failures to a file:
#   bash validate_shower_clustering_data.sh --from-list paths.txt \\
#       --functional --fail-list bad_files.txt
#

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POINTCEPT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
POINTCEPT_CONTAINER=${POINTCEPT_CONTAINER:-/cluster/tufts/wongjiradlabnu/larbys/containers/pointcept_cuml.sif}

if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer/1.2.4-suid 2>/dev/null || module load apptainer
fi

PY="${SCRIPT_DIR}/validate_shower_clustering_data.py"

apptainer exec --nv --bind /cluster:/cluster "${POINTCEPT_CONTAINER}" \
    bash -c '
set -e
UBDL_DIR=$1; PY=$2; shift 2
cd "$UBDL_DIR"
source setenv_pointcept_container.sh
exec python3 "$PY" "$@"
' _ "${UBDL_DIR}" "${PY}" "$@"
