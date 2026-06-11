#!/usr/bin/env bash
# Run any command inside the pointcept_sandbox singularity container, with:
#   - NVIDIA GPU access (--nv)
#   - the current working directory bound in (default)
#   - `src/` on PYTHONPATH (so `import condldm` works without pip install)
#
# Usage:
#   ./run_in_container.sh python scripts/train_autoencoder.py --config configs/autoencoder_protons64.yaml
#   ./run_in_container.sh pytest tests/ -q
#
# The container path can be overridden by setting the CONDLDM_CONTAINER env var.
# This script uses `SINGULARITYENV_*` env forwarding (compatible with Singularity 3.5+).

set -euo pipefail

: "${CONDLDM_CONTAINER:=/mnt/ddrive/containers/pointcept_sandbox/}"

if [[ ! -e "${CONDLDM_CONTAINER}" ]]; then
    echo "Container not found at ${CONDLDM_CONTAINER}" >&2
    echo "Set CONDLDM_CONTAINER=/path/to/sandbox to override." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Forward PYTHONPATH and CONDLDM_CACHE into the container via SINGULARITYENV_* prefix.
export SINGULARITYENV_PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SINGULARITYENV_CONDLDM_CACHE="${CONDLDM_CACHE:-${HOME}/.cache/condldm}"

# Bind-mount common data locations into the container.
# `/mnt/ddrive` holds the proton-track datasets and the container itself.
# Override / extend with CONDLDM_BINDS="-B /extra/path:/extra/path".
DEFAULT_BINDS="-B /mnt/ddrive:/mnt/ddrive"
if [[ -d /mnt/disk1 ]]; then
    DEFAULT_BINDS="${DEFAULT_BINDS} -B /mnt/disk1:/mnt/disk1"
fi

echo "exec singularity exec --nv ${DEFAULT_BINDS} ${CONDLDM_BINDS:-} \"${CONDLDM_CONTAINER}\" \"$@\""
exec singularity exec --nv ${DEFAULT_BINDS} ${CONDLDM_BINDS:-} "${CONDLDM_CONTAINER}" "$@"
