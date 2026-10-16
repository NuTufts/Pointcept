#!/bin/bash
# ---------------------------------------------------------------------------
# Globus CLI transfer between Tufts and ALCF Eagle (either direction).
#
#   bash globus_transfer.sh to-polaris <tufts dir> <eagle dir> [--wait] [--dry-run]
#   bash globus_transfer.sh to-tufts   <eagle dir> <tufts dir> [--wait] [--dry-run]
#   bash globus_transfer.sh status <task id>   |   bash globus_transfer.sh ls <to-polaris|to-tufts> <dir>
#
# Endpoint UUIDs come from lartpc/data_prep/squashfs/globus_endpoints.env
# (copy globus_endpoints.env.example; find UUIDs with `globus endpoint search
# <name>`: Eagle = "alcf#dtn_eagle", Tufts = the Tufts HPC collection you used
# for the dlmerged copy). Paths: Tufts collections expose the /cluster/... path
# as is; Eagle is /eagle/<project>/... (see docs.alcf.anl.gov data-transfer).
#
# Setup once, on any Tufts node (shared home; login nodes lack a `pip` wrapper):
#   python3 -m pip install --user globus-cli   -> ~/.local/bin/globus (add to PATH)
#   globus login            (browser flow)      globus whoami
# Both collections must be activated/consented in the browser the first time
# (ALCF needs the OTP from the CryptoCARD/MobilePASS token).
#
# The transfer is recursive with sync-level checksum + checksum verification,
# so re-running it after a partial transfer only sends what is missing/changed.
# --wait blocks until the task finishes (prints progress every 60 s).
# ---------------------------------------------------------------------------
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ENVF=${GLOBUS_ENV:-$HERE/globus_endpoints.env}
MODE=${1:?to-polaris|to-tufts|status|ls}; shift
command -v globus >/dev/null 2>&1 || { echo "ERROR: globus CLI not found: pip install --user globus-cli (and make sure ~/.local/bin is on PATH)" >&2; exit 2; }
if [ "$MODE" = status ]; then globus task show "${1:?task id}"; exit 0; fi
[ -r "$ENVF" ] || { echo "ERROR: $ENVF missing (copy globus_endpoints.env.example and fill in the UUIDs)" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENVF"
: "${GLOBUS_TUFTS_EP:?}" "${GLOBUS_EAGLE_EP:?}"
case "$MODE" in
  to-polaris) SRC_EP=$GLOBUS_TUFTS_EP; DST_EP=$GLOBUS_EAGLE_EP ;;
  to-tufts)   SRC_EP=$GLOBUS_EAGLE_EP; DST_EP=$GLOBUS_TUFTS_EP ;;
  ls)         DIR_MODE=${1:?to-polaris|to-tufts}; D=${2:?dir}
              case "$DIR_MODE" in to-polaris) EP=$GLOBUS_EAGLE_EP;; to-tufts) EP=$GLOBUS_TUFTS_EP;; *) exit 2;; esac
              globus ls -l "$EP:$D"; exit 0 ;;
  *) echo "MODE must be to-polaris|to-tufts|status|ls" >&2; exit 2 ;;
esac
SRC=${1:?source dir}; DST=${2:?destination dir}; shift 2
WAIT=0; DRY=0
for a in "$@"; do case "$a" in --wait) WAIT=1;; --dry-run) DRY=1;; *) echo "unknown option $a" >&2; exit 2;; esac; done
SRC=${SRC%/}/; DST=${DST%/}/
LABEL="larformer $MODE $(basename "${SRC%/}") $(date +%Y%m%d-%H%M)"
globus whoami >/dev/null 2>&1 || { echo "ERROR: not logged in: globus login" >&2; exit 2; }
echo ">>> $MODE: $SRC_EP:$SRC -> $DST_EP:$DST  (label: $LABEL)"
echo ">>> source listing (first 20):"; globus ls "$SRC_EP:$SRC" | head -20
CMD=(globus transfer "$SRC_EP:$SRC" "$DST_EP:$DST" --recursive --sync-level checksum --verify-checksum
     --preserve-mtime --label "$LABEL" --notify off --jmespath task_id --format unix)
echo ">>> ${CMD[*]}"
[ $DRY -eq 1 ] && { echo "(dry run)"; exit 0; }
TASK=$("${CMD[@]}")
echo ">>> task $TASK submitted   (globus task show $TASK ; globus task event-list $TASK)"
echo "$(date -Is) $MODE $SRC -> $DST task=$TASK" >> "${GLOBUS_LOG:-$HOME/.larformer_globus_tasks.log}"
if [ $WAIT -eq 1 ]; then
  echo ">>> waiting (globus task wait, 60 s polling) ..."
  globus task wait "$TASK" --polling-interval 60 --timeout 604800 || { echo "ERROR: task $TASK did not succeed"; globus task show "$TASK"; exit 1; }
  globus task show "$TASK" | grep -E "Status|Files|Bytes|Directories|Faults" || true
  echo ">>> DONE $TASK. On the destination run:  (cd <dir> && sha256sum -c <(sed -n 's/^sha256=\\(.*\\)/\\1  IMAGE/p' *.manifest ...)) -- or use verify_images.sh"
fi
