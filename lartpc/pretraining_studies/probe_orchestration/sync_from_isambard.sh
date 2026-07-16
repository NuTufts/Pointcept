#!/bin/bash
# Pull P05 snapshots + run metadata from Isambard. RUNS AT TUFTS (Tufts is
# behind a VPN; ssh must originate here). Requires a working ssh alias to
# Isambard, e.g. in ~/.ssh/config:  Host isambard  HostName ...  User twongj01.u6jo
#
# Idempotent — rerun any time (e.g. daily while the Wave A fleet trains);
# rsync only moves new snapshots. First run: add --dry-run to check patterns.

set -eu

ISAMBARD=${ISAMBARD_SSH:-isambard}
REMOTE=/projects/u6jo/work/pointcept
LOCAL=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}

echo "[sync] snapshots + run configs from ${ISAMBARD}:${REMOTE}/sonata/p05/"
rsync -av -m \
    --exclude='smoke/' \
    --include='*/' \
    --include='snapshot/*.pth' \
    --include='config.py' \
    --exclude='*' \
    "${ISAMBARD}:${REMOTE}/sonata/p05/" "${LOCAL}/sonata/p05/"

echo "[sync] run registry"
mkdir -p "${LOCAL}/exp"
rsync -av "${ISAMBARD}:${REMOTE}/exp/registry.csv" \
    "${LOCAL}/exp/registry_isambard.csv"

echo "[sync] inventory:"
for d in "${LOCAL}"/sonata/p05/P05*/snapshot; do
    [ -d "$d" ] && echo "  $(basename "$(dirname "$d")"): $(ls "$d" | wc -l) snapshots"
done
