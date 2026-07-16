#!/bin/bash
# Pull P05 snapshots + run metadata from Isambard. RUNS AT TUFTS (Tufts is
# behind a VPN; ssh must originate here). The Tufts-side alias is
# u6jo.aip2.isambard (clifton-managed ~/.ssh/config entry).
#
# NOTE: Isambard ssh uses a SHORT-LIVED (~12 h) clifton certificate. If ssh
# fails with "kex_exchange_identification: Connection closed", renew first:
#   ./clifton auth        (repo root; interactive browser OIDC — user only)
#
# Idempotent — rerun any time (e.g. daily while the Wave A fleet trains);
# rsync only moves new snapshots. First run: add --dry-run to check patterns.

set -eu

ISAMBARD=${ISAMBARD_SSH:-u6jo.aip2.isambard}
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
