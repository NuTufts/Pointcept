#!/bin/bash
# Submit the Tier-1 gap-vs-images-seen ladder for ONE run (plan figure F3):
# for every snapshot, extract mc_cosmic + data_raw and queue the pair
# battery. Snapshots whose feature files already exist are skipped.
#   ./launch_ladder.sh <RUN_TAG> <CONFIG> <SNAPSHOT_DIR>
# e.g.
#   ./launch_ladder.sh P5B.1 \
#       configs/lartpc/p05/pretrain-sonata-p5b1-mix-raw-detsym.py \
#       sonata/p5b/P5B.1-mix_raw-s0/snapshot
set -eu

RUN=${1:?run tag, e.g. P5B.1}
CFG=${2:?config}
SNAPDIR=${3:?snapshot dir}

# Snapshot paths are repo-relative; this script runs from domain_shift/.
REPO=${LOCAL_REPO:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/isambard_pointcept}
case "$SNAPDIR" in /*) ;; *) SNAPDIR="${REPO}/${SNAPDIR}" ;; esac
ls "${SNAPDIR}"/snapshot_iter*_img*.pth >/dev/null 2>&1 \
    || { echo "no snapshots match in ${SNAPDIR}"; exit 1; }

MC_LIST=lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt
DATA_LIST=lartpc/filelists/h5list_v3_extbnb_diag1k_tufts.txt
OUTDIR=lartpc/pretraining_studies/domain_shift/features

for CKPT in "${SNAPDIR}"/snapshot_iter*_img*.pth; do
    IMG=$(basename "$CKPT" | sed 's/.*_img\([0-9]*\)\.pth/\1/')
    TAG="${RUN}_img${IMG}"
    DEPS=""
    for spec in "mc_cosmic ${MC_LIST} cosmic" \
                "data_raw ${DATA_LIST} raw"; do
        set -- $spec
        FEAT="${OUTDIR}/${TAG}_$1.npz"
        if [ -f "${REPO}/${FEAT}" ]; then
            echo "[${TAG}] $1 exists, skipping extraction"
            continue
        fi
        JID=$(sbatch --parsable --job-name="dsl_${TAG}_$1" \
            submit_extract_tufts.sbatch \
            --config "$CFG" --checkpoint "$CKPT" --data-list "$2" \
            --tier "$3" --out "$FEAT")
        DEPS="${DEPS}:${JID}"
        echo "[${TAG}] extract $1 -> job ${JID}"
    done
    OUT="results/${TAG}_tier1.json"
    if [ -f "$OUT" ]; then
        echo "[${TAG}] battery result exists, skipping"
        continue
    fi
    DEPFLAG=""
    [ -n "$DEPS" ] && DEPFLAG="--dependency=afterok${DEPS}"
    BID=$(sbatch --parsable $DEPFLAG run_pair_tufts.sbatch \
        "features/${TAG}_mc_cosmic.npz" "features/${TAG}_data_raw.npz" \
        "${TAG} tier1-cosmic" "$OUT")
    echo "[${TAG}] pair battery -> job ${BID}"
done
