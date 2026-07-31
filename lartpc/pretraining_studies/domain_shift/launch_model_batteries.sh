#!/bin/bash
# Submit the full tier-extraction + battery set for ONE model snapshot.
#   ./launch_model_batteries.sh <TAG> <CONFIG> <CKPT>
# e.g.
#   ./launch_model_batteries.sh P1A.2_img6144000 \
#       configs/lartpc/p05/pretrain-sonata-p1a2-mc-ghosts-detsym.py \
#       sonata/p1a/P1A.2-mc_ghosts-s0/snapshot/snapshot_iter0128000_img6144000.pth
# Extractions run on GPU; batteries queue with afterok dependencies.
set -eu

TAG=${1:?tag, e.g. P1A.2_img6144000}
CFG=${2:?config}
CKPT=${3:?checkpoint}

MC_LIST=lartpc/filelists/h5list_v3_mc_diag1k_tufts.txt
DATA_LIST=lartpc/filelists/h5list_v3_extbnb_diag1k_tufts.txt
OUTDIR=lartpc/pretraining_studies/domain_shift/features

declare -A JID
submit_extract () {  # name list tier
    JID[$1]=$(sbatch --parsable --job-name="dsx_${TAG}_$1" \
        submit_extract_tufts.sbatch \
        --config "$CFG" --checkpoint "$CKPT" --data-list "$2" --tier "$3" \
        --points-per-event 256 --out "${OUTDIR}/${TAG}_$1.npz")
    echo "  extract $1 -> job ${JID[$1]}"
}

echo "[${TAG}] submitting extractions"
submit_extract mc_raw           "$MC_LIST"   raw
submit_extract mc_cosmic        "$MC_LIST"   cosmic
submit_extract mc_cosmicclean   "$MC_LIST"   cosmic-clean
submit_extract mc_cosmiclmclean "$MC_LIST"   cosmic-lmclean
submit_extract data_raw         "$DATA_LIST" raw
submit_extract data_cosmicclean "$DATA_LIST" cosmic-clean

DEP_BATTERY="${JID[mc_raw]}:${JID[mc_cosmic]}:${JID[mc_cosmicclean]}:${JID[data_raw]}:${JID[data_cosmicclean]}"
DEP_LM="${JID[mc_cosmiclmclean]}:${JID[data_cosmicclean]}"

B1=$(sbatch --parsable --dependency=afterok:${DEP_BATTERY} \
    run_battery_tufts.sbatch "features/${TAG}")
B2=$(sbatch --parsable --dependency=afterok:${DEP_LM} \
    run_pair_tufts.sbatch \
    "features/${TAG}_mc_cosmiclmclean.npz" \
    "features/${TAG}_data_cosmicclean.npz" \
    "${TAG} tier1-lmclean-symmetric" \
    "results/${TAG}_tier1lmclean.json")
echo "[${TAG}] battery job ${B1}, symmetric-pair job ${B2}"
